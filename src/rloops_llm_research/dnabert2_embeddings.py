"""Extract DNABERT-2 vectors and keep their peak metadata in one artifact."""

from __future__ import annotations

from io import StringIO
from pathlib import Path
import os

import pandas as pd
import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModel, AutoTokenizer

from .model_inputs import SequenceDataset, TokenizerCollator


REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
MODEL_DIR = Path("data/models/DNABERT-2-117M") / REVISION
OUTPUT_PATH = Path("artifacts/dnabert2_embeddings.pt")


def extract_embeddings(dataset: SequenceDataset, batch_size: int = 16) -> tuple[torch.Tensor, list[str]]:
    if not torch.cuda.is_available():
        raise RuntimeError("DNABERT-2 extraction requires a GPU")
    if not MODEL_DIR.is_dir():
        raise FileNotFoundError(MODEL_DIR)
    tokenizer = AutoTokenizer.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    config = AutoConfig.from_pretrained(
        MODEL_DIR, trust_remote_code=True, local_files_only=True
    )
    # Select the tested PyTorch attention branch; eval() disables dropout itself.
    config.attention_probs_dropout_prob = 0.1
    model = AutoModel.from_pretrained(
        MODEL_DIR, config=config, trust_remote_code=True, local_files_only=True
    ).cuda().eval()
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        collate_fn=TokenizerCollator(tokenizer),
    )
    vectors: list[torch.Tensor] = []
    peak_ids: list[str] = []
    with torch.inference_mode():
        for batch in loader:
            inputs = {key: value.cuda() for key, value in batch["model_inputs"].items()}
            hidden = model(**inputs)[0]
            mask = inputs["attention_mask"].unsqueeze(-1)
            vectors.append(((hidden * mask).sum(dim=1) / mask.sum(dim=1)).cpu())
            peak_ids.extend(batch["peak_ids"])
    del model
    return torch.cat(vectors, dim=0), peak_ids


def save_embeddings(
    path: Path, embeddings: torch.Tensor, metadata: pd.DataFrame, peak_ids: list[str]
) -> None:
    if peak_ids != metadata["peak_id"].astype(str).tolist():
        raise ValueError("Peak IDs differ between vectors and metadata")
    if embeddings.shape != (len(metadata), 768) or not torch.isfinite(embeddings).all():
        raise ValueError(f"Invalid embeddings: {embeddings.shape}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".pt.tmp")
    torch.save(
        {"embeddings": embeddings, "metadata_csv": metadata.to_csv(index=False),
         "checkpoint_revision": REVISION},
        temporary,
    )
    os.replace(temporary, path)


def load_embeddings(path: Path = OUTPUT_PATH) -> tuple[torch.Tensor, pd.DataFrame]:
    payload = torch.load(path, map_location="cpu", weights_only=True)
    if payload["checkpoint_revision"] != REVISION:
        raise ValueError("Unexpected DNABERT-2 checkpoint revision")
    metadata = pd.read_csv(StringIO(payload["metadata_csv"]))
    embeddings = payload["embeddings"]
    if embeddings.shape != (len(metadata), 768) or not torch.isfinite(embeddings).all():
        raise ValueError(f"Invalid saved embeddings: {embeddings.shape}")
    return embeddings, metadata
