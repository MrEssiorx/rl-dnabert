"""Run a small, offline DNABERT-2 embedding check from repo/."""

import argparse
import json
import resource
from pathlib import Path
from time import perf_counter

import torch
from transformers import AutoModel, AutoTokenizer

from fetch_dnbert2 import MODEL_DIR


DNA = "ACGTAGCATCGGATCTATCTATCGACACTTGGTTATCGATCTACGAGCATCTCGTTAGC"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-dir", type=Path, default=MODEL_DIR)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()

    if not args.model_dir.is_dir():
        parser.error(f"Model directory does not exist: {args.model_dir}")
    device = "cuda" if args.device == "auto" and torch.cuda.is_available() else args.device
    if device == "auto":
        device = "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA device requested but unavailable")

    started = perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    )
    model = AutoModel.from_pretrained(
        args.model_dir, trust_remote_code=True, local_files_only=True
    ).to(device).eval()
    load_s = perf_counter() - started

    dna = [DNA, DNA + "ACGTACGTACGT"]
    encoded = tokenizer(dna, padding=True, return_tensors="pt")
    token_lengths = encoded["attention_mask"].sum(dim=1).tolist()
    encoded = {key: value.to(device) for key, value in encoded.items()}
    if device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    started = perf_counter()
    with torch.inference_mode():
        output = model(**encoded)
        hidden = output[0]
        mask = encoded["attention_mask"].unsqueeze(-1)
        embeddings = (hidden * mask).sum(dim=1) / mask.sum(dim=1)
    if device == "cuda":
        torch.cuda.synchronize()
    inference_s = perf_counter() - started

    assert tuple(embeddings.shape) == (2, 768), embeddings.shape
    assert torch.isfinite(embeddings).all().item()
    print(json.dumps({
        "model_dir": str(args.model_dir),
        "device": device,
        "torch": torch.__version__,
        "token_lengths": token_lengths,
        "hidden_shape": list(hidden.shape),
        "embedding_shape": list(embeddings.shape),
        "embedding_l2_norms": embeddings.norm(dim=1).tolist(),
        "load_seconds": round(load_s, 3),
        "inference_seconds": round(inference_s, 3),
        "max_rss_kb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        "cuda_peak_allocated_bytes": (
            torch.cuda.max_memory_allocated() if device == "cuda" else None
        ),
    }, indent=2))


if __name__ == "__main__":
    main()
