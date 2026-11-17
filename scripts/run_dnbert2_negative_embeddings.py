"""Extract frozen DNABERT-2 embeddings for sampled genomic negatives."""

from pathlib import Path

import pandas as pd

from rloops_llm_research.dnabert2_embeddings import extract_embeddings, save_embeddings
from rloops_llm_research.model_inputs import SequenceDataset
from rloops_llm_research.sequences import attach_fasta


WINDOWS_PATH = Path("artifacts/dnabert2_negative_windows.csv")
OUTPUT_PATH = Path("artifacts/dnabert2_negative_embeddings.pt")


def main() -> None:
    windows = pd.read_csv(WINDOWS_PATH)
    if len(windows) != 71056 or windows["peak_id"].duplicated().any():
        raise ValueError("Unexpected negative window count or duplicate IDs")
    if not windows["window_length"].eq(2000).all():
        raise ValueError("Negative windows must be 2000 bp")
    examples = attach_fasta(windows, "data/hg38.fa")
    embeddings, peak_ids = extract_embeddings(SequenceDataset(examples), batch_size=16)
    save_embeddings(OUTPUT_PATH, embeddings, windows, peak_ids)
    print(f"Saved {embeddings.shape[0]} x {embeddings.shape[1]} embeddings to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
