"""Prepare all canonical peaks and save frozen DNABERT-2 embeddings."""

import pandas as pd

from rloops_llm_research.cutting import crop_around_summit
from rloops_llm_research.dnabert2_embeddings import OUTPUT_PATH, extract_embeddings, save_embeddings
from rloops_llm_research.model_inputs import SequenceDataset
from rloops_llm_research.peaks import NARROWPEAK_COLUMNS, prepare_peak_table
from rloops_llm_research.sequences import attach_fasta


def main() -> None:
    s96 = pd.read_csv("data/rlregions_S96.narrowPeak", sep="\t", header=None, names=NARROWPEAK_COLUMNS)
    drnh = pd.read_csv("data/rlregions_dRNH.narrowPeak", sep="\t", header=None, names=NARROWPEAK_COLUMNS)
    peaks = prepare_peak_table(s96, drnh, shared_threshold=0.25)
    examples = attach_fasta(crop_around_summit(peaks, max_bp=2560), "data/hg38.fa")
    embeddings, peak_ids = extract_embeddings(SequenceDataset(examples), batch_size=16)
    save_embeddings(OUTPUT_PATH, embeddings, examples.drop(columns="sequence"), peak_ids)
    print(f"Saved {embeddings.shape[0]} x {embeddings.shape[1]} embeddings to {OUTPUT_PATH}", flush=True)


if __name__ == "__main__":
    main()
