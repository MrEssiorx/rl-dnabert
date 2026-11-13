"""Keep sequence records and metadata separate from model tensor batches.

This module deliberately does not import torch or a particular genomic model.
``SequenceDataset`` implements the interface expected by a PyTorch DataLoader.
The tokenizer's ``return_tensors='pt'`` conversion happens only at batch time.
"""

from __future__ import annotations

from typing import Any

import pandas as pd


class SequenceDataset:
    def __init__(self, examples: pd.DataFrame):
        required = {"peak_id", "sequence", "source", "group"}
        missing = required - set(examples.columns)
        if missing:
            raise ValueError(f"missing columns {sorted(missing)}")
        self.examples = examples.reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> dict[str, str]:
        row = self.examples.iloc[index]
        return {
            "peak_id": str(row["peak_id"]),
            "sequence": str(row["sequence"]),
            "source": str(row["source"]),
            "group": str(row["group"]),
        }


class TokenizerCollator:
    """Tokenize DNA strings; keep identifiers and labels outside model inputs."""

    def __init__(self, tokenizer: Any):
        self.tokenizer = tokenizer

    def __call__(self, records: list[dict[str, str]]) -> dict[str, Any]:
        if not records:
            raise ValueError("cannot collate an empty batch")
        model_inputs = self.tokenizer(
            [record["sequence"].upper() for record in records],
            padding=True,
            truncation=False,
            return_tensors="pt",
        )
        return {
            "model_inputs": model_inputs,
            "peak_ids": [record["peak_id"] for record in records],
            "sources": [record["source"] for record in records],
            "groups": [record["group"] for record in records],
        }
