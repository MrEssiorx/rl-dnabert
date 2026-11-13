"""Small coordinate and data-flow checks for sequence preparation."""

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from rloops_llm_research.cutting import crop_around_summit
from rloops_llm_research.model_inputs import SequenceDataset, TokenizerCollator
from rloops_llm_research.peaks import prepare_peak_table
from rloops_llm_research.sequences import IndexedFasta, attach_fasta


class PreparationTests(unittest.TestCase):
    def test_overlap_is_relative_to_each_peak_and_nonprimary_is_excluded(self):
        s96 = pd.DataFrame([
            {"chr": "chr1", "start": 0, "end": 100, "peak": 50},
            {"chr": "chrM", "start": 0, "end": 100, "peak": 50},
        ])
        drnh = pd.DataFrame([{"chr": "chr1", "start": 80, "end": 110, "peak": 10}])
        result = prepare_peak_table(s96, drnh, shared_threshold=0.25)
        self.assertEqual(result["peak_id"].tolist(), ["S9.6:1", "dRNH:1"])
        self.assertEqual(result["overlap_bp"].tolist(), [20, 20])
        self.assertEqual(result["group"].tolist(), ["S9.6_only", "shared"])

    def test_crop_keeps_short_peaks_and_shifts_at_edge(self):
        peaks = pd.DataFrame([
            {"start": 100, "end": 5100, "peak": 50},
            {"start": 100, "end": 5100, "peak": 4950},
            {"start": 100, "end": 500, "peak": 200},
        ])
        result = crop_around_summit(peaks, max_bp=2560)
        self.assertEqual(list(zip(result["window_start"], result["window_end"])), [
            (100, 2660), (2540, 5100), (100, 500),
        ])
        with self.assertRaises(ValueError):
            crop_around_summit(pd.DataFrame([{"start": 0, "end": 10, "peak": -1}]))

    def test_fasta_slice_across_wrapped_lines_and_metadata_survives_batching(self):
        with tempfile.TemporaryDirectory() as directory:
            fasta_path = Path(directory) / "tiny.fa"
            fasta_path.write_bytes(b">chr1\nACGT\nTGCA\n")
            Path(f"{fasta_path}.fai").write_text("chr1\t8\t6\t4\t5\n")
            with IndexedFasta(fasta_path) as fasta:
                self.assertEqual(fasta.fetch("chr1", 2, 6), "GTTG")
            windows = pd.DataFrame([{
                "peak_id": "S9.6:1", "source": "S9.6", "group": "shared",
                "chr": "chr1", "window_start": 2, "window_end": 6,
            }])
            examples = attach_fasta(windows, fasta_path)
            self.assertEqual(examples.loc[0, "sequence"], "GTTG")
            dataset = SequenceDataset(examples)

            class FakeTokenizer:
                def __call__(self, sequences, **kwargs):
                    assert kwargs == {"padding": True, "truncation": False, "return_tensors": "pt"}
                    return {"sequences": sequences}

            batch = TokenizerCollator(FakeTokenizer())([dataset[0]])
            self.assertEqual(batch["peak_ids"], ["S9.6:1"])
            self.assertEqual(batch["groups"], ["shared"])
            self.assertEqual(batch["model_inputs"], {"sequences": ["GTTG"]})


if __name__ == "__main__":
    unittest.main()
