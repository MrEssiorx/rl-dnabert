"""Indexed hg38 FASTA access while retaining peak metadata in a table."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


class IndexedFasta:
    """Read half-open slices from a plain FASTA using its samtools .fai index."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        index_path = Path(f"{self.path}.fai")
        self.index: dict[str, tuple[int, int, int, int]] = {}
        with index_path.open(encoding="utf-8") as index_file:
            for line in index_file:
                name, length, offset, line_bases, line_bytes, *_ = line.rstrip("\n").split("\t")
                self.index[name] = tuple(map(int, (length, offset, line_bases, line_bytes)))
        self._file = self.path.open("rb")

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> "IndexedFasta":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def fetch(self, chrom: str, start: int, end: int) -> str:
        """Return reference bases at zero-based ``[start, end)``."""
        length, offset, line_bases, line_bytes = self.index[chrom]
        if not 0 <= start < end <= length:
            raise ValueError(f"{chrom}: interval [{start}, {end}) outside [0, {length})")

        def byte_position(base: int) -> int:
            return offset + (base // line_bases) * line_bytes + base % line_bases

        first, last = byte_position(start), byte_position(end)
        self._file.seek(first)
        sequence = self._file.read(last - first).replace(b"\n", b"").replace(b"\r", b"")
        if len(sequence) != end - start:
            raise ValueError(f"{chrom}: FASTA index and sequence disagree at [{start}, {end})")
        return sequence.decode("ascii")


def attach_fasta(windows: pd.DataFrame, fasta_path: str | Path) -> pd.DataFrame:
    """Add a ``sequence`` column without discarding any window metadata."""
    required = {"chr", "window_start", "window_end"}
    missing = required - set(windows.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")
    result = windows.copy()
    with IndexedFasta(fasta_path) as fasta:
        result["sequence"] = [
            fasta.fetch(str(row.chr), int(row.window_start), int(row.window_end))
            for row in result.itertuples(index=False)
        ]
    return result
