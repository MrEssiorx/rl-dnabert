"""Prepare S9.6 and dRNH peak tables for sequence experiments.

Coordinates are BED-style half-open. Overlap labels are assigned to original
peaks, before any sequence-window strategy is applied.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


NARROWPEAK_COLUMNS = [
    "chr", "start", "end", "name", "score", "strand",
    "signalValue", "pValue", "qValue", "peak",
]
PRIMARY_CHROMOSOMES = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def _source_table(frame: pd.DataFrame, source: str) -> pd.DataFrame:
    required = {"chr", "start", "end", "peak"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{source}: missing columns {sorted(missing)}")

    table = frame.copy().reset_index(drop=True)
    table["source_row"] = table.index + 1  # one-based line number in narrowPeak
    table = table.loc[table["chr"].isin(PRIMARY_CHROMOSOMES)].copy().reset_index(drop=True)
    for column in ("start", "end", "peak"):
        table[column] = pd.to_numeric(table[column], errors="raise").astype("int64")
    if ((table["start"] < 0) | (table["end"] <= table["start"])).any():
        raise ValueError(f"{source}: invalid peak coordinates")
    table["source"] = source
    table["peak_id"] = source + ":" + table["source_row"].astype(str)
    table["peak_length"] = table["end"] - table["start"]
    return table


def _intervals_by_chrom(table: pd.DataFrame) -> dict[str, list[tuple[int, int, int]]]:
    by_chrom: dict[str, list[tuple[int, int, int]]] = defaultdict(list)
    for index, row in enumerate(table.itertuples(index=False)):
        by_chrom[row.chr].append((int(row.start), int(row.end), index))
    for chrom, intervals in by_chrom.items():
        intervals.sort()
        if any(left[1] > right[0] for left, right in zip(intervals, intervals[1:])):
            raise ValueError(f"Overlapping peaks within {table['source'].iloc[0]} on {chrom}")
    return by_chrom


def prepare_peak_table(
    s96: pd.DataFrame,
    drnh: pd.DataFrame,
    *,
    shared_threshold: float = 0.25,
) -> pd.DataFrame:
    """Return one row per primary-chromosome peak with overlap metadata.

    A peak is ``shared`` when bases covered by any peaks in the other dataset
    occupy at least ``shared_threshold`` of its own length. The two sides of
    an overlapping pair can consequently receive different group labels.
    """
    if not 0 < shared_threshold <= 1:
        raise ValueError("shared_threshold must be in (0, 1]")
    a = _source_table(s96, "S9.6")
    b = _source_table(drnh, "dRNH")
    a_by_chrom, b_by_chrom = _intervals_by_chrom(a), _intervals_by_chrom(b)
    a_covered, b_covered = [0] * len(a), [0] * len(b)

    for chrom in a_by_chrom.keys() & b_by_chrom.keys():
        left, right = a_by_chrom[chrom], b_by_chrom[chrom]
        i = j = 0
        while i < len(left) and j < len(right):
            start_a, end_a, index_a = left[i]
            start_b, end_b, index_b = right[j]
            shared = min(end_a, end_b) - max(start_a, start_b)
            if shared > 0:
                a_covered[index_a] += shared
                b_covered[index_b] += shared
            if end_a <= end_b:
                i += 1
            else:
                j += 1

    for table, covered, exclusive in ((a, a_covered, "S9.6_only"), (b, b_covered, "dRNH_only")):
        table["overlap_bp"] = covered
        table["overlap_fraction"] = table["overlap_bp"] / table["peak_length"]
        table["group"] = table["overlap_fraction"].ge(shared_threshold).map(
            {True: "shared", False: exclusive}
        )
    return pd.concat((a, b), ignore_index=True)
