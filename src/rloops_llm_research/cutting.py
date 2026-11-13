"""Replaceable coordinate strategies for peak-derived sequence windows."""

from __future__ import annotations

import pandas as pd


def crop_around_summit(peaks: pd.DataFrame, *, max_bp: int = 2560) -> pd.DataFrame:
    """Keep at most ``max_bp`` bases within each peak, centered on its summit.

    Short peaks remain whole. Near an edge, the window shifts to retain its
    full possible length. Original ``start``/``end`` coordinates are preserved.
    """
    if max_bp <= 0:
        raise ValueError("max_bp must be positive")
    required = {"start", "end", "peak"}
    missing = required - set(peaks.columns)
    if missing:
        raise ValueError(f"missing columns {sorted(missing)}")

    result = peaks.copy()
    starts, ends, summits = [], [], []
    for row in result.itertuples(index=False):
        start, end, offset = int(row.start), int(row.end), int(row.peak)
        if start < 0 or end <= start:
            raise ValueError(f"invalid peak coordinates [{start}, {end})")
        if not 0 <= offset < end - start:
            raise ValueError(f"summit offset {offset} outside peak [{start}, {end})")
        length = min(end - start, max_bp)
        summit = start + offset
        window_start = max(start, min(summit - length // 2, end - length))
        starts.append(window_start)
        ends.append(window_start + length)
        summits.append(summit)
    result["summit"] = summits
    result["window_start"] = starts
    result["window_end"] = ends
    result["window_length"] = result["window_end"] - result["window_start"]
    return result
