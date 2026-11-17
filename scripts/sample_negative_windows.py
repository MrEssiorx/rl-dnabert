"""Tile canonical hg38 into 2-kb windows and sample R-loop negatives."""

from __future__ import annotations

import argparse
import bisect
import csv
import random
from pathlib import Path

from interval_overlap import merge_intervals, read_intervals
from rloops_llm_research.sequences import IndexedFasta


CHROMOSOMES = [f"chr{i}" for i in range(1, 23)] + ["chrX", "chrY"]


def overlaps(starts: list[int], ends: list[int], start: int, end: int) -> bool:
    index = bisect.bisect_right(ends, start)
    return index < len(starts) and starts[index] < end


def sample_windows(
    fasta_path: Path,
    positive_paths: tuple[Path, Path],
    blacklist_path: Path,
    *,
    window_bp: int,
    count: int,
    seed: int,
) -> tuple[list[tuple[str, int, int]], dict[str, int]]:
    if window_bp <= 0 or count <= 0:
        raise ValueError("window_bp and count must be positive")
    positives: dict[str, list[tuple[int, int]]] = {chrom: [] for chrom in CHROMOSOMES}
    for path in positive_paths:
        intervals, _, _ = read_intervals(path)
        for chrom, rows in intervals.items():
            positives[chrom].extend(rows)
    blacklist, _, _ = read_intervals(blacklist_path)
    positive_index = {chrom: merge_intervals(rows) for chrom, rows in positives.items()}
    blacklist_index = {
        chrom: merge_intervals(blacklist.get(chrom, [])) for chrom in CHROMOSOMES
    }
    candidates: list[tuple[str, int, int]] = []
    stats = {"tiled": 0, "positive": 0, "blacklist": 0, "n_rich": 0}
    with IndexedFasta(fasta_path) as fasta:
        for chrom in CHROMOSOMES:
            length = fasta.index[chrom][0]
            for start in range(0, length - window_bp + 1, window_bp):
                end = start + window_bp
                stats["tiled"] += 1
                if overlaps(*positive_index[chrom], start, end):
                    stats["positive"] += 1
                    continue
                if overlaps(*blacklist_index[chrom], start, end):
                    stats["blacklist"] += 1
                    continue
                if fasta.fetch(chrom, start, end).upper().count("N") > window_bp / 2:
                    stats["n_rich"] += 1
                    continue
                candidates.append((chrom, start, end))
    stats["eligible"] = len(candidates)
    if len(candidates) < count:
        raise ValueError(f"Only {len(candidates)} eligible windows, requested {count}")
    selected = random.Random(seed).sample(candidates, count)
    order = {chrom: index for index, chrom in enumerate(CHROMOSOMES)}
    selected.sort(key=lambda row: (order[row[0]], row[1]))
    return selected, stats


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fasta", type=Path, default=Path("data/hg38.fa"))
    parser.add_argument("--s96", type=Path, default=Path("data/rlregions_S96.narrowPeak"))
    parser.add_argument("--drnh", type=Path, default=Path("data/rlregions_dRNH.narrowPeak"))
    parser.add_argument("--blacklist", type=Path,
                        default=Path("data/annotations/encode_grch38_blacklist.bed.gz"))
    parser.add_argument("--window-bp", type=int, default=2000)
    parser.add_argument("--count", type=int, default=71056)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path,
                        default=Path("artifacts/dnabert2_negative_windows.csv"))
    args = parser.parse_args()
    windows, stats = sample_windows(
        args.fasta, (args.s96, args.drnh), args.blacklist,
        window_bp=args.window_bp, count=args.count, seed=args.seed,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_suffix(args.output.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["peak_id", "chr", "start", "end", "window_start",
                         "window_end", "window_length", "source", "group"])
        for chrom, start, end in windows:
            writer.writerow([f"negative:{chrom}:{start}", chrom, start, end,
                             start, end, args.window_bp, "negative", "negative"])
    temporary.replace(args.output)
    print(f"Selected {len(windows)} windows; counts: {stats}; seed={args.seed}", flush=True)
    print(f"Saved {args.output}", flush=True)


if __name__ == "__main__":
    main()
