"""Count query intervals intersecting each annotation by at least one base.

Inputs are BED-like, tab-delimited files (plain or .gz), with chromosome,
0-based start and exclusive end in the first three columns. Extra columns are
ignored. narrowPeak can be passed without conversion. Only chr1-22, chrX and
chrY are retained. Annotation sets are independent: one query may count in
several categories. The denominator is the number of retained query intervals.

Run from repo/:
  python scripts/interval_overlap.py \
    --query S96=data/rlregions_S96.narrowPeak \
    --query dRNH=data/rlregions_dRNH.narrowPeak \
    --annotation promoter=data/annotations/promoter.bed \
    --annotation GWAS=data/annotations/gwas.bed \
    --output artifacts/overlaps.csv
"""

from __future__ import annotations

import argparse
import bisect
import csv
import gzip
import sys
from collections import defaultdict
from pathlib import Path


CHROMOSOMES = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def parse_named_file(value: str) -> tuple[str, Path]:
    name, separator, path = value.partition("=")
    if not separator or not name or not path:
        raise argparse.ArgumentTypeError("expected NAME=PATH")
    return name, Path(path)


def read_intervals(path: Path) -> tuple[dict[str, list[tuple[int, int]]], int, int]:
    intervals: dict[str, list[tuple[int, int]]] = defaultdict(list)
    retained = skipped = 0
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip() or line.startswith(("#", "track ", "browser ")):
                continue
            fields = line.rstrip("\n").split("\t")
            if len(fields) < 3:
                raise ValueError(f"{path}:{line_number}: expected at least three tab-separated columns")
            chrom = fields[0]
            if chrom not in CHROMOSOMES:
                skipped += 1
                continue
            try:
                start, end = int(fields[1]), int(fields[2])
            except ValueError as error:
                raise ValueError(f"{path}:{line_number}: coordinates must be integers") from error
            if start < 0 or end <= start:
                raise ValueError(f"{path}:{line_number}: invalid half-open interval [{start}, {end})")
            intervals[chrom].append((start, end))
            retained += 1
    return intervals, retained, skipped


def merge_intervals(intervals: list[tuple[int, int]]) -> tuple[list[int], list[int]]:
    merged: list[list[int]] = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1][1] = max(end, merged[-1][1])
        else:
            merged.append([start, end])
    return [row[0] for row in merged], [row[1] for row in merged]


def count_hits(
    queries: dict[str, list[tuple[int, int]]],
    annotation: dict[str, tuple[list[int], list[int]]],
) -> int:
    count = 0
    for chrom, rows in queries.items():
        starts, ends = annotation.get(chrom, ([], []))
        for start, end in rows:
            index = bisect.bisect_right(ends, start)
            if index < len(starts) and starts[index] < end:
                count += 1
    return count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--query", action="append", type=parse_named_file, required=True, metavar="NAME=PATH")
    parser.add_argument("--annotation", action="append", type=parse_named_file, required=True, metavar="NAME=PATH")
    parser.add_argument("--output", type=Path, help="CSV path; default: stdout")
    args = parser.parse_args()
    for kind in ("query", "annotation"):
        names = [name for name, _ in getattr(args, kind)]
        if len(names) != len(set(names)):
            parser.error(f"duplicate {kind} name")

    try:
        queries = {name: read_intervals(path) for name, path in args.query}
        annotations = {
            name: {chrom: merge_intervals(rows) for chrom, rows in read_intervals(path)[0].items()}
            for name, path in args.annotation
        }
        output = args.output.open("w", encoding="utf-8", newline="") if args.output else sys.stdout
        try:
            writer = csv.writer(output)
            writer.writerow(["query", "annotation", "query_intervals", "overlapping_query_intervals", "percent", "excluded_query_intervals"])
            for query_name, _ in args.query:
                rows, total, excluded = queries[query_name]
                if total == 0:
                    raise ValueError(f"{query_name}: no intervals remain on chr1-22, chrX or chrY")
                for annotation_name, _ in args.annotation:
                    hits = count_hits(rows, annotations[annotation_name])
                    writer.writerow([query_name, annotation_name, total, hits, f"{100 * hits / total:.6f}", excluded])
        finally:
            if args.output:
                output.close()
    except (OSError, ValueError) as error:
        parser.exit(1, f"ERROR: {error}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
