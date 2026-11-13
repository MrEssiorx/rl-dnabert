"""Build seven BED annotations from a pinned GENCODE GTF and hg38 FASTA index.

This is an explicit approximation of the seven Z-DNABERT categories, not a
claim that their unpublished annotation rules have been reproduced exactly.
Run from repo/: python scripts/prepare_gencode_regions.py \
  data/gencode.v32.annotation.gtf.gz data/hg38.fa.fai data/annotations/gencode_v32
"""

from __future__ import annotations

import argparse
import gzip
from collections import defaultdict
from pathlib import Path

from interval_overlap import CHROMOSOMES, merge_intervals


NAMES = ("promoter", "5utr", "exon", "intron", "3utr", "downstream", "distal_intergenic")


def attributes(raw: str) -> dict[str, str]:
    result = {}
    for item in raw.split(";"):
        key, _, value = item.strip().partition(" ")
        if key:
            result[key] = value.strip(' "')
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("gtf", type=Path)
    parser.add_argument("fai", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()

    lengths = {}
    with args.fai.open() as handle:
        for line in handle:
            chrom, length, *_ = line.rstrip("\n").split("\t")
            if chrom in CHROMOSOMES:
                lengths[chrom] = int(length)
    if lengths.keys() != CHROMOSOMES:
        parser.error("FASTA index must contain all canonical chr1-22, chrX, chrY names")

    transcripts = {}
    genes = defaultdict(list)
    opener = gzip.open if args.gtf.suffix == ".gz" else open
    with opener(args.gtf, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.startswith("#"):
                continue
            chrom, _, feature, first, last, _, strand, _, raw = line.rstrip("\n").split("\t", 8)
            if chrom not in CHROMOSOMES or feature not in {"gene", "transcript", "exon", "CDS", "UTR"}:
                continue
            start, end = int(first) - 1, int(last)  # GTF closed, 1-based -> BED half-open
            if feature == "gene":
                genes[chrom].append((start, end))
                continue
            transcript_id = attributes(raw).get("transcript_id")
            if not transcript_id:
                continue
            row = transcripts.setdefault(transcript_id, {"chrom": chrom, "strand": strand, "span": None, "exon": [], "CDS": [], "UTR": []})
            if feature == "transcript":
                row["span"] = (start, end)
            else:
                row[feature].append((start, end))

    beds = {name: defaultdict(list) for name in NAMES}
    for row in transcripts.values():
        chrom, strand = row["chrom"], row["strand"]
        exons = sorted(row["exon"])
        span = row["span"] or ((exons[0][0], exons[-1][1]) if exons else None)
        if span is None:
            continue
        start, end = span
        if strand == "+":
            beds["promoter"][chrom].append((max(0, start - 3000), start))
            beds["downstream"][chrom].append((end, min(end + 300, lengths[chrom])))
        elif strand == "-":
            beds["promoter"][chrom].append((end, min(end + 3000, lengths[chrom])))
            beds["downstream"][chrom].append((max(0, start - 300), start))
        beds["exon"][chrom].extend(exons)
        for left, right in zip(exons, exons[1:]):
            if left[1] < right[0]:
                beds["intron"][chrom].append((left[1], right[0]))
        if row["CDS"]:
            cds_start = min(a for a, _ in row["CDS"])
            cds_end = max(b for _, b in row["CDS"])
            for a, b in row["UTR"]:
                if a < cds_start:
                    beds["5utr" if strand == "+" else "3utr"][chrom].append((a, min(b, cds_start)))
                if b > cds_end:
                    beds["3utr" if strand == "+" else "5utr"][chrom].append((max(a, cds_end), b))

    # Intergenic means outside all annotated gene bodies and the chosen flanks.
    for chrom in sorted(CHROMOSOMES):
        occupied = genes[chrom] + beds["promoter"][chrom] + beds["downstream"][chrom]
        starts, ends = merge_intervals(occupied)
        cursor = 0
        for a, b in zip(starts, ends):
            if cursor < a:
                beds["distal_intergenic"][chrom].append((cursor, a))
            cursor = max(cursor, b)
        if cursor < lengths[chrom]:
            beds["distal_intergenic"][chrom].append((cursor, lengths[chrom]))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name in NAMES:
        path = args.output_dir / f"{name}.bed"
        with path.open("w", encoding="utf-8") as output:
            for chrom in sorted(CHROMOSOMES):
                starts, ends = merge_intervals(beds[name][chrom])
                for a, b in zip(starts, ends):
                    if a < b:
                        output.write(f"{chrom}\t{a}\t{b}\n")
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
