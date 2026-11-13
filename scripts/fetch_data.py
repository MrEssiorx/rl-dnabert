"""Fetch known public inputs, reusing files only after checksum verification.

Run from any directory: python scripts/fetch_data.py [--with-genome].
The RLBase table has no confirmed public URL for this local version, so this
script verifies a manually supplied copy instead of guessing its provenance.
"""

from __future__ import annotations

import argparse
import gzip
import hashlib
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path


DATA = Path(__file__).resolve().parents[1] / "data"
PEAKS = {
    "rlregions_S96.narrowPeak": (
        "https://rlbase-data.s3.amazonaws.com/misc/rlregions_S96.narrowPeak",
        "49dae6fdf7d814044f2972801c52753844bb1eed5544e34cb741adf627b634f3",
    ),
    "rlregions_dRNH.narrowPeak": (
        "https://rlbase-data.s3.amazonaws.com/misc/rlregions_dRNH.narrowPeak",
        "20ec5e7b67c67232ac0511b48336f46a88b9346ad3648d37e0324a191b83bb73",
    ),
}
TABLE_SHA256 = "0c41899bdca78160b95928fb427141ce6321f47d0c7efddd1ef6aaa763ed14f9"
GENOME_URL = "https://hgdownload.soe.ucsc.edu/goldenPath/hg38/bigZips/hg38.fa.gz"
GENOME_MD5 = "1c9dcaddfa41027f17cd8f7a82c7293b"
GENOME_FA_SHA256 = "5be01555d98347fdb3714dc84c6f77c9d8bc774adcf32c6f7a8fa06f5baf5e51"


def digest(path: Path, algorithm: str = "sha256") -> str:
    hash_ = hashlib.new(algorithm)
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            hash_.update(chunk)
    return hash_.hexdigest()


def matches(path: Path, expected: str, algorithm: str = "sha256") -> bool:
    return path.is_file() and digest(path, algorithm) == expected


def fetch(path: Path, url: str, expected: str, algorithm: str = "sha256", verify_only: bool = False) -> None:
    if matches(path, expected, algorithm):
        print(f"OK, reused: {path.name}")
        return
    if verify_only:
        raise RuntimeError(f"Missing or checksum mismatch: {path}; expected {algorithm} {expected}")
    print(f"Downloading {path.name} from {url}", flush=True)
    with tempfile.NamedTemporaryFile(dir=DATA, prefix=f".{path.name}.", delete=False) as temp:
        temporary = Path(temp.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "rloops-research-fetch/1"})
        with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        actual = digest(temporary, algorithm)
        if actual != expected:
            raise RuntimeError(
                f"Downloaded {path.name} has {algorithm} {actual}, expected {expected}. "
                "The upstream file may have changed; inspect its provenance before updating the manifest."
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Verified: {path.name}")


def copy_s96_alias(verify_only: bool) -> None:
    source = DATA / "rlregions_S96.narrowPeak"
    target = DATA / "rloop_S96.narrowPeak"  # historical notebook input
    expected = PEAKS[source.name][1]
    if matches(target, expected):
        print(f"OK, reused: {target.name}")
        return
    if verify_only:
        raise RuntimeError(f"Missing or checksum mismatch: {target}")
    shutil.copyfile(source, target)
    print(f"Copied alias: {target.name}")


def extract_genome(verify_only: bool) -> None:
    source = DATA / "hg38.fa.gz"
    target = DATA / "hg38.fa"
    if matches(target, GENOME_FA_SHA256):
        print(f"OK, reused: {target.name}")
    elif verify_only:
        raise RuntimeError(f"Missing or checksum mismatch: {target}")
    else:
        print("Extracting hg38.fa", flush=True)
        with tempfile.NamedTemporaryFile(dir=DATA, prefix=".hg38.fa.", delete=False) as temp:
            temporary = Path(temp.name)
        try:
            with gzip.open(source, "rb") as compressed, temporary.open("wb") as output:
                shutil.copyfileobj(compressed, output, length=8 * 1024 * 1024)
            if digest(temporary) != GENOME_FA_SHA256:
                raise RuntimeError("Extracted hg38.fa checksum mismatch")
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)
    index = DATA / "hg38.fa.fai"
    if index.is_file() and index.stat().st_mtime >= target.stat().st_mtime:
        print(f"OK, reused: {index.name}")
    elif verify_only:
        raise RuntimeError(f"Missing or stale FASTA index: {index}")
    elif shutil.which("samtools"):
        subprocess.run(["samtools", "faidx", str(target)], check=True)
        print(f"Indexed: {index.name}")
    else:
        print("FASTA is ready. Install samtools and run `samtools faidx data/hg38.fa` for random access.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-genome", action="store_true", help="also get and extract UCSC hg38 (about 4 GB on disk)")
    parser.add_argument("--verify-only", action="store_true", help="check local files without downloading or changing them")
    parser.add_argument("--require-table", action="store_true", help="fail if the manually supplied RLBase table is absent")
    args = parser.parse_args()
    if not args.verify_only:
        DATA.mkdir(exist_ok=True)
    try:
        for name, (url, expected) in PEAKS.items():
            fetch(DATA / name, url, expected, verify_only=args.verify_only)
        copy_s96_alias(args.verify_only)
        table = DATA / "rlregions_table.rda"
        if table.is_file():
            if not matches(table, TABLE_SHA256):
                raise RuntimeError(f"rlregions_table.rda differs from the audited local version: {table}")
            print(f"OK, verified local table: {table.name}")
        elif args.require_table:
            raise RuntimeError("rlregions_table.rda is absent; supply the audited file manually")
        else:
            print("Table absent: rlregions_table.rda has no confirmed download URL; supply it manually if needed.")
        if args.with_genome:
            fetch(DATA / "hg38.fa.gz", GENOME_URL, GENOME_MD5, "md5", args.verify_only)
            extract_genome(args.verify_only)
    except (OSError, RuntimeError, urllib.error.URLError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
