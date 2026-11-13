"""Fetch published Z-DNABERT/GQ-DNABERT hg19 BEDs and lift them to hg38.

Run separately from training setup, on Linux x86_64:
  python scripts/fetch_flipon_predictions.py
  python scripts/fetch_flipon_predictions.py --verify-only

Raw author BEDs are kept unchanged. Lifted BEDs are derived annotations.
"""

from __future__ import annotations

import argparse
import hashlib
import platform
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ANNOTATIONS = ROOT / "data" / "annotations" / "flipons"
TOOL = ROOT / "data" / "tools" / "liftOver"
FILES = {
    ANNOTATIONS / "z_dnaberthg19.bed": (
        "https://raw.githubusercontent.com/mitiau/Z-DNABERT/main/beds/HG19_thr_0.25_minlen_12_pad.bed",
        "eeafe755958bdc85b12be292aa269fdb2450276b505204f80b0b14ee929798a3",
    ),
    ANNOTATIONS / "gq_dnaberthg19.bed": (
        "https://raw.githubusercontent.com/mitiau/G-DNABERT/master/beds/HG19_thr_0.25_minlen_6.bed",
        "bff9fb10f5e871edd90e932a0afdc8a12811918ccba8c8017d179759a91f0be1",
    ),
    ANNOTATIONS / "hg19ToHg38.over.chain.gz": (
        "https://hgdownload.soe.ucsc.edu/goldenPath/hg19/liftOver/hg19ToHg38.over.chain.gz",
        "5c0598e500ceb5a78c73086929e8ef993aec309bcafb595139b53d440b125a1d",
    ),
    TOOL: (
        "https://hgdownload.soe.ucsc.edu/admin/exe/linux.x86_64/liftOver",
        "87f202a0059b8bb6b4ab5db22f84c822413bab2dc22d3b6ad0e9e368e185bfec",
    ),
}
LIFTED = {
    "z_dnaberthg19": "a704c7081ff8a68b2b5b4b9fabf36c386cb4992616b4b851721bd8d350505525",
    "gq_dnaberthg19": "836f0d6228c5f300811b587b187f7da42a38b3cfb6f19889cc99359031e1826d",
}
PRIMARY = {
    "z_dnaberthg19": "86ada5eb66c5b6ab9eb79275d24db97b9cda827e42d3fbbb7a9f231d184093b1",
    "gq_dnaberthg19": "43c2926a161f9a486caa13dd2e8979366df8af5886bd05754687a23e2e14d073",
}
CHROMOSOMES = {f"chr{i}" for i in range(1, 23)} | {"chrX", "chrY"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def fetch(path: Path, url: str, expected: str, verify_only: bool) -> None:
    if path.is_file() and sha256(path) == expected:
        print(f"Verified {path.relative_to(ROOT)}")
        return
    if verify_only:
        raise ValueError(f"Missing or checksum mismatch: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.name}.", delete=False) as stream:
        temporary = Path(stream.name)
    try:
        request = urllib.request.Request(url, headers={"User-Agent": "rloops-research/1"})
        with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
            shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
        if sha256(temporary) != expected:
            raise ValueError(f"Downloaded file differs from pinned SHA-256: {url}")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
    print(f"Downloaded {path.relative_to(ROOT)}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    try:
        for path, (url, expected) in FILES.items():
            fetch(path, url, expected, args.verify_only)
        if platform.system() != "Linux" or platform.machine() not in ("x86_64", "AMD64"):
            raise ValueError("Pinned UCSC liftOver binary requires Linux x86_64")
        TOOL.chmod(TOOL.stat().st_mode | 0o111)
        for name, expected in LIFTED.items():
            output = ANNOTATIONS / f"{name.replace('hg19', 'hg38')}.bed"
            unmapped = ANNOTATIONS / f"{name}.unmapped"
            if not (output.is_file() and unmapped.is_file() and sha256(output) == expected):
                if args.verify_only:
                    raise ValueError(f"Missing or checksum mismatch: {output}")
                subprocess.run(
                    [str(TOOL), "-minMatch=0.95", str(ANNOTATIONS / f"{name}.bed"),
                     str(ANNOTATIONS / "hg19ToHg38.over.chain.gz"), str(output), str(unmapped)],
                    check=True,
                )
                if sha256(output) != expected:
                    raise ValueError(f"Lifted BED differs from pinned SHA-256: {output}")
                print(f"Lifted {output.relative_to(ROOT)}")
            else:
                print(f"Verified {output.relative_to(ROOT)}")
            primary = ANNOTATIONS / f"{name.replace('hg19', 'hg38')}.primary.bed"
            if primary.is_file() and sha256(primary) == PRIMARY[name]:
                print(f"Verified {primary.relative_to(ROOT)}")
                continue
            if args.verify_only:
                raise ValueError(f"Missing or checksum mismatch: {primary}")
            with output.open(encoding="utf-8") as source, primary.open("w", encoding="utf-8") as target:
                for line in source:
                    if line.split("\t", 1)[0] in CHROMOSOMES:
                        target.write(line)
            if sha256(primary) != PRIMARY[name]:
                raise ValueError(f"Primary-chromosome BED differs from pinned SHA-256: {primary}")
            print(f"Prepared {primary.relative_to(ROOT)}")
    except (OSError, ValueError, subprocess.CalledProcessError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
