"""Fetch the pinned SCREEN Registry V4 hg38 enhancer-like cCRE BED files.

Run separately from training-data setup:
  python scripts/fetch_enhancers.py
  python scripts/fetch_enhancers.py --verify-only
"""

from __future__ import annotations

import argparse
import hashlib
import shutil
import sys
import tempfile
import urllib.request
from pathlib import Path


OUTPUT = Path(__file__).resolve().parents[1] / "data" / "annotations"
FILES = {
    "pELS": (
        "https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.pELS.bed",
        "1eff7396a19acefe2e0a342279dc752c9af301e5ff553dff9cc4b5142f53c35c",
    ),
    "dELS": (
        "https://downloads.wenglab.org/Registry-V4/GRCh38-cCREs.dELS.bed",
        "7a9bdb7ddd69cda1b95fe6dcddd2843d2a6638207e9766a3729c5e869592fa60",
    ),
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if not args.verify_only:
        OUTPUT.mkdir(parents=True, exist_ok=True)
    try:
        for label, (url, expected) in FILES.items():
            path = OUTPUT / f"encode_v4_{label}.bed"
            if path.is_file() and sha256(path) == expected:
                print(f"Verified {label}: {path}")
                continue
            if args.verify_only:
                raise ValueError(f"Missing or checksum mismatch: {path}")
            with tempfile.NamedTemporaryFile(dir=OUTPUT, prefix=f".{label}.", delete=False) as temp:
                temporary = Path(temp.name)
            try:
                request = urllib.request.Request(url, headers={"User-Agent": "rloops-research/1"})
                with urllib.request.urlopen(request, timeout=60) as response, temporary.open("wb") as output:
                    shutil.copyfileobj(response, output, length=8 * 1024 * 1024)
                if sha256(temporary) != expected:
                    raise ValueError(f"Downloaded {label} differs from pinned SCREEN Registry V4 SHA-256")
                temporary.replace(path)
            finally:
                temporary.unlink(missing_ok=True)
            print(f"Downloaded {label}: {path}")
    except (OSError, ValueError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
