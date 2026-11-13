"""Download the exact DNABERT-2 checkpoint used in this project."""

from pathlib import Path

from huggingface_hub import snapshot_download


MODEL_ID = "zhihan1996/DNABERT-2-117M"
REVISION = "7bce263b15377fc15361f52cfab88f8b586abda0"
MODEL_DIR = Path("data/models/DNABERT-2-117M") / REVISION


def main() -> None:
    path = snapshot_download(
        repo_id=MODEL_ID,
        revision=REVISION,
        local_dir=MODEL_DIR,
    )
    print(path)


if __name__ == "__main__":
    main()
