"""
Download Flickr8k from Kaggle (via ``kagglehub``) into ``V2T_DATA_DIR``.

    python -m src.data.download                 # -> ./data  (the default data_dir)
    python -m src.data.download --dest D:/flickr8k

``kagglehub`` stores the files in its own cache and returns that path; this
script then copies them into ``data_dir`` so the layout the rest of the project
expects exists:

    <data_dir>/Images/*.jpg
    <data_dir>/captions.txt

If Kaggle asks for credentials, set ``KAGGLE_USERNAME`` / ``KAGGLE_KEY`` or run
``kagglehub.login()`` once.
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from src.config import cfg

KAGGLE_SLUGS = {"flickr8k": "adityajn105/flickr8k"}


def download(dataset: str = "flickr8k", dest: Path | None = None) -> Path:
    """Fetch the dataset and make sure it is present under ``dest``."""
    if dataset not in KAGGLE_SLUGS:
        raise SystemExit(
            f"No download source is configured for '{dataset}'. Download it manually "
            "and point V2T_DATA_DIR / V2T_CAPTIONS_FILE / V2T_IMAGES_SUBDIR at it."
        )
    import kagglehub  # imported lazily: only this script needs it

    dest = Path(dest or cfg.data_dir)
    source = Path(kagglehub.dataset_download(KAGGLE_SLUGS[dataset]))
    print(f"kagglehub cache: {source}")

    if source.resolve() != dest.resolve():
        dest.mkdir(parents=True, exist_ok=True)
        shutil.copytree(source, dest, dirs_exist_ok=True)
        print(f"copied into:     {dest}")

    n_images = sum(1 for _ in (dest / "Images").glob("*.jpg")) if (dest / "Images").is_dir() else 0
    print(f"{n_images:,} images, captions file: {(dest / 'captions.txt').exists()}")
    return dest


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Download Flickr8k via kagglehub.")
    ap.add_argument("--dataset", default="flickr8k", choices=sorted(KAGGLE_SLUGS))
    ap.add_argument("--dest", type=Path, default=None, help="default: V2T_DATA_DIR (./data)")
    args = ap.parse_args()
    download(args.dataset, args.dest)
