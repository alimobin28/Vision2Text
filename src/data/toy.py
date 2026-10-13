"""
A tiny synthetic captioning dataset in the Flickr8k Kaggle layout.

Exists so the *whole* pipeline (split -> vocab -> train -> export -> serve ->
evaluate) can be exercised in seconds without downloading Flickr — in tests,
in CI, and as a first sanity check on a new machine:

    python -m src.data.toy data
    V2T_MAX_EPOCHS=3 V2T_BATCH_SIZE=8 V2T_NUM_WORKERS=0 python train.py

Each image is a coloured shape on a coloured background, with five captions that
are deliberately messy (capitals, commas, apostrophes, digits) to exercise the
caption cleaner.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw

COLORS = {
    "red": (220, 40, 40), "green": (40, 170, 70), "blue": (40, 80, 220),
    "yellow": (240, 210, 40), "purple": (150, 60, 190), "orange": (240, 140, 30),
}
SHAPES = ("circle", "square", "triangle")


def _draw(shape: str, fg: tuple, bg: tuple, size: int = 96) -> Image.Image:
    img = Image.new("RGB", (size, size), bg)
    d = ImageDraw.Draw(img)
    m = size // 4
    if shape == "circle":
        d.ellipse([m, m, size - m, size - m], fill=fg)
    elif shape == "square":
        d.rectangle([m, m, size - m, size - m], fill=fg)
    else:
        d.polygon([(size // 2, m), (m, size - m), (size - m, size - m)], fill=fg)
    return img


def _captions(shape: str, fg: str, bg: str) -> list[str]:
    return [
        f"A {fg} {shape} on a {bg} background.",
        f"{bg.capitalize()} background, with a {fg} {shape}",
        f"There is a {fg} {shape} in the picture 1",
        f"The {shape}'s colour is {fg}",
        f"a {fg.upper()} {shape} on {bg}",
    ]


def make_toy_dataset(data_dir: Path, n_images: int = 60, seed: int = 0) -> Path:
    """Write ``Images/*.jpg`` and ``captions.txt`` under ``data_dir``."""
    data_dir = Path(data_dir)
    images_dir = data_dir / "Images"
    images_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(seed)
    rows = ["image,caption"]
    for i in range(n_images):
        shape = rng.choice(SHAPES)
        fg, bg = rng.sample(sorted(COLORS), 2)
        name = f"toy_{i:04d}.jpg"
        _draw(shape, COLORS[fg], COLORS[bg]).save(images_dir / name, quality=95)
        rows += [f"{name},{c}" for c in _captions(shape, fg, bg)]

    (data_dir / "captions.txt").write_text("\n".join(rows) + "\n", encoding="utf-8")
    return data_dir


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Create a tiny synthetic captioning dataset.")
    ap.add_argument("data_dir", type=Path)
    ap.add_argument("--images", type=int, default=60)
    args = ap.parse_args()
    print(f"Wrote toy dataset to {make_toy_dataset(args.data_dir, args.images)}")
