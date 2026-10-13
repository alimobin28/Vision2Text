"""
Flickr8k / Flickr30k loading.

Pipeline
--------
caption file --parse_captions--> {image: [raw captions]}
             --load_splits-----> {"train"|"val"|"test": {image: [clean captions]}}
             --FlickrDataset / FlickrEvalDataset--> tensors

The split is made BY IMAGE with a seeded shuffle in exactly one place
(:func:`split_image_names`). ``train.py`` builds the BPE vocabulary from the
same train split, so neither the vocabulary nor the model ever sees a
validation or test caption.
"""
from __future__ import annotations

import csv
import random
import warnings
from collections.abc import Callable
from pathlib import Path

import torch
from PIL import Image
from tokenizers import Tokenizer
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

from src.config import Config
from src.data.preprocessing import clean_caption
from src.data.transforms import build_transform
from src.data.vocab import encode, load_vocab

SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# caption file parsing
# ---------------------------------------------------------------------------

def _split_line(line: str, dataset_name: str) -> tuple[str, str] | None:
    """Return ``(image_name, caption)`` or None for headers / malformed rows."""
    line = line.strip()
    if not line:
        return None

    # Original Flickr token format:  "1000268201_693b08cb0e.jpg#0<TAB>caption"
    if "\t" in line:
        key, caption = line.split("\t", 1)
        return key.split("#")[0].strip(), caption.strip()

    if dataset_name == "flickr30k":
        # results.csv:  "image_name| comment_number| comment"
        parts = [p.strip() for p in line.split("|", 2)]
        if len(parts) < 3 or parts[0] == "image_name":
            return None
        return parts[0], parts[2]

    # Kaggle Flickr8k captions.txt:  "image,caption"  (captions contain commas,
    # and may or may not be quoted — csv handles both)
    row = next(csv.reader([line]))
    if len(row) < 2 or row[0] == "image":
        return None
    return row[0].strip(), ",".join(row[1:]).strip()


def parse_captions(path: Path, dataset_name: str = "flickr8k") -> dict[str, list[str]]:
    """Read a caption file into ``{image_name: [raw captions]}``."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(
            f"Caption file not found: {path}. Download the dataset and/or set "
            "V2T_DATA_DIR / V2T_CAPTIONS_FILE (see README, 'Dataset')."
        )

    captions: dict[str, list[str]] = {}
    with open(path, encoding="utf-8-sig") as fh:
        for line in fh:
            parsed = _split_line(line, dataset_name)
            if parsed is None or not parsed[1]:
                continue
            captions.setdefault(parsed[0], []).append(parsed[1])
    if not captions:
        raise ValueError(f"No captions could be parsed from {path} as {dataset_name}.")
    return captions


# ---------------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------------

def split_image_names(
    names: list[str],
    seed: int,
    val_fraction: float = 0.1,
    test_fraction: float = 0.1,
) -> dict[str, list[str]]:
    """Deterministic train/val/test split over *images*.

    Sorted first so the result does not depend on file ordering, then shuffled
    with a seeded RNG. Non-zero fractions always get at least one image (as long
    as at least one image is left to train on).
    """
    ordered = sorted(set(names))
    random.Random(seed).shuffle(ordered)
    n = len(ordered)

    def _count(fraction: float) -> int:
        if fraction <= 0 or n < 3:
            return 0
        return max(1, round(n * fraction))

    n_val, n_test = _count(val_fraction), _count(test_fraction)
    while n_val + n_test >= n and (n_val or n_test):   # keep >= 1 train image
        if n_val >= n_test:
            n_val -= 1
        else:
            n_test -= 1

    n_train = n - n_val - n_test
    return {
        "train": ordered[:n_train],
        "val": ordered[n_train : n_train + n_val],
        "test": ordered[n_train + n_val :],
    }


def load_splits(cfg: Config) -> dict[str, dict[str, list[str]]]:
    """``{split: {image_name: [cleaned captions]}}`` for the configured dataset.

    Images that are listed in the caption file but missing on disk are dropped
    with a warning; an entirely missing image directory is an error.
    """
    images_dir = cfg.images_dir
    if not images_dir.is_dir():
        raise FileNotFoundError(
            f"Image directory not found: {images_dir}. Check V2T_DATA_DIR / "
            "V2T_IMAGES_SUBDIR (see README, 'Dataset')."
        )

    raw = parse_captions(cfg.captions_path, cfg.dataset_name)

    names = sorted(raw)
    if cfg.max_images > 0:
        names = names[: cfg.max_images]

    present = [n for n in names if (images_dir / n).is_file()]
    if len(present) < len(names):
        warnings.warn(
            f"{len(names) - len(present)} of {len(names)} images listed in the "
            f"caption file were not found in {images_dir} and were skipped.",
            stacklevel=2,
        )
    if not present:
        raise FileNotFoundError(f"None of the listed images exist in {images_dir}.")

    name_splits = split_image_names(
        present, cfg.seed, cfg.val_fraction, cfg.test_fraction
    )

    out: dict[str, dict[str, list[str]]] = {}
    for split, split_names in name_splits.items():
        out[split] = {}
        for name in split_names:
            cleaned = [c for c in (clean_caption(r) for r in raw[name]) if c]
            if cleaned:
                out[split][name] = cleaned
    return out


def collect_train_captions(cfg: Config) -> list[str]:
    """Every cleaned TRAIN caption — the corpus the BPE vocabulary is built from."""
    train = load_splits(cfg)["train"]
    return [c for captions in train.values() for c in captions]


# ---------------------------------------------------------------------------
# datasets
# ---------------------------------------------------------------------------

def _load_image(path: Path, transform: Callable[[Image.Image], Tensor]) -> Tensor:
    with Image.open(path) as img:
        return transform(img.convert("RGB"))


class FlickrDataset(Dataset):
    """Training dataset: one sample per (image, caption) pair.

    ``__getitem__`` returns ``(image [3,H,W], caption_ids [max_caption_len])``.
    """

    def __init__(
        self,
        cfg: Config,
        split: str = "train",
        tokenizer: Tokenizer | None = None,
        splits: dict[str, dict[str, list[str]]] | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if tokenizer is None:
            tokenizer = load_vocab(cfg.vocab_path)
        if splits is None:
            splits = load_splits(cfg)

        self.images_dir = cfg.images_dir
        self.transform = build_transform(cfg.image_size, train=split == "train" and cfg.augment)
        self.samples: list[tuple[str, list[int]]] = [
            (name, encode(tokenizer, caption, cfg.max_caption_len))
            for name, captions in splits[split].items()
            for caption in captions
        ]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[Tensor, Tensor]:
        name, ids = self.samples[idx]
        image = _load_image(self.images_dir / name, self.transform)
        return image, torch.tensor(ids, dtype=torch.long)


class FlickrEvalDataset(Dataset):
    """Validation / test dataset: one sample per IMAGE with all its references.

    ``__getitem__`` returns ``(image [3,H,W], [reference captions])``. Scoring a
    generated caption against all (up to five) references is what makes BLEU
    meaningful; one reference per hypothesis badly deflates it.
    """

    def __init__(
        self,
        cfg: Config,
        split: str = "val",
        splits: dict[str, dict[str, list[str]]] | None = None,
    ) -> None:
        if split not in SPLITS:
            raise ValueError(f"split must be one of {SPLITS}, got {split!r}")
        if splits is None:
            splits = load_splits(cfg)

        self.images_dir = cfg.images_dir
        self.transform = build_transform(cfg.image_size, train=False)
        self.image_names: list[str] = list(splits[split])
        self.references: list[list[str]] = [splits[split][n] for n in self.image_names]

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int) -> tuple[Tensor, list[str]]:
        image = _load_image(self.images_dir / self.image_names[idx], self.transform)
        return image, self.references[idx]


def eval_collate(batch: list[tuple[Tensor, list[str]]]) -> tuple[Tensor, list[list[str]]]:
    """Stack images; keep the variable-length reference lists as plain lists."""
    return torch.stack([b[0] for b in batch]), [b[1] for b in batch]


def get_dataloader(cfg: Config, split: str = "train") -> DataLoader:
    """Loader for ``train`` (pairs, shuffled), ``val`` or ``test`` (per image)."""
    splits = load_splits(cfg)
    workers = cfg.num_workers

    if split == "train":
        return DataLoader(
            FlickrDataset(cfg, "train", splits=splits),
            batch_size=cfg.batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(cfg.seed),
            num_workers=workers,
            pin_memory=torch.cuda.is_available(),
            persistent_workers=workers > 0,
        )
    return DataLoader(
        FlickrEvalDataset(cfg, split, splits=splits),
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=eval_collate,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        persistent_workers=workers > 0,
    )
