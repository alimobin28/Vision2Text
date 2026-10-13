"""Shared test helpers: a tiny config, and a way to fabricate an exported model."""
from __future__ import annotations

from pathlib import Path

import torch
from tokenizers import Tokenizer

from src.config import Config
from src.models.captioner import ImageCaptioner

# Small enough that building/running the (randomly initialised) network takes
# a fraction of a second: 64px input -> 2x2 = 4 image patches.
TINY = dict(
    image_size=64,
    d_model=32,
    nhead=2,
    num_decoder_layers=1,
    dim_feedforward=64,
    max_caption_len=12,
    vocab_size=120,
    batch_size=4,
    num_workers=0,
    pretrained_backbone=False,   # never touch the network in tests
    max_epochs=2,
    warmup_steps=2,
    early_stopping_patience=0,
    augment=False,
)


def make_cfg(tmp_path: Path, **overrides) -> Config:
    """A Config that ignores .env and the shell's V2T_* variables where it matters."""
    values = {
        "data_dir": tmp_path / "data",
        "cache_dir": tmp_path / "cache",
        "vocab_path": tmp_path / "cache" / "vocab.json",
        "output_dir": tmp_path / "outputs",
        **TINY,
        **overrides,
    }
    return Config(_env_file=None, **values)


def write_random_export(cfg: Config, tokenizer: Tokenizer, path: Path) -> Path:
    """Save a randomly initialised model in the exported ``model.pt`` format."""
    model = ImageCaptioner(cfg, pad_id=tokenizer.token_to_id("<pad>"), pretrained=False)
    torch.save(
        {
            "format": 1,
            "architecture": cfg.architecture(),
            "tokenizer": tokenizer.to_str(),
            "state_dict": model.state_dict(),
        },
        path,
    )
    return path
