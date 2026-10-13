from __future__ import annotations

import io

import pytest
from PIL import Image

from src.data.dataset import collect_train_captions
from src.data.toy import make_toy_dataset
from src.data.vocab import build_vocab
from tests.helpers import make_cfg


@pytest.fixture(scope="session")
def toy_root(tmp_path_factory):
    """A 40-image synthetic dataset in the Flickr8k layout (built once)."""
    return make_toy_dataset(tmp_path_factory.mktemp("toy") / "data", n_images=40)


@pytest.fixture(scope="session")
def toy_env(tmp_path_factory, toy_root):
    """(config, tokenizer) for the toy dataset. Treat as read-only."""
    root = tmp_path_factory.mktemp("toy_env")
    cfg = make_cfg(root, data_dir=toy_root)
    tokenizer = build_vocab(collect_train_captions(cfg), cfg.vocab_size, cfg.vocab_path)
    return cfg, tokenizer


@pytest.fixture
def png_bytes() -> bytes:
    """A small valid PNG, as the API would receive it."""
    buf = io.BytesIO()
    Image.new("RGB", (48, 40), (200, 30, 30)).save(buf, format="PNG")
    return buf.getvalue()
