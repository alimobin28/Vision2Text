
import pytest
import torch

from src.data.dataset import (
    FlickrDataset,
    FlickrEvalDataset,
    collect_train_captions,
    get_dataloader,
    load_splits,
    parse_captions,
    split_image_names,
)
from src.data.vocab import BOS_TOKEN
from tests.helpers import make_cfg

# ---------------------------------------------------------------- parsing ----

def test_parse_flickr8k_csv(tmp_path):
    f = tmp_path / "captions.txt"
    f.write_text(
        "image,caption\n"
        "a.jpg,A dog, running fast .\n"      # unquoted comma inside the caption
        'b.jpg,"Quoted, caption"\n'          # quoted
        "\n"                                  # blank line
        "a.jpg,Second\n",
        encoding="utf-8-sig",                 # BOM must not corrupt the header
    )
    assert parse_captions(f, "flickr8k") == {
        "a.jpg": ["A dog, running fast .", "Second"],
        "b.jpg": ["Quoted, caption"],
    }


def test_parse_flickr30k_pipe_format(tmp_path):
    f = tmp_path / "results.csv"
    f.write_text(
        "image_name| comment_number| comment\n"
        "1.jpg| 0| Two young guys\n"
        "1.jpg| 1| Two men\n"
        "2.jpg| 4\n"                          # malformed row is skipped
    )
    assert parse_captions(f, "flickr30k") == {"1.jpg": ["Two young guys", "Two men"]}


def test_parse_original_token_format(tmp_path):
    f = tmp_path / "Flickr8k.token.txt"
    f.write_text("x.jpg#0\tA cat\nx.jpg#1\tA kitten\n")
    assert parse_captions(f, "flickr8k") == {"x.jpg": ["A cat", "A kitten"]}


def test_parse_missing_or_empty_file(tmp_path):
    with pytest.raises(FileNotFoundError, match="Caption file not found"):
        parse_captions(tmp_path / "nope.txt")
    empty = tmp_path / "e.txt"
    empty.write_text("image,caption\n")
    with pytest.raises(ValueError):
        parse_captions(empty)


# -------------------------------------------------------------- splitting ----

NAMES = [f"img{i:03d}.jpg" for i in range(100)]


def test_split_is_disjoint_and_complete():
    s = split_image_names(NAMES, seed=1, val_fraction=0.1, test_fraction=0.2)
    assert [len(s[k]) for k in ("train", "val", "test")] == [70, 10, 20]
    assert sorted(s["train"] + s["val"] + s["test"]) == NAMES


def test_split_is_deterministic_and_order_independent():
    a = split_image_names(NAMES, seed=7)
    b = split_image_names(list(reversed(NAMES)), seed=7)
    assert a == b
    assert a != split_image_names(NAMES, seed=8)


def test_split_tiny_dataset_keeps_a_training_image():
    for n in (1, 2, 3, 4):
        s = split_image_names(NAMES[:n], seed=0)
        assert len(s["train"]) >= 1
        assert sum(map(len, s.values())) == n


def test_split_zero_fractions():
    s = split_image_names(NAMES, seed=0, val_fraction=0, test_fraction=0)
    assert len(s["train"]) == 100 and not s["val"] and not s["test"]


# --------------------------------------------------------- on the toy data ----

def test_load_splits_no_leakage_and_cleaned(toy_env, toy_root):
    cfg, _ = toy_env
    splits = load_splits(cfg)
    train, val, test = (set(splits[k]) for k in ("train", "val", "test"))
    assert not (train & val) and not (train & test) and not (val & test)
    assert len(train | val | test) == 40
    for captions in splits["train"].values():
        assert len(captions) == 5
        for c in captions:
            assert c == c.lower() and "," not in c and "." not in c and "1" not in c.split()


def test_vocab_corpus_is_train_split_only(toy_env):
    cfg, _ = toy_env
    corpus = collect_train_captions(cfg)
    n_train_images = len(load_splits(cfg)["train"])
    assert len(corpus) == 5 * n_train_images


def test_max_images_limits_the_dataset(toy_root, tmp_path):
    cfg = make_cfg(tmp_path, data_dir=toy_root, max_images=10)
    assert sum(len(v) for v in load_splits(cfg).values()) == 10


def test_missing_image_files_are_skipped_with_warning(toy_root, tmp_path):
    import shutil
    data = tmp_path / "data"
    shutil.copytree(toy_root, data)
    (data / "Images" / "toy_0000.jpg").unlink()
    cfg = make_cfg(tmp_path, data_dir=data)
    with pytest.warns(UserWarning, match="not found"):
        splits = load_splits(cfg)
    assert "toy_0000.jpg" not in {n for s in splits.values() for n in s}


def test_missing_image_directory_is_a_clear_error(tmp_path):
    (tmp_path / "data").mkdir()
    (tmp_path / "data" / "captions.txt").write_text("image,caption\na.jpg,x\n")
    with pytest.raises(FileNotFoundError, match="Image directory not found"):
        load_splits(make_cfg(tmp_path))


def test_train_dataset_items(toy_env):
    cfg, tok = toy_env
    ds = FlickrDataset(cfg, "train", tokenizer=tok)
    image, ids = ds[0]
    assert image.shape == (3, cfg.image_size, cfg.image_size) and image.dtype == torch.float32
    assert ids.shape == (cfg.max_caption_len,) and ids.dtype == torch.long
    assert ids[0].item() == tok.token_to_id(BOS_TOKEN)
    assert len(ds) == 5 * len(load_splits(cfg)["train"])


def test_eval_dataset_returns_all_references(toy_env):
    cfg, _ = toy_env
    ds = FlickrEvalDataset(cfg, "val")
    image, refs = ds[0]
    assert image.shape == (3, cfg.image_size, cfg.image_size)
    assert len(refs) == 5 and all(isinstance(r, str) for r in refs)
    assert len(ds) == len(load_splits(cfg)["val"])    # one sample per IMAGE


def test_dataloaders(toy_env):
    cfg, _ = toy_env
    images, ids = next(iter(get_dataloader(cfg, "train")))
    assert images.shape[1:] == (3, cfg.image_size, cfg.image_size) and ids.shape[1] == cfg.max_caption_len
    images, refs = next(iter(get_dataloader(cfg, "val")))
    assert len(refs) == images.size(0) and isinstance(refs[0], list)


def test_invalid_split_name(toy_env):
    cfg, tok = toy_env
    with pytest.raises(ValueError):
        FlickrDataset(cfg, "dev", tokenizer=tok)
