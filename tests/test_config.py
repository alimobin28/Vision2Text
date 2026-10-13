from pathlib import Path

import pytest

from src.config import ARCHITECTURE_FIELDS, Config


def test_env_var_overrides_default(monkeypatch):
    monkeypatch.setenv("V2T_D_MODEL", "64")
    assert Config(_env_file=None).d_model == 64


def test_dotenv_may_contain_serving_keys_and_unknown_keys(tmp_path):
    # Regression: V2T_CHECKPOINT / V2T_DEVICE / V2T_WORKERS used to be read with
    # os.getenv() and were not Config fields, so a .env holding them (exactly
    # what the old .env.example contained) made Config() raise.
    env = tmp_path / ".env"
    env.write_text("V2T_CHECKPOINT=weights/m.pt\nV2T_DEVICE=cpu\nV2T_WORKERS=3\nV2T_NOT_A_FIELD=1\n")
    c = Config(_env_file=env)
    assert c.checkpoint == Path("weights/m.pt")
    assert (c.device, c.workers) == ("cpu", 3)


def test_default_layouts_per_dataset():
    f8 = Config(_env_file=None, dataset_name="flickr8k", data_dir=Path("d"))
    assert f8.captions_path == Path("d/captions.txt")
    assert f8.images_dir == Path("d/Images")
    f30 = Config(_env_file=None, dataset_name="flickr30k", data_dir=Path("d"))
    assert f30.captions_path == Path("d/results.csv")
    assert f30.images_dir == Path("d/flickr30k_images")


def test_explicit_captions_and_images_override_layout():
    c = Config(_env_file=None, data_dir=Path("d"), captions_file="my.csv", images_subdir="pics")
    assert c.captions_path == Path("d/my.csv")
    assert c.images_dir == Path("d/pics")


def test_captions_path_resolves_kaggle_nested_layout(tmp_path):
    # Some Kaggle archives unpack to <data_dir>/captions.txt/captions.txt
    (tmp_path / "captions.txt").mkdir()
    (tmp_path / "captions.txt" / "captions.txt").write_text("image,caption\n")
    c = Config(_env_file=None, data_dir=tmp_path)
    assert c.captions_path == tmp_path / "captions.txt" / "captions.txt"


def test_unknown_dataset_name_rejected():
    with pytest.raises(ValueError):
        Config(_env_file=None, dataset_name="coco")


def test_architecture_contains_exactly_the_shape_fields():
    arch = Config(_env_file=None).architecture()
    assert tuple(arch) == ARCHITECTURE_FIELDS
    assert arch["d_model"] == 512
