"""
End-to-end: train a tiny model on the toy dataset, export it, serve it, evaluate it.

This is the test that proves the pieces fit together (and that exercised the
checkpoint-loading, config and decoder bugs that unit tests alone had missed).
"""
from pathlib import Path

import pytest
import pytorch_lightning as pl
import torch

from src.data.dataset import collect_train_captions, get_dataloader
from src.data.vocab import build_vocab
from src.evaluation.evaluate import evaluate
from src.inference.engine import InferenceEngine
from src.training.export import export_checkpoint
from src.training.trainer import CaptioningModule, build_trainer
from tests.helpers import make_cfg


@pytest.fixture(scope="module")
def trained(tmp_path_factory, toy_root):
    tmp = tmp_path_factory.mktemp("pipeline")
    cfg = make_cfg(tmp, data_dir=toy_root, max_epochs=2)

    build_vocab(collect_train_captions(cfg), cfg.vocab_size, cfg.vocab_path)
    train_loader, val_loader = get_dataloader(cfg, "train"), get_dataloader(cfg, "val")

    module = CaptioningModule(cfg, steps_per_epoch=len(train_loader))
    trainer = build_trainer(cfg, steps_per_epoch=len(train_loader))
    trainer.fit(module, train_loader, val_loader)

    best = Path(trainer.checkpoint_callback.best_model_path)
    return cfg, trainer, best, tmp


def test_training_writes_a_checkpoint_and_logs_bleu(trained):
    cfg, trainer, best, _ = trained
    assert best.is_file() and best.parent == cfg.output_dir / "checkpoints"
    assert "val_bleu4" in trainer.callback_metrics
    assert 0.0 <= float(trainer.callback_metrics["val_bleu4"]) <= 1.0
    # filename pattern documented in the README: best-<epoch>-<bleu>.ckpt
    assert best.name.startswith("best-") and "=" not in best.name


def test_lightning_checkpoint_loads_under_weights_only(trained):
    """Regression: hyperparameters used to contain pathlib.Path objects, which the
    restricted unpickler (weights_only=True) refuses."""
    _, _, best, _ = trained
    state = torch.load(best, map_location="cpu", weights_only=True)
    assert "state_dict" in state and "v2t" in state
    assert state["hyper_parameters"]["data_dir"].__class__ is str


def test_engine_serves_raw_checkpoint_and_export_identically(trained, png_bytes):
    cfg, _, best, tmp = trained
    exported = export_checkpoint(best, tmp / "model.pt")

    raw = InferenceEngine(best, device="cpu", config=cfg).caption(png_bytes, beam_size=3)
    exp = InferenceEngine(exported, device="cpu", config=cfg).caption(png_bytes, beam_size=3)

    assert raw["caption"] == exp["caption"]
    assert raw["tokens"] == exp["tokens"]
    assert len(exp["tokens"]) == len(exp["attention_weights"])


def test_exported_file_is_plain_data(trained):
    _, _, best, tmp = trained
    payload = torch.load(export_checkpoint(best, tmp / "plain.pt"), weights_only=True)
    assert set(payload) == {"format", "architecture", "tokenizer", "state_dict"}
    assert not any(k.startswith("model.") for k in payload["state_dict"])


def test_export_rejects_foreign_checkpoints(tmp_path):
    torch.save({"state_dict": {}}, tmp_path / "other.ckpt")
    with pytest.raises(ValueError, match="missing 'v2t' metadata"):
        export_checkpoint(tmp_path / "other.ckpt", tmp_path / "out.pt")


class _StopAfterFirstEpoch(pl.Callback):
    """Simulates a Ctrl-C / crash between epoch 1 and 2."""

    def on_train_epoch_end(self, trainer, pl_module):
        trainer.should_stop = True


def test_interrupted_run_resumes_where_it_stopped(tmp_path, toy_root):
    cfg = make_cfg(tmp_path, data_dir=toy_root, max_epochs=2)
    build_vocab(collect_train_captions(cfg), cfg.vocab_size, cfg.vocab_path)
    train_loader, val_loader = get_dataloader(cfg, "train"), get_dataloader(cfg, "val")
    steps = len(train_loader)

    # --- first run: interrupted after epoch 1 of 2 ---
    first = build_trainer(cfg, steps_per_epoch=steps)
    first.callbacks.append(_StopAfterFirstEpoch())
    first.fit(CaptioningModule(cfg, steps_per_epoch=steps), train_loader, val_loader)
    assert (first.current_epoch, first.global_step) == (1, steps)

    last = cfg.output_dir / "checkpoints" / "last.ckpt"
    assert last.is_file()

    # --- second run: a fresh process in real life; resume from last.ckpt ---
    second = build_trainer(cfg, steps_per_epoch=steps)
    second.fit(
        CaptioningModule(cfg, steps_per_epoch=steps), train_loader, val_loader, ckpt_path=str(last)
    )
    assert second.current_epoch == 2                 # finished epoch 2, did not redo epoch 1
    assert second.global_step == 2 * steps           # step counter continued, not restarted


def test_offline_evaluation_runs_and_reports_all_metrics(trained):
    cfg, _, best, tmp = trained
    exported = export_checkpoint(best, tmp / "eval_model.pt")
    result = evaluate(exported, cfg, split="test", beam_size=2, device="cpu")

    assert {"BLEU-1", "BLEU-4", "ROUGE-L", "CIDEr-D"} <= set(result["scores"])
    assert result["meta"]["n_images"] == len(result["predictions"]) > 0
    first = result["predictions"][0]
    assert isinstance(first["hypothesis"], str) and len(first["references"]) == 5
