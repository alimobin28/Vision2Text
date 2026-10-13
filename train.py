"""
Entry point for training Vision2Text from scratch.

Usage:
    python train.py                  # build vocab (if missing), train, export
    python train.py --rebuild-vocab  # force the BPE vocabulary to be rebuilt
    python train.py --resume         # continue an interrupted run from last.ckpt

Everything is controlled through src/config.py (or .env / V2T_ environment
variables). The only things you need before running this are the dataset and
the folder layout described in the README ("Dataset").
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from src.config import cfg
from src.data.dataset import collect_train_captions, get_dataloader
from src.data.vocab import build_vocab, load_vocab
from src.training.export import export_checkpoint
from src.training.trainer import CaptioningModule, build_trainer


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the Vision2Text captioner.")
    parser.add_argument(
        "--rebuild-vocab", action="store_true",
        help="ignore any existing vocabulary file and train a new BPE vocab",
    )
    parser.add_argument(
        "--resume", nargs="?", const="last", default=None, metavar="CKPT",
        help="continue an interrupted run from a checkpoint (weights, optimiser, "
             "LR schedule and epoch counter are all restored). With no value, uses "
             "<output_dir>/checkpoints/last.ckpt. Keep V2T_MAX_EPOCHS unchanged.",
    )
    args = parser.parse_args()

    resume_from: Path | None = None
    if args.resume is not None:
        resume_from = (
            cfg.output_dir / "checkpoints" / "last.ckpt"
            if args.resume == "last" else Path(args.resume)
        )
        if not resume_from.is_file():
            raise SystemExit(f"Cannot resume: checkpoint not found: {resume_from}")

    # Seeds python, numpy, torch and the DataLoader workers.
    pl.seed_everything(cfg.seed, workers=True)

    # ---- 1. Vocabulary -------------------------------------------------------
    # Built from TRAIN-split captions only, using the very same split function
    # the dataloaders use, so no validation/test caption can leak into it.
    if cfg.vocab_path.exists() and not args.rebuild_vocab:
        print(f"Vocab already exists at {cfg.vocab_path} — reusing it "
              "(pass --rebuild-vocab to change it).")
        tokenizer = load_vocab(cfg.vocab_path)
    else:
        print("Building BPE vocabulary from training captions…")
        captions = collect_train_captions(cfg)
        tokenizer = build_vocab(captions, cfg.vocab_size, cfg.vocab_path)
        print(f"Vocab saved to {cfg.vocab_path}  ({tokenizer.get_vocab_size()} tokens "
              f"from {len(captions):,} captions)")

    # ---- 2. Data loaders -----------------------------------------------------
    print("Loading datasets…")
    train_loader = get_dataloader(cfg, split="train")
    val_loader = get_dataloader(cfg, split="val")
    print(
        f"  train: {len(train_loader.dataset):,} caption pairs"  # type: ignore[arg-type]
        f"  |  val: {len(val_loader.dataset):,} images"          # type: ignore[arg-type]
    )

    # ---- 3. Model + trainer --------------------------------------------------
    steps_per_epoch = len(train_loader)
    module = CaptioningModule(cfg, steps_per_epoch=steps_per_epoch)
    trainer = build_trainer(cfg, steps_per_epoch=steps_per_epoch)

    # ---- 4. Train ------------------------------------------------------------
    print(f"Training for up to {cfg.max_epochs} epochs…")
    print(f"Checkpoints will be saved to: {cfg.output_dir / 'checkpoints'}")
    if resume_from is not None:
        print(f"Resuming from {resume_from}")
    trainer.fit(
        module, train_loader, val_loader,
        ckpt_path=str(resume_from) if resume_from is not None else None,
    )

    # ---- 5. Export the best checkpoint into the single file the API serves ---
    checkpoint_cb = trainer.checkpoint_callback
    best = checkpoint_cb.best_model_path if isinstance(checkpoint_cb, ModelCheckpoint) else ""
    if not best:
        print("\nTraining finished but no checkpoint was written "
              "(was validation skipped?). Check outputs/checkpoints/.")
        return

    exported = export_checkpoint(Path(best), cfg.output_dir / "model.pt")
    print("\nTraining complete.")
    print(f"  best checkpoint : {best}")
    print(f"  exported model  : {exported}   (weights + architecture + tokenizer)")
    print("\nNext steps:")
    print("  evaluate on the held-out test split:")
    print(f"      python -m src.evaluation.evaluate --checkpoint {exported}")
    print("  serve it:")
    print(f"      V2T_CHECKPOINT={exported} uvicorn src.api.app:app --port 8000")


if __name__ == "__main__":
    main()
