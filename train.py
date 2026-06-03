"""
Entry point for training Vision2Text from scratch.

Usage:
    python train.py

Everything is controlled through src/config.py (or .env / V2T_ environment
variables). The only thing you need before running this is the dataset and
the folder structure described in the README.
"""
from __future__ import annotations

from pathlib import Path

from src.config import cfg
from src.data.dataset import FlickrDataset, get_dataloader
from src.data.preprocessing import clean_caption
from src.data.vocab import build_vocab, load_vocab
from src.training.trainer import CaptioningModule, build_trainer


def _collect_train_captions() -> list[str]:
    """Read every training caption so we can build the BPE vocabulary from them.
    """
    from src.data.dataset import _parse_captions_file

    captions_path = cfg.data_dir / cfg.captions_file
    all_captions = _parse_captions_file(captions_path)

    # Only use training-split captions for the vocab — avoids data leakage.
    image_names = sorted(all_captions.keys())
    import random
    rng = random.Random(cfg.seed)
    rng.shuffle(image_names)
    n = len(image_names)
    train_names = set(image_names[: int(0.9 * n)])

    cleaned = []
    for name in train_names:
        for raw in all_captions[name]:
            c = clean_caption(raw)
            if c:
                cleaned.append(c)
    return cleaned


def main() -> None:
    # ---- 1. Vocabulary -------------------------------------------------------
    if cfg.vocab_path.exists():
        print(f"Vocab already exists at {cfg.vocab_path} — skipping rebuild.")
        tokenizer = load_vocab(cfg.vocab_path)
    else:
        print("Building BPE vocabulary from training captions…")
        captions = _collect_train_captions()
        tokenizer = build_vocab(captions, cfg.vocab_size, cfg.vocab_path)
        print(f"Vocab saved to {cfg.vocab_path}  ({tokenizer.get_vocab_size()} tokens)")

    # ---- 2. Data loaders -----------------------------------------------------
    print("Loading datasets…")
    train_loader = get_dataloader(cfg, split="train")
    val_loader   = get_dataloader(cfg, split="val")
    print(
        f"  train: {len(train_loader.dataset):,} samples"  # type: ignore[arg-type]
        f"  |  val: {len(val_loader.dataset):,} samples"   # type: ignore[arg-type]
    )

    # ---- 3. Model + trainer --------------------------------------------------
    steps_per_epoch = len(train_loader)
    module  = CaptioningModule(cfg, steps_per_epoch=steps_per_epoch)
    trainer = build_trainer(cfg, steps_per_epoch=steps_per_epoch)

    # ---- 4. Train ------------------------------------------------------------
    print(f"Training for up to {cfg.max_epochs} epochs…")
    print(f"Checkpoints will be saved to: {cfg.output_dir / 'checkpoints'}")
    trainer.fit(module, train_loader, val_loader)

    # ---- 5. Tell the user where the best checkpoint is -----------------------
    ckpt_dir = cfg.output_dir / "checkpoints"
    checkpoints = sorted(ckpt_dir.glob("best-*.ckpt"))
    if checkpoints:
        best = checkpoints[-1]
        print(f"\nTraining complete. Best checkpoint: {best}")
        print(f"\nTo run the backend, set:")
        print(f"  V2T_CHECKPOINT={best}")
        print(f"  V2T_VOCAB_PATH={cfg.vocab_path}")
    else:
        print("\nTraining complete. Check outputs/checkpoints/ for the saved model.")


if __name__ == "__main__":
    main()
