"""
Turn a Lightning checkpoint into the single file the API serves.

    python -m src.training.export outputs/checkpoints/best-03-0.2150.ckpt outputs/model.pt

The exported ``model.pt`` contains only plain tensors, strings and numbers:

    {"format": 1,
     "architecture": {...},        # everything needed to rebuild the network
     "tokenizer": "<tokenizers JSON>",
     "state_dict": {...}}          # model weights, no Lightning "model." prefix

so it loads with ``torch.load(weights_only=True)`` on any machine and OS, needs
no separate vocab file, and cannot disagree with the environment it is served in.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

EXPORT_FORMAT = 1


def export_checkpoint(ckpt_path: Path, out_path: Path) -> Path:
    """Convert a Lightning ``.ckpt`` produced by ``train.py`` into ``model.pt``.

    The checkpoint is read with ``weights_only=False``, so only point this at a
    checkpoint you trained yourself.
    """
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    meta = ckpt.get("v2t")
    if meta is None:
        raise ValueError(
            f"{ckpt_path} was not written by this project's CaptioningModule "
            "(missing 'v2t' metadata); retrain, or export from a newer run."
        )

    state_dict = {
        k.removeprefix("model."): v
        for k, v in ckpt["state_dict"].items()
        if k.startswith("model.")
    }
    payload = {
        "format": EXPORT_FORMAT,
        "architecture": meta["architecture"],
        "tokenizer": meta["tokenizer"],
        "state_dict": state_dict,
    }

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, out_path)
    return out_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    parser.add_argument("checkpoint", type=Path, help="Lightning .ckpt from train.py")
    parser.add_argument("output", type=Path, help="where to write model.pt")
    args = parser.parse_args()
    print(f"Exported -> {export_checkpoint(args.checkpoint, args.output)}")


if __name__ == "__main__":
    main()
