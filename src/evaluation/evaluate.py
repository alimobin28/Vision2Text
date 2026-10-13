"""
Offline evaluation of an exported model on a held-out split.

    python -m src.evaluation.evaluate --checkpoint outputs/model.pt
    python -m src.evaluation.evaluate --checkpoint outputs/model.pt --split val --beam-size 5 --meteor

Decodes every image of the split with beam search and scores the result against
ALL of that image's reference captions. Writes

    outputs/metrics_<split>.json       scores + the settings that produced them
    outputs/predictions_<split>.json   hypothesis and references per image

IMPORTANT: the train/val/test split is a pure function of the data settings
(V2T_SEED, V2T_VAL_FRACTION, V2T_TEST_FRACTION, V2T_MAX_IMAGES, V2T_DATA_DIR).
Run this with the SAME values the model was trained with, otherwise "test"
images may overlap the training set.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.config import Config
from src.config import cfg as default_cfg
from src.data.dataset import FlickrEvalDataset, eval_collate, load_splits
from src.evaluation.metrics import score_all
from src.inference.engine import InferenceEngine


def evaluate(
    checkpoint: Path,
    config: Config = default_cfg,
    split: str = "test",
    beam_size: int = 4,
    limit: int = 0,
    meteor: bool = False,
    batch_size: int = 16,
    num_workers: int = 0,
    device: str = "auto",
) -> dict:
    engine = InferenceEngine(checkpoint, config.vocab_path, device, config=config)
    dataset = FlickrEvalDataset(engine.cfg, split, splits=load_splits(config))
    if len(dataset) == 0:
        raise ValueError(f"The '{split}' split is empty; check the V2T_* data settings.")

    names, hyps, refs = [], [], []
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=eval_collate,
    )
    seen = 0
    for images, batch_refs in tqdm(loader, desc=f"Captioning {split}"):
        hyps.extend(engine.caption_tensors(images, beam_size))
        refs.extend(batch_refs)
        names.extend(dataset.image_names[seen : seen + len(batch_refs)])
        seen += len(batch_refs)
        if limit and seen >= limit:
            names, hyps, refs = names[:limit], hyps[:limit], refs[:limit]
            break

    scores = score_all(hyps, refs, meteor=meteor)
    return {
        "scores": scores,
        "meta": {
            "checkpoint": str(checkpoint),
            "split": split,
            "n_images": len(hyps),
            "beam_size": beam_size,
            "device": str(engine.device),
            "torch": torch.__version__,
            "data": {
                "dataset": config.dataset_name, "seed": config.seed,
                "val_fraction": config.val_fraction, "test_fraction": config.test_fraction,
                "max_images": config.max_images,
            },
        },
        "predictions": [
            {"image": n, "hypothesis": h, "references": r}
            for n, h, r in zip(names, hyps, refs, strict=True)
        ],
    }


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1].strip())
    p.add_argument("--checkpoint", type=Path, default=default_cfg.checkpoint)
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--beam-size", type=int, default=4)
    p.add_argument("--limit", type=int, default=0, help="only score the first N images (0 = all)")
    p.add_argument("--meteor", action="store_true", help="also compute NLTK METEOR (needs wordnet)")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default=default_cfg.device)
    args = p.parse_args()

    result = evaluate(
        args.checkpoint, default_cfg, args.split, args.beam_size, args.limit,
        args.meteor, args.batch_size, device=args.device,
    )

    default_cfg.output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = default_cfg.output_dir / f"metrics_{args.split}.json"
    preds_path = default_cfg.output_dir / f"predictions_{args.split}.json"
    metrics_path.write_text(json.dumps(
        {"scores": result["scores"], "meta": result["meta"]}, indent=2))
    preds_path.write_text(json.dumps(result["predictions"], indent=2))

    print(f"\nResults on '{args.split}' ({result['meta']['n_images']} images, "
          f"beam={args.beam_size}):")
    for name, value in result["scores"].items():
        print(f"  {name:<14} {value:.4f}")
    print(f"\nWrote {metrics_path} and {preds_path}")


if __name__ == "__main__":
    main()
