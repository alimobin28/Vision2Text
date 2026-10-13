from __future__ import annotations

import math
from typing import Literal

import pytorch_lightning as pl
import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torchmetrics.text import BLEUScore

from src.config import Config
from src.data.vocab import load_vocab, special_ids
from src.models.captioner import ImageCaptioner


def recommended_precision() -> Literal["bf16-mixed", "16-mixed", "32"]:
    """Pick the best available mixed-precision mode for this machine."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    if torch.cuda.is_available():
        return "16-mixed"
    return "32"


def cosine_with_warmup(
    optimizer: torch.optim.Optimizer, warmup_steps: int, total_steps: int
) -> LambdaLR:
    """Linear warm-up to the base LR, then cosine decay to zero."""

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(1.0, progress)))

    return LambdaLR(optimizer, lr_lambda)


class CaptioningModule(pl.LightningModule):
    """Lightning wrapper around ImageCaptioner.

    Training uses teacher forcing (fast, stable).
    Validation uses greedy decoding so BLEU-4 reflects what the model actually
    generates at inference time, not what it generates when fed ground-truth
    tokens. Each generated caption is scored against ALL of its image's
    reference captions.
    """

    def __init__(self, cfg: Config, steps_per_epoch: int):
        super().__init__()
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        # mode="json" turns Paths into strings. Raw pathlib objects pickled into
        # the checkpoint make it unloadable with torch.load(weights_only=True)
        # and, if trained on Windows, on any non-Windows machine at all.
        self.save_hyperparameters(cfg.model_dump(mode="json"))

        tokenizer = load_vocab(cfg.vocab_path)
        if tokenizer.get_vocab_size() > cfg.vocab_size:
            raise ValueError(
                f"Tokenizer has {tokenizer.get_vocab_size()} tokens but "
                f"V2T_VOCAB_SIZE is {cfg.vocab_size}: the embedding table would "
                "be too small. Rebuild the vocab or raise vocab_size."
            )
        self._tokenizer = tokenizer
        self._pad_id, self._bos_id, self._eos_id = special_ids(tokenizer)

        self.model = ImageCaptioner(cfg, pad_id=self._pad_id)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing,
            ignore_index=self._pad_id,
        )

        self.val_bleu4 = BLEUScore(n_gram=4)

    # ------------------------------------------------------------------
    # helpers
    # ------------------------------------------------------------------

    def _decode_ids_to_text(self, token_ids: torch.Tensor) -> list[str]:
        """Convert a (B, T) int tensor back to a list of caption strings."""
        results = []
        for row in token_ids.tolist():
            trimmed = []
            for tok in row:
                if tok in (self._bos_id, self._pad_id):
                    continue
                if tok == self._eos_id:
                    break
                trimmed.append(tok)
            results.append(self._tokenizer.decode(trimmed))
        return results

    def on_save_checkpoint(self, checkpoint: dict) -> None:
        # Make every .ckpt self-describing: the serving side rebuilds the exact
        # architecture and tokenizer from the file instead of trusting its own
        # environment variables to match the training run.
        checkpoint["v2t"] = {
            "architecture": self.cfg.architecture(),
            "tokenizer": self._tokenizer.to_str(),
        }

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, token_ids = batch                          # (B,3,H,W), (B,T)

        features = self.model.encode(images)               # (B, N, d_model)

        # teacher forcing: feed all tokens except the last, predict all except the first
        logits = self.model(features, token_ids[:, :-1])   # (B, T-1, vocab_size)
        targets = token_ids[:, 1:]                         # (B, T-1)

        V = logits.size(-1)
        loss = self.criterion(logits.reshape(-1, V), targets.reshape(-1))

        self.log(
            "train_loss", loss,
            on_step=True, on_epoch=True, prog_bar=True, batch_size=images.size(0),
        )
        return loss

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        images, references = batch        # (B,3,H,W), list[list[str]] (B x n_refs)

        features = self.model.encode(images)
        generated = self.model.greedy_decode(
            features,
            max_len=self.cfg.max_caption_len,
            bos_id=self._bos_id,
            eos_id=self._eos_id,
        )
        preds = self._decode_ids_to_text(generated)

        # torchmetrics accumulates across the epoch, computes once at the end
        # and resets itself — Lightning handles that for metric objects.
        self.val_bleu4.update(preds, references)
        self.log(
            "val_bleu4", self.val_bleu4,
            on_step=False, on_epoch=True, prog_bar=True, batch_size=images.size(0),
        )

    # ------------------------------------------------------------------
    # optimiser + schedule
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        cfg = self.cfg
        backbone_ids = {id(p) for p in self.model.encoder.features.parameters()}

        # 4 groups: {backbone, new layers} x {weight decay, none for biases/norms/embeddings}
        groups: dict[tuple[bool, bool], list[nn.Parameter]] = {
            (b, d): [] for b in (True, False) for d in (True, False)
        }
        for name, p in self.model.named_parameters():
            if not p.requires_grad:
                continue
            decay = p.ndim > 1 and not name.endswith("pos_embed")
            groups[(id(p) in backbone_ids, decay)].append(p)

        param_groups = []
        for (is_backbone, decay), params in groups.items():
            if not params:
                continue
            param_groups.append({
                "params": params,
                "lr": cfg.learning_rate * (cfg.backbone_lr_mult if is_backbone else 1.0),
                "weight_decay": cfg.weight_decay if decay else 0.0,
            })

        optimizer = AdamW(param_groups)

        total_steps = max(1, self.steps_per_epoch * cfg.max_epochs)
        # A 1000-step warm-up is longer than an entire smoke run; cap it at 10%.
        warmup = min(cfg.warmup_steps, max(1, total_steps // 10))
        scheduler = cosine_with_warmup(optimizer, warmup, total_steps)

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step", "frequency": 1}],
        )


def build_trainer(cfg: Config, steps_per_epoch: int) -> pl.Trainer:
    """Convenience factory that wires up the standard callbacks and precision."""
    from pytorch_lightning.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(cfg.output_dir / "checkpoints"),
        monitor="val_bleu4",
        mode="max",
        save_top_k=1,
        save_last=True,                     # last.ckpt: what `train.py --resume` continues from
        filename="best-{epoch:02d}-{val_bleu4:.4f}",
        auto_insert_metric_name=False,      # best-03-0.2150.ckpt, not best-epoch=03-...
    )
    callbacks: list[pl.Callback] = [checkpoint_cb, LearningRateMonitor(logging_interval="step")]
    if cfg.early_stopping_patience > 0:
        callbacks.append(
            EarlyStopping(monitor="val_bleu4", mode="max", patience=cfg.early_stopping_patience)
        )

    return pl.Trainer(
        max_epochs=cfg.max_epochs,
        precision=recommended_precision(),
        gradient_clip_val=1.0,
        callbacks=callbacks,
        logger=CSVLogger(save_dir=str(cfg.output_dir), name="logs"),
        log_every_n_steps=max(1, min(50, steps_per_epoch)),
        default_root_dir=str(cfg.output_dir),
    )
