from __future__ import annotations

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import AdamW
from transformers import get_cosine_schedule_with_warmup
from torchmetrics.text import BLEUScore

from src.config import Config
from src.data.vocab import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, load_vocab
from src.models.captioner import ImageCaptioner


def recommended_precision() -> str:
    """Pick the best available mixed-precision mode for this machine."""
    if torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        return "bf16-mixed"
    if torch.cuda.is_available():
        return "16-mixed"
    return "32"


class CaptioningModule(pl.LightningModule):
    """Lightning wrapper around ImageCaptioner.

    Training uses teacher forcing (fast, stable).
    Validation uses greedy decoding so BLEU-4 reflects what the model actually
    generates at inference time, not what it generates when fed ground-truth tokens.
    """

    def __init__(self, cfg: Config, steps_per_epoch: int):
        super().__init__()
        self.cfg = cfg
        self.steps_per_epoch = steps_per_epoch
        self.save_hyperparameters(cfg.model_dump())

        self.model = ImageCaptioner(cfg)

        tokenizer = load_vocab(cfg.vocab_path)
        self._tokenizer = tokenizer
        self._bos_id = tokenizer.token_to_id(BOS_TOKEN)
        self._eos_id = tokenizer.token_to_id(EOS_TOKEN)
        self._pad_id = tokenizer.token_to_id(PAD_TOKEN)

        self.criterion = nn.CrossEntropyLoss(
            label_smoothing=cfg.label_smoothing,
            ignore_index=self._pad_id,
        )

        self._bleu4 = BLEUScore(n_gram=4)

        # accumulate during validation_step, score in on_validation_epoch_end
        self._val_preds: list[str] = []
        self._val_targets: list[list[str]] = []

    # ------------------------------------------------------------------
    # forward pass helpers
    # ------------------------------------------------------------------

    def _decode_ids_to_text(self, token_ids: torch.Tensor) -> list[str]:
        """Convert a (B, T) int tensor back to a list of caption strings."""
        results = []
        for row in token_ids.tolist():
            # strip BOS, EOS, PAD
            trimmed = []
            for tok in row:
                if tok in (self._bos_id, self._pad_id):
                    continue
                if tok == self._eos_id:
                    break
                trimmed.append(tok)
            results.append(self._tokenizer.decode(trimmed))
        return results

    # ------------------------------------------------------------------
    # training
    # ------------------------------------------------------------------

    def training_step(self, batch: tuple, batch_idx: int) -> torch.Tensor:
        images, token_ids = batch                          # (B,3,H,W), (B,T)

        features = self.model.encode(images)               # (B, 49, d_model)

        # teacher forcing: feed all tokens except the last, predict all except the first
        logits = self.model(features, token_ids[:, :-1])   # (B, T-1, vocab_size)
        targets = token_ids[:, 1:]                         # (B, T-1)

        V = logits.size(-1)
        loss = self.criterion(logits.reshape(-1, V), targets.reshape(-1))

        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    # ------------------------------------------------------------------
    # validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: tuple, batch_idx: int) -> None:
        images, token_ids = batch

        features = self.model.encode(images)
        generated = self.model.greedy_decode(
            features,
            max_len=self.cfg.max_caption_len,
            bos_id=self._bos_id,
            eos_id=self._eos_id,
        )

        preds = self._decode_ids_to_text(generated)
        refs = self._decode_ids_to_text(token_ids)

        self._val_preds.extend(preds)
        # torchmetrics BLEUScore expects List[List[str]] for references
        self._val_targets.extend([[ref] for ref in refs])

    def on_validation_epoch_end(self) -> None:
        if not self._val_preds:
            return

        bleu = self._bleu4(self._val_preds, self._val_targets)
        self.log("val_bleu4", bleu, prog_bar=True)

        self._val_preds.clear()
        self._val_targets.clear()

    # ------------------------------------------------------------------
    # optimiser + schedule
    # ------------------------------------------------------------------

    def configure_optimizers(self):
        optimizer = AdamW(
            self.parameters(),
            lr=self.cfg.learning_rate,
            weight_decay=self.cfg.weight_decay,
        )

        total_steps = self.steps_per_epoch * self.cfg.max_epochs
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=self.cfg.warmup_steps,
            num_training_steps=total_steps,
        )

        return (
            [optimizer],
            [{"scheduler": scheduler, "interval": "step", "frequency": 1}],
        )


def build_trainer(cfg: Config, steps_per_epoch: int) -> pl.Trainer:
    """Convenience factory that wires up the standard callbacks and precision."""
    from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor

    checkpoint_cb = ModelCheckpoint(
        dirpath=str(cfg.output_dir / "checkpoints"),
        monitor="val_bleu4",
        mode="max",
        save_top_k=1,
        filename="best-{epoch:02d}-{val_bleu4:.4f}",
    )

    lr_monitor = LearningRateMonitor(logging_interval="step")

    return pl.Trainer(
        max_epochs=cfg.max_epochs,
        precision=recommended_precision(),
        gradient_clip_val=1.0,
        callbacks=[checkpoint_cb, lr_monitor],
        default_root_dir=str(cfg.output_dir),
    )
