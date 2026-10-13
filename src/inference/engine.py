"""
Inference engine: loads the trained model once and exposes a caption() method
that runs beam search without ever touching disk for the image bytes.
"""
from __future__ import annotations

import io
import logging
import pickle
import threading
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image, ImageOps
from tokenizers import Tokenizer
from torch import Tensor

from src.config import Config
from src.config import cfg as default_cfg
from src.data.transforms import build_transform
from src.data.vocab import group_pieces, load_vocab, special_ids
from src.models.captioner import ImageCaptioner

log = logging.getLogger(__name__)

# alpha of the length penalty in Wu et al. (2016), "Google's NMT system"
LENGTH_PENALTY_ALPHA = 0.7


class InferenceEngine:
    """Wraps the trained ImageCaptioner for single-image inference.

    Instantiate once at startup — loading weights takes a few seconds and you
    definitely don't want that on the hot path.

    ``checkpoint_path`` may be

    * an exported ``model.pt`` (``python -m src.training.export``; also written
      by ``train.py``) — self-contained: weights + architecture + tokenizer, or
    * a Lightning ``.ckpt`` written by this project's trainer.

    The architecture and tokenizer are taken from the checkpoint, never from the
    serving environment, so a mismatched ``V2T_D_MODEL`` etc. cannot produce a
    model that loads but silently misbehaves. ``vocab_path`` is only used for
    old checkpoints that do not embed a tokenizer.

    ``caption()`` is safe to call from several threads at once.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        vocab_path: Path | None = None,
        device: str = "auto",
        config: Config | None = None,
    ) -> None:
        self.device = self._resolve_device(device)

        arch, tokenizer_json, state_dict = self._read_checkpoint(Path(checkpoint_path))
        self.cfg = (config or default_cfg).model_copy(update=arch)

        # --- tokenizer ---
        if tokenizer_json is not None:
            self.tokenizer = Tokenizer.from_str(tokenizer_json)
        else:
            self.tokenizer = load_vocab(Path(vocab_path or self.cfg.vocab_path))
        self.pad_id, self.bos_id, self.eos_id = special_ids(self.tokenizer)

        # --- model ---
        # pretrained=False: every weight comes from the checkpoint, so don't
        # download ImageNet weights just to overwrite them.
        self.model = ImageCaptioner(self.cfg, pad_id=self.pad_id, pretrained=False)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device).eval()

        self._transform = build_transform(self.cfg.image_size, train=False)

        # Cross-attention capture (see _get_attention_weights). Hooks are
        # installed ONCE and only act on a thread that opted in through the
        # thread-local below, so concurrent requests cannot see each other's
        # tensors and no module hook dict is mutated while another thread is
        # iterating it.
        self._tls = threading.local()
        self._install_attention_hooks()

    # ------------------------------------------------------------------
    # public interface
    # ------------------------------------------------------------------

    def caption(self, image_bytes: bytes, beam_size: int = 4) -> dict[str, Any]:
        """Run beam search and return a result dict ready for the API response.

        Two forward passes are made:
          1. Beam search → best token sequence.
          2. Single parallel pass over that sequence → cross-attention weights
             for every token, without the complexity of tracking attention
             through beam search bookkeeping.

        ``tokens`` are whole words (BPE pieces are folded back together) and
        ``attention_weights`` has one row per word: the mean of its pieces' rows.
        """
        t0 = time.perf_counter()

        img_tensor = self._preprocess(image_bytes)

        with torch.inference_mode():
            features = self.model.encode(img_tensor)       # (1, N, d_model)
            token_ids = self._beam_decode(features, beam_size)

        if not token_ids:
            return {
                "caption": "",
                "beam_size": beam_size,
                "inference_ms": round((time.perf_counter() - t0) * 1000, 2),
                "tokens": [],
                "attention_weights": [],
            }

        # Build the input sequence for the attention pass: [BOS, t_1, …, t_T].
        # Cross-attention at position i tells us which patches drove the
        # prediction of token_ids[i], so attn_raw[i] aligns with token_ids[i].
        input_ids = torch.tensor(
            [self.bos_id] + token_ids,
            dtype=torch.long,
            device=self.device,
        ).unsqueeze(0)                                     # (1, T+1)

        attn_raw = self._get_attention_weights(features, input_ids)

        # Keep content tokens (and their attention rows); drop specials.
        n_patches = features.size(1)
        uniform = [1.0 / n_patches] * n_patches
        skip = {self.bos_id, self.eos_id, self.pad_id}

        pieces: list[str] = []
        rows: list[list[float]] = []
        for idx, tid in enumerate(token_ids):
            if tid in skip:
                continue
            pieces.append(self.tokenizer.id_to_token(tid) or "")
            rows.append(attn_raw[idx] if idx < len(attn_raw) else uniform)

        # Fold sub-word pieces into words; average their attention rows.
        words: list[str] = []
        word_rows: list[list[float]] = []
        for word, piece_idxs in group_pieces(pieces):
            words.append(word)
            word_rows.append(
                [sum(rows[i][p] for i in piece_idxs) / len(piece_idxs) for p in range(n_patches)]
            )

        return {
            "caption": self.tokenizer.decode(token_ids),
            "beam_size": beam_size,
            "inference_ms": round((time.perf_counter() - t0) * 1000, 2),
            "tokens": words,
            "attention_weights": word_rows,
        }

    @torch.inference_mode()
    def caption_tensors(self, images: Tensor, beam_size: int = 4) -> list[str]:
        """Caption an already-preprocessed ``(B, 3, H, W)`` batch.

        Used by offline evaluation, which batches the CNN forward pass and skips
        the attention maps. Beam search itself still runs one image at a time.
        """
        features = self.model.encode(images.to(self.device))
        return [
            self.tokenizer.decode(self._beam_decode(features[i : i + 1], beam_size))
            for i in range(features.size(0))
        ]

    # ------------------------------------------------------------------
    # loading
    # ------------------------------------------------------------------

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    @staticmethod
    def _read_checkpoint(path: Path) -> tuple[dict, str | None, dict[str, Tensor]]:
        """Return ``(architecture overrides, tokenizer JSON or None, state_dict)``."""
        if not path.is_file():
            raise FileNotFoundError(
                f"Checkpoint not found: {path}. Train a model with `python train.py` "
                "(it writes outputs/model.pt) or set V2T_CHECKPOINT."
            )
        try:
            # weights_only=True: refuse to unpickle arbitrary objects.
            state = torch.load(path, map_location="cpu", weights_only=True)
        except pickle.UnpicklingError as exc:
            raise RuntimeError(
                f"{path} contains objects that are not plain tensors, so it cannot "
                "be loaded safely. Convert it first with "
                f"`python -m src.training.export {path} model.pt`."
            ) from exc

        # exported model.pt
        if "format" in state:
            return state["architecture"], state["tokenizer"], state["state_dict"]

        # Lightning .ckpt
        if "state_dict" in state:
            meta = state.get("v2t", {})
            sd = {
                k.removeprefix("model."): v
                for k, v in state["state_dict"].items()
                if k.startswith("model.")
            }
            return meta.get("architecture", {}), meta.get("tokenizer"), sd

        # bare state_dict
        return {}, None, state

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _preprocess(self, image_bytes: bytes) -> Tensor:
        opened = Image.open(io.BytesIO(image_bytes))
        img = ImageOps.exif_transpose(opened).convert("RGB")   # honour phone rotation
        return self._transform(img).unsqueeze(0).to(self.device)

    def _install_attention_hooks(self) -> None:
        """Register (once) the hooks that expose decoder cross-attention.

        PyTorch's ``TransformerDecoderLayer._mha_block`` calls the cross-attention
        module with ``need_weights=False`` to take the fast path, which discards
        the weight matrix. For a thread that has set ``self._tls.capture`` we
        override that to ``True`` and collect the returned weights; for every
        other call the hooks are a no-op.
        """

        def _pre_hook(module: nn.Module, args: tuple, kwargs: dict):
            if getattr(self._tls, "capture", None) is None:
                return None
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True   # average over heads → (B, T, N)
            return args, kwargs

        def _post_hook(module: nn.Module, inputs: tuple, output: object) -> None:
            capture = getattr(self._tls, "capture", None)
            if capture is None:
                return
            # MHA returns (attn_output, attn_weights); weights may still be None
            # if the kernel decided to skip them — guard defensively.
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                capture.append(output[1].detach().cpu())   # (1, T, N_patches)

        for layer in self.model.decoder.decoder.layers:
            layer.multihead_attn.register_forward_pre_hook(_pre_hook, with_kwargs=True)
            layer.multihead_attn.register_forward_hook(_post_hook)

    def _get_attention_weights(
        self,
        features: Tensor,
        token_ids: Tensor,
    ) -> list[list[float]]:
        """One forward pass; cross-attention of every decoder layer, averaged.

        Returns shape [T, N_patches].
        """
        captured: list[Tensor] = []
        self._tls.capture = captured
        try:
            with torch.inference_mode():
                self.model(features, token_ids)
        finally:
            self._tls.capture = None

        T = token_ids.size(1)
        n_patches = features.size(1)

        if not captured:
            # Shouldn't happen — fall back to uniform so the UI doesn't crash.
            log.warning("cross-attention capture returned nothing; using uniform weights")
            return [[1.0 / n_patches] * n_patches for _ in range(T)]

        # (num_layers, 1, T, N_patches) → (T, N_patches)
        return torch.stack(captured, dim=0).mean(dim=0).squeeze(0).tolist()

    @torch.inference_mode()
    def _beam_decode(self, features: Tensor, beam_size: int) -> list[int]:
        """Standard beam search over the decoder.

        Keeps up to `beam_size` hypotheses alive and collects completed ones as
        they hit EOS. Returns the highest length-normalised complete sequence,
        falling back to the best active sequence if nothing ever hit EOS.
        """
        device = features.device
        max_len = self.cfg.max_caption_len
        vocab_size = self.cfg.vocab_size

        # one active beam to start
        seqs = torch.full((1, 1), self.bos_id, dtype=torch.long, device=device)
        scores = torch.zeros(1, device=device)                 # cumulative log-probs
        completed: list[tuple[float, list[int]]] = []

        for _ in range(max_len - 1):
            k = seqs.size(0)
            mem = features.expand(k, -1, -1)                   # (k, N_patches, d_model)
            logits = self.model(mem, seqs)                     # (k, t, vocab_size)
            log_probs = torch.log_softmax(logits[:, -1, :], dim=-1)  # (k, V)

            cand_scores = scores.unsqueeze(1) + log_probs      # (k, V)

            n_needed = beam_size - len(completed)
            if n_needed <= 0:
                break

            top_scores, flat_idx = cand_scores.view(-1).topk(
                min(n_needed * 2, k * vocab_size)
            )
            beam_idx = flat_idx // vocab_size
            token_idx = flat_idx % vocab_size

            new_seqs: list[Tensor] = []
            new_scores: list[float] = []

            for i in range(top_scores.size(0)):
                if len(new_seqs) >= n_needed:
                    break
                b = int(beam_idx[i].item())
                t = int(token_idx[i].item())
                s = float(top_scores[i].item())
                new_seq = torch.cat([seqs[b], seqs.new_tensor([t])])

                if t == self.eos_id:
                    # strip BOS at index 0; EOS is the last token
                    completed.append((s, new_seq[1:-1].tolist()))
                else:
                    new_seqs.append(new_seq)
                    new_scores.append(s)

            if not new_seqs:
                break

            seqs = torch.stack(new_seqs)
            scores = torch.tensor(new_scores, device=device)

        if not completed:
            # nothing hit EOS — just return the best active beam as-is
            best = int(scores.argmax().item())
            return [
                t for t in seqs[best].tolist()
                if t not in (self.bos_id, self.pad_id, self.eos_id)
            ]

        # Length-normalize scores so short captions don't win by default. An
        # immediate EOS gives an empty caption; ignore those when anything else
        # finished, and never divide by zero length.
        candidates = [c for c in completed if c[1]] or completed
        candidates.sort(
            key=lambda c: c[0] / (max(1, len(c[1])) ** LENGTH_PENALTY_ALPHA),
            reverse=True,
        )
        return candidates[0][1]
