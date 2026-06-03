"""
Inference engine: loads the trained model once and exposes a caption() method
that runs beam search without ever touching disk for the image bytes.
"""
from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
from PIL import Image
from torch import Tensor
from torchvision import transforms

from src.config import cfg
from src.data.vocab import BOS_TOKEN, EOS_TOKEN, PAD_TOKEN, load_vocab
from src.models.captioner import ImageCaptioner


class InferenceEngine:
    """Wraps the trained ImageCaptioner for single-image inference.

    Instantiate once at startup — loading the backbone + weights takes a few
    seconds and you definitely don't want that on the hot path.
    """

    def __init__(
        self,
        checkpoint_path: Path,
        vocab_path: Path,
        device: str = "auto",
    ) -> None:
        self.device = self._resolve_device(device)

        # --- tokenizer ---
        self.tokenizer = load_vocab(vocab_path)
        self.bos_id = self.tokenizer.token_to_id(BOS_TOKEN)
        self.eos_id = self.tokenizer.token_to_id(EOS_TOKEN)
        self.pad_id = self.tokenizer.token_to_id(PAD_TOKEN)

        # --- model ---
        self.model = ImageCaptioner(cfg).to(self.device)
        self._load_checkpoint(checkpoint_path)
        self.model.eval()

        # same normalisation the training DataLoader used
        self._transform = transforms.Compose([
            transforms.Resize((cfg.image_size, cfg.image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

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
        """
        t0 = time.perf_counter()

        img_tensor = self._preprocess(image_bytes)

        with torch.inference_mode():
            features = self.model.encode(img_tensor)       # (1, 49, d_model)
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

        elapsed_ms = (time.perf_counter() - t0) * 1000

        # Zip token IDs with their attention rows; drop special tokens from both.
        _skip = {self.bos_id, self.eos_id, self.pad_id, 0}
        n_patches = features.size(1)
        uniform = [1.0 / n_patches] * n_patches

        token_strs: list[str] = []
        attn_aligned: list[list[float]] = []

        for idx, tid in enumerate(token_ids):
            if tid in _skip:
                continue
            token_strs.append(self.tokenizer.id_to_token(tid))
            attn_aligned.append(attn_raw[idx] if idx < len(attn_raw) else uniform)

        return {
            "caption": self.tokenizer.decode(token_ids),
            "beam_size": beam_size,
            "inference_ms": round(elapsed_ms, 2),
            "tokens": token_strs,
            "attention_weights": attn_aligned,
        }

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------

    def _resolve_device(self, device: str) -> torch.device:
        if device != "auto":
            return torch.device(device)
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    def _load_checkpoint(self, path: Path) -> None:
        """Handles both raw state-dicts and PyTorch Lightning checkpoints."""
        state = torch.load(path, map_location=self.device, weights_only=True)
        sd = state.get("state_dict", state)
        # Lightning wraps the model under self.model, so keys look like
        # "model.encoder.features.0.weight" — strip that prefix.
        sd = {k.removeprefix("model."): v for k, v in sd.items()}
        self.model.load_state_dict(sd)

    def _preprocess(self, image_bytes: bytes) -> Tensor:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        return self._transform(img).unsqueeze(0).to(self.device)

    def _get_attention_weights(
        self,
        features: Tensor,
        token_ids: Tensor,
    ) -> list[list[float]]:
        """Run a single forward pass and capture cross-attention from every
        decoder layer, then average them together.

        PyTorch's TransformerDecoderLayer._mha_block explicitly passes
        need_weights=False to take the fast (flash-attention) path, which
        discards the weight matrix. We temporarily patch each layer's
        multihead_attn via forward hooks to force need_weights=True before
        the call so the weights flow back through the output tuple.

        Returns shape [T, N_patches].
        """
        captured: list[Tensor] = []
        hooks: list[Any] = []

        def _pre_hook(
            module: nn.Module,
            args: tuple,
            kwargs: dict,
        ) -> tuple[tuple, dict]:
            # Override the need_weights=False that _mha_block injects.
            kwargs["need_weights"] = True
            kwargs["average_attn_weights"] = True   # average over heads → (B, T, N)
            return args, kwargs

        def _post_hook(
            module: nn.Module,
            inputs: tuple,
            output: object,
        ) -> None:
            # MHA returns (attn_output, attn_weights); attn_weights may still be
            # None if the kernel decided to skip them — guard defensively.
            if isinstance(output, tuple) and len(output) >= 2 and output[1] is not None:
                captured.append(output[1].detach().cpu())   # (1, T, N_patches)

        for layer in self.model.decoder.decoder.layers:
            mha = layer.multihead_attn
            hooks.append(mha.register_forward_pre_hook(_pre_hook, with_kwargs=True))
            hooks.append(mha.register_forward_hook(_post_hook))

        try:
            with torch.inference_mode():
                self.model(features, token_ids)
        finally:
            for h in hooks:
                h.remove()

        T = token_ids.size(1)
        n_patches = features.size(1)

        if not captured:
            # Shouldn't happen — fall back to uniform so the UI doesn't crash.
            uniform = 1.0 / n_patches
            return [[uniform] * n_patches] * T

        # Average over all decoder layers: (num_layers, 1, T, N_patches) → (T, N_patches)
        avg = torch.stack(captured, dim=0).mean(dim=0).squeeze(0)
        return avg.tolist()

    @torch.inference_mode()
    def _beam_decode(self, features: Tensor, beam_size: int) -> list[int]:
        """Standard beam search over the decoder.

        Keeps `beam_size` active hypotheses at each step and collects completed
        ones as they hit EOS. Returns the highest-scoring complete sequence,
        falling back to the best active sequence if nothing ever hit EOS.
        """
        device = features.device
        max_len = cfg.max_caption_len
        vocab_size = cfg.vocab_size

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
                b = beam_idx[i].item()
                t = token_idx[i].item()
                s = top_scores[i].item()
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
            best = scores.argmax().item()
            return [
                t for t in seqs[best].tolist()
                if t not in (self.bos_id, self.pad_id, 0)
            ]

        # Length-normalize scores so short captions don't win by default.
        # Alpha=0.7 is the standard value from Wu et al. (Google NMT paper).
        completed.sort(key=lambda x: x[0] / (len(x[1]) ** 0.7), reverse=True)
        return completed[0][1]
