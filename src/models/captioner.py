from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch import Tensor

from src.config import Config


class ImageEncoder(nn.Module):
    """CNN trunk -> spatial feature map -> project to d_model.

    Instead of pooling down to a single vector, we keep the full spatial grid
    (7x7 = 49 patches for EfficientNetV2-S at 224px) so the decoder can
    cross-attend to individual image regions.

    ``pretrained=None`` follows ``cfg.pretrained_backbone``. The serving path
    passes ``False``: the checkpoint supplies every weight, so downloading
    ImageNet weights at startup would only waste time (and fail offline).
    """

    def __init__(self, cfg: Config, pretrained: bool | None = None):
        super().__init__()
        if pretrained is None:
            pretrained = cfg.pretrained_backbone

        backbone = tv_models.get_model(
            cfg.backbone, weights="DEFAULT" if pretrained else None
        )
        if not hasattr(backbone, "features"):
            raise ValueError(
                f"Backbone '{cfg.backbone}' has no convolutional `.features` trunk; "
                "use an EfficientNet/MobileNet/ConvNeXt/RegNet-style model."
            )
        # .features stops before pooling and the classifier head
        self.features = backbone.features

        # Probe the trunk once instead of hard-coding 1280 channels / 7x7.
        was_training = self.features.training
        self.features.eval()
        with torch.no_grad():
            probe = self.features(torch.zeros(1, 3, cfg.image_size, cfg.image_size))
        self.features.train(was_training)
        _, channels, h, w = probe.shape
        self.n_patches = h * w
        self.grid = (h, w)

        self.proj = nn.Linear(channels, cfg.d_model)

        # learnable positional embedding; one vector per spatial location
        self.pos_embed = nn.Parameter(torch.zeros(1, self.n_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        """images: (B, 3, H, W) -> (B, N_patches, d_model)"""
        x = self.features(images)                  # (B, C, h, w)
        x = x.flatten(2).transpose(1, 2)           # (B, h*w, C)
        x = self.proj(x)                           # (B, h*w, d_model)
        return x + self.pos_embed                  # broadcast positional info


class CaptionDecoder(nn.Module):
    """Causal Transformer decoder with cross-attention over image patches.

    Each generated token attends to all spatial patches — the model learns
    which patches are relevant for each word rather than getting a single
    blurred summary.
    """

    def __init__(self, cfg: Config, pad_id: int = 0):
        super().__init__()
        self.d_model = cfg.d_model
        self.max_len = cfg.max_caption_len
        self.pad_id = pad_id

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=pad_id)
        # nn.Embedding defaults to N(0, 1). The output projection is tied to this
        # table, so that scale would put the initial logits at ~sqrt(d_model) and
        # the starting loss far above ln(vocab_size). Small init, like GPT-2.
        nn.init.normal_(self.token_embed.weight, mean=0.0, std=0.02)
        with torch.no_grad():
            self.token_embed.weight[pad_id].zero_()

        # one learnable position per caption slot
        self.pos_embed = nn.Parameter(torch.zeros(1, cfg.max_caption_len, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.nhead,
            dim_feedforward=cfg.dim_feedforward,
            dropout=cfg.dropout,
            batch_first=True,   # (B, T, d_model) convention throughout
            norm_first=True,    # pre-norm: more stable for smaller datasets
        )
        # Pre-norm layers leave the residual stream un-normalised, so the stack
        # needs a final LayerNorm before the output projection (standard in
        # pre-LN transformers). Without it the logits' scale is uncontrolled.
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=cfg.num_decoder_layers,
            norm=nn.LayerNorm(cfg.d_model),
        )

        # output projection — weight tied to token_embed to reduce params by ~4M
        self.output_proj = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self.output_proj.weight = self.token_embed.weight   # tied

        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, token_ids: Tensor, memory: Tensor) -> Tensor:
        """
        token_ids: (B, T)
        memory:    (B, N_patches, d_model)
        returns:   (B, T, vocab_size)
        """
        T = token_ids.size(1)
        device = token_ids.device

        x = self.token_embed(token_ids)             # (B, T, d_model)
        x = x + self.pos_embed[:, :T, :]            # add positional info
        x = self.dropout(x)

        # causal mask: -inf at (i, j) means position i must NOT attend to j.
        # Float additive mask is compatible across all PyTorch versions.
        causal_mask = torch.full((T, T), float("-inf"), device=device).triu(diagonal=1)

        # padding mask: -inf wherever the token is PAD so those positions are
        # ignored in self-attention. Same (float) type as the causal mask —
        # PyTorch deprecates mixing a bool and a float mask.
        pad_mask = torch.zeros(token_ids.shape, device=device).masked_fill(
            token_ids.eq(self.pad_id), float("-inf")
        )                                           # (B, T)

        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )                                            # (B, T, d_model)

        return self.output_proj(x)                  # (B, T, vocab_size)


class ImageCaptioner(nn.Module):
    """Full encoder-decoder captioning model."""

    def __init__(self, cfg: Config, pad_id: int = 0, pretrained: bool | None = None):
        super().__init__()
        self.encoder = ImageEncoder(cfg, pretrained=pretrained)
        self.decoder = CaptionDecoder(cfg, pad_id=pad_id)

    @property
    def n_patches(self) -> int:
        return self.encoder.n_patches

    def encode(self, images: Tensor) -> Tensor:
        """images: (B, 3, H, W) -> (B, N_patches, d_model)"""
        return self.encoder(images)

    def forward(self, image_features: Tensor, token_ids: Tensor) -> Tensor:
        """
        image_features: (B, N_patches, d_model)  — from encode()
        token_ids:      (B, T)
        returns:        (B, T, vocab_size)
        """
        return self.decoder(token_ids, image_features)

    @torch.inference_mode()
    def greedy_decode(
        self,
        image_features: Tensor,
        max_len: int,
        bos_id: int,
        eos_id: int,
    ) -> Tensor:
        """Autoregressive greedy decoding — used during validation.

        Returns (B, max_len) tensor of generated token IDs.
        Sequences that produce EOS are padded with the PAD id from that point on.
        """
        B = image_features.size(0)
        device = image_features.device
        pad_id = self.decoder.pad_id

        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(image_features, tokens)   # (B, t, vocab_size)
            next_id = logits[:, -1, :].argmax(dim=-1)       # (B,)
            next_id = next_id.masked_fill(finished, pad_id) # pad finished seqs
            tokens = torch.cat([tokens, next_id.unsqueeze(1)], dim=1)
            finished = finished | next_id.eq(eos_id)
            if finished.all():
                break

        # pad to exactly max_len if we exited the loop early
        if tokens.size(1) < max_len:
            pad = torch.full(
                (B, max_len - tokens.size(1)), pad_id, dtype=torch.long, device=device
            )
            tokens = torch.cat([tokens, pad], dim=1)

        return tokens
