from __future__ import annotations

import torch
import torch.nn as nn
import torchvision.models as tv_models
from torch import Tensor

from src.config import Config


class ImageEncoder(nn.Module):
    """EfficientNetV2-S → spatial feature map → project to d_model.

    Instead of pooling down to a single 1280-d vector, we keep the full 7×7
    spatial grid so the decoder can cross-attend to individual image regions.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        weights = tv_models.EfficientNet_V2_S_Weights.DEFAULT
        backbone = tv_models.efficientnet_v2_s(weights=weights)

        # .features is the convolutional trunk; stops before avgpool and classifier
        self.features = backbone.features          # output: (B, 1280, 7, 7) at 224px

        backbone_out_channels = 1280
        self.proj = nn.Linear(backbone_out_channels, cfg.d_model)

        # how many spatial patches the backbone produces at this image size
        spatial_size = cfg.image_size // 32        # 224 // 32 = 7
        n_patches = spatial_size * spatial_size    # 49

        # learnable positional embedding; one vector per spatial location
        self.pos_embed = nn.Parameter(torch.zeros(1, n_patches, cfg.d_model))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, images: Tensor) -> Tensor:
        """images: (B, 3, H, W) -> (B, N_patches, d_model)"""
        x = self.features(images)                  # (B, 1280, 7, 7)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)           # (B, 49, 1280)
        x = self.proj(x)                           # (B, 49, d_model)
        x = x + self.pos_embed                     # broadcast positional info
        return x


class CaptionDecoder(nn.Module):
    """4-layer causal Transformer decoder with cross-attention over image patches.

    Each generated token attends to all 49 spatial patches — the model learns
    which patches are relevant for each word rather than getting a single blurred summary.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.d_model = cfg.d_model
        self.max_len = cfg.max_caption_len

        self.token_embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=0)

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
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=cfg.num_decoder_layers,
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

        # padding mask: True wherever the token is PAD (index 0), so those
        # positions are ignored in self-attention
        pad_mask = token_ids.eq(0)                  # (B, T), True = ignore

        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=pad_mask,
        )                                            # (B, T, d_model)

        return self.output_proj(x)                  # (B, T, vocab_size)


class ImageCaptioner(nn.Module):
    """Full encoder-decoder captioning model.

    Forward accepts pre-encoded spatial features so the encoder can be run
    once and cached (useful for ablations or frozen-encoder training).
    The `encode` method is the bridge between raw images and those features.
    """

    def __init__(self, cfg: Config):
        super().__init__()
        self.encoder = ImageEncoder(cfg)
        self.decoder = CaptionDecoder(cfg)

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
        Sequences that produce EOS are padded with zeros from that point on.
        """
        B = image_features.size(0)
        device = image_features.device

        tokens = torch.full((B, 1), bos_id, dtype=torch.long, device=device)
        finished = torch.zeros(B, dtype=torch.bool, device=device)

        for _ in range(max_len - 1):
            logits = self.forward(image_features, tokens)   # (B, t, vocab_size)
            next_id = logits[:, -1, :].argmax(dim=-1)       # (B,)
            next_id = next_id.masked_fill(finished, 0)      # pad finished seqs
            tokens = torch.cat([tokens, next_id.unsqueeze(1)], dim=1)
            finished = finished | next_id.eq(eos_id)
            if finished.all():
                break

        # pad to exactly max_len if we exited the loop early
        if tokens.size(1) < max_len:
            pad = torch.zeros(B, max_len - tokens.size(1), dtype=torch.long, device=device)
            tokens = torch.cat([tokens, pad], dim=1)

        return tokens
