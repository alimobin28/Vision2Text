import math

import pytest
import torch
import torch.nn.functional as F

from src.models.captioner import ImageCaptioner
from tests.helpers import make_cfg


@pytest.fixture(scope="module")
def cfg(tmp_path_factory):
    return make_cfg(tmp_path_factory.mktemp("m"))


@pytest.fixture(scope="module")
def model(cfg):
    torch.manual_seed(0)
    return ImageCaptioner(cfg, pad_id=0, pretrained=False).eval()


def test_shapes(cfg, model):
    images = torch.randn(3, 3, cfg.image_size, cfg.image_size)
    feats = model.encode(images)
    assert feats.shape == (3, model.n_patches, cfg.d_model)
    tokens = torch.randint(4, cfg.vocab_size, (3, 7))
    assert model(feats, tokens).shape == (3, 7, cfg.vocab_size)


@pytest.mark.parametrize("size", [64, 96, 128])
def test_patch_count_follows_image_size(tmp_path, size):
    # No more hard-coded 1280 channels / 7x7 grid: probed from the backbone.
    m = ImageCaptioner(make_cfg(tmp_path, image_size=size), pretrained=False)
    assert m.n_patches == (size // 32) ** 2


def test_decoder_is_causal(cfg, model):
    """Logits at position i must not depend on tokens after i."""
    feats = model.encode(torch.randn(1, 3, cfg.image_size, cfg.image_size))
    a = torch.randint(4, cfg.vocab_size, (1, 8))
    b = a.clone()
    b[:, 5:] = torch.randint(4, cfg.vocab_size, (1, 3))      # change only the tail
    with torch.no_grad():
        la, lb = model(feats, a), model(feats, b)
    assert torch.allclose(la[:, :5], lb[:, :5], atol=1e-5)
    assert not torch.allclose(la[:, 5:], lb[:, 5:], atol=1e-5)


def test_padding_tokens_do_not_produce_nans(cfg, model):
    feats = model.encode(torch.randn(2, 3, cfg.image_size, cfg.image_size))
    tokens = torch.tensor([[2, 5, 6, 0, 0, 0], [2, 7, 0, 0, 0, 0]])
    assert torch.isfinite(model(feats, tokens)).all()


def test_initial_loss_is_near_chance(cfg, model):
    """Regression: without a final LayerNorm and with N(0,1) tied embeddings the
    initial logits were huge (smoke run started at loss ~20 for a 300-token vocab).
    A fresh model should start close to ln(vocab_size)."""
    model.train()
    feats = model.encode(torch.randn(8, 3, cfg.image_size, cfg.image_size))
    tokens = torch.randint(4, cfg.vocab_size, (8, 10))
    loss = F.cross_entropy(model(feats, tokens[:, :-1]).reshape(-1, cfg.vocab_size), tokens[:, 1:].reshape(-1))
    model.eval()
    assert abs(loss.item() - math.log(cfg.vocab_size)) < 1.0


def test_pad_id_is_configurable_and_embedding_row_is_zero(cfg):
    m = ImageCaptioner(cfg, pad_id=2, pretrained=False)
    assert m.decoder.pad_id == 2 and m.decoder.token_embed.padding_idx == 2
    assert torch.count_nonzero(m.decoder.token_embed.weight[2]) == 0


def test_output_projection_is_tied_to_embedding(model):
    assert model.decoder.output_proj.weight is model.decoder.token_embed.weight


def test_greedy_decode_pads_after_eos(cfg):
    m = ImageCaptioner(cfg, pad_id=0, pretrained=False).eval()
    V, BOS, EOS, WORD = cfg.vocab_size, 2, 3, 7

    def fake_forward(feats, toks):                 # always "WORD", then EOS at step 3
        logits = torch.zeros(toks.size(0), toks.size(1), V)
        logits[:, :, WORD] = 5.0
        if toks.size(1) >= 3:
            logits[:, -1, EOS] = 10.0
        return logits

    m.forward = fake_forward
    out = m.greedy_decode(torch.zeros(2, 4, cfg.d_model), max_len=8, bos_id=BOS, eos_id=EOS)
    assert out.shape == (2, 8)
    assert out[0].tolist() == [BOS, WORD, WORD, EOS, 0, 0, 0, 0]


def test_greedy_decode_always_returns_max_len(cfg, model):
    out = model.greedy_decode(
        model.encode(torch.randn(2, 3, cfg.image_size, cfg.image_size)),
        max_len=cfg.max_caption_len, bos_id=2, eos_id=3,
    )
    assert out.shape == (2, cfg.max_caption_len)
    assert (out[:, 0] == 2).all()
