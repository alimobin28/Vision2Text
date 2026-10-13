import io
from concurrent.futures import ThreadPoolExecutor

import pytest
import torch
from PIL import Image

from src.inference.engine import InferenceEngine
from tests.helpers import make_cfg, write_random_export


@pytest.fixture(scope="module")
def engine(toy_env, tmp_path_factory):
    cfg, tok = toy_env
    path = write_random_export(cfg, tok, tmp_path_factory.mktemp("exp") / "model.pt")
    return InferenceEngine(path, device="cpu", config=cfg)


@pytest.fixture
def image_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (80, 60), (30, 120, 220)).save(buf, format="JPEG")
    return buf.getvalue()


def _rows_are_distributions(result, n_patches):
    assert len(result["tokens"]) == len(result["attention_weights"])
    for row in result["attention_weights"]:
        assert len(row) == n_patches
        assert sum(row) == pytest.approx(1.0, abs=1e-4)


def test_caption_contract_with_a_known_sequence(engine, monkeypatch):
    ids = engine.tokenizer.encode("a red circle").ids
    monkeypatch.setattr(engine, "_beam_decode", lambda feats, beam: list(ids))
    out = engine.caption(_png(), beam_size=3)
    assert out["caption"] == "a red circle"
    assert out["tokens"] == ["a", "red", "circle"]          # whole words, not BPE pieces
    assert out["beam_size"] == 3 and out["inference_ms"] > 0
    _rows_are_distributions(out, n_patches=4)


def test_real_beam_search_runs_for_several_widths(engine, image_bytes):
    for beam in (1, 3):
        out = engine.caption(image_bytes, beam_size=beam)
        assert isinstance(out["caption"], str)
        _rows_are_distributions(out, n_patches=4)


def test_empty_caption_does_not_crash(engine, image_bytes, monkeypatch):
    """Regression: a beam that emits EOS first has length 0 and used to hit 0 ** 0.7."""
    V, EOS = engine.cfg.vocab_size, engine.eos_id

    def always_eos(feats, toks):
        logits = torch.full((toks.size(0), toks.size(1), V), -10.0)
        logits[:, :, EOS] = 10.0
        return logits

    monkeypatch.setattr(engine.model, "forward", always_eos)
    out = engine.caption(image_bytes, beam_size=3)
    assert out["caption"] == "" and out["tokens"] == [] and out["attention_weights"] == []


def test_concurrent_requests_do_not_corrupt_each_other(engine, image_bytes):
    """Regression for the shared-hook race: results under concurrency must equal
    the single-threaded result exactly."""
    baseline = engine.caption(image_bytes, beam_size=2)
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda _: engine.caption(image_bytes, beam_size=2), range(12)))
    for r in results:
        assert r["caption"] == baseline["caption"]
        assert r["tokens"] == baseline["tokens"]
        assert torch.allclose(
            torch.tensor(r["attention_weights"]), torch.tensor(baseline["attention_weights"]), atol=1e-5
        )


def test_engine_never_requests_pretrained_weights(toy_env, tmp_path, monkeypatch):
    import torchvision.models as tvm
    cfg, tok = toy_env
    path = write_random_export(cfg, tok, tmp_path / "model.pt")
    seen = []
    real = tvm.get_model
    monkeypatch.setattr(tvm, "get_model", lambda name, **kw: (seen.append(kw.get("weights")), real(name, **kw))[1])
    InferenceEngine(path, device="cpu", config=cfg)
    assert seen == [None]      # no ImageNet download at serve time


def test_architecture_comes_from_the_checkpoint_not_the_environment(toy_env, tmp_path):
    cfg, tok = toy_env
    path = write_random_export(cfg, tok, tmp_path / "model.pt")
    wrong_env = make_cfg(tmp_path, d_model=64, nhead=4, num_decoder_layers=3)
    engine = InferenceEngine(path, device="cpu", config=wrong_env)
    assert (engine.cfg.d_model, engine.cfg.num_decoder_layers) == (cfg.d_model, cfg.num_decoder_layers)


def test_missing_checkpoint_message(tmp_path):
    with pytest.raises(FileNotFoundError, match="Checkpoint not found"):
        InferenceEngine(tmp_path / "nope.pt", device="cpu")


class _NotATensor:   # module-level so it can be pickled
    pass


def test_unsafe_pickles_are_refused(toy_env, tmp_path):
    cfg, tok = toy_env
    path = tmp_path / "evil.pt"
    torch.save({"format": 1, "architecture": cfg.architecture(), "tokenizer": tok.to_str(),
                "state_dict": {}, "payload": _NotATensor()}, path)
    with pytest.raises(RuntimeError, match="cannot be loaded safely"):
        InferenceEngine(path, device="cpu", config=cfg)


def _png():
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), (10, 200, 10)).save(buf, format="PNG")
    return buf.getvalue()
