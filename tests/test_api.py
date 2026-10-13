import io
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import src.api.app as app_mod
from src.api.schemas import CaptionResponse


class StubEngine:
    """Stands in for the real model so API behaviour is tested in isolation."""
    device = "cpu"

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.calls: list[int] = []

    def caption(self, raw: bytes, beam_size: int) -> dict:
        if self.fail:
            raise RuntimeError("boom")
        self.calls.append(beam_size)
        return {
            "caption": "a dog", "beam_size": beam_size, "inference_ms": 1.5,
            "tokens": ["a", "dog"], "attention_weights": [[0.25] * 4, [0.25] * 4],
        }


@pytest.fixture
def stub():
    return StubEngine()


@pytest.fixture
def client(stub, monkeypatch):
    monkeypatch.setattr(app_mod, "_engine", stub)
    monkeypatch.setattr(app_mod, "_executor", ThreadPoolExecutor(max_workers=1))
    monkeypatch.setattr(app_mod, "_request_count", 0)
    monkeypatch.setattr(app_mod, "_error_count", 0)
    monkeypatch.setattr(app_mod, "_latencies", [])
    return TestClient(app_mod.app)    # no `with`: lifespan (model loading) is skipped


def _post(client, data: bytes, content_type="image/png", **form):
    return client.post("/caption", files={"file": ("x", data, content_type)}, data=form)


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json() == {"status": "ok", "model_loaded": True, "device": "cpu"}


def test_health_503_until_model_is_loaded(client, monkeypatch):
    monkeypatch.setattr(app_mod, "_engine", None)
    assert client.get("/health").status_code == 503


def test_caption_happy_path_matches_schema(client, stub, png_bytes):
    r = _post(client, png_bytes, beam_size="6")
    assert r.status_code == 200
    body = CaptionResponse.model_validate(r.json())
    assert body.caption == "a dog" and body.beam_size == 6 and stub.calls == [6]


def test_default_beam_size_is_4(client, stub, png_bytes):
    assert _post(client, png_bytes).status_code == 200 and stub.calls == [4]


@pytest.mark.parametrize("beam", ["0", "17", "abc"])
def test_beam_size_out_of_range_is_422(client, png_bytes, beam):
    assert _post(client, png_bytes, beam_size=beam).status_code == 422


def test_rejects_wrong_content_type(client, png_bytes):
    r = _post(client, png_bytes, content_type="application/pdf")
    assert r.status_code == 422 and "Unsupported file type" in r.json()["detail"]


def test_rejects_oversized_upload(client, png_bytes, monkeypatch):
    monkeypatch.setattr(app_mod, "MAX_BYTES", 100)
    r = _post(client, png_bytes)
    assert r.status_code == 422 and "maximum allowed size" in r.json()["detail"]


def test_rejects_corrupt_image(client):
    r = _post(client, b"this is not an image")
    assert r.status_code == 422 and "Could not decode" in r.json()["detail"]


def test_rejects_decompression_bomb_sized_image(client, png_bytes, monkeypatch):
    monkeypatch.setattr(app_mod, "MAX_PIXELS", 100)       # the 48x40 test image exceeds this
    r = _post(client, png_bytes)
    assert r.status_code == 422 and "megapixels" in r.json()["detail"]


def test_accepts_jpeg_and_webp(client):
    for fmt, mime in (("JPEG", "image/jpeg"), ("WEBP", "image/webp")):
        buf = io.BytesIO()
        Image.new("RGB", (30, 30), (1, 2, 3)).save(buf, format=fmt)
        assert _post(client, buf.getvalue(), content_type=mime).status_code == 200


def test_503_on_caption_when_model_not_loaded(client, png_bytes, monkeypatch):
    monkeypatch.setattr(app_mod, "_engine", None)
    assert _post(client, png_bytes).status_code == 503


def test_inference_failure_is_500_and_counted(client, stub, png_bytes):
    stub.fail = True
    assert _post(client, png_bytes).status_code == 500
    assert "v2t_request_errors_total 1" in client.get("/metrics").text


def test_metrics_exposition(client, png_bytes):
    _post(client, png_bytes)
    _post(client, png_bytes)
    _post(client, b"junk")
    text = client.get("/metrics").text
    assert "v2t_requests_total 3" in text
    assert "v2t_request_errors_total 1" in text
    assert 'v2t_request_duration_seconds{quantile="0.95"}' in text
    assert "v2t_request_duration_seconds_count 2" in text
