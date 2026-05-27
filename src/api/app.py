"""
FastAPI application for Vision2Text.

The model is loaded once at startup (via the lifespan context manager) and
reused for every request — no cold starts, no disk writes for uploaded images.
"""
from __future__ import annotations

import asyncio
import io
import os
import threading
import time
from contextlib import asynccontextmanager
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from PIL import Image, UnidentifiedImageError

from src.api.schemas import CaptionResponse, HealthResponse
from src.inference.engine import InferenceEngine


# ---------------------------------------------------------------------------
# in-process metrics store (Prometheus text format at GET /metrics)
# ---------------------------------------------------------------------------

_metrics_lock = threading.Lock()
_request_count: int = 0
_latencies: list[float] = []   # seconds; capped at 10 000 entries


def _record(elapsed_s: float) -> None:
    global _request_count, _latencies
    with _metrics_lock:
        _request_count += 1
        _latencies.append(elapsed_s)
        # keep memory bounded — oldest entries get dropped
        if len(_latencies) > 10_000:
            _latencies = _latencies[-10_000:]


def _percentile(sorted_data: list[float], q: float) -> float:
    if not sorted_data:
        return 0.0
    idx = int(len(sorted_data) * q)
    return sorted_data[min(idx, len(sorted_data) - 1)]


# ---------------------------------------------------------------------------
# validation constants
# ---------------------------------------------------------------------------

ALLOWED_MIME = {"image/jpeg", "image/png", "image/webp"}
MAX_BYTES = 10 * 1024 * 1024   # 10 MB


# ---------------------------------------------------------------------------
# app state (populated during lifespan)
# ---------------------------------------------------------------------------

_engine: InferenceEngine | None = None
_executor: ThreadPoolExecutor | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Everything in here runs before the server starts serving requests.
    Putting model loading here means it happens exactly once, not per request.
    """
    global _engine, _executor

    checkpoint = Path(os.getenv("V2T_CHECKPOINT", "checkpoints/best.ckpt"))
    vocab = Path(os.getenv("V2T_VOCAB_PATH", "cache/vocab.json"))
    device = os.getenv("V2T_DEVICE", "auto")
    n_workers = int(os.getenv("V2T_WORKERS", "2"))

    _engine = InferenceEngine(checkpoint, vocab, device)
    # beam search is CPU/GPU bound — run it in a thread so the event loop
    # stays free to accept new connections while inference is happening
    _executor = ThreadPoolExecutor(max_workers=n_workers, thread_name_prefix="v2t")

    yield   # server is live from here until shutdown

    _executor.shutdown(wait=True)


app = FastAPI(
    title="Vision2Text",
    version="0.1.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# routes
# ---------------------------------------------------------------------------

@app.post("/caption", response_model=CaptionResponse)
async def caption_image(
    file: UploadFile = File(..., description="JPEG, PNG, or WEBP image (max 10 MB)"),
    beam_size: int = Form(default=4, ge=1, le=16),
):
    """
    Upload an image and get back a natural-language caption.

    The image is read into memory and never written to disk — no race conditions
    between concurrent users and no leftover temp files.
    """
    # --- content-type gate ---
    if file.content_type not in ALLOWED_MIME:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Unsupported file type '{file.content_type}'. "
                "Please upload a JPEG, PNG, or WEBP image."
            ),
        )

    raw = await file.read()

    # --- size gate ---
    if len(raw) > MAX_BYTES:
        mb = len(raw) / 1024 / 1024
        raise HTTPException(
            status_code=422,
            detail=f"Image is {mb:.1f} MB — maximum allowed size is 10 MB.",
        )

    # --- actually-a-valid-image gate ---
    # Browsers and clients can lie about content-type; Pillow is the authority.
    try:
        Image.open(io.BytesIO(raw)).verify()
    except (UnidentifiedImageError, Exception) as exc:
        raise HTTPException(
            status_code=422,
            detail=f"Could not decode image: {exc}",
        ) from exc

    # --- inference (off the event loop) ---
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    result: dict = await loop.run_in_executor(_executor, _engine.caption, raw, beam_size)
    _record(time.perf_counter() - t0)

    return result


@app.get("/health", response_model=HealthResponse)
async def health():
    """Liveness + readiness in one shot: tells you whether the model is up."""
    return {
        "status": "ok",
        "model_loaded": _engine is not None,
        "device": str(_engine.device) if _engine else "unknown",
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """
    Prometheus-compatible plaintext exposition.

    Tracks total requests and p50/p95 latency since the process started.
    """
    with _metrics_lock:
        count = _request_count
        lats = sorted(_latencies)

    p50 = _percentile(lats, 0.50)
    p95 = _percentile(lats, 0.95)

    return (
        "# HELP v2t_requests_total Total number of /caption requests handled\n"
        "# TYPE v2t_requests_total counter\n"
        f"v2t_requests_total {count}\n"
        "\n"
        "# HELP v2t_request_duration_seconds End-to-end latency of /caption requests\n"
        "# TYPE v2t_request_duration_seconds summary\n"
        f'v2t_request_duration_seconds{{quantile="0.5"}} {p50:.4f}\n'
        f'v2t_request_duration_seconds{{quantile="0.95"}} {p95:.4f}\n'
        f"v2t_request_duration_seconds_count {count}\n"
    )
