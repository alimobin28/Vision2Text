"""
FastAPI application for Vision2Text.

The model is loaded once at startup (via the lifespan context manager) and
reused for every request — no cold starts, no disk writes for uploaded images.
"""
from __future__ import annotations

import asyncio
import io
import logging
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse
from PIL import Image

from src.api.schemas import CaptionResponse, HealthResponse
from src.config import cfg
from src.inference.engine import InferenceEngine

log = logging.getLogger("v2t.api")


# ---------------------------------------------------------------------------
# in-process metrics store (Prometheus text format at GET /metrics)
# ---------------------------------------------------------------------------

_metrics_lock = threading.Lock()
_request_count: int = 0        # every /caption request that reached validation
_error_count: int = 0          # of those, how many were rejected or failed
_latencies: list[float] = []   # seconds, successful requests only; capped at 10 000


def _record(elapsed_s: float | None) -> None:
    """Count a request. ``elapsed_s=None`` marks it as a failure."""
    global _request_count, _error_count, _latencies
    with _metrics_lock:
        _request_count += 1
        if elapsed_s is None:
            _error_count += 1
            return
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
MAX_BYTES = 10 * 1024 * 1024        # 10 MB on the wire
MAX_PIXELS = 40_000_000             # ~40 MP: a tiny PNG can otherwise decode to GBs


def _reject(detail: str) -> HTTPException:
    """422 for a bad upload; also counted as a failed request in /metrics."""
    _record(None)
    return HTTPException(status_code=422, detail=detail)


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

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    _engine = InferenceEngine(cfg.checkpoint, cfg.vocab_path, cfg.device)
    log.info("model loaded from %s on %s", cfg.checkpoint, _engine.device)

    # beam search is CPU/GPU bound — run it in threads so the event loop stays
    # free to accept new connections while inference is happening
    _executor = ThreadPoolExecutor(max_workers=cfg.workers, thread_name_prefix="v2t")

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
    if _engine is None or _executor is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")

    # --- content-type gate ---
    if file.content_type not in ALLOWED_MIME:
        raise _reject(
            f"Unsupported file type '{file.content_type}'. "
            "Please upload a JPEG, PNG, or WEBP image."
        )

    raw = await file.read()

    # --- size gate ---
    if len(raw) > MAX_BYTES:
        mb = len(raw) / 1024 / 1024
        raise _reject(f"Image is {mb:.1f} MB — maximum allowed size is 10 MB.")

    # --- actually-a-valid-image gate ---
    # Browsers and clients can lie about content-type; Pillow is the authority.
    try:
        with Image.open(io.BytesIO(raw)) as img:
            if img.width * img.height > MAX_PIXELS:
                raise _reject(
                    f"Image is {img.width}x{img.height} px — maximum is "
                    f"{MAX_PIXELS // 1_000_000} megapixels."
                )
            img.verify()
    except HTTPException:
        raise
    except Exception as exc:   # Pillow raises many types for corrupt input
        raise _reject(f"Could not decode image: {exc}") from exc

    # --- inference (off the event loop) ---
    loop = asyncio.get_running_loop()
    t0 = time.perf_counter()
    try:
        result: dict = await loop.run_in_executor(_executor, _engine.caption, raw, beam_size)
    except Exception:
        _record(None)
        log.exception("inference failed")
        raise HTTPException(status_code=500, detail="Inference failed.") from None

    elapsed = time.perf_counter() - t0
    _record(elapsed)
    log.info("caption ok: %.0f ms, beam=%d, %d tokens", elapsed * 1000, beam_size, len(result["tokens"]))
    return result


@app.get("/health", response_model=HealthResponse)
async def health():
    """Readiness: 200 only once the model is loaded, 503 otherwise."""
    if _engine is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return {
        "status": "ok",
        "model_loaded": True,
        "device": str(_engine.device),
    }


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics():
    """
    Prometheus-compatible plaintext exposition.

    Tracks total requests, failures and p50/p95 latency of successful requests
    since the process started. Nginx does not expose this route publicly; scrape
    it from inside the Docker network (api:8000).
    """
    with _metrics_lock:
        count = _request_count
        errors = _error_count
        lats = sorted(_latencies)

    p50 = _percentile(lats, 0.50)
    p95 = _percentile(lats, 0.95)

    return (
        "# HELP v2t_requests_total Total number of /caption requests handled\n"
        "# TYPE v2t_requests_total counter\n"
        f"v2t_requests_total {count}\n"
        "\n"
        "# HELP v2t_request_errors_total /caption requests that were rejected or failed\n"
        "# TYPE v2t_request_errors_total counter\n"
        f"v2t_request_errors_total {errors}\n"
        "\n"
        "# HELP v2t_request_duration_seconds End-to-end latency of successful /caption requests\n"
        "# TYPE v2t_request_duration_seconds summary\n"
        f'v2t_request_duration_seconds{{quantile="0.5"}} {p50:.4f}\n'
        f'v2t_request_duration_seconds{{quantile="0.95"}} {p95:.4f}\n'
        f"v2t_request_duration_seconds_count {count - errors}\n"
    )
