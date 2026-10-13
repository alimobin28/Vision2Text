# ============================================================
# API image (FastAPI + the trained model).
#
# Only the SERVING dependencies are installed (requirements-infer.txt):
# no Lightning, no dataset tooling, no test tools. torch comes from the
# CPU wheel index by default (~200 MB instead of ~2.5 GB with CUDA).
#
#   GPU build:  docker build --build-arg TORCH_INDEX=https://download.pytorch.org/whl/cu121 .
# ============================================================

# ---------- Stage 1: builder -----------------------------------------
# Install in a throwaway environment so the final image carries no pip
# caches or build tools.
FROM python:3.11 AS builder

WORKDIR /app
ARG TORCH_INDEX=https://download.pytorch.org/whl/cpu

# Manifests first: Docker skips the install layer when only code changed.
COPY requirements-infer.txt ./

RUN pip install --no-cache-dir torch torchvision --index-url ${TORCH_INDEX} \
 && pip install --no-cache-dir -r requirements-infer.txt


# ---------- Stage 2: runtime -----------------------------------------
# Slim base; model weights are NOT baked in — mount them at runtime
# (see docker-compose.yml) so a new checkpoint needs no rebuild.
FROM python:3.11-slim AS runtime

WORKDIR /app
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Source only — no notebooks, data or checkpoints.
COPY src/ ./src/

# Non-root user: a compromised container cannot write anywhere meaningful.
# The engine no longer downloads ImageNet weights, so it needs no home/cache dir.
RUN useradd --no-create-home --system --uid 1001 appuser \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# /health returns 503 until the model is loaded, so "healthy" really means
# "ready to caption". start-period gives the model time to load.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
      || exit 1

CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
