# ============================================================
# Stage 1 — builder
# Install every dependency in a throwaway environment so the
# final image doesn't carry pip, wheel caches, or build tools.
# ============================================================
FROM python:3.11 AS builder

WORKDIR /app

# Copy dependency manifests first — Docker cache will skip the
# pip install layer on every code change if these files haven't changed.
COPY requirements.txt requirements-api.txt ./

RUN pip install --no-cache-dir \
        -r requirements.txt \
        -r requirements-api.txt


# ============================================================
# Stage 2 — runtime
# Slim base image; we only copy the installed packages from
# the builder. Model weights are NOT baked in — mount them
# at runtime via a Docker volume or download at startup.
# ============================================================
FROM python:3.11-slim AS runtime

WORKDIR /app

# Pull in the installed site-packages and entry-point scripts
COPY --from=builder /usr/local/lib/python3.11/site-packages \
                    /usr/local/lib/python3.11/site-packages
COPY --from=builder /usr/local/bin /usr/local/bin

# Copy only source code — no notebooks, no data, no checkpoints
COPY src/ ./src/

# Non-root user: if the container is ever compromised, the attacker
# can't write anywhere meaningful on the host.
RUN useradd --no-create-home --system --uid 1001 appuser \
 && chown -R appuser:appuser /app

USER appuser

EXPOSE 8000

# The health check lets Docker (and docker-compose) know when the
# server is actually ready to accept traffic, not just "running".
# start-period gives the model time to load before checks begin.
HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" \
      || exit 1

CMD ["uvicorn", "src.api.app:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1"]
