# ── Builder stage ─────────────────────────────────────────────────────────────
FROM python:3.11-slim AS builder

WORKDIR /app

# Install system deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install uv for fast dependency resolution
RUN pip install --no-cache-dir uv

# Copy dependency spec first for layer caching
COPY warehouse_env/pyproject.toml ./pyproject.toml

# Install all dependencies into a venv
RUN uv venv /app/.venv && \
    . /app/.venv/bin/activate && \
    uv pip install openenv-core fastapi "uvicorn[standard]" pydantic numpy pillow openai

# Copy full project
COPY warehouse_env/ ./warehouse_env/

# ── Runtime stage ──────────────────────────────────────────────────────────────
FROM python:3.11-slim

WORKDIR /app

# Copy the venv and the code
COPY --from=builder /app/.venv /app/.venv
COPY --from=builder /app/warehouse_env /app/warehouse_env

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONPATH="/app:$PYTHONPATH"

# HF Spaces uses port 7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

CMD ["uvicorn", "warehouse_env.server.app:app", \
     "--host", "0.0.0.0", "--port", "7860", "--workers", "2"]
