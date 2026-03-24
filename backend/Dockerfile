# ── Backend Dockerfile ───────────────────────────────────────────
# Multi-stage: builder installs deps, runner is slim final image.

# Stage 1 — dependency builder
FROM python:3.11-slim AS builder

WORKDIR /app

# System deps for building native extensions
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --user --no-cache-dir -r requirements.txt


# Stage 2 — runtime
FROM python:3.11-slim AS runner

WORKDIR /app

# Runtime system deps (libgomp needed by faiss)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /root/.local /root/.local
ENV PATH=/root/.local/bin:$PATH

# Copy application code
COPY app/ ./app/

# Create data directories
RUN mkdir -p ./data/faiss_index ./models/lora_adapter

# Non-root user for security
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=15s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["uvicorn", "app.main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--workers", "1", \
     "--log-level", "info"]
