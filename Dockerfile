FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl && \
    rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . /app

# Install server dependencies (minimal — no torch needed server-side)
RUN pip install --no-cache-dir \
    "openenv-core[core]>=0.2.2" \
    "fastapi>=0.100.0" \
    "uvicorn>=0.23.0" \
    "numpy>=1.24.0"

# Install the package itself so imports work
RUN pip install --no-cache-dir --no-deps -e .

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

ENV ENABLE_WEB_INTERFACE=true
CMD uvicorn server.app:app --host "0.0.0.0" --port "7860" --workers 1
