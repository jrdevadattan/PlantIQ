# PlantIQ Backend — Docker Image
# ================================
# Multi-stage build: install dependencies, then copy app code.
# Runs FastAPI via uvicorn on port 8000.

FROM python:3.10-slim AS base

# System deps for XGBoost (libomp), numpy, torch
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        build-essential \
        libomp-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ── Install Python deps ──────────────────────────────────
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ── Copy application code ────────────────────────────────
COPY . .

# ── Expose port ──────────────────────────────────────────
EXPOSE 8000

# ── Health check ─────────────────────────────────────────
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

# ── Start ────────────────────────────────────────────────
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
