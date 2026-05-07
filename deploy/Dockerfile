# ── Stage 1: base image with CUDA + Python ────────────────────────────────────
FROM pytorch/pytorch:2.5.1-cuda12.1-cudnn9-runtime

LABEL maintainer="DATN Pancreas Segmentation"
LABEL description="Inference server for pancreas/tumor CT segmentation"

WORKDIR /app

# Install system deps for NIfTI I/O
RUN apt-get update && apt-get install -y --no-install-recommends \
        libglib2.0-0 \
        libsm6 \
        libxext6 \
    && rm -rf /var/lib/apt/lists/*

# ── Python dependencies ────────────────────────────────────────────────────────
COPY deploy/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r /app/requirements.txt

# ── Copy project source ────────────────────────────────────────────────────────
COPY monai_benchmark/ /app/monai_benchmark/
COPY deploy/server.py /app/server.py

# Checkpoints are mounted at runtime (see docker-compose.yml)
# VOLUME ["/app/checkpoints"]

# ── Runtime config (override with -e in docker run) ───────────────────────────
ENV MODEL_NAME=nnunet \
    MODEL_FOLD=0 \
    CKPT_DIR=/app/checkpoints \
    HU_MIN=-74 \
    HU_MAX=140 \
    DEVICE=cuda \
    SW_BATCH=4 \
    SW_OVERLAP=0.25

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"

CMD ["python", "-m", "uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
