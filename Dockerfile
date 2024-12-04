FROM python:3.11-slim

LABEL maintainer="Kirk Patrick"
LABEL description="Stock Prediction API with LSTM model"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/home/appuser/.local/bin:${PATH}"

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    libhdf5-dev \
    liblapack-dev \
    gfortran \
    pkg-config \
    wget \
    git \
    libblas-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/models/saved_models /app/models/checkpoints \
    && chown -R appuser:appuser /app

USER appuser

COPY --chown=appuser:appuser requirements.txt .

RUN pip install --no-cache-dir --user -r requirements.txt \
    && pip install --no-cache-dir --user uvicorn

COPY --chown=appuser:appuser . .

RUN chmod +x startup.sh

EXPOSE 8001 8000

HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

CMD ["/app/startup.sh"]
