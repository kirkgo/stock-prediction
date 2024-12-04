# Use Python 3.11 slim base image
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONPATH=/app \
    DEBIAN_FRONTEND=noninteractive \
    PATH="/home/appuser/.local/bin:${PATH}"  # Adicionando o path do usuário

# Set working directory
WORKDIR /app

# Install system dependencies
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

# Create non-root user and set up directories
RUN useradd -m -u 1000 appuser \
    && mkdir -p /app/models/saved_models /app/models/checkpoints \
    && chown -R appuser:appuser /app

# Switch to non-root user
USER appuser

# Copy requirements first (with appropriate ownership)
COPY --chown=appuser:appuser requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir --user -r requirements.txt \
    && pip install --no-cache-dir --user uvicorn

# Copy project files (with appropriate ownership)
COPY --chown=appuser:appuser . .

# Ensure startup script is executable
RUN chmod +x startup.sh

# Expose ports for the API and Prometheus
EXPOSE 8001 8000

# Healthcheck
HEALTHCHECK --interval=30s --timeout=3s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8001/health || exit 1

# Command to run the startup script
CMD ["/app/startup.sh"]
