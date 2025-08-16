# OPTIMIZED Dockerfile for 5+ sequential website scans
FROM python:3.11-slim

# Optimized environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    # Memory optimization flags
    MALLOC_TRIM_THRESHOLD=100000 \
    MALLOC_MMAP_THRESHOLD=100000

# Set work directory
WORKDIR /app

# Install minimal system dependencies for speed
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        gcc \
        g++ \
        libxml2-dev \
        libxslt-dev \
        libffi-dev \
        libssl-dev \
        curl \
        ca-certificates \
        # Minimal browser dependencies
        libnss3-dev \
        libatk-bridge2.0-dev \
        libdrm-dev \
        libxkbcommon-dev \
        libgbm-dev \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --compile -r requirements.txt

# Install only Chromium (most stable for sequential scans)
RUN python -m playwright install chromium --with-deps

# Copy application code
COPY . .

# Create optimized non-root user
RUN adduser --disabled-password --gecos '' --shell /bin/bash appuser && \
    chown -R appuser:appuser /app && \
    chown -R appuser:appuser /ms-playwright
USER appuser

# Expose port
EXPOSE 8000

# Optimized health check for faster startup
HEALTHCHECK --interval=60s --timeout=30s --start-period=60s --retries=2 \
    CMD curl -f http://localhost:8000/health || exit 1

# OPTIMIZED startup with 2 workers for concurrent handling
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "2", "--loop", "uvloop", "--http", "httptools"]
