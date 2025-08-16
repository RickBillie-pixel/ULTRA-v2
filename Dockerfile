# ALTERNATIVE SOLUTION: Ubuntu-based Dockerfile (more stable with Playwright)
FROM ubuntu:22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

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

# Install Python and system dependencies
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        python3.11 \
        python3.11-pip \
        python3.11-dev \
        gcc \
        g++ \
        libxml2-dev \
        libxslt-dev \
        libffi-dev \
        libssl-dev \
        curl \
        ca-certificates \
        # Browser dependencies that work on Ubuntu 22.04
        libnss3-dev \
        libatk-bridge2.0-dev \
        libdrm-dev \
        libxkbcomposite-dev \
        libgbm-dev \
        libxss1 \
        libasound2 \
        # Font packages that exist on Ubuntu
        fonts-liberation \
        fonts-noto-color-emoji \
        fonts-unifont \
        ttf-ubuntu-font-family \
        fontconfig \
        # Additional browser requirements
        libgtk-3-0 \
        libgdk-pixbuf2.0-0 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libpangocairo-1.0-0 \
        libatk1.0-0 \
        libcairo-gobject2 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create symlinks for python and pip
RUN ln -s /usr/bin/python3.11 /usr/bin/python \
    && ln -s /usr/bin/pip3.11 /usr/bin/pip

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --compile -r requirements.txt

# Install Chromium with dependencies (should work on Ubuntu)
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
