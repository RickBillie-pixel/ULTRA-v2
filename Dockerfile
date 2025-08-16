# Use Python 3.11 slim image for better performance and smaller size
FROM python:3.11-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONIOENCODING=utf-8 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PLAYWRIGHT_BROWSERS_PATH=/ms-playwright \
    PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD=0

# Set work directory
WORKDIR /app

# Install essential system dependencies only (skip problematic font packages)
RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        wget \
        ca-certificates \
        gnupg \
        dumb-init \
        # Essential for Playwright
        libnss3 \
        libatk-bridge2.0-0 \
        libdrm2 \
        libxkbcommon0 \
        libgtk-3-0 \
        libgbm1 \
        libasound2 \
        # Build essentials (minimal)
        gcc \
        g++ \
        # Cleanup problematic packages and cache
    && apt-get autoremove -y \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /tmp/* \
    && rm -rf /var/tmp/*

# Copy requirements first for better Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Install Playwright with Chromium only (skip font dependencies)
RUN python -m playwright install chromium --with-deps \
    && python -m playwright install-deps chromium \
    # Clean up unnecessary files
    && find /ms-playwright -name "*.log" -delete \
    && find /ms-playwright -name "*.tmp" -delete

# Copy application code
COPY . .

# Create non-root user for security
RUN groupadd --gid 1000 appuser \
    && useradd --uid 1000 --gid appuser --shell /bin/bash --create-home appuser \
    && chown -R appuser:appuser /app \
    && chown -R appuser:appuser /ms-playwright

# Switch to non-root user
USER appuser

# Expose port
EXPOSE 8000

# Health check with timeout adjustments for browser pool
HEALTHCHECK --interval=60s --timeout=30s --start-period=120s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Use dumb-init to handle signals properly and improve browser process management
ENTRYPOINT ["/usr/bin/dumb-init", "--"]

# Command to run the application with optimized settings
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1", "--access-log", "--log-level", "info"]
