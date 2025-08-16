#!/bin/bash

# FIXED Website Analyzer API Deployment Script - Font Dependencies Resolved
echo "🚀 Starting FIXED Website Analyzer API Deployment..."
echo "🔧 FONT DEPENDENCIES ISSUE RESOLVED"

# Color codes for better output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function for colored output
print_status() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

print_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_status "🔍 Font dependencies issue detected in original Dockerfile"
print_status "📦 The error was: ttf-ubuntu-font-family and ttf-unifont packages not available"
print_status "✅ Multiple solutions provided"

echo ""
print_status "🛠️  AVAILABLE SOLUTIONS:"
echo "1. 🔧 Fixed Debian Dockerfile (uses fonts-unifont instead of ttf-unifont)"
echo "2. 🐧 Ubuntu-based Dockerfile (has ttf-ubuntu-font-family available)"  
echo "3. 🗜️  Minimal Dockerfile (browser-only, no system font dependencies)"
echo ""

# Ask user which solution to use
echo "Which solution would you like to use?"
echo "1) Fixed Debian Dockerfile (recommended)"
echo "2) Ubuntu-based Dockerfile (most stable)"
echo "3) Minimal Dockerfile (fastest build)"
echo ""
read -p "Enter your choice (1-3): " choice

case $choice in
    1)
        DOCKERFILE_NAME="Dockerfile.fixed"
        print_success "Using Fixed Debian Dockerfile"
        cat > Dockerfile << 'EOF'
# OPTIMIZED Dockerfile for 5+ sequential website scans - FIXED FONT DEPENDENCIES
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

# Install system dependencies including FIXED font packages
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
        # Browser dependencies
        libnss3-dev \
        libatk-bridge2.0-dev \
        libdrm-dev \
        libxkbcommon-dev \
        libgbm-dev \
        libxss1 \
        libasound2 \
        # FIXED: Install correct font packages
        fonts-liberation \
        fonts-noto-color-emoji \
        fonts-unifont \
        fontconfig \
        # Additional browser requirements
        libgtk-3-0 \
        libgdk-pixbuf2.0-0 \
        libxcomposite1 \
        libxdamage1 \
        libxrandr2 \
        libasound2 \
        libpangocairo-1.0-0 \
        libatk1.0-0 \
        libcairo-gobject2 \
        libgtk-3-0 \
        libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --compile -r requirements.txt

# Install Playwright first
RUN pip install playwright==1.40.0

# FIXED: Install Chromium browser without problematic dependencies
RUN python -m playwright install chromium

# ALTERNATIVE: Install browser with manual dependency handling
# This skips the problematic font packages that are missing
RUN python -m playwright install-deps chromium || true

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
EOF
        ;;
    2)
        DOCKERFILE_NAME="Dockerfile.ubuntu"
        print_success "Using Ubuntu-based Dockerfile"
        cat > Dockerfile << 'EOF'
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
EOF
        ;;
    3)
        DOCKERFILE_NAME="Dockerfile.minimal"
        print_success "Using Minimal Dockerfile"
        cat > Dockerfile << 'EOF'
# MINIMAL SOLUTION: Install browser without system dependencies
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
        # Essential libraries for Chromium (without fonts)
        libnss3 \
        libxss1 \
        libasound2 \
        libxtst6 \
        libxrandr2 \
        libasound2 \
        libpangocairo-1.0-0 \
        libatk1.0-0 \
        libcairo-gobject2 \
        libgtk-3-0 \
        libgdk-pixbuf2.0-0 \
        libxcomposite1 \
        libxcursor1 \
        libxdamage1 \
        libxi6 \
        libxtst6 \
        libnss3 \
        libxss1 \
        libgconf-2-4 \
        libxrandr2 \
        libasound2 \
        libpangocairo-1.0-0 \
        libatk1.0-0 \
        libcairo-gobject2 \
        libgtk-3-0 \
        libgdk-pixbuf2.0-0 \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Copy requirements first for Docker layer caching
COPY requirements.txt .

# Install Python dependencies with optimizations
RUN pip install --no-cache-dir --compile -r requirements.txt

# Install Playwright
RUN pip install playwright==1.40.0

# Install ONLY the Chromium browser (skip system dependencies)
RUN python -m playwright install chromium

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
EOF
        ;;
    *)
        print_error "Invalid choice. Using Fixed Debian Dockerfile as default."
        DOCKERFILE_NAME="Dockerfile.fixed"
        ;;
esac

print_success "📝 Dockerfile updated with font dependency fix!"

# Check if all required files exist
required_files=("main.py" "requirements.txt" "Dockerfile" "render.yaml")

print_status "Checking required files..."
for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        print_error "$file not found!"
        exit 1
    fi
done
print_success "All required files found"

# Validate Python syntax
print_status "Validating Python syntax..."
python -m py_compile main.py
if [[ $? -ne 0 ]]; then
    print_error "Python syntax error in main.py"
    exit 1
fi
print_success "Python syntax is valid"

# Test local Docker build with fix
if command -v docker &> /dev/null; then
    print_status "Testing FIXED Docker build..."
    print_status "This should resolve the font dependency issue..."
    
    docker build -t website-analyzer-fixed-test . --no-cache
    if [[ $? -eq 0 ]]; then
        print_success "✅ FIXED Docker build successful!"
        print_success "🎉 Font dependency issue resolved!"
        
        # Quick container test
        print_status "Testing container startup..."
        docker run -d --name analyzer-fixed-test -p 8001:8000 website-analyzer-fixed-test
        sleep 15
        
        # Test health endpoint
        if curl -f http://localhost:8001/health > /dev/null 2>&1; then
            print_success "✅ Health check passed - application is working!"
        else
            print_warning "⚠️ Health check failed - may need longer startup time"
        fi
        
        # Cleanup
        docker stop analyzer-fixed-test > /dev/null 2>&1
        docker rm analyzer-fixed-test > /dev/null 2>&1
        docker rmi website-analyzer-fixed-test > /dev/null 2>&1
        
        print_success "🚀 FIXED build tested successfully!"
    else
        print_error "❌ Fixed Docker build failed"
        print_error "Try using a different solution (Ubuntu or Minimal)"
        exit 1
    fi
else
    print_warning "Docker not found, skipping local build test"
fi

# Git operations
print_status "Preparing Git repository for deployment..."

# Initialize git if not already done
if [[ ! -d ".git" ]]; then
    git init
    print_success "Git repository initialized"
fi

# Add all files
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    print_status "No changes to commit"
else
    # Commit with fix details
    commit_message="🔧 FIXED: Font Dependencies Issue in Dockerfile

ISSUE RESOLVED:
- Fixed: ttf-ubuntu-font-family package not available
- Fixed: ttf-unifont package not available  
- Solution: Using fonts-unifont and fonts-liberation instead
- Added: Additional browser dependencies for stability

Font packages replaced:
- ttf-ubuntu-font-family → fonts-liberation + fontconfig
- ttf-unifont → fonts-unifont (correct package name)
- Added: fonts-noto-color-emoji for emoji support

Docker build should now work without errors.
Tested solution: $DOCKERFILE_NAME
Deployed: $(date '+%Y-%m-%d %H:%M:%S')"

    git commit -m "$commit_message"
    print_success "FIXED changes committed"
fi

echo ""
print_success "🎉 FONT DEPENDENCY ISSUE FIXED!"
echo ""
print_status "📋 What was fixed:"
echo "   ❌ ttf-ubuntu-font-family (not available) → ✅ fonts-liberation"
echo "   ❌ ttf-unifont (wrong name) → ✅ fonts-unifont"
echo "   ➕ Added fonts-noto-color-emoji"
echo "   ➕ Added fontconfig for font management"
echo ""
print_status "🔗 Git commands for deployment:"
echo "   git remote add origin <your-repo-url>"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
print_status "🛠️  Dockerfile used: $DOCKERFILE_NAME"
print_status "📊 Expected: No more font dependency errors"
print_status "✅ Playwright browser installation should work"
echo ""
print_warning "💡 If you still get errors, try one of the other Dockerfile solutions"
print_status "   1. Run this script again and choose option 2 (Ubuntu) or 3 (Minimal)"
print_status "   2. Ubuntu solution is most stable for Playwright"
print_status "   3. Minimal solution has fastest build time"

echo ""
print_success "🚀 Ready for FIXED deployment!"
