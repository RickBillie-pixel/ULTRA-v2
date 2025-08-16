#!/bin/bash

# Website Analyzer API with Browser Pool Management - Deployment Script for Render
# Make sure this file is executable: chmod +x deploy.sh

echo "🚀 Starting Website Analyzer API with Browser Pool Deployment..."

# Check if all required files exist
required_files=("main.py" "requirements.txt" "Dockerfile" "render.yaml")

for file in "${required_files[@]}"; do
    if [[ ! -f "$file" ]]; then
        echo "❌ Error: $file not found!"
        exit 1
    fi
done

echo "✅ All required files found"

# Validate Python syntax
echo "🔍 Validating Python syntax..."
python -m py_compile main.py
if [[ $? -ne 0 ]]; then
    echo "❌ Python syntax error in main.py"
    exit 1
fi

echo "✅ Python syntax is valid"

# Test local Docker build (optional)
if command -v docker &> /dev/null; then
    echo "🐳 Testing Docker build locally..."
    docker build -t website-analyzer-pooled-test .
    if [[ $? -eq 0 ]]; then
        echo "✅ Docker build successful"
        docker rmi website-analyzer-pooled-test
    else
        echo "❌ Docker build failed"
        exit 1
    fi
else
    echo "⚠️  Docker not found, skipping local build test"
fi

# Git operations
echo "📦 Preparing Git repository..."

# Initialize git if not already done
if [[ ! -d ".git" ]]; then
    git init
    echo "✅ Git repository initialized"
fi

# Add gitignore if it doesn't exist
if [[ ! -f ".gitignore" ]]; then
    cat > .gitignore << EOF
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg
MANIFEST
.env
.venv
env/
venv/
ENV/
env.bak/
venv.bak/
.pytest_cache/
.coverage
htmlcov/
.tox/
.cache
nosetests.xml
coverage.xml
*.cover
.hypothesis/
.DS_Store
EOF
    echo "✅ .gitignore created"
fi

# Add all files
git add .

# Check if there are changes to commit
if git diff --staged --quiet; then
    echo "ℹ️  No changes to commit"
else
    # Commit changes
    git commit -m "Deploy Website Analyzer API v4.1.1 with Browser Pool Management - $(date '+%Y-%m-%d %H:%M:%S')"
    echo "✅ Changes committed"
fi

echo ""
echo "🎉 Deployment preparation complete!"
echo ""
echo "📋 Next steps:"
echo "1. Push to your Git repository (GitHub, GitLab, etc.)"
echo "2. Connect your repository to Render"
echo "3. Render will automatically detect render.yaml and deploy"
echo ""
echo "🔗 Useful commands:"
echo "   git remote add origin <your-repo-url>"
echo "   git branch -M main"
echo "   git push -u origin main"
echo ""
echo "📊 API Endpoints after deployment:"
echo "   GET  / - API documentation"
echo "   GET  /analyze/{url} - Quick website analysis" 
echo "   POST /analyze/combined - Full analysis with browser pool optimization"
echo ""
echo "💡 Example usage:"
echo "   curl https://your-app.onrender.com/analyze/combined/google.com"
echo ""
echo "🔧 Browser Pool Features:"
echo "   • Max 2 concurrent browsers for efficiency"
echo "   • Browser reuse across sequential scans"
echo "   • Automatic memory cleanup every 5 requests"
echo "   • Optimized for 5+ sequential scans without timeouts"
echo ""

# Show file structure
echo "📁 Project structure:"
find . -type f -name "*.py" -o -name "*.txt" -o -name "*.yaml" -o -name "Dockerfile" | head -20
