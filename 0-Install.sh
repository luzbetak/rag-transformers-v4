#!/bin/bash

# 0-Installation.sh
# Transformers v5 RAG System - Setup Script
# Run this first to install all dependencies

set -e

echo "🚀 Installing Transformers v5 RAG System..."
echo ""

# Upgrade pip
echo "📦 Step 1: Upgrading pip..."
pip install --upgrade pip setuptools wheel > /dev/null 2>&1

# Install torch FIRST (it's required by flash-attn)
echo "📦 Step 2: Installing PyTorch (this may take a minute)..."
pip install torch==2.1.0 > /dev/null 2>&1

# Install other requirements
echo "📦 Step 3: Installing dependencies from requirements.txt..."
pip install -r requirements.txt

echo ""
echo "✅ Installation complete!"
echo ""
echo "Optional: To enable Flash Attention 2 (faster), run:"
echo "  pip install flash-attn==2.3.5"
echo ""
echo "Next steps:"
echo "  1. If you have documents, run: python 1-Large-Text-Chunking.py"
echo "  2. Then run: python 2-RAG-Indexer.py"
echo "  3. Finally run: python 4-RAG-Search.py"
echo ""
