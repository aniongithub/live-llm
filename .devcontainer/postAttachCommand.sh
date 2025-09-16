#!/bin/bash

set -euo pipefail

echo "🚀 Setting up Live LLM development environment..."

# Authenticate with Hugging Face
if [ -n "$HUGGINGFACE_TOKEN" ]; then
    echo "🔐 Authenticating with Hugging Face..."
    huggingface-cli login --token "$HUGGINGFACE_TOKEN"
    
    if [ $? -eq 0 ]; then
        echo "✅ Hugging Face authentication successful"
    else
        echo "❌ Hugging Face authentication failed"
        exit 1
    fi
else
    echo "❌ HUGGINGFACE_TOKEN not found in .env file"
    exit 1
fi

# Install requirements if needed
echo "📦 Installing Python requirements..."
pip install -r requirements.txt

# Install client requirements
echo "📦 Installing client requirements..."
npm install --prefix client

echo "✅ Setup complete! You can now run the Live LLM models."
echo ""
echo "To test the setup, run the launch configuration \"Live LLM (web)\""
