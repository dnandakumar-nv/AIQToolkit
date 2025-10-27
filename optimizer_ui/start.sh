#!/bin/bash

# SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Startup script for NeMo Agent Toolkit Optimizer UI

set -e

echo "🚀 Starting NeMo Agent Toolkit Optimizer UI..."
echo ""

# Check if we're in the correct directory
if [ ! -f "backend/main.py" ]; then
    echo "❌ Error: Please run this script from the optimizer_ui directory"
    exit 1
fi

# Check Python version
PYTHON_VERSION=$(python3 --version 2>&1 | awk '{print $2}')
echo "✓ Python version: $PYTHON_VERSION"

# Check if virtual environment exists
#if [ ! -d "venv" ]; then
#    echo "📦 Creating virtual environment..."
#    python3 -m venv venv
#fi
#
## Activate virtual environment
#echo "🔧 Activating virtual environment..."
#source venv/bin/activate

# Install dependencies
echo "📦 Installing dependencies..."
uv pip install -q --upgrade pip
uv pip install -r requirements.txt

# Navigate to parent directory to ensure NAT is importable
cd ..

# Start the backend server
echo ""
echo "✅ Starting backend server on http://localhost:8080"
echo "📊 Open http://localhost:8080 in your browser"
echo ""
echo "Press Ctrl+C to stop the server"
echo ""

python3 -m uvicorn optimizer_ui.backend.main:app --host 0.0.0.0 --port 8080 --reload
