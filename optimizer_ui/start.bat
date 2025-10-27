@echo off
REM SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
REM SPDX-License-Identifier: Apache-2.0

REM Startup script for NeMo Agent Toolkit Optimizer UI (Windows)

echo Starting NeMo Agent Toolkit Optimizer UI...
echo.

REM Check if we're in the correct directory
if not exist "backend\main.py" (
    echo Error: Please run this script from the optimizer_ui directory
    exit /b 1
)

REM Check if virtual environment exists
if not exist "venv" (
    echo Creating virtual environment...
    python -m venv venv
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat

REM Install dependencies
echo Installing dependencies...
pip install --upgrade pip
pip install -r requirements.txt

REM Navigate to parent directory
cd ..

REM Start the backend server
echo.
echo Starting backend server on http://localhost:8080
echo Open http://localhost:8080 in your browser
echo.
echo Press Ctrl+C to stop the server
echo.

python -m uvicorn optimizer_ui.backend.main:app --host 0.0.0.0 --port 8080 --reload
