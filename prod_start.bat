@echo off
chcp 65001 >nul
title tAIres Studio [PROD]
cd /d "%~dp0"

set PYTHONPATH=src

where python >nul 2>nul
IF ERRORLEVEL 1 (
    echo [ERROR] Python not found. Install Python 3.10+ and add it to PATH.
    pause
    exit /b 1
)

IF NOT EXIST ".venv\Scripts\activate.bat" (
    echo [1/2] Initializing clean Python environment...
    python -m venv .venv

    echo [2/2] Installing dependencies from requirements.txt...
    call .venv\Scripts\activate.bat
    python -m pip install --upgrade pip >nul
    pip install -r requirements.txt --quiet
) ELSE (
    call .venv\Scripts\activate.bat
)

echo Launching tAIres Studio...
python -m streamlit run src/taires/ui/app.py