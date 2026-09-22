@echo off
title tAIres Studio [DEV]
cd /d "%~dp0"

set PYTHONPATH=src

poetry run streamlit run src/taires/ui/app.py