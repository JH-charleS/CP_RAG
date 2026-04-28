@echo off
setlocal

cd /d "%~dp0"

echo [CP_RAG] Checking Python...
python --version
if errorlevel 1 (
  echo Python not found. Please activate your Conda environment first.
  pause
  exit /b 1
)

echo [CP_RAG] Installing dependencies...
python -m pip install -r requirements.txt
if errorlevel 1 (
  echo Dependency installation failed.
  pause
  exit /b 1
)

echo [CP_RAG] Starting server at http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"
python -m uvicorn main:app --host 127.0.0.1 --port 8000
