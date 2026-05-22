@echo off
REM ----------------------------------------------------------------------------
REM  MedLens backend launcher (Windows)
REM  Activates the local venv if it exists, then starts uvicorn on port 8000.
REM ----------------------------------------------------------------------------

cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo [medlens] Starting FastAPI on http://127.0.0.1:8000 ...
echo [medlens] Swagger UI:  http://127.0.0.1:8000/docs
echo.

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
