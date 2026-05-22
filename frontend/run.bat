@echo off
REM ----------------------------------------------------------------------------
REM  MedLens frontend launcher (Windows)
REM ----------------------------------------------------------------------------

cd /d "%~dp0"

if exist "venv\Scripts\activate.bat" (
    call venv\Scripts\activate.bat
)

echo [medlens] Starting Streamlit on http://localhost:8501 ...
echo.

streamlit run streamlit_app.py
