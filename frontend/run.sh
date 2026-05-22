#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# MedLens frontend launcher (Linux / macOS)
# -----------------------------------------------------------------------------
set -e

cd "$(dirname "$0")"

if [ -d "venv" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

echo "[medlens] Starting Streamlit on http://localhost:8501 ..."
echo ""

streamlit run streamlit_app.py
