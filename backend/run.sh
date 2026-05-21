#!/usr/bin/env bash
# -----------------------------------------------------------------------------
# MedLens backend launcher (Linux / macOS)
# Activates the local venv if it exists, then starts uvicorn on port 8000.
# -----------------------------------------------------------------------------
set -e

cd "$(dirname "$0")"

if [ -d "venv" ]; then
  # shellcheck disable=SC1091
  source venv/bin/activate
fi

echo "[medlens] Starting FastAPI on http://127.0.0.1:8000 ..."
echo "[medlens] Swagger UI:  http://127.0.0.1:8000/docs"
echo ""

uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
