#!/bin/bash
set -e

echo "=== Paper RAG Startup ==="

# Download models if the cache is empty (first run)
python scripts/download_models.py

# Start FastAPI in background
echo "Starting FastAPI on 0.0.0.0:8000 ..."
uvicorn app.main:app --host 0.0.0.0 --port 8000 --workers 1 &
FASTAPI_PID=$!

# Short delay to let FastAPI start
sleep 2

# Start Gradio in foreground
echo "Starting Gradio on 0.0.0.0:7860 ..."
python -c "
from app.config import get_config
from app.frontend.gradio_app import run_gradio
run_gradio()
" &
GRADIO_PID=$!

echo "FastAPI PID: $FASTAPI_PID"
echo "Gradio PID: $GRADIO_PID"

# Trap signals to shut down gracefully
trap "kill $FASTAPI_PID $GRADIO_PID 2>/dev/null; exit" SIGTERM SIGINT

# Wait for any process to exit
wait
