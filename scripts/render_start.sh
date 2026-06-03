#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${ECOSORT_MODEL_PATH:-checkpoints/best_model.pth}"
MODEL_URL="${ECOSORT_MODEL_URL:-}"

if [[ ! -f "$MODEL_PATH" && -n "$MODEL_URL" ]]; then
  echo "[render-start] checkpoint not found, downloading from ECOSORT_MODEL_URL..."
  mkdir -p "$(dirname "$MODEL_PATH")"
  curl -fL "$MODEL_URL" -o "$MODEL_PATH"
  echo "[render-start] checkpoint downloaded to $MODEL_PATH"
fi

if [[ ! -f "$MODEL_PATH" ]]; then
  echo "[render-start] warning: checkpoint missing at $MODEL_PATH"
  echo "[render-start] service will start, but /predict returns model not loaded until checkpoint is provided"
fi

exec gunicorn backend.wsgi:app --workers 1 --threads 4 --timeout 180 --bind "0.0.0.0:${PORT:-5000}"
