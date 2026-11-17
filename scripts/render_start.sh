#!/usr/bin/env bash
set -euo pipefail

MODEL_PATH="${ECOSORT_MODEL_PATH:-checkpoints/best_model.pth}"
MODEL_URL="${ECOSORT_MODEL_URL:-}"
EXPECTED_SHA256="${ECOSORT_EXPECTED_MODEL_SHA256:-9ba817cdf0712b597e1eb322d722e130707d60d6733cf46d8d87149456946234}"

is_valid_checkpoint() {
  local candidate="$1"
  [[ -f "$candidate" ]] || return 1
  [[ $(wc -c < "$candidate") -gt 1048576 ]] || return 1
  if head -c 64 "$candidate" | grep -q 'git-lfs.github.com/spec'; then
    return 1
  fi
  if [[ -n "$EXPECTED_SHA256" ]]; then
    [[ "$(sha256sum "$candidate" | awk '{print $1}')" == "$EXPECTED_SHA256" ]] || return 1
  fi
}

if ! is_valid_checkpoint "$MODEL_PATH" && [[ -n "$MODEL_URL" ]]; then
  echo "[render-start] downloading audited checkpoint from ECOSORT_MODEL_URL..."
  mkdir -p "$(dirname "$MODEL_PATH")"
  DOWNLOAD_PATH="${MODEL_PATH}.download"
  curl -fL "$MODEL_URL" -o "$DOWNLOAD_PATH"
  if ! is_valid_checkpoint "$DOWNLOAD_PATH"; then
    echo "[render-start] downloaded checkpoint failed size/LFS/SHA256 validation" >&2
    exit 1
  fi
  mv "$DOWNLOAD_PATH" "$MODEL_PATH"
  echo "[render-start] audited checkpoint installed at $MODEL_PATH"
fi

if ! is_valid_checkpoint "$MODEL_PATH"; then
  echo "[render-start] warning: runnable audited checkpoint unavailable at $MODEL_PATH"
  echo "[render-start] web UI and health endpoints remain available; /predict stays disabled"
fi

exec gunicorn backend.wsgi:app --workers 1 --threads 4 --timeout 180 --bind "0.0.0.0:${PORT:-5000}"
