#!/usr/bin/env bash
set -euo pipefail

# One-command, no-network release gate. Run inside the cptac environment:
# conda run -n cptac bash scripts/release_check.sh

python -m py_compile \
  backend/app.py \
  backend/wsgi.py \
  backend/web.py \
  backend/store.py \
  src/inference/file_lock.py \
  src/inference/cascade.py \
  src/inference/seed_vlm.py \
  scripts/run_ecosort_vlm_fallback.py \
  scripts/run_ecosort_seed_vlm_fallback.py \
  scripts/analyze_ecosort_vlm_cascade.py \
  scripts/analyze_ecosort_cascade_robustness.py \
  scripts/plot_ecosort_cascade_portfolio.py \
  scripts/validate_public_release.py

bash -n scripts/render_start.sh

PYTHONPATH=. pytest -q tests

python scripts/validate_public_release.py
git diff --check
