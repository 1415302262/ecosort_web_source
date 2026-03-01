# ♻️ EcoSort: Edge-Cloud Cascade Waste Classification System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://students-cs-ecosort-14class-demo.hf.space)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Live Demo: https://students-cs-ecosort-14class-demo.hf.space/

## Project Overview
EcoSort is an intelligent waste classification system built with an edge-cloud cascade pipeline.

- Edge: local CNN inference (EfficientNet/ResNet) for fast response.
- Cloud fallback: optional VLM verification for low-confidence or ambiguous cases.
- Production-oriented preprocessing: letterbox padding and robust runtime mapping.

## Repository Structure
```text
ecosort-web/
├── backend/                # Flask API for inference and cascade fallback
├── src/                    # Core modules: data, models, train, utils
├── experiments/            # Training and evaluation entry scripts
├── configs/                # Training and dataset mapping configurations
├── docs/                   # Technical and dataset-related documentation
├── scripts/                # Utility scripts for data and training workflow
├── data/                   # Local data workspace (ignored by Git)
├── checkpoints/            # Model checkpoints (ignored by Git, managed by DVC)
├── streamlit_demo.py       # Local demo app
├── environment.yml         # Conda environment definition
└── requirements.txt        # Pip dependencies
```

## Quick Start
### 1) Environment
```bash
conda env create -f environment.yml
conda activate ecosort
pip install -r requirements.txt
```

### 2) Training
```bash
python experiments/train_baseline.py \
  --config configs/efficientnet_b3.yaml \
  --data-root data/proc
```

### 3) Evaluation
```bash
python experiments/evaluate.py \
  --checkpoint checkpoints/<exp_name>/best_model.pth \
  --data-root data/proc \
  --model-type efficientnet
```

### 4) Run API
```bash
python backend/app.py
```

### 5) Run Streamlit Demo
```bash
streamlit run streamlit_demo.py
```

## Data & Checkpoints
- Large datasets and model weights are not committed to Git.
- This repo uses DVC workflow for `data/` and `checkpoints/` artifacts.
- Keep full training state in checkpoints: model, optimizer, scheduler, epoch, best score.

## Deliverable Scope
This repository is cleaned for delivery and focuses on:
- reproducible training/evaluation entry points,
- runnable API/demo code,
- essential configs and docs.

For handover details, see `HANDOVER_CHECKLIST.md` and `QUICK_START.md`.

