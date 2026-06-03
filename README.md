# EcoSort Web: Edge-Cloud Waste Classification

EcoSort is an end-to-end computer vision project for real-world waste classification. It includes the training pipeline, model evaluation utilities, a Flask inference backend, and deployment adapters for GitHub, Render, and Hugging Face Spaces.

The project is packaged as a clean public repository: source code, configs, sample dataset structure, and deployment docs are included; raw datasets, experiment logs, local environments, and large temporary artifacts are excluded.

## Core Features

* **Edge-cloud cascade inference**: EfficientNet-B3 handles local low-latency classification, while an optional cloud VLM fallback can be enabled for uncertain samples.
* **Robust preprocessing**: `Letterbox` resizing preserves aspect ratio and avoids distortion on real user photos.
* **Practical taxonomy**: The runtime mapping supports a daily-use waste taxonomy with hazardous, recyclable, kitchen, and residual handling logic.
* **Deployment-ready layout**: Flask backend, static frontend, Hugging Face Space entry point, and Render startup files are organized for direct upload.

## Repository Structure
```text
ecosort_web_source/
├── backend/
│   ├── app.py                  # Flask REST API with cascade logic
│   └── static/index.html       # Frontend interface
├── experiments/
│   ├── train_baseline.py       # Training pipeline entry point
│   └── evaluate.py             # Model evaluation and metrics
├── src/                        # Core ML modules (models, datasets, transforms)
├── configs/                    # YAML configs for training and label mapping
├── data/
│   └── sample/                 # Directory structure template for dataset
├── checkpoints/                # Model weights directory (see README inside)
├── ecosort_mapping.yaml        # Runtime confidence thresholds and fallback settings
├── render.yaml                 # Render Blueprint deployment config
├── environment.yml             # Conda environment definition
└── requirements.txt            # Python dependencies
```

## Installation

We recommend using Conda to manage the environment:

```bash
conda env create -f environment.yml
conda activate ecosort
pip install -r requirements.txt
```

## Quick Start

### 1. Training
Train the model using the 14-class configuration with letterbox padding enabled:
```bash
python experiments/train_baseline.py \
  --config configs/ecosort_v5_14class_effb3_letterbox224.yaml \
  --data-root data/raw
```

### 2. Evaluation
Evaluate the trained model against the validation/test set:
```bash
python experiments/evaluate.py \
  --checkpoint checkpoints/best_model.pth \
  --data-root data/raw \
  --model-type efficientnet
```

### 3. Serving (Backend API)
Launch the Flask inference server. The server exposes REST endpoints for both local inference and VLM fallback execution.
```bash
python backend/app.py \
  --model-path checkpoints/best_model.pth \
  --host 0.0.0.0 \
  --port 5000
```

## Data & Checkpoints

Following version-control best practices for machine learning projects:

* **Dataset**: the full raw dataset is not committed. Use `data/sample/` as the expected directory template and place custom images under `data/raw/<class_name>/`.
* **Model weights**: `checkpoints/best_model.pth` is prepared through Git LFS for GitHub upload. If Git LFS is unavailable in your deployment target, use `ECOSORT_MODEL_URL` or `HF_MODEL_URL` to download the model at startup instead.

See `checkpoints/README.md` and `GITHUB_UPLOAD.md` for the exact upload and deployment workflow.
