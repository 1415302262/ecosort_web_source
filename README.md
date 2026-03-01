# ♻️ EcoSort: Edge-Cloud Cascade Waste Classification System

[![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://students-cs-ecosort-14class-demo.hf.space)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

**Live Interactive Demo:** [Experience EcoSort on Hugging Face Spaces](https://students-cs-ecosort-14class-demo.hf.space)

## 📌 Project Overview
EcoSort is an end-to-end intelligent waste classification system designed to address the challenges of real-world, high-entropy waste sorting. By leveraging an **Edge-Cloud Cascade Architecture**, the system achieves ultra-low latency for common waste items locally while maintaining high accuracy for ambiguous, hard-to-distinguish materials (e.g., e-waste, complex miscellaneous items) via a cloud-based Vision Large Language Model (VLM) fallback.

## 🏗️ System Architecture
<img width="2816" height="1536" alt="_20260227214111_32183_14" src="https://github.com/user-attachments/assets/36069f6d-8cb0-479c-b6b1-c7413c96332e" />

* **Edge (Local CNN - EfficientNet-B3):** Handles 80%+ of standard inferences locally, ensuring rapid response times and reducing cloud compute costs.
* **Cloud (VLM Fallback):** Triggers conditionally based on the CNN's confidence threshold, providing robust secondary verification for out-of-distribution or highly ambiguous samples.
* **Robust Vision Pipeline:** Features custom **Letterbox (SquarePad)** preprocessing to eliminate geometric distortion for non-square image inputs during real-world mobile inference.

## 📊 Performance & Evaluation
Rigorous evaluation was conducted to ensure real-world reliability across a customized 14-class dataset.

### Model & Data Ablation Study
We compared multiple backbone architectures and data preprocessing strategies. The EfficientNet-B3 model paired with Letterbox padding demonstrated the optimal balance between inference speed and accuracy. 

*(Insert your model/data comparison bar chart or line graph here)*
`![Model Comparison](docs/images/model_comparison.png)`

### Confusion Matrix
The confusion matrix below highlights the system's ability to distinguish between historically difficult classes, such as hazardous batteries versus standard metal waste.

*(Insert your confusion matrix image here)*
`![Confusion Matrix](docs/images/confusion_matrix.png)`

### Per-Class Metrics
| Class | Precision | Recall | F1-Score |
|-------|-----------|--------|----------|
| *Recyclable - Cardboard* | 0.94 | 0.95 | 0.94 |
| *Hazardous - Battery* | 0.89 | 0.91 | 0.90 |
| *Residual - E-Waste* | 0.85 | 0.82 | 0.83 |
| ... | ... | ... | ... |

*(Note: Replace with your actual metrics or a screenshot of your classification report)*
`![Per Class Metrics](docs/images/class_metrics.png)`

## 📓 Research & Development Journal
Building this system involved significant experimentation with data pipelines, model quantization, and threshold tuning. 

I have documented the entire engineering and research process, including technical roadblocks, literature reviews, and iterative optimizations in my development notes:
👉 **[Read the EcoSort R&D Journal](link-to-your-blog-or-pdf)**

## 📂 Repository Structure
```text
ecosort-web/
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
├── checkpoints/                # Model weights directory
├── ecosort_mapping.yaml        # Runtime confidence thresholds and fallback settings
└── environment.yml             # Conda environment definition
```

## 🚀 Reproducibility & Local Setup
While the production application is hosted on Hugging Face, the core training and evaluation pipelines can be fully reproduced locally.

### 1. Environment Setup

```bash
conda env create -f environment.yml
conda activate ecosort
pip install -r requirements.txt
```

### 2. Training
Train the baseline EfficientNet model using the 14-class configuration with letterbox padding:

```bash
python experiments/train_baseline.py \
  --config configs/ecosort_v5_14class_effb3_letterbox224.yaml \
  --data-root data/raw
```
### 3. Evaluation

```bash
python experiments/evaluate.py \
  --checkpoint checkpoints/ecosort_v5_14class_effb3_20260227/best_model.pth \
  --data-root data/raw \
  --model-type efficientnet

```
Note on Data & Weights: Due to size constraints, the full 1.5GB raw dataset and the .pth model weights are excluded from version control. Please refer to data/sample/ for dataset hierarchy constraints if utilizing custom data.

