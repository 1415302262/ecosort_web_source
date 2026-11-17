#!/usr/bin/env python3
"""Collect reproducible EcoSort local-model predictions and single-image latency.

This is an offline experiment harness. It deliberately reuses the current
EfficientNet model factory, checkpoint format, and letterbox preprocessing;
it does not alter the product inference path.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import platform
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from PIL import Image

from src.data.dataset import get_data_transforms
from src.models.efficientnet_classifier import create_efficientnet_model
from src.models.resnet_classifier import create_resnet_model


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def percentile_summary(values: List[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean_ms": float(array.mean()),
        "p50_ms": float(np.percentile(array, 50)),
        "p95_ms": float(np.percentile(array, 95)),
        "min_ms": float(array.min()),
        "max_ms": float(array.max()),
    }


def load_current_model(checkpoint_path: Path, device: torch.device):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if not isinstance(checkpoint, dict):
        raise TypeError("Expected a dictionary checkpoint")

    config = checkpoint.get("config", {})
    model_config = config.get("model", {})
    data_config = config.get("data", {})
    state_dict = checkpoint.get("model_state_dict", checkpoint)

    class_names = data_config.get("class_names")
    if not isinstance(class_names, list) or not class_names:
        raise ValueError("Checkpoint does not contain data.class_names")

    model_type = model_config.get("type", "efficientnet")
    backbone = model_config.get("backbone", "efficientnet-b3")
    dropout = float(model_config.get("dropout", 0.3))
    num_classes = len(class_names)

    if model_type == "efficientnet":
        model = create_efficientnet_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
        )
    elif model_type == "resnet":
        model = create_resnet_model(
            backbone=backbone,
            num_classes=num_classes,
            pretrained=False,
            dropout=dropout,
            use_attention=bool(model_config.get("use_attention", False)),
        )
    else:
        raise ValueError(f"Unsupported checkpoint model type: {model_type}")

    model.load_state_dict(state_dict, strict=True)
    model = model.to(device).eval()
    image_size = int(data_config.get("img_size", 224))
    transform = get_data_transforms("test", image_size)
    effective_model_spec = {
        "model_type": model_type,
        "backbone": backbone,
        "dropout": dropout,
        "num_classes": num_classes,
        "image_size": image_size,
        "preprocess": "ResizeLongestSideWithPadding/letterbox + ImageNet normalization",
        "model_type_source": (
            "checkpoint.config.model.type" if "type" in model_config else "script_default"
        ),
        "backbone_source": (
            "checkpoint.config.model.backbone"
            if "backbone" in model_config else "script_default"
        ),
    }
    return (
        model,
        transform,
        [str(name) for name in class_names],
        config,
        effective_model_spec,
    )


def enumerate_split(data_root: Path, split: str, class_names: List[str]):
    samples = []
    for label_index, class_name in enumerate(class_names):
        class_dir = data_root / split / class_name
        if not class_dir.is_dir():
            raise FileNotFoundError(f"Missing class directory: {class_dir}")
        paths = sorted(
            path for path in class_dir.iterdir()
            if path.suffix.lower() in IMAGE_SUFFIXES and path.is_file()
        )
        for path in paths:
            samples.append((path, label_index, class_name))
    if not samples:
        raise ValueError(f"No samples found for split={split}")
    return samples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint",
        default="checkpoints/champion_ecosort_target_20260301/best_model.pth",
    )
    parser.add_argument("--data-root", default="data/proc/ecosort_v6_15class")
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output-dir", default="reports/ecosort_experiment_results/raw")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--torch-threads", type=int, default=1)
    parser.add_argument("--disable-autocast", action="store_true")
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.torch_threads))
    checkpoint_path = Path(args.checkpoint).resolve()
    data_root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_device = torch.device(args.device)
    if requested_device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    device = requested_device
    use_autocast = device.type == "cuda" and not args.disable_autocast

    model, transform, class_names, config, effective_model_spec = load_current_model(
        checkpoint_path, device
    )
    samples = enumerate_split(data_root, args.split, class_names)

    with Image.open(samples[0][0]) as first_image:
        warmup_tensor = transform(first_image.convert("RGB")).unsqueeze(0).to(device)
    with torch.inference_mode():
        for _ in range(max(0, args.warmup)):
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    model(warmup_tensor)
            else:
                model(warmup_tensor)
    sync_if_cuda(device)

    rows = []
    forward_latencies = []
    end_to_end_latencies = []
    with torch.inference_mode():
        for index, (image_path, true_index, true_label) in enumerate(samples):
            sync_if_cuda(device)
            end_to_end_start = time.perf_counter()
            with Image.open(image_path) as image:
                tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)

            sync_if_cuda(device)
            forward_start = time.perf_counter()
            if use_autocast:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    logits = model(tensor)
            else:
                logits = model(tensor)
            probabilities = torch.softmax(logits.float(), dim=1)[0]
            sync_if_cuda(device)
            forward_end = time.perf_counter()

            probability_array = probabilities.detach().cpu().numpy().astype(np.float64)
            order = np.argsort(-probability_array)
            pred_index = int(order[0])
            top2_index = int(order[1])
            confidence = float(probability_array[pred_index])
            top2_confidence = float(probability_array[top2_index])
            margin = confidence - top2_confidence
            normalized_entropy = float(
                -(probability_array * np.log(probability_array + 1e-12)).sum()
                / np.log(len(class_names))
            )
            end_to_end_end = time.perf_counter()
            forward_ms = (forward_end - forward_start) * 1000.0
            end_to_end_ms = (end_to_end_end - end_to_end_start) * 1000.0
            forward_latencies.append(forward_ms)
            end_to_end_latencies.append(end_to_end_ms)

            row = {
                "sample_id": str(image_path.relative_to(data_root)),
                "source_path": str(image_path.resolve()),
                "split": args.split,
                "true_index": true_index,
                "true_label": true_label,
                "pred_index": pred_index,
                "pred_label": class_names[pred_index],
                "top2_index": top2_index,
                "top2_label": class_names[top2_index],
                "confidence": confidence,
                "top2_confidence": top2_confidence,
                "margin": margin,
                "normalized_entropy": normalized_entropy,
                "correct": int(pred_index == true_index),
                "forward_latency_ms": forward_ms,
                "end_to_end_latency_ms": end_to_end_ms,
            }
            for class_index, class_name in enumerate(class_names):
                row[f"prob_{class_name}"] = float(probability_array[class_index])
            rows.append(row)

            if (index + 1) % 100 == 0 or index + 1 == len(samples):
                print(f"[{args.split}] {index + 1}/{len(samples)}")

    csv_path = output_dir / f"local_predictions_{args.split}.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    device_name = (
        torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()
    )
    metadata = {
        "created_at_unix": time.time(),
        "split": args.split,
        "sample_count": len(rows),
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "data_root": str(data_root),
        "class_names": class_names,
        "checkpoint_config": config,
        "checkpoint_config_sha256": hashlib.sha256(
            json.dumps(config, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest(),
        "effective_model_spec": effective_model_spec,
        "device": str(device),
        "device_name": device_name,
        "torch_version": torch.__version__,
        "autocast_fp16": use_autocast,
        "warmup_iterations": args.warmup,
        "latency_definition": {
            "forward_latency_ms": "H2D-complete through logits, softmax, and GPU synchronization",
            "end_to_end_latency_ms": "PIL decode, letterbox preprocessing, H2D, forward, softmax, and postprocessing",
        },
        "forward_latency": percentile_summary(forward_latencies),
        "end_to_end_latency": percentile_summary(end_to_end_latencies),
        "prediction_csv": str(csv_path),
    }
    metadata_path = output_dir / f"local_inference_{args.split}_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
