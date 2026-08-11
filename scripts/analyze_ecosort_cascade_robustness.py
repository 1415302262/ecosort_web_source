#!/usr/bin/env python3
"""Paired statistical audit for the completed EcoSort cascade experiment.

This script never calls a model or provider.  It consumes the fixed per-sample
cascade predictions, performs a paired bootstrap, runs an exact McNemar test,
and writes machine-readable robustness and per-class delta artifacts.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from scipy.stats import binomtest


DEFAULT_SEED = 20260811
DEFAULT_BOOTSTRAPS = 10_000


def metrics_from_encoded(
    y_true: np.ndarray, y_pred: np.ndarray, class_count: int
) -> Tuple[float, float, float, float, np.ndarray, np.ndarray, np.ndarray]:
    confusion = np.bincount(
        y_true * class_count + y_pred,
        minlength=class_count * class_count,
    ).reshape(class_count, class_count)
    true_positive = np.diag(confusion).astype(float)
    predicted = confusion.sum(axis=0).astype(float)
    actual = confusion.sum(axis=1).astype(float)
    precision = np.divide(
        true_positive,
        predicted,
        out=np.zeros_like(true_positive),
        where=predicted > 0,
    )
    recall = np.divide(
        true_positive,
        actual,
        out=np.zeros_like(true_positive),
        where=actual > 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_positive),
        where=(precision + recall) > 0,
    )
    accuracy = float(true_positive.sum() / confusion.sum())
    return (
        accuracy,
        float(precision.mean()),
        float(recall.mean()),
        float(f1.mean()),
        precision,
        recall,
        f1,
    )


def percentile_interval(values: np.ndarray) -> Dict[str, float]:
    low, high = np.percentile(values, [2.5, 97.5])
    return {
        "estimate": float(values[0]),
        "ci95_low": float(low),
        "ci95_high": float(high),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-csv",
        default=(
            "reports/ecosort_experiment_results/vlm_seed2_lite_260428/"
            "cascade_predictions.csv"
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="reports/ecosort_experiment_results/vlm_seed2_lite_260428",
    )
    parser.add_argument("--bootstrap-replicates", type=int, default=DEFAULT_BOOTSTRAPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    if args.bootstrap_replicates < 1_000:
        raise ValueError("Use at least 1,000 paired bootstrap replicates")

    frame = pd.read_csv(args.predictions_csv)
    required = {"true_label", "pred_label", "final_pred_label", "vlm_attempted"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    class_names = sorted(frame["true_label"].astype(str).unique())
    class_to_index = {name: index for index, name in enumerate(class_names)}
    encode = np.vectorize(class_to_index.__getitem__)
    y_true = encode(frame["true_label"].astype(str).to_numpy()).astype(np.int16)
    y_local = encode(frame["pred_label"].astype(str).to_numpy()).astype(np.int16)
    y_cascade = encode(frame["final_pred_label"].astype(str).to_numpy()).astype(np.int16)
    class_count = len(class_names)

    local = metrics_from_encoded(y_true, y_local, class_count)
    cascade = metrics_from_encoded(y_true, y_cascade, class_count)
    point_delta = np.asarray(cascade[:4]) - np.asarray(local[:4])

    rng = np.random.default_rng(args.seed)
    bootstrap_rows = np.empty((args.bootstrap_replicates, 12), dtype=float)
    class_f1_deltas = np.empty((args.bootstrap_replicates, class_count), dtype=float)
    sample_count = len(frame)
    for bootstrap_index in range(args.bootstrap_replicates):
        sampled = rng.integers(0, sample_count, size=sample_count)
        local_boot = metrics_from_encoded(
            y_true[sampled], y_local[sampled], class_count
        )
        cascade_boot = metrics_from_encoded(
            y_true[sampled], y_cascade[sampled], class_count
        )
        local_values = np.asarray(local_boot[:4])
        cascade_values = np.asarray(cascade_boot[:4])
        bootstrap_rows[bootstrap_index] = np.concatenate(
            [local_values, cascade_values, cascade_values - local_values]
        )
        class_f1_deltas[bootstrap_index] = cascade_boot[6] - local_boot[6]

    bootstrap_columns = [
        "local_accuracy",
        "local_precision_macro",
        "local_recall_macro",
        "local_f1_macro",
        "cascade_accuracy",
        "cascade_precision_macro",
        "cascade_recall_macro",
        "cascade_f1_macro",
        "delta_accuracy",
        "delta_precision_macro",
        "delta_recall_macro",
        "delta_f1_macro",
    ]
    bootstrap = pd.DataFrame(bootstrap_rows, columns=bootstrap_columns)

    local_correct = y_local == y_true
    cascade_correct = y_cascade == y_true
    local_only_correct = int((local_correct & ~cascade_correct).sum())
    cascade_only_correct = int((~local_correct & cascade_correct).sum())
    discordant = local_only_correct + cascade_only_correct
    mcnemar = binomtest(
        min(local_only_correct, cascade_only_correct),
        n=discordant,
        p=0.5,
        alternative="two-sided",
    )

    delta_keys = ["accuracy", "precision_macro", "recall_macro", "f1_macro"]
    intervals: Dict[str, Dict[str, float]] = {}
    for index, key in enumerate(delta_keys):
        values = np.concatenate(([point_delta[index]], bootstrap_rows[:, 8 + index]))
        intervals[key] = percentile_interval(values)

    routed = frame["vlm_attempted"].astype(bool).to_numpy()
    routed_local_accuracy = float(local_correct[routed].mean())
    routed_cascade_accuracy = float(cascade_correct[routed].mean())

    per_class_rows = []
    for class_index, class_name in enumerate(class_names):
        class_mask = y_true == class_index
        corrected = int((class_mask & ~local_correct & cascade_correct).sum())
        introduced = int((class_mask & local_correct & ~cascade_correct).sum())
        low, high = np.percentile(class_f1_deltas[:, class_index], [2.5, 97.5])
        per_class_rows.append({
            "class_name": class_name,
            "support": int(class_mask.sum()),
            "local_precision": float(local[4][class_index]),
            "local_recall": float(local[5][class_index]),
            "local_f1": float(local[6][class_index]),
            "cascade_precision": float(cascade[4][class_index]),
            "cascade_recall": float(cascade[5][class_index]),
            "cascade_f1": float(cascade[6][class_index]),
            "delta_f1": float(cascade[6][class_index] - local[6][class_index]),
            "delta_f1_ci95_low": float(low),
            "delta_f1_ci95_high": float(high),
            "corrected_local_errors": corrected,
            "introduced_errors": introduced,
            "net_corrected": corrected - introduced,
        })
    per_class = pd.DataFrame(per_class_rows).sort_values(
        ["delta_f1", "class_name"], ascending=[False, True]
    )

    payload = {
        "analysis_status": "completed",
        "analysis_type": "paired bootstrap and exact McNemar test",
        "sample_count": sample_count,
        "class_count": class_count,
        "bootstrap_replicates": args.bootstrap_replicates,
        "random_seed": args.seed,
        "bootstrap_sampling": "paired nonparametric sample-level bootstrap",
        "point_metrics": {
            "local": dict(zip(delta_keys, map(float, local[:4]))),
            "cascade": dict(zip(delta_keys, map(float, cascade[:4]))),
        },
        "paired_delta": intervals,
        "mcnemar_exact": {
            "local_correct_cascade_wrong": local_only_correct,
            "local_wrong_cascade_correct": cascade_only_correct,
            "discordant_pair_count": discordant,
            "two_sided_p_value": float(mcnemar.pvalue),
        },
        "routed_subset": {
            "sample_count": int(routed.sum()),
            "local_accuracy": routed_local_accuracy,
            "cascade_accuracy": routed_cascade_accuracy,
            "accuracy_delta": routed_cascade_accuracy - routed_local_accuracy,
        },
        "interpretation_boundary": (
            "Confidence intervals quantify paired sampling uncertainty on this fixed "
            "test set; they do not cover dataset shift, label noise, provider drift, "
            "or threshold-selection uncertainty."
        ),
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = output_dir / "cascade_robustness_metrics.json"
    bootstrap_path = output_dir / "paired_bootstrap_deltas.csv"
    per_class_path = output_dir / "cascade_per_class_delta.csv"
    metrics_path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    bootstrap.to_csv(bootstrap_path, index=False)
    per_class.to_csv(per_class_path, index=False)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
