#!/usr/bin/env python3
"""Export a sanitized, recruiter-facing result bundle without secrets or host paths."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="reports/ecosort_experiment_results/vlm_seed2_lite_260428",
    )
    parser.add_argument("--output-dir", default="docs/portfolio/data")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((results_dir / "cascade_metrics.json").read_text())
    robust = json.loads(
        (results_dir / "cascade_robustness_metrics.json").read_text()
    )
    transitions = pd.read_csv(results_dir / "cascade_routed_transitions.csv")
    per_class = pd.read_csv(results_dir / "cascade_per_class_delta.csv")
    bad_cases = pd.read_csv(results_dir / "cascade_bad_cases.csv")

    summary = {
        "release": "ecosort-portfolio-2026-08-11",
        "evaluation_status": metrics["evaluation_status"],
        "test_sample_count": metrics["test_sample_count"],
        "routing": {
            "strategy": "confidence_only",
            "threshold": 0.85,
            "local_coverage": 1 - metrics["actual_vlm_call_rate"],
            "actual_vlm_call_rate": metrics["actual_vlm_call_rate"],
            "actual_calls": metrics["actual_vlm_attempt_count"],
            "successful_calls": metrics["successful_vlm_response_count"],
            "failed_or_unmapped_calls": metrics[
                "failed_or_unmapped_vlm_response_count"
            ],
        },
        "model": {
            "local": "EfficientNet-B3",
            "vlm": "doubao-seed-2-0-lite-260428",
        },
        "local_metrics": metrics["local_metrics"],
        "cascade_metrics": metrics["cascade_metrics"],
        "paired_delta": robust["paired_delta"],
        "mcnemar_exact": robust["mcnemar_exact"],
        "error_accounting": {
            "routed_local_errors": metrics["routed_local_error_count"],
            "corrected": metrics["vlm_corrected_local_errors"],
            "uncorrected": metrics["vlm_uncorrected_local_errors"],
            "introduced": metrics["vlm_introduced_errors"],
            "net_corrected": metrics["net_corrected"],
            "transition_counts": metrics["transition_counts"],
        },
        "latency_ms": {
            "cascade_all_samples": metrics["cascade_latency"],
            "vlm_attempts": metrics["vlm_latency_on_attempts"],
        },
        "provider_usage": metrics["provider_usage"],
        "limitations": [
            "One request failed the provider pixel limit and retained the local prediction.",
            "One valid provider response returned the unmapped open-set label 'trash'.",
            "Confidence intervals quantify fixed-test-set sampling uncertainty only.",
            "Results cover one checkpoint, one provider version and one fixed threshold.",
            "Directory-level labels contain material versus disposal-semantics ambiguity.",
        ],
    }
    summary_path = output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    transition_columns = [
        "sample_id",
        "true_label",
        "pred_label",
        "vlm_pred_label",
        "final_pred_label",
        "correct",
        "final_correct",
        "vlm_success",
        "transition_type",
        "vlm_outcome_type",
        "confidence",
        "vlm_confidence",
        "vlm_latency_ms",
        "cascade_latency_ms",
    ]
    transitions[transition_columns].to_csv(
        output_dir / "routed_transitions.csv", index=False
    )
    per_class.to_csv(output_dir / "per_class_delta.csv", index=False)

    bad_case_columns = [
        column
        for column in [
            "bad_case_type",
            "sample_id",
            "true_label",
            "local_pred_label",
            "vlm_pred_label",
            "final_pred_label",
            "local_confidence",
            "vlm_success",
        ]
        if column in bad_cases.columns
    ]
    bad_cases[bad_case_columns].to_csv(output_dir / "bad_cases.csv", index=False)

    public_files = [
        summary_path,
        output_dir / "routed_transitions.csv",
        output_dir / "per_class_delta.csv",
        output_dir / "bad_cases.csv",
    ]
    checksum_lines = [
        f"{sha256_file(path)}  {path.name}" for path in sorted(public_files)
    ]
    (output_dir / "SHA256SUMS.txt").write_text(
        "\n".join(checksum_lines) + "\n", encoding="utf-8"
    )
    print(json.dumps({"output_dir": str(output_dir), "files": len(public_files)}, indent=2))


if __name__ == "__main__":
    main()
