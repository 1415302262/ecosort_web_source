#!/usr/bin/env python3
"""Evaluate the EcoSort local-to-Gemini cascade from actual cached calls only.

No cache means no cascade claim: the script writes a machine-readable
``not_evaluated`` status. Partial caches are labeled ``partial`` and are never
presented as the completed 142-sample experiment. Failed/unmapped VLM outputs
retain the local prediction and are counted separately.
"""

from __future__ import annotations

import argparse
import json
import os
from collections import Counter
from pathlib import Path
from typing import Dict, List

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from run_ecosort_vlm_fallback import (
    PROMPT_VERSION,
    confidence_only_rejection,
    exact_leakage_test_ids,
    function_source_sha256,
    load_cache_rows,
    sha256_file,
)


def json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def latency_summary(values: pd.Series) -> Dict[str, float] | None:
    if values.empty:
        return None
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(values.quantile(0.50)),
        "p95_ms": float(values.quantile(0.95)),
    }


def aggregate_provider_usage(rows: List[Dict]) -> Dict | None:
    if not rows:
        return None
    usages = [row.get("provider_usage") for row in rows]
    valid = [usage for usage in usages if isinstance(usage, dict)]
    numeric_totals: Dict[str, float | int] = {}
    for usage in valid:
        for key, value in usage.items():
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                numeric_totals[key] = numeric_totals.get(key, 0) + value
    return {
        "requests_with_usage_metadata": len(valid),
        "requests_without_usage_metadata": len(rows) - len(valid),
        "numeric_totals": numeric_totals,
    }


def classification_metrics(frame: pd.DataFrame, pred_column: str, class_names: List[str]) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(frame["true_label"], frame[pred_column])),
        "precision_macro": float(precision_score(
            frame["true_label"], frame[pred_column], labels=class_names,
            average="macro", zero_division=0,
        )),
        "recall_macro": float(recall_score(
            frame["true_label"], frame[pred_column], labels=class_names,
            average="macro", zero_division=0,
        )),
        "f1_macro": float(f1_score(
            frame["true_label"], frame[pred_column], labels=class_names,
            average="macro", zero_division=0,
        )),
    }


def empty_bad_cases(path: Path) -> None:
    pd.DataFrame(columns=[
        "bad_case_type", "sample_id", "true_label", "local_pred_label",
        "vlm_pred_label", "final_pred_label", "local_confidence",
        "vlm_success", "vlm_latency_ms", "vlm_failure_reason", "source_path",
    ]).to_csv(path, index=False)


def empty_routed_transitions(path: Path) -> None:
    pd.DataFrame(columns=[
        "sample_id", "true_label", "pred_label", "vlm_pred_label",
        "final_pred_label", "correct", "final_correct", "vlm_attempted",
        "vlm_success", "transition_type", "vlm_outcome_type",
        "vlm_failure_reason", "confidence", "vlm_confidence",
        "vlm_latency_ms", "cascade_latency_ms", "source_path",
    ]).to_csv(path, index=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", default="reports/ecosort_experiment_results/raw/local_predictions_test.csv")
    parser.add_argument("--overlap-csv", default="reports/ecosort_experiment_results/data_audit/cross_split_overlaps.csv")
    parser.add_argument("--mapping-yaml", default="configs/dataset_mapping.yaml")
    parser.add_argument("--backend-path", default="backend/app.py")
    parser.add_argument("--vlm-dir", default="reports/ecosort_experiment_results/vlm")
    parser.add_argument("--provider", choices=("gemini", "seed"), default="seed")
    args = parser.parse_args()

    vlm_dir = Path(args.vlm_dir).resolve()
    vlm_dir.mkdir(parents=True, exist_ok=True)
    cache_path = vlm_dir / "vlm_response_cache.jsonl"
    ledger_path = vlm_dir / "vlm_attempt_ledger.jsonl"
    metrics_path = vlm_dir / "cascade_metrics.json"
    predictions_path = vlm_dir / "cascade_predictions.csv"
    routed_transitions_path = vlm_dir / "cascade_routed_transitions.csv"
    bad_cases_path = vlm_dir / "cascade_bad_cases.csv"

    predictions_full = pd.read_csv(args.predictions_csv)
    leaked_ids = exact_leakage_test_ids(Path(args.overlap_csv))
    predictions = predictions_full[~predictions_full["sample_id"].isin(leaked_ids)].copy()
    class_names = [
        column.removeprefix("prob_")
        for column in predictions.columns
        if column.startswith("prob_")
    ]
    routed_mask = confidence_only_rejection(predictions)
    routed_ids = set(predictions.loc[routed_mask, "sample_id"].astype(str))
    if args.provider == "seed":
        from run_ecosort_seed_vlm_fallback import (
            DEFAULT_MODEL as provider_default_model,
            PROMPT_VERSION as provider_prompt_version,
        )

        model = os.getenv("ECOSORT_ARK_MODEL", provider_default_model).strip()
        prompt_version = provider_prompt_version
        prompt_hash = function_source_sha256(
            Path(__file__).with_name("run_ecosort_seed_vlm_fallback.py"),
            "build_prompt",
        )
        provider_name = "volcano_ark"
    else:
        model = os.getenv("ECOSORT_GEMINI_MODEL", "gemini-2.5-flash").strip()
        prompt_version = PROMPT_VERSION
        prompt_hash = function_source_sha256(
            Path(args.backend_path).resolve(), "_call_vlm_primary"
        )
        provider_name = None
    mapping_hash = sha256_file(Path(args.mapping_yaml).resolve())

    cache_rows = load_cache_rows(cache_path)
    ledger_rows = load_cache_rows(ledger_path)
    source_hashes = {
        str(row["sample_id"]): sha256_file(Path(row["source_path"]))
        for _, row in predictions.loc[routed_mask].iterrows()
    }

    def matches(row: Dict) -> bool:
        sample_id = str(row.get("sample_id", ""))
        return (
            sample_id in routed_ids
            and (provider_name is None or row.get("provider") == provider_name)
            and row.get("model") == model
            and row.get("prompt_version") == prompt_version
            and row.get("prompt_source_sha256") == prompt_hash
            and row.get("mapping_sha256") == mapping_hash
            and row.get("image_sha256") == source_hashes.get(sample_id)
        )

    matching_rows = [row for row in cache_rows if matches(row)]
    matching_reservations = [row for row in ledger_rows if matches(row)]
    result_attempt_ids = {
        str(row["attempt_id"])
        for row in matching_rows
        if str(row.get("attempt_id", "")).strip()
    }
    ambiguous_reservations = [
        row for row in matching_reservations
        if str(row.get("attempt_id", "")) not in result_attempt_ids
    ]
    current_attempt_ids = result_attempt_ids | {
        str(row["attempt_id"])
        for row in matching_reservations
        if str(row.get("attempt_id", "")).strip()
    }
    legacy_matching_results = [
        row for row in matching_rows if not str(row.get("attempt_id", "")).strip()
    ]
    reserved_attempt_count = len(current_attempt_ids) + len(legacy_matching_results)
    if not matching_rows:
        predictions_path.unlink(missing_ok=True)
        empty_routed_transitions(routed_transitions_path)
        payload = {
            "provider": args.provider,
            "evaluation_status": (
                "partial_ambiguous" if ambiguous_reservations else "not_evaluated"
            ),
            "blocked_reason": (
                "Matching reservations exist without durable responses."
                if ambiguous_reservations else
                "No matching actual VLM cache entries were found."
            ),
            "analysis_set": "test excluding confirmed cross-split exact duplicates",
            "test_sample_count": int(len(predictions)),
            "planned_routed_sample_count": int(routed_mask.sum()),
            "planned_vlm_call_rate": float(routed_mask.mean()),
            "actual_vlm_attempt_count": None if ambiguous_reservations else 0,
            "actual_vlm_call_rate": None if ambiguous_reservations else 0.0,
            "actual_vlm_attempt_count_lower_bound": 0,
            "actual_vlm_attempt_count_upper_bound": int(reserved_attempt_count),
            "actual_vlm_call_rate_lower_bound": 0.0,
            "actual_vlm_call_rate_upper_bound": float(
                reserved_attempt_count / len(predictions)
            ),
            "reserved_attempt_budget_count": int(reserved_attempt_count),
            "ambiguous_reserved_attempt_count": int(len(ambiguous_reservations)),
            "recorded_vlm_response_count": 0,
            "successful_vlm_response_count": 0,
            "failed_or_unmapped_vlm_response_count": 0,
            "local_metrics": classification_metrics(predictions, "pred_label", class_names),
            "cascade_metrics": None,
            "partial_provisional_metrics": None,
            "cascade_latency": None,
            "partial_provisional_latency": None,
            "vlm_corrected_local_errors": None,
            "routed_local_error_count": int(
                ((routed_mask) & (~predictions["correct"].astype(bool))).sum()
            ),
            "vlm_uncorrected_local_errors": None,
            "vlm_introduced_errors": None,
            "net_corrected": None,
            "transition_counts": None,
            "provider_usage": None,
            "routed_transitions_csv": str(routed_transitions_path),
            "cache_path": str(cache_path),
            "attempt_ledger_path": str(ledger_path),
        }
        json_dump(metrics_path, payload)
        empty_bad_cases(bad_cases_path)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    latest_by_sample: Dict[str, Dict] = {}
    for row in matching_rows:
        latest_by_sample[str(row["sample_id"])] = row

    predictions["routed_by_confidence_only_085"] = routed_mask.astype(int)
    predictions["vlm_attempted"] = predictions["sample_id"].isin(latest_by_sample).astype(int)
    predictions["vlm_success"] = 0
    predictions["vlm_pred_label"] = ""
    predictions["vlm_confidence"] = 0.0
    predictions["vlm_latency_ms"] = 0.0
    predictions["final_pred_label"] = predictions["pred_label"]
    predictions["vlm_failure_reason"] = ""

    for index, sample in predictions[predictions["vlm_attempted"] == 1].iterrows():
        cached = latest_by_sample[str(sample["sample_id"])]
        success = bool(cached.get("success")) and bool(cached.get("mapped_target14"))
        predictions.at[index, "vlm_success"] = int(success)
        predictions.at[index, "vlm_pred_label"] = str(cached.get("mapped_target14", ""))
        response = cached.get("response") if isinstance(cached.get("response"), dict) else {}
        predictions.at[index, "vlm_confidence"] = float(response.get("confidence", 0.0))
        predictions.at[index, "vlm_latency_ms"] = float(cached.get("latency_ms", 0.0))
        if success:
            predictions.at[index, "final_pred_label"] = str(cached["mapped_target14"])
        else:
            predictions.at[index, "vlm_failure_reason"] = str(cached.get("map_method", "failed_or_unmapped"))

    predictions["final_correct"] = (
        predictions["final_pred_label"] == predictions["true_label"]
    ).astype(int)
    predictions["cascade_latency_ms"] = (
        predictions["end_to_end_latency_ms"] + predictions["vlm_latency_ms"]
    )
    routed = predictions["routed_by_confidence_only_085"].astype(bool)
    local_correct = predictions["correct"].astype(bool)
    final_correct = predictions["final_correct"].astype(bool)
    predictions["transition_type"] = ""
    predictions.loc[routed & ~local_correct & final_correct, "transition_type"] = (
        "local_wrong_to_vlm_correct"
    )
    predictions.loc[routed & ~local_correct & ~final_correct, "transition_type"] = (
        "local_wrong_to_vlm_still_wrong"
    )
    predictions.loc[routed & local_correct & final_correct, "transition_type"] = (
        "local_correct_to_vlm_correct"
    )
    predictions.loc[routed & local_correct & ~final_correct, "transition_type"] = (
        "local_correct_to_vlm_wrong"
    )
    predictions["vlm_outcome_type"] = ""
    attempted = predictions["vlm_attempted"].astype(bool)
    successful = predictions["vlm_success"].astype(bool)
    predictions.loc[routed & ~attempted, "vlm_outcome_type"] = "not_attempted"
    predictions.loc[routed & attempted & ~successful, "vlm_outcome_type"] = (
        "api_failure_local_retained"
    )
    predictions.loc[routed & successful, "vlm_outcome_type"] = predictions.loc[
        routed & successful, "transition_type"
    ]
    predictions.to_csv(predictions_path, index=False)
    routed_columns = [
        "sample_id", "true_label", "pred_label", "vlm_pred_label",
        "final_pred_label", "correct", "final_correct", "vlm_attempted",
        "vlm_success", "transition_type", "vlm_outcome_type",
        "vlm_failure_reason", "confidence", "margin", "normalized_entropy",
        "vlm_confidence", "end_to_end_latency_ms", "vlm_latency_ms",
        "cascade_latency_ms", "source_path",
    ]
    predictions.loc[routed, routed_columns].to_csv(routed_transitions_path, index=False)

    attempted_unique = int(predictions["vlm_attempted"].sum())
    success_unique = int(predictions["vlm_success"].sum())
    completed = (
        attempted_unique == len(routed_ids)
        and not ambiguous_reservations
    )
    corrected = (~local_correct) & final_correct
    introduced = local_correct & (~final_correct)
    routed_local_errors = routed & (~local_correct)
    uncorrected = routed_local_errors & (~final_correct)
    successful_routed = predictions["vlm_success"].astype(bool)
    provisional_metrics = classification_metrics(
        predictions, "final_pred_label", class_names
    )
    provisional_latency = latency_summary(predictions["cascade_latency_ms"])
    provider_usage = aggregate_provider_usage(matching_rows)
    provider_model_versions = Counter(
        str(row.get("provider_model_version"))
        for row in matching_rows
        if row.get("provider_model_version")
    )
    provider_sdk_versions = Counter(
        str(row.get("provider_sdk_version"))
        for row in matching_rows
        if row.get("provider_sdk_version")
    )
    successful_response_count = sum(
        bool(row.get("success")) and bool(row.get("mapped_target14"))
        for row in matching_rows
    )
    transition_counts = {
        str(key): int(value)
        for key, value in predictions.loc[routed, "transition_type"].value_counts().items()
    }

    payload = {
        "provider": args.provider,
        "evaluation_status": "completed" if completed else "partial",
        "analysis_set": "test excluding confirmed cross-split exact duplicates",
        "test_sample_count": int(len(predictions)),
        "planned_routed_sample_count": int(len(routed_ids)),
        "planned_vlm_call_rate": float(len(routed_ids) / len(predictions)),
        # A durable response proves that one helper request returned. An
        # unresolved reservation conservatively consumes budget but is not
        # presented as a verified provider call.
        "actual_vlm_attempt_count": (
            None if ambiguous_reservations else int(len(matching_rows))
        ),
        "actual_unique_routed_sample_count": attempted_unique,
        "actual_vlm_call_rate": (
            None if ambiguous_reservations else
            float(len(matching_rows) / len(predictions))
        ),
        "actual_vlm_attempt_count_lower_bound": int(len(matching_rows)),
        "actual_vlm_attempt_count_upper_bound": int(reserved_attempt_count),
        "actual_vlm_call_rate_lower_bound": float(
            len(matching_rows) / len(predictions)
        ),
        "actual_vlm_call_rate_upper_bound": float(
            reserved_attempt_count / len(predictions)
        ),
        "recorded_vlm_response_count": int(len(matching_rows)),
        "successful_vlm_response_count": int(successful_response_count),
        "failed_or_unmapped_vlm_response_count": int(
            len(matching_rows) - successful_response_count
        ),
        "reserved_attempt_budget_count": int(reserved_attempt_count),
        "reserved_attempt_budget_rate": float(reserved_attempt_count / len(predictions)),
        "ambiguous_reserved_attempt_count": int(len(ambiguous_reservations)),
        "successful_unique_vlm_count": success_unique,
        "failed_or_unmapped_unique_vlm_count": attempted_unique - success_unique,
        "local_metrics": classification_metrics(predictions, "pred_label", class_names),
        "cascade_metrics": provisional_metrics if completed else None,
        "partial_provisional_metrics": None if completed else provisional_metrics,
        "cascade_latency": provisional_latency if completed else None,
        "partial_provisional_latency": None if completed else provisional_latency,
        "vlm_latency_on_attempts": latency_summary(
            predictions.loc[predictions["vlm_attempted"] == 1, "vlm_latency_ms"]
        ),
        "vlm_corrected_local_errors": int(corrected.sum()) if completed else None,
        "routed_local_error_count": int(routed_local_errors.sum()),
        "vlm_uncorrected_local_errors": int(uncorrected.sum()) if completed else None,
        "vlm_introduced_errors": int(introduced.sum()) if completed else None,
        "net_corrected": (
            int(corrected.sum() - introduced.sum()) if completed else None
        ),
        "transition_counts": transition_counts if completed else None,
        "partial_provisional_corrected_local_errors": (
            None if completed else int(corrected.sum())
        ),
        "partial_provisional_uncorrected_local_errors": (
            None if completed else int(uncorrected.sum())
        ),
        "partial_provisional_introduced_errors": (
            None if completed else int(introduced.sum())
        ),
        "partial_provisional_net_corrected": (
            None if completed else int(corrected.sum() - introduced.sum())
        ),
        "partial_provisional_transition_counts": (
            None if completed else transition_counts
        ),
        "successful_vlm_routed_accuracy": (
            float(predictions.loc[successful_routed, "final_correct"].mean())
            if success_unique else None
        ),
        "provider_usage": provider_usage,
        "provider_model_versions": dict(provider_model_versions),
        "provider_sdk_versions": dict(provider_sdk_versions),
        "provider_usage_note": (
            "Numeric usage_metadata fields are summed across recorded responses; "
            "no monetary price is inferred."
        ),
        "partial_result_warning": (
            None if completed else
            "Formal cascade metrics remain null. Provisional metrics retain local "
            "predictions for routed samples without a durable VLM response."
        ),
        "predictions_csv": str(predictions_path),
        "routed_transitions_csv": str(routed_transitions_path),
        "bad_cases_csv": str(bad_cases_path),
        "cache_path": str(cache_path),
        "attempt_ledger_path": str(ledger_path),
    }
    json_dump(metrics_path, payload)

    bad_case_rows = []
    if completed:
        categories = (
            ("local_wrong_to_vlm_correct", corrected),
            ("local_wrong_to_vlm_still_wrong", uncorrected),
            ("local_correct_to_vlm_wrong", introduced),
        )
        for category, mask in categories:
            subset = predictions[mask].sort_values("confidence").head(20)
            for _, row in subset.iterrows():
                bad_case_rows.append({
                    "bad_case_type": category,
                    "sample_id": row["sample_id"],
                    "true_label": row["true_label"],
                    "local_pred_label": row["pred_label"],
                    "vlm_pred_label": row["vlm_pred_label"],
                    "final_pred_label": row["final_pred_label"],
                    "local_confidence": float(row["confidence"]),
                    "vlm_success": int(row["vlm_success"]),
                    "vlm_latency_ms": float(row["vlm_latency_ms"]),
                    "vlm_failure_reason": row["vlm_failure_reason"],
                    "source_path": row["source_path"],
                })
    if bad_case_rows:
        pd.DataFrame(bad_case_rows).to_csv(bad_cases_path, index=False)
    else:
        empty_bad_cases(bad_cases_path)
    print(json.dumps(payload, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
