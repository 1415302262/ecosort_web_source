#!/usr/bin/env python3
"""Legacy Gemini experiment harness with a hard call cap and cache.

The active EcoSort cascade uses the Seed runner. This retained harness requires
an explicit compatible backend helper and is covered by no-network regression
tests. External requests require ``--execute`` and ``ECOSORT_GEMINI_API_KEY``. Every
attempt is durably reserved in a locked JSONL ledger before the request; the
result cache is appended separately. An interrupted run therefore consumes a
budget slot conservatively instead of repeating a possibly paid call.
"""

from __future__ import annotations

import argparse
import ast
from src.inference import file_lock as fcntl
import hashlib
import json
import os
import signal
import threading
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Dict, Tuple

import pandas as pd
import yaml
from PIL import Image


HARD_MAX_CALLS = 142
ROUTING_CONFIDENCE_THRESHOLD = 0.85
PROMPT_VERSION = "root_backend_call_vlm_primary_42class_2026-03"
DEFAULT_CALL_TIMEOUT_SECONDS = 120
VLM_PROMPT_LABELS = frozenset({
    "cardboard", "glass", "metal", "paper", "plastic", "clothes", "shoes",
    "bag", "e_waste", "plastic_toy", "battery", "bulb", "chemical", "paint",
    "medicine", "thermometer", "cosmetic", "fruit_peel", "vegetable", "leftover",
    "tea", "egg_shell", "bone", "flower", "cake", "cigarette", "tissue",
    "chopsticks", "toothpick", "cup", "mask", "ceramic", "brick", "diaper",
    "wet_wipe", "plastic_bag", "pen", "comb", "towel", "lighter", "foam", "trash",
})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def exact_leakage_test_ids(overlap_csv: Path) -> set:
    if not overlap_csv.exists():
        return set()
    overlaps = pd.read_csv(overlap_csv)
    mask = (
        (overlaps["overlap_type"] == "exact_sha256")
        & overlaps["group_splits"].astype(str).str.contains("test")
        & overlaps["group_splits"].astype(str).str.contains("train|val", regex=True)
        & (overlaps["split"] == "test")
    )
    return set(overlaps.loc[mask, "sample_id"].astype(str))


def confidence_only_rejection(frame: pd.DataFrame) -> pd.Series:
    """Fixed cascade routing selected on validation: confidence < 0.85."""
    return frame["confidence"] < ROUTING_CONFIDENCE_THRESHOLD


def load_v6_mapping(path: Path) -> Dict[str, str]:
    config = yaml.safe_load(path.read_text(encoding="utf-8"))
    raw_mapping = config["daily_life_mapping_v6_15class"]["raw_to_target"]
    return {str(key).strip().lower(): str(value) for key, value in raw_mapping.items()}


def map_vlm_result(response: Dict, mapping: Dict[str, str]) -> Tuple[str, str]:
    # Use VLM top-1 only. Falling through to top-2/top-3 after seeing whether
    # top-1 maps would selectively improve the measured cascade.
    candidate = str(response.get("class_name", "")).strip()
    normalized_candidate = candidate.lower()
    if normalized_candidate not in VLM_PROMPT_LABELS:
        return "", "out_of_42class_vocabulary"
    mapped = mapping.get(normalized_candidate)
    if mapped:
        return mapped, f"v6_raw_to_target_top1:{candidate}"
    return "", "unmapped_vlm_output"


def function_source_sha256(path: Path, function_name: str) -> str:
    """Hash the exact helper source so prompt/code edits invalidate cache."""
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source)
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == function_name:
            lines = source.splitlines(keepends=True)
            function_source = "".join(lines[node.lineno - 1:node.end_lineno])
            return hashlib.sha256(function_source.encode("utf-8")).hexdigest()
    raise ValueError(f"Function {function_name!r} not found in {path}")


def cache_key(
    sample_id: str,
    image_sha256: str,
    model: str,
    prompt_version: str,
    prompt_source_sha256: str,
    mapping_sha256: str,
) -> str:
    return "|".join((
        sample_id,
        image_sha256,
        model,
        prompt_version,
        prompt_source_sha256,
        mapping_sha256,
    ))


def load_cache_rows(path: Path) -> list[Dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_cache(path: Path) -> Dict[str, Dict]:
    cache = {}
    for row in load_cache_rows(path):
        key = cache_key(
            str(row["sample_id"]),
            str(row.get("image_sha256", "")),
            str(row.get("model", "")),
            str(row.get("prompt_version", "")),
            str(row.get("prompt_source_sha256", "")),
            str(row.get("mapping_sha256", "")),
        )
        cache[key] = row
    return cache


def append_cache(path: Path, row: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def load_api_key() -> Tuple[str, str | None]:
    """Load the API key without requiring it in a process environment value."""
    environment_key = os.getenv("ECOSORT_GEMINI_API_KEY", "").strip()
    if environment_key:
        return environment_key, "environment"

    key_file_value = os.getenv("ECOSORT_GEMINI_API_KEY_FILE", "").strip()
    if not key_file_value:
        return "", None
    key_path = Path(key_file_value)
    if key_path.is_symlink():
        raise ValueError("ECOSORT_GEMINI_API_KEY_FILE must not be a symbolic link")
    if not key_path.is_file():
        raise ValueError("ECOSORT_GEMINI_API_KEY_FILE is not a regular file")
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("ECOSORT_GEMINI_API_KEY_FILE is empty")
    return key, "protected_file"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions-csv", default="reports/ecosort_experiment_results/raw/local_predictions_test.csv")
    parser.add_argument("--overlap-csv", default="reports/ecosort_experiment_results/data_audit/cross_split_overlaps.csv")
    parser.add_argument("--mapping-yaml", default="configs/dataset_mapping.yaml")
    parser.add_argument(
        "--backend-path",
        required=True,
        help="Backend source containing the legacy _call_vlm_primary helper.",
    )
    parser.add_argument("--output-dir", default="reports/ecosort_experiment_results/vlm")
    parser.add_argument("--max-calls", type=int, default=HARD_MAX_CALLS)
    parser.add_argument("--timeout-seconds", type=int, default=DEFAULT_CALL_TIMEOUT_SECONDS)
    parser.add_argument(
        "--retry-failures",
        action="store_true",
        help="Retry explicit failures only while the same 142-attempt global budget remains.",
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.max_calls < 0 or args.max_calls > HARD_MAX_CALLS:
        raise ValueError(f"--max-calls must be between 0 and {HARD_MAX_CALLS}")
    if args.timeout_seconds <= 0 or args.timeout_seconds > 300:
        raise ValueError("--timeout-seconds must be between 1 and 300")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "vlm_response_cache.jsonl"
    ledger_path = output_dir / "vlm_attempt_ledger.jsonl"
    lock_path = output_dir / ".vlm_attempt.lock"
    plan_path = output_dir / "vlm_call_plan.csv"
    summary_path = output_dir / "vlm_run_summary.json"

    frame = pd.read_csv(args.predictions_csv)
    leaked_ids = exact_leakage_test_ids(Path(args.overlap_csv))
    frame = frame[~frame["sample_id"].isin(leaked_ids)].copy()
    frame = frame[confidence_only_rejection(frame)].copy()
    frame = frame.sort_values(["confidence", "sample_id"], ascending=[True, True])
    frame.to_csv(plan_path, index=False)

    mapping_path = Path(args.mapping_yaml).resolve()
    backend_path = Path(args.backend_path).resolve()
    mapping = load_v6_mapping(mapping_path)
    mapping_sha256 = sha256_file(mapping_path)
    prompt_source_sha256 = function_source_sha256(backend_path, "_call_vlm_primary")
    model_name = os.getenv("ECOSORT_GEMINI_MODEL", "gemini-2.5-flash").strip()
    planned_hashes = {
        str(row["sample_id"]): sha256_file(Path(row["source_path"]))
        for _, row in frame.iterrows()
    }
    def matches_current_experiment(row: Dict) -> bool:
        sample_id = str(row.get("sample_id", ""))
        return (
            row.get("model") == model_name
            and row.get("prompt_version") == PROMPT_VERSION
            and row.get("prompt_source_sha256") == prompt_source_sha256
            and row.get("mapping_sha256") == mapping_sha256
            and row.get("image_sha256") == planned_hashes.get(sample_id)
        )

    def build_attempt_state(cache_rows: list[Dict], ledger_rows: list[Dict]) -> Dict:
        """Separate permanent call budget from current-experiment cache reuse."""
        matching_results = [row for row in cache_rows if matches_current_experiment(row)]
        matching_reservations = [
            row for row in ledger_rows if matches_current_experiment(row)
        ]

        # The approved 142-attempt budget is permanent for this output directory.
        # Changing a model, prompt, mapping, or image can invalidate reuse, but it
        # never refunds a prior reservation or legacy result from the budget.
        all_attempt_ids = {
            str(row["attempt_id"])
            for row in cache_rows + ledger_rows
            if str(row.get("attempt_id", "")).strip()
        }
        unidentified_budget_rows = [
            row for row in cache_rows + ledger_rows
            if not str(row.get("attempt_id", "")).strip()
        ]
        global_attempt_count = len(all_attempt_ids) + len(unidentified_budget_rows)

        completed_attempt_ids = {
            str(row["attempt_id"])
            for row in matching_results
            if str(row.get("attempt_id", "")).strip()
        }
        unresolved_reservations = [
            row for row in matching_reservations
            if str(row.get("attempt_id", "")) not in completed_attempt_ids
        ]
        valid_cached_sample_ids = {
            str(row["sample_id"])
            for row in matching_results
            if bool(row.get("success"))
        }
        result_sample_ids = {
            str(row["sample_id"]) for row in matching_results
        }
        attempted_sample_ids = result_sample_ids | {
            str(row["sample_id"]) for row in matching_reservations
        }
        ambiguous_reserved_sample_ids = {
            str(row["sample_id"]) for row in unresolved_reservations
        }
        failed_or_unmapped_sample_ids = result_sample_ids - valid_cached_sample_ids
        return {
            "matching_results": matching_results,
            "matching_reservations": matching_reservations,
            "global_attempt_count": global_attempt_count,
            "valid_cached_sample_ids": valid_cached_sample_ids,
            "attempted_sample_ids": attempted_sample_ids,
            "ambiguous_reserved_sample_ids": ambiguous_reserved_sample_ids,
            "failed_or_unmapped_sample_ids": failed_or_unmapped_sample_ids,
        }

    cache_rows = load_cache_rows(cache_path)
    ledger_rows = load_cache_rows(ledger_path)
    state = build_attempt_state(cache_rows, ledger_rows)
    valid_cached_sample_ids = state["valid_cached_sample_ids"]
    attempted_sample_ids = state["attempted_sample_ids"]
    ambiguous_reserved_sample_ids = state["ambiguous_reserved_sample_ids"]
    excluded_from_pending = (
        valid_cached_sample_ids | ambiguous_reserved_sample_ids
        if args.retry_failures
        else attempted_sample_ids
    )
    pending = frame[~frame["sample_id"].isin(excluded_from_pending)].copy()
    prior_attempts = int(state["global_attempt_count"])
    remaining_global_calls = max(0, HARD_MAX_CALLS - prior_attempts)
    allowed_this_run = min(args.max_calls, remaining_global_calls, len(pending))

    summary = {
        "mode": "execute" if args.execute else "dry_run",
        "routing_strategy": "Confidence-only routing on standardized checkpoint outputs",
        "thresholds": {
            "confidence": ROUTING_CONFIDENCE_THRESHOLD,
        },
        "test_sample_count_after_exact_leakage_exclusion": int(
            len(pd.read_csv(args.predictions_csv)) - len(leaked_ids)
        ),
        "planned_vlm_samples": int(len(frame)),
        "hard_max_calls": HARD_MAX_CALLS,
        "invocation_max_calls": args.max_calls,
        "model": model_name,
        "prompt_version": PROMPT_VERSION,
        "prompt_source_sha256": prompt_source_sha256,
        "mapping_sha256": mapping_sha256,
        "cached_samples": int(len(valid_cached_sample_ids)),
        "prior_attempts_in_global_budget": int(prior_attempts),
        "prior_failed_or_unmapped_samples": int(
            len(state["failed_or_unmapped_sample_ids"])
        ),
        "ambiguous_reserved_samples": int(len(ambiguous_reserved_sample_ids)),
        "pending_samples": int(len(pending)),
        "remaining_global_calls_before_run": int(remaining_global_calls),
        "allowed_calls_this_run": int(allowed_this_run),
        "retry_failures": bool(args.retry_failures),
        "call_timeout_seconds": int(args.timeout_seconds),
        "executed_this_run": 0,
        "successful_this_run": 0,
        "failed_this_run": 0,
        "provider_sdk_version_detected": installed_package_version("google-genai"),
        "usage_note": (
            "Gemini usage_metadata is cached when returned; no price conversion is performed."
        ),
        "budget_scope_note": (
            "Every durable reservation in this output directory consumes the same "
            "142-attempt budget, even if model, prompt, mapping, or image hashes change."
        ),
        "call_plan_csv": str(plan_path),
        "cache_jsonl": str(cache_path),
        "attempt_ledger_jsonl": str(ledger_path),
        "lock_file": str(lock_path),
    }

    if not args.execute:
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    try:
        api_key, api_key_source = load_api_key()
    except Exception as exc:
        summary["blocked_reason"] = (
            "Gemini API key loading failed before any reservation: "
            f"{type(exc).__name__}: {exc}"
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(summary["blocked_reason"]) from exc
    summary["api_key_source"] = api_key_source
    if not api_key:
        summary["blocked_reason"] = (
            "Neither ECOSORT_GEMINI_API_KEY nor ECOSORT_GEMINI_API_KEY_FILE is set"
        )
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
        raise RuntimeError(summary["blocked_reason"])

    # Import the existing helper with the key temporarily hidden. This avoids
    # backend import-time ``models.list()`` network traffic outside the capped
    # generation-request budget, while still calling its exact VLM function.
    environment_api_key = os.environ.pop("ECOSORT_GEMINI_API_KEY", None)
    try:
        try:
            from google import genai
            from backend import app as backend_app
        except Exception as exc:
            summary["blocked_reason"] = (
                "VLM runtime import failed before any reservation: "
                f"{type(exc).__name__}: {exc}"
            )
            summary_path.write_text(
                json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
            )
            raise RuntimeError(summary["blocked_reason"]) from exc
    finally:
        if environment_api_key is not None:
            os.environ["ECOSORT_GEMINI_API_KEY"] = environment_api_key
    backend_app.GEMINI_API_KEY = api_key
    backend_app.GEMINI_MODEL_NAME = model_name
    try:
        backend_app.gemini_client = genai.Client(api_key=api_key)
    except Exception as exc:
        summary["blocked_reason"] = (
            "Gemini client initialization failed before any reservation: "
            f"{type(exc).__name__}: {exc}"
        )
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        raise RuntimeError(summary["blocked_reason"]) from exc
    provider_sdk_version = installed_package_version("google-genai")
    summary["provider_sdk_version_detected"] = provider_sdk_version

    def reserve_attempt(sample: pd.Series, image_hash: str) -> str | None:
        """Atomically reserve one permanent budget slot before the API call."""
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            locked_state = build_attempt_state(
                load_cache_rows(cache_path), load_cache_rows(ledger_path)
            )
            if int(locked_state["global_attempt_count"]) >= HARD_MAX_CALLS:
                return None

            sample_id = str(sample["sample_id"])
            if sample_id in locked_state["valid_cached_sample_ids"]:
                return None
            if sample_id in locked_state["ambiguous_reserved_sample_ids"]:
                return None
            if (
                not args.retry_failures
                and sample_id in locked_state["attempted_sample_ids"]
            ):
                return None

            attempt_id = uuid.uuid4().hex
            append_cache(ledger_path, {
                "record_type": "attempt_reservation",
                "attempt_id": attempt_id,
                "sample_id": sample_id,
                "source_path": str(sample["source_path"]),
                "image_sha256": image_hash,
                "model": model_name,
                "prompt_version": PROMPT_VERSION,
                "prompt_source_sha256": prompt_source_sha256,
                "mapping_sha256": mapping_sha256,
                "reserved_at": datetime.now(timezone.utc).isoformat(),
            })
            return attempt_id

    def _timeout_handler(signum, frame):
        raise TimeoutError(f"VLM call exceeded {args.timeout_seconds}s wall timeout")

    use_signal_timer = hasattr(signal, "setitimer")
    if use_signal_timer:
        previous_handler = signal.signal(signal.SIGALRM, _timeout_handler)

    try:
        for _, sample in pending.iterrows():
            if summary["executed_this_run"] >= args.max_calls:
                break
            source_path = Path(sample["source_path"])
            image_hash = sha256_file(source_path)
            decode_started = time.perf_counter()
            with Image.open(source_path) as image:
                request_image = image.convert("RGB").copy()
            image_decode_latency_ms = (time.perf_counter() - decode_started) * 1000.0

            attempt_id = reserve_attempt(sample, image_hash)
            if attempt_id is None:
                continue
            # A durable reservation is deliberately counted before the request.
            # If the process dies now, the slot remains consumed and the sample
            # is not retried automatically, preventing duplicate paid calls.
            summary["executed_this_run"] += 1

            request_started = time.perf_counter()
            if use_signal_timer:
                signal.setitimer(signal.ITIMER_REAL, args.timeout_seconds)
            else:
                # Windows CLI: a timed-out request is ambiguous. Stop this worker;
                # its durable reservation prevents a duplicate paid request.
                timer = threading.Timer(args.timeout_seconds, os._exit, args=(124,))
                timer.daemon = True
                timer.start()
            try:
                response = backend_app._call_vlm_primary(request_image)
            finally:
                if use_signal_timer:
                    signal.setitimer(signal.ITIMER_REAL, 0)
                else:
                    timer.cancel()
                    timer.join()
            request_latency_ms = (time.perf_counter() - request_started) * 1000.0
            mapped_target, map_method = map_vlm_result(response, mapping)
            success = not bool(response.get("vlm_raw_output")) and bool(mapped_target)
            row = {
                "record_type": "attempt_result",
                "attempt_id": attempt_id,
                "sample_id": sample["sample_id"],
                "source_path": str(source_path),
                "image_sha256": image_hash,
                "true_label": sample["true_label"],
                "local_pred_label": sample["pred_label"],
                "local_correct": int(sample["correct"]),
                "local_confidence": float(sample["confidence"]),
                "local_margin": float(sample["margin"]),
                "local_normalized_entropy": float(sample["normalized_entropy"]),
                "model": model_name,
                "prompt_version": PROMPT_VERSION,
                "prompt_source_sha256": prompt_source_sha256,
                "mapping_sha256": mapping_sha256,
                "called_at": datetime.now(timezone.utc).isoformat(),
                "latency_ms": request_latency_ms,
                "request_latency_ms": request_latency_ms,
                "image_decode_latency_ms": image_decode_latency_ms,
                "offline_vlm_stage_latency_ms": (
                    request_latency_ms + image_decode_latency_ms
                ),
                "success": success,
                "mapped_target14": mapped_target,
                "map_method": map_method,
                "provider_usage": response.get("provider_usage"),
                "provider_model_version": response.get("provider_model_version"),
                "provider_sdk_version": provider_sdk_version,
                "response": response,
            }
            with lock_path.open("a+", encoding="utf-8") as lock_handle:
                fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
                append_cache(cache_path, row)
            summary["successful_this_run"] += int(success)
            summary["failed_this_run"] += int(not success)
            print(
                f"{summary['executed_this_run']}/{args.max_calls} "
                f"{sample['sample_id']} success={success} "
                f"request_latency_ms={request_latency_ms:.1f}"
            )
            provider_error = str(response.get("vlm_raw_output", ""))
            if "429 RESOURCE_EXHAUSTED" in provider_error:
                summary["stopped_early_reason"] = "provider_quota_exhausted"
                summary["quota_error_sample_id"] = str(sample["sample_id"])
                print("Stopping after provider quota exhaustion; no further samples reserved.")
                break
    finally:
        if use_signal_timer:
            signal.setitimer(signal.ITIMER_REAL, 0)
            signal.signal(signal.SIGALRM, previous_handler)

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        final_state = build_attempt_state(
            load_cache_rows(cache_path), load_cache_rows(ledger_path)
        )
    summary["cached_samples_after_run"] = int(
        len(final_state["valid_cached_sample_ids"])
    )
    summary["global_attempts_after_run"] = int(final_state["global_attempt_count"])
    summary["ambiguous_reserved_samples_after_run"] = int(
        len(final_state["ambiguous_reserved_sample_ids"])
    )
    summary["remaining_global_calls_after_run"] = (
        HARD_MAX_CALLS - summary["global_attempts_after_run"]
    )
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
