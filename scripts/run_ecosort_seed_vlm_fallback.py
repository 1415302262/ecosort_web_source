#!/usr/bin/env python3
"""Run the fixed EcoSort confidence route with Doubao Seed2.0 Lite.

This provider-specific runner preserves the existing Gemini pilot.  It writes
to a separate output directory, reserves every request durably before network
I/O, caches every response immediately, and never exceeds the fixed 142-sample
plan selected by ``confidence < 0.85``.
"""

from __future__ import annotations

import argparse
import base64
from src.inference import file_lock as fcntl
import io
import json
import math
import os
import re
import time
import uuid
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Dict, Tuple

import pandas as pd
import requests
from PIL import Image

from run_ecosort_vlm_fallback import (
    HARD_MAX_CALLS,
    ROUTING_CONFIDENCE_THRESHOLD,
    append_cache,
    cache_key,
    confidence_only_rejection,
    exact_leakage_test_ids,
    function_source_sha256,
    load_cache_rows,
    load_v6_mapping,
    map_vlm_result,
    sha256_file,
)


PROVIDER = "volcano_ark"
DEFAULT_MODEL = "doubao-seed-2-0-lite-260428"
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_OUTPUT_DIR = "reports/ecosort_experiment_results/vlm_seed2_lite_260428"
DEFAULT_KEY_FILE = str(Path.home() / ".config" / "ecosort" / "ark_api_key")
DEFAULT_CALL_TIMEOUT_SECONDS = 120
DEFAULT_MAX_OUTPUT_TOKENS = 256
ARK_MAX_TOTAL_IMAGE_PIXELS = 36_000_000
# Keep a small safety margin for format conversion and provider-side checks.
ARK_SAFE_TOTAL_IMAGE_PIXELS = 35_000_000
PROMPT_VERSION = "seed2_lite_42class_same_semantics_2026-08-10_v1"


def build_prompt() -> str:
    """Return the same 42-class task definition used by the Gemini helper."""
    return """
You are a top expert in waste classification standards.
Identify the main object in the image and strictly select the most accurate
classification from the following 42 labels:
cardboard, glass, metal, paper, plastic, clothes, shoes, bag, e_waste,
plastic_toy, battery, bulb, chemical, paint, medicine, thermometer, cosmetic,
fruit_peel, vegetable, leftover, tea, egg_shell, bone, flower, cake,
cigarette, tissue, chopsticks, toothpick, cup, mask, ceramic, brick, diaper,
wet_wipe, plastic_bag, pen, comb, towel, lighter, foam, trash.
Also provide its coarse category (recyclable, hazardous, kitchen, other).

Return the top three classifications and confidence values from 0 to 1.  The
first item must have the highest confidence.  Output only one valid JSON object
with exactly this structure and no Markdown fence or explanation:
{
  "class_name": "one of the 42 labels",
  "coarse_category": "recyclable/hazardous/kitchen/other",
  "confidence": 0.95,
  "top_predictions": [
    {"class_name": "first label", "confidence": 0.95},
    {"class_name": "second label", "confidence": 0.04},
    {"class_name": "third label", "confidence": 0.01}
  ]
}
""".strip()


def installed_package_version(package_name: str) -> str | None:
    try:
        return version(package_name)
    except PackageNotFoundError:
        return None


def load_api_key() -> Tuple[str, str]:
    environment_key = os.getenv("ECOSORT_ARK_API_KEY", "").strip()
    if environment_key:
        return environment_key, "environment"

    key_path = Path(
        os.getenv("ECOSORT_ARK_API_KEY_FILE", DEFAULT_KEY_FILE).strip()
    )
    if key_path.is_symlink():
        raise ValueError("ECOSORT_ARK_API_KEY_FILE must not be a symbolic link")
    if not key_path.is_file():
        raise ValueError("ECOSORT_ARK_API_KEY_FILE is not a regular file")
    key = key_path.read_text(encoding="utf-8").strip()
    if not key:
        raise ValueError("ECOSORT_ARK_API_KEY_FILE is empty")
    return key, "protected_file"


def resize_for_ark(
    image: Image.Image,
    max_total_pixels: int = ARK_SAFE_TOTAL_IMAGE_PIXELS,
) -> Image.Image:
    """Downscale oversized inputs while preserving aspect ratio.

    The completed experiment intentionally retains its one historical HTTP 400
    result.  This guard only prevents the same provider input failure in future
    runs and does not alter cached experiment outputs.
    """
    width, height = image.size
    if width * height <= max_total_pixels:
        return image
    scale = math.sqrt(max_total_pixels / float(width * height))
    resized_width = max(1, int(math.floor(width * scale)))
    resized_height = max(1, int(math.floor(height * scale)))
    while resized_width * resized_height > max_total_pixels:
        if resized_width >= resized_height:
            resized_width -= 1
        else:
            resized_height -= 1
    return image.resize(
        (resized_width, resized_height),
        resample=Image.Resampling.LANCZOS,
    )


def image_data_uri(
    image: Image.Image,
    max_total_pixels: int = ARK_SAFE_TOTAL_IMAGE_PIXELS,
) -> str:
    image = resize_for_ark(image, max_total_pixels=max_total_pixels)
    buffer = io.BytesIO()
    image.save(buffer, format="JPEG", quality=95, optimize=True)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def parse_json_object(text: str) -> Dict[str, Any]:
    cleaned = (text or "").strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    data = json.loads(cleaned)
    if not isinstance(data, dict):
        raise ValueError("VLM response is not a JSON object")
    return data


def normalize_seed_response(
    payload: Dict[str, Any], http_status: int
) -> Dict[str, Any]:
    provider_usage = payload.get("usage") if isinstance(payload, dict) else None
    provider_model_version = payload.get("model") if isinstance(payload, dict) else None
    provider_response_id = payload.get("id") if isinstance(payload, dict) else None
    if http_status >= 400:
        error = payload.get("error", payload)
        raise RuntimeError(
            f"Ark HTTP {http_status}: {json.dumps(error, ensure_ascii=False)}"
        )

    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices:
        raise ValueError("Ark response has no choices")
    message = choices[0].get("message", {})
    content = message.get("content", "")
    if isinstance(content, list):
        content = "".join(
            str(item.get("text", "")) if isinstance(item, dict) else str(item)
            for item in content
        )
    data = parse_json_object(str(content))

    top_predictions = []
    for prediction in data.get("top_predictions", []):
        if not isinstance(prediction, dict):
            continue
        label = str(prediction.get("class_name", "trash"))
        top_predictions.append({
            "class_name": label,
            "daily_label": label,
            "confidence": float(prediction.get("confidence", 0.0)),
            "coarse_category": str(data.get("coarse_category", "other")),
        })

    main_label = str(data.get("class_name", "trash"))
    return {
        "class_name": main_label,
        "raw_top1_class": main_label,
        "daily_label": main_label,
        "coarse_category": str(data.get("coarse_category", "other")),
        "confidence": float(data.get("confidence", 0.0)),
        "top_predictions": top_predictions,
        "vlm_fallback_used": True,
        "provider_usage": provider_usage,
        "provider_model_version": provider_model_version,
        "provider_response_id": provider_response_id,
        "provider_http_status": http_status,
        "uncertain": False,
    }


def call_seed(
    session: requests.Session,
    api_key: str,
    model: str,
    base_url: str,
    image: Image.Image,
    timeout_seconds: int,
    max_output_tokens: int,
) -> Dict[str, Any]:
    """Make exactly one Ark Chat request and preserve usage on parse failure."""
    payload = {
        "model": model,
        "messages": [{
            "role": "user",
            "content": [
                {"type": "text", "text": build_prompt()},
                {
                    "type": "image_url",
                    "image_url": {"url": image_data_uri(image)},
                },
            ],
        }],
        "stream": False,
        "temperature": 0,
        "max_tokens": max_output_tokens,
        "thinking": {"type": "disabled"},
        "service_tier": "default",
    }
    provider_payload: Dict[str, Any] | None = None
    http_status = 0
    try:
        response = session.post(
            f"{base_url.rstrip('/')}/chat/completions",
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=(10, timeout_seconds),
        )
        http_status = response.status_code
        try:
            provider_payload = response.json()
        except Exception as exc:
            raise ValueError(
                f"Ark returned non-JSON HTTP {http_status}: {response.text[:300]}"
            ) from exc
        return normalize_seed_response(provider_payload, http_status)
    except Exception as exc:
        return {
            "class_name": "trash",
            "raw_top1_class": "trash",
            "daily_label": "trash",
            "coarse_category": "other",
            "confidence": 0.0,
            "top_predictions": [],
            "vlm_fallback_used": True,
            "provider_usage": (
                provider_payload.get("usage")
                if isinstance(provider_payload, dict) else None
            ),
            "provider_model_version": (
                provider_payload.get("model")
                if isinstance(provider_payload, dict) else None
            ),
            "provider_response_id": (
                provider_payload.get("id")
                if isinstance(provider_payload, dict) else None
            ),
            "provider_http_status": http_status,
            "vlm_raw_output": f"{type(exc).__name__}: {exc}",
            "uncertain": True,
        }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--predictions-csv",
        default="reports/ecosort_experiment_results/raw/local_predictions_test.csv",
    )
    parser.add_argument(
        "--overlap-csv",
        default="reports/ecosort_experiment_results/data_audit/cross_split_overlaps.csv",
    )
    parser.add_argument("--mapping-yaml", default="configs/dataset_mapping.yaml")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--max-calls", type=int, default=HARD_MAX_CALLS)
    parser.add_argument(
        "--timeout-seconds", type=int, default=DEFAULT_CALL_TIMEOUT_SECONDS
    )
    parser.add_argument(
        "--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS
    )
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()

    if args.max_calls < 0 or args.max_calls > HARD_MAX_CALLS:
        raise ValueError(f"--max-calls must be between 0 and {HARD_MAX_CALLS}")
    if args.timeout_seconds <= 0 or args.timeout_seconds > 300:
        raise ValueError("--timeout-seconds must be between 1 and 300")
    if args.max_output_tokens < 64 or args.max_output_tokens > 512:
        raise ValueError("--max-output-tokens must be between 64 and 512")

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "vlm_response_cache.jsonl"
    ledger_path = output_dir / "vlm_attempt_ledger.jsonl"
    lock_path = output_dir / ".vlm_attempt.lock"
    plan_path = output_dir / "vlm_call_plan.csv"
    summary_path = output_dir / "vlm_run_summary.json"

    predictions_full = pd.read_csv(args.predictions_csv)
    leaked_ids = exact_leakage_test_ids(Path(args.overlap_csv))
    frame = predictions_full[~predictions_full["sample_id"].isin(leaked_ids)].copy()
    frame = frame[confidence_only_rejection(frame)].copy()
    frame = frame.sort_values(["confidence", "sample_id"], ascending=[True, True])
    frame.to_csv(plan_path, index=False)
    if len(frame) != HARD_MAX_CALLS:
        raise RuntimeError(
            f"Fixed route changed: expected {HARD_MAX_CALLS}, found {len(frame)}"
        )

    mapping_path = Path(args.mapping_yaml).resolve()
    runner_path = Path(__file__).resolve()
    mapping = load_v6_mapping(mapping_path)
    mapping_sha256 = sha256_file(mapping_path)
    prompt_source_sha256 = function_source_sha256(runner_path, "build_prompt")
    model_name = os.getenv("ECOSORT_ARK_MODEL", DEFAULT_MODEL).strip()
    base_url = os.getenv("ECOSORT_ARK_BASE_URL", DEFAULT_BASE_URL).strip()
    planned_hashes = {
        str(row["sample_id"]): sha256_file(Path(row["source_path"]))
        for _, row in frame.iterrows()
    }

    def matches_current_experiment(row: Dict[str, Any]) -> bool:
        sample_id = str(row.get("sample_id", ""))
        return (
            row.get("provider") == PROVIDER
            and row.get("model") == model_name
            and row.get("prompt_version") == PROMPT_VERSION
            and row.get("prompt_source_sha256") == prompt_source_sha256
            and row.get("mapping_sha256") == mapping_sha256
            and row.get("image_sha256") == planned_hashes.get(sample_id)
        )

    def build_attempt_state() -> Dict[str, Any]:
        cache_rows = load_cache_rows(cache_path)
        ledger_rows = load_cache_rows(ledger_path)
        matching_results = [row for row in cache_rows if matches_current_experiment(row)]
        matching_reservations = [
            row for row in ledger_rows if matches_current_experiment(row)
        ]
        all_rows = cache_rows + ledger_rows
        identified = {
            str(row["attempt_id"])
            for row in all_rows
            if str(row.get("attempt_id", "")).strip()
        }
        unidentified = [
            row for row in all_rows if not str(row.get("attempt_id", "")).strip()
        ]
        result_attempt_ids = {
            str(row["attempt_id"])
            for row in matching_results
            if str(row.get("attempt_id", "")).strip()
        }
        unresolved = [
            row for row in matching_reservations
            if str(row.get("attempt_id", "")) not in result_attempt_ids
        ]
        result_ids = {str(row["sample_id"]) for row in matching_results}
        reserved_ids = {str(row["sample_id"]) for row in matching_reservations}
        successful_ids = {
            str(row["sample_id"])
            for row in matching_results
            if bool(row.get("success"))
        }
        return {
            "global_attempt_count": len(identified) + len(unidentified),
            "attempted_sample_ids": result_ids | reserved_ids,
            "successful_sample_ids": successful_ids,
            "unresolved_sample_ids": {str(row["sample_id"]) for row in unresolved},
            "matching_results": matching_results,
        }

    state = build_attempt_state()
    pending = frame[
        ~frame["sample_id"].astype(str).isin(state["attempted_sample_ids"])
    ].copy()
    remaining_budget = max(0, HARD_MAX_CALLS - state["global_attempt_count"])
    allowed_this_run = min(args.max_calls, remaining_budget, len(pending))
    summary: Dict[str, Any] = {
        "mode": "execute" if args.execute else "dry_run",
        "provider": PROVIDER,
        "model": model_name,
        "routing_strategy": "Confidence-only",
        "confidence_threshold": ROUTING_CONFIDENCE_THRESHOLD,
        "test_sample_count_after_exact_leakage_exclusion": int(
            len(predictions_full) - len(leaked_ids)
        ),
        "planned_vlm_samples": int(len(frame)),
        "hard_max_calls": HARD_MAX_CALLS,
        "invocation_max_calls": args.max_calls,
        "prompt_version": PROMPT_VERSION,
        "prompt_source_sha256": prompt_source_sha256,
        "mapping_sha256": mapping_sha256,
        "prior_attempts_in_global_budget": int(state["global_attempt_count"]),
        "cached_successful_samples": int(len(state["successful_sample_ids"])),
        "ambiguous_reserved_samples": int(len(state["unresolved_sample_ids"])),
        "pending_samples": int(len(pending)),
        "remaining_global_calls_before_run": int(remaining_budget),
        "allowed_calls_this_run": int(allowed_this_run),
        "call_timeout_seconds": int(args.timeout_seconds),
        "max_output_tokens": int(args.max_output_tokens),
        "thinking": "disabled",
        "executed_this_run": 0,
        "successful_this_run": 0,
        "failed_this_run": 0,
        "provider_sdk": "requests",
        "provider_sdk_version": installed_package_version("requests"),
        "usage_note": "Provider usage is cached verbatim; no monetary cost is inferred.",
        "call_plan_csv": str(plan_path),
        "cache_jsonl": str(cache_path),
        "attempt_ledger_jsonl": str(ledger_path),
    }
    if not args.execute:
        summary_path.write_text(
            json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return

    api_key, key_source = load_api_key()
    summary["api_key_source"] = key_source
    session = requests.Session()
    session.trust_env = False

    def reserve_attempt(sample: pd.Series, image_hash: str) -> str | None:
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            locked = build_attempt_state()
            if locked["global_attempt_count"] >= HARD_MAX_CALLS:
                return None
            sample_id = str(sample["sample_id"])
            if sample_id in locked["attempted_sample_ids"]:
                return None
            attempt_id = uuid.uuid4().hex
            append_cache(ledger_path, {
                "record_type": "attempt_reservation",
                "attempt_id": attempt_id,
                "provider": PROVIDER,
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

    for _, sample in pending.head(allowed_this_run).iterrows():
        source_path = Path(sample["source_path"])
        image_hash = sha256_file(source_path)
        decode_started = time.perf_counter()
        with Image.open(source_path) as image:
            request_image = image.convert("RGB").copy()
        decode_latency_ms = (time.perf_counter() - decode_started) * 1000.0
        attempt_id = reserve_attempt(sample, image_hash)
        if attempt_id is None:
            continue
        summary["executed_this_run"] += 1
        request_started = time.perf_counter()
        response = call_seed(
            session=session,
            api_key=api_key,
            model=model_name,
            base_url=base_url,
            image=request_image,
            timeout_seconds=args.timeout_seconds,
            max_output_tokens=args.max_output_tokens,
        )
        request_latency_ms = (time.perf_counter() - request_started) * 1000.0
        mapped_target, map_method = map_vlm_result(response, mapping)
        success = not bool(response.get("vlm_raw_output")) and bool(mapped_target)
        result_row = {
            "record_type": "attempt_result",
            "attempt_id": attempt_id,
            "provider": PROVIDER,
            "sample_id": str(sample["sample_id"]),
            "source_path": str(source_path),
            "image_sha256": image_hash,
            "true_label": str(sample["true_label"]),
            "local_pred_label": str(sample["pred_label"]),
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
            "image_decode_latency_ms": decode_latency_ms,
            "offline_vlm_stage_latency_ms": request_latency_ms + decode_latency_ms,
            "success": success,
            "mapped_target14": mapped_target,
            "map_method": map_method,
            "provider_usage": response.get("provider_usage"),
            "provider_model_version": response.get("provider_model_version"),
            "provider_response_id": response.get("provider_response_id"),
            "provider_http_status": response.get("provider_http_status"),
            "provider_sdk_version": installed_package_version("requests"),
            "response": response,
        }
        with lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            append_cache(cache_path, result_row)
        summary["successful_this_run"] += int(success)
        summary["failed_this_run"] += int(not success)
        print(
            f"{summary['executed_this_run']}/{allowed_this_run} "
            f"{sample['sample_id']} success={success} "
            f"http={response.get('provider_http_status')} "
            f"request_latency_ms={request_latency_ms:.1f}",
            flush=True,
        )
        error_text = str(response.get("vlm_raw_output", ""))
        if any(marker in error_text for marker in (
            "HTTP 401", "HTTP 403", "HTTP 429", "Quota", "quota",
        )):
            summary["stopped_early_reason"] = "provider_auth_or_quota_error"
            summary["stopped_on_sample_id"] = str(sample["sample_id"])
            break

    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        final_state = build_attempt_state()
    summary["global_attempts_after_run"] = int(final_state["global_attempt_count"])
    summary["cached_successful_samples_after_run"] = int(
        len(final_state["successful_sample_ids"])
    )
    summary["ambiguous_reserved_samples_after_run"] = int(
        len(final_state["unresolved_sample_ids"])
    )
    summary["remaining_global_calls_after_run"] = int(
        HARD_MAX_CALLS - final_state["global_attempt_count"]
    )
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
