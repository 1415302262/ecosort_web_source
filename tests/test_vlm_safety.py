"""No-network regression tests for the bounded EcoSort VLM experiment harness."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from run_ecosort_vlm_fallback import (  # noqa: E402
    VLM_PROMPT_LABELS,
    load_v6_mapping,
    map_vlm_result,
)


RUNNER = ROOT / "scripts/run_ecosort_vlm_fallback.py"
ANALYZER = ROOT / "scripts/analyze_ecosort_vlm_cascade.py"
EXPECTED_PLAN = 142


def synthetic_input_args(tmp_path: Path) -> list[str]:
    """Create a self-contained 142-sample plan instead of relying on private data."""
    input_dir = tmp_path / "synthetic_inputs"
    input_dir.mkdir(parents=True, exist_ok=True)
    image_path = input_dir / "sample.jpg"
    if not image_path.exists():
        Image.new("RGB", (16, 16), "white").save(image_path)
    predictions_path = input_dir / "local_predictions_test.csv"
    if not predictions_path.exists():
        rows = []
        for index in range(EXPECTED_PLAN):
            true_label = "plastic" if index < 56 else "foam"
            pred_label = "foam"
            rows.append({
                "sample_id": f"test/synthetic/sample_{index:03d}.jpg",
                "source_path": str(image_path),
                "true_label": true_label,
                "pred_label": pred_label,
                "confidence": 0.50 + index * 0.001,
                "margin": 0.20,
                "normalized_entropy": 0.30,
                "correct": int(true_label == pred_label),
                "end_to_end_latency_ms": 10.0,
                "prob_foam": 0.6,
                "prob_plastic": 0.4,
            })
        pd.DataFrame(rows).to_csv(predictions_path, index=False)
    overlap_path = input_dir / "cross_split_overlaps.csv"
    if not overlap_path.exists():
        pd.DataFrame(columns=[
            "overlap_type", "fingerprint", "group_splits", "split",
            "class_name", "sample_id", "source_path",
        ]).to_csv(overlap_path, index=False)
    return [
        "--predictions-csv", str(predictions_path),
        "--overlap-csv", str(overlap_path),
    ]


def run_analyzer(output_dir: Path, fake_root: Path) -> None:
    subprocess.run(
        [
            sys.executable,
            str(ANALYZER),
            "--vlm-dir", str(output_dir),
            "--provider", "gemini",
            "--backend-path", str(fake_root / "backend/app.py"),
            *synthetic_input_args(output_dir.parent),
        ],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )


def make_fake_runtime(tmp_path: Path, behavior: str) -> Path:
    fake_root = tmp_path / f"fake_runtime_{behavior}"
    backend = fake_root / "backend"
    google = fake_root / "google"
    backend.mkdir(parents=True)
    google.mkdir(parents=True)
    (backend / "__init__.py").write_text("", encoding="utf-8")
    (google / "__init__.py").write_text("", encoding="utf-8")
    (google / "genai.py").write_text(
        "class Client:\n"
        "    def __init__(self, api_key):\n"
        "        self.api_key = api_key\n",
        encoding="utf-8",
    )
    if behavior == "crash":
        helper = (
            "import os\n"
            "GEMINI_API_KEY = ''\nGEMINI_MODEL_NAME = ''\ngemini_client = None\n"
            "def _call_vlm_primary(image):\n"
            "    os._exit(97)\n"
        )
    elif behavior == "quota":
        helper = (
            "GEMINI_API_KEY = ''\nGEMINI_MODEL_NAME = ''\ngemini_client = None\n"
            "def _call_vlm_primary(image):\n"
            "    return {\n"
            "        'class_name': 'trash',\n"
            "        'provider_usage': None,\n"
            "        'vlm_raw_output': '429 RESOURCE_EXHAUSTED synthetic quota',\n"
            "    }\n"
        )
    else:
        helper = (
            "GEMINI_API_KEY = ''\nGEMINI_MODEL_NAME = ''\ngemini_client = None\n"
            "def _call_vlm_primary(image):\n"
            "    return {\n"
            "        'class_name': 'foam',\n"
            "        'provider_usage': {\n"
            "            'prompt_token_count': 10,\n"
            "            'candidates_token_count': 2,\n"
            "            'total_token_count': 12,\n"
            "        },\n"
            "        'provider_model_version': 'synthetic-no-network',\n"
            "    }\n"
        )
    (backend / "app.py").write_text(helper, encoding="utf-8")
    return fake_root


def command_env(fake_root: Path) -> dict[str, str]:
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join((str(fake_root), str(ROOT)))
    env["ECOSORT_GEMINI_API_KEY"] = "synthetic-test-key-no-network"
    return env


def run_runner(output_dir: Path, fake_root: Path, *args: str, check: bool = True):
    return subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--output-dir",
            str(output_dir),
            "--backend-path",
            str(fake_root / "backend/app.py"),
            *synthetic_input_args(output_dir.parent),
            *args,
        ],
        cwd=ROOT,
        env=command_env(fake_root),
        check=check,
        capture_output=True,
        text=True,
        timeout=60,
    )


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def test_dry_run_plan_is_exact_confidence_only_085(tmp_path: Path):
    fake_root = make_fake_runtime(tmp_path, "success")
    output_dir = tmp_path / "dry_plan_output"
    run_runner(output_dir, fake_root, "--max-calls", str(EXPECTED_PLAN))

    plan = pd.read_csv(output_dir / "vlm_call_plan.csv")
    summary = read_json(output_dir / "vlm_run_summary.json")
    assert len(plan) == EXPECTED_PLAN
    assert (plan["confidence"] < 0.85).all()
    assert int((plan["correct"] == 0).sum()) == 56
    assert int((plan["correct"] == 1).sum()) == 86
    assert "test/e_waste/e_waste_000006.jpg" not in set(plan["sample_id"])
    assert summary["routing_strategy"].startswith("Confidence-only")
    assert summary["thresholds"] == {"confidence": 0.85}
    assert summary["planned_vlm_samples"] == EXPECTED_PLAN
    assert summary["hard_max_calls"] == EXPECTED_PLAN


def test_mapping_accepts_only_prompt_42_class_vocabulary():
    mapping = load_v6_mapping(ROOT / "configs/dataset_mapping.yaml")
    assert len(VLM_PROMPT_LABELS) == 42
    assert map_vlm_result({"class_name": "foam"}, mapping) == (
        "foam", "v6_raw_to_target_top1:foam"
    )
    assert map_vlm_result({"class_name": "trash"}, mapping) == (
        "", "unmapped_vlm_output"
    )
    assert map_vlm_result({"class_name": "textile"}, mapping) == (
        "", "out_of_42class_vocabulary"
    )
    assert map_vlm_result({"class_name": "plastic shaker"}, mapping) == (
        "", "out_of_42class_vocabulary"
    )


def test_protected_key_file_is_supported_without_key_environment_value(tmp_path: Path):
    fake_root = make_fake_runtime(tmp_path, "success")
    output_dir = tmp_path / "key_file_output"
    key_file = tmp_path / "gemini_api_key"
    key_file.write_text("synthetic-test-key-no-network", encoding="utf-8")
    key_file.chmod(0o600)
    env = command_env(fake_root)
    env.pop("ECOSORT_GEMINI_API_KEY", None)
    env["ECOSORT_GEMINI_API_KEY_FILE"] = str(key_file)

    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--output-dir",
            str(output_dir),
            "--backend-path",
            str(fake_root / "backend/app.py"),
            *synthetic_input_args(output_dir.parent),
            "--execute",
            "--max-calls",
            "1",
        ],
        cwd=ROOT,
        env=env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    summary = read_json(output_dir / "vlm_run_summary.json")
    assert summary["api_key_source"] == "protected_file"
    assert summary["executed_this_run"] == 1
    assert summary["successful_this_run"] == 1


def test_quota_error_stops_before_reserving_more_samples(tmp_path: Path):
    fake_root = make_fake_runtime(tmp_path, "quota")
    output_dir = tmp_path / "quota_output"
    run_runner(output_dir, fake_root, "--execute", "--max-calls", "10")
    summary = read_json(output_dir / "vlm_run_summary.json")
    assert summary["executed_this_run"] == 1
    assert summary["failed_this_run"] == 1
    assert summary["stopped_early_reason"] == "provider_quota_exhausted"
    assert len((output_dir / "vlm_attempt_ledger.jsonl").read_text().splitlines()) == 1
    assert len((output_dir / "vlm_response_cache.jsonl").read_text().splitlines()) == 1


def test_crash_after_reservation_consumes_budget_and_is_not_retried(tmp_path: Path):
    fake_root = make_fake_runtime(tmp_path, "crash")
    output_dir = tmp_path / "crash_output"
    result = run_runner(output_dir, fake_root, "--execute", "--max-calls", "1", check=False)
    assert result.returncode == 97
    assert len((output_dir / "vlm_attempt_ledger.jsonl").read_text().splitlines()) == 1
    assert not (output_dir / "vlm_response_cache.jsonl").exists()

    run_runner(output_dir, fake_root, "--max-calls", str(EXPECTED_PLAN))
    summary = read_json(output_dir / "vlm_run_summary.json")
    assert summary["prior_attempts_in_global_budget"] == 1
    assert summary["ambiguous_reserved_samples"] == 1
    assert summary["pending_samples"] == 141
    assert summary["allowed_calls_this_run"] == 141

    run_runner(
        output_dir, fake_root, "--max-calls", str(EXPECTED_PLAN), "--retry-failures"
    )
    retry_summary = read_json(output_dir / "vlm_run_summary.json")
    assert retry_summary["ambiguous_reserved_samples"] == 1
    assert retry_summary["pending_samples"] == 141
    assert retry_summary["allowed_calls_this_run"] == 141

    run_analyzer(output_dir, fake_root)
    metrics = read_json(output_dir / "cascade_metrics.json")
    assert metrics["evaluation_status"] == "partial_ambiguous"
    assert metrics["actual_vlm_attempt_count"] is None
    assert metrics["actual_vlm_attempt_count_lower_bound"] == 0
    assert metrics["actual_vlm_attempt_count_upper_bound"] == 1


def test_budget_is_not_refunded_when_model_changes(tmp_path: Path):
    fake_root = make_fake_runtime(tmp_path, "success")
    output_dir = tmp_path / "model_change_output"
    run_runner(output_dir, fake_root, "--execute", "--max-calls", "1")

    changed_env = command_env(fake_root)
    changed_env["ECOSORT_GEMINI_MODEL"] = "synthetic-different-model"
    subprocess.run(
        [
            sys.executable,
            str(RUNNER),
            "--output-dir",
            str(output_dir),
            "--backend-path",
            str(fake_root / "backend/app.py"),
            *synthetic_input_args(output_dir.parent),
            "--max-calls",
            str(EXPECTED_PLAN),
        ],
        cwd=ROOT,
        env=changed_env,
        check=True,
        capture_output=True,
        text=True,
        timeout=60,
    )
    summary = read_json(output_dir / "vlm_run_summary.json")
    assert summary["prior_attempts_in_global_budget"] == 1
    assert summary["remaining_global_calls_before_run"] == 141
    assert summary["allowed_calls_this_run"] == 141


def test_two_concurrent_runners_share_one_142_attempt_budget(tmp_path: Path):
    fake_root = make_fake_runtime(tmp_path, "success")
    output_dir = tmp_path / "concurrent_output"
    cmd = [
        sys.executable,
        str(RUNNER),
        "--output-dir",
        str(output_dir),
        "--backend-path",
        str(fake_root / "backend/app.py"),
        *synthetic_input_args(output_dir.parent),
        "--execute",
        "--max-calls",
        str(EXPECTED_PLAN),
    ]
    env = command_env(fake_root)
    processes = [
        subprocess.Popen(
            cmd, cwd=ROOT, env=env, stdout=subprocess.PIPE,
            stderr=subprocess.PIPE, text=True,
        )
        for _ in range(2)
    ]
    for process in processes:
        stdout, stderr = process.communicate(timeout=60)
        assert process.returncode == 0, (stdout, stderr)

    ledger_rows = (output_dir / "vlm_attempt_ledger.jsonl").read_text().splitlines()
    cache_rows = (output_dir / "vlm_response_cache.jsonl").read_text().splitlines()
    assert len(ledger_rows) == EXPECTED_PLAN
    assert len(cache_rows) == EXPECTED_PLAN
    assert len({json.loads(row)["attempt_id"] for row in ledger_rows}) == EXPECTED_PLAN

    run_runner(output_dir, fake_root, "--max-calls", str(EXPECTED_PLAN))
    summary = read_json(output_dir / "vlm_run_summary.json")
    assert summary["prior_attempts_in_global_budget"] == EXPECTED_PLAN
    assert summary["remaining_global_calls_before_run"] == 0
    assert summary["allowed_calls_this_run"] == 0

    run_analyzer(output_dir, fake_root)
    metrics = read_json(output_dir / "cascade_metrics.json")
    transitions = pd.read_csv(output_dir / "cascade_routed_transitions.csv")
    assert metrics["evaluation_status"] == "completed"
    assert metrics["planned_routed_sample_count"] == EXPECTED_PLAN
    assert metrics["actual_vlm_attempt_count"] == EXPECTED_PLAN
    assert metrics["routed_local_error_count"] == 56
    assert metrics["vlm_uncorrected_local_errors"] == (
        56 - metrics["vlm_corrected_local_errors"]
    )
    assert metrics["net_corrected"] == (
        metrics["vlm_corrected_local_errors"] - metrics["vlm_introduced_errors"]
    )
    assert sum(metrics["transition_counts"].values()) == EXPECTED_PLAN
    assert len(transitions) == EXPECTED_PLAN


def test_partial_cascade_keeps_formal_metrics_null_and_aggregates_usage(tmp_path: Path):
    fake_root = make_fake_runtime(tmp_path, "success")
    output_dir = tmp_path / "partial_output"
    run_runner(output_dir, fake_root, "--execute", "--max-calls", "1")
    run_analyzer(output_dir, fake_root)
    metrics = read_json(output_dir / "cascade_metrics.json")
    assert metrics["evaluation_status"] == "partial"
    assert metrics["cascade_metrics"] is None
    assert metrics["partial_provisional_metrics"] is not None
    assert metrics["provider_usage"]["requests_with_usage_metadata"] == 1
    assert metrics["provider_usage"]["numeric_totals"]["total_token_count"] == 12
