"""No-network regression tests for the audited online cascade runtime."""

from __future__ import annotations

import math
from pathlib import Path

import yaml
from PIL import Image

from src.inference.cascade import (
    AUDITED_CHECKPOINT_SHA256,
    AUDITED_CLASS_NAMES,
    AUDITED_CONFIDENCE_THRESHOLD,
    AUDITED_IMAGE_SIZE,
    AUDITED_VLM_MODEL,
    AUDITED_VLM_PROMPT_SHA256,
    AUDITED_VLM_PROMPT_VERSION,
    CascadePolicy,
    RuntimeTelemetry,
    map_vlm_label,
    runtime_parity,
)
from src.inference.seed_vlm import (
    PROMPT,
    PROMPT_SHA256,
    SeedClientConfig,
    SeedVLMClient,
)


ROOT = Path(__file__).resolve().parents[1]


class FakeResponse:
    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {
            "model": "doubao-seed-2-0-lite-260428",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 20,
                "total_tokens": 120,
            },
            "choices": [{
                "message": {
                    "content": '{"class_name":"battery","confidence":0.97,'
                    '"top_predictions":[]}'
                }
            }],
        }


class FakeSession:
    def __init__(self) -> None:
        self.trust_env = True
        self.calls = 0

    def post(self, *args, **kwargs) -> FakeResponse:
        self.calls += 1
        return FakeResponse()


def test_confidence_policy_matches_evaluated_strict_rule():
    policy = CascadePolicy()
    assert policy.should_route(0.849999)
    assert not policy.should_route(0.85)
    assert not policy.should_route(0.95)
    assert policy.should_route(math.nan)


def test_mapping_reuses_existing_42_to_14_semantics():
    mapping = {"battery": "haz_battery", "tissue": "hygiene_contaminated"}
    assert map_vlm_label("Battery", mapping) == "haz_battery"
    assert map_vlm_label("trash", mapping) is None
    assert map_vlm_label("unknown", mapping) is None


def test_runtime_parity_requires_all_audited_dimensions():
    exact = runtime_parity(
        checkpoint_sha256=AUDITED_CHECKPOINT_SHA256,
        class_names=AUDITED_CLASS_NAMES,
        image_size=AUDITED_IMAGE_SIZE,
        confidence_threshold=AUDITED_CONFIDENCE_THRESHOLD,
        provider_model=AUDITED_VLM_MODEL,
        prompt_version=AUDITED_VLM_PROMPT_VERSION,
        prompt_sha256=AUDITED_VLM_PROMPT_SHA256,
        raw_to_target=yaml.safe_load((ROOT / "ecosort_mapping.yaml").read_text())[
            "ecosort_config"
        ]["raw_to_target"],
    )
    assert exact["audited_parity"]
    mismatch = runtime_parity(
        checkpoint_sha256="wrong",
        class_names=AUDITED_CLASS_NAMES,
        image_size=AUDITED_IMAGE_SIZE,
        confidence_threshold=AUDITED_CONFIDENCE_THRESHOLD,
        provider_model=AUDITED_VLM_MODEL,
        prompt_version=AUDITED_VLM_PROMPT_VERSION,
        prompt_sha256=AUDITED_VLM_PROMPT_SHA256,
        raw_to_target=yaml.safe_load((ROOT / "ecosort_mapping.yaml").read_text())[
            "ecosort_config"
        ]["raw_to_target"],
    )
    assert not mismatch["audited_parity"]
    assert not mismatch["checks"]["checkpoint_sha256"]


def test_online_seed_prompt_is_locked_to_completed_experiment_contract():
    assert PROMPT_SHA256 == AUDITED_VLM_PROMPT_SHA256
    assert "Also provide its coarse category" in PROMPT
    assert '"third label"' in PROMPT
    assert "42 labels" in PROMPT


def test_runtime_mapping_file_is_locked_to_audited_policy():
    config = yaml.safe_load((ROOT / "ecosort_mapping.yaml").read_text())
    runtime = config["ecosort_config"]
    assert runtime["routing_strategy"] == "confidence_only"
    assert runtime["confidence_threshold"] == AUDITED_CONFIDENCE_THRESHOLD
    assert runtime["image_size"] == AUDITED_IMAGE_SIZE
    assert tuple(runtime["train_classes"]) == AUDITED_CLASS_NAMES
    assert "trash" not in runtime["raw_to_target"]


def test_telemetry_separates_routing_from_actual_provider_attempts():
    metrics = RuntimeTelemetry()
    metrics.record(
        routed=False,
        vlm_attempted=False,
        vlm_success=False,
        fallback_reason=None,
        latency_ms=10,
    )
    metrics.record(
        routed=True,
        vlm_attempted=False,
        vlm_success=False,
        fallback_reason="execution_disabled",
        latency_ms=20,
    )
    snapshot = metrics.snapshot()
    assert snapshot["local_coverage"] == 0.5
    assert snapshot["routing_rate"] == 0.5
    assert snapshot["actual_vlm_call_rate"] == 0.0
    assert snapshot["counts"].get("provider_attempts", 0) == 0


def test_seed_client_is_disabled_by_default_and_does_not_reserve(tmp_path):
    client = SeedVLMClient(
        SeedClientConfig(
            execute=False,
            api_key="",
            max_calls=0,
            state_dir=tmp_path,
        )
    )
    result = client.review(Image.new("RGB", (16, 16), "white"))
    assert result["failure_reason"] == "execution_disabled"
    assert not result["attempted"]
    assert client.status()["reserved_calls"] == 0


def test_seed_client_caches_success_and_never_repeats_a_paid_call(tmp_path):
    session = FakeSession()
    client = SeedVLMClient(
        SeedClientConfig(
            execute=True,
            api_key="synthetic-key",
            max_calls=1,
            state_dir=tmp_path,
        ),
        session=session,
    )
    image = Image.new("RGB", (16, 16), "white")
    first = client.review(image)
    second = client.review(image)
    assert first["success"] and first["attempted"]
    assert second["success"] and second["cached"] and not second["attempted"]
    assert first["usage"]["total_tokens"] == 120
    assert session.calls == 1
    assert client.status()["reserved_calls"] == 1

    exhausted = client.review(Image.new("RGB", (16, 16), "black"))
    assert exhausted["failure_reason"] == "budget_exhausted"
    assert session.calls == 1
