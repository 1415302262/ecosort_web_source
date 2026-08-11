"""Canonical, provider-independent EcoSort cascade policy.

This module is deliberately lightweight so the backend, tests, and offline
tools can share the audited routing semantics without importing Torch.
"""

from __future__ import annotations

import math
import threading
from collections import Counter, deque
from dataclasses import dataclass
from statistics import mean
from typing import Any, Dict, Iterable, Mapping, Optional, Sequence


AUDITED_CONFIDENCE_THRESHOLD = 0.85
AUDITED_IMAGE_SIZE = 224
AUDITED_CHECKPOINT_SHA256 = (
    "9ba817cdf0712b597e1eb322d722e130707d60d6733cf46d8d87149456946234"
)
AUDITED_CLASS_NAMES = (
    "brick_ceramic",
    "e_waste",
    "foam",
    "glass",
    "haz_battery",
    "haz_device",
    "haz_medicine",
    "hygiene_contaminated",
    "metal",
    "organic_food",
    "paper_family",
    "plastic",
    "small_hard_items",
    "textile",
)

CLASS_TO_COARSE = {
    "brick_ceramic": "other",
    "e_waste": "hazardous",
    "foam": "other",
    "glass": "recyclable",
    "haz_battery": "hazardous",
    "haz_device": "hazardous",
    "haz_medicine": "hazardous",
    "hygiene_contaminated": "other",
    "metal": "recyclable",
    "organic_food": "kitchen",
    "paper_family": "recyclable",
    "plastic": "recyclable",
    "small_hard_items": "other",
    "textile": "recyclable",
}


def normalize_label(value: str) -> str:
    return str(value).strip().lower().replace("-", "_").replace(" ", "_")


@dataclass(frozen=True)
class CascadePolicy:
    confidence_threshold: float = AUDITED_CONFIDENCE_THRESHOLD

    def __post_init__(self) -> None:
        if not 0.0 <= self.confidence_threshold <= 1.0:
            raise ValueError("confidence_threshold must be within [0, 1]")

    def should_route(self, confidence: float) -> bool:
        """Use the exact evaluated rule: route only when confidence < 0.85."""
        value = float(confidence)
        if not math.isfinite(value):
            return True
        return value < self.confidence_threshold


def coarse_category(class_name: str) -> str:
    return CLASS_TO_COARSE.get(normalize_label(class_name), "other")


def map_vlm_label(
    raw_label: Optional[str], raw_to_target: Mapping[str, str]
) -> Optional[str]:
    """Map one of the existing 42 prompt labels to the audited 14 classes."""
    if raw_label is None:
        return None
    normalized = normalize_label(raw_label)
    if normalized == "trash":
        return None
    normalized_mapping = {
        normalize_label(source): normalize_label(target)
        for source, target in raw_to_target.items()
    }
    mapped = normalized_mapping.get(normalized)
    return mapped if mapped in AUDITED_CLASS_NAMES else None


def runtime_parity(
    *,
    checkpoint_sha256: Optional[str],
    class_names: Sequence[str],
    image_size: int,
    confidence_threshold: float,
) -> Dict[str, Any]:
    checks = {
        "checkpoint_sha256": checkpoint_sha256 == AUDITED_CHECKPOINT_SHA256,
        "class_names": tuple(class_names) == AUDITED_CLASS_NAMES,
        "image_size": int(image_size) == AUDITED_IMAGE_SIZE,
        "confidence_threshold": math.isclose(
            float(confidence_threshold), AUDITED_CONFIDENCE_THRESHOLD, abs_tol=1e-12
        ),
    }
    return {
        "audited_parity": all(checks.values()),
        "checks": checks,
        "expected_checkpoint_sha256": AUDITED_CHECKPOINT_SHA256,
    }


def _percentile(values: Iterable[float], percentile: float) -> Optional[float]:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return None
    position = (len(ordered) - 1) * percentile
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


class RuntimeTelemetry:
    """Thread-safe, bounded in-memory operational metrics without image data."""

    def __init__(self, latency_window: int = 10_000) -> None:
        if latency_window < 1:
            raise ValueError("latency_window must be positive")
        self._lock = threading.Lock()
        self._counts: Counter[str] = Counter()
        self._latencies = deque(maxlen=latency_window)
        self._vlm_latencies = deque(maxlen=latency_window)
        self._token_usage: Counter[str] = Counter()

    def record(
        self,
        *,
        routed: bool,
        vlm_attempted: bool,
        vlm_success: bool,
        fallback_reason: Optional[str],
        latency_ms: float,
        vlm_latency_ms: Optional[float] = None,
        usage: Optional[Mapping[str, Any]] = None,
    ) -> None:
        with self._lock:
            self._counts["requests"] += 1
            self._counts["local_accepted" if not routed else "routed"] += 1
            if routed:
                self._counts["vlm_success" if vlm_success else "vlm_failure"] += 1
            if vlm_attempted:
                self._counts["provider_attempts"] += 1
            if fallback_reason:
                self._counts[f"reason:{fallback_reason}"] += 1
            self._latencies.append(float(latency_ms))
            if vlm_latency_ms is not None:
                self._vlm_latencies.append(float(vlm_latency_ms))
            for field in ("prompt_tokens", "completion_tokens", "total_tokens"):
                value = (usage or {}).get(field)
                if isinstance(value, (int, float)):
                    self._token_usage[field] += int(value)

    @staticmethod
    def _latency_summary(values: Iterable[float]) -> Dict[str, Optional[float]]:
        collected = list(values)
        return {
            "mean_ms": mean(collected) if collected else None,
            "p50_ms": _percentile(collected, 0.50),
            "p95_ms": _percentile(collected, 0.95),
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._lock:
            counts = dict(self._counts)
            requests = counts.get("requests", 0)
            routed = counts.get("routed", 0)
            provider_attempts = counts.get("provider_attempts", 0)
            return {
                "counts": counts,
                "local_coverage": (requests - routed) / requests if requests else None,
                "routing_rate": routed / requests if requests else None,
                "actual_vlm_call_rate": (
                    provider_attempts / requests if requests else None
                ),
                "latency": self._latency_summary(tuple(self._latencies)),
                "vlm_latency": self._latency_summary(tuple(self._vlm_latencies)),
                "token_usage": dict(self._token_usage),
            }
