"""Budgeted Doubao Seed VLM adapter for online EcoSort fallback.

Real calls are disabled unless all of the following are configured:
`ECOSORT_VLM_EXECUTE=1`, a non-empty key, and a positive
`ECOSORT_VLM_MAX_CALLS`. Every attempt is reserved durably before network I/O.
"""

from __future__ import annotations

import base64
import fcntl
import hashlib
import io
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import requests
from PIL import Image

from .cascade import AUDITED_VLM_MODEL, AUDITED_VLM_PROMPT_VERSION


DEFAULT_MODEL = AUDITED_VLM_MODEL
DEFAULT_BASE_URL = "https://ark.cn-beijing.volces.com/api/v3"
DEFAULT_TIMEOUT_SECONDS = 120
ARK_SAFE_TOTAL_IMAGE_PIXELS = 35_000_000
PROMPT_VERSION = AUDITED_VLM_PROMPT_VERSION


PROMPT = """
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
PROMPT_SHA256 = hashlib.sha256(PROMPT.encode("utf-8")).hexdigest()


def _append_jsonl(path: Path, row: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


def _load_jsonl(path: Path) -> list[Dict[str, Any]]:
    if not path.is_file():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _extract_json(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", stripped, re.DOTALL)
    if fenced:
        stripped = fenced.group(1)
    else:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start >= 0 and end > start:
            stripped = stripped[start : end + 1]
    parsed = json.loads(stripped)
    if not isinstance(parsed, dict):
        raise ValueError("provider output is not a JSON object")
    return parsed


def _resize_for_provider(image: Image.Image) -> Image.Image:
    width, height = image.size
    if width * height <= ARK_SAFE_TOTAL_IMAGE_PIXELS:
        return image
    scale = math.sqrt(ARK_SAFE_TOTAL_IMAGE_PIXELS / float(width * height))
    new_width = max(1, int(math.floor(width * scale)))
    new_height = max(1, int(math.floor(height * scale)))
    while new_width * new_height > ARK_SAFE_TOTAL_IMAGE_PIXELS:
        if new_width >= new_height:
            new_width -= 1
        else:
            new_height -= 1
    return image.resize((new_width, new_height), Image.Resampling.LANCZOS)


def _jpeg_bytes(image: Image.Image) -> bytes:
    converted = _resize_for_provider(image.convert("RGB"))
    buffer = io.BytesIO()
    converted.save(buffer, format="JPEG", quality=90, optimize=True)
    return buffer.getvalue()


@dataclass(frozen=True)
class SeedClientConfig:
    execute: bool
    api_key: str
    max_calls: int
    model: str = DEFAULT_MODEL
    base_url: str = DEFAULT_BASE_URL
    timeout_seconds: int = DEFAULT_TIMEOUT_SECONDS
    state_dir: Path = Path("runtime/seed_vlm")

    def __post_init__(self) -> None:
        if self.max_calls < 0:
            raise ValueError("max_calls must be non-negative")
        if self.timeout_seconds < 1:
            raise ValueError("timeout_seconds must be positive")


class SeedVLMClient:
    def __init__(
        self,
        config: SeedClientConfig,
        session: Optional[requests.Session] = None,
    ) -> None:
        self.config = config
        self.state_dir = Path(config.state_dir)
        self.cache_path = self.state_dir / "response_cache.jsonl"
        self.ledger_path = self.state_dir / "attempt_ledger.jsonl"
        self.lock_path = self.state_dir / ".attempt.lock"
        # A caller-supplied session is useful for deterministic tests. In the
        # threaded web runtime, use one short-lived session per request because
        # requests.Session does not promise cross-thread safety.
        self.session = session
        if self.session is not None:
            self.session.trust_env = False

    @classmethod
    def from_env(cls) -> "SeedVLMClient":
        execute = os.getenv("ECOSORT_VLM_EXECUTE", "0").strip().lower() in {
            "1", "true", "yes", "on"
        }
        key = os.getenv("ECOSORT_ARK_API_KEY", "").strip()
        key_file_value = os.getenv("ECOSORT_ARK_API_KEY_FILE", "").strip()
        if not key and key_file_value:
            key_path = Path(key_file_value).expanduser()
            if key_path.is_symlink() or not key_path.is_file():
                raise ValueError("ECOSORT_ARK_API_KEY_FILE must be a regular file")
            key = key_path.read_text(encoding="utf-8").strip()
        config = SeedClientConfig(
            execute=execute,
            api_key=key,
            max_calls=int(os.getenv("ECOSORT_VLM_MAX_CALLS", "0")),
            model=os.getenv("ECOSORT_ARK_MODEL", DEFAULT_MODEL).strip(),
            base_url=os.getenv("ECOSORT_ARK_BASE_URL", DEFAULT_BASE_URL).rstrip("/"),
            timeout_seconds=int(
                os.getenv("ECOSORT_VLM_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS))
            ),
            state_dir=Path(os.getenv("ECOSORT_VLM_STATE_DIR", "runtime/seed_vlm")),
        )
        return cls(config)

    def status(self) -> Dict[str, Any]:
        self.state_dir.mkdir(parents=True, exist_ok=True)
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            cache = _load_jsonl(self.cache_path)
            ledger = _load_jsonl(self.ledger_path)
        return {
            "provider": "volcano_ark",
            "model": self.config.model,
            "execute": self.config.execute,
            "configured": bool(self.config.api_key),
            "max_calls": self.config.max_calls,
            "reserved_calls": len(ledger),
            "cached_results": len(cache),
            "remaining_calls": max(0, self.config.max_calls - len(ledger)),
        }

    def _cache_key(self, image_bytes: bytes) -> str:
        digest = hashlib.sha256()
        digest.update(image_bytes)
        digest.update(self.config.model.encode())
        digest.update(PROMPT_VERSION.encode())
        return digest.hexdigest()

    @staticmethod
    def _result(
        *,
        success: bool,
        cache_key: str,
        failure_reason: Optional[str] = None,
        cached: bool = False,
        attempted: bool = False,
        **extra: Any,
    ) -> Dict[str, Any]:
        return {
            "success": success,
            "cache_key": cache_key,
            "failure_reason": failure_reason,
            "cached": cached,
            "attempted": attempted,
            **extra,
        }

    def review(self, image: Image.Image) -> Dict[str, Any]:
        image_bytes = _jpeg_bytes(image)
        cache_key = self._cache_key(image_bytes)
        self.state_dir.mkdir(parents=True, exist_ok=True)

        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            cache_rows = _load_jsonl(self.cache_path)
            cached = next(
                (row for row in reversed(cache_rows) if row.get("cache_key") == cache_key),
                None,
            )
            if cached is not None:
                result = dict(cached)
                result["cached"] = True
                result["attempted"] = False
                return result

            ledger_rows = _load_jsonl(self.ledger_path)
            if any(row.get("cache_key") == cache_key for row in ledger_rows):
                return self._result(
                    success=False,
                    cache_key=cache_key,
                    failure_reason="reserved_without_result",
                )
            if not self.config.execute:
                return self._result(
                    success=False, cache_key=cache_key, failure_reason="execution_disabled"
                )
            if not self.config.api_key:
                return self._result(
                    success=False, cache_key=cache_key, failure_reason="missing_api_key"
                )
            if len(ledger_rows) >= self.config.max_calls:
                return self._result(
                    success=False, cache_key=cache_key, failure_reason="budget_exhausted"
                )
            attempt_id = uuid.uuid4().hex
            _append_jsonl(
                self.ledger_path,
                {
                    "attempt_id": attempt_id,
                    "cache_key": cache_key,
                    "model": self.config.model,
                    "reserved_at_unix": time.time(),
                },
            )

        data_uri = "data:image/jpeg;base64," + base64.b64encode(image_bytes).decode()
        payload = {
            "model": self.config.model,
            "messages": [{
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {"type": "image_url", "image_url": {"url": data_uri}},
                ],
            }],
            "max_tokens": 256,
            "temperature": 0,
        }
        started = time.perf_counter()
        request_session = self.session or requests.Session()
        request_session.trust_env = False
        try:
            response = request_session.post(
                f"{self.config.base_url}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.config.api_key}",
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=self.config.timeout_seconds,
            )
            latency_ms = (time.perf_counter() - started) * 1000.0
            response.raise_for_status()
            provider_payload = response.json()
            content = provider_payload["choices"][0]["message"]["content"]
            parsed = _extract_json(content)
            result = self._result(
                success=True,
                cache_key=cache_key,
                attempted=True,
                raw_label=str(parsed.get("class_name", "")),
                confidence=float(parsed.get("confidence", 0.0)),
                usage=provider_payload.get("usage") or {},
                latency_ms=latency_ms,
                provider_model=provider_payload.get("model") or self.config.model,
            )
        except Exception as exc:
            result = self._result(
                success=False,
                cache_key=cache_key,
                attempted=True,
                failure_reason=type(exc).__name__,
                latency_ms=(time.perf_counter() - started) * 1000.0,
            )
        finally:
            if self.session is None:
                request_session.close()
        with self.lock_path.open("a+", encoding="utf-8") as lock_handle:
            fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
            _append_jsonl(self.cache_path, result)
        return result
