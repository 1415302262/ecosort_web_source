"""No-network unit tests for the Doubao Seed VLM adapter."""

from __future__ import annotations

import base64
import io
import sys
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from run_ecosort_seed_vlm_fallback import (  # noqa: E402
    ARK_MAX_TOTAL_IMAGE_PIXELS,
    ARK_SAFE_TOTAL_IMAGE_PIXELS,
    DEFAULT_MODEL,
    HARD_MAX_CALLS,
    image_data_uri,
    normalize_seed_response,
    resize_for_ark,
)


def synthetic_response(label: str = "battery") -> dict:
    return {
        "id": "synthetic-response",
        "model": DEFAULT_MODEL,
        "usage": {
            "prompt_tokens": 100,
            "completion_tokens": 20,
            "total_tokens": 120,
        },
        "choices": [{
            "message": {
                "content": (
                    "```json\n"
                    f'{{"class_name":"{label}","coarse_category":"hazardous",'
                    '"confidence":0.96,"top_predictions":[]}'
                    "\n```"
                )
            }
        }],
    }


def test_seed_response_parser_preserves_usage_and_model_version():
    parsed = normalize_seed_response(synthetic_response(), 200)
    assert parsed["raw_top1_class"] == "battery"
    assert parsed["provider_usage"]["total_tokens"] == 120
    assert parsed["provider_model_version"] == DEFAULT_MODEL
    assert parsed["provider_http_status"] == 200


def test_oversized_image_is_reduced_below_provider_limit():
    image = Image.new("RGB", (1000, 800), color="white")
    resized = resize_for_ark(image, max_total_pixels=500_000)
    assert resized.width * resized.height <= 500_000
    assert abs((resized.width / resized.height) - (1000 / 800)) < 0.01


def test_data_uri_contains_decodable_jpeg_with_safe_pixel_count():
    image = Image.new("RGB", (1000, 800), color="white")
    uri = image_data_uri(image, max_total_pixels=500_000)
    prefix, encoded = uri.split(",", 1)
    assert prefix == "data:image/jpeg;base64"
    decoded = Image.open(io.BytesIO(base64.b64decode(encoded)))
    assert decoded.width * decoded.height <= 500_000


def test_release_constants_lock_model_and_budget():
    assert DEFAULT_MODEL == "doubao-seed-2-0-lite-260428"
    assert HARD_MAX_CALLS == 142
    assert 0 < ARK_SAFE_TOTAL_IMAGE_PIXELS < ARK_MAX_TOTAL_IMAGE_PIXELS
