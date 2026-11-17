#!/usr/bin/env python3
"""Fail fast when public EcoSort artifacts drift or expose private state."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


EXPECTED = {
    "test_sample_count": 911,
    "actual_calls": 142,
    "local_f1": 0.9120789156401253,
    "cascade_f1": 0.9633674228632408,
    "net_corrected": 45,
}
SECRET_PATTERNS = {
    "host_absolute_path": re.compile(r"/public/home/"),
    "ark_key": re.compile(r"AQ\.[A-Za-z0-9_-]{16,}"),
    "bearer_token": re.compile(r"Bearer\s+[A-Za-z0-9_-]{16,}"),
    "generic_secret": re.compile(
        r"(?i)(?:api[_-]?key|secret|token)\s*[:=]\s*['\"][A-Za-z0-9_-]{16,}"
    ),
    "provider_response_id": re.compile(r"provider_response_id|attempt_id"),
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", default=".")
    args = parser.parse_args()
    root = Path(args.repo_root).resolve()
    public = root / "docs" / "portfolio"
    data = public / "data"
    errors: list[str] = []

    summary_path = data / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    checks = {
        "test_sample_count": summary["test_sample_count"],
        "actual_calls": summary["routing"]["actual_calls"],
        "local_f1": summary["local_metrics"]["f1_macro"],
        "cascade_f1": summary["cascade_metrics"]["f1_macro"],
        "net_corrected": summary["error_accounting"]["net_corrected"],
    }
    for name, expected in EXPECTED.items():
        if checks[name] != expected:
            errors.append(f"metric drift: {name}={checks[name]!r}, expected {expected!r}")

    checksum_path = data / "SHA256SUMS.txt"
    for line in checksum_path.read_text(encoding="utf-8").splitlines():
        expected_hash, filename = line.split(maxsplit=1)
        target = data / filename.strip()
        if not target.is_file():
            errors.append(f"missing checksummed file: {target.name}")
        elif sha256_file(target) != expected_hash:
            errors.append(f"checksum mismatch: {target.name}")

    public_text_files = [
        path for path in public.rglob("*")
        if path.is_file() and path.suffix.lower() in {".html", ".json", ".csv", ".txt", ".md", ".svg"}
    ]
    for path in public_text_files:
        text = path.read_text(encoding="utf-8", errors="replace")
        for name, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                errors.append(f"privacy scan {name}: {path.relative_to(root)}")

    html = (public / "index.html").read_text(encoding="utf-8")
    for required in (
        "96.337%", "15.59%", "Net Corrected", "data/summary.json",
        "assets/ecosort_cascade_summary.svg",
    ):
        if required not in html:
            errors.append(f"portfolio missing required evidence: {required}")

    if errors:
        raise SystemExit("\n".join(f"ERROR: {error}" for error in errors))
    print(
        json.dumps(
            {
                "status": "ok",
                "public_files_scanned": len(public_text_files),
                "metrics": checks,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
