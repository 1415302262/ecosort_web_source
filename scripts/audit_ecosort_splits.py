#!/usr/bin/env python3
"""Audit EcoSort train/val/test integrity without modifying the dataset."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_dhash(path: Path) -> str:
    with Image.open(path) as image:
        gray = image.convert("L").resize((9, 8), Image.Resampling.LANCZOS)
        pixels = np.asarray(gray, dtype=np.int16)
    bits = pixels[:, 1:] > pixels[:, :-1]
    value = 0
    for bit in bits.reshape(-1):
        value = (value << 1) | int(bit)
    return f"{value:016x}"


def enumerate_samples(root: Path, split: str):
    split_root = root / split
    rows = []
    for class_dir in sorted(path for path in split_root.iterdir() if path.is_dir()):
        for path in sorted(class_dir.iterdir()):
            if path.suffix.lower() not in IMAGE_SUFFIXES:
                continue
            rows.append({
                "split": split,
                "class_name": class_dir.name,
                "sample_id": str(path.relative_to(root)),
                "link_path": str(path),
                "source_path": str(path.resolve(strict=False)),
                "exists": path.exists(),
            })
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/proc/ecosort_v6_15class")
    parser.add_argument("--output-dir", default="reports/ecosort_experiment_results/data_audit")
    args = parser.parse_args()

    root = Path(args.data_root).resolve()
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for split in ("train", "val", "test"):
        rows.extend(enumerate_samples(root, split))

    for index, row in enumerate(rows):
        if row["exists"]:
            path = Path(row["link_path"])
            row["sha256"] = sha256_file(path)
            row["dhash64"] = perceptual_dhash(path)
        else:
            row["sha256"] = ""
            row["dhash64"] = ""
        if (index + 1) % 1000 == 0:
            print(f"Hashed {index + 1}/{len(rows)}")

    canonical_groups = defaultdict(list)
    exact_groups = defaultdict(list)
    dhash_groups = defaultdict(list)
    for row in rows:
        canonical_groups[row["source_path"]].append(row)
        if row["sha256"]:
            exact_groups[row["sha256"]].append(row)
        if row["dhash64"]:
            dhash_groups[row["dhash64"]].append(row)

    overlap_rows = []
    for overlap_type, groups in (
        ("same_source_path", canonical_groups),
        ("exact_sha256", exact_groups),
        ("identical_dhash64_candidate", dhash_groups),
    ):
        for fingerprint, group in groups.items():
            splits = sorted({row["split"] for row in group})
            if len(splits) < 2:
                continue
            for row in group:
                overlap_rows.append({
                    "overlap_type": overlap_type,
                    "fingerprint": fingerprint,
                    "group_splits": ",".join(splits),
                    "split": row["split"],
                    "class_name": row["class_name"],
                    "sample_id": row["sample_id"],
                    "source_path": row["source_path"],
                })

    manifest_path = output_dir / "split_manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    overlap_path = output_dir / "cross_split_overlaps.csv"
    fieldnames = [
        "overlap_type", "fingerprint", "group_splits", "split",
        "class_name", "sample_id", "source_path",
    ]
    with overlap_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(overlap_rows)

    def cross_split_group_count(groups):
        return sum(1 for group in groups.values() if len({row["split"] for row in group}) >= 2)

    summary = {
        "data_root": str(root),
        "sample_counts": {
            split: sum(row["split"] == split for row in rows)
            for split in ("train", "val", "test")
        },
        "broken_link_count": sum(not row["exists"] for row in rows),
        "unique_source_path_counts": {
            split: len({row["source_path"] for row in rows if row["split"] == split})
            for split in ("train", "val", "test")
        },
        "within_split_exact_duplicate_samples": {
            split: sum(
                max(0, len([row for row in group if row["split"] == split]) - 1)
                for group in exact_groups.values()
            )
            for split in ("train", "val", "test")
        },
        "cross_split_same_source_groups": cross_split_group_count(canonical_groups),
        "cross_split_exact_sha256_groups": cross_split_group_count(exact_groups),
        "cross_split_identical_dhash_candidate_groups": cross_split_group_count(dhash_groups),
        "notes": [
            "Exact SHA256 overlap is confirmed leakage.",
            "Identical dHash is only a near-duplicate candidate and requires visual review.",
            "This audit does not prove absence of semantic or source-level near duplicates.",
        ],
        "manifest_csv": str(manifest_path),
        "overlap_csv": str(overlap_path),
    }
    summary_path = output_dir / "leakage_audit.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
