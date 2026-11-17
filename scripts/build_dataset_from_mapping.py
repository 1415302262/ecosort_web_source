"""Build a NEW stratified split; deduplicate exact bytes before assigning splits.

Existing frozen experiments are never rewritten. Symlinks are optional because
Windows may require elevated privileges; the portable default copies files.
"""

import argparse
import csv
import json
import random
import re
import shutil
from collections import defaultdict
from pathlib import Path

import yaml
from PIL import Image
from src.inference.checksum import sha256_file


def build_dataset(
    raw_root, output_root, raw_to_target, target_classes, seed=42, mode="copy"
):
    raw_root, output_root = Path(raw_root).resolve(), Path(output_root).resolve()
    if output_root.exists():
        raise ValueError("Output already exists; choose a new output directory")
    if raw_root == output_root or raw_root in output_root.parents:
        raise ValueError("Output must not be inside the raw data directory")
    if mode not in {"copy", "symlink"}:
        raise ValueError("mode must be copy or symlink")
    if not target_classes or any(
        not re.fullmatch(r"[a-z0-9_]+", c) for c in target_classes
    ):
        raise ValueError("Invalid target class names")
    by_class, seen, duplicates = defaultdict(list), {}, []
    for directory in sorted(raw_root.iterdir()):
        if not directory.is_dir() or directory.name not in raw_to_target:
            continue
        label = raw_to_target[directory.name]
        if label not in target_classes:
            raise ValueError("Mapped class is absent from target_classes")
        for path in sorted(directory.rglob("*")):
            if not path.is_file() or path.suffix.lower() not in {
                ".jpg",
                ".jpeg",
                ".png",
                ".webp",
            }:
                continue
            with Image.open(path) as image:
                image.verify()
            digest = sha256_file(path)
            if digest in seen:
                if seen[digest][0] != label:
                    raise ValueError(f"Conflicting labels for exact duplicate: {path}")
                duplicates.append(
                    {"source": str(path), "retained": seen[digest][1], "sha256": digest}
                )
                continue
            seen[digest] = (label, str(path))
            by_class[label].append((path, digest))
    if set(by_class) != set(target_classes) or any(
        len(by_class[c]) < 3 for c in target_classes
    ):
        raise ValueError(
            "Each target class requires at least three unique readable images"
        )
    rng = random.Random(seed)
    manifest = []
    output_root.mkdir(parents=True)
    for label in target_classes:
        samples = by_class[label]
        rng.shuffle(samples)
        n = len(samples)
        train_n = min(n - 2, max(1, int(n * 0.8)))
        val_n = min(n - train_n - 1, max(1, int(n * 0.1)))
        chunks = {
            "train": samples[:train_n],
            "val": samples[train_n : train_n + val_n],
            "test": samples[train_n + val_n :],
        }
        for split, entries in chunks.items():
            directory = output_root / split / label
            directory.mkdir(parents=True)
            for index, (source, digest) in enumerate(entries):
                target = directory / f"{label}_{index:06d}{source.suffix.lower()}"
                if mode == "symlink":
                    target.symlink_to(source)
                else:
                    shutil.copyfile(source, target)
                manifest.append(
                    {
                        "split": split,
                        "class_name": label,
                        "sample_id": target.relative_to(output_root).as_posix(),
                        "source_path": str(source),
                        "sha256": digest,
                    }
                )
    with (output_root / "manifest.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(manifest[0]))
        writer.writeheader()
        writer.writerows(manifest)
    summary = {
        "seed": seed,
        "mode": mode,
        "unique_images": len(manifest),
        "duplicates_removed": len(duplicates),
        "splits": {
            split: sum(row["split"] == split for row in manifest)
            for split in ("train", "val", "test")
        },
        "duplicates": duplicates,
        "limitation": "Exact-byte deduplication does not rule out near-duplicates or source-level leakage.",
    }
    (output_root / "metadata.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--raw-root", required=True)
    parser.add_argument("--output-root", required=True)
    parser.add_argument("--config-path", default="configs/dataset_mapping.yaml")
    parser.add_argument("--mapping-key", default="daily_life_mapping_v6_15class")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--mode", choices=["copy", "symlink"], default="copy")
    args = parser.parse_args()
    mapping = yaml.safe_load(Path(args.config_path).read_text(encoding="utf-8"))[
        args.mapping_key
    ]
    result = build_dataset(
        args.raw_root,
        args.output_root,
        mapping["raw_to_target"],
        mapping["target_classes"],
        args.seed,
        args.mode,
    )
    print(
        json.dumps(
            {k: v for k, v in result.items() if k != "duplicates"},
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
