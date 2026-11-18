"""Build a deterministic manifest from live files; never modify source images."""
import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

SPLITS = ("train", "val", "test")
EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def inventory(root):
    groups = defaultdict(list)
    labels = sorted(p.name for p in (root / "train").iterdir() if p.is_dir())
    for split in SPLITS:
        dirs = sorted(p.name for p in (root / split).iterdir() if p.is_dir())
        if dirs != labels:
            raise ValueError(f"Inconsistent class directories: {split}")
        for label in labels:
            for p in sorted((root / split / label).iterdir()):
                if p.suffix.lower() not in EXT:
                    continue
                digest = hashlib.sha256(p.read_bytes()).hexdigest()
                groups[digest].append(dict(sample_id=p.relative_to(root).as_posix(),
                    split=split, label=label, sha256=digest, source_folder=p.resolve(strict=True).parent.name))
    rows = []
    for group in groups.values():
        conflict = len({r["label"] for r in group}) > 1
        for i, row in enumerate(group):
            row["decision"] = "label_conflict" if conflict else ("duplicate" if i else "keep")
            rows.append(row)
    kept = Counter((r["split"], r["label"]) for r in rows if r["decision"] == "keep")
    if not labels or any(not kept[s, c] for s in SPLITS for c in labels):
        raise ValueError("At least one required split/class has no retained images")
    return sorted(rows, key=lambda r: r["sample_id"])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("root", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    rows = inventory(a.root)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out / "manifest.csv").open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    summary = {s:dict(Counter(r["decision"] for r in rows if r["split"] == s)) for s in SPLITS}
    (a.out / "audit.json").write_text(json.dumps(summary, indent=2))
    print(summary)


if __name__ == "__main__":
    main()
