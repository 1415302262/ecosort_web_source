"""Materialize a verified, deduplicated snapshot in a new private directory."""

import csv
import hashlib
import sys
from pathlib import Path


def main(manifest, source, dest):
    if (
        hashlib.sha256(manifest.read_bytes()).hexdigest()
        != "eb30ec7afb832a35e3500c4fd3ec5262efd1b95a57a6eba0b6c97d61e9fc073c"
    ):
        raise ValueError(
            "Unexpected input manifest (expected the reviewed 14-class split)"
        )
    dest.mkdir(mode=0o700)
    seen = set()
    counts = {"train": 0, "val": 0, "test": 0}
    with manifest.open() as f:
        for r in csv.DictReader(f):
            if r["decision"] != "keep":
                continue
            name = Path(r["sample_id"])
            if (
                len(name.parts) != 3
                or name.parts[:2] != (r["split"], r["label"])
                or ".." in name.parts
            ):
                raise ValueError(name)
            data = (source / name).read_bytes()
            digest = hashlib.sha256(data).hexdigest()
            if digest != r["sha256"] or digest in seen:
                raise ValueError(f"Image mismatch or duplicate: {name}")
            seen.add(digest)
            counts[r["split"]] += 1
            target = dest / name
            target.parent.mkdir(exist_ok=True, parents=True)
            target.write_bytes(data)
    if counts != {"train": 7206, "val": 895, "test": 911}:
        raise ValueError(counts)
    print(counts)


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
