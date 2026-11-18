"""Attribute only byte-identical samples to a verified upstream TrashNet archive."""
import argparse
import csv
import hashlib
import io
import json
import zipfile
from collections import Counter
from pathlib import Path

BLOB = "4184b4e9d359026a49e5d34ac2cead944f53be88"


def upstream_index(blob):
    if hashlib.sha1(b"blob " + str(len(blob)).encode() + b"\0" + blob).hexdigest() != BLOB:
        raise ValueError("Archive does not match the verified upstream TrashNet git blob")
    index = {}
    with zipfile.ZipFile(io.BytesIO(blob)) as archive:
        for item in archive.namelist():
            if not item.startswith("dataset-resized/") or not item.lower().endswith(".jpg"):
                continue
            index[hashlib.sha256(archive.read(item)).hexdigest()] = item
    if not index:
        raise ValueError("No upstream images found")
    return index


def main():
    p = argparse.ArgumentParser()
    p.add_argument("manifest", type=Path)
    p.add_argument("trashnet_master_zip", type=Path)
    p.add_argument("out", type=Path)
    a = p.parse_args()
    with zipfile.ZipFile(a.trashnet_master_zip) as outer:
        index = upstream_index(outer.read("trashnet-master/data/dataset-resized.zip"))
    counts = Counter()
    per_class = {}
    a.out.mkdir(parents=True, exist_ok=True)
    with a.manifest.open(encoding="utf-8") as inp, (a.out / "sources.csv").open("w", newline="", encoding="utf-8") as out:
        w = csv.writer(out)
        w.writerow(["sample_id", "sha256", "source_folder", "verified_trashnet_member", "decision"])
        for r in csv.DictReader(inp):
            member = index.get(r["sha256"], "")
            w.writerow([r["sample_id"], r["sha256"], r["source_folder"], member, r["decision"]])
            status = "trashnet_exact" if member else "not_matched_to_trashnet"
            counts[status] += 1
            per_class.setdefault(r["label"], Counter())[status] += 1
    (a.out / "source_summary.json").write_text(json.dumps(dict(upstream_git_blob=BLOB,
        totals=counts, per_class=per_class), indent=2))


if __name__ == "__main__":
    main()
