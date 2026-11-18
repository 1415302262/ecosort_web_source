import csv
import json
import sys
from pathlib import Path


def validate(folder, manifest_path):
    p = json.loads((folder / "paired.json").read_text())
    with (folder / "predictions.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == p["n"] == len({r["sample_id"] for r in rows}) == 911
    with manifest_path.open(encoding="utf-8") as f:
        manifest = {
            r["sample_id"]: r["label"]
            for r in csv.DictReader(f)
            if r["split"] == "test" and r["decision"] == "keep"
        }
    assert {r["sample_id"]: r["truth"] for r in rows} == manifest
    assert all(r["sample_id"].split("/")[1] == r["truth"] for r in rows)
    for key, name in (("clean", "clean_b3"), ("old", "deployed_b3")):
        n = sum(r[key] == r["truth"] for r in rows)
        assert abs(n / len(rows) - p[name]["accuracy"]) < 1e-12
        assert (
            sum(float(r[key + "_conf"]) < 0.85 for r in rows)
            == p[name]["route_count_at_085"]
        )
        f1 = []
        for label in set(manifest.values()):
            tp = sum(r[key] == label == r["truth"] for r in rows)
            pred = sum(r[key] == label for r in rows)
            truth = sum(r["truth"] == label for r in rows)
            f1.append(2 * tp / (pred + truth))
        assert abs(sum(f1) / 14 - p[name]["macro_f1"]) < 1e-12
    assert (
        p["deployed_b3"]["sha256"]
        == "9ba817cdf0712b597e1eb322d722e130707d60d6733cf46d8d87149456946234"
    )
    print("911 labeled pairs, both metrics and both route counts verified")


if __name__ == "__main__":
    validate(Path(sys.argv[1]), Path(sys.argv[2]))
