"""Compare two fixed B3 checkpoints on identical inputs, after validation-only selection."""

import csv
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch
from scripts.run_ecosort_local_inference import load_current_model
from sklearn.metrics import accuracy_score, classification_report, f1_score
from src.data.dataset import TrashDataset
from torch.utils.data import DataLoader


def main(root, new, old, out):
    torch.set_num_threads(2)
    models = [load_current_model(p, torch.device("cuda")) for p in (new, old)]
    names = models[0][2]
    assert names == models[1][2]
    ds = TrashDataset(
        str(root), split="test", transform=models[0][1], class_names=names
    )
    assert len(ds) == 911
    y, probabilities = [], [[], []]
    with torch.inference_mode():
        for x, true in DataLoader(ds, batch_size=32, num_workers=4):
            y.extend(true.tolist())
            x = x.cuda()
            for i, (model, *_) in enumerate(models):
                probabilities[i].extend(model(x).softmax(dim=1).cpu().tolist())
    y = np.array(y)
    report = {
        "precision": "FP32",
        "n": len(ds),
        "gpu": torch.cuda.get_device_name(),
        "batch_size": 32,
    }
    preds = []
    for name, probs, path in zip(
        ("clean_b3", "deployed_b3"), probabilities, (new, old)
    ):
        probs = np.array(probs)
        pred = probs.argmax(1)
        preds.append(pred)
        report[name] = {
            "accuracy": accuracy_score(y, pred),
            "macro_f1": f1_score(y, pred, average="macro"),
            "route_count_at_085": int((probs.max(1) < 0.85).sum()),
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "per_class": classification_report(
                y, pred, target_names=names, output_dict=True, zero_division=0
            ),
        }
    a, b = preds
    report.update(
        corrected=int(((a == y) & (b != y)).sum()),
        regressed=int(((a != y) & (b == y)).sum()),
    )
    rng = np.random.default_rng(20261118)
    diffs = []
    for _ in range(4000):
        idx = rng.integers(0, len(y), len(y))
        diffs.append(
            float(((a[idx] == y[idx]).sum() - (b[idx] == y[idx]).sum()) / len(y))
        )
    report["paired_accuracy_delta_95pct_bootstrap"] = np.percentile(
        diffs, [2.5, 97.5]
    ).tolist()
    with (out / "predictions.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["sample_id", "truth", "clean", "old", "clean_conf", "old_conf"])
        for i in range(len(y)):
            path = (
                ds.samples[i][0] if isinstance(ds.samples[i], tuple) else ds.samples[i]
            )
            w.writerow(
                [
                    str(Path(path).relative_to(root)),
                    names[y[i]],
                    names[a[i]],
                    names[b[i]],
                    max(probabilities[0][i]),
                    max(probabilities[1][i]),
                ]
            )
    (out / "paired.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    main(*map(Path, sys.argv[1:]))
