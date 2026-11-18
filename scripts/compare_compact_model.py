"""Train an independent compact candidate and compare with frozen B3 on identical images."""
import argparse
import csv
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import mobilenet_v3_small

from scripts.run_ecosort_local_inference import load_current_model
from src.data.dataset import get_data_transforms


class Photos(Dataset):
    def __init__(self, rows, labels, root, mode):
        self.rows, self.labels, self.root = rows, labels, root
        self.transform = get_data_transforms(mode, 224)

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        row = self.rows[i]
        with Image.open(self.root / row["sample_id"]) as im:
            img = self.transform(im.convert("RGB"))
        return img, self.labels.index(row["label"])


@torch.inference_mode()
def predict(model, loader):
    model.eval()
    actual, guesses = [], []
    for x, y in loader:
        actual.extend(y.tolist())
        guesses.extend(model(x.cuda()).argmax(1).tolist())
    return actual, guesses


@torch.inference_mode()
def latency(model, sample):
    model.eval()
    times = []
    for i in range(110):
        torch.cuda.synchronize()
        t = time.perf_counter()
        model(sample)
        torch.cuda.synchronize()
        if i >= 10:
            times.append(1000 * (time.perf_counter() - t))
    return {"p50_ms": statistics.median(times), "p95_ms":float(np.percentile(times, 95))}


def run():
    p = argparse.ArgumentParser()
    for key in ("root", "manifest", "weights", "baseline", "out"):
        p.add_argument("--" + key, required=True, type=Path)
    p.add_argument("--epochs", type=int, default=6)
    a = p.parse_args()
    seed = 20261118
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    a.out.mkdir(parents=True, exist_ok=True)
    with a.manifest.open(encoding="utf-8") as f:
        rows = [r for r in csv.DictReader(f) if r["decision"] == "keep"]
    labels = sorted({r["label"] for r in rows})
    loaders = {s:DataLoader(Photos([r for r in rows if r["split"] == s], labels, a.root, s),
                            batch_size=64, shuffle=s=="train", num_workers=4, persistent_workers=s=="train")
               for s in ("train", "val", "test")}
    if hashlib.sha256(a.weights.read_bytes()).hexdigest() != "047dcff4addef86ea5bc2eff13c9614dc11f47ab1160d0a71a25e7db994f4e1f":
        raise ValueError("Unexpected ImageNet initialization")
    model = mobilenet_v3_small(weights=None)
    model.load_state_dict(torch.load(a.weights, weights_only=True))
    model.classifier[-1] = torch.nn.Linear(model.classifier[-1].in_features, len(labels))
    model.cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=0.00035)
    loss = torch.nn.CrossEntropyLoss(label_smoothing=0.03)
    best, history = -1, []
    for epoch in range(1, a.epochs + 1):
        model.train()
        for x, y in loaders["train"]:
            opt.zero_grad(set_to_none=True)
            loss(model(x.cuda()), y.cuda()).backward()
            opt.step()
        actual, guesses = predict(model, loaders["val"])
        score = f1_score(actual, guesses, average="macro")
        history.append(float(score))
        print(f"epoch {epoch} validation macro-F1 {score:.4f}", flush=True)
        if score > best:
            best, best_epoch = score, epoch
            torch.save({"weights":model.state_dict(), "labels":labels}, a.out / "compact.pth")
    model.load_state_dict(torch.load(a.out / "compact.pth", weights_only=True)["weights"])
    baseline, _, expected, _, _ = load_current_model(a.baseline, torch.device("cuda"))
    if expected != labels:
        raise ValueError("Inconsistent checkpoint class order")
    report = {"seed":seed, "device":torch.cuda.get_device_name(), "torch":str(torch.__version__),
              "manifest_sha256":hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
              "best_epoch":best_epoch, "validation_history":history, "precision":"float32",
              "baseline_sha256":hashlib.sha256(a.baseline.read_bytes()).hexdigest()}
    sample = next(iter(loaders["test"]))[0][:1].cuda()
    for name, m in (("compact", model), ("baseline", baseline)):
        y, pred = predict(m, loaders["test"])
        report[name] = {"accuracy":accuracy_score(y, pred), "macro_f1":f1_score(y,pred,average="macro"),
                        "test_n":len(y), "parameters":sum(t.numel() for t in m.parameters()),
                        "per_class":classification_report(y, pred, target_names=labels, output_dict=True, zero_division=0),
                        "batch1_forward":latency(m,sample)}
        print(name, report[name]["macro_f1"], flush=True)
    (a.out / "comparison.json").write_text(json.dumps(report, indent=2))


if __name__ == "__main__":
    run()
