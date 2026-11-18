"""Measure single-image FP32 forward time with the production CPU thread count."""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torchvision.models import mobilenet_v3_small

from scripts.run_ecosort_local_inference import load_current_model


def main():
    p = argparse.ArgumentParser()
    for k in ("baseline", "compact", "out"):
        p.add_argument("--" + k, type=Path, required=True)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.set_num_interop_threads(1)
    ck = torch.load(a.compact, weights_only=True, map_location="cpu")
    model = mobilenet_v3_small(num_classes=len(ck["labels"]))
    model.load_state_dict(ck["weights"])
    base, _, labels, _, spec = load_current_model(a.baseline, torch.device("cpu"))
    assert labels == ck["labels"]
    torch.manual_seed(0)
    sample = torch.randn(1, 3, spec["image_size"], spec["image_size"])
    result = {}
    with torch.inference_mode():
        for name, net in (("compact", model.eval()), ("baseline", base.eval())):
            times = []
            for i in range(110):
                start = time.perf_counter()
                net(sample)
                if i >= 10:
                    times.append((time.perf_counter() - start) * 1000)
            result[name] = {"p50_ms":float(np.median(times)), "p95_ms":float(np.percentile(times, 95))}
    a.out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
