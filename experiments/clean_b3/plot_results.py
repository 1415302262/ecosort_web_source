"""Plot validation selection and paired per-class performance from finished artifacts."""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter


def make(folder):
    history = json.loads((folder / "validation-history.json").read_text())
    p = json.loads((folder / "paired.json").read_text())
    fig, (curve, dots) = plt.subplots(
        1,
        2,
        figsize=(9, 4.8),
        constrained_layout=True,
        gridspec_kw={"width_ratios": [1, 1.25]},
    )
    epochs = [x["epoch"] + 1 for x in history]
    scores = [x["val_f1"] for x in history]
    peak = scores.index(max(scores))
    curve.plot(epochs, scores, color="#256354")
    curve.scatter(epochs[peak], scores[peak], color="#b1522d", zorder=3)
    curve.set(
        xlabel="Epoch", ylabel="Validation macro-F1", title="Checkpoint selection"
    )
    curve.yaxis.set_major_formatter(PercentFormatter(1))
    curve.annotate(
        f"Selected: epoch {epochs[peak]}",
        (epochs[peak], scores[peak]),
        xytext=(-5, -17),
        textcoords="offset points",
        ha="right",
    )
    a = p["clean_b3"]["per_class"]
    b = p["deployed_b3"]["per_class"]
    labels = sorted(
        (k for k in a if k not in ("accuracy", "macro avg", "weighted avg")),
        key=lambda k: a[k]["f1-score"] - b[k]["f1-score"],
    )
    for i, k in enumerate(labels):
        dots.plot([a[k]["f1-score"], b[k]["f1-score"]], [i, i], color="#aaa", lw=1)
    dots.scatter(
        [b[k]["f1-score"] for k in labels],
        range(len(labels)),
        color="#888",
        marker="s",
        s=24,
        label="Deployed",
    )
    dots.scatter(
        [a[k]["f1-score"] for k in labels],
        range(len(labels)),
        color="#256354",
        marker="o",
        s=20,
        label="Retrained",
    )
    dots.set(
        yticks=range(len(labels)),
        yticklabels=[x.replace("_", " ") for x in labels],
        xlabel="Test F1",
        title="Same 911 test images",
    )
    dots.xaxis.set_major_formatter(PercentFormatter(1))
    dots.legend(loc="lower center", bbox_to_anchor=(0.5, -0.27), ncol=2, frameon=False)
    for ax in (curve, dots):
        ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(folder / "comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("folder", type=Path)
    make(p.parse_args().folder)
