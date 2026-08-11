#!/usr/bin/env python3
"""Create the recruiter-facing EcoSort cascade summary from real results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans']
plt.rcParams['svg.fonttype'] = 'none'

import matplotlib as mpl
import numpy as np
import pandas as pd


LOCAL = "#4D4D4D"
CASCADE = "#0F4D92"
CASCADE_LIGHT = "#B4C0E4"
POSITIVE = "#2E9E44"
NEGATIVE = "#B64342"
NEUTRAL = "#A8A8A8"
GRID = "#E5E5E5"


def add_panel_label(ax: mpl.axes.Axes, label: str) -> None:
    ax.text(
        -0.12,
        1.04,
        label,
        transform=ax.transAxes,
        fontsize=10,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def empirical_cdf(values: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = np.sort(np.asarray(values, dtype=float))
    y = np.arange(1, len(x) + 1) / len(x)
    return x, y


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        default="reports/ecosort_experiment_results/vlm_seed2_lite_260428",
    )
    parser.add_argument("--output-dir", default="docs/portfolio/assets")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = json.loads((results_dir / "cascade_metrics.json").read_text())
    robust = json.loads(
        (results_dir / "cascade_robustness_metrics.json").read_text()
    )
    predictions = pd.read_csv(results_dir / "cascade_predictions.csv")
    per_class = pd.read_csv(results_dir / "cascade_per_class_delta.csv")

    mpl.rcParams.update({
        "font.size": 7.2,
        "axes.linewidth": 0.8,
        "axes.spines.right": False,
        "axes.spines.top": False,
        "legend.frameon": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "xtick.major.width": 0.8,
        "ytick.major.width": 0.8,
    })

    fig = plt.figure(figsize=(7.09, 6.18), facecolor="white")
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[0.92, 1.25],
        height_ratios=[0.88, 1.20],
        left=0.075,
        right=0.98,
        bottom=0.10,
        top=0.855,
        wspace=0.38,
        hspace=0.58,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1])

    fig.suptitle(
        "EcoSort: selective VLM routing improves accuracy at a measurable latency cost",
        x=0.075,
        y=0.982,
        ha="left",
        fontsize=11.2,
        fontweight="bold",
    )
    fig.text(
        0.075,
        0.943,
        "Fixed test set n=911  •  Confidence < 0.85  •  142 VLM calls (15.6%)  •  Doubao-Seed-2.0-Lite",
        ha="left",
        fontsize=7.3,
        color="#606060",
    )

    # a — paired headline metrics
    metric_labels = ["Accuracy", "Macro-F1", "Macro-Recall"]
    local_values = np.array([
        metrics["local_metrics"]["accuracy"],
        metrics["local_metrics"]["f1_macro"],
        metrics["local_metrics"]["recall_macro"],
    ]) * 100
    cascade_values = np.array([
        metrics["cascade_metrics"]["accuracy"],
        metrics["cascade_metrics"]["f1_macro"],
        metrics["cascade_metrics"]["recall_macro"],
    ]) * 100
    y = np.arange(len(metric_labels))[::-1]
    for yi, local_value, cascade_value in zip(y, local_values, cascade_values):
        ax_a.plot(
            [local_value, cascade_value],
            [yi, yi],
            color=CASCADE_LIGHT,
            linewidth=3.2,
            solid_capstyle="round",
            zorder=1,
        )
        ax_a.scatter(local_value, yi, s=31, color=LOCAL, zorder=2)
        ax_a.scatter(cascade_value, yi, s=38, color=CASCADE, zorder=3)
        ax_a.text(
            local_value - 0.22,
            yi - 0.24,
            f"{local_value:.1f}",
            ha="right",
            color=LOCAL,
            fontsize=6.8,
        )
        ax_a.text(
            cascade_value + 0.22,
            yi - 0.24,
            f"{cascade_value:.1f}",
            ha="left",
            color=CASCADE,
            fontsize=6.8,
            fontweight="bold",
        )
    ax_a.set_yticks(y, metric_labels)
    ax_a.set_xlim(89.5, 98.0)
    ax_a.set_xlabel("Score (%)")
    ax_a.grid(axis="x", color=GRID, linewidth=0.7)
    ax_a.scatter([], [], s=25, color=LOCAL, label="Local")
    ax_a.scatter([], [], s=30, color=CASCADE, label="Cascade")
    ax_a.legend(
        loc="lower right",
        bbox_to_anchor=(1.0, 1.005),
        ncol=2,
        borderaxespad=0,
        handletextpad=0.35,
        columnspacing=0.75,
        fontsize=6.6,
    )
    delta = robust["paired_delta"]["f1_macro"]
    ax_a.text(
        0.0,
        -0.31,
        (
            f"Macro-F1 Δ +{delta['estimate']*100:.2f} pp "
            f"(95% CI +{delta['ci95_low']*100:.2f} to +{delta['ci95_high']*100:.2f}); "
            f"McNemar p={robust['mcnemar_exact']['two_sided_p_value']:.1e}"
        ),
        transform=ax_a.transAxes,
        fontsize=6.5,
        color="#4D4D4D",
    )
    ax_a.set_title("Paired performance", loc="left", fontweight="bold")
    add_panel_label(ax_a, "a")

    # b — error accounting
    local_errors = int((~predictions["correct"].astype(bool)).sum())
    cascade_errors = int((~predictions["final_correct"].astype(bool)).sum())
    corrected = metrics["vlm_corrected_local_errors"]
    introduced = metrics["vlm_introduced_errors"]
    positions = np.arange(4)
    values = [local_errors, -corrected, introduced, cascade_errors]
    colors = [LOCAL, POSITIVE, NEGATIVE, CASCADE]
    labels = ["Local\nerrors", "VLM\ncorrected", "VLM\nintroduced", "Cascade\nerrors"]
    bottoms = [0, local_errors - corrected, local_errors - corrected, 0]
    heights = [local_errors, corrected, introduced, cascade_errors]
    ax_b.bar(
        positions,
        heights,
        bottom=bottoms,
        color=colors,
        width=0.62,
        edgecolor="white",
        linewidth=0.8,
    )
    ax_b.plot([0.31, 0.69], [local_errors, local_errors], color=NEUTRAL, lw=0.8)
    ax_b.plot(
        [1.31, 1.69],
        [local_errors - corrected, local_errors - corrected],
        color=NEUTRAL,
        lw=0.8,
    )
    ax_b.plot(
        [2.31, 2.69],
        [cascade_errors, cascade_errors],
        color=NEUTRAL,
        lw=0.8,
    )
    annotations = [f"{local_errors}", f"−{corrected}", f"+{introduced}", f"{cascade_errors}"]
    annotation_y = [local_errors + 2, local_errors - corrected / 2, cascade_errors + 2, cascade_errors + 2]
    annotation_colors = [LOCAL, "white", NEGATIVE, CASCADE]
    for x_value, text_value, y_value, color in zip(
        positions, annotations, annotation_y, annotation_colors
    ):
        ax_b.text(
            x_value,
            y_value,
            text_value,
            ha="center",
            va="center",
            color=color,
            fontsize=8,
            fontweight="bold",
        )
    ax_b.set_xticks(positions, labels)
    ax_b.set_ylabel("Number of errors")
    ax_b.set_ylim(0, 90)
    ax_b.grid(axis="y", color=GRID, linewidth=0.7)
    ax_b.set_title("Error accounting (Net Corrected = 45)", loc="left", fontweight="bold")
    ax_b.text(
        0.99,
        0.93,
        "49 fixed / 4 introduced",
        transform=ax_b.transAxes,
        ha="right",
        va="top",
        color="#4D4D4D",
        fontsize=6.7,
    )
    add_panel_label(ax_b, "b")

    # c — class-level delta forest plot
    class_plot = per_class.sort_values("delta_f1", ascending=True).reset_index(drop=True)
    y_class = np.arange(len(class_plot))
    estimates = class_plot["delta_f1"].to_numpy() * 100
    low = class_plot["delta_f1_ci95_low"].to_numpy() * 100
    high = class_plot["delta_f1_ci95_high"].to_numpy() * 100
    point_colors = np.where(estimates >= 0, POSITIVE, NEGATIVE)
    for yi, estimate, lo, hi, color in zip(y_class, estimates, low, high, point_colors):
        ax_c.plot([lo, hi], [yi, yi], color=color, lw=1.2, alpha=0.8)
        ax_c.scatter(estimate, yi, s=18, color=color, zorder=2)
    ax_c.axvline(0, color=NEUTRAL, linestyle="--", linewidth=0.9)
    ax_c.set_yticks(y_class, class_plot["class_name"].str.replace("_", " "))
    ax_c.set_xlabel("Macro class F1 change (percentage points)")
    ax_c.grid(axis="x", color=GRID, linewidth=0.7)
    ax_c.set_title("Class-level effect (bootstrap 95% CI)", loc="left", fontweight="bold")
    add_panel_label(ax_c, "c")

    # d — latency ECDF
    local_latency = predictions["end_to_end_latency_ms"].to_numpy(dtype=float)
    cascade_latency = predictions["cascade_latency_ms"].to_numpy(dtype=float)
    if np.any(local_latency <= 0) or np.any(cascade_latency <= 0):
        raise ValueError("Latency values must be strictly positive for a log-scale ECDF")
    x_local, y_local = empirical_cdf(local_latency)
    x_cascade, y_cascade = empirical_cdf(cascade_latency)
    ax_d.step(x_local, y_local * 100, where="post", color=LOCAL, lw=1.8, label="Local")
    ax_d.step(
        x_cascade,
        y_cascade * 100,
        where="post",
        color=CASCADE,
        lw=2.0,
        label="Cascade",
    )
    ax_d.set_xscale("log")
    ax_d.set_xlim(10, 10_000)
    ax_d.set_ylim(0, 101)
    ax_d.set_xlabel("End-to-end latency per sample (ms, log scale)")
    ax_d.set_ylabel("Samples completed (%)")
    ax_d.grid(color=GRID, linewidth=0.7, which="both")
    ax_d.legend(loc="lower right")
    ax_d.axhline(95, color=NEUTRAL, linestyle=":", linewidth=0.9)
    ax_d.text(
        70,
        93.0,
        "Local P95 65.6 ms",
        color=LOCAL,
        fontsize=6.6,
        ha="left",
    )
    ax_d.text(
        metrics["cascade_latency"]["p95_ms"],
        89,
        "Cascade P95 4.14 s",
        color=CASCADE,
        fontsize=6.6,
        ha="right",
        fontweight="bold",
    )
    ax_d.set_title("Latency distribution", loc="left", fontweight="bold")
    add_panel_label(ax_d, "d")

    fig.text(
        0.98,
        0.02,
        "Paired bootstrap: 10,000 resamples. Error bars reflect sampling uncertainty only.",
        ha="right",
        fontsize=6.1,
        color="#606060",
    )

    stem = output_dir / "ecosort_cascade_summary"
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)

    manifest = {
        "figure": "ecosort_cascade_summary",
        "source_test_samples": int(len(predictions)),
        "excluded_rows": 0,
        "panels": ["paired metrics", "error accounting", "per-class delta", "latency ECDF"],
        "outputs": [
            str(stem.with_suffix(ext))
            for ext in (".svg", ".pdf", ".png", ".tiff")
        ],
    }
    (output_dir / "ecosort_cascade_summary_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
