#!/usr/bin/env python3
"""Analyze EcoSort baseline and existing uncertainty-routing strategies.

Thresholds are selected on validation predictions. Test predictions are used
only for final reporting. Rejection metrics never pretend that a VLM result
exists: pre-fallback overall metrics remain the local-model baseline.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    precision_score,
    recall_score,
)


# This deliberately reproduces the currently implemented Streamlit set.
# For the 14-class checkpoint, only ``e_waste`` actually matches.
CURRENT_STREAMLIT_HAZARDOUS_CLASSES = {
    "battery", "bulb", "medicine", "e_waste",
    "Container for household chemicals",
}

OKABE_ITO = {
    "blue": "#0072B2",
    "orange": "#D55E00",
    "green": "#009E73",
    "purple": "#CC79A7",
    "gray": "#6B7280",
}


def configure_plot_style() -> None:
    sns.set_theme(context="paper", style="ticks")
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "savefig.facecolor": "white",
    })


def save_figure(fig: plt.Figure, output_dir: Path, stem: str, vector: bool = True) -> None:
    fig.savefig(output_dir / f"{stem}.png", dpi=300, bbox_inches="tight", facecolor="white")
    if vector:
        fig.savefig(output_dir / f"{stem}.pdf", bbox_inches="tight", facecolor="white")
        fig.savefig(output_dir / f"{stem}.svg", bbox_inches="tight", facecolor="white")


def json_dump(path: Path, payload) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def percentile_summary(values: pd.Series) -> Dict[str, float]:
    return {
        "mean_ms": float(values.mean()),
        "p50_ms": float(values.quantile(0.50)),
        "p95_ms": float(values.quantile(0.95)),
        "min_ms": float(values.min()),
        "max_ms": float(values.max()),
    }


def baseline_metrics(frame: pd.DataFrame, class_names: List[str]) -> Dict:
    true = frame["true_label"]
    pred = frame["pred_label"]
    report = classification_report(
        true, pred, labels=class_names, target_names=class_names,
        zero_division=0, output_dict=True,
    )
    return {
        "sample_count": int(len(frame)),
        "accuracy": float(accuracy_score(true, pred)),
        "precision_macro": float(precision_score(true, pred, labels=class_names, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(true, pred, labels=class_names, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(true, pred, labels=class_names, average="macro", zero_division=0)),
        "forward_latency": percentile_summary(frame["forward_latency_ms"]),
        "end_to_end_latency": percentile_summary(frame["end_to_end_latency_ms"]),
        "per_class": {name: report[name] for name in class_names},
    }


def rejected_mask(
    frame: pd.DataFrame,
    strategy: str,
    confidence_threshold: float,
    margin_threshold: float = 0.0,
    entropy_threshold: float = 1.0,
    hazardous_min_conf: float = 0.92,
) -> pd.Series:
    mask = frame["confidence"] < confidence_threshold
    if strategy in {"B_confidence_margin", "C_confidence_margin_entropy", "D_current_full"}:
        mask = mask | (frame["margin"] < margin_threshold)
    if strategy in {"C_confidence_margin_entropy", "D_current_full"}:
        mask = mask | (frame["normalized_entropy"] > entropy_threshold)
    if strategy == "D_current_full":
        mask = mask | (
            frame["pred_label"].isin(CURRENT_STREAMLIT_HAZARDOUS_CLASSES)
            & (frame["confidence"] < hazardous_min_conf)
        )
    return mask


def routing_metrics(
    frame: pd.DataFrame,
    class_names: List[str],
    strategy: str,
    confidence_threshold: float,
    margin_threshold: float = 0.0,
    entropy_threshold: float = 1.0,
    hazardous_min_conf: float = 0.92,
) -> Dict:
    rejected = rejected_mask(
        frame, strategy, confidence_threshold, margin_threshold,
        entropy_threshold, hazardous_min_conf,
    )
    accepted = ~rejected
    errors = ~frame["correct"].astype(bool)
    rejected_count = int(rejected.sum())
    accepted_count = int(accepted.sum())
    total_errors = int(errors.sum())
    rejected_errors = int((rejected & errors).sum())
    accepted_frame = frame.loc[accepted]

    if accepted_count:
        accepted_accuracy = float(accuracy_score(accepted_frame["true_label"], accepted_frame["pred_label"]))
        accepted_f1 = float(f1_score(
            accepted_frame["true_label"], accepted_frame["pred_label"],
            labels=class_names, average="macro", zero_division=0,
        ))
    else:
        accepted_accuracy = float("nan")
        accepted_f1 = float("nan")

    error_precision = rejected_errors / rejected_count if rejected_count else 0.0
    error_recall = rejected_errors / total_errors if total_errors else 0.0
    error_f1 = (
        2 * error_precision * error_recall / (error_precision + error_recall)
        if error_precision + error_recall else 0.0
    )
    overall_accuracy = frame.attrs.get("baseline_accuracy")
    overall_macro_f1 = frame.attrs.get("baseline_macro_f1")
    if overall_accuracy is None or overall_macro_f1 is None:
        overall_accuracy = float(accuracy_score(frame["true_label"], frame["pred_label"]))
        overall_macro_f1 = float(f1_score(
            frame["true_label"], frame["pred_label"], labels=class_names,
            average="macro", zero_division=0,
        ))
        frame.attrs["baseline_accuracy"] = overall_accuracy
        frame.attrs["baseline_macro_f1"] = overall_macro_f1
    return {
        "strategy": strategy,
        "confidence_threshold": float(confidence_threshold),
        "margin_threshold": float(margin_threshold),
        "entropy_threshold": float(entropy_threshold),
        "hazardous_min_conf": float(hazardous_min_conf),
        "sample_count": int(len(frame)),
        "local_coverage": accepted_count / len(frame),
        "reject_vlm_call_rate": rejected_count / len(frame),
        "accepted_count": accepted_count,
        "rejected_count": rejected_count,
        "total_local_errors": total_errors,
        "rejected_local_errors": rejected_errors,
        "accepted_local_errors": int((accepted & errors).sum()),
        "rejected_true_error_fraction": error_precision,
        "local_accepted_accuracy": accepted_accuracy,
        "local_accepted_macro_f1": accepted_f1,
        "overall_pre_fallback_accuracy": overall_accuracy,
        "overall_pre_fallback_macro_f1": overall_macro_f1,
        "error_detection_precision": error_precision,
        "error_detection_recall": error_recall,
        "error_detection_f1": error_f1,
        "overall_metrics_note": "Rejection alone has no replacement prediction; overall pre-fallback metrics equal local baseline.",
    }


def candidate_thresholds() -> Iterable[Tuple[str, float, float, float, float]]:
    confidence_values = sorted(set(np.round(np.arange(0.50, 0.951, 0.05), 2).tolist() + [0.70, 0.85]))
    margin_values = sorted(set(np.round(np.arange(0.05, 0.401, 0.05), 2).tolist() + [0.15, 0.25]))
    entropy_values = sorted(set(np.round(np.arange(0.40, 0.901, 0.05), 2).tolist() + [0.65, 0.72, 0.80]))

    for confidence in confidence_values:
        yield "A_confidence_only", confidence, 0.0, 1.0, 0.92
    for confidence in confidence_values:
        for margin in margin_values:
            yield "B_confidence_margin", confidence, margin, 1.0, 0.92
    for confidence in confidence_values:
        for margin in margin_values:
            for entropy in entropy_values:
                yield "C_confidence_margin_entropy", confidence, margin, entropy, 0.92
    # Current Streamlit strict-mode defaults, including its implemented
    # hazardous-class set. This is a fixed reference, not a new strategy.
    yield "D_current_full", 0.85, 0.25, 0.65, 0.92


def default_ablation(frame: pd.DataFrame, class_names: List[str]) -> pd.DataFrame:
    settings = [
        ("A_confidence_only", 0.85, 0.0, 1.0, 0.92),
        ("B_confidence_margin", 0.85, 0.25, 1.0, 0.92),
        ("C_confidence_margin_entropy", 0.85, 0.25, 0.65, 0.92),
        ("D_current_full", 0.85, 0.25, 0.65, 0.92),
    ]
    return pd.DataFrame([
        routing_metrics(frame, class_names, *setting) for setting in settings
    ])


def select_validation_points(sweep: pd.DataFrame) -> pd.DataFrame:
    selected = []
    for strategy in sorted(sweep["strategy"].unique()):
        strategy_rows = sweep[sweep["strategy"] == strategy]
        for budget in (0.10, 0.15, 0.20):
            eligible = strategy_rows[
                (strategy_rows["reject_vlm_call_rate"] > 0)
                & (strategy_rows["reject_vlm_call_rate"] <= budget)
            ]
            if eligible.empty:
                continue
            best = eligible.sort_values(
                ["error_detection_recall", "error_detection_precision", "reject_vlm_call_rate"],
                ascending=[False, False, True],
            ).iloc[0].to_dict()
            best["selection_budget"] = budget
            selected.append(best)
    return pd.DataFrame(selected)


def apply_selected_to_test(
    selected_val: pd.DataFrame,
    test: pd.DataFrame,
    class_names: List[str],
) -> pd.DataFrame:
    rows = []
    for _, val_row in selected_val.iterrows():
        result = routing_metrics(
            test,
            class_names,
            val_row["strategy"],
            float(val_row["confidence_threshold"]),
            float(val_row["margin_threshold"]),
            float(val_row["entropy_threshold"]),
            float(val_row["hazardous_min_conf"]),
        )
        result["selection_budget"] = float(val_row["selection_budget"])
        result["selected_on"] = "validation"
        result["validation_call_rate"] = float(val_row["reject_vlm_call_rate"])
        result["validation_error_detection_precision"] = float(val_row["error_detection_precision"])
        result["validation_error_detection_recall"] = float(val_row["error_detection_recall"])
        result["validation_error_detection_f1"] = float(val_row["error_detection_f1"])
        rows.append(result)
    return pd.DataFrame(rows)


def exact_leakage_ids(overlap_csv: Path, split: str) -> set:
    if not overlap_csv.exists():
        return set()
    overlaps = pd.read_csv(overlap_csv)
    mask = (
        (overlaps["overlap_type"] == "exact_sha256")
        & overlaps["group_splits"].astype(str).str.contains(split)
        & (overlaps["split"] == split)
    )
    return set(overlaps.loc[mask, "sample_id"].astype(str))


def dhash_candidate_ids(overlap_csv: Path, split: str) -> set:
    if not overlap_csv.exists():
        return set()
    overlaps = pd.read_csv(overlap_csv)
    mask = (
        (overlaps["overlap_type"] == "identical_dhash64_candidate")
        & overlaps["group_splits"].astype(str).str.contains(split)
        & (overlaps["split"] == split)
    )
    return set(overlaps.loc[mask, "sample_id"].astype(str))


def save_confusion_outputs(frame: pd.DataFrame, class_names: List[str], output_dir: Path) -> None:
    matrix = confusion_matrix(frame["true_label"], frame["pred_label"], labels=class_names)
    pd.DataFrame(matrix, index=class_names, columns=class_names).to_csv(output_dir / "baseline_confusion_matrix.csv")
    json_dump(output_dir / "baseline_confusion_matrix.json", {
        "class_names": class_names,
        "matrix": matrix.tolist(),
    })
    configure_plot_style()
    for grayscale, cmap, stem in (
        (False, "Blues", "baseline_confusion_matrix"),
        (True, "Greys", "baseline_confusion_matrix_grayscale"),
    ):
        fig, axis = plt.subplots(figsize=(9.2, 7.8))
        sns.heatmap(
            matrix,
            annot=True,
            fmt="d",
            cmap=cmap,
            linewidths=0.25,
            linecolor="white",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar_kws={"label": "Sample count", "shrink": 0.78},
            ax=axis,
        )
        axis.set_xlabel("Predicted label")
        axis.set_ylabel("True label")
        axis.set_title("Local EfficientNet-B3: leakage-excluded test confusion matrix")
        axis.tick_params(axis="x", rotation=45)
        for label in axis.get_xticklabels():
            label.set_horizontalalignment("right")
        axis.tick_params(axis="y", rotation=0)
        fig.tight_layout()
        save_figure(fig, output_dir, stem, vector=not grayscale)
        plt.close(fig)


def save_plots(sweep_test: pd.DataFrame, output_dir: Path) -> None:
    a = sweep_test[sweep_test["strategy"] == "A_confidence_only"].sort_values("confidence_threshold")
    configure_plot_style()
    operating = a.iloc[(a["confidence_threshold"] - 0.85).abs().argsort().iloc[0]]
    baseline_f1 = float(a["overall_pre_fallback_macro_f1"].iloc[0]) * 100

    def render_threshold(grayscale: bool) -> None:
        blue = "#222222" if grayscale else OKABE_ITO["blue"]
        orange = "#777777" if grayscale else OKABE_ITO["orange"]
        fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
        axes[0].plot(
            a["confidence_threshold"], a["local_accepted_macro_f1"] * 100,
            marker="o", linewidth=1.8, color=blue, label="Accepted-subset Macro-F1",
        )
        axes[0].axhline(
            baseline_f1, color=OKABE_ITO["gray"] if not grayscale else "#999999",
            linestyle="--", linewidth=1.2, label="Unrouted overall Macro-F1",
        )
        axes[0].set(
            xlabel="Top-1 confidence threshold",
            ylabel="Macro-F1 (%)",
            title="Selective performance",
        )
        axes[0].legend(
            frameon=True,
            facecolor="white",
            edgecolor="none",
            framealpha=0.95,
            loc="lower right",
        )
        axes[1].plot(
            a["confidence_threshold"], a["reject_vlm_call_rate"] * 100,
            marker="s", linewidth=1.8, color=orange,
        )
        axes[1].set(
            xlabel="Top-1 confidence threshold",
            ylabel="Prospective VLM call rate (%)",
            title="Routing volume",
        )
        for axis in axes:
            axis.axvline(0.85, color="#555555", linestyle=":", linewidth=1.2)
            axis.grid(axis="y", alpha=0.2)
        axes[1].annotate(
            f"15.59% at 0.85",
            xy=(0.85, operating["reject_vlm_call_rate"] * 100),
            xytext=(0.72, 25),
            arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
        )
        fig.suptitle("Confidence routing trade-off on the fixed 911-sample test set", y=1.02)
        fig.tight_layout()
        save_figure(
            fig,
            output_dir,
            "threshold_tradeoff_grayscale" if grayscale else "threshold_tradeoff",
            vector=not grayscale,
        )
        plt.close(fig)

    render_threshold(False)
    render_threshold(True)

    ordered = a.sort_values("local_coverage")
    fig, axis = plt.subplots(figsize=(6.4, 4.5))
    axis.plot(
        ordered["local_coverage"] * 100,
        ordered["local_accepted_accuracy"] * 100,
        marker="o", linewidth=1.8, color=OKABE_ITO["blue"], label="Accepted accuracy",
    )
    axis.plot(
        ordered["local_coverage"] * 100,
        ordered["local_accepted_macro_f1"] * 100,
        marker="s", linewidth=1.8, linestyle="--", color=OKABE_ITO["green"],
        label="Accepted Macro-F1",
    )
    axis.scatter(
        [operating["local_coverage"] * 100],
        [operating["local_accepted_accuracy"] * 100],
        s=55, facecolors="white", edgecolors=OKABE_ITO["orange"], linewidths=1.5, zorder=5,
        label="Threshold 0.85",
    )
    axis.set(
        xlabel="Local coverage (%)",
        ylabel="Accepted-subset metric (%)",
        title="Coverage versus selective local-model performance",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "coverage_vs_accuracy")
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(6.4, 4.5))
    axis.plot(
        a["reject_vlm_call_rate"] * 100,
        a["error_detection_precision"] * 100,
        marker="o", linewidth=1.8, color=OKABE_ITO["orange"], label="Error-detection precision",
    )
    axis.plot(
        a["reject_vlm_call_rate"] * 100,
        a["error_detection_recall"] * 100,
        marker="s", linewidth=1.8, linestyle="--", color=OKABE_ITO["blue"],
        label="Error-detection recall",
    )
    axis.scatter(
        [operating["reject_vlm_call_rate"] * 100],
        [operating["error_detection_recall"] * 100],
        s=55, facecolors="white", edgecolors=OKABE_ITO["green"], linewidths=1.5, zorder=5,
        label="Threshold 0.85",
    )
    axis.set(
        xlabel="Prospective VLM call rate (%)",
        ylabel="Error-detection metric (%)",
        title="Can uncertainty identify local-model errors?",
    )
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False)
    fig.tight_layout()
    save_figure(fig, output_dir, "error_detection_tradeoff")
    plt.close(fig)


def save_bad_cases(
    test: pd.DataFrame,
    selected_test: pd.DataFrame,
    output_dir: Path,
) -> None:
    rows = []
    confident_wrong = test[~test["correct"].astype(bool)].sort_values("confidence", ascending=False).head(12)
    for _, row in confident_wrong.iterrows():
        item = row.to_dict()
        item["bad_case_type"] = "local_confident_but_wrong"
        rows.append(item)

    if not selected_test.empty:
        preferred = selected_test.iloc[
            (selected_test["selection_budget"] - 0.15).abs().argsort().iloc[0]
        ]
        rejected = rejected_mask(
            test,
            preferred["strategy"],
            float(preferred["confidence_threshold"]),
            float(preferred["margin_threshold"]),
            float(preferred["entropy_threshold"]),
            float(preferred["hazardous_min_conf"]),
        )
        uncertain_wrong = test[rejected & ~test["correct"].astype(bool)].sort_values("confidence").head(12)
        uncertain_correct = test[rejected & test["correct"].astype(bool)].sort_values("confidence").head(8)
        for category, subset in (
            ("local_uncertain_local_wrong_vlm_pending", uncertain_wrong),
            ("local_uncertain_local_correct_vlm_risk_pending", uncertain_correct),
        ):
            for _, row in subset.iterrows():
                item = row.to_dict()
                item["bad_case_type"] = category
                rows.append(item)
    pd.DataFrame(rows).to_csv(output_dir / "bad_cases_local.csv", index=False)


def save_bad_case_summary(test: pd.DataFrame, class_names: List[str], output_dir: Path) -> None:
    errors = test[~test["correct"].astype(bool)].copy()
    confidence_rejected = test["confidence"] < 0.85
    high_confidence_wrong = errors[errors["confidence"] >= 0.85].copy()

    directed = (
        errors.groupby(["true_label", "pred_label"])
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    directed.to_csv(output_dir / "bad_case_confusion_pairs_directed.csv", index=False)

    unordered_rows = []
    for _, row in errors.iterrows():
        left, right = sorted((str(row["true_label"]), str(row["pred_label"])))
        unordered_rows.append((left, right))
    unordered = (
        pd.DataFrame(unordered_rows, columns=["class_a", "class_b"])
        .value_counts()
        .rename("count")
        .reset_index()
        .sort_values("count", ascending=False)
    )
    unordered["fraction_of_all_errors"] = unordered["count"] / len(errors)
    unordered.to_csv(output_dir / "bad_case_confusion_pairs_unordered.csv", index=False)

    support = test["true_label"].value_counts().reindex(class_names, fill_value=0)
    error_counts = errors["true_label"].value_counts().reindex(class_names, fill_value=0)
    captured_counts = (
        test[confidence_rejected & ~test["correct"].astype(bool)]["true_label"]
        .value_counts().reindex(class_names, fill_value=0)
    )
    class_summary = pd.DataFrame({
        "class_name": class_names,
        "support": support.values,
        "local_errors": error_counts.values,
        "error_rate": (error_counts / support).fillna(0).values,
        "errors_captured_at_confidence_0.85": captured_counts.values,
    })
    class_summary.to_csv(output_dir / "bad_case_class_summary.csv", index=False)

    reviewed_notes = {
        "test/organic_food/organic_food_000029.jpg": (
            "label_or_subject_mismatch",
            "Mixed-waste pile dominates the frame; flowers occupy only the foreground, while the source folder maps flower to organic_food.",
        ),
        "test/plastic/plastic_000181.jpg": (
            "probable_label_noise",
            "Painted duck toy appears wooden rather than clearly plastic; folder-level plastic_toy mapping is not material-verified.",
        ),
        "test/plastic/plastic_000014.jpg": (
            "probable_label_noise",
            "Plush toy does not present a clear plastic material cue despite the plastic_toy source mapping.",
        ),
        "test/hygiene_contaminated/hygiene_contaminated_000074.jpg": (
            "taxonomy_boundary",
            "Cigarette butts visually resemble small hard items; disposal semantics conflict with shape-based appearance.",
        ),
        "test/haz_device/haz_device_000002.jpg": (
            "taxonomy_boundary",
            "A bulb-like electronic object plausibly fits both haz_device and e_waste.",
        ),
        "test/foam/foam_000010.jpg": (
            "visual_ambiguity",
            "Rigid porous white blocks resemble ceramic or masonry in texture and shape.",
        ),
        "test/paper_family/paper_family_000002.jpg": (
            "clear_model_error",
            "The newspaper target is visually clear, but the model predicts small_hard_items with high confidence.",
        ),
        "test/haz_battery/haz_battery_000011.png": (
            "clear_model_or_calibration_error",
            "Multiple batteries are visible, but the model predicts plastic with high confidence.",
        ),
        "test/plastic/plastic_000050.jpg": (
            "clear_model_or_calibration_error",
            "LEGO-like blocks are clear, but the model predicts haz_medicine with high confidence.",
        ),
        "test/haz_medicine/haz_medicine_000005.png": (
            "taxonomy_boundary",
            "A medicine container has plastic appearance; material cues conflict with the disposal label.",
        ),
        "test/brick_ceramic/brick_ceramic_000035.jpg": (
            "context_bias",
            "The ceramic plate is clear, but residue and scene context appear to drive an organic_food prediction.",
        ),
        "test/brick_ceramic/brick_ceramic_000038.jpg": (
            "context_bias_near_duplicate",
            "Perceptual-hash duplicate of another ceramic-plate scene; the repeated context produces the same error.",
        ),
    }
    reviewed_rows = []
    for sample_id, (diagnosis, observation) in reviewed_notes.items():
        match = test[test["sample_id"] == sample_id]
        if match.empty:
            continue
        row = match.iloc[0]
        reviewed_rows.append({
            "sample_id": sample_id,
            "true_label": row["true_label"],
            "pred_label": row["pred_label"],
            "confidence": float(row["confidence"]),
            "margin": float(row["margin"]),
            "normalized_entropy": float(row["normalized_entropy"]),
            "diagnosis": diagnosis,
            "visual_observation": observation,
            "review_basis": "manual visual inspection of the source image",
        })
    pd.DataFrame(reviewed_rows).to_csv(output_dir / "bad_case_reviewed_representatives.csv", index=False)

    summary = {
        "analysis_set": "test excluding confirmed cross-split exact duplicates",
        "sample_count": int(len(test)),
        "local_error_count": int(len(errors)),
        "confidence_0_85_routing": {
            "uncertain_and_wrong": int((confidence_rejected & ~test["correct"].astype(bool)).sum()),
            "uncertain_but_correct": int((confidence_rejected & test["correct"].astype(bool)).sum()),
            "confident_but_wrong": int(len(high_confidence_wrong)),
            "error_detection_precision": float(
                (confidence_rejected & ~test["correct"].astype(bool)).sum() / confidence_rejected.sum()
            ),
            "error_detection_recall": float(
                (confidence_rejected & ~test["correct"].astype(bool)).sum() / len(errors)
            ),
        },
        "high_confidence_wrong": {
            "confidence_ge_0_85_count": int(len(high_confidence_wrong)),
            "confidence_ge_0_90_count": int((errors["confidence"] >= 0.90).sum()),
            "confidence_ge_0_95_count": int((errors["confidence"] >= 0.95).sum()),
            "mean_confidence": float(high_confidence_wrong["confidence"].mean()),
            "mean_margin": float(high_confidence_wrong["margin"].mean()),
            "mean_normalized_entropy": float(high_confidence_wrong["normalized_entropy"].mean()),
            "true_label_in_top2_count": int((high_confidence_wrong["top2_label"] == high_confidence_wrong["true_label"]).sum()),
        },
        "class_imbalance": {
            "minimum_test_support": int(support.min()),
            "maximum_test_support": int(support.max()),
            "max_to_min_support_ratio": float(support.max() / support.min()),
            "interpretation": "Imbalance is present but does not alone explain class performance.",
        },
        "top_unordered_confusion_pairs": unordered.head(10).to_dict(orient="records"),
        "vlm_dependent_bad_case_categories": {
            "local_uncertain_and_vlm_corrected": "not_evaluated_no_actual_vlm_results",
            "local_uncertain_and_vlm_still_wrong": "not_evaluated_no_actual_vlm_results",
            "vlm_introduced_error": "not_evaluated_no_actual_vlm_results",
        },
        "reviewed_representatives_csv": str(output_dir / "bad_case_reviewed_representatives.csv"),
    }
    json_dump(output_dir / "bad_case_summary.json", summary)


def save_leakage_sensitivity(
    test: pd.DataFrame,
    strict_metrics: Dict,
    overlap_csv: Path,
    class_names: List[str],
    output_dir: Path,
) -> Dict:
    candidate_ids = dhash_candidate_ids(overlap_csv, "test") & set(test["sample_id"].astype(str))
    sensitivity = test[~test["sample_id"].isin(candidate_ids)].copy()
    sensitivity_metrics = baseline_metrics(sensitivity, class_names)
    candidate_frame = test[test["sample_id"].isin(candidate_ids)]
    payload = {
        "interpretation": (
            "Identical dHash entries are cross-split near-duplicate candidates, not confirmed duplicates. "
            "This exclusion is a sensitivity analysis, not the primary result."
        ),
        "primary_sample_count": int(len(test)),
        "cross_split_dhash_candidate_test_count": int(len(candidate_ids)),
        "candidate_correct_count": int(candidate_frame["correct"].astype(bool).sum()),
        "sensitivity_sample_count": int(len(sensitivity)),
        "primary_accuracy": float(strict_metrics["accuracy"]),
        "sensitivity_accuracy": float(sensitivity_metrics["accuracy"]),
        "accuracy_delta_percentage_points": float(
            (sensitivity_metrics["accuracy"] - strict_metrics["accuracy"]) * 100
        ),
        "primary_macro_f1": float(strict_metrics["f1_macro"]),
        "sensitivity_macro_f1": float(sensitivity_metrics["f1_macro"]),
        "macro_f1_delta_percentage_points": float(
            (sensitivity_metrics["f1_macro"] - strict_metrics["f1_macro"]) * 100
        ),
        "candidate_sample_ids": sorted(candidate_ids),
    }
    json_dump(output_dir / "leakage_sensitivity.json", payload)
    return payload


def save_method_comparison(
    strict_metrics: Dict,
    test_ablation: pd.DataFrame,
    output_dir: Path,
) -> pd.DataFrame:
    rows = [{
        "method": "Local EfficientNet-B3",
        "evaluation_status": "measured",
        "overall_accuracy": strict_metrics["accuracy"],
        "overall_macro_f1": strict_metrics["f1_macro"],
        "overall_macro_recall": strict_metrics["recall_macro"],
        "local_coverage": 1.0,
        "prospective_vlm_call_rate": 0.0,
        "actual_vlm_call_rate": 0.0,
        "actual_vlm_call_count": 0,
        "accepted_subset_accuracy": strict_metrics["accuracy"],
        "accepted_subset_macro_f1": strict_metrics["f1_macro"],
        "error_detection_precision": np.nan,
        "error_detection_recall": np.nan,
        "mean_latency_ms": strict_metrics["end_to_end_latency"]["mean_ms"],
        "p50_latency_ms": strict_metrics["end_to_end_latency"]["p50_ms"],
        "p95_latency_ms": strict_metrics["end_to_end_latency"]["p95_ms"],
        "vlm_corrected_local_errors": np.nan,
        "vlm_introduced_errors": np.nan,
        "note": "Single-image end-to-end local inference; no routing or VLM.",
    }]

    method_labels = {
        "A_confidence_only": "+ Confidence routing (0.85)",
        "B_confidence_margin": "+ Confidence + margin routing",
        "C_confidence_margin_entropy": "+ Confidence + margin + entropy routing",
        "D_current_full": (
            "+ Streamlit-threshold full routing (standardized inference)"
        ),
    }
    for _, ablation in test_ablation.iterrows():
        rows.append({
            "method": method_labels[str(ablation["strategy"])],
            "evaluation_status": "measured_rejection_only_no_replacement",
            "overall_accuracy": strict_metrics["accuracy"],
            "overall_macro_f1": strict_metrics["f1_macro"],
            "overall_macro_recall": strict_metrics["recall_macro"],
            "local_coverage": float(ablation["local_coverage"]),
            "prospective_vlm_call_rate": float(ablation["reject_vlm_call_rate"]),
            "actual_vlm_call_rate": 0.0,
            "actual_vlm_call_count": 0,
            "accepted_subset_accuracy": float(ablation["local_accepted_accuracy"]),
            "accepted_subset_macro_f1": float(ablation["local_accepted_macro_f1"]),
            "error_detection_precision": float(ablation["error_detection_precision"]),
            "error_detection_recall": float(ablation["error_detection_recall"]),
            "mean_latency_ms": strict_metrics["end_to_end_latency"]["mean_ms"],
            "p50_latency_ms": strict_metrics["end_to_end_latency"]["p50_ms"],
            "p95_latency_ms": strict_metrics["end_to_end_latency"]["p95_ms"],
            "vlm_corrected_local_errors": np.nan,
            "vlm_introduced_errors": np.nan,
            "note": (
                "Rejected samples have no replacement prediction, so overall metrics remain the local baseline; "
                "latency is the measured local pipeline only."
            ),
        })

    current = test_ablation[test_ablation["strategy"] == "D_current_full"].iloc[0]
    rows.append({
        "method": "+ Uncertainty routing + VLM fallback",
        "evaluation_status": "not_evaluated_missing_api_key",
        "overall_accuracy": np.nan,
        "overall_macro_f1": np.nan,
        "overall_macro_recall": np.nan,
        "local_coverage": float(current["local_coverage"]),
        "prospective_vlm_call_rate": float(current["reject_vlm_call_rate"]),
        "actual_vlm_call_rate": 0.0,
        "actual_vlm_call_count": 0,
        "accepted_subset_accuracy": float(current["local_accepted_accuracy"]),
        "accepted_subset_macro_f1": float(current["local_accepted_macro_f1"]),
        "error_detection_precision": float(current["error_detection_precision"]),
        "error_detection_recall": float(current["error_detection_recall"]),
        "mean_latency_ms": np.nan,
        "p50_latency_ms": np.nan,
        "p95_latency_ms": np.nan,
        "vlm_corrected_local_errors": np.nan,
        "vlm_introduced_errors": np.nan,
        "note": (
            "143 calls planned; a locked pre-call reservation ledger enforces one "
            "permanent cumulative cap across processes and configuration changes. "
            "Zero actual calls and no VLM outcome cache."
        ),
    })
    comparison = pd.DataFrame(rows)
    comparison.to_csv(output_dir / "method_comparison.csv", index=False)
    serializable = comparison.astype(object).where(pd.notna(comparison), None)
    json_dump(output_dir / "method_comparison.json", serializable.to_dict(orient="records"))
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--val-csv", default="reports/ecosort_experiment_results/raw/local_predictions_val.csv")
    parser.add_argument("--test-csv", default="reports/ecosort_experiment_results/raw/local_predictions_test.csv")
    parser.add_argument("--overlap-csv", default="reports/ecosort_experiment_results/data_audit/cross_split_overlaps.csv")
    parser.add_argument("--output-dir", default="reports/ecosort_experiment_results/analysis")
    args = parser.parse_args()

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    validation_full = pd.read_csv(args.val_csv)
    test_full = pd.read_csv(args.test_csv)
    class_names = [column.removeprefix("prob_") for column in test_full.columns if column.startswith("prob_")]

    leaked_validation_ids = exact_leakage_ids(Path(args.overlap_csv), "val")
    leaked_test_ids = exact_leakage_ids(Path(args.overlap_csv), "test")
    validation = validation_full[~validation_full["sample_id"].isin(leaked_validation_ids)].copy()
    test = test_full[~test_full["sample_id"].isin(leaked_test_ids)].copy()

    full_metrics = baseline_metrics(test_full, class_names)
    strict_metrics = baseline_metrics(test, class_names)
    baseline_payload = {
        "primary_analysis_set": (
            "test-side evaluation excluding confirmed cross-split exact duplicates"
        ),
        "checkpoint_selection_leakage_warning": (
            "The champion epoch was selected on the original validation split, which "
            "contained one train-validation exact duplicate. Post-hoc exclusion cannot "
            "remove its unquantified effect on checkpoint selection."
        ),
        "confirmed_leaked_validation_sample_ids": sorted(leaked_validation_ids),
        "confirmed_leaked_test_sample_ids": sorted(leaked_test_ids),
        "full_test": full_metrics,
        "leakage_excluded_test": strict_metrics,
    }
    json_dump(output_dir / "baseline_metrics.json", baseline_payload)

    per_class = pd.DataFrame(strict_metrics["per_class"]).T.reset_index().rename(columns={"index": "class_name"})
    per_class.to_csv(output_dir / "baseline_per_class_metrics.csv", index=False)
    save_confusion_outputs(test, class_names, output_dir)
    leakage_sensitivity = save_leakage_sensitivity(
        test, strict_metrics, Path(args.overlap_csv), class_names, output_dir,
    )

    val_sweep_rows = [routing_metrics(validation, class_names, *candidate) for candidate in candidate_thresholds()]
    test_sweep_rows = [routing_metrics(test, class_names, *candidate) for candidate in candidate_thresholds()]
    val_sweep = pd.DataFrame(val_sweep_rows)
    test_sweep = pd.DataFrame(test_sweep_rows)
    val_sweep.to_csv(output_dir / "routing_threshold_sweep_validation.csv", index=False)
    test_sweep.to_csv(output_dir / "routing_threshold_sweep_test.csv", index=False)
    test_sweep[test_sweep["strategy"] == "A_confidence_only"].to_csv(
        output_dir / "threshold_sweep_confidence_only.csv", index=False,
    )

    default_ablation(validation, class_names).assign(split="validation").to_csv(
        output_dir / "routing_ablation_validation.csv", index=False,
    )
    test_ablation = default_ablation(test, class_names).assign(split="test")
    test_ablation.to_csv(output_dir / "routing_ablation_test.csv", index=False)
    method_comparison = save_method_comparison(strict_metrics, test_ablation, output_dir)

    selected_val = select_validation_points(val_sweep)
    selected_val.to_csv(output_dir / "selected_operating_points_validation.csv", index=False)
    selected_test = apply_selected_to_test(selected_val, test, class_names)
    selected_test.to_csv(output_dir / "selected_operating_points_test.csv", index=False)

    save_plots(test_sweep, output_dir)
    save_bad_cases(test, selected_test, output_dir)
    save_bad_case_summary(test, class_names, output_dir)

    summary = {
        "class_names": class_names,
        "full_validation_sample_count": int(len(validation_full)),
        "validation_sample_count": int(len(validation)),
        "full_test_sample_count": int(len(test_full)),
        "primary_test_sample_count": int(len(test)),
        "confirmed_leaked_test_count": int(len(leaked_test_ids)),
        "confirmed_leaked_validation_count": int(len(leaked_validation_ids)),
        "checkpoint_selection_leakage_warning": (
            "Champion selection used the original validation split containing one "
            "train-validation exact duplicate; its effect cannot be removed post hoc."
        ),
        "routing_input_scope": (
            "Streamlit threshold formulas applied to checkpoint-consistent letterbox224 "
            "L20/FP16 outputs, not Streamlit's 300x300 CPU/FP32 end-to-end pipeline."
        ),
        "cross_split_dhash_sensitivity": leakage_sensitivity,
        "baseline": strict_metrics,
        "current_default_routing_test": test_ablation.to_dict(orient="records"),
        "method_comparison": method_comparison.astype(object).where(
            pd.notna(method_comparison), None,
        ).to_dict(orient="records"),
        "selected_operating_points_test": selected_test.to_dict(orient="records"),
        "vlm_status": "not_evaluated; no cache or API result was used",
    }
    json_dump(output_dir / "routing_analysis_summary.json", summary)
    print(json.dumps({
        "baseline": strict_metrics,
        "selected_operating_points_test": selected_test.to_dict(orient="records"),
    }, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
