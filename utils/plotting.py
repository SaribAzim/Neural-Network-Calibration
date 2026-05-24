"""
plotting.py
-----------
All visualisation functions for the calibration study.

Three plots are produced:
1. Reliability Diagram  - bar chart of accuracy vs confidence per bin
2. Confidence Histogram - how often the model uses each confidence level
3. ECE Comparison Bar   - before vs after calibration
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches


# Shared style
STYLE = {
    "before_color": "#E07B54",   # warm orange
    "after_color": "#5B8DB8",    # calm blue
    "perfect_color": "#2ECC71",  # green
    "bg_color": "#F9F9F9",
    "grid_color": "#E0E0E0",
    "font_size": 12,
}


def _apply_base_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent styling to plots."""
    ax.set_facecolor(STYLE["bg_color"])
    ax.set_title(
        title,
        fontsize=STYLE["font_size"] + 2,
        fontweight="bold",
        pad=12
    )

    ax.set_xlabel(xlabel, fontsize=STYLE["font_size"])
    ax.set_ylabel(ylabel, fontsize=STYLE["font_size"])

    ax.grid(
        True,
        color=STYLE["grid_color"],
        linewidth=0.8
    )

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ==========================================================
# Plot 1 - Reliability Diagram
# ==========================================================

def plot_reliability_diagram(
    bins_before,
    bins_after,
    ece_before,
    ece_after,
    save_path="results/reliability_diagram.png"
):
    """
    Side-by-side reliability diagrams.
    """

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 6)
    )

    fig.suptitle(
        "Reliability Diagram - Before vs After Temperature Scaling",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )

    for ax, bins, color, label, ece in zip(
        axes,
        [bins_before, bins_after],
        [STYLE["before_color"], STYLE["after_color"]],
        [
            "Before (Uncalibrated)",
            "After (Temperature Scaled)"
        ],
        [ece_before, ece_after]
    ):

        _apply_base_style(
            ax,
            f"{label}\nECE = {ece:.4f}",
            "Confidence",
            "Accuracy"
        )

        if len(bins) > 0:

            centers = np.array([
                (b["bin_lower"] + b["bin_upper"]) / 2
                for b in bins
            ])

            widths = np.array([
                b["bin_upper"] - b["bin_lower"]
                for b in bins
            ])

            accs = np.array([
                b["accuracy"]
                for b in bins
            ])

            confs = np.array([
                b["confidence"]
                for b in bins
            ])

            ax.bar(
                centers,
                accs,
                width=widths * 0.9,
                color=color,
                alpha=0.75,
                label="Accuracy",
                zorder=2
            )

            gap = confs - accs
            positive_gap = np.maximum(gap, 0)

            ax.bar(
                centers,
                positive_gap,
                width=widths * 0.9,
                bottom=accs,
                color="#E74C3C",
                alpha=0.35,
                label="Overconfidence Gap",
                zorder=2
            )

        ax.plot(
            [0, 1],
            [0, 1],
            linestyle="--",
            color=STYLE["perfect_color"],
            linewidth=2,
            label="Perfect Calibration",
            zorder=3
        )

        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        ax.legend(fontsize=10)

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[Plot] Saved -> {save_path}")


# ==========================================================
# Plot 2 - Confidence Histogram
# ==========================================================

def plot_confidence_histogram(
    logits_before,
    logits_after,
    n_bins=20,
    save_path="results/confidence_histogram.png"
):

    import torch.nn.functional as F

    conf_before = (
        F.softmax(
            logits_before,
            dim=1
        )
        .max(dim=1)
        .values
        .numpy()
    )

    conf_after = (
        F.softmax(
            logits_after,
            dim=1
        )
        .max(dim=1)
        .values
        .numpy()
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(14, 5),
        sharey=True
    )

    fig.suptitle(
        "Confidence Distribution - Before vs After Temperature Scaling",
        fontsize=14,
        fontweight="bold",
        y=1.02
    )

    for ax, conf, color, label in zip(
        axes,
        [conf_before, conf_after],
        [STYLE["before_color"], STYLE["after_color"]],
        [
            "Before (Uncalibrated)",
            "After (Temperature Scaled)"
        ]
    ):

        _apply_base_style(
            ax,
            label,
            "Max Confidence",
            "Count"
        )

        ax.hist(
            conf,
            bins=n_bins,
            range=(0, 1),
            color=color,
            alpha=0.8,
            edgecolor="white",
            linewidth=0.5
        )

        ax.axvline(
            conf.mean(),
            color="#333333",
            linestyle="--",
            linewidth=1.5,
            label=f"Mean = {conf.mean():.3f}"
        )

        ax.legend(fontsize=10)

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[Plot] Saved -> {save_path}")


# ==========================================================
# Plot 3 - ECE Comparison
# ==========================================================

def plot_ece_comparison(
    metrics,
    save_path="results/ece_comparison.png"
):

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    _apply_base_style(
        ax,
        "Calibration Error - Before vs After Temperature Scaling",
        "Metric",
        "Error"
    )

    labels = ["ECE", "MCE"]

    x = np.arange(
        len(labels)
    )

    width = 0.35

    bars_before = [
        metrics["ece_before"],
        metrics["mce_before"]
    ]

    bars_after = [
        metrics["ece_after"],
        metrics["mce_after"]
    ]

    ax.bar(
        x - width / 2,
        bars_before,
        width,
        color=STYLE["before_color"],
        alpha=0.85,
        label="Before"
    )

    ax.bar(
        x + width / 2,
        bars_after,
        width,
        color=STYLE["after_color"],
        alpha=0.85,
        label="After"
    )

    for positions, values in [
        (x - width / 2, bars_before),
        (x + width / 2, bars_after)
    ]:

        for xpos, val in zip(
            positions,
            values
        ):

            ax.text(
                xpos,
                val + 0.002,
                f"{val:.4f}",
                ha="center",
                va="bottom",
                fontsize=10,
                fontweight="bold"
            )

    ax.set_xticks(x)
    ax.set_xticklabels(labels)

    ax.set_ylim(
        0,
        max(
            bars_before +
            bars_after
        ) * 1.3
    )

    ax.legend()

    plt.tight_layout()

    os.makedirs(
        os.path.dirname(save_path),
        exist_ok=True
    )

    plt.savefig(
        save_path,
        dpi=150,
        bbox_inches="tight"
    )

    plt.close()

    print(f"[Plot] Saved -> {save_path}")