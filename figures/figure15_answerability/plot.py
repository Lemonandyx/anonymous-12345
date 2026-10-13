#!/usr/bin/env python3
"""Draw the updated two-panel answerability-gate figure.

Both panels read audited strict single-request measurements: panel (a) varies
the gate components and panel (b) varies the confidence margin.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT.parents[1] / "data" / "figure15_answerability"
SUMMARY = DATA_DIR / "components.csv"
MARGIN_SUMMARY = DATA_DIR / "margins.csv"

ORDER = [
    "ungated",
    "confidence_first",
    "confidence_structure",
    "full_gate",
    "full_context",
]
LABELS = {
    "ungated": "Ungated",
    "confidence_first": "+ Confidence",
    "confidence_structure": "+ Structure",
    "full_gate": "Full Gate",
    "full_context": "Full Context",
}

BLUE = "#4C78A8"
ORANGE = "#F58518"
NAVY = "#17263A"
GRID = "#D7DEE7"


def load_component_data() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with SUMMARY.open(newline="") as handle:
        rows = {row["mode"]: row for row in csv.DictReader(handle)}
    missing = [mode for mode in ORDER if mode not in rows]
    if missing:
        raise ValueError(f"Missing gate-component rows: {missing}")
    acc = np.array([float(rows[m]["acc_pct_mean"]) for m in ORDER])
    f1 = np.array([float(rows[m]["f1_pct_mean"]) for m in ORDER])
    latency = np.array([float(rows[m]["mean_latency_s_mean"]) for m in ORDER])
    return acc, f1, latency


def load_margin_data() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    with MARGIN_SUMMARY.open(newline="") as handle:
        rows = {int(row["gamma"]): row for row in csv.DictReader(handle)}
    gammas = np.arange(0, 33, 4)
    missing = [int(gamma) for gamma in gammas if int(gamma) not in rows]
    if missing:
        raise ValueError(f"Missing gate-margin rows: {missing}")
    acc = np.array([float(rows[int(g)]["acc_pct_mean"]) for g in gammas])
    f1 = np.array([float(rows[int(g)]["f1_pct_mean"]) for g in gammas])
    latency = np.array(
        [float(rows[int(g)]["mean_latency_s_mean"]) for g in gammas]
    )
    return gammas, acc, f1, latency


def style_quality_axis(axis: plt.Axes) -> None:
    axis.grid(
        axis="both",
        color=GRID,
        linestyle=(0, (3, 3)),
        linewidth=0.8,
        alpha=0.95,
        zorder=0,
    )
    axis.tick_params(axis="both", direction="out", length=4.0, width=1.0)
    for spine in axis.spines.values():
        spine.set_color("#222222")
        spine.set_linewidth(1.05)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--detail-inset",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="include the post-confidence detail inset in panel (a)",
    )
    parser.add_argument(
        "--output-stem",
        default="Figure15-Answerability-Gate",
        help="basename written under the figures directory",
    )
    args = parser.parse_args()
    output = Path("output/figure15") / args.output_stem

    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 10.5,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12.0,
            "axes.linewidth": 1.05,
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 10.0,
            "legend.fontsize": 11.5,
            "legend.frameon": False,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
        }
    )

    component_acc, component_f1, component_latency = load_component_data()
    gamma, margin_acc, margin_f1, margin_latency = load_margin_data()

    fig, axes = plt.subplots(1, 2, figsize=(10.8, 3.42), constrained_layout=False)

    # (a) Audited strict single-request component ablation.
    left = axes[0]
    x = np.arange(len(ORDER))
    quality_acc = left.plot(
        x,
        component_acc,
        color=BLUE,
        marker="o",
        linewidth=2.15,
        markersize=4.8,
        label="ACC",
        zorder=3,
    )[0]
    quality_f1 = left.plot(
        x,
        component_f1,
        color=ORANGE,
        marker="s",
        linewidth=2.15,
        markersize=4.6,
        label="F1",
        zorder=3,
    )[0]
    left.set_title("(a) Gate component ablation", fontweight="bold", pad=6)
    left.set_ylabel("Quality (%)")
    left.set_xlabel("Gate configuration", labelpad=4)
    left.set_ylim(20, 60)
    left.set_yticks([20, 30, 40, 50, 60])
    left.set_xticks(x)
    left.set_xticklabels([LABELS[m] for m in ORDER], fontsize=8.5)
    style_quality_axis(left)

    left_latency = left.twinx()
    latency_line = left_latency.plot(
        x,
        component_latency,
        color=NAVY,
        marker="o",
        linewidth=1.95,
        markersize=4.2,
        linestyle=(0, (4, 3)),
        label="Mean E2E Latency",
        zorder=4,
    )[0]
    left_latency.set_ylabel("Latency (s/query)", labelpad=6)
    left_latency.set_ylim(0, 0.8)
    left_latency.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    left_latency.tick_params(axis="y", direction="out", length=4.0, width=1.0)
    left_latency.spines["right"].set_color("#222222")
    left_latency.spines["right"].set_linewidth(1.05)

    if args.detail_inset:
        # Preserve the full-range comparison above while exposing the smaller
        # quality differences after confidence is enabled. Placing the inset
        # in the otherwise empty lower portion avoids obscuring main data.
        detail = left.inset_axes([0.385, 0.075, 0.565, 0.34], zorder=8)
        detail.plot(
            x[1:],
            component_acc[1:],
            color=BLUE,
            marker="o",
            linewidth=1.25,
            markersize=3.0,
        )
        detail.plot(
            x[1:],
            component_f1[1:],
            color=ORANGE,
            marker="s",
            linewidth=1.25,
            markersize=2.9,
        )
        detail.set_xlim(0.85, float(x[-1]) + 0.15)
        detail.set_ylim(49.2, 54.5)
        detail.set_yticks([50, 52, 54])
        detail.set_xticks(x[1:])
        detail.set_xticklabels(
            ["Conf.", "Struct.", "Gate", "Context"], fontsize=5.8
        )
        detail.tick_params(axis="y", labelsize=6.0, length=2.2, width=0.65)
        detail.tick_params(axis="x", length=2.2, width=0.65, pad=1.5)
        detail.grid(
            axis="y",
            color=GRID,
            linestyle=(0, (2, 2)),
            linewidth=0.5,
            alpha=0.9,
        )
        detail.set_facecolor((1.0, 1.0, 1.0, 0.97))
        detail.set_title("Post-confidence detail", fontsize=6.8, pad=2.0)
        for spine in detail.spines.values():
            spine.set_color("#7A7A7A")
            spine.set_linewidth(0.65)

    # (b) Gate-margin sensitivity under the same strict single-request protocol.
    right = axes[1]
    right.plot(
        gamma,
        margin_acc,
        color=BLUE,
        marker="o",
        linewidth=2.15,
        markersize=4.8,
        zorder=3,
    )
    right.plot(
        gamma,
        margin_f1,
        color=ORANGE,
        marker="s",
        linewidth=2.15,
        markersize=4.6,
        zorder=3,
    )
    right.axvline(16, color="#666666", linewidth=1.2, linestyle=(0, (4, 3)))
    right.set_title(r"(b) Gate margin $\gamma$", fontweight="bold", pad=6)
    right.set_ylabel("Quality (%)")
    right.set_xlabel(r"Gate margin $\gamma$", labelpad=4)
    right.set_xlim(0, 32)
    right.set_xticks(gamma)
    right.set_ylim(45, 55)
    right.set_yticks([45, 47.5, 50, 52.5, 55])
    style_quality_axis(right)

    right_latency = right.twinx()
    right_latency.plot(
        gamma,
        margin_latency,
        color=NAVY,
        marker="o",
        linewidth=1.95,
        markersize=4.2,
        linestyle=(0, (4, 3)),
        zorder=4,
    )
    right_latency.set_ylabel("Latency (s/query)", labelpad=6)
    right_latency.set_ylim(0, 0.8)
    right_latency.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8])
    right_latency.tick_params(axis="y", direction="out", length=4.0, width=1.0)
    right_latency.spines["right"].set_color("#222222")
    right_latency.spines["right"].set_linewidth(1.05)

    fig.legend(
        (quality_acc, quality_f1, latency_line),
        ("ACC", "F1", "Mean E2E Latency"),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        columnspacing=2.1,
        handlelength=2.8,
        handletextpad=0.55,
    )
    fig.subplots_adjust(left=0.065, right=0.935, bottom=0.20, top=0.78, wspace=0.31)

    output.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(
            output.with_suffix(f".{suffix}"),
            dpi=300,
            bbox_inches="tight",
            pad_inches=0.04,
        )
    plt.close(fig)


if __name__ == "__main__":
    main()
