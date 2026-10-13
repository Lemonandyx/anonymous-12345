#!/usr/bin/env python3
"""Plot Figure 12 from an external aggregate CSV."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Patch
from matplotlib.ticker import FormatStrFormatter


SYSTEMS = ["Top-5", "Top-10", "Adaptive-k", "HarmoRAG-5", "HarmoRAG-10"]
AXIS_SYSTEMS = [
    "Top-5",
    "Top-10",
    r"Adaptive-$k$",
    "HarmoRAG\n(k=5)",
    "HarmoRAG\n(k=10)",
]
COLORS = ["#A9C1D5", "#7FA5C7", "#A5CDA3", "#F5BE7D", "#E49A50"]
EDGE = "#62676D"

FIELDS = {
    "context_docs",
    "retrieval_s",
    "generation_s",
    "e2e_s",
    "f1_pct",
    "acc_pct",
    "prefix_hit_pct",
}


def load_data(path: Path) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = {row["system"]: row for row in csv.DictReader(handle)}
    if set(rows) != set(SYSTEMS):
        raise ValueError(f"system mismatch: expected={SYSTEMS}, found={sorted(rows)}")
    missing = FIELDS - set(next(iter(rows.values())))
    if missing:
        raise ValueError(f"missing columns: {sorted(missing)}")
    return {
        field: np.asarray([float(rows[system][field]) for system in SYSTEMS], dtype=float)
        for field in FIELDS
    }


def style_axis(ax):
    ax.set_axisbelow(True)
    ax.grid(axis="y", linestyle="--", linewidth=0.65, color="#C9CED4", alpha=0.9)
    for spine in ax.spines.values():
        spine.set_color("#50555A")
        spine.set_linewidth(0.85)
    ax.tick_params(axis="both", labelsize=10.8, width=0.8, length=3.5, pad=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/figure12"))
    args = parser.parse_args()
    data = load_data(args.input_csv)
    context = data["context_docs"]
    retrieval = data["retrieval_s"]
    generation = data["generation_s"]
    e2e = data["e2e_s"]
    f1 = data["f1_pct"]
    acc = data["acc_pct"]
    prefix_hit = data["prefix_hit_pct"]
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 11,
            "axes.titlesize": 16,
            "axes.titleweight": "bold",
            "axes.labelsize": 14,
            "axes.linewidth": 0.85,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    # Preserve the 1350:432 aspect ratio of the paper figure.
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 4.32))
    ax_a, ax_b, ax_c = axes

    # (a) Context size.
    x = np.arange(len(SYSTEMS))
    bars = ax_a.bar(
        x,
        context,
        width=0.58,
        color=COLORS,
        edgecolor=EDGE,
        linewidth=0.75,
        zorder=3,
    )
    ax_a.set_title("(a) Context Size", pad=5)
    ax_a.set_ylabel("Mean Documents/query")
    ax_a.set_xticks(x, AXIS_SYSTEMS)
    ax_a.set_ylim(0, 12.5)
    ax_a.set_yticks(np.arange(0, 12.51, 2.5))
    ax_a.yaxis.set_major_formatter(FormatStrFormatter("%g"))
    style_axis(ax_a)
    ax_a.tick_params(axis="x", labelsize=8.8)
    for bar, value in zip(bars, context):
        ax_a.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.18,
            f"{value:.2f}",
            ha="center",
            va="bottom",
            fontsize=11.5,
        )

    # (b) Latency breakdown.
    width = 0.25
    ax_b.bar(
        x - width,
        retrieval,
        width,
        label="Retrieval",
        color="#86A9C8",
        edgecolor=EDGE,
        linewidth=0.7,
        zorder=3,
    )
    ax_b.bar(
        x,
        generation,
        width,
        label="Generation",
        color="#A9CFA6",
        edgecolor=EDGE,
        linewidth=0.7,
        zorder=3,
    )
    e2e_bars = ax_b.bar(
        x + width,
        e2e,
        width,
        label="E2E",
        color="#F2BC7A",
        edgecolor=EDGE,
        linewidth=0.7,
        zorder=3,
    )
    ax_b.set_title("(b) Latency Breakdown", pad=5)
    ax_b.set_ylabel("Latency (s/query)")
    ax_b.set_xticks(x, AXIS_SYSTEMS)
    ax_b.set_ylim(0, 0.8)
    ax_b.set_yticks(np.arange(0, 0.801, 0.16))
    ax_b.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))
    style_axis(ax_b)
    ax_b.tick_params(axis="x", labelsize=8.8)
    ax_b.legend(
        loc="upper left",
        bbox_to_anchor=(0.01, 0.995),
        borderaxespad=0,
        fontsize=10.2,
        handlelength=0.8,
        handletextpad=0.35,
        labelspacing=0.15,
    )
    for bar, value in zip(e2e_bars, e2e):
        ax_b.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.018,
            f"{value:.3f}",
            ha="center",
            va="bottom",
            fontsize=11.2,
        )

    # (c) Answer quality and vLLM token-level prefix-cache hit rate.
    metrics = ["F1", "ACC", "Prefix Hit Rate"]
    metric_values = np.vstack([f1, acc, prefix_hit])
    metric_x = np.arange(len(metrics))
    quality_width = 0.145
    offsets = (np.arange(len(SYSTEMS)) - (len(SYSTEMS) - 1) / 2) * quality_width
    for system_idx, (system, color) in enumerate(zip(SYSTEMS, COLORS)):
        ax_c.bar(
            metric_x + offsets[system_idx],
            metric_values[:, system_idx],
            quality_width,
            color=color,
            edgecolor=EDGE,
            linewidth=0.7,
            zorder=3,
        )
    ax_c.set_title("(c) Quality & Prefix Hit Rate", pad=5)
    ax_c.set_ylabel("Score / Hit Rate (%)")
    ax_c.set_xticks(metric_x, metrics)
    ax_c.set_xlim(-0.55, len(metrics) - 0.45)
    ax_c.set_ylim(0, 60)
    ax_c.set_yticks(np.arange(0, 61, 20))
    style_axis(ax_c)

    handles = [
        Patch(facecolor=color, edgecolor=EDGE, linewidth=0.7, label=system)
        for system, color in zip(SYSTEMS, COLORS)
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=5,
        fontsize=13.0,
        handlelength=1.05,
        handletextpad=0.35,
        columnspacing=1.4,
    )

    fig.subplots_adjust(left=0.055, right=0.993, bottom=0.20, top=0.82, wspace=0.18)
    png_path = args.output_dir / "Figure12-Generation-Performance-QPS1-Prefix-Hit.png"
    pdf_path = args.output_dir / "Figure12-Generation-Performance-QPS1-Prefix-Hit.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(png_path)
    print(pdf_path)


if __name__ == "__main__":
    main()
