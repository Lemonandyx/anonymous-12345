#!/usr/bin/env python3
"""Publication-ready throughput comparison across LLM scales."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np


SOURCE_DIR = Path(__file__).resolve().parent
DATA_PATH = SOURCE_DIR.parents[1] / "data" / "figure10_peak_throughput" / "points.csv"
MODEL_SIZES = ["8B", "32B", "70B"]
MODEL_LABELS = [
    "Llama3-8B (1 GPU)",
    "Qwen3-32B (4 GPUs)",
    "Llama3-70B (8 GPUs)",
]
SYSTEMS = [
    "Adaptive-k",
    "Self-RAG (k=10)",
    "Adaptive-k + Self-RAG",
    "HedraRAG",
    "HarmoRAG",
]

# Preserve the system mapping used by the existing nprobe/top-k figure.
COLORS = {
    "Adaptive-k": "#80A2C5",
    "Self-RAG (k=10)": "#B79AC8",
    "Adaptive-k + Self-RAG": "#E9A6A1",
    "HedraRAG": "#A3CFA2",
    "HarmoRAG": "#F7BD81",
}
BAR_EDGE = "#4D4D4D"


def load_data(path: Path) -> dict[tuple[str, str], float]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    values: dict[tuple[str, str], float] = {}
    for row in rows:
        key = (row["model_size"], row["system"])
        value = float(row["throughput_req_s"])
        if key in values or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Duplicate or invalid measurement: {key}")
        values[key] = value

    expected = {(model, system) for model in MODEL_SIZES for system in SYSTEMS}
    if set(values) != expected:
        raise ValueError(
            f"Unexpected table entries: missing={expected - set(values)}, "
            f"extra={set(values) - expected}"
        )
    return values


def main() -> None:
    values = load_data(DATA_PATH)

    matplotlib.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 8.0,
            "axes.labelsize": 9.0,
            "axes.titlesize": 10.0,
            "xtick.labelsize": 8.2,
            "ytick.labelsize": 8.0,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "legend.frameon": False,
        }
    )

    fig, ax = plt.subplots(figsize=(6.0, 3.05))
    x = np.arange(len(MODEL_SIZES), dtype=float) * 0.62
    total_width = 0.46
    bar_width = total_width / len(SYSTEMS)

    for index, system in enumerate(SYSTEMS):
        offset = (index - (len(SYSTEMS) - 1) / 2) * bar_width
        heights = [values[(model, system)] for model in MODEL_SIZES]
        bars = ax.bar(
            x + offset,
            heights,
            width=bar_width,
            color=COLORS[system],
            edgecolor=BAR_EDGE,
            linewidth=0.45,
            zorder=3,
        )
        for bar, height in zip(bars, heights):
            ax.annotate(
                f"{height:.2f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 2.0),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=6.6,
                color="#303840",
                clip_on=False,
            )

    ax.set_xticks(x, MODEL_LABELS)
    ax.set_xlabel("LLM Backbones", labelpad=3.0)
    ax.set_ylabel("Peak Throughput (req/s)", labelpad=3.0)
    ax.set_ylim(0, 14.0)
    ax.set_yticks(np.arange(0, 14.1, 2.0))
    ax.yaxis.grid(True, color="#D7E0E8", linestyle="--", linewidth=0.6, zorder=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)

    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#303840")
    ax.tick_params(axis="both", which="both", length=2.0, width=0.65, pad=1.8)

    legend_handles = [
        Patch(
            facecolor=COLORS[system],
            edgecolor=BAR_EDGE,
            linewidth=0.45,
            label=(
                "Self-RAG"
                if system == "Self-RAG (k=10)"
                else "HarmoRAG"
                if system == "HarmoRAG"
                else system
            ),
        )
        for system in SYSTEMS
    ]
    fig.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.54, 0.925),
        ncol=5,
        columnspacing=1.35,
        handlelength=1.25,
        handleheight=0.75,
        handletextpad=0.42,
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.085, right=0.992, bottom=0.19, top=0.84)

    output_dir = Path("output/figure10")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_base = output_dir / "Throughput-Across-Model-Sizes"
    fig.savefig(output_base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(
        output_base.with_suffix(".png"),
        dpi=360,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    plt.close(fig)
    print(output_base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
