#!/usr/bin/env python3
"""Render the paper's matched-recall IVF (Figure 11) or HNSW (Figure 17) plot."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator


DATASETS = ("sift100m", "deep100m", "gist1m")
DATASET_LABELS = {
    "sift100m": "SIFT100M",
    "deep100m": "DEEP100M",
    "gist1m": "GIST1M",
}
METHODS = ("LAET", "DARTH", "Terminus", "HarmoRAG")
STYLE = {
    "LAET": ("#4C78A8", "o"),
    "DARTH": ("#54A24B", "s"),
    "Terminus": ("#9C6FB6", "D"),
    "HarmoRAG": ("#F58518", "^"),
}


def load_rows(path: Path, index: str) -> list[dict[str, object]]:
    with path.open(newline="", encoding="utf-8") as handle:
        raw = list(csv.DictReader(handle))
    rows: list[dict[str, object]] = []
    for row in raw:
        if row.get("index") and row["index"].upper() != index:
            continue
        measured_key = "recall_at_100" if "recall_at_100" in row else "measured_recall_at_100"
        rows.append(
            {
                "dataset": row["dataset"],
                "method": row["method"],
                "target": float(row["target_recall_at_100"]),
                "measured": float(row[measured_key]),
                "latency_ms": float(row["latency_ms"]),
            }
        )
    expected = {(d, m, r) for d in DATASETS for m in METHODS for r in (0.9, 0.95, 0.99)}
    actual = {(r["dataset"], r["method"], r["target"]) for r in rows}
    if actual != expected:
        raise ValueError(f"incomplete matched-recall grid: missing={expected - actual}, extra={actual - expected}")
    for row in rows:
        if row["measured"] + 1e-12 < row["target"]:
            raise ValueError(f"quality target missed: {row}")
    return rows


def render(rows: list[dict[str, object]], output_stem: Path) -> None:
    plt.rcParams.update(
        {
            "font.family": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 9,
            "axes.titlesize": 10.5,
            "axes.labelsize": 10,
            "axes.linewidth": 0.9,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "legend.fontsize": 9,
            "legend.frameon": False,
            "pdf.fonttype": 42,
            "svg.fonttype": "none",
        }
    )
    fig, axes = plt.subplots(1, 3, figsize=(8.35, 2.55), sharex=True)
    for panel, (axis, dataset) in enumerate(zip(axes, DATASETS)):
        maximum = 0.0
        for method in METHODS:
            series = sorted(
                (row for row in rows if row["dataset"] == dataset and row["method"] == method),
                key=lambda row: row["target"],
            )
            x = [row["target"] for row in series]
            y = [row["latency_ms"] for row in series]
            maximum = max(maximum, max(y))
            color, marker = STYLE[method]
            axis.plot(x, y, label=method, color=color, marker=marker, linewidth=1.45,
                      markersize=4.6, markeredgewidth=0.55, zorder=3)
        axis.set_title(f"({chr(97 + panel)}) {DATASET_LABELS[dataset]}", fontweight="bold", pad=3.5)
        axis.set_xlim(0.89, 1.0)
        axis.set_xticks((0.90, 0.92, 0.94, 0.96, 0.98, 1.00))
        axis.set_xlabel("Recall@100", labelpad=2)
        axis.set_ylim(0.0, maximum * 1.08)
        axis.yaxis.set_major_locator(MaxNLocator(nbins=5, min_n_ticks=4))
        axis.grid(True, color="#B7CBE2", linestyle=(0, (2.5, 2.5)), linewidth=0.7, zorder=0)
        for spine in axis.spines.values():
            spine.set_color("#3F444A")
            spine.set_linewidth(0.85)
        axis.tick_params(direction="out", length=3.0, width=0.8, pad=2)
    axes[0].set_ylabel("Latency (ms/query)", labelpad=3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(0.5, 0.94), ncol=4,
               handlelength=2.0, handletextpad=0.45, columnspacing=1.35)
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.22, top=0.79, wspace=0.20)
    output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", pad_inches=0.025)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", choices=("IVF", "HNSW"), required=True)
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-stem", type=Path, required=True)
    args = parser.parse_args()
    render(load_rows(args.input_csv, args.index), args.output_stem)


if __name__ == "__main__":
    main()
