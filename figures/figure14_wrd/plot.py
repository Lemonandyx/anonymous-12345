#!/usr/bin/env python3
"""Render the six-panel WRD ablation and sensitivity figure from released data."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "figure14_wrd" / "points.csv"
PANELS = ("a", "b", "c", "d", "e", "f")
DATASET_STYLE = {"NQ": "#4C78A8", "HotpotQA": "#F58518"}


def load(path: Path):
    grouped = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            grouped[(row["panel"], row["dataset"], row["metric"])].append(row)
    for key, rows in grouped.items():
        if not rows:
            raise ValueError(f"empty series: {key}")
    return grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-stem", type=Path, default=Path("output/figure14/Figure14-WRD"))
    args = parser.parse_args()
    grouped = load(args.input_csv)
    plt.rcParams.update({"font.family": ["Arial", "DejaVu Sans", "sans-serif"], "font.size": 7.5,
                         "axes.titlesize": 8.4, "axes.labelsize": 8.0, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "legend.frameon": False})
    fig, axes = plt.subplots(2, 3, figsize=(9.0, 4.4))
    legend_handles = []
    for panel, axis in zip(PANELS, axes.flat):
        sample = next(rows for (p, _, _), rows in grouped.items() if p == panel)
        title = sample[0]["panel_title"]
        parameter = sample[0]["parameter"]
        twin = axis.twinx()
        for dataset, color in DATASET_STYLE.items():
            recall = grouped[(panel, dataset, "recall_at_5_percent")]
            latency = grouped[(panel, dataset, "latency_as_plotted")]
            labels = [row["parameter_value"] for row in recall]
            x = list(range(len(labels)))
            recall_line = axis.plot(x, [float(row["value"]) for row in recall], color=color,
                                    marker="o", linewidth=1.4, markersize=3.3,
                                    label=f"{dataset} Recall@5", zorder=3)[0]
            latency_line = twin.plot(x, [float(row["value"]) for row in latency], color=color,
                                     marker="s", linestyle="--", linewidth=1.2, markersize=3.0,
                                     label=f"{dataset} Latency", zorder=3)[0]
            if panel == "a":
                legend_handles.extend((recall_line, latency_line))
            axis.set_xticks(x, labels, rotation=0)
        axis.set_title(f"({panel}) {title}", fontweight="bold", pad=3)
        axis.set_xlabel(parameter.replace("epsilon", r"$\epsilon$").replace("mu", r"$\mu$"))
        if panel in {"a", "d"}:
            axis.set_ylabel("Recall@5 (%)")
        if panel in {"c", "f"}:
            twin.set_ylabel("Latency (ms/query)")
        axis.grid(True, color="#D5E0EB", linestyle=(0, (2.5, 2.5)), linewidth=0.55)
        for current in (axis, twin):
            current.tick_params(length=2.5, width=0.7, pad=1.5)
            for spine in current.spines.values():
                spine.set_linewidth(0.7)
    fig.legend(legend_handles, [h.get_label() for h in legend_handles], loc="upper center",
               bbox_to_anchor=(0.5, 0.995), ncol=4, handlelength=2.2, columnspacing=1.2)
    fig.subplots_adjust(left=0.065, right=0.945, bottom=0.11, top=0.86, wspace=0.36, hspace=0.42)
    args.output_stem.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "svg", "png"):
        fig.savefig(args.output_stem.with_suffix(f".{suffix}"), dpi=300, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


if __name__ == "__main__":
    main()
