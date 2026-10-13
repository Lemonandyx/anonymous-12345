#!/usr/bin/env python3
"""Replot the audited Top-k=1 figure with larger panel titles."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt


SYSTEMS = ["Faiss", "HedraRAG", "VectoLiteRAG", "HarmoRAG"]
NPROBES = [128, 256, 512]
RATES = [1, 2, 4, 6, 8, 10, 12, 16, 20, 24, 28, 32]
COLORS = {
    "Faiss": "#8C8C8C",
    "HedraRAG": "#3A9D5D",
    "VectoLiteRAG": "#477DB3",
    "HarmoRAG": "#F57C00",
}
MARKERS = {"Faiss": "s", "HedraRAG": "o", "VectoLiteRAG": "D", "HarmoRAG": "^"}


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    rows = list(csv.DictReader(args.input_csv.open(newline="", encoding="utf-8")))
    expected = {(n, s, q) for n in NPROBES for s in SYSTEMS for q in RATES}
    observed = {
        (int(row["nprobe"]), row["system"], int(row["offered_rate_req_s"]))
        for row in rows
    }
    if len(rows) != 144 or observed != expected:
        raise RuntimeError("input is not the audited 144-point Top-k=1 grid")
    if any(int(row["topk"]) != 1 or int(row["sample_count"]) != 512 for row in rows):
        raise RuntimeError("Top-k/sample-count contract failed")

    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans", "Liberation Sans"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 8,
            "axes.linewidth": 0.8,
            "axes.spines.top": True,
            "axes.spines.right": True,
            "legend.frameon": False,
        }
    )

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.45), sharex=True, sharey=True)
    for panel, (ax, nprobe) in enumerate(zip(axes, NPROBES), start=1):
        for system in SYSTEMS:
            subset = sorted(
                (
                    row
                    for row in rows
                    if int(row["nprobe"]) == nprobe and row["system"] == system
                ),
                key=lambda row: int(row["offered_rate_req_s"]),
            )
            ax.plot(
                [int(row["offered_rate_req_s"]) for row in subset],
                [float(row["active_retrieval_overlap_pct"]) for row in subset],
                color=COLORS[system],
                marker=MARKERS[system],
                markersize=3.0,
                markeredgewidth=0,
                linewidth=1.35,
                label=system,
            )
        ax.set_title(
            f"({chr(96 + panel)}) nprobe={nprobe}",
            fontsize=9.5,
            fontweight="bold",
            pad=2.0,
        )
        ax.set_xlim(0, 33)
        ax.set_ylim(0, 102)
        ax.set_xticks([0, 8, 16, 24, 32])
        ax.set_yticks([0, 20, 40, 60, 80, 100])
        ax.grid(True, color="#D7E2EF", linewidth=0.65, linestyle=(0, (3, 3)))
        ax.tick_params(direction="out", width=0.8, length=3)
        ax.set_xlabel("Request rate (req/s)")

    axes[0].set_ylabel("Active Retrieval Overlap (%)")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="upper center",
        # Compensate for the one-point title-size increase so the visual
        # legend-to-title separation remains unchanged.
        bbox_to_anchor=(0.5, 0.9580),
        ncol=4,
        handlelength=2.0,
        columnspacing=1.25,
        handletextpad=0.45,
    )
    # Raise the panels slightly and lower the legend to reduce the former gap.
    fig.subplots_adjust(left=0.085, right=0.995, bottom=0.20, top=0.805, wspace=0.12)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    stem = args.output_dir / "figure13_topk1_three_nprobe_four_systems_layout_v5"
    outputs = []
    for suffix, options in (("svg", {}), ("pdf", {}), ("png", {"dpi": 600})):
        path = stem.with_suffix(f".{suffix}")
        fig.savefig(path, bbox_inches="tight", **options)
        outputs.append(str(path))
    plt.close(fig)

    manifest = {
        "input_csv": str(args.input_csv),
        "input_csv_sha256": sha256(args.input_csv),
        "point_count": 144,
        "topk": 1,
        "sample_count_each": 512,
        "data_changed": False,
        "changes": [
            "all subplot borders enabled",
            "panel titles moved four points closer to the subplot borders",
            "panel-title font size increased from 8.5 to 9.5 points",
            "legend position compensated for the title-size increase to preserve legend-title spacing",
            "Y-axis title words capitalized",
        ],
        "outputs": outputs,
    }
    (args.output_dir / "layout_v5_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
