#!/usr/bin/env python3
"""Two-panel publication figure from the two supplied NQ/8B mean-E2E tables.

Panel a varies nprobe at top-k=10. Panel b varies top-k at nprobe=128.
Each bar is normalized to HarmoRAG within the same category and panel.
The supplied panels are distinct sweeps, not pooled measurements.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Arial", "DejaVu Sans", "Liberation Sans"]
plt.rcParams["svg.fonttype"] = "none"

SOURCE_DIR = Path(__file__).resolve().parent
DATA_DIR = SOURCE_DIR.parents[1] / "data" / "figure09_retrieval_configuration"
# Four hues are taken directly from the supplied pastel references. The
# references contain no red, so the combination baseline uses soft coral.
COLORS = {
    "Adaptive-k": "#80A2C5",
    "Self-RAG (k=10)": "#B79AC8",
    "Adaptive-k + Self-RAG": "#E9A6A1",
    "HedraRAG": "#A3CFA2",
    "HarmoRAG": "#F7BD81",
}
BAR_EDGE = "#4D4D4D"
ORDER_A = list(COLORS)
ORDER_B = ["Self-RAG (k=10)", "HedraRAG", "HarmoRAG"]
NP_VALUES = [128, 256, 512]
TOPK_VALUES = [1, 5, 10]


def load_table(path: Path, category_field: str, categories: list[int], systems: list[str]):
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    data = {}
    for row in rows:
        system = row["system"]
        category = int(row[category_field])
        value = float(row["mean_e2e_s"])
        key = (system, category)
        if key in data or not np.isfinite(value) or value <= 0:
            raise ValueError(f"Duplicate or invalid value: {key}")
        data[key] = value
    expected = {(system, category) for system in systems for category in categories}
    if set(data) != expected:
        raise ValueError(f"Unexpected table entries: missing={expected-set(data)}, extra={set(data)-expected}")
    return data


def normalize(data: dict[tuple[str, int], float], categories: list[int], systems: list[str]):
    return {
        (system, category): data[(system, category)] / data[("HarmoRAG", category)]
        for system in systems
        for category in categories
    }


def draw_panel(ax, data, categories, systems, *, panel_title: str, xlabel: str, ylabel: str | None, ymax: float, step: float):
    x = np.arange(len(categories), dtype=float)
    total_width = 0.78
    bar_width = total_width / len(systems)
    for index, system in enumerate(systems):
        offset = (index - (len(systems) - 1) / 2) * bar_width
        values = [data[(system, category)] for category in categories]
        ax.bar(
            x + offset,
            values,
            width=bar_width,
            color=COLORS[system],
            edgecolor=BAR_EDGE,
            linewidth=0.40,
            zorder=3,
        )
    ax.set_xticks(x, [str(value) for value in categories])
    ax.set_xlabel(xlabel, labelpad=2.0)
    if ylabel:
        ax.set_ylabel(ylabel, labelpad=2.0)
    ax.set_title(panel_title, fontweight="bold", pad=5.0)
    ax.set_ylim(0, ymax)
    ax.set_yticks(np.arange(0, ymax + 0.01, step))
    ax.axhline(1.0, color="#828C96", linewidth=0.65, linestyle=(0, (2, 2)), zorder=2)
    ax.yaxis.grid(True, color="#D7E0E8", linestyle="--", linewidth=0.6, zorder=1)
    ax.xaxis.grid(False)
    ax.set_axisbelow(True)
    for spine in ax.spines.values():
        spine.set_linewidth(0.65)
        spine.set_color("#303840")
    ax.tick_params(axis="both", which="both", length=2.0, width=0.65, pad=1.5)


def write_normalized_csv(path: Path, a, b, a_raw, b_raw):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["panel", "fixed_setting", "varying_parameter", "value", "system", "mean_e2e_s", "harmorag_mean_e2e_s", "normalized_latency"],
        )
        writer.writeheader()
        for panel, fixed, param, categories, systems, normalized, raw in (
            ("a", "topk=10", "nprobe", NP_VALUES, ORDER_A, a, a_raw),
            ("b", "nprobe=128", "topk", TOPK_VALUES, ORDER_B, b, b_raw),
        ):
            for category in categories:
                for system in systems:
                    writer.writerow({
                        "panel": panel,
                        "fixed_setting": fixed,
                        "varying_parameter": param,
                        "value": category,
                        "system": system,
                        "mean_e2e_s": f"{raw[(system, category)]:.6f}",
                        "harmorag_mean_e2e_s": f"{raw[('HarmoRAG', category)]:.6f}",
                        "normalized_latency": f"{normalized[(system, category)]:.6f}",
                    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("output/figure09"))
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)

    a_raw = load_table(DATA_DIR / "nprobe_topk10.csv", "nprobe", NP_VALUES, ORDER_A)
    b_raw = load_table(DATA_DIR / "topk_nprobe128.csv", "topk", TOPK_VALUES, ORDER_B)
    a = normalize(a_raw, NP_VALUES, ORDER_A)
    b = normalize(b_raw, TOPK_VALUES, ORDER_B)
    write_normalized_csv(out / "normalized_latency_source.csv", a, b, a_raw, b_raw)

    matplotlib.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7.7,
        "axes.labelsize": 8.5,
        "axes.titlesize": 9.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "legend.frameon": False,
    })

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.85))
    draw_panel(
        axes[0], a, NP_VALUES, ORDER_A,
        panel_title="(a) nprobe (Top-k=10)", xlabel="nprobe", ylabel="Normalized Latency", ymax=14.0, step=2.0,
    )
    draw_panel(
        axes[1], b, TOPK_VALUES, ORDER_B,
        panel_title="(b) Top-k (nprobe=128)", xlabel="Top-k", ylabel=None, ymax=4.0, step=1.0,
    )

    legend = [
        Patch(
            facecolor=COLORS[system],
            edgecolor=BAR_EDGE,
            linewidth=0.40,
            label="Self-RAG" if system == "Self-RAG (k=10)" else system,
        )
        for system in ORDER_A
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.985),
        ncol=5,
        columnspacing=1.45,
        handlelength=1.25,
        handleheight=0.75,
        handletextpad=0.45,
        fontsize=8.0,
    )
    fig.subplots_adjust(left=0.092, right=0.992, bottom=0.19, top=0.83, wspace=0.08)

    base = out / "NQ-8B-Normalized-E2E-nprobe-topk"
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.04)
    fig.savefig(base.with_suffix(".png"), dpi=360, bbox_inches="tight", pad_inches=0.04)
    plt.close(fig)

    cross_panel_same_setting = {
        system: (a_raw[(system, 128)], b_raw[(system, 10)])
        for system in ORDER_B
    }
    mismatched = {name: values for name, values in cross_panel_same_setting.items() if abs(values[0] - values[1]) > 1e-6}
    if mismatched:
        print("NOTE: These are separate supplied sweeps. Matching-setting entries differ:", mismatched)
    print(base.with_suffix(".pdf"))


if __name__ == "__main__":
    main()
