#!/usr/bin/env python3
"""Redraw the original retrieval panels with maximum-split measurements."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DATASETS = ("NQ", "HotpotQA", "SQuAD")
PREFIX = {"NQ": "nq", "HotpotQA": "hotpotqa", "SQuAD": "squad"}

# Pastel colors shared with the current HarmoRAG evaluation figures.
COLORS = {"NQ": "#80A2C5", "HotpotQA": "#E39A59", "SQuAD": "#A3CFA2"}
INK = "#25313C"
GRID = "#87919B"
BASELINE_NPROBES = [1, 2, 4, 8, 16, 32]
SCAN_NPROBES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]


def load_retention_matrix(path: Path) -> np.ndarray:
    """Load reference_nprobe,evaluation_nprobe,retention_pct rows."""
    values: dict[tuple[int, int], float] = {}
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            key = (int(row["reference_nprobe"]), int(row["evaluation_nprobe"]))
            if key in values:
                raise ValueError(f"duplicate retention point: {key}")
            values[key] = float(row["retention_pct"])
    expected = {(reference, evaluation) for reference in BASELINE_NPROBES for evaluation in SCAN_NPROBES}
    if set(values) != expected:
        raise ValueError(f"retention grid mismatch: missing={expected - set(values)}, extra={set(values) - expected}")
    return np.asarray(
        [[values[(reference, evaluation)] for evaluation in SCAN_NPROBES] for reference in BASELINE_NPROBES],
        dtype=np.float64,
    )


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def pchip_endpoint_slope(h0: float, h1: float, d0: float, d1: float) -> float:
    slope = ((2.0 * h0 + h1) * d0 - h0 * d1) / (h0 + h1)
    if slope * d0 <= 0.0:
        return 0.0
    if d0 * d1 < 0.0 and abs(slope) > abs(3.0 * d0):
        return 3.0 * d0
    return slope


def pchip(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    """Dependency-free, shape-preserving cubic Hermite interpolation."""
    h = np.diff(x)
    delta = np.diff(y) / h
    slopes = np.zeros_like(y, dtype=np.float64)
    if len(x) == 2:
        slopes[:] = delta[0]
    else:
        slopes[0] = pchip_endpoint_slope(h[0], h[1], delta[0], delta[1])
        slopes[-1] = pchip_endpoint_slope(h[-1], h[-2], delta[-1], delta[-2])
        for index in range(1, len(x) - 1):
            if delta[index - 1] * delta[index] <= 0:
                slopes[index] = 0.0
            else:
                w1 = 2.0 * h[index] + h[index - 1]
                w2 = h[index] + 2.0 * h[index - 1]
                slopes[index] = (w1 + w2) / (w1 / delta[index - 1] + w2 / delta[index])
    segment = np.clip(np.searchsorted(x, x_new, side="right") - 1, 0, len(x) - 2)
    x0, x1 = x[segment], x[segment + 1]
    y0, y1 = y[segment], y[segment + 1]
    m0, m1 = slopes[segment], slopes[segment + 1]
    width = x1 - x0
    t = (x_new - x0) / width
    return (
        (2 * t**3 - 3 * t**2 + 1) * y0
        + (t**3 - 2 * t**2 + t) * width * m0
        + (-2 * t**3 + 3 * t**2) * y1
        + (t**3 - t**2) * width * m1
    )


def smooth_cdf(cdf: dict[str, float], maximum: int, *, log_x: bool) -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(1, maximum + 1, dtype=np.float64)
    y = 100.0 * np.asarray([cdf[str(value)] for value in range(1, maximum + 1)], dtype=np.float64)
    if not log_x:
        x = np.insert(x, 0, 0.0)
        y = np.insert(y, 0, 0.0)
    interpolation_x = np.log2(x) if log_x else x
    dense_interpolation_x = np.linspace(interpolation_x[0], interpolation_x[-1], 700)
    dense_x = np.power(2.0, dense_interpolation_x) if log_x else dense_interpolation_x
    dense_y = np.clip(pchip(interpolation_x, y, dense_interpolation_x), 0.0, 100.0)
    # A CDF must remain monotonic even under finite-precision interpolation.
    dense_y = np.maximum.accumulate(dense_y)
    return dense_x, dense_y


def smooth_topk_cdf(cdf: dict[str, float], maximum: int) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Fit a smooth, monotone, endpoint-anchored stretched-exponential CDF."""
    measured_x = np.arange(0, maximum + 1, dtype=np.float64)
    measured_y = np.asarray(
        [0.0] + [100.0 * float(cdf[str(value)]) for value in range(1, maximum + 1)],
        dtype=np.float64,
    )
    best = None
    for scale in np.geomspace(0.001, 20.0, 240):
        for shape in np.linspace(0.05, 3.0, 240):
            unscaled = 1.0 - np.exp(-scale * measured_x**shape)
            prediction = 100.0 * unscaled / unscaled[-1]
            error = float(np.mean((prediction - measured_y) ** 2))
            if best is None or error < best[0]:
                best = (error, float(scale), float(shape))
    if best is None:
        raise RuntimeError("top-k CDF fit failed")
    error, scale, shape = best
    dense_x = np.linspace(0.0, float(maximum), 700)
    unscaled = 1.0 - np.exp(-scale * dense_x**shape)
    dense_y = 100.0 * unscaled / unscaled[-1]
    return dense_x, dense_y, {"scale": scale, "shape": shape, "rmse_pp": error**0.5}


def fit_biexponential(x: np.ndarray, y: np.ndarray) -> tuple[float, float, float, float]:
    """Fit the constrained two-timescale decay used in the original figure."""
    t = x - 1.0
    weights = 1.0 + 1.0 / np.sqrt(x)
    best = None
    for fast_decay in np.arange(0.20, 3.0001, 0.025):
        fast = np.exp(-fast_decay * t)
        for slow_decay in np.arange(0.0, 0.3001, 0.005):
            slow = np.exp(-slow_decay * t)
            delta = fast - slow
            denominator = float(np.sum(weights * delta * delta))
            fast_weight = 0.0 if denominator == 0.0 else float(
                np.clip(np.sum(weights * (y - slow) * delta) / denominator, 0.0, 1.0)
            )
            prediction = slow + fast_weight * delta
            error = float(np.sum(weights * (y - prediction) ** 2))
            if best is None or error < best[0]:
                best = (error, fast_weight, float(fast_decay), float(slow_decay))
    if best is None:
        raise RuntimeError("rank-loss fit failed")
    _, fast_weight, fast_decay, slow_decay = best
    prediction = fast_weight * np.exp(-fast_decay * t) + (1 - fast_weight) * np.exp(-slow_decay * t)
    total = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - float(np.sum((y - prediction) ** 2)) / total
    return fast_weight, fast_decay, slow_decay, r2


def style_curve_axis(ax) -> None:
    ax.tick_params(axis="both", labelsize=13, width=1.2, length=5, direction="in")
    for spine in ax.spines.values():
        spine.set_linewidth(1.25)
        spine.set_color(INK)


def horizontal_guides(ax, *, threshold: bool = False) -> None:
    for value in (20, 40, 60):
        ax.axhline(value, color=GRID, linestyle=(0, (2, 4)), linewidth=0.8, zorder=0)
    if threshold:
        ax.axhline(80, color="#5F6871", linestyle=(0, (5, 3, 1, 3)), linewidth=1.0, zorder=0)


def add_legend(fig, axes) -> None:
    handles, labels = axes[0].get_legend_handles_labels()
    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.955),
        ncol=3,
        frameon=False,
        fontsize=12.5,
        handlelength=2.0,
        handletextpad=0.45,
        columnspacing=1.25,
        borderaxespad=0.0,
    )
    for line in legend.get_lines():
        line.set_linewidth(2.8)


def draw_nprobe(ax, summaries: dict[str, dict]) -> None:
    for name in DATASETS:
        section = summaries[name]["nprobe_top5"]
        x, y = smooth_cdf(section["cdf"], 128, log_x=True)
        ax.plot(x, y, color=COLORS[name], linewidth=3.0, label=name)
        marker = int(section["p80_required_nprobe"])
        ax.vlines(marker, 0, 80, color=COLORS[name], linestyle=(0, (5, 5)), linewidth=1.7)
    for name in DATASETS:
        marker = int(summaries[name]["nprobe_top5"]["p80_required_nprobe"])
        ax.text(marker, 3.0, str(marker), ha="center", va="bottom", fontsize=11,
                color="black")
    ax.set_xscale("log", base=2)
    ax.set_xlim(1, 128)
    ax.set_ylim(0, 100)
    ax.set_xticks([1, 2, 4, 8, 16, 32, 64, 128])
    ax.set_xticklabels(["1", "2", "4", "8", "16", "32", "64", "128"])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    horizontal_guides(ax, threshold=True)
    style_curve_axis(ax)
    ax.set_xlabel("Nprobe", fontsize=14)
    ax.set_ylabel("CDF (%)", fontsize=14)
    ax.set_title("(a) Cumulative Nprobe Coverage", fontsize=14, fontweight="bold", pad=6)


def draw_rank_loss(ax, summaries: dict[str, dict], maximum_rank: int) -> dict[str, dict]:
    fit_metadata = {}
    measured_x = np.arange(1, maximum_rank + 1, dtype=np.float64)
    dense_x = np.linspace(1, maximum_rank, 800)
    for name in DATASETS:
        section = summaries[name]["rank_dependent_evidence_loss"][str(maximum_rank)]
        measured_y = np.asarray(
            [section["normalized_coverage_loss_by_rank"][str(rank)] for rank in range(1, maximum_rank + 1)],
            dtype=np.float64,
        )
        fast_weight, fast_decay, slow_decay, r2 = fit_biexponential(measured_x, measured_y)
        t = dense_x - 1.0
        dense_y = fast_weight * np.exp(-fast_decay * t) + (1 - fast_weight) * np.exp(-slow_decay * t)
        ax.plot(dense_x, dense_y, color=COLORS[name], linewidth=3.0, label=name)
        fit_metadata[name] = {
            "fast_weight": fast_weight,
            "fast_decay": fast_decay,
            "slow_decay": slow_decay,
            "r_squared": r2,
        }
    ax.set_xlim(1, maximum_rank)
    ax.set_ylim(0, 1.05)
    ax.set_xticks([1, 5, 10] if maximum_rank == 10 else [1, 5, 10, 15, 20])
    ax.set_yticks([0.0, 0.2, 0.4, 0.6, 0.8, 1.0])
    for value in (0.2, 0.4, 0.6, 0.8):
        ax.axhline(value, color=GRID, linestyle=(0, (2, 4)), linewidth=0.8, zorder=0)
    style_curve_axis(ax)
    ax.set_xlabel("Document Rank", fontsize=14)
    ax.set_ylabel(
        "Normalized\nAnswer-Coverage Loss",
        fontsize=13.5,
        linespacing=1.0,
    )
    ax.set_title("(b) Rank-Dependent Retrieval Loss", fontsize=14, fontweight="bold", pad=6)
    return fit_metadata


def draw_topk(ax, summaries: dict[str, dict]) -> dict[str, dict[str, float]]:
    fit_metadata = {}
    for name in DATASETS:
        section = summaries[name]["topk_sufficiency"]["10"]
        x, y, fit_metadata[name] = smooth_topk_cdf(section["cdf"], 10)
        ax.plot(x, y, color=COLORS[name], linewidth=3.0, label=name)
        marker = int(section["p80_required_topk"])
        ax.vlines(marker, 0, 80, color=COLORS[name], linestyle=(0, (5, 5)), linewidth=1.7)
        ax.text(marker, 3.0, str(marker), ha="center", va="bottom", fontsize=11,
                color="black")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 100)
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    horizontal_guides(ax, threshold=True)
    style_curve_axis(ax)
    ax.set_xlabel("Top-k", fontsize=14)
    ax.set_ylabel("CDF (%)", fontsize=14)
    ax.set_title("(a) Cumulative Top-k Coverage", fontsize=14, fontweight="bold", pad=6)
    return fit_metadata


def annotation_color(image, value: float) -> str:
    red, green, blue, _ = image.cmap(image.norm(value))
    luminance = 0.2126 * red + 0.7152 * green + 0.0722 * blue
    return "black" if luminance > 0.58 else "white"


def draw_retention(ax, fig, retention_matrix: np.ndarray) -> None:
    from matplotlib.colors import LinearSegmentedColormap

    display_nprobes = SCAN_NPROBES
    display_matrix = retention_matrix
    cmap = LinearSegmentedColormap.from_list(
        "harmorag_blue_retention", ["#F4F7FA", "#DDE7F0", "#BCD0E2", "#80A2C5", "#496B8E"]
    )
    image = ax.imshow(display_matrix, aspect="auto", cmap=cmap, vmin=45, vmax=100)
    ax.set_xticks(range(len(display_nprobes)))
    ax.set_xticklabels([str(value) for value in display_nprobes], fontsize=10.5)
    ax.set_yticks(range(len(BASELINE_NPROBES)))
    ax.set_yticklabels([str(value) for value in BASELINE_NPROBES], fontsize=12)
    ax.tick_params(axis="both", length=0)
    ax.set_xlabel("Evaluation Nprobe", fontsize=14)
    ax.set_ylabel("Reference Nprobe", fontsize=14)
    ax.set_title("(b) Cross-Nprobe Top-1 Retention", fontsize=14, fontweight="bold", pad=6)
    ax.set_xticks(np.arange(-0.5, len(display_nprobes), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(BASELINE_NPROBES), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=0.9)
    ax.tick_params(which="minor", bottom=False, left=False)
    for row in range(display_matrix.shape[0]):
        for column in range(display_matrix.shape[1]):
            value = float(display_matrix[row, column])
            ax.text(column, row, f"{value:.2f}", ha="center", va="center", fontsize=7.8, color=annotation_color(image, value))
    for spine in ax.spines.values():
        spine.set_linewidth(1.2)
        spine.set_color(INK)
    colorbar = fig.colorbar(image, ax=ax, fraction=0.032, pad=0.018, aspect=25)
    colorbar.set_label("Top-1 Retention@10 (%)", fontsize=12.5)
    colorbar.set_ticks([50, 60, 70, 80, 90, 100])
    colorbar.ax.tick_params(labelsize=10.5)


def save(fig, stem: Path, suffixes: tuple[str, ...] = ("png", "pdf", "svg")) -> list[str]:
    outputs = []
    for suffix in suffixes:
        path = stem.with_suffix("." + suffix)
        fig.savefig(path, dpi=320 if suffix == "png" else None, bbox_inches="tight", facecolor="white")
        outputs.append(str(path))
    return outputs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--retention-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("output/figure03_04"))
    args = parser.parse_args()
    out = args.output_dir
    nprobe_paths = {
        name: args.data_dir / f"{PREFIX[name]}_retrieval_observation_summary.json"
        for name in DATASETS
    }
    topk_paths = {
        name: args.data_dir / f"{PREFIX[name]}_topk_rank_summary.json"
        for name in DATASETS
    }
    nprobe = {name: json.loads(path.read_text()) for name, path in nprobe_paths.items()}
    topk = {name: json.loads(path.read_text()) for name, path in topk_paths.items()}
    retention_matrix = load_retention_matrix(args.retention_csv)
    for name in DATASETS:
        assert nprobe[name]["dataset"] == topk[name]["dataset"] == name
        assert nprobe[name]["query_count"] == topk[name]["query_count"]
        assert nprobe[name]["audit"]["reference_nprobe"] == 128
        assert topk[name]["audit"]["reference_nprobe"] == 128
        assert topk[name]["topk_sufficiency"]["10"]["reference_topk"] == 10

    plt.rcParams.update({"font.family": "DejaVu Sans", "axes.unicode_minus": False, "svg.fonttype": "none", "pdf.fonttype": 42})
    out.mkdir(parents=True, exist_ok=True)

    fig3, axes3 = plt.subplots(1, 2, figsize=(11.6, 4.25), dpi=180)
    draw_nprobe(axes3[0], nprobe)
    fits20 = draw_rank_loss(axes3[1], topk, 20)
    add_legend(fig3, axes3)
    fig3.subplots_adjust(left=0.075, right=0.985, bottom=0.18, top=0.815, wspace=0.30)
    figure3_outputs = save(
        fig3, out / "Figure3-Nprobe-Coverage-Rank-Loss-Latest", suffixes=("png", "pdf")
    )
    plt.close(fig3)

    fig4, axes4 = plt.subplots(
        1,
        2,
        figsize=(12.2, 4.25),
        dpi=180,
        gridspec_kw={"width_ratios": [1.0, 1.38]},
    )
    topk_fits = draw_topk(axes4[0], topk)
    draw_retention(axes4[1], fig4, retention_matrix)
    add_legend(fig4, axes4)
    fig4.subplots_adjust(left=0.065, right=0.982, bottom=0.18, top=0.815, wspace=0.18)
    figure4_outputs = save(
        fig4, out / "Figure4-Topk-Coverage-Retention-Latest", suffixes=("png", "pdf")
    )
    plt.close(fig4)

    metadata = {
        "status": "PASS",
        "datasets": {name: nprobe[name]["query_count"] for name in DATASETS},
        "figure3": {
            "panel_a": "maximum-split nprobe CDF; terminal nprobe=128, top-k=5 reference",
            "panel_b": "maximum-split nprobe=128, top-20 rank-dependent answer-coverage loss",
            "p80_nprobe": {name: nprobe[name]["nprobe_top5"]["p80_required_nprobe"] for name in DATASETS},
            "rank_loss_fit": fits20,
        },
        "figure4": {
            "panel_a": "maximum-split nprobe=128, K=10 top-k CDF",
            "panel_b": "retained original NQ cross-nprobe top-1 retention matrix; evaluation nprobe 1..512",
            "p80_topk": {name: topk[name]["topk_sufficiency"]["10"]["p80_required_topk"] for name in DATASETS},
            "topk_cdf_fit": topk_fits,
        },
        "smoothing": {
            "nprobe_cdf": "shape-preserving PCHIP in log2(nprobe); measured P80 unchanged",
            "topk_cdf": "smooth monotone endpoint-anchored stretched-exponential fit; measured P80 unchanged",
            "rank_loss": "constrained two-timescale biexponential fit over measured ranks 1..20",
        },
        "colors": COLORS,
        "input_sha256": {
            str(path): sha256(path)
            for path in list(nprobe_paths.values()) + list(topk_paths.values()) + [args.retention_csv]
        },
        "outputs": figure3_outputs + figure4_outputs,
    }
    (out / "FIGURE3_4_LATEST_METADATA.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
