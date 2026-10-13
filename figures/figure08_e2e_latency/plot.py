#!/usr/bin/env python3
"""Reproduce Figure 8's 3x3 layout from the released CSV or an XLSX file."""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
import re
import zipfile
import xml.etree.ElementTree as ET

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


REQUIRED_COLUMNS = {
    "dataset",
    "model_family",
    "system",
    "rate_req_s",
    "mean_e2e_s",
}
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = REPO_ROOT / "data" / "figure08_e2e_latency" / "points.csv"

MODEL_ORDER = ["8b", "32b", "70b"]
MODEL_TITLES = {
    "8b": "Llama3-8B (1 GPU)",
    "32b": "Qwen3-32B (4 GPUs)",
    "70b": "Llama3-70B (8 GPUs)",
}
DATASET_ORDER = ["nq", "hotpotqa", "squad"]
DATASET_LABELS = {"nq": "NQ", "hotpotqa": "HotpotQA", "squad": "SQuAD"}

SYSTEM_ORDER = [
    "adaptive_k",
    "selfrag_k10",
    "adaptive_selfrag_cap10",
    "hedrarag",
    "harmorag",
]
SYSTEM_STYLES = {
    "adaptive_k": {
        "label": r"Adaptive-$k$",
        "color": "#4C78A8",
        "marker": "o",
    },
    "selfrag_k10": {
        "label": "Self-RAG",
        "color": "#B279A2",
        "marker": "s",
    },
    "adaptive_selfrag_cap10": {
        "label": r"Adaptive-$k$ + Self-RAG",
        "color": "#E45756",
        "marker": "X",
    },
    "hedrarag": {
        "label": "HedraRAG",
        "color": "#54A24B",
        "marker": "D",
    },
    "harmorag": {
        "label": "HarmoRAG",
        "color": "#F58518",
        "marker": "^",
    },
}


def _xlsx_column_index(cell_ref: str) -> int:
    letters = re.match(r"[A-Z]+", cell_ref).group(0)
    index = 0
    for char in letters:
        index = index * 26 + (ord(char) - ord("A") + 1)
    return index - 1


def read_first_xlsx_sheet(path: Path) -> list[dict[str, object]]:
    """Read the first worksheet using only Python's standard library."""
    ns_main = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
    ns_rel = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
    ns_pkg = "http://schemas.openxmlformats.org/package/2006/relationships"

    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for item in root.findall(f"{{{ns_main}}}si"):
                text = "".join(t.text or "" for t in item.iter(f"{{{ns_main}}}t"))
                shared_strings.append(text)

        workbook = ET.fromstring(archive.read("xl/workbook.xml"))
        first_sheet = workbook.find(f".//{{{ns_main}}}sheet")
        if first_sheet is None:
            raise ValueError("Workbook has no worksheets")
        relation_id = first_sheet.attrib[f"{{{ns_rel}}}id"]

        relationships = ET.fromstring(archive.read("xl/_rels/workbook.xml.rels"))
        target = None
        for rel in relationships.findall(f"{{{ns_pkg}}}Relationship"):
            if rel.attrib.get("Id") == relation_id:
                target = rel.attrib["Target"]
                break
        if target is None:
            raise ValueError("Could not resolve the first worksheet")
        sheet_path = target.lstrip("/")
        if not sheet_path.startswith("xl/"):
            sheet_path = f"xl/{sheet_path}"

        sheet = ET.fromstring(archive.read(sheet_path))
        matrix: list[list[object]] = []
        for row in sheet.findall(f".//{{{ns_main}}}row"):
            values: dict[int, object] = {}
            for cell in row.findall(f"{{{ns_main}}}c"):
                idx = _xlsx_column_index(cell.attrib["r"])
                cell_type = cell.attrib.get("t")
                value_node = cell.find(f"{{{ns_main}}}v")
                if cell_type == "inlineStr":
                    value = "".join(
                        t.text or "" for t in cell.iter(f"{{{ns_main}}}t")
                    )
                elif value_node is None:
                    value = ""
                else:
                    raw = value_node.text or ""
                    if cell_type == "s":
                        value = shared_strings[int(raw)]
                    elif cell_type in {"str", "e"}:
                        value = raw
                    else:
                        number = float(raw)
                        value = int(number) if number.is_integer() else number
                values[idx] = value
            if values:
                width = max(values) + 1
                matrix.append([values.get(i, "") for i in range(width)])

    if not matrix:
        raise ValueError("First worksheet is empty")
    headers = [str(value).strip() for value in matrix[0]]
    records: list[dict[str, object]] = []
    for row in matrix[1:]:
        padded = row + [""] * (len(headers) - len(row))
        records.append(dict(zip(headers, padded)))
    return records


def read_rows(path: Path) -> list[dict[str, object]]:
    if path.suffix.lower() == ".csv":
        with path.open(newline="", encoding="utf-8") as handle:
            return list(csv.DictReader(handle))
    if path.suffix.lower() == ".xlsx":
        return read_first_xlsx_sheet(path)
    raise ValueError("--input must be a .csv or .xlsx file")


def validate_data(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    if not rows:
        raise ValueError("Workbook contains no data rows")
    missing = REQUIRED_COLUMNS - set(rows[0])
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out: list[dict[str, object]] = []
    seen: set[tuple[object, ...]] = set()
    for source in rows:
        row = {
            "dataset": str(source["dataset"]).strip().lower(),
            "model_family": str(source["model_family"]).strip().lower(),
            "system": str(source["system"]).strip().lower(),
            "rate_req_s": float(source["rate_req_s"]),
            "mean_e2e_s": float(source["mean_e2e_s"]),
        }
        if not math.isfinite(row["rate_req_s"]) or not math.isfinite(row["mean_e2e_s"]):
            raise ValueError("Workbook contains non-finite numeric values")
        if row["rate_req_s"] < 0 or row["mean_e2e_s"] < 0:
            raise ValueError("Rates and latencies must be non-negative")
        key = (row["dataset"], row["model_family"], row["system"], row["rate_req_s"])
        if key in seen:
            raise ValueError(f"Duplicate plotting key found: {key}")
        seen.add(key)
        out.append(row)

    for column, expected in [
        ("dataset", set(DATASET_ORDER)),
        ("model_family", set(MODEL_ORDER)),
        ("system", set(SYSTEM_ORDER)),
    ]:
        unexpected = {row[column] for row in out} - expected
        if unexpected:
            raise ValueError(f"Unexpected {column} values: {sorted(unexpected)}")

    return sorted(
        out,
        key=lambda row: (
            row["dataset"],
            row["model_family"],
            row["system"],
            row["rate_req_s"],
        ),
    )


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "font.size": 7.0,
            "axes.titlesize": 8.5,
            "axes.titleweight": "bold",
            "axes.labelsize": 7.5,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "legend.fontsize": 6.8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.15,
            "lines.markersize": 3.2,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
            "figure.facecolor": "white",
        }
    )


def plot_figure(data: list[dict[str, object]], output_stem: Path) -> None:
    configure_style()
    width_in = 183 / 25.4
    height_in = 122 / 25.4
    fig, axes = plt.subplots(
        3,
        3,
        figsize=(width_in, height_in),
        sharey=True,
        constrained_layout=False,
    )

    legend_handles = []
    for system in SYSTEM_ORDER:
        style = SYSTEM_STYLES[system]
        handle, = axes[0, 0].plot(
            [],
            [],
            color=style["color"],
            marker=style["marker"],
            markeredgewidth=0,
            label=style["label"],
        )
        legend_handles.append(handle)

    for row_idx, dataset in enumerate(DATASET_ORDER):
        for col_idx, model in enumerate(MODEL_ORDER):
            ax = axes[row_idx, col_idx]
            panel = [
                row
                for row in data
                if row["dataset"] == dataset and row["model_family"] == model
            ]

            for system in SYSTEM_ORDER:
                series = sorted(
                    (row for row in panel if row["system"] == system),
                    key=lambda row: row["rate_req_s"],
                )
                if not series:
                    continue
                style = SYSTEM_STYLES[system]
                ax.plot(
                    [row["rate_req_s"] for row in series],
                    [row["mean_e2e_s"] for row in series],
                    color=style["color"],
                    marker=style["marker"],
                    markeredgewidth=0,
                    solid_capstyle="round",
                    clip_on=True,
                    zorder=3,
                )

            ax.set_ylim(0, 10)
            ax.set_yticks(np.arange(0, 11, 2))
            if model == "8b":
                ax.set_xlim(0, 20)
                ax.set_xticks([0, 4, 8, 12, 16, 20])
            else:
                ax.set_xlim(0, 8)
                ax.set_xticks([0, 2, 4, 6, 8])

            ax.grid(
                True,
                which="major",
                axis="both",
                color="#C8D6E5",
                linewidth=0.55,
                linestyle=(0, (3, 3)),
                zorder=0,
            )
            ax.tick_params(direction="out", length=2.4, width=0.65, pad=1.5)
            for spine in ax.spines.values():
                spine.set_visible(True)
                spine.set_color("#222222")
                spine.set_linewidth(0.7)

            if row_idx == 0:
                ax.set_title(MODEL_TITLES[model], pad=3.5)
            if col_idx == 0:
                ax.set_ylabel(DATASET_LABELS[dataset], labelpad=5.5)
            if row_idx == 2:
                ax.set_xlabel("Arrival Rate (req/s)", labelpad=2.5)

    fig.legend(
        handles=legend_handles,
        labels=[SYSTEM_STYLES[s]["label"] for s in SYSTEM_ORDER],
        loc="upper center",
        bbox_to_anchor=(0.54, 0.966),
        ncol=5,
        frameon=False,
        handlelength=1.7,
        handletextpad=0.35,
        columnspacing=1.2,
        borderaxespad=0,
    )
    fig.text(
        0.013,
        0.49,
        "End-to-End Latency (s/query)",
        rotation=90,
        va="center",
        ha="center",
        fontsize=8.2,
    )
    fig.subplots_adjust(left=0.075, right=0.995, bottom=0.085, top=0.90, wspace=0.12, hspace=0.13)

    output_stem.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".png"), dpi=400, bbox_inches="tight")
    fig.savefig(output_stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output-dir", type=Path, default=Path("output/figure08"))
    args = parser.parse_args()

    data = validate_data(read_rows(args.input))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with (args.output_dir / "figure8_source_data.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["dataset", "model_family", "system", "rate_req_s", "mean_e2e_s"],
        )
        writer.writeheader()
        writer.writerows(data)
    plot_figure(data, args.output_dir / "figure8_latest")

    print(f"rows={len(data)}")
    print(f"output={args.output_dir / 'figure8_latest.png'}")


if __name__ == "__main__":
    main()
