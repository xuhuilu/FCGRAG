#!/usr/bin/env python3
"""Histogram of per-item paired Qual differences (FCG-RAG minus Code+AST).

Data default: analysis/item_level/output/local_item_structural_features_rerun.csv
(post-rerun deltas); pass --input to use another delta source. Mean/median
marker labels are computed from the data.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "local_item_structural_features_rerun.csv"
DEFAULT_OUT = ROOT / "output" / "figures" / "local_paired_qual_difference.pdf"

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "font.family": "sans-serif",
})


def main(data: Path, out: Path) -> None:
    deltas = pd.read_csv(data)["delta_qual"].to_numpy()
    mean, median = float(np.mean(deltas)), float(np.median(deltas))

    bins = np.arange(-0.80, 0.96, 0.05)
    fig, ax = plt.subplots(figsize=(6.2, 2.2))
    counts, edges, patches = ax.hist(deltas, bins=bins, edgecolor="#333333",
                                     linewidth=0.5, zorder=3)
    for left, patch in zip(edges[:-1], patches):
        patch.set_facecolor("#c7c7c7" if left + 0.05 <= 0 else "#666666")

    annotations = ((0.0, "zero", (-46, -14)),
                   (mean, f"mean = {mean:.3f}", (5, -14)),
                   (median, f"median = {median:.3f}", (5, -44)))
    for value, label, offset in annotations:
        ax.axvline(value, color="#111111",
                   linestyle=(0, (4, 3)) if value == 0.0 else ("solid" if value == mean else (0, (2, 2))),
                   linewidth=1.2, zorder=4)
        ax.annotate(label, xy=(value, ax.get_ylim()[1]), xytext=offset,
                    textcoords="offset points", fontsize=7.5)

    ax.set_xlabel(r"Per-item $\Delta$Qual. (FCG-RAG $-$ Code+AST)")
    ax.set_ylabel("Number of items")
    ax.grid(axis="y", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"written {out}  (mean={mean:.4f}, median={median:.4f}, n={len(deltas)})")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.input, args.output)
