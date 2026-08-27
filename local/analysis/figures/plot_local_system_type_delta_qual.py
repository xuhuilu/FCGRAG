#!/usr/bin/env python3
"""Redraw figures/local_system_type_delta_qual.pdf in grayscale.

Data default: analysis/item_level/output/system_type_qual_rerun.csv (post-rerun
scores). The frozen pre-rerun variant is
analysis/frozen_local/output/system_type_qual.csv. Columns are ordered by the
aggregate FCG-RAG advantage, matching figures/local_question_type_performance.pdf.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "system_type_qual_rerun.csv"
DEFAULT_OUT = ROOT / "output" / "figures" / "local_system_type_delta_qual.pdf"

SYSTEMS = ["LangChain/Python", "Ollama/C--C++", "Ollama/Go", "OpenClaw/TS--JS"]
# ordered by aggregate FCG-RAG advantage, matching the dot plot
COLS = [
    ("impact_scope", "Impact scope"),
    ("execution_flow", "Execution flow"),
    ("error_handling_path", "Error-handling path"),
    ("dependency_trace", "Dependency trace"),
    ("conditional_path", "Conditional path"),
]

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "font.family": "sans-serif",
})


def main(data: Path, out: Path) -> None:
    df = pd.read_csv(data)
    lut = {(r["system"], r["question_type"]): r for _, r in df.iterrows()}

    fig, ax = plt.subplots(figsize=(6.4, 2.2))
    vmax = df.loc[df["n"] > 0, "delta_qual"].astype(float).abs().max()

    for row, system in enumerate(SYSTEMS):
        ax.text(-0.55, row, system, ha="right", va="center", fontsize=8)
        for col, (qtype, _) in enumerate(COLS):
            r = lut[(system, qtype)]
            if r["n"] == 0:
                ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                           facecolor="white", edgecolor="#bbbbbb",
                                           linewidth=0.8))
                ax.text(col, row + 0.08, "N/A", ha="center", va="center",
                        fontsize=8, color="#666666")
                ax.text(col, row + 0.26, "N=0", ha="center", va="center",
                        fontsize=6.5, color="#666666")
                continue
            v = float(r["delta_qual"])
            # diverging grayscale: 0 -> mid gray, +|max| -> dark, -|max| -> light
            g = 0.82 - 0.62 * (v + vmax) / (2 * vmax)  # gray level in [0.2, 0.82]
            ax.add_patch(plt.Rectangle((col - 0.5, row - 0.5), 1, 1,
                                       facecolor=str(round(g, 3)),
                                       edgecolor="white", linewidth=1.5))
            text_color = "white" if g < 0.45 else "#111111"
            ax.text(col, row - 0.10, f"{v:+.3f}", ha="center", va="center",
                    fontsize=8, fontweight="bold", color=text_color)
            ax.text(col, row + 0.24, f"N={r['n']}", ha="center", va="center",
                    fontsize=6.5, color=text_color)

    for col, (_, label) in enumerate(COLS):
        words = label.split(" ")
        if len(label) > 13 and len(words) > 1:
            label = " ".join(words[:-1]) + "\n" + words[-1]
        ax.text(col, -0.62, label, ha="center", va="bottom", fontsize=8)

    ax.set_xlim(-0.5, len(COLS) - 0.5)
    ax.set_ylim(len(SYSTEMS) - 0.5, -1.3)
    ax.axis("off")
    ax.text((len(COLS) - 1) / 2, len(SYSTEMS) - 0.28,
            r"$\Delta$Qual. = FCG-RAG $-$ Code+AST (descriptive)",
            ha="center", va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"written {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.input, args.output)
