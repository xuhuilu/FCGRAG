#!/usr/bin/env python3
"""Dot plot of Qual by local question type (FCG-RAG minus baselines view).

Data default: analysis/item_level/output/qtype_method_qual_rerun.csv
(post-rerun scores). Question types are ordered by the aggregate FCG-RAG
advantage; markers of the four configurations are vertically offset within
each row; FCG-RAG value labels are bolded.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA = ROOT / "data" / "qtype_method_qual_rerun.csv"
DEFAULT_OUT = ROOT / "output" / "figures" / "local_question_type_performance.pdf"

QTYPE_LABEL = {
    "impact_scope": "Impact scope",
    "dependency_trace": "Dependency trace",
    "execution_flow": "Execution flow",
    "conditional_path": "Conditional path",
    "error_handling_path": "Error-handling path",
}
# marker, gray, filled?, bold label
METHODS = {
    "direct":     dict(label="Direct",    marker="o", color="#4d4d4d", fill=True,  dy=-0.33),
    "code_only":  dict(label="Code-only", marker="s", color="#969696", fill=True,  dy=-0.11),
    "source_ast": dict(label="Code+AST",  marker="^", color="#111111", fill=False, dy=0.11),
    "fcg_rag":    dict(label="FCG-RAG",   marker="D", color="#000000", fill=True,  dy=0.33),
}

plt.rcParams.update({
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "font.size": 8, "font.family": "sans-serif",
})


def main(data: Path, out: Path) -> None:
    df = pd.read_csv(data)
    wide = df.pivot(index="question_type", columns="method", values="qual")
    wide["adv"] = wide["fcg_rag"] - wide[["direct", "code_only", "source_ast"]].max(axis=1)
    wide = wide.sort_values("adv", ascending=False)

    fig, ax = plt.subplots(figsize=(6.2, 2.8))
    for row, (qtype, r) in enumerate(wide.iterrows()):
        for method, st in METHODS.items():
            y = row + st["dy"]
            v = r[method]
            ax.scatter(v, y, marker=st["marker"], s=26,
                       facecolors=st["color"] if st["fill"] else "none",
                       edgecolors=st["color"], linewidths=0.9, zorder=3)
            bold = method == "fcg_rag"
            ax.text(v + 0.012, y, f"{v:.3f}", va="center", ha="left",
                    fontsize=7.5 if not bold else 8,
                    fontweight="bold" if bold else "normal",
                    color="#111111" if bold else "#444444", zorder=3)

    ax.set_yticks(range(len(wide)))
    ax.set_yticklabels([QTYPE_LABEL[q] for q in wide.index])
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(len(wide) - 0.5, -0.5)
    ax.set_xlabel("Qual.")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6, zorder=0)
    ax.set_axisbelow(True)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    handles = [plt.Line2D([], [], marker=st["marker"], linestyle="none",
                          markerfacecolor=st["color"] if st["fill"] else "none",
                          markeredgecolor=st["color"], markersize=6,
                          label=st["label"])
               for st in METHODS.values()]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0),
              ncol=4, frameon=False, handletextpad=0.3, columnspacing=1.2)
    fig.tight_layout()
    fig.savefig(out, bbox_inches="tight")
    print(f"written {out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    main(args.input, args.output)
