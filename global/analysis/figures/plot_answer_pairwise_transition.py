#!/usr/bin/env python3
"""Regenerate figures/answer_pairwise_transition.pdf (paper Figure 5)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from scipy import stats

from common import METHODS, ROOT, pivot, load_results

OUT = ROOT / "analysis" / "output" / "figures"


def mcnemar_p(b: int, c: int) -> float:
    return float(stats.chi2.sf((abs(b - c) - 1) ** 2 / (b + c), 1))


def main() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    df = load_results()
    piv = pivot(df)
    rows = []
    for base in METHODS[:-1]:
        fcg = piv["Ans.R"]["FCG-RAG"] == 1
        other = piv["Ans.R"][base] == 1
        fo = piv[fcg & ~other]
        bo = piv[~fcg & other]
        rows.append((f"vs {base}", len(fo), fo["RA-FC"]["FCG-RAG"].mean(),
                     len(bo), bo["RA-FC"][base].mean()))

    fig, ax = plt.subplots(figsize=(6.6, 2.6))
    y = list(range(len(rows)))[::-1]
    for yi, (label, b, fr, c, br) in zip(y, rows):
        net = b * fr - c * br
        p = mcnemar_p(b, c)
        ax.barh(yi, b, color="#333333", height=0.55)
        ax.barh(yi, -c, color="#bbbbbb", height=0.55)
        ax.text(b / 2, yi, f"RA-FC {fr:.3f}", va="center", ha="center",
                color="white", fontsize=7.5)
        ax.text(-c / 2, yi, f"{br:.3f}", va="center", ha="center",
                color="#333333", fontsize=7.5)
        ax.text(b + 5, yi, f"n={b}", va="center", fontsize=8)
        ax.text(-c - 5, yi, f"n={c}", va="center", ha="right", fontsize=8)
        ax.text(228, yi, f"p={p:.0e}\nnet {net:+.1f}", va="center",
                ha="left", fontsize=7, style="italic")
    ax.axvline(0, color="#111111", lw=1)
    ax.set_yticks(y)
    ax.set_yticklabels([r[0] for r in rows])
    ax.set_xlim(-110, 285)
    ax.set_xticks([-100, -50, 0, 50, 100, 150, 200, 250])
    ax.set_xlabel("Number of items answered by only one method")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="x", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(handles=[Patch(color="#333333", label="FCG-RAG only"),
                       Patch(color="#bbbbbb", label="Baseline only")],
              loc="lower center", ncol=2, frameon=False,
              bbox_to_anchor=(0.5, -0.38))
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "answer_pairwise_transition.pdf", bbox_inches="tight")
    print(f"wrote {OUT / 'answer_pairwise_transition.pdf'}")


if __name__ == "__main__":
    main()
