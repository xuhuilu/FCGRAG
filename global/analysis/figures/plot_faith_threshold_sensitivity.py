#!/usr/bin/env python3
"""Regenerate figures/faith_threshold_sensitivity.pdf (paper Appendix Figure)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import METHODS, ROOT, load_results

OUT = ROOT / "analysis" / "output" / "figures"
STYLES = {"Code-RAG": ("-", "#111111"), "Report-RAG": ("--", "#777777"),
          "Int-RAG": ("-.", "#555555"), "FCG-RAG": ("-", "#333333")}
MARKERS = {"Code-RAG": "o", "Report-RAG": "s", "Int-RAG": "^", "FCG-RAG": "D"}


def main() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    df = load_results()
    taus = np.linspace(0.0, 1.0, 21)
    fig, ax = plt.subplots(figsize=(4.2, 2.8))
    for m in METHODS:
        s = df[df.method == m]
        ans = s["Ans.R"] == 1
        rates = [((ans) & (s["Faith"] > t)).mean() for t in taus]
        ls, color = STYLES[m]
        ax.plot(taus, rates, ls, color=color, marker=MARKERS[m],
                markersize=3, markevery=4, lw=1.4 if m == "FCG-RAG" else 1.0,
                label=m)
    ax.axvspan(0.95, 1.0, color="#eeeeee")
    ax.text(0.975, 0.40, "Code-RAG\novertakes", ha="center", fontsize=6.5,
            color="#555555")
    ax.set_xlabel("Faithfulness threshold $\\tau$ (answer counted if Faith > $\\tau$)")
    ax.set_ylabel("Faithful-answer rate over all items")
    ax.set_xlim(0, 1.0)
    ax.set_ylim(0, 0.5)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower left", fontsize=7.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "faith_threshold_sensitivity.pdf", bbox_inches="tight")
    print(f"wrote {OUT / 'faith_threshold_sensitivity.pdf'}")


if __name__ == "__main__":
    main()
