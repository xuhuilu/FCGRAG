#!/usr/bin/env python3
"""Regenerate figures/rafc_branch_contribution.pdf (paper Figure 7)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common import METHODS, ROOT, load_results

OUT = ROOT / "analysis" / "output" / "figures"


def main() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    df = load_results()
    cov, unc = [], []
    for m in METHODS:
        s = df[df.method == m]
        cov.append(s[s["Ctx.R"] > 0]["RA-FC"].mean() * (s["Ctx.R"] > 0).mean())
        unc.append(s[s["Ctx.R"] == 0]["RA-FC"].mean() * (s["Ctx.R"] == 0).mean())
    x = np.arange(len(METHODS))
    fig, ax = plt.subplots(figsize=(4.0, 2.8))
    ax.bar(x, cov, 0.55, color="#333333",
           label="Ctx.R > 0 branch (evidence use)")
    ax.bar(x, unc, 0.55, bottom=cov, color="#bbbbbb",
           label="Ctx.R = 0 branch (parametric knowledge)")
    for xi, (c, u) in enumerate(zip(cov, unc)):
        ax.text(xi, c / 2, f"{c:.3f}", ha="center", va="center",
                color="white", fontsize=7.5)
        ax.text(xi, c + u / 2, f"{u:.3f}", ha="center", va="center",
                color="#333333", fontsize=7.5)
        ax.text(xi, c + u + 0.008, f"{c + u:.3f}", ha="center",
                fontweight="bold", fontsize=8.5)
    ax.set_xticks(x)
    ax.set_xticklabels(METHODS)
    ax.set_ylim(0, 0.23)
    ax.set_ylabel("Contribution to overall RA-FC")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left", fontsize=7.5)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "rafc_branch_contribution.pdf", bbox_inches="tight")
    print(f"wrote {OUT / 'rafc_branch_contribution.pdf'}")


if __name__ == "__main__":
    main()
