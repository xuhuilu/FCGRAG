#!/usr/bin/env python3
"""Regenerate figures/conditional_answer_rate_by_coverage.pdf (paper Figure 6)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats

from common import METHODS, ROOT, load_results

OUT = ROOT / "analysis" / "output" / "figures"


def wilson(k: int, n: int) -> tuple[float, float]:
    lo, hi = stats.binomtest(k, n).proportion_ci(confidence_level=0.95,
                                                 method="wilson")
    return float(lo), float(hi)


def main() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    df = load_results()
    fig, ax = plt.subplots(figsize=(4.4, 2.8))
    x = np.arange(len(METHODS))
    w = 0.38
    for j, (stratum, color, label) in enumerate([
            ("hit", "#333333", "P(answer | Ctx.R > 0)"),
            ("miss", "#bbbbbb", "P(answer | Ctx.R = 0)")]):
        vals, errs = [], []
        for m in METHODS:
            s = df[df.method == m]
            sub = s[(s["Ctx.R"] > 0) if stratum == "hit" else (s["Ctx.R"] == 0)]
            k, n = int(sub["Ans.R"].sum()), len(sub)
            p = k / n
            lo, hi = wilson(k, n)
            vals.append(p)
            errs.append([p - lo, hi - p])
        errs = np.array(errs).T
        bars = ax.bar(x + (j - 0.5) * w, vals, w, color=color, label=label,
                      yerr=errs, capsize=3, error_kw=dict(ecolor="#666666", lw=1))
        for xi, v in zip(x, vals):
            ax.text(xi + (j - 0.5) * w, v + 0.055, f"{v:.3f}", ha="center",
                    fontsize=7.5)
    labels = []
    for m in METHODS:
        s = df[df.method == m]
        labels.append(f"{m}\nhit rate {(s['Ctx.R'] > 0).mean():.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7.5)
    ax.tick_params(axis="x", pad=2)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Conditional answer rate")
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#dddddd", lw=0.5)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="upper left")
    fig.subplots_adjust(bottom=0.22)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "conditional_answer_rate_by_coverage.pdf",
                bbox_inches="tight")
    print(f"wrote {OUT / 'conditional_answer_rate_by_coverage.pdf'}")


if __name__ == "__main__":
    main()
