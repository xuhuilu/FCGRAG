#!/usr/bin/env python3
"""Regenerate figures/global_metric_correlation_heatmap.pdf (paper Figure 8)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common import ROOT, load_results

OUT = ROOT / "analysis" / "output" / "figures"
METRICS = ["CtxTok", "Ctx.R", "Faith", "Ans.C", "RA-FC"]


def main() -> None:
    plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.size": 8})
    df = load_results()
    fcg = df[df.method == "FCG-RAG"]
    M = fcg[METRICS].corr(method="spearman").to_numpy()
    fig, ax = plt.subplots(figsize=(3.2, 2.8))
    im = ax.imshow(M, cmap="Greys", vmin=-1, vmax=1)
    ax.set_xticks(range(5))
    ax.set_xticklabels(METRICS, rotation=45, ha="right")
    ax.set_yticks(range(5))
    ax.set_yticklabels(METRICS)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center",
                    fontsize=7,
                    color="white" if abs(M[i, j]) > 0.55 else "black")
    fig.colorbar(im, shrink=0.8, label="Spearman $\\rho$")
    fig.tight_layout()
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "global_metric_correlation_heatmap.pdf",
                bbox_inches="tight")
    print(f"wrote {OUT / 'global_metric_correlation_heatmap.pdf'}")


if __name__ == "__main__":
    main()
