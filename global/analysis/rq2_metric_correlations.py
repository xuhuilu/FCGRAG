#!/usr/bin/env python3
"""RQ2: item-level metric correlations and faithfulness-threshold analysis.

Reproduces Figure global_metric_correlation_heatmap values (FCG-RAG, pooled,
N=574): rho(CtxTok, RA-FC)=0.028 [-0.056, 0.112]; rho(Ctx.R, RA-FC)=0.489
[0.414, 0.559]; rho(Faith, RA-FC)=0.645 pooled but -0.111 [-0.228, 0.003]
among answered items; rho(Ctx.R, RA-FC)=0.401 among answered items.
Also reproduces the faithful-answer rates: lenient (Faith>0) 0.467 for
FCG-RAG vs 0.256-0.341 for baselines; strict full-support (Faith~1)
0.237 for Code-RAG vs 0.214 for FCG-RAG.
"""
import numpy as np
from scipy import stats

from common import METHODS, load_results

N_BOOT = 10_000
SEED = 20260731
PAIRS = [("CtxTok", "RA-FC"), ("Ctx.R", "RA-FC"), ("Faith", "RA-FC"),
         ("Ans.C", "RA-FC"), ("CtxTok", "Ans.C"), ("Ctx.R", "Faith"),
         ("Ctx.R", "Ans.C"), ("Faith", "Ans.C"), ("CtxTok", "Ctx.R"),
         ("CtxTok", "Faith")]


def spearman_boot(df, x, y):
    d = df[[x, y]].dropna()
    rx = stats.rankdata(d[x].to_numpy())
    ry = stats.rankdata(d[y].to_numpy())
    rho, p = stats.spearmanr(d[x], d[y])
    rng = np.random.default_rng(SEED)
    n = len(rx)
    idx = rng.integers(0, n, size=(N_BOOT, n))
    bx = rx[idx] - rx[idx].mean(axis=1, keepdims=True)
    by = ry[idx] - ry[idx].mean(axis=1, keepdims=True)
    denom = np.sqrt((bx ** 2).sum(axis=1) * (by ** 2).sum(axis=1))
    with np.errstate(invalid="ignore", divide="ignore"):
        boots = (bx * by).sum(axis=1) / denom
    boots = boots[np.isfinite(boots)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return rho, lo, hi, p


def bh_adjust(pvals):
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / (np.arange(len(p)) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(len(p))
    out[order] = np.minimum(ranked, 1.0)
    return out


def main() -> None:
    df = load_results()
    fcg = df[df.method == "FCG-RAG"]
    rows = []
    for x, y in PAIRS:
        rho, lo, hi, p = spearman_boot(fcg, x, y)
        rows.append(dict(x=x, y=y, rho=rho, ci_lo=lo, ci_hi=hi, p=p))
    res = __import__("pandas").DataFrame(rows)
    res["p_adj"] = bh_adjust(res["p"])
    print("FCG-RAG pooled item-level Spearman correlations (N=574):")
    print(res.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    ans = fcg[fcg["Ans.R"] == 1]
    for x, y in [("Ctx.R", "RA-FC"), ("Faith", "RA-FC")]:
        rho, lo, hi, p = spearman_boot(ans, x, y)
        print(f"answered-only rho({x}, {y}) = {rho:.3f} [{lo:.3f}, {hi:.3f}]")

    print("\nfaithful-answer rates (fraction of all items answered with "
          "Faith above threshold):")
    for m in METHODS:
        s = df[df.method == m]
        lenient = ((s["Ans.R"] == 1) & (s["Faith"] > 0)).mean()
        strict = ((s["Ans.R"] == 1) & (s["Faith"] >= 0.999)).mean()
        print(f"  {m:11s} lenient (Faith>0) = {lenient:.3f}   "
              f"strict (Faith=1) = {strict:.3f}")


if __name__ == "__main__":
    main()
