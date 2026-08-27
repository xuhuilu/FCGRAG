#!/usr/bin/env python3
"""RQ2: conditional answer rates stratified by evidence coverage.

Reproduces Figure conditional_answer_rate_by_coverage and Appendix
Table coverage_stratified: hit rates (0.226/0.157/0.199/0.199), P(ans|hit)
0.685-0.868, P(ans|miss) 0.190-0.415, per-system ranges, and conditional
RA-FC / Faithfulness per stratum.
"""
from scipy import stats

from common import METHODS, SYSTEMS, load_results


def wilson(k: int, n: int) -> tuple[float, float]:
    lo, hi = stats.binomtest(k, n).proportion_ci(confidence_level=0.95,
                                                 method="wilson")
    return float(lo), float(hi)


def main() -> None:
    df = load_results()
    print(f"{'method':11s} {'hit':>5s} {'P(a|h)':>7s} {'95% Wilson':>15s} "
          f"{'RA-FC|h':>8s} {'Fa|h':>6s} {'P(a|m)':>7s} {'RA-FC|m':>8s} {'Fa|m':>6s}")
    for m in METHODS:
        s = df[df.method == m]
        hit, miss = s[s["Ctx.R"] > 0], s[s["Ctx.R"] == 0]
        ah, am = hit[hit["Ans.R"] == 1], miss[miss["Ans.R"] == 1]
        ci = wilson(int(hit["Ans.R"].sum()), len(hit))
        print(f"{m:11s} {len(hit) / len(s):5.3f} {hit['Ans.R'].mean():7.3f} "
              f"  [{ci[0]:.3f},{ci[1]:.3f}] {ah['RA-FC'].mean():8.3f} "
              f"{ah['Faith'].mean():6.3f} {miss['Ans.R'].mean():7.3f} "
              f"{am['RA-FC'].mean():8.3f} {am['Faith'].mean():6.3f}")
    print("\nstratum sizes: hit n =", {m: int((df[df.method == m]['Ctx.R'] > 0).sum()) for m in METHODS})
    print("answered in hit stratum n =",
          {m: int(((df[df.method == m]['Ctx.R'] > 0) & (df[df.method == m]['Ans.R'] == 1)).sum()) for m in METHODS})

    print("\nper-system hit rate / P(ans|hit) / P(ans|miss):")
    for sys in SYSTEMS.values():
        for m in METHODS:
            s = df[(df.method == m) & (df.system == sys)]
            hit, miss = s[s["Ctx.R"] > 0], s[s["Ctx.R"] == 0]
            print(f"  {sys:9s} {m:11s} hit={len(hit) / len(s):.3f} "
                  f"P(ans|hit)={hit['Ans.R'].mean():.3f} "
                  f"P(ans|miss)={miss['Ans.R'].mean():.3f}")


if __name__ == "__main__":
    main()
