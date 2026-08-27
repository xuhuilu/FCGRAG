#!/usr/bin/env python3
"""RQ2: RA-FC branch decomposition and the closed-book control.

Reproduces Figure rafc_branch_contribution and the closed-book comparison:
covered-branch contributions 0.056-0.090, uncovered branch 0.106 for FCG-RAG
vs 0.052-0.078 for baselines; on answered covered items FCG-RAG RA-FC 0.524
vs closed-book 0.360; on answered uncovered items FCG-RAG 0.320 vs
closed-book 0.339 (statistically indistinguishable, wins/losses even).
"""
import numpy as np

from common import METHODS, load_direct, load_results

N_BOOT = 10_000
SEED = 20260819


def main() -> None:
    df = load_results()
    direct = load_direct().set_index(["system", "item_id"])
    print("branch contributions to overall RA-FC (pooled):")
    for m in METHODS:
        s = df[df.method == m]
        cov = s[s["Ctx.R"] > 0]["RA-FC"].mean() * (s["Ctx.R"] > 0).mean()
        unc = s[s["Ctx.R"] == 0]["RA-FC"].mean() * (s["Ctx.R"] == 0).mean()
        print(f"  {m:11s} covered={cov:.3f}  uncovered={unc:.3f}  "
              f"overall={s['RA-FC'].mean():.3f}")

    fcg = df[df.method == "FCG-RAG"].set_index(["system", "item_id"])
    fcg = fcg.join(direct)
    rng = np.random.default_rng(SEED)
    print("\nFCG-RAG answered items vs closed-book control (same generator, "
          "no retrieved context):")
    for name, sub in [("covered (Ctx.R>0)", fcg[(fcg["Ans.R"] == 1) & (fcg["Ctx.R"] > 0)]),
                      ("uncovered (Ctx.R=0)", fcg[(fcg["Ans.R"] == 1) & (fcg["Ctx.R"] == 0)])]:
        d = (sub["RA-FC"] - sub["Direct.RA-FC"]).to_numpy()
        n = len(d)
        boots = d[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        print(f"  {name}: n={n}  FCG RA-FC={sub['RA-FC'].mean():.3f}  "
              f"direct RA-FC={sub['Direct.RA-FC'].mean():.3f}  "
              f"diff={d.mean():+.3f} 95% CI [{lo:.3f}, {hi:.3f}]  "
              f"FCG wins {(d > 0.01).mean():.3f} / ties "
              f"{((d >= -0.01) & (d <= 0.01)).mean():.3f} / "
              f"direct wins {(d < -0.01).mean():.3f}")


if __name__ == "__main__":
    main()
