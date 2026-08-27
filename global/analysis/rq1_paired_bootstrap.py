#!/usr/bin/env python3
"""RQ1: paired RA-FC gains of FCG-RAG over each baseline (pooled, 574 items).

Reproduces: "+0.035 [0.010, 0.060] over Code-RAG, +0.061 [0.035, 0.086] over
Int-RAG, +0.089 [0.064, 0.114] over Report-RAG" (10,000 bootstrap resamples
over items; CIs are percentile intervals and vary by seed at the third
decimal).
"""
import numpy as np

from common import METHODS, load_results, pivot

N_BOOT = 10_000
SEED = 20260819


def main() -> None:
    df = load_results()
    piv = pivot(df)
    rng = np.random.default_rng(SEED)
    print("paired RA-FC difference (FCG-RAG minus baseline), pooled 574 items")
    for base in METHODS[:-1]:
        diff = (piv["RA-FC"]["FCG-RAG"] - piv["RA-FC"][base]).to_numpy()
        n = len(diff)
        boots = diff[rng.integers(0, n, (N_BOOT, n))].mean(axis=1)
        lo, hi = np.percentile(boots, [2.5, 97.5])
        one_sided = float((boots <= 0).mean())
        print(f"  vs {base:10s}: {diff.mean():+.3f}  "
              f"95% CI [{lo:.3f}, {hi:.3f}]  one-sided p~{one_sided:.4f}")


if __name__ == "__main__":
    main()
