#!/usr/bin/env python3
"""RQ2: pairwise answer-transition analysis (Figure answer_pairwise_transition).

Reproduces: transition counts 125/46, 172/36, 150/37; exclusive-answer mean
RA-FC 0.378-0.398 vs 0.439-0.509; net quality contribution +24.2 to +52.6;
McNemar p = 2.4e-09, 7.9e-21, 2.6e-16 (chi-square with continuity
correction); both-answered paired differences -0.013 to -0.024 (CIs include
zero); 71-74% of uniquely answered items fall in the Ctx.R=0 stratum.
"""
import numpy as np
from scipy import stats

from common import METHODS, load_results, pivot

N_BOOT = 10_000
SEED = 20260819


def mcnemar_p(b: int, c: int) -> float:
    """Continuity-corrected McNemar chi-square p-value."""
    chi2 = (abs(b - c) - 1) ** 2 / (b + c)
    return float(stats.chi2.sf(chi2, 1))


def main() -> None:
    df = load_results()
    piv = pivot(df)
    rng = np.random.default_rng(SEED)
    for base in METHODS[:-1]:
        fcg = piv["Ans.R"]["FCG-RAG"] == 1
        other = piv["Ans.R"][base] == 1
        b, c = int((fcg & ~other).sum()), int((~fcg & other).sum())
        fcg_only = piv[fcg & ~other]
        base_only = piv[~fcg & other]
        net = b * fcg_only["RA-FC"]["FCG-RAG"].mean() - \
            c * base_only["RA-FC"][base].mean()
        uncovered = (fcg_only["Ctx.R"]["FCG-RAG"] == 0).mean()
        both = piv[fcg & other]
        d = (both["RA-FC"]["FCG-RAG"] - both["RA-FC"][base]).to_numpy()
        n = len(d)
        lo, hi = np.percentile(
            d[rng.integers(0, n, (N_BOOT, n))].mean(axis=1), [2.5, 97.5])
        print(f"FCG-RAG vs {base}")
        print(f"  FCG-only={b} (RA-FC {fcg_only['RA-FC']['FCG-RAG'].mean():.3f}), "
              f"{base}-only={c} (RA-FC {base_only['RA-FC'][base].mean():.3f}), "
              f"ratio {b / c:.1f}x, McNemar p={mcnemar_p(b, c):.1e}")
        print(f"  net quality contribution {net:+.1f}; "
              f"FCG-only items in Ctx.R=0 stratum: {uncovered:.0%}")
        print(f"  both-answered n={n}: paired diff {d.mean():+.3f} "
              f"95% CI [{lo:.3f}, {hi:.3f}]")


if __name__ == "__main__":
    main()
