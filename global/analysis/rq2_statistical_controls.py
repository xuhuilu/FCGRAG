#!/usr/bin/env python3
"""RQ2: statistical controls (logistic regression and answered-item OLS).

Reproduces: logit P(answer) ~ hit*method + log CtxTok + system with
item-clustered SEs over 2296 observations -- FCG-RAG OR=2.14 [1.15, 3.99]
p=0.016; hit OR=5.91 [3.86, 9.04]; log CtxTok p=0.666; hit x FCG-RAG
OR=1.57 [0.80, 3.08] p=0.195. OLS on answered items: FCG-RAG RA-FC
coefficient -0.039 (p=0.470), hit +0.184 (p<0.001); Faith coefficient
-0.062 (p=0.280).

Requires statsmodels.
"""
import numpy as np
import pandas as pd
import statsmodels.formula.api as smf

from common import load_results


def main() -> None:
    df = load_results()
    df["hit"] = (df["Ctx.R"] > 0).astype(int)
    df["log_tok"] = np.log(df["CtxTok"])
    df["answered"] = df["Ans.R"].astype(int)
    df["method"] = pd.Categorical(
        df["method"], categories=["Code-RAG", "Report-RAG", "Int-RAG", "FCG-RAG"])

    logit = smf.logit("answered ~ hit*method + log_tok + C(system)", data=df)
    res = logit.fit(cov_type="cluster", cov_kwds={"groups": df["item_id"]}, disp=0)
    print("logistic regression: answered ~ hit*method + log CtxTok + system "
          "(item-clustered SE, n=2296)")
    for k in res.params.index:
        if k == "Intercept":
            continue
        ci = res.conf_int().loc[k]
        print(f"  {k:32s} coef={res.params[k]:+.3f} OR={np.exp(res.params[k]):5.2f} "
              f"OR-CI=[{np.exp(ci[0]):5.2f},{np.exp(ci[1]):5.2f}] p={res.pvalues[k]:.4f}")

    a = df[df.answered == 1]
    ols = smf.ols('Q("RA-FC") ~ hit*method + log_tok + C(system)', data=a)
    r2 = ols.fit(cov_type="cluster", cov_kwds={"groups": a["item_id"]})
    print("\nOLS on answered items: RA-FC ~ hit*method + log CtxTok + system")
    for k in ["method[T.FCG-RAG]", "hit", "hit:method[T.FCG-RAG]"]:
        print(f"  {k:32s} coef={r2.params[k]:+.3f} p={r2.pvalues[k]:.4f}")
    r3 = smf.ols("Faith ~ hit*method + log_tok + C(system)", data=a).fit(
        cov_type="cluster", cov_kwds={"groups": a["item_id"]})
    print(f"\nOLS Faith | answered: FCG-RAG coef="
          f"{r3.params['method[T.FCG-RAG]']:+.3f} "
          f"p={r3.pvalues['method[T.FCG-RAG]']:.4f}")


if __name__ == "__main__":
    main()
