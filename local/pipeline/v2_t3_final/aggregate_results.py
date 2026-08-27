#!/usr/bin/env python3
"""Aggregate final 572-item v2 results.

Reads scores/scores_v22_all_methods.jsonl (2288 long-format records) and
computes:
  - overall / by-dataset / by-qtype / by-method means (Struct, Expl, Consist,
    Qual=(Struct+Expl)/2), on the 572-item denominator
  - pairwise significance: exact two-sided sign test + paired bootstrap 95% CI
    (4000 resamples, percentile, seeded) for all 6 method pairs x 4 metrics
    x (overall + qtype + dataset scopes)
  - confidence intervals: per method x metric x scope bootstrap CI of the mean

Significance/CI methods are identical to analyze_three_methods_clean_v1_en.py
(frozen). No v1/632 denominator is used for v2 aggregates.
"""
from __future__ import annotations

import csv
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analyze_three_methods_clean_v1_en import bootstrap_ci, exact_sign_test_pvalue  # noqa: E402

from v2_t3_final import (  # noqa: E402
    AGG_DIR,
    DATASETS,
    METHODS,
    SCORE_DIR,
    SIG_DIR,
    ensure_out_dirs,
    load_jsonl,
)

M_LABEL = {"direct": "Direct", "code_only": "Code-only",
           "source_ast": "Code+AST", "fcg_rag": "FCG-RAG"}
METRICS = ["struct", "expl", "consist", "qual"]
N_BOOT = 4000


def qual(s):
    return (s["structure"] + s["explanation"]) / 2


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def write_csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def main() -> int:
    ensure_out_dirs()
    rows = load_jsonl(SCORE_DIR / "scores_v22_all_methods.jsonl")
    assert len(rows) == 2288, len(rows)
    for r in rows:
        r["qual"] = round(qual(r), 6)

    # index: (scope, bucket) -> method -> list of records
    by_item = defaultdict(dict)  # item_id -> method -> rec
    for r in rows:
        by_item[r["item_id"]][r["method"]] = r

    item_meta = {r["item_id"]: (r["dataset"], r["question_type"]) for r in rows}

    # scopes
    scopes = {"overall": list(by_item.keys())}
    for ds in DATASETS:
        scopes[f"dataset:{ds}"] = [i for i in by_item if item_meta[i][0] == ds]
    for q in sorted({item_meta[i][1] for i in by_item}):
        scopes[f"qtype:{q}"] = [i for i in by_item if item_meta[i][1] == q]

    # ---- aggregate tables ----
    def agg_table(bucket_fn, scope_name):
        out = []
        for scope, item_ids in scopes.items():
            if not item_ids:
                continue
            for m in METHODS:
                vals = [by_item[i][m] for i in item_ids if m in by_item[i]]
                if not vals:
                    continue
                out.append({
                    "scope": scope,
                    "bucket": bucket_fn(scope),
                    "n_items": len(vals),
                    "method": m,
                    "method_label": M_LABEL[m],
                    "struct": round(mean([v["structure"] for v in vals]), 4),
                    "expl": round(mean([v["explanation"] for v in vals]), 4),
                    "consist": round(mean([v["consistency"] for v in vals]), 4),
                    "qual": round(mean([v["qual"] for v in vals]), 4),
                })
        return out

    overall = agg_table(lambda s: "overall", None)
    by_dataset = [r for r in overall if r["scope"].startswith("dataset:")]
    by_qtype = [r for r in overall if r["scope"].startswith("qtype:")]
    overall_only = [r for r in overall if r["scope"] == "overall"]

    fields_agg = ["scope", "bucket", "n_items", "method", "method_label",
                  "struct", "expl", "consist", "qual"]
    write_csv(AGG_DIR / "aggregate_overall_v22.csv", overall_only, fields_agg)
    write_csv(AGG_DIR / "aggregate_by_dataset_v22.csv", by_dataset, fields_agg)
    write_csv(AGG_DIR / "aggregate_by_qtype_v22.csv", by_qtype, fields_agg)
    write_csv(AGG_DIR / "aggregate_by_method_v22.csv", overall_only, fields_agg)

    # ---- pairwise significance ----
    sig_rows = []
    ci_rows = []
    metric_field = {"struct": "structure", "expl": "explanation",
                    "consist": "consistency", "qual": "qual"}
    for scope, item_ids in scopes.items():
        if not item_ids:
            continue
        for mi in METRICS:
            fld = metric_field[mi]
            # per-method CI of mean
            for m in METHODS:
                vals = [by_item[i][m][fld] for i in item_ids if m in by_item[i]]
                if not vals:
                    continue
                lo, hi = bootstrap_ci(
                    [v - mean(vals) for v in vals], n_boot=N_BOOT,
                    seed=hash((scope, mi, m, "cimean")) & 0xFFFFFFFF,
                )
                ci_rows.append({
                    "scope": scope, "metric": mi, "method": m,
                    "n_items": len(vals), "mean": round(mean(vals), 4),
                    "ci_low": round(mean(vals) + lo, 4), "ci_high": round(mean(vals) + hi, 4),
                })
            # pairwise
            for a in range(len(METHODS)):
                for b in range(a + 1, len(METHODS)):
                    ma, mb = METHODS[a], METHODS[b]
                    common = [i for i in item_ids if ma in by_item[i] and mb in by_item[i]]
                    if not common:
                        continue
                    av = [by_item[i][ma][fld] for i in common]
                    bv = [by_item[i][mb][fld] for i in common]
                    diffs = [x - y for x, y in zip(av, bv)]
                    wins = sum(1 for d in diffs if d > 1e-9)
                    losses = sum(1 for d in diffs if d < -1e-9)
                    ties = len(diffs) - wins - losses
                    lo, hi = bootstrap_ci(
                        diffs, n_boot=N_BOOT,
                        seed=hash((scope, mi, ma, mb)) & 0xFFFFFFFF,
                    )
                    sig_rows.append({
                        "scope": scope, "metric": mi,
                        "method_a": ma, "method_b": mb,
                        "n_items": len(common),
                        "mean_a": round(mean(av), 4), "mean_b": round(mean(bv), 4),
                        "mean_diff_a_minus_b": round(mean(diffs), 4),
                        "wins_a": wins, "ties": ties, "losses_a": losses,
                        "sign_test_p": round(exact_sign_test_pvalue(wins, losses), 6),
                        "bootstrap_ci_low": lo, "bootstrap_ci_high": hi,
                    })

    write_csv(SIG_DIR / "pairwise_significance_v22.csv", sig_rows,
              ["scope", "metric", "method_a", "method_b", "n_items", "mean_a", "mean_b",
               "mean_diff_a_minus_b", "wins_a", "ties", "losses_a", "sign_test_p",
               "bootstrap_ci_low", "bootstrap_ci_high"])
    write_csv(SIG_DIR / "confidence_intervals_v22.csv", ci_rows,
              ["scope", "metric", "method", "n_items", "mean", "ci_low", "ci_high"])

    print("aggregates written.")
    print("overall means (Qual):")
    for r in overall_only:
        print(f"  {r['method_label']:<10} struct={r['struct']} expl={r['expl']} "
              f"consist={r['consist']} qual={r['qual']}")
    # ranking
    rank = sorted(overall_only, key=lambda r: -r["qual"])
    print("Qual ranking: " + " > ".join(r["method_label"] for r in rank))
    return 0


if __name__ == "__main__":
    sys.exit(main())
