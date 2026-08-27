#!/usr/bin/env python3
"""v1 -> v2 result difference and contribution decomposition.

Compares the frozen v1 results (632 items, wide per-item scores) against the
final v2 results (572 items), decomposing the overall change per method into:
  - deletion contribution : v1(632) vs v1 restricted to the 572 retained items
  - retained change       : v2(572) vs v1(572 retained), further split by item
                            disposition group (unchanged / fix_explanation /
                            relabel / gold-changed reeval / rerun)

Qual = (Struct+Expl)/2 (paper metric). Outputs reports/V1_V2_RESULT_DIFF.csv.
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2_t3_final import (  # noqa: E402
    DATASETS,
    METHODS,
    PLAN_CSV,
    SCORE_DIR,
    V1_SCORE_DIR,
    load_jsonl,
)

REPO_REPORTS = Path("/home/lxh/fcg-v2-implementation/reports")
M_LABEL = {"direct": "Direct", "code_only": "Code-only",
           "source_ast": "Code+AST", "fcg_rag": "FCG-RAG"}
GROUPS = ["unchanged", "fix_explanation", "relabel", "gold_changed_reeval", "rerun"]


def qual(s, e):
    return (s + e) / 2


def mean(xs):
    return sum(xs) / len(xs) if xs else 0.0


def main() -> int:
    # v1 long (632)
    v1 = {}
    for ds in DATASETS:
        for row in load_jsonl(V1_SCORE_DIR / f"{ds}_per_item_scores_v2_en.jsonl"):
            for m in METHODS:
                v1[(row["item_id"], m)] = {
                    "struct": float(row[f"{m}_structure"]),
                    "expl": float(row[f"{m}_explanation"]),
                    "consist": float(row[f"{m}_consistency"]),
                    "qual": qual(float(row[f"{m}_structure"]), float(row[f"{m}_explanation"])),
                }
    # v2 long (572)
    v2 = {(r["item_id"], r["method"]): {
        "struct": r["structure"], "expl": r["explanation"],
        "consist": r["consistency"], "qual": qual(r["structure"], r["explanation"]),
    } for r in load_jsonl(SCORE_DIR / "scores_v22_all_methods.jsonl")}

    v2_items = {iid for (iid, _) in v2}
    v1_items = {iid for (iid, _) in v1}
    retained = v1_items & v2_items  # 572

    # disposition groups
    group_of = {}
    with PLAN_CSV.open() as f:
        for r in csv.DictReader(f):
            d = r["prediction_disposition"]
            a = r["action"]
            iid = r["item_id"]
            if d == "REUSE_UNCHANGED":
                g = "unchanged"
            elif d == "RERUN_ALL_4":
                g = "rerun"
            elif d.startswith("DROP"):
                g = "dropped"
            elif a == "FIX_EXPLANATION":
                g = "fix_explanation"
            elif iid == "ollama_cpp_0026":
                g = "relabel"
            else:
                g = "gold_changed_reeval"
            group_of[iid] = g

    rows = []
    for m in METHODS:
        v1_632 = [v1[(i, m)] for i in v1_items if (i, m) in v1]
        v1_ret = [v1[(i, m)] for i in retained if (i, m) in v1]
        v2_ret = [v2[(i, m)] for i in retained if (i, m) in v2]
        rec = {
            "method": m, "method_label": M_LABEL[m],
            "n_v1_632": len(v1_632), "n_retained": len(retained), "n_v2_572": len(v2_ret),
        }
        for tag, src in [("v1_632", v1_632), ("v1_retained", v1_ret), ("v2", v2_ret)]:
            for met in ("struct", "expl", "consist", "qual"):
                rec[f"{tag}_{met}"] = round(mean([x[met] for x in src]), 4)
        # contributions (Qual)
        rec["deletion_contrib_qual"] = round(rec["v1_632_qual"] - rec["v1_retained_qual"], 4)
        rec["retained_change_qual"] = round(rec["v2_qual"] - rec["v1_retained_qual"], 4)
        rec["total_change_qual"] = round(rec["v2_qual"] - rec["v1_632_qual"], 4)
        # per-group delta qual on retained
        for g in GROUPS:
            g_items = [i for i in retained if group_of.get(i) == g]
            deltas = [v2[(i, m)]["qual"] - v1[(i, m)]["qual"]
                      for i in g_items if (i, m) in v2 and (i, m) in v1]
            rec[f"group_{g}_n"] = len(g_items)
            rec[f"group_{g}_mean_delta_qual"] = round(mean(deltas), 4) if deltas else 0.0
        rows.append(rec)

    fields = list(rows[0].keys())
    out = REPO_REPORTS / "V1_V2_RESULT_DIFF.csv"
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}")

    # ranking
    rank_v1 = sorted(rows, key=lambda r: -r["v1_632_qual"])
    rank_v2 = sorted(rows, key=lambda r: -r["v2_qual"])
    print("v1(632) Qual ranking: " + " > ".join(r["method_label"] for r in rank_v1))
    print("v2(572) Qual ranking: " + " > ".join(r["method_label"] for r in rank_v2))
    print("ranking changed:", [r["method_label"] for r in rank_v1] != [r["method_label"] for r in rank_v2])
    return 0


if __name__ == "__main__":
    sys.exit(main())
