"""Emit the final T3.7 / T3.8 input plans for the accepted 572-item v2 candidate.

Deterministic, local-only. Reads the candidate mappings and before/after
reports; writes:

- reports/T3_7_FINAL_INPUT_PLAN.csv        (all 632 v1 items, dispositions)
- reports/T3_7_METRIC_DEPENDENCY_PLAN.csv  (572 items x 4 methods)
- reports/T3_8_RERUN_ITEM_LIST.csv         (23 items x 4 generation calls)

Metric-dependency rule (per author instruction T3.6A, step 5):
- prediction, gold fields, and judge inputs unchanged -> REUSE_ALL_SCORES.
- The two LLM judges (claim coverage, quality) consume item["gold_answer"],
  question, and the model response (fcg_bench/explanation_eval.py:
  judge_explanation). The free-text gold *explanation* field is NOT a judge
  input and NOT a scored field, so FIX_EXPLANATION-only items reuse all
  scores.
- gold_answer changed -> claims and gold summary change -> both judges must
  re-run (REJUDGE_BOTH); struct metrics are recomputed locally.
- label-only change with unchanged gold hash (ollama_go_0014, score-neutral
  relabel) -> RECOMPUTE_STRUCT_ONLY (deterministic, no LLM judge).
- question changed -> all such items are RERUN_ALL_4 -> WAIT_FOR_RERUN.
"""

import csv
import json
import os

BASE = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    "experiments", "multilingual_en",
    "benchmarks_with_explanations_v22_t3_remediated",
)
MAPPINGS = os.path.join(BASE, "mappings")
REPORTS = os.path.join(BASE, "reports")
OUT_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "reports",
)

METHODS = ["code_only", "direct", "fcg_rag", "source_ast"]


def load_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    mapping = load_csv(os.path.join(MAPPINGS, "v1_to_v2_item_mapping.csv"))
    assert len(mapping) == 632, f"expected 632 mapping rows, got {len(mapping)}"

    gold_changed = {r["item_id"] for r in load_csv(os.path.join(REPORTS, "gold_before_after.csv"))}
    expl_changed = {r["item_id"] for r in load_csv(os.path.join(REPORTS, "explanation_before_after.csv"))}
    label_changed = {r["item_id"] for r in load_csv(os.path.join(REPORTS, "label_before_after.csv"))}

    # ---- 1. T3_7_FINAL_INPUT_PLAN.csv (all 632) --------------------------
    input_plan = os.path.join(OUT_DIR, "T3_7_FINAL_INPUT_PLAN.csv")
    counts = {}
    with open(input_plan, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "dataset", "qtype", "action",
                    "prediction_disposition", "v2_item_id", "planned_generation_calls"])
        for r in mapping:
            d = r["prediction_disposition"]
            counts[d] = counts.get(d, 0) + 1
            calls = 4 if d == "RERUN_ALL_4" else 0
            w.writerow([r["original_item_id"], r["source_system"], r["qtype"],
                        r["action"], d, r["v2_item_id"], calls])
    reuse = counts.get("REUSE_UNCHANGED", 0)
    reev = counts.get("REUSE_REEVALUATE", 0)
    rerun = counts.get("RERUN_ALL_4", 0)
    drop = counts.get("DROP_DEDUPLICATE", 0) + counts.get("DROP_EXCLUDE", 0)
    assert reuse + reev + rerun + drop == 632

    # ---- 2. T3_7_METRIC_DEPENDENCY_PLAN.csv (572 x 4) --------------------
    metric_plan = os.path.join(OUT_DIR, "T3_7_METRIC_DEPENDENCY_PLAN.csv")
    n_struct = n_judge = 0
    with open(metric_plan, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "method", "metric_action",
                    "local_struct_recompute", "llm_judge_required", "notes"])
        for r in mapping:
            iid = r["original_item_id"]
            d = r["prediction_disposition"]
            if d.startswith("DROP"):
                continue  # drops do not enter v2 results
            if d == "REUSE_UNCHANGED":
                action, struct, judge, note = "REUSE_ALL_SCORES", "no", "no", ""
            elif d == "RERUN_ALL_4":
                action, struct, judge, note = "WAIT_FOR_RERUN", "deferred", "deferred", "question/root changed; all scores after T3.8 rerun"
            elif iid in gold_changed:
                action, struct, judge, note = "REJUDGE_BOTH", "yes", "yes", "gold_answer changed; claim and quality judge inputs changed"
                n_struct += 1
                n_judge += 1
            elif iid in expl_changed:
                action, struct, judge, note = (
                    "REUSE_ALL_SCORES", "no", "no",
                    "gold explanation text is not a scored field and not a judge input "
                    "(judge_explanation consumes gold_answer/question/response only)",
                )
            elif iid in label_changed:
                action, struct, judge, note = "RECOMPUTE_STRUCT_ONLY", "yes", "no", "score-neutral relabel; gold hash unchanged"
                n_struct += 1
            else:
                raise AssertionError(f"unclassified REUSE_REEVALUATE item: {iid}")
            for m in METHODS:
                w.writerow([iid, m, action, struct, judge, note])

    # ---- 3. T3_8_RERUN_ITEM_LIST.csv (23 items) --------------------------
    rerun_plan = os.path.join(OUT_DIR, "T3_8_RERUN_ITEM_LIST.csv")
    with open(rerun_plan, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["item_id", "dataset", "qtype", "action", "methods", "planned_generation_calls"])
        for r in mapping:
            if r["prediction_disposition"] == "RERUN_ALL_4":
                w.writerow([r["original_item_id"], r["source_system"], r["qtype"],
                            r["action"], ";".join(METHODS), 4])

    summary = {
        "dispositions": counts,
        "reuse_unchanged": reuse,
        "reuse_reevaluate": reev,
        "rerun_all_4": rerun,
        "drop_total": drop,
        "sum_check": reuse + reev + rerun + drop,
        "metric_plan_records": 572 * len(METHODS),
        "local_struct_recompute_items": n_struct,
        "local_struct_recompute_records": n_struct * len(METHODS),
        "llm_judge_items": n_judge,
        "llm_judge_records": n_judge * len(METHODS),
        "rerun_generation_calls": rerun * len(METHODS),
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
