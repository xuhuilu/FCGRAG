#!/usr/bin/env python3
"""T3.7 prediction rebinding (no API calls).

Builds the reuse portion of the v2 prediction/score bundle:
- 2140 records (457 REUSE_UNCHANGED + 78 REUSE_REEVALUATE items x 4 methods).

Unified normalization rule (per author directive):
  - Struct is recomputed for ALL records with the frozen v2 evaluator
    (values taken from reports/T3_7_FULL_STRUCT_RECOMPUTE.csv).
  - Explanation and Consistency (LLM-judge scores) are reused from the frozen
    v1 hybrid bundle UNLESS the record is REJUDGE_BOTH (gold change alters a
    judge input). REJUDGE_BOTH records carry pending explanation/consistency,
    filled by reevaluate_t3_7.py.
  - The 92 RERUN_ALL_4 records are produced by rerun_t3_8.py.

Hard stops:
  - any reused prediction row with error / empty raw_output / empty structure
  - missing v2 struct value for a reused record
  - any reused prediction whose raw content differs from the v1 line hash
  - dropped item leaking into the bundle
"""
from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fcg_benchmark/

from v2_t3_final import (  # noqa: E402
    DATASETS,
    EXT_RUN_DIR,
    MANIFEST_DIR,
    CHECKPOINT_DIR,
    METHODS,
    METRIC_PLAN_CSV,
    PLAN_CSV,
    V1_SCORE_DIR,
    V2_BENCH,
    ensure_out_dirs,
    load_jsonl,
    sha256_file,
    sha256_text,
    verify_frozen_inputs,
    write_jsonl,
)

REPO = Path("/home/lxh/fcg-v2-implementation")
STRUCT_RECOMPUTE_CSV = REPO / "reports" / "T3_7_FULL_STRUCT_RECOMPUTE.csv"


def load_plan() -> dict[str, dict]:
    with PLAN_CSV.open() as f:
        return {r["item_id"]: r for r in csv.DictReader(f)}


def load_metric_plan() -> dict[tuple[str, str], dict]:
    with METRIC_PLAN_CSV.open() as f:
        return {(r["item_id"], r["method"]): r for r in csv.DictReader(f)}


def load_v2_struct() -> dict[tuple[str, str], dict]:
    out = {}
    with STRUCT_RECOMPUTE_CSV.open() as f:
        for r in csv.DictReader(f):
            out[(r["item_id"], r["method"])] = r
    return out


def load_v1_predictions() -> dict[tuple[str, str, str], dict]:
    out = {}
    for ds in DATASETS:
        for m in METHODS:
            path = EXT_RUN_DIR / f"{ds}_pred_{m}_v2_en.jsonl"
            file_sha = sha256_file(path)
            for idx, line in enumerate(path.read_bytes().split(b"\n")):
                if not line.strip():
                    continue
                row = json.loads(line.decode("utf-8"))
                key = (ds, m, row["item_id"])
                assert key not in out, f"duplicate v1 pred row: {key}"
                out[key] = {
                    "pred_file": path.name,
                    "pred_file_sha256": file_sha,
                    "row_index": idx,
                    "line_sha256": hashlib.sha256(line).hexdigest(),
                    "row": row,
                }
    return out


def load_v1_scores() -> dict[str, dict]:
    out = {}
    for ds in DATASETS:
        for row in load_jsonl(V1_SCORE_DIR / f"{ds}_per_item_scores_v2_en.jsonl"):
            out[row["item_id"]] = row
    return out


def main() -> None:
    ensure_out_dirs()
    verify_frozen_inputs()

    plan = load_plan()
    metric_plan = load_metric_plan()
    v2_struct = load_v2_struct()
    v2_items = {it["item_id"]: it for it in load_jsonl(V2_BENCH)}
    assert len(v2_items) == 572
    v1_preds = load_v1_predictions()
    v1_scores = load_v1_scores()

    predictions = []
    scores = []
    rejudge_worklist = []

    for item_id in sorted(v2_items):
        p = plan[item_id]
        disp = p["prediction_disposition"]
        if disp == "RERUN_ALL_4":
            continue
        assert disp in ("REUSE_UNCHANGED", "REUSE_REEVALUATE"), (item_id, disp)
        ds = p["dataset"]
        item = v2_items[item_id]
        v1_score_row = v1_scores[item_id]

        for method in METHODS:
            mp = metric_plan[(item_id, method)]
            ma = mp["metric_action"]
            src = v1_preds[(ds, method, item_id)]
            row = src["row"]

            # integrity of reused prediction content
            assert row.get("error") in (None, "", "None"), (
                f"v1 prediction has error: {ds}|{method}|{item_id}: {row.get('error')}"
            )
            assert (row.get("raw_output") or "").strip(), (
                f"empty raw_output: {ds}|{method}|{item_id}"
            )
            sa = row.get("structure_answer") or {}
            assert isinstance(sa, dict) and sa, f"empty struct: {ds}|{method}|{item_id}"
            assert (row.get("explanation_answer") or "").strip(), (
                f"empty explanation: {ds}|{method}|{item_id}"
            )

            pred_sha = sha256_text(
                json.dumps(
                    {
                        "structure_answer": sa,
                        "explanation_answer": row["explanation_answer"],
                        "raw_output": row["raw_output"],
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                )
            )
            predictions.append({
                "item_id": item_id,
                "dataset": ds,
                "question_type": item["question_type"],
                "root_function": item["root_function"],
                "method": method,
                "structure_answer": sa,
                "explanation_answer": row["explanation_answer"],
                "raw_output": row["raw_output"],
                "error": None,
                "binding": "reused_v1",
                "metric_action": ma,
                "source": {
                    "pred_file": src["pred_file"],
                    "pred_file_sha256": src["pred_file_sha256"],
                    "row_index": src["row_index"],
                    "line_sha256": src["line_sha256"],
                },
                "question_sha256_v2": sha256_text(item["question"]),
                "prediction_sha256": pred_sha,
            })

            # unified v2 struct
            sr = v2_struct[(item_id, method)]
            struct_v2 = float(sr["v2_struct_final"])
            assert sr["attribution"] != "ERROR", f"missing struct recompute: {item_id}|{method}"

            old_e = v1_score_row[f"{method}_explanation"]
            old_c = v1_score_row[f"{method}_consistency"]

            if ma in ("REUSE_ALL_SCORES", "RECOMPUTE_STRUCT_ONLY"):
                # judge inputs unchanged -> reuse v1 explanation/consistency
                scores.append({
                    "item_id": item_id,
                    "dataset": ds,
                    "question_type": item["question_type"],
                    "method": method,
                    "structure": struct_v2,
                    "explanation": old_e,
                    "consistency": old_c,
                    "combined": round((struct_v2 + old_e + old_c) / 3, 4),
                    "struct_origin": "v2_evaluator_recompute",
                    "judge_origin": "reused_v1",
                    "struct_v1": float(sr["v1_struct"]),
                    "struct_delta": float(sr["delta"]),
                    "struct_attribution": sr["attribution"],
                })
            elif ma == "REJUDGE_BOTH":
                # gold change alters a judge input -> rejudge explanation+consistency
                scores.append({
                    "item_id": item_id,
                    "dataset": ds,
                    "question_type": item["question_type"],
                    "method": method,
                    "structure": struct_v2,
                    "explanation": None,
                    "consistency": None,
                    "combined": None,
                    "struct_origin": "v2_evaluator_recompute",
                    "judge_origin": "pending_rejudge",
                    "struct_v1": float(sr["v1_struct"]),
                    "struct_delta": float(sr["delta"]),
                    "struct_attribution": sr["attribution"],
                    "judge_v1": {"explanation": old_e, "consistency": old_c},
                })
                rejudge_worklist.append({
                    "item_id": item_id,
                    "dataset": ds,
                    "question_type": item["question_type"],
                    "method": method,
                })
            else:
                raise AssertionError(f"unexpected metric_action {ma} for {item_id}|{method}")

    # dropped items must never appear
    drop_ids = {iid for iid, p in plan.items() if p["prediction_disposition"].startswith("DROP")}
    assert sum(1 for r in predictions if r["item_id"] in drop_ids) == 0

    assert len(predictions) == 2140, len(predictions)
    assert len(scores) == 2140, len(scores)
    assert len(rejudge_worklist) == 240, len(rejudge_worklist)

    write_jsonl(CHECKPOINT_DIR / "rebind_predictions.jsonl", predictions)
    write_jsonl(CHECKPOINT_DIR / "rebind_scores.jsonl", scores)
    write_jsonl(MANIFEST_DIR / "rejudge_worklist.jsonl", rejudge_worklist)

    n_reuse_judge = sum(1 for s in scores if s["judge_origin"] == "reused_v1")
    n_pending = sum(1 for s in scores if s["judge_origin"] == "pending_rejudge")
    summary = {
        "reused_prediction_records": len(predictions),
        "struct_unified_v2_recompute": len(scores),
        "judge_reused_v1": n_reuse_judge,
        "judge_pending_rejudge": n_pending,
        "rejudge_worklist_records": len(rejudge_worklist),
        "planned_rejudge_api_calls": len(rejudge_worklist) * 2,
    }
    (MANIFEST_DIR / "rebind_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
