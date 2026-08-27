#!/usr/bin/env python3
"""Assemble the final v2 prediction + score bundles (2288 records each).

Sources:
  predictions:
    - rebind_predictions.jsonl     : 2140 reused-v1 predictions
    - rerun_gen_checkpoint.jsonl   : 148 rerun-v2 predictions (success only)
  scores:
    - rebind_scores.jsonl          : 2140 (1900 reused judge + 240 pending)
        pending explanation/consistency filled from reevaluate_checkpoint.jsonl
    - rerun_judge_checkpoint.jsonl : 148 rerun scores

Performs the full integrity gate (spec section 7):
  - 572 item ids x 4 methods = 2288, 0 dup, 0 missing
  - question/root hashes match the v2 benchmark
  - dropped items absent
  - 0 placeholder/error output, 0 schema-invalid prediction
  - 0 NaN/Inf, 0 missing scores

Writes:
  predictions/predictions_v22_all_methods.jsonl
  scores/scores_v22_all_methods.jsonl
"""
from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2_t3_final import (  # noqa: E402
    CHECKPOINT_DIR,
    METHODS,
    MAPPING_CSV,
    OUT_DIR,
    PLAN_CSV,
    PRED_DIR,
    SCORE_DIR,
    V2_BENCH,
    load_jsonl,
    sha256_text,
    write_jsonl,
)

REPO = Path("/home/lxh/fcg-v2-implementation")


def is_finite_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def main() -> int:
    v2_items = {it["item_id"]: it for it in load_jsonl(V2_BENCH)}
    assert len(v2_items) == 572

    # dropped item ids (must not appear)
    drop_ids = set()
    with PLAN_CSV.open() as f:
        for r in csv.DictReader(f):
            if r["prediction_disposition"].startswith("DROP"):
                drop_ids.add(r["item_id"])

    # ---- predictions ----
    rebind_preds = {(r["item_id"], r["method"]): r for r in load_jsonl(CHECKPOINT_DIR / "rebind_predictions.jsonl")}
    assert len(rebind_preds) == 2140

    rerun_gen = {}
    for r in load_jsonl(CHECKPOINT_DIR / "rerun_gen_checkpoint.jsonl"):
        if r.get("final_status") == "success":
            rerun_gen[(r["item_id"], r["method"])] = r
    assert len(rerun_gen) == 148, f"rerun gen success={len(rerun_gen)} != 148"

    predictions = []
    for (iid, method), rp in rebind_preds.items():
        item = v2_items[iid]
        predictions.append({
            "item_id": iid,
            "dataset": rp["dataset"],
            "question_type": rp["question_type"],
            "root_function": item["root_function"],
            "method": method,
            "structure_answer": rp["structure_answer"],
            "explanation_answer": rp["explanation_answer"],
            "raw_output": rp["raw_output"],
            "error": None,
            "binding": "reused_v1",
            "source": rp["source"],
            "prediction_sha256": rp["prediction_sha256"],
            "question_sha256_v2": sha256_text(item["question"]),
        })
    for (iid, method), g in rerun_gen.items():
        item = v2_items[iid]
        predictions.append({
            "item_id": iid,
            "dataset": g["dataset"],
            "question_type": g["qtype"],
            "root_function": item["root_function"],
            "method": method,
            "structure_answer": g["structure_answer"],
            "explanation_answer": g["explanation_answer"],
            "raw_output": g["raw_output"],
            "error": None,
            "binding": "rerun_v2",
            "source": {
                "pred_file": "rerun_gen_checkpoint.jsonl",
                "raw_output_sha256": g["raw_output_sha256"],
                "prompt_sha256": g["prompt_sha256"],
                "context_sha256": g["context_sha256"],
                "request_id": g.get("request_id"),
                "retry_count": g.get("retry_count", 0),
                "model": g.get("model"),
                "endpoint": g.get("endpoint"),
            },
            "prediction_sha256": g["prediction_sha256"],
            "question_sha256_v2": sha256_text(item["question"]),
        })

    # ---- scores ----
    rebind_scores = {(r["item_id"], r["method"]): r for r in load_jsonl(CHECKPOINT_DIR / "rebind_scores.jsonl")}
    assert len(rebind_scores) == 2140

    reeval = {(r["item_id"], r["method"]): r for r in load_jsonl(CHECKPOINT_DIR / "reevaluate_checkpoint.jsonl")
              if r.get("status") == "success"}
    assert len(reeval) == 296, f"reeval success={len(reeval)} != 296"

    rerun_judge = {(r["item_id"], r["method"]): r for r in load_jsonl(CHECKPOINT_DIR / "rerun_judge_checkpoint.jsonl")}
    assert len(rerun_judge) == 148, f"rerun judge={len(rerun_judge)} != 148"

    scores = []
    for (iid, method), rs in rebind_scores.items():
        item = v2_items[iid]
        struct = rs["structure"]
        if rs["judge_origin"] == "pending_rejudge":
            rj = reeval[(iid, method)]
            expl = rj["explanation"]["total_score"]
            cons = rj["consistency"]["total_score"]
            judge_origin = "rejudged_v2"
        else:
            expl = rs["explanation"]
            cons = rs["consistency"]
            judge_origin = "reused_v1"
        scores.append({
            "item_id": iid,
            "dataset": rs["dataset"],
            "question_type": rs["question_type"],
            "method": method,
            "structure": struct,
            "explanation": expl,
            "consistency": cons,
            "combined": round((struct + expl + cons) / 3, 4),
            "struct_origin": rs["struct_origin"],
            "judge_origin": judge_origin,
        })
    for (iid, method), rj in rerun_judge.items():
        struct = rj["structure"]
        expl = rj["explanation"]["total_score"]
        cons = rj["consistency"]["total_score"]
        scores.append({
            "item_id": iid,
            "dataset": rj["dataset"],
            "question_type": rj["qtype"],
            "method": method,
            "structure": struct,
            "explanation": expl,
            "consistency": cons,
            "combined": round((struct + expl + cons) / 3, 4),
            "struct_origin": "v2_evaluator_rerun",
            "judge_origin": "rejudged_v2",
        })

    # ============ INTEGRITY GATE ============
    errors = []

    def e(msg):
        errors.append(msg)

    pred_keys = {(p["item_id"], p["method"]) for p in predictions}
    score_keys = {(s["item_id"], s["method"]) for s in scores}
    expected = {(iid, m) for iid in v2_items for m in METHODS}

    if len(predictions) != 2288:
        e(f"predictions count {len(predictions)} != 2288")
    if len(scores) != 2288:
        e(f"scores count {len(scores)} != 2288")
    if len(pred_keys) != len(predictions):
        e(f"duplicate item x method in predictions ({len(pred_keys)} vs {len(predictions)})")
    if len(score_keys) != len(scores):
        e(f"duplicate item x method in scores ({len(score_keys)} vs {len(scores)})")
    if pred_keys != expected:
        e(f"prediction key set mismatch: missing={len(expected-pred_keys)} extra={len(pred_keys-expected)}")
    if score_keys != expected:
        e(f"score key set mismatch: missing={len(expected-score_keys)} extra={len(score_keys-expected)}")

    # each item exactly 4 methods
    from collections import Counter
    pm = Counter(p["item_id"] for p in predictions)
    sm = Counter(s["item_id"] for s in scores)
    for iid in v2_items:
        if pm.get(iid, 0) != 4:
            e(f"item {iid} has {pm.get(iid,0)} predictions (expect 4)")
        if sm.get(iid, 0) != 4:
            e(f"item {iid} has {sm.get(iid,0)} scores (expect 4)")

    # dropped items absent
    leaked = [p["item_id"] for p in predictions if p["item_id"] in drop_ids]
    if leaked:
        e(f"dropped items leaked into predictions: {set(leaked)}")

    # question hash matches v2 benchmark
    qh = {it["item_id"]: sha256_text(it["question"]) for it in load_jsonl(V2_BENCH)}
    bad_q = [p["item_id"] for p in predictions if p["question_sha256_v2"] != qh[p["item_id"]]]
    if bad_q:
        e(f"question hash mismatch on {len(bad_q)} predictions")

    # no placeholder/error output
    bad_pred = [p["item_id"] + "|" + p["method"] for p in predictions
                if p["error"] or not (p["raw_output"] or "").strip()
                or not (p.get("explanation_answer") or "").strip()
                or not isinstance(p.get("structure_answer"), dict) or not p["structure_answer"]]
    if bad_pred:
        e(f"placeholder/error predictions: {len(bad_pred)} ({bad_pred[:3]})")

    # scores finite, in range, no NaN/Inf
    bad_sc = []
    for s in scores:
        for f in ("structure", "explanation", "consistency", "combined"):
            v = s[f]
            if not is_finite_num(v) or v < 0 or v > 1:
                bad_sc.append(f"{s['item_id']}|{s['method']}.{f}={v}")
    if bad_sc:
        e(f"out-of-range/NaN scores: {len(bad_sc)} ({bad_sc[:5]})")

    if errors:
        (OUT_DIR / "reports" / "ASSEMBLE_GATE_FAILURES.json").write_text(
            json.dumps(errors, indent=2), encoding="utf-8"
        )
        print("GATE FAILED:")
        for x in errors:
            print(" -", x)
        return 2

    # sort deterministically
    order = {m: i for i, m in enumerate(METHODS)}
    predictions.sort(key=lambda p: (p["item_id"], order[p["method"]]))
    scores.sort(key=lambda s: (s["item_id"], order[s["method"]]))

    write_jsonl(PRED_DIR / "predictions_v22_all_methods.jsonl", predictions)
    write_jsonl(SCORE_DIR / "scores_v22_all_methods.jsonl", scores)

    print(f"wrote {len(predictions)} predictions, {len(scores)} scores")
    print("integrity gate: PASS (2288 each, 0 dup/missing, 0 placeholder, 0 NaN)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
