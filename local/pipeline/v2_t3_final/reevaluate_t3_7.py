#!/usr/bin/env python3
"""T3.7 selective reevaluation (LLM judge).

Re-judges ONLY the 296 REJUDGE_BOTH records (74 gold-changed items x 4 methods)
whose gold change alters a judge input. Each record gets exactly two judge
calls (explanation + consistency), matching the frozen DeepSeekJudge prompts
and params (gpt-5.5 via SU8, temperature=0.0, json_object, max_tokens=1400).

Features:
  - checkpoint/resume: one JSONL line per completed record; successful records
    are skipped on resume (no double billing).
  - per-call request id + error logging.
  - single-record failure does NOT abort the batch; the record is marked failed
    and re-queued on the next run.
  - no missing values reach the formal bundle: a final assertion requires all
    296 records to be status=success.

Does NOT overwrite v1 score files. Writes only under results_v22_t3_remediated/.
"""
from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fcg_benchmark/

from evaluate_three_methods_clean_v1_en import (  # noqa: E402
    evaluate_consistency_item,
    evaluate_explanation_item,
)
from run_three_methods_clean_v1_en import load_jsonl  # noqa: E402

from v2_t3_final import (  # noqa: E402
    CHECKPOINT_DIR,
    DATASETS,
    EXT_RUN_DIR,
    MANIFEST_DIR,
    METHODS,
    V2_BENCH,
    ensure_out_dirs,
    verify_frozen_inputs,
)
from v2_t3_final.su8_client import SU8Judge  # noqa: E402

CKPT = CHECKPOINT_DIR / "reevaluate_checkpoint.jsonl"
WORKLIST = MANIFEST_DIR / "rejudge_worklist.jsonl"
LOG = CHECKPOINT_DIR / "reevaluate_error.log"
CK_LOCK = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def load_done() -> set[tuple[str, str]]:
    done = set()
    if CKPT.exists():
        for line in CKPT.read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            r = json.loads(line)
            if r.get("status") == "success":
                done.add((r["item_id"], r["method"]))
    return done


def append_checkpoint(rec: dict) -> None:
    with CK_LOCK:
        with CKPT.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def judge_one(task: dict, v2_item: dict, pred: dict) -> dict:
    judge = SU8Judge()
    rec: dict = {
        "item_id": task["item_id"],
        "dataset": task["dataset"],
        "qtype": task["question_type"],
        "method": task["method"],
        "started_at": utc_now(),
    }
    try:
        exp = evaluate_explanation_item(judge, v2_item, v2_item, pred)
        exp_rid = judge.last_request_id
        cons = evaluate_consistency_item(judge, v2_item, pred)
        cons_rid = judge.last_request_id
        rec.update({
            "status": "success",
            "explanation": exp,
            "explanation_request_id": exp_rid,
            "consistency": cons,
            "consistency_request_id": cons_rid,
            "completed_at": utc_now(),
        })
    except Exception as exc:
        rec.update({
            "status": "failed",
            "error_type": type(exc).__name__,
            "error": str(exc)[:800],
            "status_code": getattr(exc, "status_code", None),
            "completed_at": utc_now(),
        })
        with CK_LOCK, LOG.open("a", encoding="utf-8") as f:
            f.write(f"{utc_now()} {task['item_id']}|{task['method']} "
                    f"{type(exc).__name__} {getattr(exc,'status_code','?')} {str(exc)[:300]}\n")
    return rec


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    ensure_out_dirs()
    verify_frozen_inputs()

    worklist = load_jsonl(WORKLIST)
    assert len(worklist) == 296, len(worklist)
    v2_items = {it["item_id"]: it for it in load_jsonl(V2_BENCH)}

    # reused predictions for the 74 gold-changed items
    preds = {}
    for ds in DATASETS:
        for m in METHODS:
            for row in load_jsonl(EXT_RUN_DIR / f"{ds}_pred_{m}_v2_en.jsonl"):
                preds[(ds, m, row["item_id"])] = row

    done = load_done()
    pending = [t for t in worklist if (t["item_id"], t["method"]) not in done]
    if args.limit > 0:
        pending = pending[: args.limit]
    print(f"worklist={len(worklist)} done={len(done)} pending={len(pending)} workers={args.workers}", flush=True)

    tasks = []
    for t in pending:
        pred = preds[(t["dataset"], t["method"], t["item_id"])]
        tasks.append((t, v2_items[t["item_id"]], pred))

    n_ok = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(judge_one, t, vi, p): t for (t, vi, p) in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            rec = fut.result()
            append_checkpoint(rec)
            if rec["status"] == "success":
                n_ok += 1
            else:
                n_fail += 1
            if i % 20 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} ok={n_ok} fail={n_fail} "
                      f"elapsed={time.time()-t0:.0f}s", flush=True)

    # reload and verify completeness
    all_done = load_done()
    missing = [(t["item_id"], t["method"]) for t in worklist if (t["item_id"], t["method"]) not in all_done]
    print(f"total success={len(all_done)}/{len(worklist)} missing={len(missing)} "
          f"this-run ok={n_ok} fail={n_fail}", flush=True)
    if missing:
        print("STOP: not all 296 records judged successfully; re-run to retry failures.", flush=True)
        (MANIFEST_DIR / "reevaluate_missing.json").write_text(
            json.dumps(missing, indent=2), encoding="utf-8"
        )
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
