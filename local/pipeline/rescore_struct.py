#!/usr/bin/env python3
"""Level-2 verification: re-score all frozen local predictions with the
deterministic Struct evaluator and compare against the frozen score bundle.

Usage (from local/pipeline/):
    python3 rescore_struct.py [--limit N]

Expected: every record matches the frozen Struct value up to a tiny
floating-point deviation (max |diff| ~ 5e-5, caused by set-order
nondeterminism in tie handling; the frozen bundle is authoritative).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fcg_bench.evaluator import evaluate_prediction  # noqa: E402
from fcg_bench.schema import BenchmarkItem  # noqa: E402

LOCAL = Path(__file__).resolve().parents[1]
BENCH = LOCAL / "benchmark" / "all_multilingual_v22_t3_remediated_en_with_explanations.jsonl"
PREDS = LOCAL / "results" / "predictions" / "predictions_v22_all_methods.jsonl"
SCORES = LOCAL / "results" / "scores" / "scores_v22_all_methods.jsonl"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    bench: dict[str, BenchmarkItem] = {}
    for line in open(BENCH, encoding="utf-8"):
        d = json.loads(line)
        bench[d["item_id"]] = BenchmarkItem(
            question_type=d["question_type"], root_function=d["root_function"],
            question=d["question"], gold_answer=d["gold_answer"])
    scores: dict[tuple[str, str], float] = {}
    for line in open(SCORES, encoding="utf-8"):
        s = json.loads(line)
        scores[(s["item_id"], s["method"])] = float(s["structure"])

    preds = [json.loads(l) for l in open(PREDS, encoding="utf-8")]
    if args.limit:
        preds = preds[: args.limit]

    max_diff = 0.0
    worst = None
    for p in preds:
        s = evaluate_prediction(bench[p["item_id"]], p["structure_answer"])
        frozen = scores[(p["item_id"], p["method"])]
        d = abs(s - frozen)
        if d > max_diff:
            max_diff, worst = d, (p["item_id"], p["method"], s, frozen)

    print(f"re-scored {len(preds)} records with the frozen v2 evaluator")
    print(f"max |diff| vs frozen Struct = {max_diff:.2e}  (worst: {worst})")
    if max_diff > 1e-3:
        sys.exit("FAIL: deviation exceeds tolerance 1e-3")
    print("OK: all records within tolerance 1e-3")


if __name__ == "__main__":
    main()
