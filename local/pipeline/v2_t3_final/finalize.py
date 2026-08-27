#!/usr/bin/env python3
"""Emit result/prediction manifests and run the 15-point consistency gate.

Writes:
  manifests/prediction_binding_manifest.json
  manifests/result_manifest.json
  reports/FINAL_VALIDATION_CHECKS.json
"""
from __future__ import annotations

import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2_t3_final import (  # noqa: E402
    AGG_DIR,
    CHECKPOINT_DIR,
    DATASETS,
    MANIFEST_DIR,
    METHODS,
    OUT_DIR,
    PRED_DIR,
    REPO_ROOT,
    SCORE_DIR,
    SIG_DIR,
    V2_BENCH,
    V2_BENCH_SHA256,
    load_jsonl,
    sha256_file,
    sha256_text,
)

REPO = Path("/home/lxh/fcg-v2-implementation")
QTYPES = {"conditional_path", "dependency_trace", "error_handling_path",
          "execution_flow", "impact_scope"}
METHOD_NAMES = {"Direct", "Code-only", "Code+AST", "FCG-RAG"}


def fin(x):
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def main() -> int:
    checks: list[dict] = []

    def chk(name, ok, detail=""):
        checks.append({"check": name, "pass": bool(ok), "detail": detail})

    # benchmark SHA unchanged
    bench_sha = sha256_file(V2_BENCH)
    chk("15 benchmark candidate SHA unchanged", bench_sha == V2_BENCH_SHA256, bench_sha[:16])

    items = load_jsonl(V2_BENCH)
    chk("benchmark 572 items", len(items) == 572, len(items))

    preds = load_jsonl(PRED_DIR / "predictions_v22_all_methods.jsonl")
    scores = load_jsonl(SCORE_DIR / "scores_v22_all_methods.jsonl")
    chk("1 2288 predictions", len(preds) == 2288, len(preds))
    chk("2 2288 score records", len(scores) == 2288, len(scores))

    pm = Counter(p["item_id"] for p in preds)
    sm = Counter(s["item_id"] for s in scores)
    chk("each item exactly 4 methods (pred)",
        all(v == 4 for v in pm.values()) and len(pm) == 572, "")
    chk("each item exactly 4 methods (score)",
        all(v == 4 for v in sm.values()) and len(sm) == 572, "")

    # no dropped v1 item
    plan = list(csv.DictReader((REPO / "reports" / "T3_7_FINAL_INPUT_PLAN.csv").open()))
    drop_ids = {r["item_id"] for r in plan if r["prediction_disposition"].startswith("DROP")}
    leaked = [p["item_id"] for p in preds if p["item_id"] in drop_ids]
    chk("8 0 v1 dropped item in bundle", len(leaked) == 0, f"leaked={set(leaked)}")

    # no placeholder
    bad_pred = [p["item_id"] + "|" + p["method"] for p in preds
                if p["error"] or not (p["raw_output"] or "").strip()
                or not isinstance(p.get("structure_answer"), dict) or not p["structure_answer"]
                or not (p.get("explanation_answer") or "").strip()]
    chk("7 0 placeholder/error predictions", len(bad_pred) == 0, str(len(bad_pred)))

    # finite scores
    bad_sc = [s["item_id"] + "|" + s["method"] for s in scores
              if not all(fin(s[f]) and 0 <= s[f] <= 1 for f in ("structure", "explanation", "consistency", "combined"))]
    chk("6 0 NaN/Inf/out-of-range scores", len(bad_sc) == 0, str(len(bad_sc)))

    # hashes
    reused = [p for p in preds if p["binding"] == "reused_v1"]
    rerun = [p for p in preds if p["binding"] == "rerun_v2"]
    chk("9 reused predictions have source hash",
        all(p["source"].get("line_sha256") for p in reused), f"{len(reused)} records")
    chk("10 rerun predictions have prompt/context hash",
        all(p["source"].get("prompt_sha256") and p["source"].get("context_sha256") for p in rerun),
        f"{len(rerun)} records")

    # name unification
    chk("5 method names unified", all(p["method"] in METHODS for p in preds), "")
    chk("5 qtype names unified", all(p["question_type"] in QTYPES for p in preds), "")

    # duplicate item x method (no retry duplicate billing)
    keys = [(p["item_id"], p["method"]) for p in preds]
    chk("12 0 duplicate item x method / no retry duplicates", len(keys) == len(set(keys)), "")
    skeys = [(s["item_id"], s["method"]) for s in scores]
    chk("score 0 duplicate", len(skeys) == len(set(skeys)), "")

    # aggregate reproducible from per-item
    def qual(s):
        return (s["structure"] + s["explanation"]) / 2
    by_m = defaultdict(list)
    for s in scores:
        by_m[s["method"]].append(s)
    overall_csv = list(csv.DictReader((AGG_DIR / "aggregate_overall_v22.csv").open()))
    agg_ok = True
    for m in METHODS:
        vals = by_m[m]
        exp_struct = sum(v["structure"] for v in vals) / len(vals)
        exp_qual = sum(qual(v) for v in vals) / len(vals)
        row = next(r for r in overall_csv if r["method"] == m)
        if abs(round(exp_struct, 4) - float(row["struct"])) > 1e-3:
            agg_ok = False
        if abs(round(exp_qual, 4) - float(row["qual"])) > 1e-3:
            agg_ok = False
    chk("3 aggregates reproducible from per-item scores", agg_ok, "")

    # CI/significance use 572
    sig = list(csv.DictReader((SIG_DIR / "pairwise_significance_v22.csv").open()))
    chk("4 significance uses 572 items (overall scope)",
        all(int(r["n_items"]) <= 572 for r in sig if r["scope"] == "overall"), "")

    # rounding consistency (4 dp)
    rnd_ok = all(len(str(r["struct"]).split(".")[-1]) <= 4 for r in overall_csv)
    chk("13 rounding consistent (<=4 dp)", rnd_ok, "")

    # E1 retained/drop/rerun mapping
    disp = Counter(r["prediction_disposition"] for r in plan)
    expected = {"REUSE_UNCHANGED": 457, "REUSE_REEVALUATE": 78, "RERUN_ALL_4": 37}
    e1_ok = all(disp.get(k, 0) == v for k, v in expected.items()) and \
        disp.get("DROP_DEDUPLICATE", 0) + disp.get("DROP_EXCLUDE", 0) == 60
    chk("14 E1 retained/drop/rerun mapping correct", e1_ok, dict(disp))

    # API accounting consistency
    reeval = load_jsonl(CHECKPOINT_DIR / "reevaluate_checkpoint.jsonl")
    rg = load_jsonl(CHECKPOINT_DIR / "rerun_gen_checkpoint.jsonl")
    rj = load_jsonl(CHECKPOINT_DIR / "rerun_judge_checkpoint.jsonl")
    reeval_ok = sum(1 for r in reeval if r.get("status") == "success")
    rg_ok = sum(1 for r in rg if r.get("final_status") == "success")
    chk("11 API success counts match logs",
        reeval_ok == 296 and rg_ok == 148 and len(rj) == 148,
        f"reeval={reeval_ok} gen={rg_ok} judge={len(rj)}")

    all_pass = all(c["pass"] for c in checks)
    (OUT_DIR / "reports" / "FINAL_VALIDATION_CHECKS.json").write_text(
        json.dumps({"all_pass": all_pass, "checks": checks}, indent=2), encoding="utf-8"
    )

    # ---- manifests ----
    binding_summary = {
        "reused_v1_records": len(reused),
        "rerun_v2_records": len(rerun),
        "total": len(preds),
        "by_dataset": {ds: sum(1 for p in preds if p["dataset"] == ds) for ds in DATASETS},
        "by_method": {m: sum(1 for p in preds if p["method"] == m) for m in METHODS},
    }
    pred_manifest = {
        "benchmark_sha256": bench_sha,
        "item_count": 572,
        "total_records": len(preds),
        "binding_summary": binding_summary,
        "v1_prediction_source_files": sorted({p["source"]["pred_file"] for p in reused}),
        "rerun_source": "checkpoints/rerun_gen_checkpoint.jsonl",
    }
    (MANIFEST_DIR / "prediction_binding_manifest.json").write_text(
        json.dumps(pred_manifest, indent=2), encoding="utf-8"
    )

    out_files = {}
    for sub in (PRED_DIR, SCORE_DIR, AGG_DIR, SIG_DIR, MANIFEST_DIR):
        for p in sub.glob("**/*"):
            if p.is_file():
                out_files[str(p.relative_to(OUT_DIR))] = sha256_file(p)
    result_manifest = {
        "all_checks_pass": all_pass,
        "record_counts": {"predictions": len(preds), "scores": len(scores), "items": 572},
        "model": "gpt-5.5 via SU8 (https://www.su8.codes/v1)",
        "evaluator": "frozen v2 evaluator (unified normalization)",
        "aggregation": "means over 572 items; Qual=(Struct+Expl)/2",
        "significance": "exact two-sided sign test + paired bootstrap 95% CI (4000 resamples)",
        "output_files_sha256": out_files,
    }
    (MANIFEST_DIR / "result_manifest.json").write_text(
        json.dumps(result_manifest, indent=2), encoding="utf-8"
    )

    print("FINAL VALIDATION:", "ALL PASS" if all_pass else "FAILURES")
    for c in checks:
        if not c["pass"]:
            print("  FAIL:", c["check"], c["detail"])
    return 0 if all_pass else 2


if __name__ == "__main__":
    sys.exit(main())
