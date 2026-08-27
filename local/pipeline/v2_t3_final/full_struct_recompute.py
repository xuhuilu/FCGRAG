#!/usr/bin/env python3
"""Zero-API deterministic compatibility check.

Recomputes the Struct score for every reusable prediction (records whose
prediction is reused from v1: 457 REUSE_UNCHANGED + 92 REUSE_REEVALUATE =
2196 records) with the frozen v2 evaluator. The 92 RERUN_ALL_4 records are
deferred to T3.8 (their predictions do not exist yet and will be scored with
the same frozen v2 evaluator during rerun evaluation).

For each record computes four controlled scores by temporarily switching the
evaluator's entity-normalization function:

  S_old            = v1 frozen Struct   (v1 norm `_norm_v1` + v1 gold)
  S_norm_only      = v2 norm on v1 gold (isolates normalization contribution)
  S_gold_only      = v1 norm on v2 gold (isolates gold-change contribution)
  S_new (final)    = v2 norm on v2 gold (the value the v2 bundle will carry)

Attribution rule (prediction is reused, so prediction contribution == 0):
  - |S_norm_only - S_old| > eps and |S_gold_only - S_old| <= eps  -> normalization
  - |S_norm_only - S_old| <= eps and |S_gold_only - S_old| > eps  -> gold
  - both > eps                                                    -> normalization+gold
  - both <= eps                                                   -> none

Determinism guard: S_new is computed twice; the two passes must be byte-identical.

Outputs:
  reports/T3_7_FULL_STRUCT_RECOMPUTE.csv
  reports/T3_7_NORMALIZATION_IMPACT.md

No model calls, no judge calls. Hard stops on:
  - non-deterministic evaluator reproduction
  - attribution = none but delta != 0 (unattributable drift)
  - binding/schema failure
"""
from __future__ import annotations

import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fcg_benchmark/

import fcg_bench.evaluator as ev  # noqa: E402
from fcg_bench.evaluator import evaluate_prediction  # noqa: E402
from fcg_bench.schema import BenchmarkItem  # noqa: E402

from v2_t3_final import (  # noqa: E402
    DATASETS,
    EXT_RUN_DIR,
    METHODS,
    MAPPING_CSV,
    V1_BENCH,
    V1_SCORE_DIR,
    V2_BENCH,
    load_jsonl,
    verify_frozen_inputs,
)

REPO_REPORTS = Path("/home/lxh/fcg-v2-implementation/reports")
EPS = 1e-9


def bench_item_of(item: dict) -> BenchmarkItem:
    return BenchmarkItem(
        question_type=item["question_type"],
        root_function=item["root_function"],
        question=item.get("question_en", item.get("question", "")),
        gold_answer=item["gold_answer"],
        natural_answer="",
        subgraph_nodes=item.get("subgraph_nodes", []),
        structure_gold_hash=item.get("structure_gold_hash", ""),
        benchmark_tier=item.get("benchmark_tier", "main_experiment"),
    )


def score_with(norm_fn, bench_item, struct_pred):
    """Evaluate with a specific normalization function (temporary monkeypatch)."""
    saved = ev._norm
    ev._norm = norm_fn
    try:
        return round(evaluate_prediction(bench_item, struct_pred), 4)
    finally:
        ev._norm = saved


def main() -> None:
    verify_frozen_inputs()

    v1_items = {it["item_id"]: it for it in load_jsonl(V1_BENCH)}
    v2_items = {it["item_id"]: it for it in load_jsonl(V2_BENCH)}
    assert len(v2_items) == 572

    # mapping: gold_changed flag + disposition
    gold_changed = {}
    disposition = {}
    action = {}
    with MAPPING_CSV.open() as f:
        for r in csv.DictReader(f):
            disposition[r["original_item_id"]] = r["prediction_disposition"]
            action[r["original_item_id"]] = r["action"]
            gold_changed[r["original_item_id"]] = r["old_gold_hash"] != r["new_gold_hash"]

    # v1 scores (frozen hybrid bundle)
    v1_scores = {}
    for ds in DATASETS:
        for row in load_jsonl(V1_SCORE_DIR / f"{ds}_per_item_scores_v2_en.jsonl"):
            v1_scores[row["item_id"]] = row

    # v1 predictions (reused)
    v1_preds = {}
    for ds in DATASETS:
        ds_items = {it["item_id"]: it for it in load_jsonl(V1_BENCH) if it["dataset"] == ds}
        for m in METHODS:
            for row in load_jsonl(EXT_RUN_DIR / f"{ds}_pred_{m}_v2_en.jsonl"):
                v1_preds[(ds, m, row["item_id"])] = row

    rows = []
    n_rerun = 0
    for item_id in sorted(v2_items):
        disp = disposition.get(item_id)
        if disp == "RERUN_ALL_4":
            n_rerun += 1
            continue  # deferred to T3.8
        if disp is None or disp.startswith("DROP"):
            continue
        v2_item = v2_items[item_id]
        v1_item = v1_items[item_id]
        bi_v2 = bench_item_of(v2_item)
        bi_v1 = bench_item_of(v1_item)
        qtype = v2_item["question_type"]
        ds = v2_item["dataset"]
        v1_score_row = v1_scores[item_id]

        for method in METHODS:
            pred = v1_preds.get((ds, method, item_id))
            assert pred is not None, f"missing reused prediction {ds}|{method}|{item_id}"
            struct = pred.get("structure_answer") or {}
            assert isinstance(struct, dict) and struct, f"empty struct {ds}|{method}|{item_id}"

            s_old = float(v1_score_row[f"{method}_structure"])     # v1norm+v1gold (frozen)
            s_v2n_v1g = score_with(ev._norm, bi_v1, struct)        # v2norm+v1gold
            s_v1n_v2g = score_with(ev._norm_v1, bi_v2, struct)     # v1norm+v2gold
            s_new_a = score_with(ev._norm, bi_v2, struct)          # v2norm+v2gold (final)
            s_new_b = score_with(ev._norm, bi_v2, struct)          # determinism guard

            # determinism hard stop
            assert s_new_a == s_new_b, (
                f"non-deterministic Struct: {item_id}|{method} {s_new_a} != {s_new_b}"
            )

            delta = s_new_a - s_old
            # Four-corner decomposition (prediction reused -> prediction contribution == 0).
            norm_eff_on_v1gold = s_v2n_v1g - s_old
            norm_eff_on_v2gold = s_new_a - s_v1n_v2g
            gold_eff_on_v1norm = s_v1n_v2g - s_old
            gold_eff_on_v2norm = s_new_a - s_v2n_v1g
            norm_involved = abs(norm_eff_on_v1gold) > EPS or abs(norm_eff_on_v2gold) > EPS
            gold_involved = abs(gold_eff_on_v1norm) > EPS or abs(gold_eff_on_v2norm) > EPS

            if not norm_involved and not gold_involved:
                attr = "none"
            elif norm_involved and not gold_involved:
                attr = "normalization"
            elif gold_involved and not norm_involved:
                attr = "gold"
            else:
                attr = "normalization+gold"

            # true unattributable drift: delta != 0 but neither norm nor gold involved
            if abs(delta) > EPS and attr == "none":
                print(f"STOP: unattributable drift {item_id}|{method} delta={delta}")
                sys.exit(2)

            rows.append({
                "item_id": item_id,
                "dataset": ds,
                "qtype": qtype,
                "method": method,
                "disposition": disp,
                "action": action[item_id],
                "gold_changed": str(gold_changed[item_id]).lower(),
                "v1_struct": s_old,
                "v2norm_v1gold": s_v2n_v1g,
                "v1norm_v2gold": s_v1n_v2g,
                "v2_struct_final": s_new_a,
                "delta": round(delta, 4),
                "norm_eff_v1gold": round(norm_eff_on_v1gold, 4),
                "norm_eff_v2gold": round(norm_eff_on_v2gold, 4),
                "gold_eff_v1norm": round(gold_eff_on_v1norm, 4),
                "gold_eff_v2norm": round(gold_eff_on_v2norm, 4),
                "attribution": attr,
            })

    out_csv = REPO_REPORTS / "T3_7_FULL_STRUCT_RECOMPUTE.csv"
    fields = list(rows[0].keys())
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out_csv} ({len(rows)} records)")

    # ---- summary statistics ----
    changed = [r for r in rows if abs(r["delta"]) > EPS]
    n_items_changed = len({r["item_id"] for r in changed})
    deltas = [abs(r["delta"]) for r in changed]
    max_abs = max(deltas) if deltas else 0.0
    mean_abs = (sum(deltas) / len(deltas)) if deltas else 0.0

    by_method = Counter(r["method"] for r in changed)
    by_qtype = Counter(r["qtype"] for r in changed)
    by_dataset = Counter(r["dataset"] for r in changed)
    by_attr = Counter(r["attribution"] for r in changed)

    # ranking impact: recompute per-method overall Struct mean under v1 vs v2
    by_m_v1 = defaultdict(list)
    by_m_v2 = defaultdict(list)
    for r in rows:
        by_m_v1[r["method"]].append(r["v1_struct"])
        by_m_v2[r["method"]].append(r["v2_struct_final"])
    rank_v1 = sorted(METHODS, key=lambda m: -sum(by_m_v1[m]) / len(by_m_v1[m]))
    rank_v2 = sorted(METHODS, key=lambda m: -sum(by_m_v2[m]) / len(by_m_v2[m]))
    means = {
        m: {
            "v1": round(sum(by_m_v1[m]) / len(by_m_v1[m]), 4),
            "v2": round(sum(by_m_v2[m]) / len(by_m_v2[m]), 4),
        }
        for m in METHODS
    }

    lines = []
    lines.append("# T3.7 Normalization Impact — Full Struct Recompute (zero API)")
    lines.append("")
    lines.append("All reusable predictions (2196 records) were re-scored with the frozen v2")
    lines.append("evaluator. Struct is recomputed uniformly; no model or judge calls were made.")
    lines.append(f"The {n_rerun*4} RERUN_ALL_4 records are deferred to T3.8 and will be scored with")
    lines.append("the same frozen v2 evaluator during rerun evaluation, so the final bundle uses a")
    lines.append("single normalization rule throughout.")
    lines.append("")
    lines.append("## Headline")
    lines.append("")
    lines.append(f"- records recomputed: **{len(rows)}**")
    lines.append(f"- records with changed Struct: **{len(changed)}** ({len(changed)/len(rows)*100:.2f}%)")
    lines.append(f"- distinct items affected: **{n_items_changed}**")
    lines.append(f"- max |delta|: **{max_abs:.4f}**")
    lines.append(f"- mean |delta| over changed: **{mean_abs:.4f}**")
    lines.append("")
    lines.append("## Attribution (delta decomposed into normalization vs gold change;")
    lines.append("prediction is reused so its contribution is zero)")
    lines.append("")
    lines.append("| attribution | records |")
    lines.append("|---|---|")
    for k in ("none", "normalization", "gold", "normalization+gold"):
        lines.append(f"| {k} | {by_attr.get(k,0)} |")
    lines.append("")
    lines.append("## Distribution of changed records")
    lines.append("")
    lines.append("By method:")
    lines.append("")
    lines.append("| method | changed |")
    lines.append("|---|---|")
    for m in METHODS:
        lines.append(f"| {m} | {by_method.get(m,0)} |")
    lines.append("")
    lines.append("By question type:")
    lines.append("")
    lines.append("| qtype | changed |")
    lines.append("|---|---|")
    for q in sorted(by_qtype):
        lines.append(f"| {q} | {by_qtype[q]} |")
    lines.append("")
    lines.append("By dataset:")
    lines.append("")
    lines.append("| dataset | changed |")
    lines.append("|---|---|")
    for d in sorted(by_dataset):
        lines.append(f"| {d} | {by_dataset[d]} |")
    lines.append("")
    lines.append("## Method-ranking impact (Struct overall mean, reusable records only)")
    lines.append("")
    lines.append("| method | v1 struct mean | v2 struct mean | delta |")
    lines.append("|---|---|---|---|")
    for m in METHODS:
        lines.append(
            f"| {m} | {means[m]['v1']:.4f} | {means[m]['v2']:.4f} | "
            f"{means[m]['v2']-means[m]['v1']:+.4f} |"
        )
    lines.append("")
    lines.append(f"- v1 ranking (best→worst): {' > '.join(rank_v1)}")
    lines.append(f"- v2 ranking (best→worst): {' > '.join(rank_v2)}")
    lines.append(f"- ranking changed: **{'yes' if rank_v1 != rank_v2 else 'no'}**")
    lines.append("")
    lines.append("## Determinism / binding checks")
    lines.append("")
    lines.append("- S_new computed twice per record: all identical (deterministic).")
    lines.append(f"- unattributable drift records (delta!=0, attribution=none): **0**.")
    lines.append(f"- all {len(rows)} reusable records bound to a v1 prediction and v1 score.")

    out_md = REPO_REPORTS / "T3_7_NORMALIZATION_IMPACT.md"
    out_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"wrote {out_md}")
    print(f"changed records: {len(changed)}/{len(rows)}; items: {n_items_changed}; "
          f"ranking changed: {rank_v1 != rank_v2}")


if __name__ == "__main__":
    main()
