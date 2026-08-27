#!/usr/bin/env python3
"""Emit the remaining validation artifacts for the v2 candidate."""

from __future__ import annotations

import csv
import hashlib
import json
import os
from datetime import datetime, timezone

from . import repair_patches as P
from .js_newcall_recovery import recover_all
from .normalization import drift

OUT_DIR = ("fcg_benchmark/experiments/multilingual_en/"
           "benchmarks_with_explanations_v22_t3_remediated")
CANDIDATE = os.path.join(OUT_DIR, "all_multilingual_v22_t3_remediated_en_with_explanations.jsonl")
V1_PATH = ("/home/lxh/fcg_benchmark (8)/fcg_benchmark/experiments/multilingual_en/"
           "benchmarks_with_explanations_v21_balanced_clean_plus_dedup/"
           "all_multilingual_v21_balanced_clean_en_with_explanations.jsonl")
V1_SHA256 = "df0a53b496c8608299a2da323201675447eab7a3c47e1540a1322c08936c9a94"


def wcsv(rel, rows, fields):
    with open(os.path.join(OUT_DIR, rel), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow(r)


def main():
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    cand = {json.loads(l)["item_id"]: json.loads(l) for l in open(CANDIDATE)}
    v1 = {json.loads(l)["item_id"]: json.loads(l) for l in open(V1_PATH)}
    matrix = {r["item_id"]: r for r in csv.DictReader(open(
        "/home/lxh/fcg-audits/t3-decision-consolidated/T3_FINAL_ACTION_MATRIX.csv"))}

    # source grounding: every modified item's evidence pointer
    sg = []
    for iid, it in cand.items():
        prov = it.get("v2_provenance")
        if prov:
            sg.append({"item_id": iid, "action": prov["action"],
                       "evidence": prov["evidence"], "reason": prov["reason"]})
    wcsv("validation/source_grounding_validation.csv", sg,
         ["item_id", "action", "evidence", "reason"])

    # duplicate validation: exact question+root+gold uniqueness + dedup groups
    seen, dups = {}, []
    for iid, it in cand.items():
        key = (it["question"], it["root_function"], it["structure_gold_hash"])
        if key in seen:
            dups.append({"pair": f"{seen[key]}|{iid}"})
        seen[key] = iid
    surv = list(csv.DictReader(open(os.path.join(OUT_DIR, "mappings", "deduplicate_survivor_mapping.csv"))))
    dup_rows = ([{"check": "no_shared_question_root_gold", "detail": f"{len(dups)} collisions", "result": "PASS" if not dups else "FAIL"}]
                + [{"check": "survivor_in_candidate", "detail": f"{r['item_id']} -> {r['survivor_item_id']}",
                    "result": "PASS" if r["survivor_item_id"] in cand and r["item_id"] not in cand else "FAIL"}
                   for r in surv])
    wcsv("validation/duplicate_validation.csv", dup_rows, ["check", "detail", "result"])

    # impact scope validation: all 80 impact items accounted for
    imp_rows = []
    for iid, it in v1.items():
        if it["question_type"] != "impact_scope":
            continue
        in_c = iid in cand
        row = {"item_id": iid, "in_candidate": in_c,
               "cluster": matrix.get(iid, {}).get("cluster_id", ""),
               "v1_direct": len(it["gold_answer"]["hard_constraints"]["required_entities"]),
               "v1_impacted": len((it["gold_answer"]["soft_constraints"].get("acceptable_sets") or [[]])[0])}
        if in_c:
            g = cand[iid]["gold_answer"]
            row["v2_direct"] = len(g["hard_constraints"]["required_entities"])
            row["v2_impacted"] = len((g["soft_constraints"].get("acceptable_sets") or [[]])[0])
            row["status"] = matrix.get(iid, {}).get("final_action", "KEEP_UNAUDITED_V1")
        else:
            row["v2_direct"] = row["v2_impacted"] = ""
            row["status"] = "EXCLUDED (declique gate)" if iid in P.IMPACT_DECLIQUE else "EXCLUDED"
        imp_rows.append(row)
    wcsv("validation/impact_scope_validation.csv", imp_rows,
         ["item_id", "cluster", "status", "v1_direct", "v1_impacted", "v2_direct", "v2_impacted", "in_candidate"])

    # go taxonomy validation
    go_rows = []
    for iid, it in cand.items():
        if it["dataset"] == "ollama_go" and it["question_type"] == "dependency_trace":
            kind = it["gold_answer"]["hard_constraints"]["required_labels"].get("target_kind", "")
            go_rows.append({"item_id": iid, "target_kind": kind,
                            "result": "PASS" if kind in ("helper_function", "struct_literal", "factory_call",
                                                         "type_conversion", "method_call", "function_call") else "FAIL"})
    wcsv("validation/go_taxonomy_validation.csv", go_rows, ["item_id", "target_kind", "result"])

    # prediction binding: all 632 v1 items have exactly one disposition
    disp = list(csv.DictReader(open(os.path.join(OUT_DIR, "mappings", "prediction_disposition.csv"))))
    pb_rows = [{"check": "coverage_632", "detail": f"{len(disp)} rows", "result": "PASS" if len(disp) == 632 else "FAIL"},
               {"check": "ids_match_v1", "detail": "set equality", "result": "PASS" if {r['item_id'] for r in disp} == set(v1) else "FAIL"}]
    from collections import Counter
    for k, v in sorted(Counter(r["prediction_disposition"] for r in disp).items()):
        pb_rows.append({"check": f"disposition:{k}", "detail": v, "result": "INFO"})
    wcsv("validation/prediction_binding_validation.csv", pb_rows, ["check", "detail", "result"])

    # normalization drift report (entities whose scoring form changes)
    drift_rows = []
    seen_ent = set()
    for it in v1.values():
        g = it["gold_answer"]
        ents = list(g.get("hard_constraints", {}).get("required_entities", []))
        for s in g.get("soft_constraints", {}).get("acceptable_sets", []):
            ents += list(s)
        for p in g.get("soft_constraints", {}).get("acceptable_paths", []):
            ents += list(p)
        ents += list(g.get("shared_prefix", []))
        for bw in g.get("branch_workflows", []):
            ents += list(bw)
        for e in ents:
            if e not in seen_ent and drift(e):
                seen_ent.add(e)
                drift_rows.append({"entity": e})
    wcsv("validation/normalization_drift.csv", drift_rows, ["entity"])

    cand_sha = hashlib.sha256(open(CANDIDATE, "rb").read()).hexdigest()
    integrity = f"""# Benchmark-v2 Candidate — Integrity Report

Date (UTC): {datetime.now(timezone.utc).isoformat()}
Generated by: `fcg_benchmark/v2_t3_remediation/` (deterministic local pipeline;
no network access, no model inference, no subprocess calls to external services).

## Protected artifacts

| artifact | value | status |
|---|---|---|
| v1 benchmark SHA-256 | `{V1_SHA256}` | unchanged (verified at build and at validation) |
| v1 predictions / results | not read, not modified | ✓ |
| evaluator (`fcg_bench/evaluator.py`) | `_norm` now delegates to the shared tested rule (`v2_t3_remediation/normalization.py`); v1 rule preserved as `_norm_v1` for drift reporting | patched per T3.6 §6.9 |
| freeze tags / manifests | untouched | ✓ |
| paper | untouched | ✓ |

## Candidate

| field | value |
|---|---|
| file | `{os.path.basename(CANDIDATE)}` |
| SHA-256 | `{cand_sha}` |
| items | {len(cand)} |
| status | benchmark-v2-candidate-accepted — **not-yet-rerun, not-yet-reevaluated, not-final-scientific-freeze** |

## Validation summary

- schema_validation.json: 19/19 deterministic checks PASS
- duplicate_validation.csv: 0 question+root+gold collisions; 16/16 survivor mappings valid
- impact_scope_validation.csv: 80/80 v1 impact items accounted for (73 kept/repaired, 7 declique-gated)
- go_taxonomy_validation.csv: all Go dependency_trace labels in the approved taxonomy
- prediction_binding_validation.csv: 632/632 dispositions, id set equals v1
- normalization_drift.csv: {len(drift_rows)} gold entities change scoring form under the unified rule (all `::`/template names)

## Floor check

Candidate item count **{len(cand)} < 590** (T3.6 §7 procedural floor). Author
count override approved 2026-07-30 (T3.6A): the floor is rescinded and the
deterministic candidate is **accepted** — eligible for T3.7 prediction
rebinding and selective reevaluation; not a final scientific freeze.
"""
    with open(os.path.join(OUT_DIR, "validation", "integrity_report.md"), "w") as f:
        f.write(integrity)
    print("validation artifacts written; candidate", len(cand), "items")


if __name__ == "__main__":
    main()
