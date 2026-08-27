# Benchmark-v2 Candidate Report (T3.6)

Date: 2026-07-30. Status: **benchmark-v2-candidate-accepted — not-yet-rerun,
not-yet-reevaluated, not-final-scientific-freeze**. Built deterministically from the canonical v1 benchmark
(SHA `df0a53b4…c9a94`, verified unchanged) per the author-approved
`T3_FINAL_ACTION_MATRIX.csv` and `T3_V2_IMPLEMENTATION_SPEC.md`.
**No model was run; no v1 artifact was modified.**

> **Author count override (T3.6A, 2026-07-30): candidate = 572 items accepted.
> The 590 procedural floor is rescinded; the deterministic candidate is
> accepted and eligible for T3.7 prediction rebinding and selective
> reevaluation. See `reports/AUTHOR_CANDIDATE_COUNT_OVERRIDE.md`.**

## 1. Headline numbers

| measure | value |
|---|---|
| candidate item count | **572** |
| candidate SHA-256 | `cd321211ae5ede2e8d150fbedd41bb5dfeb7617e5ea77f172b9d7bf205478b60` |
| rerun items / runs | **23 items / 92 runs** (unchanged from T3.5 plan) |
| re-evaluation-only items | **92** |
| reused unchanged | **457** (343 unaudited + 114 KEEP) |
| drops (total) | **60** (16 dedup + 44 exclude) |
| gate fires | **31** (23 JS-MAJOR + 7 impact declique + 1 condition G2) |
| candidate commit | **accepted under author count override (T3.6A)** |

## 2. Composition (572)

| dataset | v1 | v2 | qtype | v1 | v2 |
|---|---|---|---|---|---|
| langchain_python | 159 | **147** | conditional_path | 156 | **143** |
| ollama_cpp | 131 | **123** | dependency_trace | 123 | **85** |
| ollama_go | 159 | **155** | error_handling_path | 100 | **98** |
| openclaw_ts_js | 183 | **147** | execution_flow | 173 | **173** |
| | | | impact_scope | 80 | **73** |

G5 (composition): dependency_trace −30.9% and openclaw_ts_js −19.7% are the
largest relative losses, driven by the JS-MAJOR gates. Reported only — no
resampling, no new items generated (author principles 4–5).

## 3. Final actions applied (632 v1 items)

| action | items | disposition |
|---|---|---|
| KEEP_UNAUDITED_V1 | 343 | REUSE_UNCHANGED (byte-identical) |
| KEEP | 114 | REUSE_UNCHANGED (43 C-IMPACT-PASS + 43 C-UNVERIF-VERIFIABLE + 13 dedup survivors + 1 C-IMPACT-AMBIG no-change + 16 unverif implementation-verified) |
| FIX_EXPLANATION | 17 | REUSE_REEVALUATE (6 span + 2 external + 9 unverif external/stdlib documented) |
| FIX_LABEL | 28 | REUSE_REEVALUATE (27 Go MINOR + ollama_go_0014, see §6) |
| EXPAND_ACCEPTABLE | 2 | REUSE_REEVALUATE (impact_v21_0017, impact_v21_0005) |
| REBUILD_GOLD | 32 | 31 REUSE_REEVALUATE + 1 RERUN (error_v21_0022) |
| REWRITE_QUESTION | 7 | RERUN_ALL_4 (C-DUP-SHARED, fully qualified roots) |
| REBUILD_ITEM | 29 | 15 RERUN_ALL_4 (13 Go MAJOR + cpp_0088 + ts_js_0090) + 14 REUSE_REEVALUATE (JS-MAJOR NEW_CALL rebuilds) |
| DEDUPLICATE | 16 | DROP_DEDUPLICATE (qualname-pinned survivors; clean_v1_0112 kept) |
| EXCLUDE | 13 | DROP_EXCLUDE (C-CLASS-JS-FAIL) |
| GATE_EXCLUDE | 31 | DROP_EXCLUDE (see §5) |

## 4. Prediction dispositions (all 632)

| disposition | items |
|---|---|
| REUSE_UNCHANGED | 457 |
| REUSE_REEVALUATE | 92 |
| RERUN_ALL_4 | 23 (92 runs) |
| DROP_DEDUPLICATE | 16 |
| DROP_EXCLUDE | 44 |
| PENDING_GATE | 0 |

## 5. Implementation gates (G1–G5)

| gate | fired | detail |
|---|---|---|
| G1 (question/root/context change discovered) | 7 | impact declique items (ollama_cpp_impact_v21_0013–0019): removing the destructor clique leaves an **empty** scored caller set (static destruction has no call site) → excluded |
| G2 (extraction vs source conflict) | 1 | ollama_cpp_0017: condition `src1_cont` absent from frozen source; branch identity between two candidate guards not auto-adjudicable |
| G3 (not uniquely recoverable) | 23 | JS-MAJOR: 3 empty recovered target sets; 2 roots absent from extraction; 1 root pooled across 15 defs; 17 roots whose extraction anchors collapse to the same minified-bundle line (viewer-runtime.js:9 / :11) making per-root sets identical and unrecoverable |
| G4 (new duplicates after repair) | 0 | full dedup re-run: 0 question+root+gold collisions |
| G5 (composition imbalance) | reported | see §2; no resampling performed |

JS-MAJOR split: **14 rebuilt** from file-anchored NEW_CALL recovery (8-item
audit-verified cross-check passed: exact for 0005/0014/0007/0017/0020, lower-bound
for 0002; zero contradictions) / **23 gate-excluded**.

## 6. Deviations from the T3.5 plan (all evidence-backed)

| deviation | plan | actual | reason |
|---|---|---|---|
| candidate count | 603 | **572** | 31 gate exclusions (§5) |
| drops | 29 | **60** | same |
| re-eval-only items | 138 | **92** | 23 JS-MAJOR gated instead of rebuilt; unverif reclassification (below) |
| C-UNVERIF-UNRESOLVED (27) | REBUILD_GOLD (drop entity) | 16 KEEP + 9 FIX_EXPLANATION + 1 REBUILD_GOLD + 1 G2 gate | implementation-time frozen-source verification per the author policy KEEP_ONLY_SOURCE_VERIFIABLE_ENTITIES: 16 entities/conditions located verbatim (e.g. `if error_to_raise:` tools/base.py, `if run_managers:` chat_models.py:939, `hasQuantize` x/create/client/create.go:284-293 — the mechanical screen had the wrong file), 9 are stdlib/builtin (fmt.Printf, np.linalg.norm, byte) documented |
| ollama_go_0014 | KEEP (matrix) | FIX_LABEL | dedup survivor carrying the audited MINOR label_only_error (score-neutral relabel; required by validation #11) |
| rerun | 23 items / 92 runs | **unchanged** | — |
| E1 | RETAIN 32 / DROP 1 / RERUN 1 | **unchanged** | clean_v1_0058 dropped, clean_v1_0112 kept |

Also note: C-IMPACT-PASS (43) kept unchanged — the generator-level impact fix
(uncap, comprehension mapping, declique, normalization) was applied as
*verified repairs* where the audit documented changes; PASS items had none
(gate G4 in the spec did not fire).

## 7. Evaluator/generator normalization (T3.6 §6.9)

`fcg_bench/evaluator.py::_norm` now delegates to the shared, tested rule in
`v2_t3_remediation/normalization.py` (strips `<...>` template args, splits
`::`, preserves `<anon@…>` pseudo-names). The v1 rule is preserved as
`_norm_v1`. Drift is confined to `::`/template entities
(validation/normalization_drift.csv); plain Python/Go/JS names are
bit-identical under both rules (unit-tested).

## 8. Validation

- 19/19 deterministic checks PASS (`validation/schema_validation.json`)
- 16/16 unit tests PASS (`test_v2_remediation.py`)
- 632/632 prediction dispositions; 16/16 survivor mappings; 80/80 impact items
  accounted for; 0 duplicates; v1 SHA unchanged; no network/model calls
- zh fields updated in lockstep for all modified items (en/zh consistency)

## 9. Floor breach and author override (T3.6A)

Candidate **572 < 590** procedural floor. The 31 gate exclusions are itemized
with evidence in `reports/implementation_gate_failures.csv`. The author
adjudicated the floor question on 2026-07-30 (option (a)):
**accepted the 572-item candidate and rescinded the floor** — validity and
source recoverability prioritized over nominal size; no resampling and no
minified-bundle re-extraction authorized (see
`reports/AUTHOR_CANDIDATE_COUNT_OVERRIDE.md` and the manifest
`author_count_override` block). The original gate records above are retained
unchanged.
**T3.7 (deterministic prediction rebinding and selective reevaluation) is
unblocked for this accepted candidate.** This is not a final scientific
freeze.

## 10. Artifact index

- `all_multilingual_v22_t3_remediated_en_with_explanations.jsonl` (572 items)
- `manifest/benchmark_v2_candidate_manifest.json`
- `mappings/` v1_to_v2_item_mapping, deduplicate_survivor_mapping, excluded_items, prediction_disposition
- `reports/` action_application, impact_scope_before_after, gold/question/label/explanation before_after, implementation_gate_failures, composition, e1_item_disposition
- `validation/` schema_validation.json, source_grounding, duplicate, impact_scope, go_taxonomy, prediction_binding, normalization_drift, integrity_report.md
- code: `fcg_benchmark/v2_t3_remediation/` (normalization, js_newcall_recovery, repair_patches, build, validate, emit, tests) + evaluator `_norm` patch
