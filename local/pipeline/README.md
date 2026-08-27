# Local pipeline code

Code for the local implementation-reasoning benchmark (RQ3/RQ4).

## Reproduction path (start here)

- `v2_t3_final/paper_data_pack.py` — recomputes every local table and
  statistic in the paper (overall/system/question-type aggregates, paired
  bootstrap CIs, Pratt–Wilcoxon tests, Holm families) directly from
  `../results/`. This is the Level-1 reproduction entry point; it takes the
  results directory as an argument and has no machine-specific paths.
- `fcg_bench/` — core library: call-graph construction, per-question-type
  fact extractors, the deterministic Struct evaluator (no LLM), and the
  explanation/consistency LLM-judge implementations.
- `build_and_generate.py` — builds the four `CallGraph` objects from the
  call-graph CSVs (available in the Zenodo record under `call_graphs/`).

## Historical remediation drivers (reference only)

The remaining scripts under `v2_t3_final/` (`rebind_t3_7.py`,
`reevaluate_t3_7.py`, `rerun_t3_8.py`, `assemble_bundle.py`, `finalize.py`,
`generate_reports.py`, `full_struct_recompute.py`, `v1_v2_diff.py`,
`preflight_smoke.py`) are the one-off drivers used for the T3 remediation
campaign (gold rebinding, selective re-evaluation, targeted reruns). They
are provided for transparency of the remediation process. They contain
hard-coded paths from the authors' machines and are **not** part of the
reproduction path; nothing in the paper requires re-running them.
