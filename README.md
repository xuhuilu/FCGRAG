# FCG-RAG Replication Package

Replication package for the FCG-RAG paper, covering both benchmarks and all
four RQ results.

- `global/` — global functionality-understanding benchmark (RQ1/RQ2):
  574 items x 4 methods, frozen per-item results with SHA-256 manifest,
  analysis scripts reproducing every reported number and figure, and the
  generation/evaluation pipeline scripts (reference). See `global/README.md`.
- `local/` — local implementation-reasoning benchmark (RQ3/RQ4):
  - `benchmark/` frozen 572-item benchmark (JSONL + validation/manifest
    reports), post-T3-remediation;
  - `results/` prediction and score bundles (2,288 records each),
    significance pack `paper_inference_v22.json`, and binding manifests;
  - `pipeline/` benchmark construction and evaluation code
    (`fcg_bench/`: call-graph builder, fact extractors, deterministic Struct
    evaluator, explanation/consistency judges; `v2_t3_final/`: T3.7/T3.8
    rebind/rerun pipeline and `paper_data_pack.py`, which recomputes every
    local table and statistic in the paper from `results/`).
- `zenodo_upload/` (git-ignored) — large artifacts staged for the Zenodo
  data record: community reports for all three subject systems (12,813
  full-content reports), call graphs, and exact subject source snapshots.
  Download: https://doi.org/10.5281/zenodo.22128120

Subject-system source trees are NOT redistributed here; see `SUBJECTS.md`
for version anchors and clone instructions.

## Reproduction levels

1. **Level 1 (no API key)**: verify checksums and recompute every table.
   Global: run the scripts under `global/analysis/`. Local: run
   `python3 local/pipeline/v2_t3_final/paper_data_pack.py` against
   `local/results/` (see that script's usage).
2. **Level 2 (offline replay)**: re-score the frozen predictions with the
   deterministic evaluator and judge prompts under `local/pipeline/`,
   and re-derive global metrics from the contexts embedded in
   `global/data/results/`.
3. **Level 3 (full rerun, needs LLM API)**: regenerate predictions via the
   pipeline scripts; requires an OpenAI-compatible endpoint and, for global
   retrieval, a RAGFlow deployment (see `global/pipeline/README.md`).

## Licensing

- Code (`global/analysis`, `global/pipeline`, `local/pipeline`): MIT, see
  `LICENSE`.
- Data (benchmarks, frozen results, community reports, call graphs):
  CC-BY-4.0, see `DATA_LICENSE`.
