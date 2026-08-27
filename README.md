# FCG-RAG Replication Package

This package reproduces the results of the FCG-RAG paper:

- **Global functionality understanding (RQ1/RQ2)** — 574 open-ended QA items
  over three subject systems (Ollama 228, OpenClaw 245, LangChain 101);
  Code-RAG, Report-RAG, Int-RAG, FCG-RAG, plus closed-book controls.
- **Local implementation reasoning (RQ3/RQ4)** — 572 structured-prediction
  items over four project–language settings (LangChain/Python 147,
  Ollama/C–C++ 123, Ollama/Go 155, OpenClaw/TS–JS 147); Direct, Code-only,
  Code+AST, FCG-RAG.

Companion data record on Zenodo (subject source snapshots, community
reports, call graphs): **https://doi.org/10.5281/zenodo.22128120**
(only needed for full pipeline reruns, NOT for reproducing the paper's
numbers — see below).

## Repository layout

```
global/
  data/benchmark/     global gold QA (3 systems: questions, gold answers, atomic facts)
  data/results/       frozen per-item predictions + metrics (.json.gz), MANIFEST.json (SHA-256)
  analysis/           one script per reported analysis + figures/ (all RQ1/RQ2 figures)
  pipeline/           generation/evaluation scripts that produced the frozen results,
                      ragflow_config/ (exported retrieval-service configurations)
local/
  benchmark/          frozen 572-item benchmark (JSONL + mappings + validation reports)
  results/            predictions/ + scores/ (2,288 records each) + significance pack + manifests
  analysis/           local figure scripts (figures/) and their input data (data/)
  pipeline/           fcg_bench/ (evaluator, extractors, judges), v2_t3_final/paper_data_pack.py,
                      rescore_struct.py
SUBJECTS.md           version anchors (content hashes) for the three subject systems
SHA256SUMS            checksums of all data files in this repository
LICENSE / DATA_LICENSE / CITATION.cff
```

## Environment setup

Python 3.10+ (verified on 3.10):

```bash
git clone <this-repo-url> && cd fcg-rag-replication
pip install -r global/requirements.txt   # numpy, pandas, scipy, statsmodels, matplotlib
```

No API keys and no Zenodo download are needed for anything in
"Reproducing the paper" below: all analyses start from the frozen per-item
results shipped in this repository.

Optional integrity check of the shipped data:

```bash
sha256sum -c SHA256SUMS     # 32 files, all OK
```

## Reproducing the paper

### RQ1 — Global main results

| Paper artifact | Command (from `global/analysis/`) | Expected output |
|---|---|---|
| Main results table (aggregate metrics, 4 methods x 3 systems) | `python export_item_level_csv.py` | all 12 aggregate cells print `MATCH` (two documented rounding exceptions, see `data/results/MANIFEST.json`); writes `output/item_level_all_methods.csv` |
| Paired RA-FC gains + 95% CIs | `python rq1_paired_bootstrap.py` | **+0.035 [0.010, 0.060]** vs Code-RAG; **+0.089 [0.064, 0.114]** vs Report-RAG; **+0.061 [0.036, 0.087]** vs Int-RAG |

### RQ2 — Global performance factors

| Paper artifact | Command (from `global/analysis/`) | Expected output |
|---|---|---|
| Answer-transition decomposition (figure) | `python rq2_answer_transition.py` then `python figures/plot_answer_pairwise_transition.py` | e.g. vs Int-RAG: FCG-only 150 vs Int-only 37 answers, McNemar p = 2.6e-16 |
| Coverage-stratified answer rates (figure + appendix table) | `python rq2_conditional_answer_rate.py` then `python figures/plot_conditional_answer_rate.py` | FCG-RAG answers 86.8% of hit items vs 68.5–71.1% (baselines); 41.5% of miss items vs 19.0–27.5% |
| Covered/uncovered split vs closed-book (figure) | `python rq2_branch_contribution.py` then `python figures/plot_rafc_branch_contribution.py` | covered: FCG 0.524 vs closed-book 0.360; uncovered: 0.320 vs 0.339 |
| Logistic regression + answered-item OLS (statistical controls) | `python rq2_statistical_controls.py` | FCG-RAG answer odds 2.14x vs Code-RAG (p = 0.016) |
| Metric correlations + faithful-answer rates (figures) | `python rq2_metric_correlations.py` then `python figures/plot_global_metric_correlation_heatmap.py` and `python figures/plot_faith_threshold_sensitivity.py` | FCG-RAG faithful-answer rate 0.467, highest up to threshold 0.75 |

Every analysis script first re-verifies the frozen data: SHA-256 against
`data/results/MANIFEST.json` and positional item alignment across methods
(items have no stable IDs; the 574-question order is identical in every
file).

### RQ3 — Local effectiveness

| Paper artifact | Command | Expected output |
|---|---|---|
| Main local table (Struct/Expl/Qual/Consist., overall + 4 systems) | `cd local/pipeline/v2_t3_final && python paper_data_pack.py` | Overall Qual: Direct 0.429 / Code-only 0.523 / Code+AST 0.595 / **FCG-RAG 0.693** |
| Paired inference (CI, Pratt–Wilcoxon, effect size, win/tie/loss) | same script | FCG − Code+AST = **+0.0981 [0.0762, 0.1208]**, p = 3.44e-08, r_rb = 0.2717; improved/tied/degraded = **284/84/204** |
| Per-system comparisons + Holm family | same script | written to `local/results/significance/paper_inference_v22.json` (`by_system`, `rq2_system_family`) |
| Paired-difference histogram (figure) | `cd local/analysis/figures && python plot_local_paired_qual_difference.py` | mean = 0.098, median = 0.000, n = 572 |

The script recomputes everything from `local/results/` (Qual = (Struct +
Expl)/2; root-clustered bootstrap, seed 20260731) and rewrites
`paper_inference_v22.json` **bit-identically** (SHA-256 in `SHA256SUMS`).

Level-2 check — re-score the frozen predictions with the deterministic
Struct evaluator (no LLM):

```bash
cd local/pipeline && python rescore_struct.py    # 2,288 records; max |diff| ~ 5e-5 vs frozen
```

### RQ4 — Question-type variation

| Paper artifact | Command | Expected output |
|---|---|---|
| Qual by question type (4 methods x 5 types) | `paper_data_pack.py` (as above) | `by_qtype` in `paper_inference_v22.json`; FCG-RAG best on dependency trace / execution flow / impact scope; Code+AST best on conditional path / error handling |
| Question-type metric decomposition (ΔStruct/ΔExpl/ΔQual) | same | `qtype_decomposition`: impact scope +0.931 Struct, dependency trace +0.137/+0.106 |
| System x type delta matrix | same | `system_type_delta_qual`: impact-scope gains +0.581 to +0.727 in all four systems |
| Per-type Holm family | same | `rq3_qtype_family` |
| Question-type bars + system–type heatmap (figures) | `cd local/analysis/figures && python plot_local_question_type_performance.py && python plot_local_system_type_delta_qual.py` | PDFs in `local/analysis/output/figures/` |

The local figure scripts read precomputed per-item aggregates from
`local/analysis/data/` (derived from the same frozen score bundle).

## Full pipeline rerun (from scratch)

Reproducing the *numbers* above does not require rerunning the experiments.
Rerunning generation/evaluation end-to-end additionally needs:

1. **Zenodo artifacts** (https://doi.org/10.5281/zenodo.22128120):
   `subject_snapshots/` (exact source trees, hash-anchored in `SUBJECTS.md`),
   `call_graphs/`, `community_reports/`; verify with the record's
   `SHA256SUMS.zenodo`.
2. **An OpenAI-compatible LLM endpoint** for generation, atomic-fact
   decomposition, and LLM-judged metrics (the paper used GPT-5.5,
   temperature 0, for all roles).
3. **Global only**: a RAGFlow deployment hosting the retrieval corpora
   (`global/pipeline/README.md` documents the setup and per-method
   commands). The exact dataset and chat-assistant configurations used in
   the paper are exported in `global/pipeline/ragflow_config/` (14 datasets,
   25 assistants; re-export with `export_ragflow_config.py`). Remaining
   limitation: bit-identical retrieval additionally requires the same
   RAGFlow version (v0.25.0), the same embedding model
   (`nomic-embed-text@Ollama`, per `ragflow_config/datasets.json`), and the
   code-snapshot documents, whose retrieval inputs are the subject source
   trees on Zenodo; the retrieved contexts are embedded verbatim in the
   frozen per-item files, which is why all analyses above work without
   retrieval.

Local benchmark construction from the call graphs uses
`local/pipeline/build_and_generate.py::build_graphs()` and the extractors
in `local/pipeline/fcg_bench/`. The T3 remediation drivers under
`local/pipeline/v2_t3_final/` (`rebind_t3_7.py`, `rerun_t3_8.py`, …) are
historical one-off scripts, not part of the reproduction path
(see `local/pipeline/README.md`).

## Quick sanity table

| Claim | Value | Verify with |
|---|---|---|
| Global pooled RA-FC: Code / Report / Int / FCG | 0.162 / 0.108 / 0.135 / 0.197 | `export_item_level_csv.py` |
| FCG-RAG pooled gains | +0.035 / +0.089 / +0.061 | `rq1_paired_bootstrap.py` |
| FCG-RAG answer rate vs baselines | 0.505 vs 0.268–0.368 | `rq2_conditional_answer_rate.py` |
| Local overall Qual: Direct / Code-only / Code+AST / FCG | 0.429 / 0.523 / 0.595 / 0.693 | `paper_data_pack.py` |
| Local paired gain (FCG − Code+AST) | +0.098 [0.076, 0.121], p = 3.4e-08 | `paper_data_pack.py` |
| Local win/tie/loss | 284 / 84 / 204 | `paper_data_pack.py` |

## Citing and licensing

See `CITATION.cff`; please cite the paper and the Zenodo record
(https://doi.org/10.5281/zenodo.22128120). Code is MIT (`LICENSE`); data
and benchmarks are CC-BY-4.0 (`DATA_LICENSE`); subject snapshots remain
under their original MIT licenses.
