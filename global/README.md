# FCG-RAG Global Functionality-Understanding Benchmark — Reproduction Package

This package reproduces the **global functionality-understanding** results
(RQ1/RQ2) of the FCG-RAG paper: the benchmark, the frozen per-item model
outputs, and every number, table, and figure reported for the four
source-derived methods (Code-RAG, Report-RAG, Int-RAG, FCG-RAG) across three
subject systems (Ollama N=228, OpenClaw N=245, LangChain N=101; 574 items
total).

## Layout

```
data/
  benchmark/          gold QA benchmark (574 items, gold answers + atomic facts)
  results/            frozen per-item predictions + metrics, gzipped
                      (12 method x system files + 3 closed-book direct controls)
    MANIFEST.json     sha256 checksums, provenance, known limitations
analysis/
  common.py           shared loader: checksum verification, positional
                      alignment check, aggregate check vs the paper table
  export_item_level_csv.py        item_level_all_methods.csv (2296 rows)
  rq1_paired_bootstrap.py         paired RA-FC gains + bootstrap CIs
  rq2_answer_transition.py        transition counts, McNemar, net quality
  rq2_conditional_answer_rate.py  coverage-stratified answer rates (appendix table)
  rq2_branch_contribution.py      RA-FC branch split + closed-book control
  rq2_statistical_controls.py     logistic regression + answered-item OLS
  rq2_metric_correlations.py      Spearman correlations + faithful-answer rates
  figures/                        scripts regenerating every paper figure
pipeline/
  generation + evaluation scripts used to produce the frozen results
  (reference only; see pipeline/README.md for external dependencies)
```

## Quick start

```bash
pip install -r requirements.txt   # numpy, pandas, scipy, statsmodels, matplotlib

cd analysis
python export_item_level_csv.py         # verifies checksums + reproduces the
                                        # published aggregate table (12 cells)
python rq1_paired_bootstrap.py
python rq2_answer_transition.py
python rq2_conditional_answer_rate.py
python rq2_branch_contribution.py
python rq2_statistical_controls.py      # needs statsmodels
python rq2_metric_correlations.py

cd figures
for f in plot_*.py; do python "$f"; done  # PDFs in analysis/output/figures/
```

Every analysis script first re-verifies the data: SHA-256 against
`data/results/MANIFEST.json`, positional item alignment across methods
(items have no stable IDs; the 574-question order is identical in every
file), and agreement of all 12 aggregate table cells with the published
paper at display precision (two documented rounding exceptions, see
MANIFEST.json).

## What each artifact is

- `data/benchmark/*.json` — the gold benchmark: 574 functionality-understanding
  questions derived from official documentation, each with a gold answer and
  gold atomic facts.
- `data/results/<method>/<system>_all_metrics.json.gz` — frozen per-item
  records: question, gold answer/facts, retrieved contexts, generated answer
  and its atomic facts, and all metrics (CtxTok, context recall, faithfulness,
  answer rate, answer correctness, RA-FC). These are the exact files the paper
  numbers were computed from (see MANIFEST.json provenance notes).
- `data/results/direct/` — closed-book control: the same generator answering
  without retrieved context, used in RQ2 to separate evidence use from
  parametric knowledge.

## Regenerating model outputs

The `pipeline/` scripts regenerate answers and recompute metrics, but they
require external services that are not part of this package:

- a **RAGFlow** instance hosting the retrieval corpora (code snapshots and
  community reports), and
- an **OpenAI-compatible API** endpoint for generation, fact decomposition,
  and judging (the paper used GPT-5.5 for all three roles).

Note that the retrieval corpora themselves (code snapshots, community
reports) were not preserved as standalone files (a documented provenance
limitation); the retrieved contexts are, however, embedded verbatim in the
frozen per-item result files, so all metrics can be re-derived from the
frozen data without re-running retrieval. See `pipeline/README.md` for
details and exact commands.

## Citation

If you use this benchmark or these results, please cite the FCG-RAG paper
(citation TBD).
