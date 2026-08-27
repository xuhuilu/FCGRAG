# Pipeline scripts (reference)

These are the exact scripts used to produce the frozen results in
`data/results/`. They are provided for transparency. Re-running them
end-to-end requires two external services:

1. **RAGFlow** (`--ragflow_base_url`, `--ragflow_api_key`,
   `--ragflow_chat_id`) hosting the retrieval corpora. Each method retrieves
   through its own RAGFlow chat assistant:
   - Code-RAG: code snapshots only
   - Report-RAG: community reports only
   - Int-RAG: naive combination of code + report retrieval
   - FCG-RAG: two-stage — community reports produce routing hints, then
     code retrieval produces the final context
2. **An OpenAI-compatible API** (`export SU8_API_KEY=...`) for generation,
   atomic-fact decomposition, and metric judging. The paper used GPT-5.5
   for all three roles, temperature 0.

> Provenance note: the community-report retrieval corpora are preserved in
> the Zenodo data record (`community_reports/`: 2,562 LangChain + 8,585
> OpenClaw + 1,666 Ollama full-content reports), and the subject source
> trees are preserved as `subject_snapshots/` with content-hash anchors.
> The RAGFlow dataset configurations themselves (chunking, embeddings,
> chat-assistant settings) are not preserved, so the retrieval services
> cannot be rebuilt bit-identically; the retrieved contexts are embedded
> verbatim in the frozen per-item files under `data/results/`, which allows
> re-deriving every metric without re-running retrieval.

## Scripts

| Script | Role |
|---|---|
| `generate_closed_book_answer_su8_facts_multi_backend_ragflow_refs_retry5_error_safe.py` | Formal generator: closed-book direct and single-source RAGFlow methods (Code-RAG, Report-RAG); unified answer cleaning + inline atomic-fact decomposition |
| `generate_twostage_int_rag_v5_1.py` | Int-RAG: naive two-source retrieval under a token budget |
| `generate_twostage_mixed_rag.py` | FCG-RAG: two-stage report-guided retrieval + mixed (code + report snippet) context |
| `clean_rag_answer.py` | Shared answer cleaner (required import of the generators) |
| `evaluate_all_metrics_su8_final.py` | Unified evaluator: Ans.R, Ans.C, RA-FC, Faithfulness, Context Recall |
| `evaluate_all_metrics_su8_strict_embedding.py` | Strict-embedding evaluator variant used for Report-RAG on Ollama/LangChain |

## Example commands

```bash
# Code-RAG on Ollama
python generate_closed_book_answer_su8_facts_multi_backend_ragflow_refs_retry5_error_safe.py \
  --system_name Ollama --provider su8 --model gpt-5.5 \
  --method ragflow --method_name Code-RAG \
  --ragflow_base_url "http://your-ragflow-host/api/v1" \
  --ragflow_api_key "$RAGFLOW_API_KEY" \
  --ragflow_chat_id "$OLLAMA_CODERAG_CHAT_ID" \
  --inputs ../data/benchmark/ollama_qa_gold_final_atomic.json \
  --output_dir out/code_rag/ollama --run_metrics

# FCG-RAG on Ollama
python generate_twostage_mixed_rag.py \
  --system_name Ollama --provider su8 --model gpt-5.5 \
  --report_chat_id "$OLLAMA_REPORTRAG_CHAT_ID" \
  --code_chat_id "$OLLAMA_CODERAG_CHAT_ID" \
  --inputs ../data/benchmark/ollama_qa_gold_final_atomic.json \
  --output_dir out/fcg_rag/ollama

# Recompute metrics on a frozen (or regenerated) predictions file
python evaluate_all_metrics_su8_final.py \
  --input out/code_rag/ollama/*_facts_su8_gpt-5.5.json \
  --output out/code_rag/ollama/all_metrics.json
```

Exact flag names may differ slightly per script; run each with `--help`.
The two-stage scripts import the formal generator module at runtime
(`BASE_GEN_MODULE` env var overrides the module name).
