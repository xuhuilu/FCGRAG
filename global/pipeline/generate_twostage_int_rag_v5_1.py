#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TwoStage-Int-RAG generation (v5: decoupled retrieval / fusion / generation).

Architecture (final):
  Stage 1  RAGFlow report chat  -> retrieve community reports
             -> extract routing hints (report text NEVER enters the code query)
             -> remember which local report files (source) were hit
  Stage 2  RAGFlow code chat    -> retrieve code chunks ONLY (its answer is discarded)
             query = question + hints
  Fusion   local report files   -> slice by '## ' sections, pick the most
             question-relevant 1-2 sections (no embedding; lexical overlap),
             truncate to a token budget
  Gen      script-side SU8 (gpt-5.5) or Ollama (qwen3.6:27b)
             prompt = question + code_chunks (primary) + report_snippets (weak concept background)

Why this design:
  - RAGFlow's report text used to leak into the code query and into generation,
    polluting both. Now report only (a) routes via hints and (b) provides a short,
    relevant concept snippet at generation/eval time. Retrieval stays clean.
  - Generation moves to the script so the final context is exactly what we control
    and exactly what faithfulness is scored against. Underlying model (gpt-5.5 /
    qwen3.6) matches the RAGFlow-side baselines, so RA-FC stays comparable.
  - Length is budgeted so "code-only" vs "code+report" differ in CONTENT, not length.

Ablation switch (--use_report_context):
  none       : Code-only. retrieved_contexts = code_chunks. (report used only for hints)
  eval_only  : report_snippets added to retrieved_contexts for EVALUATION, but the
               generator is given code only. Diagnoses "does report help recall
               without touching generation".
  generate   : report_snippets given to the generator AND in retrieved_contexts.
               This is the real fusion (main method).

Field names match the formal generator so evaluate_all_metrics_su8.py works directly.

Example (Code-only baseline, gpt-5.5):
  python generate_twostage_int_rag_v5.py \
    --input langchain_qa_gold_final_atomic.json \
    --output_dir outputs/langchain/gpt55/twostage_v5_codeonly \
    --method_name TwoStage-CodeOnly \
    --report_chat_id "$LANGCHAIN_REPORT_CHAT_ID" \
    --code_chat_id   "$LANGCHAIN_CODERAG_CHAT_ID" \
    --report_md_dir /path/to/community_reports_md \
    --use_report_context none \
    --gen_backend su8 --gen_model gpt-5.5 \
    --limit 10

Example (Mixed / main method, gpt-5.5):
  ... --method_name TwoStage-Mixed --use_report_context generate \
      --code_token_budget 3000 --report_token_budget 700

Example (qwen):
  ... --gen_backend ollama --gen_model qwen3.6:27b --ollama_base_url http://localhost:11434
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


# =============================================================================
# Formal module resolution (RAGFlow retrieval / cleaning / fact decomposition)
# =============================================================================

_BASE_CANDIDATES = [
    "generate_closed_book_answer_su8_facts_multi_backend",
    "generate_closed_book_answer_su8_facts_multi_backend_ragflow_refs_retry5_error_safe",
]

base: Any = None


def resolve_base(explicit_name: str | None) -> Any:
    names: list[str] = []
    if explicit_name:
        names.append(explicit_name)
    if os.environ.get("BASE_GEN_MODULE"):
        names.append(os.environ["BASE_GEN_MODULE"])
    names.extend(_BASE_CANDIDATES)
    last_err: Exception | None = None
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise SystemExit(
        "Could not import the formal generator module. Tried: " + ", ".join(names)
        + f"\nLast error: {last_err}\nUse --base_module or set BASE_GEN_MODULE."
    )


# =============================================================================
# Routing-hint extraction (preserved from prior versions)
# =============================================================================

FUNC_RE = re.compile(r"\b[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)?\b")
PATH_RE = re.compile(
    r"\b(?:libs|langchain|langchain_core|langchain_community|tests|docs|src)/"
    r"[A-Za-z0-9_./-]+\.(?:py|md|rst|toml|yaml|yml)\b"
)
STOP = {
    "the", "and", "for", "with", "from", "this", "that", "when", "then",
    "true", "false", "none", "null", "return", "returns",
    "class", "classes", "function", "functions", "module", "modules",
    "api", "apis", "concept", "concepts",
    "data", "entity", "entities", "relationship", "relationships",
    "community", "report", "project", "part", "rank", "title",
    "source", "document", "chunk", "content",
    "knowledge_type", "community_report_full_content", "report_key",
    "community_id", "doc_id", "chunk_id", "dataset_id",
    "references", "reference", "raw", "metadata",
    "relevant", "related", "generic", "required", "error",
    "message", "messages", "model", "models", "tool", "tools",
    "agent", "agents", "loop", "directives", "test", "tests",
    "code", "routing", "useful", "python", "langchain",
}


def get_question(item: dict[str, Any]) -> str:
    return str(item.get("question") or item.get("user_question") or item.get("query") or "").strip()


def ctx_text(ctx: Any) -> str:
    if isinstance(ctx, dict):
        return str(ctx.get("text") or ctx.get("content") or ctx)
    return str(ctx)


def is_bad_hint(h: str) -> bool:
    h = str(h).strip()
    if not h:
        return True
    low = h.lower().strip()
    if low in STOP:
        return True
    if re.fullmatch(r"c\d+", low):
        return True
    if re.fullmatch(r"\d+", low):
        return True
    if low.startswith("[") and low.endswith("]"):
        return True
    if len(h) < 4 and "_" not in h and "." not in h:
        return True
    if len(h.split()) > 8 and "/" not in h:
        return True
    return False


def is_code_like(tok: str) -> bool:
    if not tok:
        return False
    if "/" in tok and "." in tok:
        return True
    if "_" in tok:
        return True
    if "." in tok:
        return True
    if any(c.isupper() for c in tok[1:]):
        return True
    if tok.startswith(("create_", "init_", "load_", "get_", "set_", "format_", "parse_", "convert_", "bind_")):
        return True
    return False


def extract_question_identifiers(question: str, max_items: int = 12) -> list[str]:
    hints: list[str] = []
    for m in re.finditer(r"`([^`]+)`", question):
        x = m.group(1).strip()
        if x and not is_bad_hint(x):
            hints.append(x)
    for tok in FUNC_RE.findall(question):
        if not is_bad_hint(tok) and is_code_like(tok):
            hints.append(tok)
    out, seen = [], set()
    for h in hints:
        k = h.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
        if len(out) >= max_items:
            break
    return out


def add_question_aware_anchors(question: str, hints: list[str]) -> list[str]:
    q = question.lower()
    anchors: list[str] = []
    if "create_agent" in question or "create agent" in q or "create an agent" in q or "creating an agent" in q:
        anchors += ["create_agent", "tools", "model"]
    if "specific model" in q and "tools" in q:
        anchors += ["create_agent", "model", "tools", "init_chat_model"]
    if "custom tool" in q or "tool function" in q:
        anchors += ["create_agent", "tool", "tools", "def", "@tool"]
    if "standardization" in q or "model provider" in q or "different model providers" in q or "provider lock" in q:
        anchors += ["init_chat_model", "BaseChatModel", "BaseLanguageModel", "model provider", "ChatOpenAI"]
    if "simple agent" in q or "effort" in q or "get started" in q:
        anchors += ["create_agent", "simple agent", "tools", "model"]
    if "alternative way" in q and "model instance" in q:
        anchors += ["ChatOpenAI", "provider-specific", "init_chat_model", "BaseChatModel"]
    if "tool calling" in q:
        anchors += ["tool_calls", "bind_tools", "ToolCall", "BaseTool", "tool inputs"]
    if "llm agent" in q or "core behavior" in q:
        anchors += ["AgentAction", "AgentFinish", "_iter_next_step", "AgentExecutor", "tool"]
    if "quality and capabilities" in q or "chosen model" in q:
        anchors += ["model", "tool calling", "reliability", "performance", "agent trajectory"]
    out, seen = [], set()
    for x in anchors + hints:
        x = str(x).strip()
        if not x:
            continue
        k = x.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(x)
    return out


def extract_hints(question: str, report_answer: str, report_contexts: list[Any], max_hints: int = 15) -> list[str]:
    hints: list[str] = []
    hints.extend(extract_question_identifiers(question))
    texts = [str(report_answer or "")]
    texts.extend(ctx_text(c) for c in (report_contexts or []))
    joined = "\n".join(texts)
    for m in PATH_RE.finditer(joined):
        hints.append(m.group(0))
    for line in joined.splitlines():
        line = line.strip()
        title = None
        if line.startswith("[title]:"):
            title = line.replace("[title]:", "").strip()
        elif line.startswith("# "):
            title = line[2:].strip()
        if not title:
            continue
        if len(title.split()) <= 6 or any(is_code_like(t) for t in FUNC_RE.findall(title)):
            hints.append(title)
    for tok in FUNC_RE.findall(joined):
        if not is_bad_hint(tok) and is_code_like(tok):
            hints.append(tok)
    seen, out = set(), []
    for h in hints:
        h = str(h).strip()
        if is_bad_hint(h):
            continue
        k = h.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(h)
        if len(out) >= max_hints:
            break
    return out


# =============================================================================
# Local community-report slicing  (## section level, no embedding)
# =============================================================================

_META_LINE_RE = re.compile(r"^\[[a-z_]+\]:\s")
_DATA_REF_RE = re.compile(r"\s*\[Data:[^\]]*\]\s*$")
_WORD_RE = re.compile(r"[A-Za-z0-9_]+")


def _strip_data_refs(text: str) -> str:
    out = []
    for line in text.splitlines():
        out.append(_DATA_REF_RE.sub("", line).rstrip())
    return "\n".join(l for l in out if l.strip())


def slice_report_sections(md_text: str) -> list[dict[str, str]]:
    """Split a community-report markdown into '## ' sections.

    Returns list of {title, body, full}. The leading [key]: metadata block and
    the top-level '# ' overview are dropped from section bodies; the overview is
    kept as one extra candidate section so high-level questions can still match.
    """
    lines = md_text.splitlines()
    # drop leading metadata block
    i = 0
    while i < len(lines) and (_META_LINE_RE.match(lines[i]) or not lines[i].strip()):
        i += 1
    body_lines = lines[i:]

    sections: list[dict[str, str]] = []
    cur_title = None
    cur_buf: list[str] = []
    overview_buf: list[str] = []
    in_overview = False

    def flush():
        nonlocal cur_title, cur_buf
        if cur_title is not None:
            text = _strip_data_refs("\n".join(cur_buf)).strip()
            if text:
                sections.append({"title": cur_title, "body": text, "full": f"{cur_title}\n{text}"})
        cur_title, cur_buf = None, []

    for ln in body_lines:
        if ln.startswith("## "):
            if in_overview:
                in_overview = False
            flush()
            cur_title = ln[3:].strip()
        elif ln.startswith("# "):
            in_overview = True
            overview_buf = []
        else:
            if in_overview:
                overview_buf.append(ln)
            elif cur_title is not None:
                cur_buf.append(ln)
    flush()

    ov = _strip_data_refs("\n".join(overview_buf)).strip()
    if ov:
        sections.insert(0, {"title": "Overview", "body": ov, "full": ov})
    return sections


def _tokset(text: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(text or "") if len(w) > 2}


def score_section(section: dict[str, str], q_terms: set[str], hint_terms: set[str]) -> float:
    sec_terms = _tokset(section["full"])
    if not sec_terms:
        return 0.0
    q_overlap = len(sec_terms & q_terms)
    h_overlap = len(sec_terms & hint_terms)
    # question terms weighted higher than hints; normalize lightly by sqrt length
    return (2.0 * q_overlap + 1.0 * h_overlap) / (len(sec_terms) ** 0.5)


def select_report_snippets(
    question: str,
    hints: list[str],
    report_contexts: list[dict[str, Any]],
    report_md_dir: Path,
    top_files: int,
    top_sections: int,
    token_budget: int,
) -> list[dict[str, Any]]:
    """Pick the most question-relevant '## ' sections from the hit report files."""
    if not report_md_dir or token_budget <= 0:
        return []

    # rank report files by Stage-1 score, take top_files distinct sources
    ranked = sorted(
        [c for c in (report_contexts or []) if isinstance(c, dict) and c.get("source")],
        key=lambda c: float(c.get("score") or c.get("similarity") or 0.0),
        reverse=True,
    )
    seen_src, src_list = set(), []
    for c in ranked:
        s = c["source"]
        if s in seen_src:
            continue
        seen_src.add(s)
        src_list.append(s)
        if len(src_list) >= top_files:
            break

    q_terms = _tokset(question)
    hint_terms = set()
    for h in hints:
        hint_terms |= _tokset(h)

    candidates: list[dict[str, Any]] = []
    for src in src_list:
        fp = report_md_dir / src
        if not fp.exists():
            continue
        try:
            md = fp.read_text(encoding="utf-8", errors="ignore")
        except Exception:
            continue
        secs = slice_report_sections(md)
        scored = sorted(secs, key=lambda s: score_section(s, q_terms, hint_terms), reverse=True)
        for sec in scored[:top_sections]:
            candidates.append({
                "source": src,
                "title": sec["title"],
                "text": sec["full"],
                "score": round(score_section(sec, q_terms, hint_terms), 4),
                "origin": "community_report_local",
            })

    # global rank across files, fill to token budget
    candidates.sort(key=lambda c: c["score"], reverse=True)
    snippets: list[dict[str, Any]] = []
    used = 0
    for c in candidates:
        tk = base.count_tokens(c["text"])
        if used + tk > token_budget and snippets:
            continue
        c["token_count"] = tk
        snippets.append(c)
        used += tk
        if used >= token_budget:
            break
    return snippets


def budget_code_chunks(code_contexts: list[dict[str, Any]], token_budget: int) -> list[dict[str, Any]]:
    """Keep highest-ranked code chunks up to a token budget (preserves order)."""
    if token_budget <= 0:
        return code_contexts
    out, used = [], 0
    for c in code_contexts:
        tk = int(c.get("token_count") or base.count_tokens(c.get("text", "")))
        if used + tk > token_budget and out:
            break
        out.append(c)
        used += tk
    return out


# =============================================================================
# Script-side generation (SU8 / Ollama), context fully controlled here
# =============================================================================

GEN_SYSTEM_PROMPT = """You answer questions about software systems using retrieved evidence.

Evidence priority:
- CODE EVIDENCE is the primary source. Base concrete facts (commands, APIs, function names, arguments, code) on it.
- CONCEPT BACKGROUND (community-report summaries) is secondary. Use it only for high-level framing; do not treat it as a source of exact code-level details.

Answering rules:
- Answer only what the question asks. Prefer 1-3 concise sentences.
- Use only the provided evidence. Do not invent commands, ports, endpoints, key bindings, parameters, default values, configuration keys, or code snippets not present in the evidence.
- Prefer current/non-deprecated APIs when the evidence distinguishes old and new.
- If the evidence does not support an answer, answer exactly: I don't know.
- Do not mention "evidence", "context", "report", or that you are using retrieval.
- Output only the answer text. Do not add any preamble, labels, headings, or formatting.
"""


def build_gen_user_prompt(question: str, code_chunks: list[dict[str, Any]], report_snippets: list[dict[str, Any]]) -> str:
    code_block = "\n\n".join(
        f"[Code {i+1}] {c.get('text','')}" for i, c in enumerate(code_chunks)
    ) or "(none)"
    if report_snippets:
        concept_block = "\n\n".join(
            f"[Concept {i+1}] {s.get('text','')}" for i, s in enumerate(report_snippets)
        )
        concept_section = f"\n\nConcept background (secondary, high-level only):\n{concept_block}"
    else:
        concept_section = ""
    return f"""Question:
{question}

Code evidence (primary):
{code_block}{concept_section}

Answer the question following the rules. Code evidence is authoritative for concrete details."""


def generate_answer(question: str, code_chunks: list[dict[str, Any]], report_snippets: list[dict[str, Any]], cfg: dict[str, Any]) -> str:
    """Generate the final answer on the script side using SU8 or Ollama."""
    system = GEN_SYSTEM_PROMPT
    user = build_gen_user_prompt(question, code_chunks, report_snippets)

    if cfg["gen_backend"] == "su8":
        # reuse base.call_su8_answer_json shape but with our own prompts
        client = cfg["su8_client"]
        last_err = None
        for attempt in range(1, 6):
            try:
                kwargs = {
                    "model": cfg["gen_model"],
                    "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                    "temperature": 0,
                    "max_tokens": 400,
                    "response_format": base.answer_response_format_schema(),
                }
                try:
                    resp = client.chat.completions.create(**kwargs)
                except Exception as e:
                    if "json_schema" in str(e) or "response_format" in str(e):
                        kwargs["response_format"] = {"type": "json_object"}
                        resp = client.chat.completions.create(**kwargs)
                    else:
                        raise
                content = resp.choices[0].message.content or ""
                parsed = base.extract_json_object(content)
                return base.normalize_answer_text(str(parsed.get("answer", "")).strip())
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(60, 2 * attempt))
        raise RuntimeError(f"SU8 generation failed: {last_err}")

    elif cfg["gen_backend"] == "ollama":
        # Plain-text generation for qwen. We deliberately do NOT use Ollama's
        # structured-output (`format` schema) here: with long code context it
        # collides with qwen's generation and yields empty/invalid JSON
        # ("No JSON object found"). The system prompt asks for answer-only text,
        # and we take message.content directly. Thinking is left ON to stay
        # consistent with prior qwen experiments; any stray <think> block is
        # stripped defensively.
        gen_opts = {"temperature": 0, "top_p": 0.9, "num_predict": 300, "num_ctx": 8192}
        last_err = None
        for attempt in range(1, 4):
            try:
                payload = {
                    "model": cfg["gen_model"],
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                    "options": gen_opts,
                    # no `format`: plain text, taken as-is
                }
                resp = base.post_ollama_chat(cfg["ollama_base_url"], payload)
                content = resp.get("message", {}).get("content", "") or ""
                content = re.sub(r"(?is)<think>.*?</think>", "", content).strip()
                if not content:
                    raise ValueError("empty content from ollama")
                return base.normalize_answer_text(content)
            except Exception as e:  # noqa: BLE001
                last_err = e
                time.sleep(min(30, 2 * attempt))
        raise RuntimeError(f"Ollama generation failed: {last_err}")

    raise ValueError(f"Unknown gen_backend: {cfg['gen_backend']}")


# =============================================================================
# Prompts for RAGFlow retrieval stages
# =============================================================================

def build_report_prompt(question: str) -> str:
    return f"""You are performing routing, not final answering.

Original question:
{question}

Retrieve relevant community reports. Return concise routing hints (function/class/method names, file paths, modules, APIs, config keys). Do not answer the question directly."""


def build_code_query(question: str, hints: list[str]) -> str:
    hints_text = "\n".join(f"- {h}" for h in hints[:15])
    return f"""Retrieve code evidence to answer the question.

Question:
{question}

Search hints:
{hints_text}

Return the most relevant code for this exact question."""


# =============================================================================
# Two-stage retrieval + fusion + generation
# =============================================================================

def _ragflow_retrieve(prompt: str, chat_id: str, cfg: dict[str, Any]):
    raw_answer, reference, raw_info = base.run_ragflow_mode(
        question=prompt,
        ragflow_base_url=cfg["base_url"],
        ragflow_api_key=cfg["api_key"],
        ragflow_chat_id=chat_id,
        ragflow_model=cfg["ragflow_model"],
        endpoint_style=cfg["endpoint_style"],
    )
    contexts = base.extract_contexts_from_ragflow_reference(reference)
    return str(raw_answer or ""), reference, raw_info, contexts


def run_two_stage(question: str, cfg: dict[str, Any]) -> dict[str, Any]:
    # Stage 1: report routing (hints only; report text not used as code query)
    report_raw, report_ref, report_info, report_contexts = _ragflow_retrieve(
        build_report_prompt(question), cfg["report_chat_id"], cfg
    )
    hints = extract_hints(question, report_raw, report_contexts, max_hints=cfg["max_hints"])
    if not cfg["no_question_anchors"]:
        hints = add_question_aware_anchors(question, hints)
    hints = hints[: cfg["max_hints"]]

    # Stage 2: code retrieval ONLY (discard RAGFlow's answer)
    _code_raw, code_ref, code_info, code_contexts = _ragflow_retrieve(
        build_code_query(question, hints), cfg["code_chat_id"], cfg
    )
    code_contexts = budget_code_chunks(code_contexts, cfg["code_token_budget"])

    # Fusion: local report sections, question-relevant, budgeted
    report_snippets: list[dict[str, Any]] = []
    if cfg["use_report_context"] in {"eval_only", "generate"}:
        report_snippets = select_report_snippets(
            question=question,
            hints=hints,
            report_contexts=report_contexts,
            report_md_dir=cfg["report_md_dir"],
            top_files=cfg["report_top_files"],
            top_sections=cfg["report_top_sections"],
            token_budget=cfg["report_token_budget"],
        )

    # Generation: code is primary; report given to generator only in 'generate' mode
    gen_report = report_snippets if cfg["use_report_context"] == "generate" else []
    answer = generate_answer(question, code_contexts, gen_report, cfg)

    # Evaluation contexts: code always; report added in eval_only/generate
    if cfg["use_report_context"] in {"eval_only", "generate"}:
        eval_contexts = list(code_contexts) + list(report_snippets)
    else:
        eval_contexts = list(code_contexts)

    return {
        "answer": answer,
        "code_contexts": code_contexts,
        "code_ref": code_ref,
        "code_info": code_info,
        "report_contexts": report_contexts,
        "report_snippets": report_snippets,
        "report_ref": report_ref,
        "report_info": report_info,
        "hints": hints,
        "eval_contexts": eval_contexts,
    }


# =============================================================================
# Item processing
# =============================================================================

def is_complete_item(item: dict[str, Any], method: str, method_name: str, fact_model: str) -> bool:
    answer_ok = (
        item.get("method") == method
        and item.get("method_name") == method_name
        and bool(item.get("format_answer"))
        and item.get("answer_status") == "success"
    )
    facts_ok = (
        item.get("fact_model") == fact_model
        and "format_answer_atomic_facts" in item
        and item.get("fact_status") == "success"
    )
    return bool(answer_ok and facts_ok)


def process_item(item: dict[str, Any], question: str, cfg: dict[str, Any], fact_client: Any) -> None:
    gold_answer = base.extract_gold_answer(item)
    gold_atomic_facts = base.extract_gold_atomic_facts(item)

    gen = run_two_stage(question, cfg)
    clean_answer = base.normalize_answer_text(base.clean_rag_answer_for_eval(gen["answer"] or ""))

    facts = base.call_su8_decompose_facts(client=fact_client, model=cfg["fact_model"], answer=clean_answer)

    eval_contexts = gen["eval_contexts"]
    ctx_tok = sum(int(c.get("token_count") or base.count_tokens(c.get("text", ""))) for c in eval_contexts)

    item.update({
        "system_name": cfg["system_name"],
        "generator_provider": cfg["gen_backend"],
        "generator_model": cfg["gen_model"],
        "method": cfg["method"],
        "method_name": cfg["method_name"],
        "fact_provider": "su8",
        "fact_model": cfg["fact_model"],
        "fact_base_url": cfg["fact_base_url"],
        "answer_cleaner": getattr(base, "ANSWER_CLEANER_VERSION", "clean_rag_answer_v3"),
        "use_report_context": cfg["use_report_context"],
    })

    item["gold_answer"] = gold_answer
    item["gold_atomic_facts"] = gold_atomic_facts
    item["format_answer"] = clean_answer
    item["format_answer_atomic_facts"] = facts
    item["generated_answer"] = clean_answer
    item["generated_atomic_facts"] = facts

    item["retrieved_contexts"] = eval_contexts
    item["retrieved_context_token_count"] = ctx_tok
    item["ctx_tok"] = ctx_tok
    item["ragflow_reference_contexts"] = gen["code_contexts"]
    item["ragflow_reference_context_count"] = len(gen["code_contexts"])
    item["raw_ragflow_reference"] = gen["code_ref"]
    item["ragflow_reference"] = gen["code_ref"]
    item["raw_generation_info"] = gen["code_info"]

    # diagnostics
    item["code_context_count"] = len(gen["code_contexts"])
    item["code_context_token_count"] = sum(int(c.get("token_count") or 0) for c in gen["code_contexts"])
    item["report_snippet_count"] = len(gen["report_snippets"])
    item["report_snippet_token_count"] = sum(int(c.get("token_count") or 0) for c in gen["report_snippets"])
    item["report_snippet_sources"] = [s.get("source") for s in gen["report_snippets"]]
    item["report_snippet_titles"] = [s.get("title") for s in gen["report_snippets"]]
    item["twostage_report_contexts"] = gen["report_contexts"]
    item["twostage_report_context_count"] = len(gen["report_contexts"])
    item["twostage_routing_hints"] = gen["hints"]

    item["answer_word_count"] = base.word_count(clean_answer)
    item["answer_token_count"] = base.count_tokens(clean_answer)
    item["atomic_fact_count"] = len(facts)
    item["answer_status"] = "success"
    item["fact_status"] = "success"


# =============================================================================
# File loop
# =============================================================================

def _save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def process_file(input_path: Path, output_path: Path, cfg: dict[str, Any], fact_client: Any) -> None:
    if output_path.exists() and not cfg["overwrite"]:
        print(f"[Resume] {output_path}", flush=True)
        data = base.load_json(output_path)
    else:
        print(f"[Read] {input_path}", flush=True)
        data = base.load_json(input_path)
    items = data if isinstance(data, list) else data.get("items", data)

    processed = success = failed = skipped = 0
    total = len(items) if hasattr(items, "__len__") else 0
    print(f"[Process] -> {output_path.name} total={total} method={cfg['method_name']} "
          f"report_mode={cfg['use_report_context']} gen={cfg['gen_backend']}:{cfg['gen_model']}", flush=True)

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue
        if cfg["limit"] is not None and processed >= cfg["limit"]:
            break
        if is_complete_item(item, cfg["method"], cfg["method_name"], cfg["fact_model"]) and not cfg["overwrite"]:
            skipped += 1
            continue

        question = get_question(item)
        print(f"[{i+1}/{total}] {question[:90]}", flush=True)
        if not question:
            item["format_answer"] = ""
            item["format_answer_atomic_facts"] = []
            item["answer_status"] = "skipped_missing_question"
            item["fact_status"] = "skipped_missing_question"
            processed += 1
            continue

        try:
            process_item(item, question, cfg, fact_client)
            ans_tag = "IDK" if item["format_answer"].lower().startswith("i don't know") else "ANS"
            print(f"  -> facts={item['atomic_fact_count']} code={item['code_context_count']} "
                  f"report_snip={item['report_snippet_count']} ctx_tok={item['ctx_tok']} "
                  f"ans={ans_tag}", flush=True)
            success += 1
        except Exception as e:  # noqa: BLE001
            item["format_answer"] = ""
            item["format_answer_atomic_facts"] = []
            item["retrieved_contexts"] = []
            item["retrieved_context_token_count"] = None
            item["ctx_tok"] = None
            item["answer_status"] = "failed"
            item["fact_status"] = "failed"
            item["generation_error"] = str(e)
            failed += 1
            print(f"  [Failed] {i}: {e}", flush=True)

        processed += 1
        if cfg["save_every"] > 0 and processed % cfg["save_every"] == 0:
            _save(output_path, data)
            print(f"  [Checkpoint] {output_path}", flush=True)
        if cfg["sleep"] > 0:
            time.sleep(cfg["sleep"])

    _save(output_path, data)
    print(f"[Done] processed={processed} success={success} failed={failed} skipped={skipped}\n  {output_path}\n")


def count_answer_status(items: list[dict[str, Any]]) -> dict[str, Any]:
    answered = idk = empty = facts_ne = 0
    ctx_toks = []
    for it in items:
        ans = str(it.get("format_answer") or "").strip()
        if not ans:
            empty += 1
        elif ans.lower().startswith("i don't know"):
            idk += 1
        else:
            answered += 1
        if it.get("format_answer_atomic_facts"):
            facts_ne += 1
        if it.get("ctx_tok"):
            ctx_toks.append(it["ctx_tok"])
    n = len(items)
    done = [x for x in items if x.get("format_answer")]
    nd = len(done) or 1
    return {
        "N": n,
        "generated": len(done),
        "answered": answered,
        "answer_rate_over_generated": round(answered / nd, 4),
        "idk": idk,
        "empty": empty,
        "facts_nonempty": facts_ne,
        "avg_ctx_tok_over_generated": round(sum(ctx_toks) / len(ctx_toks), 1) if ctx_toks else None,
    }


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="TwoStage v5: RAGFlow retriever + local report fusion + script-side generation.")
    ap.add_argument("--input", default="langchain_qa_gold_final_atomic.json")
    ap.add_argument("--output_dir", default="outputs/langchain/twostage_v5")
    ap.add_argument("--output", default=None)
    ap.add_argument("--system_name", default=None)
    ap.add_argument("--method", default="TwoStage-Int-RAG")
    ap.add_argument("--method_name", default="TwoStage-Int-RAG")

    # RAGFlow (retriever only)
    ap.add_argument("--report_chat_id", default=os.environ.get("LANGCHAIN_REPORT_CHAT_ID", ""))
    ap.add_argument("--code_chat_id", default=os.environ.get("LANGCHAIN_CODERAG_CHAT_ID", os.environ.get("LANGCHAIN_CODE_CHAT_ID", "")))
    ap.add_argument("--ragflow_base_url", default=os.environ.get("RAGFLOW_BASE_URL", ""))
    ap.add_argument("--ragflow_api_key", default=os.environ.get("RAGFLOW_API_KEY", ""))
    ap.add_argument("--ragflow_model", default=os.environ.get("RAGFLOW_MODEL", "ragflow-model"))
    ap.add_argument("--ragflow_endpoint_style", default="chats_openai", choices=["auto", "openai", "chats_openai"])

    # hints
    ap.add_argument("--max_hints", type=int, default=15)
    ap.add_argument("--no_question_anchors", action="store_true")

    # local report fusion
    ap.add_argument("--report_md_dir", default=None,
                    help="directory of exported community-report markdown files (see Zenodo data record)")
    ap.add_argument("--use_report_context", choices=["none", "eval_only", "generate"], default="none")
    ap.add_argument("--report_top_files", type=int, default=2)
    ap.add_argument("--report_top_sections", type=int, default=1)
    ap.add_argument("--report_token_budget", type=int, default=700)
    ap.add_argument("--code_token_budget", type=int, default=3500,
                    help="Token cap for code chunks. Lower it when adding report to keep total length comparable to code-only.")

    # generation (script-side)
    ap.add_argument("--gen_backend", choices=["su8", "ollama"], default="su8")
    ap.add_argument("--gen_model", default="gpt-5.5")
    ap.add_argument("--su8_base_url", default="https://www.su8.codes/codex/v1")
    ap.add_argument("--ollama_base_url", default="http://localhost:11434")

    # facts (always SU8 for cross-model comparability)
    ap.add_argument("--fact_model", default="gpt-5.5")
    ap.add_argument("--fact_base_url", default="https://www.su8.codes/codex/v1")

    # control
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--base_module", default=None)

    args = ap.parse_args()

    global base
    base = resolve_base(args.base_module)

    for need, val in [("--report_chat_id", args.report_chat_id), ("--code_chat_id", args.code_chat_id),
                      ("--ragflow_base_url", args.ragflow_base_url), ("--ragflow_api_key", args.ragflow_api_key)]:
        if not val:
            raise SystemExit(f"Missing {need}")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")
    report_md_dir = Path(args.report_md_dir)
    if args.use_report_context != "none" and not report_md_dir.exists():
        raise SystemExit(f"--report_md_dir not found: {report_md_dir}")

    system_name = args.system_name or base.guess_system_name(input_path)

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("pip install openai") from exc
    su8_api_key = base.get_su8_api_key()
    fact_client = OpenAI(base_url=args.fact_base_url, api_key=su8_api_key, timeout=120)
    su8_gen_client = OpenAI(base_url=args.su8_base_url, api_key=su8_api_key, timeout=120) if args.gen_backend == "su8" else None

    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        out_name = (input_path.stem
                    + f"_{base.sanitize_name(args.method_name)}"
                    + f"_{args.gen_backend}_{base.sanitize_name(args.gen_model)}"
                    + f"_rep-{args.use_report_context}_facts_su8_{base.sanitize_name(args.fact_model)}.json")
        out_path = out_dir / out_name

    cfg = {
        "system_name": system_name, "method": args.method, "method_name": args.method_name,
        "base_url": args.ragflow_base_url, "api_key": args.ragflow_api_key,
        "ragflow_model": args.ragflow_model, "endpoint_style": args.ragflow_endpoint_style,
        "report_chat_id": args.report_chat_id, "code_chat_id": args.code_chat_id,
        "max_hints": args.max_hints, "no_question_anchors": args.no_question_anchors,
        "report_md_dir": report_md_dir, "use_report_context": args.use_report_context,
        "report_top_files": args.report_top_files, "report_top_sections": args.report_top_sections,
        "report_token_budget": args.report_token_budget, "code_token_budget": args.code_token_budget,
        "gen_backend": args.gen_backend, "gen_model": args.gen_model,
        "ollama_base_url": args.ollama_base_url, "su8_client": su8_gen_client,
        "fact_model": args.fact_model, "fact_base_url": args.fact_base_url,
        "limit": args.limit, "save_every": args.save_every, "sleep": args.sleep, "overwrite": args.overwrite,
    }

    print("\n========== TwoStage v5 Config ==========")
    print("Base module:", base.__name__)
    print("Output:", out_path)
    print("System:", system_name, "| Method:", args.method_name)
    print("Report mode:", args.use_report_context, "| report_md_dir:", report_md_dir)
    print("Gen:", args.gen_backend, args.gen_model, "| Fact:", args.fact_model)
    print("Budgets: code=", args.code_token_budget, "report=", args.report_token_budget)
    print("========================================\n")

    process_file(input_path, out_path, cfg, fact_client)

    data = base.load_json(out_path)
    items = data if isinstance(data, list) else data.get("items", data)
    stats = count_answer_status([x for x in items if isinstance(x, dict)])
    _save(out_path.with_name(out_path.stem + "_genstats.json"), stats)
    print("[GenStats]")
    for k, v in stats.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
