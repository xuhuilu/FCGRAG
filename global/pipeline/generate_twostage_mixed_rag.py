#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
TwoStage-Int-RAG / TwoStage-Mixed generation (clean v5).

What changed vs the old MVP (v311):
  - The old MVP set `format_answer_atomic_facts = []` and relied on the
    evaluator to decompose later. That fallback only reached answer_correctness,
    so RA-FC / faithfulness collapsed to 0 / None. THIS script decomposes the
    final answer into atomic facts INLINE (same path as the formal multi-backend
    generator), so format_answer_atomic_facts is populated at generation time.
  - All RAGFlow calling, error-string detection, context standardization, the
    unified v3 answer cleaner, and SU8 fact decomposition are REUSED from the
    formal module (no duplicated logic, no signature-guessing via inspect).
  - Field names match the formal generator so evaluate_all_metrics_su8.py and
    the summary builder work without any fallback flags.
  - Resume treats RAGFlow-error items as NOT complete, so transient failures are
    retried instead of being frozen as "I don't know.".

Pipeline:
  Stage 1 (router): community-report RAGFlow chat -> routing hints for code search.
  Stage 2 (answerer): code-RAG RAGFlow chat -> final evidence + answer.

  Default / ablation mode:
    - retrieved_contexts = Stage 2 code contexts only.
    - Stage 1 report contexts are kept as diagnostics.

  Mixed mode (--mixed_context):
    - Select compact community-report snippets from Stage 1 contexts.
    - Put selected report snippets into the Stage 2 final-answer prompt.
    - retrieved_contexts = selected report snippets + Stage 2 code contexts.

  Community reports are assumed to be source-code-derived summaries, not
  external documentation or third-party tutorials.

The formal module is imported as `base`. By default it is:
  generate_closed_book_answer_su8_facts_multi_backend
Override with --base_module or env BASE_GEN_MODULE if your filename differs
(e.g. generate_closed_book_answer_su8_facts_multi_backend_ragflow_refs_retry5_error_safe).

Example:
  export SU8_API_KEY=...
  python generate_twostage_int_rag.py \
    --input langchain_qa_gold_final_atomic.json \
    --output_dir outputs/langchain/gpt55/twostage_int_rag_v4 \
    --ragflow_base_url "$RAGFLOW_BASE_URL" \
    --ragflow_api_key  "$RAGFLOW_API_KEY" \
    --report_chat_id   "$LANGCHAIN_REPORT_CHAT_ID" \
    --code_chat_id     "$LANGCHAIN_CODERAG_CHAT_ID" \
    --method_name TwoStage-Int-RAG \
    --run_metrics
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
# Formal module resolution (reused for RAGFlow / cleaning / fact decomposition)
# =============================================================================

_BASE_CANDIDATES = [
    "generate_closed_book_answer_su8_facts_multi_backend",
    "generate_closed_book_answer_su8_facts_multi_backend_ragflow_refs_retry5_error_safe",
]

base: Any = None  # set in resolve_base()


def resolve_base(explicit_name: str | None) -> Any:
    names: list[str] = []
    if explicit_name:
        names.append(explicit_name)
    env_name = os.environ.get("BASE_GEN_MODULE")
    if env_name:
        names.append(env_name)
    names.extend(_BASE_CANDIDATES)

    last_err: Exception | None = None
    for name in names:
        try:
            return importlib.import_module(name)
        except Exception as e:  # noqa: BLE001 - we want to try the next candidate
            last_err = e
            continue

    raise SystemExit(
        "Could not import the formal generator module. Tried: "
        + ", ".join(names)
        + f"\nLast error: {last_err}\n"
        "Pass the correct module name with --base_module, or set BASE_GEN_MODULE. "
        "The module (and clean_rag_answer.py) must be importable from this directory."
    )


# =============================================================================
# Routing-hint extraction (domain-specific; preserved from the MVP)
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
    "code", "routing", "useful",
    "python", "langchain",
}


def get_question(item: dict[str, Any]) -> str:
    return str(
        item.get("question")
        or item.get("user_question")
        or item.get("query")
        or ""
    ).strip()


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
    if re.fullmatch(r"c\d+", low):          # GraphRAG community IDs
        return True
    if re.fullmatch(r"\d+", low):           # pure numbers / ids
        return True
    if low.startswith("[") and low.endswith("]"):
        return True
    if len(h) < 4 and "_" not in h and "." not in h:
        return True
    if len(h.split()) > 8 and "/" not in h:  # long report titles dominate retrieval
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
    if any(c.isupper() for c in tok[1:]):    # CamelCase / PascalCase
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
        if is_bad_hint(tok):
            continue
        if is_code_like(tok):
            hints.append(tok)

    out: list[str] = []
    seen: set[str] = set()
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
    """Fair query-aware anchors based only on terms present in the question."""
    q = question.lower()
    anchors: list[str] = []

    if "create_agent" in question or "create agent" in q or "create an agent" in q or "creating an agent" in q:
        anchors.extend(["create_agent", "tools", "model"])
    if "specific model" in q and "tools" in q:
        anchors.extend(["create_agent", "model", "tools", "init_chat_model"])
    if "custom tool" in q or "tool function" in q:
        anchors.extend(["create_agent", "tool", "tools", "def", "@tool"])
    if "standardization" in q or "model provider" in q or "different model providers" in q or "provider lock" in q:
        anchors.extend(["init_chat_model", "BaseChatModel", "BaseLanguageModel", "model provider", "ChatOpenAI"])
    if "simple agent" in q or "effort" in q or "get started" in q:
        anchors.extend(["create_agent", "simple agent", "tools", "model"])
    if "alternative way" in q and "model instance" in q:
        anchors.extend(["ChatOpenAI", "provider-specific", "init_chat_model", "BaseChatModel"])
    if "tool calling" in q:
        anchors.extend(["tool_calls", "bind_tools", "ToolCall", "BaseTool", "tool inputs"])
    if "llm agent" in q or "core behavior" in q:
        anchors.extend(["AgentAction", "AgentFinish", "_iter_next_step", "AgentExecutor", "tool"])
    if "quality and capabilities" in q or "chosen model" in q:
        anchors.extend(["model", "tool calling", "reliability", "performance", "agent trajectory"])

    out: list[str] = []
    seen: set[str] = set()
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


def extract_hints(question: str, report_answer: str, report_contexts: list[Any], max_hints: int = 25) -> list[str]:
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
        has_code_like = any(is_code_like(t) for t in FUNC_RE.findall(title))
        if len(title.split()) <= 6 or has_code_like:
            hints.append(title)

    for tok in FUNC_RE.findall(joined):
        if is_bad_hint(tok):
            continue
        if is_code_like(tok):
            hints.append(tok)

    seen: set[str] = set()
    out: list[str] = []
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


def _count_tokens_safe(text: str) -> int:
    """Token counting helper that falls back to a rough estimate if base is unavailable."""
    text = str(text or "")
    try:
        if base is not None and hasattr(base, "count_tokens"):
            return int(base.count_tokens(text))
    except Exception:
        pass
    # Rough fallback: English/code tokens are usually close to 3-4 chars/token.
    return max(1, len(text) // 4) if text else 0


def _split_report_sections(text: str) -> list[dict[str, str]]:
    """
    Split a community-report context into [title]-anchored sections.

    The user's report format is expected to contain lines like:
      [title]: LangChain Tool Calling, Runnable Tracing, ...

    Some RAGFlow chunks may contain exactly one report; others may contain
    multiple reports or no explicit title. This function handles all cases.
    """
    text = str(text or "").strip()
    if not text:
        return []

    title_re = re.compile(r"(?m)^\s*(?:\[title\]:|#)\s*(.+?)\s*$")
    matches = list(title_re.finditer(text))

    if not matches:
        # Keep a title-less fallback instead of dropping potentially useful report evidence.
        return [{"title": "Untitled community report", "body": text}]

    sections: list[dict[str, str]] = []
    for i, m in enumerate(matches):
        title = m.group(1).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        if not body:
            # If the title line itself is the only evidence, still keep it.
            body = title
        sections.append({"title": title, "body": body})

    return sections


def _question_terms(question: str) -> set[str]:
    """Extract query terms used for simple, transparent snippet scoring."""
    terms: set[str] = set()
    for t in re.findall(r"[A-Za-z_][A-Za-z0-9_./-]*", str(question or "")):
        low = t.lower().strip()
        if len(low) < 4:
            continue
        if low in STOP:
            continue
        terms.add(low)
    return terms


def _score_report_section(question: str, title: str, body: str) -> float:
    """Score a report section for inclusion in the mixed final context."""
    q_terms = _question_terms(question)
    hay = f"{title}\n{body}".lower()

    overlap = sum(1 for t in q_terms if t in hay)
    title_overlap = sum(1 for t in q_terms if t in str(title or "").lower())

    # Code-like identifiers in the question are especially important.
    q_ids = [t for t in extract_question_identifiers(question, max_items=20)]
    id_overlap = sum(1 for t in q_ids if t.lower() in hay)

    has_title = 1.0 if title and title != "Untitled community report" else 0.0
    code_like_in_title = 1.0 if any(is_code_like(t) for t in FUNC_RE.findall(title or "")) else 0.0

    return (
        1.00 * overlap
        + 1.50 * title_overlap
        + 2.00 * id_overlap
        + 0.50 * has_title
        + 0.50 * code_like_in_title
    )


def _compact_report_body(body: str, question: str, max_chars: int) -> str:
    """
    Keep report details compact. Prefer lines overlapping the question, but fall
    back to the report prefix when no line-level overlap exists.
    """
    body = str(body or "").strip()
    if len(body) <= max_chars:
        return body

    q_terms = _question_terms(question)
    lines = [ln.strip() for ln in body.splitlines() if ln.strip()]

    scored: list[tuple[float, int, str]] = []
    for i, ln in enumerate(lines):
        low = ln.lower()
        score = sum(1 for t in q_terms if t in low)
        # Keep lines with identifiers / paths because they are useful bridges.
        if any(is_code_like(tok) for tok in FUNC_RE.findall(ln)):
            score += 0.5
        scored.append((score, -i, ln))

    useful = [ln for score, _neg_i, ln in sorted(scored, reverse=True) if score > 0]
    if not useful:
        return body[:max_chars].strip()

    picked: list[str] = []
    total = 0
    for ln in useful:
        add = len(ln) + 1
        if total + add > max_chars:
            continue
        picked.append(ln)
        total += add
        if total >= max_chars:
            break

    if not picked:
        return body[:max_chars].strip()
    return "\n".join(picked).strip()


def select_report_snippets(
    question: str,
    report_contexts: list[Any],
    max_reports: int = 3,
    max_chars_per_report: int = 1200,
    min_score: float = 0.0,
) -> list[dict[str, Any]]:
    """
    Select compact source-code-derived community-report snippets for mixed RAG.

    These snippets are intended as high-level evidence: they explain module-level
    intent and relationships. Source code remains authoritative for exact API
    behavior.
    """
    candidates: list[tuple[float, int, dict[str, Any]]] = []

    for ctx_idx, ctx in enumerate(report_contexts or []):
        raw_text = ctx_text(ctx).strip()
        if not raw_text:
            continue

        sections = _split_report_sections(raw_text)
        for sec_idx, sec in enumerate(sections):
            title = str(sec.get("title") or "Untitled community report").strip()
            body = str(sec.get("body") or "").strip()
            if not body and not title:
                continue

            score = _score_report_section(question, title, body)
            compact_body = _compact_report_body(body, question, max_chars_per_report)
            snippet_text = (
                "[Community Report Evidence]\n"
                f"Title: {title}\n"
                "Relevant details:\n"
                f"{compact_body}"
            ).strip()

            # Preserve some original metadata if present, but keep the output stable.
            out: dict[str, Any] = {
                "text": snippet_text,
                "content": snippet_text,
                "source": "community_report_selected_snippet",
                "source_stage": "stage1_report",
                "title": title,
                "score": score,
                "original_context_rank": ctx_idx + 1,
                "section_rank_in_context": sec_idx + 1,
                "token_count": _count_tokens_safe(snippet_text),
            }
            if isinstance(ctx, dict):
                for k in ("doc_id", "chunk_id", "document_id", "dataset_id", "id", "filename"):
                    if k in ctx:
                        out[f"original_{k}"] = ctx.get(k)

            candidates.append((score, ctx_idx, out))

    # Higher score first; if tied, keep earlier RAGFlow-ranked context.
    candidates.sort(key=lambda x: (-x[0], x[1]))

    selected: list[dict[str, Any]] = []
    seen_titles: set[str] = set()
    for score, _idx, out in candidates:
        if score < min_score and selected:
            continue
        title_key = str(out.get("title") or "").strip().lower()
        # Avoid duplicate sections with the same title unless it is untitled.
        if title_key and title_key != "untitled community report" and title_key in seen_titles:
            continue
        seen_titles.add(title_key)
        selected.append(out)
        if len(selected) >= max_reports:
            break

    return selected


# =============================================================================
# Prompts
# =============================================================================

def build_report_prompt(question: str) -> str:
    return f"""You are performing routing, not final answering.

Original question:
{question}

Task:
Retrieve relevant community reports and identify concise routing hints for code search.

Return only concise routing hints such as:
- function names
- class names
- method names
- file paths
- module/package names
- APIs
- config keys
- error names

Do not answer the question directly.
Avoid generic words and metadata fields.
"""


def build_code_prompt(
    question: str,
    hints: list[str],
    report_snippets: list[dict[str, Any]] | None = None,
) -> str:
    hints_text = "\n".join(f"- {h}" for h in hints[:15])
    report_text = ""
    if report_snippets:
        report_text = "\n\n".join(str(x.get("text") or x.get("content") or "") for x in report_snippets).strip()

    return f"""Answer the original question using the provided evidence.

Original question:
{question}

High-level source-code-derived community-report evidence:
{report_text if report_text else "None."}

Compact routing/search hints:
{hints_text}

Important instructions:
- The original question is the primary query.
- Community reports are source-code-derived high-level evidence.
- Use community reports to understand module-level intent, relationships, and likely code areas.
- Use retrieved source code as the authoritative evidence for concrete API behavior.
- If report evidence and source-code evidence conflict, follow the source code.
- Prefer code evidence that directly matches the exact terms in the original question.
- Prefer current/non-deprecated LangChain APIs when the evidence distinguishes old and new APIs.
- Do not replace `create_agent` with adjacent APIs such as `create_react_agent`, `initialize_agent`, or `create_self_ask_with_search_agent` unless the question explicitly asks for those APIs.
- Do not answer a different example than the one asked.
- If the retrieved evidence only partially supports the answer, give the supported part and avoid unsupported details.
- Do not invent exact model names, tool names, function names, argument values, or code examples that are not present in the retrieved evidence.
- If the question asks how to do something but the retrieved evidence lacks the exact example, describe the supported API pattern without fabricating a concrete example.
- If there is no relevant evidence at all, answer exactly: I don't know.
"""


# =============================================================================
# Two-stage generation
# =============================================================================

def _ragflow_call(prompt: str, chat_id: str, cfg: dict[str, Any]) -> tuple[str, Any, dict[str, Any], list[dict[str, Any]]]:
    """Single RAGFlow turn via the formal module. Returns raw_answer, ref, info, contexts."""
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
    # ---- Stage 1: report routing -------------------------------------------
    report_raw, report_ref, report_info, report_contexts = _ragflow_call(
        build_report_prompt(question), cfg["report_chat_id"], cfg
    )
    report_is_error = base.is_ragflow_error_answer(report_raw)

    hints = extract_hints(question, report_raw, report_contexts, max_hints=cfg["max_hints"])
    if not cfg["no_question_anchors"]:
        hints = add_question_aware_anchors(question, hints)
    hints = hints[: cfg["max_hints"]]

    # Select compact report evidence. In CodeOnly ablation mode these snippets
    # are stored for diagnostics but are NOT put into the final prompt/context.
    report_snippets = select_report_snippets(
        question=question,
        report_contexts=report_contexts,
        max_reports=cfg.get("max_report_snippets", 3),
        max_chars_per_report=cfg.get("max_report_chars", 1200),
        min_score=cfg.get("min_report_score", 0.0),
    )

    prompt_report_snippets = report_snippets if cfg.get("mixed_context", False) else None

    # ---- Stage 2: code evidence + final answer -----------------------------
    code_prompt = build_code_prompt(question, hints, prompt_report_snippets)
    code_raw, code_ref, code_info, code_contexts = _ragflow_call(
        code_prompt, cfg["code_chat_id"], cfg
    )
    code_is_error = base.is_ragflow_error_answer(code_raw)

    # Evaluation-facing answer: unified v3 cleaner + error->"I don't know.".
    clean_answer = base.normalize_ragflow_answer_for_eval(code_raw)

    code_ctx_tok = sum(int(c.get("token_count") or 0) for c in code_contexts if isinstance(c, dict))
    report_ctx_tok = sum(int(c.get("token_count") or 0) for c in report_snippets if isinstance(c, dict))
    ctx_tok = code_ctx_tok + (report_ctx_tok if cfg.get("mixed_context", False) else 0)

    return {
        "clean_answer": clean_answer,
        # Stage 2 (final) raw + contexts.
        "code_raw": code_raw,
        "code_ref": code_ref,
        "code_info": code_info,
        "code_contexts": code_contexts,
        "code_is_error": code_is_error,
        "code_ctx_tok": code_ctx_tok,
        "ctx_tok": ctx_tok,
        # Stage 1 (router) diagnostics and optional mixed evidence.
        "report_raw": report_raw,
        "report_ref": report_ref,
        "report_info": report_info,
        "report_contexts": report_contexts,
        "report_snippets": report_snippets,
        "report_ctx_tok": report_ctx_tok,
        "report_is_error": report_is_error,
        "hints": hints,
        "code_prompt": code_prompt,
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
        # IMPORTANT: do not treat a RAGFlow-error item as complete; retry it.
        and not item.get("raw_ragflow_answer_is_error")
    )
    facts_ok = (
        item.get("fact_model") == fact_model
        and "format_answer_atomic_facts" in item
        and item.get("fact_status") == "success"
    )
    return bool(answer_ok and facts_ok)


def process_item(
    item: dict[str, Any],
    question: str,
    cfg: dict[str, Any],
    fact_client: Any,
) -> None:
    gold_answer = base.extract_gold_answer(item)
    gold_atomic_facts = base.extract_gold_atomic_facts(item)

    common_meta = {
        "system_name": cfg["system_name"],
        "generator_provider": "ragflow",
        "generator_model": cfg["ragflow_model"],
        "method": cfg["method"],
        "method_name": cfg["method_name"],
        "fact_provider": "su8",
        "fact_model": cfg["fact_model"],
        "fact_base_url": cfg["fact_base_url"],
        "answer_cleaner": getattr(base, "ANSWER_CLEANER_VERSION", "clean_rag_answer_v3"),
    }

    gen = run_two_stage(question, cfg)

    # Final eval answer is already cleaned (normalize_ragflow_answer_for_eval ->
    # clean_rag_answer_for_eval). It is idempotent, but keep the invariant explicit.
    clean_answer = base.normalize_answer_text(
        base.clean_rag_answer_for_eval(gen["clean_answer"] or "")
    )

    # ---- Inline atomic-fact decomposition (the whole point of this fix) -----
    facts = base.call_su8_decompose_facts(
        client=fact_client,
        model=cfg["fact_model"],
        answer=clean_answer,
    )

    code_contexts = gen["code_contexts"] or []
    report_snippets = gen.get("report_snippets") or []

    if cfg.get("mixed_context", False):
        final_contexts = report_snippets + code_contexts
    else:
        final_contexts = code_contexts

    ctx_tok = sum(
        int(c.get("token_count") or 0)
        for c in final_contexts
        if isinstance(c, dict)
    )

    item.update(common_meta)
    item["gold_answer"] = gold_answer
    item["gold_atomic_facts"] = gold_atomic_facts

    # Evaluation-compatible fields.
    item["format_answer"] = clean_answer
    item["format_answer_atomic_facts"] = facts
    item["generated_answer"] = clean_answer
    item["generated_atomic_facts"] = facts

    # Retrieved contexts.
    # CodeOnly ablation: Stage 2 code evidence only.
    # Mixed main method: selected report snippets + Stage 2 code evidence.
    item["retrieved_contexts"] = final_contexts
    item["retrieved_context_token_count"] = ctx_tok
    item["ctx_tok"] = ctx_tok
    item["mixed_context_enabled"] = bool(cfg.get("mixed_context", False))

    # RAGFlow raw + reference (Stage 2 is the answer-producing stage).
    item["raw_ragflow_answer"] = gen["code_raw"]
    item["raw_ragflow_answer_is_error"] = bool(gen["code_is_error"])
    item["raw_ragflow_reference"] = gen["code_ref"]
    item["ragflow_reference"] = gen["code_ref"]
    item["ragflow_reference_contexts"] = final_contexts
    item["ragflow_reference_context_count"] = len(final_contexts)
    item["ragflow_reference_token_count"] = ctx_tok
    item["raw_generation_info"] = gen["code_info"]

    # Two-stage diagnostics.
    item["twostage_report_answer"] = gen["report_raw"]
    item["twostage_report_is_error"] = bool(gen["report_is_error"])
    item["twostage_report_contexts"] = gen["report_contexts"]
    item["twostage_report_context_count"] = len(gen["report_contexts"])
    item["twostage_selected_report_snippets"] = report_snippets
    item["twostage_selected_report_snippet_count"] = len(report_snippets)
    item["twostage_selected_report_token_count"] = sum(
        int(c.get("token_count") or 0)
        for c in report_snippets
        if isinstance(c, dict)
    )
    item["twostage_code_contexts"] = code_contexts
    item["twostage_code_context_count"] = len(code_contexts)
    item["twostage_code_context_token_count"] = sum(
        int(c.get("token_count") or 0)
        for c in code_contexts
        if isinstance(c, dict)
    )
    item["twostage_raw_report_reference"] = gen["report_ref"]
    item["twostage_report_info"] = gen["report_info"]
    item["twostage_routing_hints"] = gen["hints"]
    item["twostage_routing_hint_count"] = len(gen["hints"])
    item["twostage_code_prompt"] = gen["code_prompt"]
    item["twostage_report_chat_id"] = cfg["report_chat_id"]
    item["twostage_code_chat_id"] = cfg["code_chat_id"]

    # Bookkeeping.
    item["answer_word_count"] = base.word_count(clean_answer)
    item["answer_token_count"] = base.count_tokens(clean_answer)
    item["atomic_fact_count"] = len(facts)

    # A clean "I don't know." from an error is still a deterministic answer, but
    # we flag the error so resume retries it and analysis can exclude it.
    item["answer_status"] = "success" if not gen["code_is_error"] else "ragflow_error"
    item["fact_status"] = "success"


# =============================================================================
# File processing (resume + checkpoint)
# =============================================================================

def process_file(input_path: Path, output_path: Path, cfg: dict[str, Any], fact_client: Any) -> None:
    if output_path.exists() and not cfg["overwrite"]:
        print(f"[Resume] Loading existing output: {output_path}", flush=True)
        data = base.load_json(output_path)
    else:
        print(f"[Read] {input_path}", flush=True)
        data = base.load_json(input_path)

    items = data if isinstance(data, list) else data.get("items", data)

    processed = 0
    success = 0
    failed = 0
    skipped = 0

    total = len(items) if hasattr(items, "__len__") else 0
    print(f"[Process] {input_path.name} -> {output_path.name}, total={total}", flush=True)
    print(f"  system={cfg['system_name']} method={cfg['method_name']} fact_model={cfg['fact_model']}", flush=True)

    for i, item in enumerate(items):
        if not isinstance(item, dict):
            continue

        if cfg["limit"] is not None and processed >= cfg["limit"]:
            break

        if is_complete_item(item, cfg["method"], cfg["method_name"], cfg["fact_model"]) and not cfg["overwrite"]:
            skipped += 1
            continue

        question = get_question(item)
        print(f"[{i + 1}/{total}] {question[:100]}", flush=True)

        if not question:
            item["format_answer"] = ""
            item["format_answer_atomic_facts"] = []
            item["answer_status"] = "skipped_missing_question"
            item["fact_status"] = "skipped_missing_question"
            processed += 1
            continue

        try:
            process_item(item, question, cfg, fact_client)
            print(
                f"  -> facts={item.get('atomic_fact_count')} "
                f"ctx={item.get('ragflow_reference_context_count')} "
                f"code_ctx={item.get('twostage_code_context_count')} "
                f"report_snip={item.get('twostage_selected_report_snippet_count')} "
                f"hints={item.get('twostage_routing_hint_count')} "
                f"status={item.get('answer_status')}",
                flush=True,
            )
            if item.get("answer_status") == "success":
                success += 1
            else:
                failed += 1
        except Exception as e:  # noqa: BLE001
            item["format_answer"] = ""
            item["format_answer_atomic_facts"] = []
            item["generated_answer"] = ""
            item["generated_atomic_facts"] = []
            item["retrieved_contexts"] = []
            item["retrieved_context_token_count"] = None
            item["ctx_tok"] = None
            item["raw_ragflow_answer"] = None
            item["raw_ragflow_answer_is_error"] = True
            item["answer_status"] = "failed"
            item["fact_status"] = "failed"
            item["generation_error"] = str(e)
            failed += 1
            print(f"  [Failed] item {i}: {e}", flush=True)

        processed += 1

        if cfg["save_every"] > 0 and processed % cfg["save_every"] == 0:
            _save(output_path, data)
            print(f"  [Checkpoint] {output_path}", flush=True)

        if cfg["sleep"] > 0:
            time.sleep(cfg["sleep"])

    _save(output_path, data)
    print(f"[Done] {input_path.name}")
    print(f"  output: {output_path}")
    print(f"  processed={processed} success={success} failed={failed} skipped_existing={skipped}")
    print()


def _save(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


# =============================================================================
# Summary
# =============================================================================

def count_answer_status(items: list[dict[str, Any]]) -> dict[str, Any]:
    answered = idk = empty = errors = with_code = with_report = facts_nonempty = 0

    for it in items:
        ans = str(it.get("format_answer") or "").strip()
        if not ans:
            empty += 1
        elif ans.lower().startswith("i don't know"):
            idk += 1
        else:
            answered += 1

        if it.get("raw_ragflow_answer_is_error"):
            errors += 1
        if it.get("retrieved_contexts"):
            with_code += 1
        if it.get("twostage_report_contexts"):
            with_report += 1
        if it.get("format_answer_atomic_facts"):
            facts_nonempty += 1

    n = len(items)
    return {
        "N": n,
        "answered": answered,
        "answer_rate": answered / n if n else None,
        "idk": idk,
        "idk_rate": idk / n if n else None,
        "empty": empty,
        "ragflow_errors": errors,
        "with_code_contexts": with_code,
        "with_report_contexts": with_report,
        "facts_nonempty": facts_nonempty,
        "facts_nonempty_rate": facts_nonempty / n if n else None,
    }


# =============================================================================
# Optional metrics
# =============================================================================

def run_metrics(generated_path: Path, args: argparse.Namespace) -> None:
    script = Path(args.evaluate_all_metrics_script)
    if not script.exists():
        print(f"[Metric] script not found: {script}; skipping.", flush=True)
        return

    metrics_path = generated_path.with_name(generated_path.stem + "_all_metrics.json")
    cmd = [
        sys.executable, str(script),
        "--input", str(generated_path),
        "--output", str(metrics_path),
        "--method", "ragflow",
        "--method_name", args.method_name or "TwoStage-Int-RAG",
        "--su8_model", args.fact_model,
        # Belt-and-suspenders: facts are already inline, but enable fallback so a
        # rare missing-fact item is still recovered (it now propagates to ALL metrics).
        "--fallback_decompose_missing_facts",
    ]
    if args.metric_disable_similarity:
        cmd.append("--disable_similarity")
    if args.direct_summary_path:
        cmd.extend(["--direct_summary_path", args.direct_summary_path])

    print(f"[Metric] Running: {' '.join(cmd)}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        print(f"[Metric] failed: {e}", flush=True)


# =============================================================================
# Main
# =============================================================================

def main() -> None:
    ap = argparse.ArgumentParser(description="TwoStage-Int-RAG / TwoStage-Mixed generation with inline atomic-fact decomposition.")

    ap.add_argument("--input", default="langchain_qa_gold_final_atomic.json")
    ap.add_argument("--output_dir", default="outputs/langchain/gpt55/twostage_int_rag_v4")
    ap.add_argument("--output", default=None, help="Explicit output path. Overrides --output_dir naming.")
    ap.add_argument("--system_name", default=None, help="Default: guessed from input filename.")
    ap.add_argument("--method", default="TwoStage-Int-RAG", help="method tag stored per item.")
    ap.add_argument("--method_name", default="TwoStage-Int-RAG", help="display name for tables/summary.")

    # RAGFlow
    ap.add_argument("--report_chat_id", default=os.environ.get("LANGCHAIN_REPORT_CHAT_ID", ""))
    ap.add_argument(
        "--code_chat_id",
        default=os.environ.get("LANGCHAIN_CODERAG_CHAT_ID", os.environ.get("LANGCHAIN_CODE_CHAT_ID", "")),
    )
    ap.add_argument("--ragflow_base_url", default=os.environ.get("RAGFLOW_BASE_URL", ""))
    ap.add_argument("--ragflow_api_key", default=os.environ.get("RAGFLOW_API_KEY", ""))
    ap.add_argument("--ragflow_model", default=os.environ.get("RAGFLOW_MODEL", "ragflow-model"))
    ap.add_argument("--ragflow_endpoint_style", default="chats_openai", choices=["auto", "openai", "chats_openai"])

    # Hints
    ap.add_argument("--max_hints", type=int, default=15)
    ap.add_argument("--no_question_anchors", action="store_true",
                    help="Disable the LangChain-specific question-aware anchors.")

    # Mixed report+code context
    ap.add_argument("--mixed_context", action="store_true",
                    help="Use selected community-report snippets together with code contexts as final generation/evaluation contexts.")
    ap.add_argument("--max_report_snippets", type=int, default=3,
                    help="Maximum selected community-report snippets to add in --mixed_context mode.")
    ap.add_argument("--max_report_chars", type=int, default=1200,
                    help="Maximum characters kept per selected community-report snippet.")
    ap.add_argument("--min_report_score", type=float, default=0.0,
                    help="Minimum report snippet score after at least one snippet has been selected.")

    # Facts (SU8)
    ap.add_argument("--fact_model", default="gpt-5.5")
    ap.add_argument("--fact_base_url", default="https://www.su8.codes/codex/v1")

    # Control
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=5)
    ap.add_argument("--sleep", type=float, default=0.2)
    ap.add_argument("--overwrite", action="store_true")

    # Module + metrics
    ap.add_argument("--base_module", default=None,
                    help="Import name of the formal multi-backend generator module.")
    ap.add_argument("--run_metrics", action="store_true")
    ap.add_argument("--evaluate_all_metrics_script", default="evaluate_all_metrics_su8.py")
    ap.add_argument("--metric_disable_similarity", action="store_true")
    ap.add_argument("--direct_summary_path", default=None)

    args = ap.parse_args()

    # Resolve and bind the formal module globally.
    global base
    base = resolve_base(args.base_module)

    # Validation.
    if not args.report_chat_id:
        raise SystemExit("Missing --report_chat_id or LANGCHAIN_REPORT_CHAT_ID")
    if not args.code_chat_id:
        raise SystemExit("Missing --code_chat_id or LANGCHAIN_CODERAG_CHAT_ID")
    if not args.ragflow_base_url:
        raise SystemExit("Missing --ragflow_base_url or RAGFLOW_BASE_URL")
    if not args.ragflow_api_key:
        raise SystemExit("Missing --ragflow_api_key or RAGFLOW_API_KEY")

    input_path = Path(args.input)
    if not input_path.exists():
        raise SystemExit(f"Input not found: {input_path}")

    system_name = args.system_name or base.guess_system_name(input_path)

    # SU8 client for fact decomposition.
    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install openai") from exc

    su8_api_key = base.get_su8_api_key()
    fact_client = OpenAI(base_url=args.fact_base_url, api_key=su8_api_key, timeout=120)

    # Output path.
    if args.output:
        out_path = Path(args.output)
    else:
        out_dir = Path(args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out_name = (
            input_path.stem
            + f"_{base.sanitize_name(args.method_name)}"
            + f"_ragflow_{base.sanitize_name(args.ragflow_model)}"
            + f"_facts_su8_{base.sanitize_name(args.fact_model)}.json"
        )
        out_path = out_dir / out_name

    cfg = {
        "system_name": system_name,
        "method": args.method,
        "method_name": args.method_name,
        "base_url": args.ragflow_base_url,
        "api_key": args.ragflow_api_key,
        "ragflow_model": args.ragflow_model,
        "endpoint_style": args.ragflow_endpoint_style,
        "report_chat_id": args.report_chat_id,
        "code_chat_id": args.code_chat_id,
        "max_hints": args.max_hints,
        "no_question_anchors": args.no_question_anchors,
        "mixed_context": args.mixed_context,
        "max_report_snippets": args.max_report_snippets,
        "max_report_chars": args.max_report_chars,
        "min_report_score": args.min_report_score,
        "fact_model": args.fact_model,
        "fact_base_url": args.fact_base_url,
        "limit": args.limit,
        "save_every": args.save_every,
        "sleep": args.sleep,
        "overwrite": args.overwrite,
    }

    print()
    print("========== TwoStage-Int-RAG Config ==========")
    print("Base module:", base.__name__)
    print("Input:", input_path)
    print("Output:", out_path)
    print("System:", system_name)
    print("Method:", args.method, "/", args.method_name)
    print("RAGFlow base URL:", args.ragflow_base_url)
    print("Report chat ID:", args.report_chat_id)
    print("Code chat ID:", args.code_chat_id)
    print("Endpoint style:", args.ragflow_endpoint_style)
    print("Max hints:", args.max_hints, "| question anchors:", not args.no_question_anchors)
    print("Mixed context:", args.mixed_context,
          "| max report snippets:", args.max_report_snippets,
          "| max report chars:", args.max_report_chars)
    print("Fact model:", args.fact_model)
    print("Answer cleaner:", getattr(base, "ANSWER_CLEANER_VERSION", "clean_rag_answer_v3"))
    print("=============================================")
    print()

    process_file(input_path, out_path, cfg, fact_client)

    # Summary.
    data = base.load_json(out_path)
    items = data if isinstance(data, list) else data.get("items", data)
    summary = count_answer_status([x for x in items if isinstance(x, dict)])
    summary_fp = out_path.with_name(out_path.stem + "_genstats.json")
    _save(summary_fp, summary)

    print("[GenStats]")
    for k, v in summary.items():
        print(f"  {k}: {round(v, 4) if isinstance(v, float) else v}")
    print("[genstats]", summary_fp, flush=True)

    if args.run_metrics:
        run_metrics(out_path, args)


if __name__ == "__main__":
    main()