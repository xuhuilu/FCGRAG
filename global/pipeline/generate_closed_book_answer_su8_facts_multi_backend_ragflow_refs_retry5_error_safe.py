#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generate answers for QA gold datasets under multiple knowledge-access methods,
then decompose generated answers into atomic facts using SU8 GPT-5.5.

MERGE NOTE (v3 cleaner integration):
  - All evaluation-facing answers (format_answer) now pass through the single
    deterministic cleaner in clean_rag_answer.py, for EVERY method
    (Direct / MaxCtx / Doc-RAG / Code-RAG / Report-RAG / Int-RAG / TwoStage).
  - The old strip_ragflow_citations() contained a regex that corrupted code
    answers (e.g. `argv[0]` -> `argv`). It is now DEPRECATED for answer
    cleaning and kept only for error-string detection.
  - Each item records answer_cleaner="clean_rag_answer_v3" for provenance.
  - clean_rag_answer.py must sit in the same directory as this script.

Supported answer-generation methods:
  1. direct
     - No external context.
     - Uses either SU8 OpenAI-compatible API or local Ollama.

  2. maxctx
     - Loads a full context file, truncates it by token budget, and injects it
       into the prompt.
     - Uses either SU8 OpenAI-compatible API or local Ollama.

  3. ragflow
     - Calls a RAGFlow OpenAI-compatible chat endpoint.
     - Extracts answer + reference.
     - Standardizes retrieved contexts for downstream metrics.

Fact decomposition:
  - Always done by SU8 GPT-5.5 by default.
  - This keeps atomic-fact decomposition consistent across generators and methods.

Outputs:
  Per-item JSON:
    - raw_ragflow_answer, for RAGFlow runs, preserving [ID:n] citations/errors for debugging
    - raw_ragflow_answer_is_error, whether raw RAGFlow output was a server/error string
    - format_answer, cleaned by the unified v3 cleaner, used for evaluation
    - format_answer_atomic_facts
    - retrieved_contexts
    - retrieved_context_token_count
    - generation metadata (including answer_cleaner version)
    - optional metric fields from external scripts

  Method-level summary JSON:
    - System / N / Generator / Method / CtxTok / Ctx.R / Faith. / Ans.R. /
      Ans.C. / RA-FC / DeltaRA (if a direct summary is provided)

Example usage:

    python generate_closed_book_answer_su8_facts_multi_backend.py \
  --system_name LangChain \
  --provider su8 \
  --model gpt-5.5 \
  --method direct \
  --method_name Direct \
  --inputs langchain_qa_gold_final_atomic.json \
  --output_dir outputs/langchain/gpt55/direct \
  --run_metrics

  python generate_closed_book_answer_su8_facts_multi_backend.py \
  --system_name LangChain \
  --provider su8 \
  --model gpt-5.5 \
  --method ragflow \
  --method_name Doc-RAG \
  --ragflow_base_url "http://your-ragflow-host/api/v1" \
  --ragflow_api_key "$RAGFLOW_API_KEY" \
  --ragflow_chat_id "$LANGCHAIN_DOCRAG_CHAT_ID" \
  --inputs langchain_qa_gold_final_atomic.json \
  --output_dir outputs/langchain/gpt55/doc_rag \
  --run_metrics

  python generate_closed_book_answer_su8_facts_multi_backend.py \
  --system_name LangChain \
  --provider su8 \
  --model gpt-5.5 \
  --method maxctx \
  --method_name Int.-MaxCtx \
  --context_file langchain.graphml \
  --max_context_tokens 30000 \
  --inputs langchain_qa_gold_final_atomic.json \
  --output_dir outputs/langchain/gpt55/int_maxctx \
  --run_metrics
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

# =========================
# Unified v3 answer cleaner (REQUIRED)
# =========================
# All methods' format_answer must pass through this single cleaner so that
# answer cleaning is uniform and auditable across the whole experiment.
# Hard-fail if missing: silently falling back to the old buggy regexes would
# corrupt code answers such as `argv[0]`.
try:
    from clean_rag_answer import clean_rag_answer_for_eval
except ImportError as exc:
    raise SystemExit(
        "Missing clean_rag_answer.py (unified v3 answer cleaner). "
        "Place clean_rag_answer.py in the same directory as this script."
    ) from exc

ANSWER_CLEANER_VERSION = "clean_rag_answer_v3"


# =========================
# Defaults
# =========================

DEFAULT_INPUTS = [
    "ollama_qa_gold_final_atomic.json",
    "openclaw_qa_gold_final_atomic.json",
    "langchain_qa_gold_final_atomic.json",
]

DEFAULT_PROVIDER = "su8"
DEFAULT_SU8_MODEL = "gpt-5.5"
DEFAULT_SU8_BASE_URL = "https://www.su8.codes/codex/v1"

DEFAULT_OLLAMA_MODEL = "qwen3-coder"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

DEFAULT_FACT_PROVIDER = "su8"
DEFAULT_FACT_MODEL = "gpt-5.5"
DEFAULT_FACT_BASE_URL = DEFAULT_SU8_BASE_URL

DEFAULT_METHOD = "direct"

DEFAULT_RAGFLOW_BASE_URL = os.environ.get("RAGFLOW_BASE_URL", "")
DEFAULT_RAGFLOW_API_KEY = os.environ.get("RAGFLOW_API_KEY", "")
DEFAULT_RAGFLOW_CHAT_ID = os.environ.get("RAGFLOW_CHAT_ID", "")
DEFAULT_RAGFLOW_MODEL = os.environ.get("RAGFLOW_MODEL", "ragflow-model")

# Local Ollama answer-generation models all use this fixed configuration.
LLM_FIXED_PARAMS = {
    "temperature": 0,
    "top_p": 0.9,
    "num_predict": 300,
    "num_ctx": 4096,
}


# =========================
# Prompts
# =========================

ANSWER_SYSTEM_PROMPT = """You are answering closed-book questions about software systems and developer documentation.

You must answer using only your internal model knowledge. You are NOT given the source documentation, evidence, source URL, or document title.

Answering rules:
- Answer only what the question asks.
- Prefer 1-2 concise sentences.
- Do not add examples, background, alternatives, or implementation details unless the question explicitly asks for them.
- Do not invent commands, ports, endpoints, key bindings, protocol fields, default values, configuration keys, or code snippets.
- For questions asking for exact commands, API paths, key bindings, ports, default values, configuration keys, protocol frames, JSON fields, or field names, answer only if you are certain from your internal knowledge.
- If you are not certain, answer exactly: I don't know.
- If the question mentions evidence, documentation, "shown", "according to", or "provided", do not refer to evidence. Since no evidence is provided, answer only if you know the underlying fact; otherwise answer exactly: I don't know.
- Do not say "the provided evidence", "the evidence", or "the documentation" in your answer.
- Do not speculate.
- Do not cite or mention source documents.
- Do not mention that you are answering closed-book.

Return JSON only with this exact schema:
{
  "answer": "..."
}
"""


CONTEXT_ANSWER_SYSTEM_PROMPT = """You answer questions about software systems and developer documentation.

You are given retrieved or provided contexts.

Answering rules:
- Use only the provided contexts.
- Answer only what the question asks.
- Prefer 1-2 concise sentences.
- Do not add examples, background, alternatives, or implementation details unless the question explicitly asks for them.
- Do not invent commands, ports, endpoints, key bindings, protocol fields, default values, configuration keys, or code snippets unless they are explicitly supported by the provided contexts.
- If the provided contexts do not contain enough information, answer exactly: I don't know.
- Do not say "the provided context", "the evidence", or "the documentation" in your answer.
- Do not speculate.
- Do not cite or mention source documents.
- Do not mention that you are using retrieval or context.

Return JSON only with this exact schema:
{
  "answer": "..."
}
"""


FACT_SYSTEM_PROMPT = """You decompose a given answer into independent atomic factual statements.

Rules:
- Decompose ONLY the given answer.
- Do not use outside knowledge.
- Do not add facts that are not stated in the answer.
- If the answer is exactly "I don't know.", return an empty atomic_facts list.
- Each atomic fact must be independently understandable.
- Preserve exact technical terms, commands, API paths, configuration keys, parameters, model names, numbers, ports, endpoints, and code identifiers.
- Split conjunctions and enumerations when doing so makes each fact independently checkable.
- Do not over-split simple commands or single factual answers.
- Return at most 6 atomic facts.

Return JSON only with this exact schema:
{
  "atomic_facts": ["...", "..."]
}
"""


# =========================
# General helpers
# =========================

def get_su8_api_key() -> str:
    api_key = (
        os.environ.get("SU8_API_KEY")
        or os.environ.get("CODEX_API_KEY")
        or os.environ.get("OPENAI_API_KEY")
    )
    if not api_key:
        raise SystemExit(
            "Missing SU8 API key. Please run:\n"
            "  export SU8_API_KEY='your_key'\n"
            "or:\n"
            "  export CODEX_API_KEY='your_key'"
        )
    return api_key


def sanitize_name(name: str) -> str:
    name = str(name or "").strip()
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", name)
    name = name.replace(":", "_")
    return name.strip("_") or "unknown"


def strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = strip_code_fence(text)
    try:
        return json.loads(text)
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:500]}")
    return json.loads(match.group(0))


def try_extract_answer_from_text(text: str) -> str:
    """
    For RAGFlow answers, content may be plain text rather than JSON.
    If it is JSON and contains "answer", use it; otherwise use raw text.
    """
    text = (text or "").strip()
    if not text:
        return ""

    try:
        parsed = extract_json_object(text)
        if isinstance(parsed, dict) and "answer" in parsed:
            return normalize_answer_text(str(parsed.get("answer", "")))
    except Exception:
        pass

    return normalize_answer_text(text)


def normalize_answer_text(answer: str) -> str:
    answer = str(answer or "").strip()
    if is_unknown_answer(answer):
        return "I don't know."
    return answer


def strip_ragflow_citations(text: str) -> str:
    """
    DEPRECATED FOR ANSWER CLEANING — kept ONLY as a preprocessing step inside
    is_ragflow_error_answer(), where it strips citation noise from error
    strings before prefix matching.

    DO NOT use this function to clean evaluation answers: the bare numeric
    pattern `\\[\\d+\\]` corrupts code answers (e.g. `argv[0]` -> `argv`).
    All evaluation-facing cleaning goes through
    clean_rag_answer.clean_rag_answer_for_eval (v3), which preserves
    identifier-attached brackets.
    """
    text = str(text or "")

    # RAGFlow citation markers: [ID:1], [ID: 2], [ID:1, 3]
    text = re.sub(r"\s*\[ID:\s*[\d,\s]+\]", "", text)

    # Simple numeric citations if RAGFlow/model emits them: [1], [2]
    # (Unsafe for code answers; acceptable here because this function is only
    # used for error-string detection, never for evaluation answers.)
    text = re.sub(r"\s*\[\d+\]", "", text)

    # Collapse excessive whitespace while preserving a clean answer string.
    text = re.sub(r"\s+", " ", text).strip()
    return text


def is_ragflow_error_answer(answer: str) -> bool:
    """
    Detect real RAGFlow/server error strings.

    Be conservative. Do not treat normal technical words such as
    timeout, timeoutSeconds, timeoutSec, timeouts, rate limits, max_tokens,
    or base_url as errors unless the answer clearly has an error format.
    """
    s = str(answer or "").strip()
    if not s:
        return False

    cleaned = strip_ragflow_citations(s).strip()
    low = cleaned.lower()

    # Strong error prefixes only.
    if low.startswith("**error**"):
        return True
    if low.startswith("error:"):
        return True
    if low.startswith("[error]"):
        return True

    # Explicit machine-style error codes.
    if re.search(r"\b(connection_error|http_error|timeout_error|ragflow_call_failed)\b", low):
        return True

    # Natural-language error messages only when they are the beginning of
    # the returned content, not merely technical terms inside a valid answer.
    prefix_patterns = [
        r"^connection error\b",
        r"^bad gateway\b",
        r"^gateway timeout\b",
        r"^gateway time-out\b",
        r"^request timeout\b",
        r"^request timed out\b",
        r"^server error\b",
        r"^internal server error\b",
        r"^service unavailable\b",
        r"^too many requests\b",
        r"^model unavailable\b",
        r"^upstream error\b",
    ]

    return any(re.search(pattern, low) for pattern in prefix_patterns)


def normalize_ragflow_answer_for_eval(answer: str) -> str:
    """
    Convert raw RAGFlow answer into evaluation-ready answer text.

    v3: cleaning is delegated to the unified cleaner
    (clean_rag_answer_for_eval), which removes citation markers WITHOUT
    corrupting identifier-attached brackets, strips grounding prefaces in a
    loop, removes inline/trailing meta phrases, and normalizes abstention.

    If the raw answer is actually a RAGFlow/server error string, return
    exactly "I don't know." so downstream atomic facts and metrics are not
    polluted by error messages.
    """
    if is_ragflow_error_answer(answer):
        return "I don't know."

    cleaned = clean_rag_answer_for_eval(answer)

    if is_ragflow_error_answer(cleaned):
        return "I don't know."

    return normalize_answer_text(cleaned)


def normalize_facts(raw: Any, max_facts: int = 6) -> list[str]:
    if raw is None or not isinstance(raw, list):
        return []

    facts: list[str] = []
    seen: set[str] = set()

    for x in raw:
        if not isinstance(x, str):
            continue

        fact = x.strip()
        fact = re.sub(r"^\s*[-*•]\s*", "", fact)
        fact = re.sub(r"\s+", " ", fact).strip()

        if not fact:
            continue

        key = fact.lower()
        if key in seen:
            continue

        seen.add(key)
        facts.append(fact)

        if len(facts) >= max_facts:
            break

    return facts


def is_unknown_answer(answer: str) -> bool:
    normalized = (answer or "").strip().lower()
    normalized = normalized.rstrip()
    return normalized in {
        "",
        "i don't know",
        "i don't know.",
        "i do not know",
        "i do not know.",
        "unknown",
        "unknown.",
        "not sure",
        "not sure.",
        "无法确定",
        "不知道",
        "不确定",
    }


def is_non_empty_answer(answer: str) -> bool:
    return not is_unknown_answer(answer)


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(re.findall(r"[A-Za-z0-9_./:=-]+", text))


_TOKENIZER = None


def get_tokenizer():
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    try:
        import tiktoken
        try:
            _TOKENIZER = tiktoken.get_encoding("cl100k_base")
        except Exception:
            _TOKENIZER = tiktoken.get_encoding("gpt2")
        return _TOKENIZER
    except Exception:
        _TOKENIZER = False
        return None


def count_tokens(text: str) -> int:
    if text is None:
        return 0

    text = str(text)
    tokenizer = get_tokenizer()
    if tokenizer:
        try:
            return len(tokenizer.encode(text))
        except Exception:
            pass

    # Fallback approximation.
    return len(re.findall(r"\S+", text))


def truncate_text_by_tokens(text: str, max_tokens: int | None) -> tuple[str, int, int, bool]:
    """
    Returns:
      truncated_text, kept_tokens, original_tokens, was_truncated
    """
    text = text or ""
    original_tokens = count_tokens(text)

    if max_tokens is None or max_tokens <= 0 or original_tokens <= max_tokens:
        return text, original_tokens, original_tokens, False

    tokenizer = get_tokenizer()
    if tokenizer:
        tokens = tokenizer.encode(text)
        truncated_tokens = tokens[:max_tokens]
        truncated_text = tokenizer.decode(truncated_tokens)
        return truncated_text + "\n\n[CONTENT TRUNCATED]", max_tokens, len(tokens), True

    # Fallback word-based truncation.
    words = re.findall(r"\S+", text)
    truncated_words = words[:max_tokens]
    return " ".join(truncated_words) + "\n\n[CONTENT TRUNCATED]", max_tokens, len(words), True


def load_raw_file_truncated(filepath: str | Path, max_tokens: int | None) -> tuple[str, int, int, bool]:
    path = Path(filepath)
    if not path.exists():
        raise FileNotFoundError(f"Context file not found: {path}")

    raw_content = path.read_text(encoding="utf-8", errors="ignore")
    return truncate_text_by_tokens(raw_content, max_tokens=max_tokens)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def iter_items(data: Any):
    if isinstance(data, list):
        for idx, item in enumerate(data):
            yield idx, item
    elif isinstance(data, dict):
        for key, item in data.items():
            yield key, item
    else:
        raise ValueError("Input JSON root must be a list or dict.")


def flatten_items(data: Any) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return [x for x in data if isinstance(x, dict)]

    if isinstance(data, dict):
        if isinstance(data.get("items"), list):
            return [x for x in data["items"] if isinstance(x, dict)]
        return [x for x in data.values() if isinstance(x, dict)]

    return []


def guess_system_name(input_path: Path) -> str:
    stem = input_path.stem.lower()
    if "ollama" in stem:
        return "Ollama"
    if "openclaw" in stem:
        return "OpenClaw"
    if "langchain" in stem:
        return "LangChain"
    return input_path.stem


# =========================
# Prompt builders
# =========================

def build_answer_user_prompt(question: str) -> str:
    return f"""Question:
{question}

Generate a concise closed-book answer."""


def build_context_user_prompt(question: str, contexts: list[dict[str, Any]]) -> str:
    context_text = "\n\n".join(
        f"[Context {i + 1}]\n{ctx.get('text', '')}"
        for i, ctx in enumerate(contexts)
    )

    return f"""Question:
{question}

Provided contexts:
{context_text}

Generate a concise answer using only the provided contexts."""


def build_fact_user_prompt(answer: str) -> str:
    return f"""Answer:
{answer}

Decompose the answer into atomic facts."""


def build_ragflow_user_prompt(question: str) -> str:
    return f"""Question:
{question}

Please answer concisely using the retrieved knowledge. If the retrieved knowledge does not contain enough information, answer exactly: I don't know."""


# =========================
# Response schemas
# =========================

def answer_response_format_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "answer_only",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "answer": {
                        "type": "string",
                        "description": "A concise answer to the question."
                    }
                },
                "required": ["answer"],
                "additionalProperties": False
            }
        }
    }


def fact_response_format_schema() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "atomic_fact_decomposition",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "atomic_facts": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": "Independent atomic factual statements decomposed from the answer."
                    }
                },
                "required": ["atomic_facts"],
                "additionalProperties": False
            }
        }
    }


def ollama_answer_json_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "answer": {"type": "string"},
        },
        "required": ["answer"],
        "additionalProperties": False,
    }


# =========================
# API retry helpers
# =========================

def retry_wait_seconds(error: Exception, attempt: int) -> int:
    msg = str(error)

    m = re.search(r"'retry_after':\s*(\d+)", msg)
    if not m:
        m = re.search(r'"retry_after":\s*(\d+)', msg)

    if m:
        return max(int(m.group(1)), 120)

    if "504" in msg or "Gateway time-out" in msg or "gateway_timeout" in msg:
        return 120

    if "Connection error" in msg:
        return min(60, 5 * attempt)

    return min(60, 2 ** attempt)


# =========================
# SU8 answer generation
# =========================

def parse_answer(data: dict[str, Any]) -> str:
    answer = str(data.get("answer", "")).strip()

    if not answer:
        raise ValueError(f"Empty answer returned: {data}")

    return normalize_answer_text(answer)


def parse_facts(data: dict[str, Any], answer: str) -> list[str]:
    if is_unknown_answer(answer):
        return []

    facts = normalize_facts(data.get("atomic_facts", []), max_facts=6)

    # If decomposition fails for a non-empty non-unknown answer, keep the whole
    # answer as one fact so downstream scoring does not silently ignore it.
    if not facts and answer.strip():
        facts = [answer.strip()]

    return facts


def call_su8_answer_json(
    client: Any,
    model: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 5,
) -> str:
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": 300,
                "response_format": answer_response_format_schema(),
            }

            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as e:
                if "json_schema" in str(e) or "response_format" in str(e):
                    kwargs["response_format"] = {"type": "json_object"}
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            content = response.choices[0].message.content or ""
            parsed = extract_json_object(content)
            return parse_answer(parsed)

        except Exception as e:
            last_error = e
            wait = retry_wait_seconds(e, attempt)
            print(
                f"  [Retry:SU8-answer] failed attempt {attempt}/{max_retries}: {e}. "
                f"Retrying in {wait}s...",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(f"SU8 answer generation failed after {max_retries} attempts: {last_error}")


def call_su8_answer_only(
    client: Any,
    model: str,
    question: str,
    max_retries: int = 5,
) -> str:
    return call_su8_answer_json(
        client=client,
        model=model,
        system_prompt=ANSWER_SYSTEM_PROMPT,
        user_prompt=build_answer_user_prompt(question),
        max_retries=max_retries,
    )


def call_su8_answer_with_context(
    client: Any,
    model: str,
    question: str,
    contexts: list[dict[str, Any]],
    max_retries: int = 5,
) -> str:
    return call_su8_answer_json(
        client=client,
        model=model,
        system_prompt=CONTEXT_ANSWER_SYSTEM_PROMPT,
        user_prompt=build_context_user_prompt(question, contexts),
        max_retries=max_retries,
    )


# =========================
# Ollama answer generation
# =========================

def post_ollama_chat(
    base_url: str,
    payload: dict[str, Any],
    timeout: int = 600,
) -> dict[str, Any]:
    import requests

    url = base_url.rstrip("/") + "/api/chat"
    response = requests.post(url, json=payload, timeout=timeout)
    response.raise_for_status()
    return response.json()


def call_ollama_answer_json(
    model: str,
    base_url: str,
    system_prompt: str,
    user_prompt: str,
    max_retries: int = 3,
) -> str:
    last_error: Exception | None = None

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for attempt in range(1, max_retries + 1):
        try:
            payload = {
                "model": model,
                "messages": messages,
                "stream": False,
                "options": LLM_FIXED_PARAMS,
                "format": ollama_answer_json_schema(),
            }

            try:
                response = post_ollama_chat(base_url, payload)
            except Exception as e:
                if "format" in str(e) or "schema" in str(e) or "400" in str(e):
                    payload["format"] = "json"
                    response = post_ollama_chat(base_url, payload)
                else:
                    raise

            content = response.get("message", {}).get("content", "") or ""
            parsed = extract_json_object(content)
            return parse_answer(parsed)

        except Exception as e:
            last_error = e
            wait = min(30, 2 ** attempt)
            print(
                f"  [Retry:Ollama-answer] failed attempt {attempt}/{max_retries}: {e}. "
                f"Retrying in {wait}s...",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(f"Ollama answer generation failed after {max_retries} attempts: {last_error}")


def call_ollama_answer_only(
    model: str,
    base_url: str,
    question: str,
    max_retries: int = 3,
) -> str:
    return call_ollama_answer_json(
        model=model,
        base_url=base_url,
        system_prompt=ANSWER_SYSTEM_PROMPT,
        user_prompt=build_answer_user_prompt(question),
        max_retries=max_retries,
    )


def call_ollama_answer_with_context(
    model: str,
    base_url: str,
    question: str,
    contexts: list[dict[str, Any]],
    max_retries: int = 3,
) -> str:
    return call_ollama_answer_json(
        model=model,
        base_url=base_url,
        system_prompt=CONTEXT_ANSWER_SYSTEM_PROMPT,
        user_prompt=build_context_user_prompt(question, contexts),
        max_retries=max_retries,
    )


# =========================
# SU8 atomic fact decomposition
# =========================

def call_su8_decompose_facts(
    client: Any,
    model: str,
    answer: str,
    max_retries: int = 5,
) -> list[str]:
    if is_unknown_answer(answer):
        return []

    user_prompt = build_fact_user_prompt(answer)
    last_error: Exception | None = None

    for attempt in range(1, max_retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": [
                    {"role": "system", "content": FACT_SYSTEM_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                "temperature": 0,
                "max_tokens": 300,
                "response_format": fact_response_format_schema(),
            }

            try:
                response = client.chat.completions.create(**kwargs)
            except Exception as e:
                if "json_schema" in str(e) or "response_format" in str(e):
                    kwargs["response_format"] = {"type": "json_object"}
                    response = client.chat.completions.create(**kwargs)
                else:
                    raise

            content = response.choices[0].message.content or ""
            parsed = extract_json_object(content)
            return parse_facts(parsed, answer=answer)

        except Exception as e:
            last_error = e
            wait = retry_wait_seconds(e, attempt)
            print(
                f"  [Retry:SU8-facts] failed attempt {attempt}/{max_retries}: {e}. "
                f"Retrying in {wait}s...",
                flush=True,
            )
            time.sleep(wait)

    raise RuntimeError(f"SU8 fact decomposition failed after {max_retries} attempts: {last_error}")


# =========================
# RAGFlow
# =========================

def build_ragflow_urls(
    base_url: str,
    chat_id: str,
    endpoint_style: str = "auto",
) -> list[str]:
    """
    Supports both:
      Newer style:
        /api/v1/openai/{chat_id}/chat/completions
      Older style:
        /api/v1/chats_openai/{chat_id}/chat/completions
    """
    base = base_url.rstrip("/")

    if base.endswith("/chat/completions"):
        return [base]

    if re.search(r"/(openai|chats_openai)/[^/]+$", base):
        return [base + "/chat/completions"]

    if endpoint_style == "openai":
        return [f"{base}/openai/{chat_id}/chat/completions"]

    if endpoint_style == "chats_openai":
        return [f"{base}/chats_openai/{chat_id}/chat/completions"]

    return [
        f"{base}/openai/{chat_id}/chat/completions",
        f"{base}/chats_openai/{chat_id}/chat/completions",
    ]


def run_ragflow_mode(
    question: str,
    ragflow_base_url: str,
    ragflow_api_key: str,
    ragflow_chat_id: str,
    ragflow_model: str = DEFAULT_RAGFLOW_MODEL,
    endpoint_style: str = "auto",
    timeout: int = 600,
    max_retries: int = 5,
) -> tuple[str, Any, dict[str, Any]]:
    """
    Call RAGFlow OpenAI-compatible chat endpoint and request reference contexts.

    Retry policy:
      1. HTTP errors, request exceptions, timeout, and non-JSON responses retry.
      2. HTTP 200 responses whose answer content is an error string such as
         "**ERROR**: CONNECTION_ERROR - Connection error. [ID:0]" also retry.
      3. After max_retries failures, return a synthetic raw error answer.
         Downstream normalize_ragflow_answer_for_eval() converts it to
         "I don't know.", while raw_ragflow_answer keeps the error text and
         raw_ragflow_answer_is_error is set to true.
    """
    import requests

    if not ragflow_base_url:
        raise ValueError("Missing RAGFlow base URL. Use --ragflow_base_url or export RAGFLOW_BASE_URL.")
    if not ragflow_api_key:
        raise ValueError("Missing RAGFlow API key. Use --ragflow_api_key or export RAGFLOW_API_KEY.")
    if not ragflow_chat_id:
        raise ValueError("Missing RAGFlow chat ID. Use --ragflow_chat_id or export RAGFLOW_CHAT_ID.")

    urls = build_ragflow_urls(
        base_url=ragflow_base_url,
        chat_id=ragflow_chat_id,
        endpoint_style=endpoint_style,
    )

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {ragflow_api_key}",
    }

    payload = {
        "model": ragflow_model,
        "messages": [
            {"role": "user", "content": build_ragflow_user_prompt(question)}
        ],
        "stream": False,
        "extra_body": {
            "reference": True
        }
    }

    last_error: Exception | None = None
    last_response_text = ""
    last_error_answer = ""
    last_reference = None
    last_raw_info: dict[str, Any] = {
        "endpoint_style": endpoint_style,
        "ragflow_model": ragflow_model,
        "status_code": None,
        "has_reference": False,
        "reference_type": None,
        "final_failure": False,
        "retry_count": 0,
    }

    for attempt in range(1, max_retries + 1):
        attempt_error: Exception | None = None

        for url in urls:
            try:
                response = requests.post(url, headers=headers, json=payload, timeout=timeout)
                last_response_text = response.text[:2000]

                # In auto mode, try the alternative endpoint when one style is missing.
                if response.status_code == 404 and endpoint_style == "auto":
                    attempt_error = RuntimeError(
                        f"RAGFlow endpoint returned 404 for {url}. Trying alternate endpoint."
                    )
                    continue

                response.raise_for_status()

                try:
                    result = response.json()
                except ValueError as e:
                    raise RuntimeError(
                        f"RAGFlow returned non-JSON response. status={response.status_code}, "
                        f"text={response.text[:1000]}"
                    ) from e

                answer = ""
                reference = None
                message = None

                if "choices" in result and result["choices"]:
                    message = result["choices"][0].get("message", {}) or {}
                    answer = message.get("content", "") or ""
                    reference = message.get("reference", None)
                else:
                    raise RuntimeError(
                        "RAGFlow response does not contain OpenAI-compatible choices. "
                        f"Raw response: {json.dumps(result, ensure_ascii=False)[:1500]}"
                    )

                # Some RAGFlow versions put reference at top level.
                if reference is None:
                    reference = result.get("reference", None)

                answer = try_extract_answer_from_text(answer)

                raw_info = {
                    "url": url,
                    "endpoint_style": endpoint_style,
                    "ragflow_model": ragflow_model,
                    "status_code": response.status_code,
                    "response_keys": list(result.keys()) if isinstance(result, dict) else [],
                    "choice_message_keys": list(message.keys()) if isinstance(message, dict) else [],
                    "has_reference": reference is not None,
                    "reference_type": type(reference).__name__,
                    "raw_response_id": result.get("id") if isinstance(result, dict) else None,
                    "retry_count": attempt - 1,
                    "final_failure": False,
                    "ragflow_answer_is_error": False,
                }

                # Critical: RAGFlow may return HTTP 200 with an error string as
                # message.content. Treat this as a failed attempt and retry.
                if is_ragflow_error_answer(answer):
                    last_error_answer = answer
                    last_reference = reference
                    last_raw_info = {
                        **raw_info,
                        "final_failure": False,
                        "ragflow_answer_is_error": True,
                        "error_answer_attempt": attempt,
                    }
                    attempt_error = RuntimeError(
                        f"RAGFlow returned error answer content: {answer}"
                    )
                    continue

                return answer, reference, raw_info

            except requests.exceptions.Timeout as e:
                last_error = e
                attempt_error = e
                last_error_answer = f"**ERROR**: TIMEOUT - request timed out after {timeout}s"
                last_raw_info = {
                    "url": url,
                    "endpoint_style": endpoint_style,
                    "ragflow_model": ragflow_model,
                    "status_code": 408,
                    "has_reference": False,
                    "reference_type": None,
                    "retry_count": attempt,
                    "final_failure": False,
                    "exception_type": type(e).__name__,
                    "exception": str(e),
                }
                continue

            except requests.exceptions.HTTPError as e:
                last_error = e
                attempt_error = e
                status_code = getattr(e.response, "status_code", None)
                body = getattr(e.response, "text", "")[:1000] if getattr(e, "response", None) is not None else ""
                last_error_answer = f"**ERROR**: HTTP_ERROR - status={status_code}; {body or str(e)}"
                last_raw_info = {
                    "url": url,
                    "endpoint_style": endpoint_style,
                    "ragflow_model": ragflow_model,
                    "status_code": status_code,
                    "has_reference": False,
                    "reference_type": None,
                    "retry_count": attempt,
                    "final_failure": False,
                    "exception_type": type(e).__name__,
                    "exception": str(e),
                    "response_preview": body,
                }
                continue

            except Exception as e:
                last_error = e
                attempt_error = e
                last_error_answer = f"**ERROR**: {type(e).__name__} - {str(e)}"
                last_raw_info = {
                    "url": url,
                    "endpoint_style": endpoint_style,
                    "ragflow_model": ragflow_model,
                    "status_code": None,
                    "has_reference": last_reference is not None,
                    "reference_type": type(last_reference).__name__ if last_reference is not None else None,
                    "retry_count": attempt,
                    "final_failure": False,
                    "exception_type": type(e).__name__,
                    "exception": str(e),
                    "last_response_preview": last_response_text[:500],
                }
                continue

        last_error = attempt_error or last_error
        if attempt < max_retries:
            wait = min(60, 2 ** attempt)
            print(
                f"  [Retry:RAGFlow] failed attempt {attempt}/{max_retries}: {last_error}. "
                f"Last response: {last_response_text[:300]}. Retrying in {wait}s...",
                flush=True,
            )
            time.sleep(wait)

    # After max_retries, do not raise. Return a raw error answer so the caller
    # can store it as raw_ragflow_answer while format_answer becomes I don't know.
    if not last_error_answer:
        last_error_answer = f"**ERROR**: RAGFLOW_CALL_FAILED - {last_error}"

    final_info = {
        **last_raw_info,
        "final_failure": True,
        "retry_count": max_retries,
        "max_retries": max_retries,
        "ragflow_answer_is_error": True,
        "last_error": str(last_error) if last_error else None,
        "last_response_preview": last_response_text[:500],
    }

    return last_error_answer, last_reference, final_info


def _first_non_empty(d: dict[str, Any], keys: list[str]) -> Any:
    for key in keys:
        value = d.get(key)
        if value is not None and value != "":
            return value
    return None


def _as_float_or_none(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def extract_contexts_from_ragflow_reference(reference: Any) -> list[dict[str, Any]]:
    """
    Standardize RAGFlow references into a stable list of retrieved contexts.

    Output schema:
      {
        "rank": 1,
        "source": "...",
        "doc_id": "...",
        "chunk_id": "...",
        "text": "...",
        "score": 0.87,
        "similarity": 0.87,
        "vector_similarity": 0.80,
        "term_similarity": 0.40,
        "token_count": 123,
        "raw": {...}
      }

    RAGFlow versions/configs vary. This parser handles common layouts:
      reference.chunks[]
      reference.references[]
      reference.documents[]
      result-level reference lists
      plain string contexts
    """
    contexts: list[dict[str, Any]] = []
    seen_text: set[str] = set()

    def add_candidate(obj: Any) -> None:
        if obj is None:
            return

        if isinstance(obj, str):
            text = obj.strip()
            if not text:
                return
            key = text[:500]
            if key in seen_text:
                return
            seen_text.add(key)
            contexts.append({
                "rank": len(contexts) + 1,
                "source": None,
                "doc_id": None,
                "chunk_id": None,
                "text": text,
                "score": None,
                "similarity": None,
                "vector_similarity": None,
                "term_similarity": None,
                "token_count": count_tokens(text),
                "raw": obj,
            })
            return

        if not isinstance(obj, dict):
            return

        text = _first_non_empty(
            obj,
            [
                "content",
                "text",
                "chunk",
                "chunk_text",
                "page_content",
                "document",
                "body",
                "content_with_weight",
            ],
        )
        if text is not None:
            text = str(text).strip()

        if not text:
            return

        key = text[:500]
        if key in seen_text:
            return
        seen_text.add(key)

        source = _first_non_empty(
            obj,
            [
                "doc_name",
                "document_name",
                "docnm_kwd",
                "source",
                "url",
                "file",
                "filename",
                "dataset_name",
                "kb_name",
            ],
        )
        doc_id = _first_non_empty(
            obj,
            [
                "doc_id",
                "document_id",
                "docid",
                "dataset_id",
                "kb_id",
            ],
        )
        chunk_id = _first_non_empty(obj, ["chunk_id", "chunkid", "id"])

        score = _first_non_empty(obj, ["score", "similarity", "rerank_score"])
        similarity = _first_non_empty(obj, ["similarity"])
        vector_similarity = _first_non_empty(obj, ["vector_similarity"])
        term_similarity = _first_non_empty(obj, ["term_similarity"])

        contexts.append({
            "rank": len(contexts) + 1,
            "source": source,
            "doc_id": doc_id,
            "chunk_id": chunk_id,
            "text": text,
            "score": _as_float_or_none(score),
            "similarity": _as_float_or_none(similarity),
            "vector_similarity": _as_float_or_none(vector_similarity),
            "term_similarity": _as_float_or_none(term_similarity),
            "token_count": count_tokens(text),
            "raw": obj,
        })

    def walk(obj: Any) -> None:
        if obj is None:
            return
        if isinstance(obj, str):
            add_candidate(obj)
            return
        if isinstance(obj, list):
            for x in obj:
                walk(x)
            return
        if isinstance(obj, dict):
            # Prefer explicit containers first to preserve retrieval order.
            for key in [
                "chunks",
                "references",
                "reference",
                "documents",
                "docs",
                "results",
                "items",
                "data",
                "contexts",
            ]:
                value = obj.get(key)
                if isinstance(value, (list, dict)):
                    walk(value)

            # Then try the dict itself as a chunk-like object.
            add_candidate(obj)

    walk(reference)

    for i, ctx in enumerate(contexts):
        ctx["rank"] = i + 1

    return contexts


def build_ragflow_reference_export_item(
    index: Any,
    item: dict[str, Any],
    include_raw_reference: bool = True,
) -> dict[str, Any]:
    contexts = item.get("retrieved_contexts") or []
    export_item = {
        "index": index,
        "id": item.get("id") or item.get("qid") or item.get("question_id"),
        "system_name": item.get("system_name"),
        "method": item.get("method"),
        "method_name": item.get("method_name"),
        "question": item.get("question", ""),
        # answer / clean_answer are evaluation-ready and do not contain [ID:n].
        "answer": item.get("format_answer") or item.get("generated_answer") or "",
        "clean_answer": item.get("format_answer") or item.get("generated_answer") or "",
        # raw_ragflow_answer preserves RAGFlow citation markers for debugging.
        "raw_ragflow_answer": item.get("raw_ragflow_answer"),
        "raw_ragflow_answer_is_error": item.get("raw_ragflow_answer_is_error"),
        "atomic_facts": item.get("format_answer_atomic_facts") or item.get("generated_atomic_facts") or [],
        "answer_status": item.get("answer_status"),
        "fact_status": item.get("fact_status"),
        "retrieved_context_count": len(contexts),
        "retrieved_context_token_count": item.get("retrieved_context_token_count"),
        "ctx_tok": item.get("ctx_tok"),
        "retrieved_contexts": contexts,
        "raw_generation_info": item.get("raw_generation_info"),
    }
    if include_raw_reference:
        export_item["raw_ragflow_reference"] = item.get("raw_ragflow_reference")
    return export_item


def save_ragflow_reference_export(
    data: Any,
    output_path: Path,
    include_raw_reference: bool = True,
) -> dict[str, Any]:
    refs = []
    for key, item in iter_items(data):
        if not isinstance(item, dict):
            continue
        refs.append(build_ragflow_reference_export_item(
            index=key,
            item=item,
            include_raw_reference=include_raw_reference,
        ))

    summary = {
        "output_path": str(output_path),
        "num_items": len(refs),
        "num_success": sum(1 for x in refs if x.get("answer_status") == "success"),
        "num_with_contexts": sum(1 for x in refs if x.get("retrieved_context_count", 0) > 0),
        "avg_context_count": (
            sum(x.get("retrieved_context_count", 0) for x in refs) / len(refs)
            if refs else None
        ),
        "items": refs,
    }
    save_json(output_path, summary)
    return summary


# =========================
# Generation wrappers
# =========================

def generate_direct_answer(
    provider: str,
    model: str,
    base_url: str,
    question: str,
    answer_client: Any | None,
) -> dict[str, Any]:
    if provider == "su8":
        if answer_client is None:
            raise RuntimeError("SU8 answer client is not initialized.")
        answer = call_su8_answer_only(
            client=answer_client,
            model=model,
            question=question,
        )
    elif provider == "ollama":
        answer = call_ollama_answer_only(
            model=model,
            base_url=base_url,
            question=question,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return {
        "answer": answer,
        "contexts": [],
        "ctx_tok": None,
        "raw_reference": None,
        "raw_generation_info": None,
    }


def generate_maxctx_answer(
    provider: str,
    model: str,
    base_url: str,
    question: str,
    answer_client: Any | None,
    maxctx_context: dict[str, Any],
) -> dict[str, Any]:
    context_text = maxctx_context["text"]
    context_token_count = maxctx_context["token_count"]
    context_file = maxctx_context.get("context_file")

    contexts = [
        {
            "rank": 1,
            "source": str(context_file) if context_file else "max_context_file",
            "doc_id": None,
            "chunk_id": None,
            "text": context_text,
            "score": None,
            "token_count": context_token_count,
            "raw": {
                "context_file": str(context_file) if context_file else None,
                "original_token_count": maxctx_context.get("original_token_count"),
                "was_truncated": maxctx_context.get("was_truncated"),
            },
        }
    ]

    if provider == "su8":
        if answer_client is None:
            raise RuntimeError("SU8 answer client is not initialized.")
        answer = call_su8_answer_with_context(
            client=answer_client,
            model=model,
            question=question,
            contexts=contexts,
        )
    elif provider == "ollama":
        answer = call_ollama_answer_with_context(
            model=model,
            base_url=base_url,
            question=question,
            contexts=contexts,
        )
    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return {
        "answer": answer,
        "contexts": contexts,
        "ctx_tok": context_token_count,
        "raw_reference": None,
        "raw_generation_info": {
            "context_file": str(context_file) if context_file else None,
            "original_token_count": maxctx_context.get("original_token_count"),
            "was_truncated": maxctx_context.get("was_truncated"),
        },
    }


def generate_ragflow_answer(
    question: str,
    ragflow_base_url: str,
    ragflow_api_key: str,
    ragflow_chat_id: str,
    ragflow_model: str,
    ragflow_endpoint_style: str,
) -> dict[str, Any]:
    raw_answer, reference, raw_info = run_ragflow_mode(
        question=question,
        ragflow_base_url=ragflow_base_url,
        ragflow_api_key=ragflow_api_key,
        ragflow_chat_id=ragflow_chat_id,
        ragflow_model=ragflow_model,
        endpoint_style=ragflow_endpoint_style,
    )

    # Keep raw_ragflow_answer for debugging citations; the evaluation-facing
    # answer goes through the unified v3 cleaner.
    clean_answer = normalize_ragflow_answer_for_eval(raw_answer)

    contexts = extract_contexts_from_ragflow_reference(reference)
    ctx_tok = sum(int(ctx.get("token_count") or 0) for ctx in contexts)

    return {
        "answer": clean_answer,
        "raw_answer": raw_answer,
        "raw_answer_is_error": is_ragflow_error_answer(raw_answer),
        "contexts": contexts,
        "ctx_tok": ctx_tok,
        "raw_reference": reference,
        "raw_generation_info": raw_info,
    }


def generate_answer_for_method(
    method: str,
    provider: str,
    model: str,
    base_url: str,
    question: str,
    answer_client: Any | None,
    maxctx_context: dict[str, Any] | None,
    ragflow_config: dict[str, Any] | None,
) -> dict[str, Any]:
    start_t = time.time()

    if method == "direct":
        result = generate_direct_answer(
            provider=provider,
            model=model,
            base_url=base_url,
            question=question,
            answer_client=answer_client,
        )
    elif method == "maxctx":
        if maxctx_context is None:
            raise ValueError("maxctx method requires --context_file.")
        result = generate_maxctx_answer(
            provider=provider,
            model=model,
            base_url=base_url,
            question=question,
            answer_client=answer_client,
            maxctx_context=maxctx_context,
        )
    elif method == "ragflow":
        if ragflow_config is None:
            raise ValueError("ragflow method requires RAGFlow configuration.")
        result = generate_ragflow_answer(
            question=question,
            ragflow_base_url=ragflow_config["base_url"],
            ragflow_api_key=ragflow_config["api_key"],
            ragflow_chat_id=ragflow_config["chat_id"],
            ragflow_model=ragflow_config["model"],
            ragflow_endpoint_style=ragflow_config["endpoint_style"],
        )
    else:
        raise ValueError(f"Unsupported method: {method}")

    elapsed = time.time() - start_t
    answer = result.get("answer", "") or ""
    result["time"] = round(elapsed, 4)
    result["tokens_generated"] = count_tokens(answer)
    result["tokens_per_sec"] = (
        None if elapsed <= 0 else round(result["tokens_generated"] / elapsed, 4)
    )

    return result


# =========================
# Output path and resume
# =========================

def display_method_name(method: str, method_name: str | None) -> str:
    if method_name:
        return method_name
    if method == "direct":
        return "Direct"
    if method == "maxctx":
        return "MaxContext"
    if method == "ragflow":
        return "RAGFlow"
    return method


def output_path_for(
    input_path: Path,
    output_dir: Path | None,
    provider: str,
    model: str,
    method: str,
    method_name: str,
    fact_provider: str,
    fact_model: str,
) -> Path:
    suffix = (
        f"_{sanitize_name(method_name or method)}"
        f"_{provider}_{sanitize_name(model)}"
        f"_facts_{fact_provider}_{sanitize_name(fact_model)}.json"
    )
    out_name = input_path.stem + suffix
    if output_dir:
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir / out_name
    return input_path.with_name(out_name)


def get_existing_complete_item(
    item: dict[str, Any],
    provider: str,
    model: str,
    method: str,
    method_name: str,
    fact_provider: str,
    fact_model: str,
) -> bool:
    answer_ok = (
        item.get("generator_provider") == provider
        and item.get("generator_model") == model
        and item.get("method") == method
        and item.get("method_name") == method_name
        and bool(item.get("format_answer"))
        and item.get("answer_status") == "success"
    )

    facts_ok = (
        item.get("fact_provider") == fact_provider
        and item.get("fact_model") == fact_model
        and "format_answer_atomic_facts" in item
        and item.get("fact_status") == "success"
    )

    return bool(answer_ok and facts_ok)


# =========================
# Dataset processing
# =========================

def extract_gold_answer(item: dict[str, Any]) -> str:
    return (
        item.get("gold_answer")
        or item.get("answer")
        or item.get("reference_answer")
        or ""
    )


def extract_gold_atomic_facts(item: dict[str, Any]) -> list[str]:
    raw = (
        item.get("gold_atomic_facts")
        or item.get("answer_atomic_facts")
        or item.get("atomic_facts")
        or item.get("gold_answer_atomic_facts")
        or []
    )
    if isinstance(raw, list):
        return [str(x).strip() for x in raw if str(x).strip()]
    return []


def process_file(
    input_path: Path,
    output_path: Path,
    system_name: str,
    provider: str,
    model: str,
    base_url: str,
    method: str,
    method_name: str,
    fact_provider: str,
    fact_model: str,
    fact_base_url: str,
    answer_client: Any | None,
    fact_client: Any,
    maxctx_context: dict[str, Any] | None,
    ragflow_config: dict[str, Any] | None,
    overwrite: bool,
    save_every: int,
    limit: int | None,
    sleep_seconds: float,
) -> None:
    if output_path.exists() and not overwrite:
        print(f"[Resume] Loading existing output: {output_path}")
        data = load_json(output_path)
    else:
        print(f"[Read] {input_path}")
        data = load_json(input_path)

    total = len(data) if hasattr(data, "__len__") else 0
    processed_this_run = 0
    success_this_run = 0
    failed_this_run = 0
    skipped_existing = 0

    print(f"[Process] {input_path.name} -> {output_path.name}, total={total}")
    print(f"  system={system_name}, generator={model}, method={method_name}")

    for key, item in iter_items(data):
        if not isinstance(item, dict):
            continue

        if get_existing_complete_item(
            item=item,
            provider=provider,
            model=model,
            method=method,
            method_name=method_name,
            fact_provider=fact_provider,
            fact_model=fact_model,
        ) and not overwrite:
            skipped_existing += 1
            continue

        if limit is not None and processed_this_run >= limit:
            break

        question = item.get("question", "") or ""
        gold_answer = extract_gold_answer(item)
        gold_atomic_facts = extract_gold_atomic_facts(item)

        common_meta = {
            "system_name": system_name,
            "generator_provider": provider,
            "generator_model": model,
            "generator_base_url": base_url,
            "method": method,
            "method_name": method_name,
            "fact_provider": fact_provider,
            "fact_model": fact_model,
            "fact_base_url": fact_base_url,
            # Provenance: which cleaner produced format_answer.
            "answer_cleaner": ANSWER_CLEANER_VERSION,
        }

        if not question.strip():
            item.update(common_meta)
            item["gold_answer"] = gold_answer
            item["gold_atomic_facts"] = gold_atomic_facts
            item["format_answer"] = ""
            item["format_answer_atomic_facts"] = []
            item["generated_answer"] = ""
            item["generated_atomic_facts"] = []
            item["retrieved_contexts"] = []
            item["retrieved_context_token_count"] = None
            item["ctx_tok"] = None
            item["raw_ragflow_answer"] = None
            item["raw_ragflow_answer_is_error"] = None
            item["answer_status"] = "skipped_missing_question"
            item["fact_status"] = "skipped_missing_question"
            processed_this_run += 1
            continue

        print(f"  Generating item {key}...", flush=True)

        try:
            generation = generate_answer_for_method(
                method=method,
                provider=provider,
                model=model,
                base_url=base_url,
                question=question,
                answer_client=answer_client,
                maxctx_context=maxctx_context,
                ragflow_config=ragflow_config,
            )

            # v3: ALL methods' evaluation answers pass through the same unified
            # cleaner. For ragflow, generation["answer"] is already cleaned by
            # normalize_ragflow_answer_for_eval; the cleaner is idempotent, so
            # applying it again here is harmless and keeps the invariant
            # "every format_answer went through clean_rag_answer_for_eval".
            generated_answer = clean_rag_answer_for_eval(generation.get("answer", "") or "")
            generated_answer = normalize_answer_text(generated_answer)
            retrieved_contexts = generation.get("contexts", []) or []
            retrieved_context_token_count = generation.get("ctx_tok", None)

            generated_facts = call_su8_decompose_facts(
                client=fact_client,
                model=fact_model,
                answer=generated_answer,
            )

            item.update(common_meta)

            item["gold_answer"] = gold_answer
            item["gold_atomic_facts"] = gold_atomic_facts

            # Evaluation-compatible fields.
            item["format_answer"] = generated_answer
            item["format_answer_atomic_facts"] = generated_facts

            # Generic generated fields.
            item["generated_answer"] = generated_answer
            item["generated_atomic_facts"] = generated_facts

            # Backward-compatible fields for previous closed-book script.
            item["closed_book_answer"] = generated_answer if method == "direct" else ""
            item["closed_book_atomic_facts"] = generated_facts if method == "direct" else []

            if provider == "su8" and model == "gpt-5.5":
                item["gpt55_answer"] = generated_answer
                item["gpt55_atomic_facts"] = generated_facts

            item["retrieved_contexts"] = retrieved_contexts
            item["retrieved_context_token_count"] = retrieved_context_token_count
            item["ctx_tok"] = retrieved_context_token_count

            # For RAGFlow, keep the raw answer with [ID:n] citations for debugging,
            # but use the cleaned generated_answer / format_answer for atomic
            # facts and evaluation.
            if method == "ragflow":
                item["raw_ragflow_answer"] = generation.get("raw_answer")
                item["raw_ragflow_answer_is_error"] = bool(generation.get("raw_answer_is_error"))
            else:
                item["raw_ragflow_answer"] = None
                item["raw_ragflow_answer_is_error"] = None

            item["raw_ragflow_reference"] = generation.get("raw_reference")
            item["raw_generation_info"] = generation.get("raw_generation_info")

            # RAGFlow-reference export fields. These duplicate the standardized
            # context fields under explicit names so downstream scripts and
            # manual checks can find each question's reference contexts easily.
            if method == "ragflow":
                item["ragflow_reference"] = generation.get("raw_reference")
                item["ragflow_reference_contexts"] = retrieved_contexts
                item["ragflow_reference_context_count"] = len(retrieved_contexts)
                item["ragflow_reference_token_count"] = retrieved_context_token_count

            item["answer_word_count"] = word_count(generated_answer)
            item["answer_token_count"] = count_tokens(generated_answer)
            item["atomic_fact_count"] = len(generated_facts)

            item["generation_time"] = generation.get("time")
            item["tokens_generated"] = generation.get("tokens_generated")
            item["tokens_per_sec"] = generation.get("tokens_per_sec")

            item["answer_status"] = "success"
            item["fact_status"] = "success"

            if provider == "ollama":
                item["ollama_options"] = LLM_FIXED_PARAMS

            success_this_run += 1

        except Exception as e:
            item.update(common_meta)

            item["gold_answer"] = gold_answer
            item["gold_atomic_facts"] = gold_atomic_facts

            item["format_answer"] = ""
            item["format_answer_atomic_facts"] = []
            item["generated_answer"] = ""
            item["generated_atomic_facts"] = []
            item["retrieved_contexts"] = []
            item["retrieved_context_token_count"] = None
            item["ctx_tok"] = None
            item["raw_ragflow_answer"] = None
            item["raw_ragflow_answer_is_error"] = None
            if method == "ragflow":
                item["raw_ragflow_reference"] = None
                item["ragflow_reference"] = None
                item["ragflow_reference_contexts"] = []
                item["ragflow_reference_context_count"] = 0
                item["ragflow_reference_token_count"] = None

            item["answer_status"] = "failed"
            item["fact_status"] = "failed"
            item["generation_error"] = str(e)

            failed_this_run += 1
            print(f"  [Failed] item {key}: {e}", flush=True)

        processed_this_run += 1

        if save_every > 0 and processed_this_run % save_every == 0:
            save_json(output_path, data)
            print(f"  [Saved checkpoint] {output_path}", flush=True)

        if sleep_seconds > 0:
            time.sleep(sleep_seconds)

    save_json(output_path, data)

    print(f"[Done] {input_path.name}")
    print(f"  output: {output_path}")
    print(f"  processed_this_run: {processed_this_run}")
    print(f"  success_this_run: {success_this_run}")
    print(f"  failed_this_run: {failed_this_run}")
    print(f"  skipped_existing: {skipped_existing}")
    print()


# =========================
# External metrics
# =========================

def run_external_metric_script(
    script_path: str,
    input_path: Path,
    output_path: Path,
    metric_name: str,
    input_arg: str = "--input",
    output_arg: str = "--output",
    extra_args: list[str] | None = None,
    continue_on_error: bool = False,
) -> Path:
    if not script_path:
        print(f"[Metric:{metric_name}] skipped: no script path")
        return input_path

    script = Path(script_path)
    if not script.exists():
        msg = f"[Metric:{metric_name}] script not found: {script_path}"
        if continue_on_error:
            print(msg)
            return input_path
        raise FileNotFoundError(msg)

    cmd = [
        sys.executable,
        str(script),
        input_arg,
        str(input_path),
        output_arg,
        str(output_path),
    ]

    if extra_args:
        cmd.extend(extra_args)

    print(f"[Metric:{metric_name}] Running: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as e:
        if continue_on_error:
            print(f"[Metric:{metric_name}] failed but continuing: {e}")
            return input_path
        raise

    if not output_path.exists():
        msg = f"[Metric:{metric_name}] output file not found after running: {output_path}"
        if continue_on_error:
            print(msg)
            return input_path
        raise FileNotFoundError(msg)

    return output_path


def run_all_metrics(
    generated_path: Path,
    method: str,
    args: argparse.Namespace,
) -> Path:
    """
    Unified metric execution.

    Direct:
      Ans.R + Answer Correctness + RA-FC.
    Non-Direct:
      Ans.R + Answer Correctness + RA-FC + Faithfulness + Context Recall.

    The unified evaluator internally shares the retrieved-context NLI pass for
    faithfulness and context recall, so we no longer need the old chained
    answer_correctness -> faithfulness -> context_recall execution.
    """
    metrics_path = generated_path.with_name(generated_path.stem + "_all_metrics.json")

    extra_args = [
        "--method", method,
        "--method_name", display_method_name(method, args.method_name),
        "--su8_model", args.metric_su8_model,
        "--su8_base_url", args.metric_su8_base_url,
        "--ollama_base_url", args.metric_ollama_base_url,
        "--embedding_model", args.metric_embedding_model,
        "--weights", str(args.metric_factual_weight), str(args.metric_similarity_weight),
        "--beta", str(args.metric_beta),
        "--ra_beta", str(args.metric_ra_beta),
        "--eval_chunk_size", str(args.metric_eval_chunk_size),
        "--eval_chunk_overlap", str(args.metric_eval_chunk_overlap),
        "--nli_batch_size", str(args.metric_nli_batch_size),
        "--max_retries", str(args.metric_max_retries),
        "--sleep_between_retries", str(args.metric_sleep_between_retries),
        "--save_every", str(args.metric_save_every),
        "--sleep", str(args.metric_sleep),
    ]

    if args.direct_summary_path:
        extra_args.extend(["--direct_summary_path", args.direct_summary_path])

    if args.run_context_metrics_for_direct:
        extra_args.append("--run_context_metrics_for_direct")

    if args.metric_skip_existing_metrics:
        extra_args.append("--skip_existing_metrics")

    if args.metric_disable_similarity:
        extra_args.append("--disable_similarity")

    if args.metric_no_details:
        extra_args.append("--no_details")

    if args.metric_disable_strict_schema:
        extra_args.append("--disable_strict_schema")

    if args.metric_disable_schema_fallback:
        extra_args.append("--disable_schema_fallback")

    if args.metric_errors_continue:
        # The external runner handles continue_on_error. This flag is not passed
        # to evaluate_all_metrics_su8.py because it is a generation-script policy.
        pass

    return run_external_metric_script(
        script_path=args.evaluate_all_metrics_script,
        input_path=generated_path,
        output_path=metrics_path,
        metric_name="all_metrics",
        input_arg="--input",
        output_arg="--output",
        extra_args=extra_args,
        continue_on_error=args.metric_errors_continue,
    )


# =========================
# Summary
# =========================

def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        try:
            return float(value)
        except Exception:
            return None
    return None


def normalize_metric_key(key: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", str(key).lower())


def recursive_find_numeric(
    obj: Any,
    candidate_keys: set[str],
    depth: int = 0,
    max_depth: int = 4,
) -> float | None:
    if depth > max_depth:
        return None

    if isinstance(obj, dict):
        for key, value in obj.items():
            norm_key = normalize_metric_key(key)
            if norm_key in candidate_keys:
                f = safe_float(value)
                if f is not None:
                    return f

                if isinstance(value, dict):
                    for sub_key in ["score", "value", "mean"]:
                        if sub_key in value:
                            f = safe_float(value[sub_key])
                            if f is not None:
                                return f

        for value in obj.values():
            if isinstance(value, (dict, list)):
                found = recursive_find_numeric(value, candidate_keys, depth + 1, max_depth)
                if found is not None:
                    return found

    elif isinstance(obj, list):
        for value in obj:
            found = recursive_find_numeric(value, candidate_keys, depth + 1, max_depth)
            if found is not None:
                return found

    return None


def get_metric_value(item: dict[str, Any], keys: list[str]) -> float | None:
    candidate_keys = {normalize_metric_key(k) for k in keys}
    return recursive_find_numeric(item, candidate_keys)


def safe_mean(values: list[Any]) -> float | None:
    numeric_values: list[float] = []
    for value in values:
        f = safe_float(value)
        if f is not None:
            numeric_values.append(f)

    if not numeric_values:
        return None

    return sum(numeric_values) / len(numeric_values)


def build_summary(
    data: Any,
    result_path: Path,
    direct_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    items = flatten_items(data)
    n = len(items)

    first = items[0] if items else {}

    ans_rate = safe_mean([
        1.0 if is_non_empty_answer(item.get("format_answer", "")) else 0.0
        for item in items
    ])

    ctx_tok_values = [
        item.get("retrieved_context_token_count")
        for item in items
        if item.get("retrieved_context_token_count") is not None
    ]
    ctx_tok = safe_mean(ctx_tok_values)

    context_recall = safe_mean([
        get_metric_value(
            item,
            [
                "context_recall",
                "context_recall_score",
                "ctx_recall",
                "ctx.r",
                "Ctx.R",
            ],
        )
        for item in items
    ])

    faithfulness = safe_mean([
        get_metric_value(
            item,
            [
                "faithfulness",
                "faithfulness_score",
                "faith",
                "Faith.",
            ],
        )
        for item in items
    ])

    answer_correctness = safe_mean([
        get_metric_value(
            item,
            [
                "answer_correctness",
                "answer_correctness_score",
                "ans_correctness",
                "ans.c",
                "Ans.C.",
                "ragas_answer_correctness",
            ],
        )
        for item in items
    ])

    ra_fc = safe_mean([
        get_metric_value(
            item,
            [
                "ra_fc",
                "ra_fc_score",
                "rafc",
                "RA-FC",
                "relation_aware_factual_correctness",
                "ra_relation_aware_factual_correctness",
                "rarelationawarefactualcorrectness",
                "relation_aware_fc",
                "relationawarefactualcorrectness",
            ],
        )
        for item in items
    ])

    summary = {
        "system": first.get("system_name"),
        "n": n,
        "generator": first.get("generator_model"),
        "provider": first.get("generator_provider"),
        "method": first.get("method_name") or first.get("method"),
        "method_raw": first.get("method"),
        "answer_cleaner": first.get("answer_cleaner"),
        "ctx_tok": ctx_tok,
        "context_recall": context_recall,
        "faithfulness": faithfulness,
        "answer_rate": ans_rate,
        "answer_correctness": answer_correctness,
        "ra_fc": ra_fc,
        "delta_ra": None,
        "result_path": str(result_path),
    }

    if direct_summary is not None:
        direct_ra = safe_float(direct_summary.get("ra_fc"))
        current_ra = safe_float(ra_fc)
        same_system = direct_summary.get("system") == summary.get("system")
        same_generator = direct_summary.get("generator") == summary.get("generator")

        if (
            direct_ra is not None
            and current_ra is not None
            and same_system
            and same_generator
            and summary.get("method") != "Direct"
        ):
            summary["delta_ra"] = current_ra - direct_ra

    return summary


def save_method_summary(
    final_result_path: Path,
    summary_output: Path | None,
    direct_summary_path: Path | None,
) -> dict[str, Any]:
    data = load_json(final_result_path)

    direct_summary = None
    if direct_summary_path and direct_summary_path.exists():
        direct_summary = load_json(direct_summary_path)

    summary = build_summary(
        data=data,
        result_path=final_result_path,
        direct_summary=direct_summary,
    )

    default_summary_path = final_result_path.with_name(final_result_path.stem + "_summary.json")
    save_json(default_summary_path, summary)
    print(f"[Summary] Saved: {default_summary_path}")

    if summary_output:
        save_json(summary_output, summary)
        print(f"[Summary] Saved also to: {summary_output}")

    return summary


# =========================
# Main
# =========================

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Generate answers with Direct, MaxContext, or RAGFlow-RAG, then decompose "
            "all generated answers into atomic facts using SU8 GPT-5.5."
        )
    )

    parser.add_argument(
        "--method",
        choices=["direct", "maxctx", "ragflow"],
        default=DEFAULT_METHOD,
        help="Answer generation method.",
    )
    parser.add_argument(
        "--method_name",
        default=None,
        help="Display name for table, e.g., Direct, Doc-RAG, Code-RAG, Report-RAG, Int.-RAG, Int.-MaxCtx.",
    )
    parser.add_argument(
        "--system_name",
        default=None,
        help="Subject system name, e.g., Ollama, OpenClaw, LangChain. If omitted, guessed from input filename.",
    )

    parser.add_argument(
        "--provider",
        choices=["su8", "ollama"],
        default=DEFAULT_PROVIDER,
        help="Answer-generation backend provider for direct/maxctx. For ragflow, this is metadata for the generator configuration.",
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=DEFAULT_INPUTS,
        help="Input JSON files.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Optional output directory. Default: same directory as each input file.",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Generator model. Default: gpt-5.5 for su8, qwen3-coder for ollama.",
    )
    parser.add_argument(
        "--base_url",
        default=None,
        help="Answer-generation base URL. Default: SU8 codex URL for su8, http://localhost:11434 for ollama.",
    )

    # MaxContext
    parser.add_argument(
        "--context_file",
        default=None,
        help="Context file for maxctx method, e.g., langchain.graphml.",
    )
    parser.add_argument(
        "--max_context_tokens",
        type=int,
        default=None,
        help=(
            "Maximum context tokens for maxctx. "
            "Default: derived from Ollama num_ctx when provider=ollama; otherwise 30000."
        ),
    )

    # RAGFlow
    parser.add_argument(
        "--ragflow_base_url",
        default=DEFAULT_RAGFLOW_BASE_URL,
        help="RAGFlow base URL, e.g., http://host:80/api/v1. Can also use RAGFLOW_BASE_URL.",
    )
    parser.add_argument(
        "--ragflow_api_key",
        default=DEFAULT_RAGFLOW_API_KEY,
        help="RAGFlow API key. Prefer environment variable RAGFLOW_API_KEY.",
    )
    parser.add_argument(
        "--ragflow_chat_id",
        default=DEFAULT_RAGFLOW_CHAT_ID,
        help="RAGFlow chat ID. Can also use RAGFLOW_CHAT_ID.",
    )
    parser.add_argument(
        "--ragflow_model",
        default=DEFAULT_RAGFLOW_MODEL,
        help="Model name sent to RAGFlow OpenAI-compatible endpoint. Default: ragflow-model.",
    )
    parser.add_argument(
        "--ragflow_endpoint_style",
        choices=["auto", "openai", "chats_openai"],
        default="auto",
        help="RAGFlow endpoint style. auto tries /openai first, then /chats_openai.",
    )
    parser.add_argument(
        "--ragflow_mode_name",
        default=None,
        help="Optional metadata label, e.g., doc, code, report, integrated, hybrid.",
    )
    parser.add_argument(
        "--no_ragflow_reference_export",
        action="store_true",
        help="Do not write the separate *_ragflow_references.json file for ragflow runs.",
    )
    parser.add_argument(
        "--ragflow_references_output",
        default=None,
        help=(
            "Optional path for the separate RAGFlow reference export JSON. "
            "Only use with one input; otherwise it would be overwritten."
        ),
    )
    parser.add_argument(
        "--ragflow_reference_export_no_raw",
        action="store_true",
        help="Export standardized contexts but omit raw_ragflow_reference from the separate reference file.",
    )

    # Fact decomposition
    parser.add_argument(
        "--fact_model",
        default=DEFAULT_FACT_MODEL,
        help="SU8 model used to decompose generated answers into atomic facts.",
    )
    parser.add_argument(
        "--fact_base_url",
        default=DEFAULT_FACT_BASE_URL,
        help="SU8 base URL used for atomic fact decomposition.",
    )

    # Resume/control
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate answers and facts even if complete outputs already exist.",
    )
    parser.add_argument(
        "--save_every",
        type=int,
        default=10,
        help="Save checkpoint every N processed items.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Process only N new items per file, useful for testing.",
    )
    parser.add_argument(
        "--sleep",
        type=float,
        default=0.2,
        help="Sleep seconds between items.",
    )

    # External metrics
    parser.add_argument(
        "--run_metrics",
        action="store_true",
        help=(
            "Run unified evaluate_all_metrics_su8.py after generation. "
            "Direct computes Ans.R/Ans.C/RA-FC; non-Direct also computes Faith./Ctx.R."
        ),
    )
    parser.add_argument(
        "--evaluate_all_metrics_script",
        default="evaluate_all_metrics_su8.py",
        help="Unified metric script that computes Ans.R, Ans.C, RA-FC, and optional Faith./Ctx.R.",
    )
    parser.add_argument(
        "--metric_skip_existing_metrics",
        action="store_true",
        help="Pass --skip_existing_metrics to evaluate_all_metrics_su8.py so existing per-item metrics such as RA-FC are reused.",
    )
    parser.add_argument(
        "--metric_su8_model",
        default=DEFAULT_SU8_MODEL,
        help="SU8 model used by the unified evaluator.",
    )
    parser.add_argument(
        "--metric_su8_base_url",
        default=DEFAULT_SU8_BASE_URL,
        help="SU8 base URL used by the unified evaluator.",
    )
    parser.add_argument(
        "--metric_ollama_base_url",
        default=DEFAULT_OLLAMA_BASE_URL,
        help="Local Ollama base URL used for answer-correctness semantic similarity.",
    )
    parser.add_argument(
        "--metric_embedding_model",
        default=DEFAULT_EMBEDDING_MODEL,
        help="Ollama embedding model used for semantic similarity.",
    )
    parser.add_argument("--metric_factual_weight", type=float, default=0.75)
    parser.add_argument("--metric_similarity_weight", type=float, default=0.25)
    parser.add_argument("--metric_beta", type=float, default=1.0)
    parser.add_argument("--metric_ra_beta", type=float, default=1.0)
    parser.add_argument("--metric_eval_chunk_size", type=int, default=3000)
    parser.add_argument("--metric_eval_chunk_overlap", type=int, default=300)
    parser.add_argument("--metric_nli_batch_size", type=int, default=5)
    parser.add_argument("--metric_max_retries", type=int, default=3)
    parser.add_argument("--metric_sleep_between_retries", type=float, default=2.0)
    parser.add_argument("--metric_save_every", type=int, default=10)
    parser.add_argument("--metric_sleep", type=float, default=0.0)
    parser.add_argument(
        "--metric_disable_similarity",
        action="store_true",
        help="Disable local Ollama embedding similarity in Answer Correctness.",
    )
    parser.add_argument(
        "--metric_no_details",
        action="store_true",
        help="Do not save detailed item-level metric explanations.",
    )
    parser.add_argument(
        "--metric_disable_strict_schema",
        action="store_true",
        help="Disable strict JSON schema in metric evaluator.",
    )
    parser.add_argument(
        "--metric_disable_schema_fallback",
        action="store_true",
        help="Do not fallback to json_object if strict schema fails in metric evaluator.",
    )
    parser.add_argument(
        "--metric_errors_continue",
        action="store_true",
        help="Continue even if the unified metric script fails.",
    )
    parser.add_argument(
        "--run_context_metrics_for_direct",
        action="store_true",
        help="Also run faithfulness/context_recall for Direct. Default: skip because Direct has no retrieved contexts.",
    )

    # Summary
    parser.add_argument(
        "--summary_output",
        default=None,
        help="Optional path for method-level summary JSON. If multiple inputs, only use this for one input or it will be overwritten.",
    )
    parser.add_argument(
        "--direct_summary_path",
        default=None,
        help="Optional direct summary JSON used to compute DeltaRA for non-Direct methods.",
    )
    parser.add_argument(
        "--all_summaries_output",
        default=None,
        help="Optional path to save a list of summaries when multiple input files are processed.",
    )

    args = parser.parse_args()

    method = args.method
    method_name = display_method_name(method, args.method_name)

    provider = args.provider
    if args.model:
        model = args.model
    else:
        model = DEFAULT_SU8_MODEL if provider == "su8" else DEFAULT_OLLAMA_MODEL

    if args.base_url:
        base_url = args.base_url
    else:
        base_url = DEFAULT_SU8_BASE_URL if provider == "su8" else DEFAULT_OLLAMA_BASE_URL

    fact_provider = DEFAULT_FACT_PROVIDER
    fact_model = args.fact_model
    fact_base_url = args.fact_base_url

    # MaxContext setup.
    maxctx_context = None
    if method == "maxctx":
        if not args.context_file:
            raise SystemExit("maxctx method requires --context_file.")

        if args.max_context_tokens is not None:
            max_context_tokens = args.max_context_tokens
        elif provider == "ollama":
            max_context_tokens = (
                int(LLM_FIXED_PARAMS.get("num_ctx", 4096))
                - int(LLM_FIXED_PARAMS.get("num_predict", 300))
                - 500
            )
            max_context_tokens = max(max_context_tokens, 512)
        else:
            max_context_tokens = 30000

        print(f"[MaxContext] Loading context file: {args.context_file}")
        print(f"[MaxContext] Token budget: {max_context_tokens}")

        context_text, kept_tokens, original_tokens, was_truncated = load_raw_file_truncated(
            filepath=args.context_file,
            max_tokens=max_context_tokens,
        )

        maxctx_context = {
            "context_file": args.context_file,
            "text": context_text,
            "token_count": kept_tokens,
            "original_token_count": original_tokens,
            "was_truncated": was_truncated,
            "max_context_tokens": max_context_tokens,
        }

        print(
            f"[MaxContext] Loaded: kept={kept_tokens}, "
            f"original={original_tokens}, truncated={was_truncated}"
        )

    # RAGFlow setup.
    ragflow_config = None
    if method == "ragflow":
        ragflow_config = {
            "base_url": args.ragflow_base_url,
            "api_key": args.ragflow_api_key,
            "chat_id": args.ragflow_chat_id,
            "model": args.ragflow_model,
            "endpoint_style": args.ragflow_endpoint_style,
            "mode_name": args.ragflow_mode_name,
        }

        if not ragflow_config["base_url"]:
            raise SystemExit("ragflow method requires --ragflow_base_url or RAGFLOW_BASE_URL.")
        if not ragflow_config["api_key"]:
            raise SystemExit("ragflow method requires --ragflow_api_key or RAGFLOW_API_KEY.")
        if not ragflow_config["chat_id"]:
            raise SystemExit("ragflow method requires --ragflow_chat_id or RAGFLOW_CHAT_ID.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise SystemExit("Missing dependency. Install with: pip install openai") from exc

    su8_api_key = get_su8_api_key()

    answer_client = None
    if provider == "su8" and method in {"direct", "maxctx"}:
        answer_client = OpenAI(
            base_url=base_url,
            api_key=su8_api_key,
            timeout=120,
        )

    fact_client = OpenAI(
        base_url=fact_base_url,
        api_key=su8_api_key,
        timeout=120,
    )

    output_dir = Path(args.output_dir) if args.output_dir else None
    direct_summary_path = Path(args.direct_summary_path) if args.direct_summary_path else None

    print()
    print("========== Experiment Config ==========")
    print("System name:", args.system_name or "(guessed per input)")
    print("Method:", method)
    print("Method name:", method_name)
    print("Answer provider:", provider)
    print("Answer model:", model)
    print("Answer base URL:", base_url)
    print("Answer cleaner:", ANSWER_CLEANER_VERSION)
    if provider == "ollama":
        print("Ollama fixed params:", json.dumps(LLM_FIXED_PARAMS, ensure_ascii=False))
    if method == "ragflow":
        print("RAGFlow base URL:", args.ragflow_base_url)
        print("RAGFlow chat ID:", args.ragflow_chat_id)
        print("RAGFlow model:", args.ragflow_model)
        print("RAGFlow endpoint style:", args.ragflow_endpoint_style)
        print("RAGFlow mode name:", args.ragflow_mode_name)
    print("Fact provider:", fact_provider)
    print("Fact model:", fact_model)
    print("Fact base URL:", fact_base_url)
    print("Inputs:", args.inputs)
    print("Run metrics:", args.run_metrics)
    print("=======================================")
    print()

    all_summaries: list[dict[str, Any]] = []

    for input_name in args.inputs:
        input_path = Path(input_name)
        if not input_path.exists():
            print(f"[Skip] Input file not found: {input_path}")
            continue

        system_name = args.system_name or guess_system_name(input_path)

        out_path = output_path_for(
            input_path=input_path,
            output_dir=output_dir,
            provider=provider,
            model=model,
            method=method,
            method_name=method_name,
            fact_provider=fact_provider,
            fact_model=fact_model,
        )

        process_file(
            input_path=input_path,
            output_path=out_path,
            system_name=system_name,
            provider=provider,
            model=model,
            base_url=base_url,
            method=method,
            method_name=method_name,
            fact_provider=fact_provider,
            fact_model=fact_model,
            fact_base_url=fact_base_url,
            answer_client=answer_client,
            fact_client=fact_client,
            maxctx_context=maxctx_context,
            ragflow_config=ragflow_config,
            overwrite=args.overwrite,
            save_every=args.save_every,
            limit=args.limit,
            sleep_seconds=args.sleep,
        )

        if method == "ragflow" and not args.no_ragflow_reference_export:
            if args.ragflow_references_output and len(args.inputs) == 1:
                ragflow_reference_path = Path(args.ragflow_references_output)
            else:
                ragflow_reference_path = out_path.with_name(out_path.stem + "_ragflow_references.json")

            reference_export_summary = save_ragflow_reference_export(
                data=load_json(out_path),
                output_path=ragflow_reference_path,
                include_raw_reference=not args.ragflow_reference_export_no_raw,
            )
            print(
                f"[RAGFlow references] Saved: {ragflow_reference_path} | "
                f"items={reference_export_summary.get('num_items')} | "
                f"with_contexts={reference_export_summary.get('num_with_contexts')}",
                flush=True,
            )

        final_result_path = out_path

        if args.run_metrics:
            final_result_path = run_all_metrics(
                generated_path=out_path,
                method=method,
                args=args,
            )

        # If multiple inputs are processed, avoid overwriting --summary_output repeatedly
        # unless the user explicitly knows they are processing one input.
        summary_output = None
        if args.summary_output and len(args.inputs) == 1:
            summary_output = Path(args.summary_output)

        summary = save_method_summary(
            final_result_path=final_result_path,
            summary_output=summary_output,
            direct_summary_path=direct_summary_path,
        )
        all_summaries.append(summary)

    if args.all_summaries_output:
        save_json(Path(args.all_summaries_output), all_summaries)
        print(f"[All summaries] Saved: {args.all_summaries_output}")


if __name__ == "__main__":
    main()
