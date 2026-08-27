#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Unified evaluator for QA generation outputs.

This script is intended to replace the old chained metric execution:
  answer_correctness_su8.py -> faithfulness_su8.py -> context_recall_su8.py
plus a separate relation_aware_factual_correctness_su8.py pass.

Metric logic:
  - Direct method:
      Ans.R, Answer Correctness, RA-FC
      Faithfulness and Context Recall are skipped by default.

  - Context methods, such as maxctx / ragflow:
      Ans.R, Answer Correctness, RA-FC, Faithfulness, Context Recall
      Faithfulness and Context Recall share one context-support NLI pass per chunk:
        statements = generated_facts + gold_facts

Per-item fields written:
  - answer_rate / ans_r
  - answer_correctness
  - factual_correctness
  - similarity_score
  - answer_correctness_details
  - ra_relation_precision
  - ra_relation_recall
  - ra_relation_aware_factual_correctness
  - ra_fc / relation_aware_factual_correctness
  - faithfulness
  - context_recall
  - retrieved_context_token_count / ctx_tok

FIX (this version):
  The generated-fact fallback decomposition is now performed ONCE in the outer
  evaluation loop, BEFORE any calculator is called. Previously the fallback ran
  inside AnswerCorrectnessCalculator and its rebound local variable never reached
  RA-FC or the context-metric calls, so those metrics still saw an empty
  pred_facts list and collapsed to 0 / None even though the log printed
  "[Fallback] ... decomposing text with SU8.". Now the decomposed facts are
  shared by all metrics and written back into the item.

Usage:
  python evaluate_all_metrics_su8.py \
    --input generated.json \
    --output generated_all_metrics.json \
    --method direct \
    --su8_model gpt-5.5 \
    --ollama_base_url http://localhost:11434 \
    --embedding_model nomic-embed-text

  python evaluate_all_metrics_su8.py \
    --input generated_rag.json \
    --output generated_rag_all_metrics.json \
    --method ragflow \
    --fallback_decompose_missing_facts \
    --direct_summary_path outputs/direct_summary.json

Environment:
  export SU8_API_KEY="your_key"
  # or export CODEX_API_KEY="your_key"

For Answer Correctness similarity:
  ollama pull nomic-embed-text
"""

from __future__ import annotations

import argparse
import asyncio
import json
import math
import os
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx


# =============================================================================
# Defaults
# =============================================================================

DEFAULT_SU8_MODEL = "gpt-5.5"
DEFAULT_SU8_BASE_URL = "https://www.su8.codes/codex/v1"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"
DEFAULT_EMBEDDING_MODEL = "nomic-embed-text"

UNKNOWN_ANSWERS = {
    "",
    "i don't know",
    "i don't know.",
    "unknown",
    "unknown.",
    "not sure",
    "not sure.",
    "无法确定",
    "不知道",
    "不确定",
}

RELATIONS = [
    "equivalent",
    "pred_entails_gold",
    "gold_entails_pred",
    "partial_overlap",
    "coordinate",
    "contradiction",
    "unrelated",
]

PRECISION_CREDIT = {
    "equivalent": 1.0,
    "gold_entails_pred": 1.0,
    "pred_entails_gold": 0.7,
    "partial_overlap": 0.5,
    "coordinate": 0.2,
    "contradiction": 0.0,
    "unrelated": 0.0,
}

RECALL_CREDIT = {
    "equivalent": 1.0,
    "pred_entails_gold": 1.0,
    "gold_entails_pred": 0.6,
    "partial_overlap": 0.5,
    "coordinate": 0.0,
    "contradiction": 0.0,
    "unrelated": 0.0,
}


# =============================================================================
# Config
# =============================================================================

@dataclass
class EvalConfig:
    weights: list[float] = field(default_factory=lambda: [0.75, 0.25])
    beta: float = 1.0
    ra_beta: float = 1.0

    eval_chunk_size: int = 3000
    eval_chunk_overlap: int = 300
    nli_batch_size: int = 5
    max_retries: int = 3
    sleep_between_retries: float = 2.0

    strict_schema: bool = True
    allow_json_object_fallback: bool = True
    save_details: bool = True

    fallback_decompose_missing_facts: bool = False
    max_fallback_claims: int = 12


# =============================================================================
# JSON / dataset helpers
# =============================================================================

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


def load_json(path: str | Path) -> Any:
    path = Path(path)
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path: str | Path, data: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")


def iter_items(data: Any):
    if isinstance(data, list):
        for idx, item in enumerate(data):
            yield idx, item
    elif isinstance(data, dict):
        if isinstance(data.get("items"), list):
            for idx, item in enumerate(data["items"]):
                yield idx, item
        else:
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


def count_items(data: Any) -> int:
    return len(flatten_items(data))


def strip_code_fence(text: str) -> str:
    text = (text or "").strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    return text.strip()


def extract_json_object(text: str) -> dict[str, Any]:
    text = strip_code_fence(text)
    try:
        parsed = json.loads(text)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        raise ValueError(f"No JSON object found in response: {text[:500]}")
    parsed = json.loads(match.group(0))
    if not isinstance(parsed, dict):
        raise ValueError(f"Parsed JSON is not object: {type(parsed)}")
    return parsed


def safe_round(x: float | None, ndigits: int = 4) -> float | None:
    if x is None:
        return None
    if isinstance(x, float) and (math.isnan(x) or math.isinf(x)):
        return None
    return round(float(x), ndigits)


def safe_float(value: Any) -> float | None:
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    if isinstance(value, str):
        try:
            f = float(value)
            if math.isnan(f) or math.isinf(f):
                return None
            return f
        except Exception:
            return None
    return None


def safe_mean(values: list[Any]) -> float | None:
    numeric: list[float] = []
    for v in values:
        f = safe_float(v)
        if f is not None:
            numeric.append(f)
    if not numeric:
        return None
    return sum(numeric) / len(numeric)


def is_unknown_or_empty(text: str) -> bool:
    return (text or "").strip().lower().rstrip() in UNKNOWN_ANSWERS


def is_non_empty_answer(text: str) -> bool:
    return not is_unknown_or_empty(text)


def normalize_fact_list(raw: Any) -> list[str]:
    if raw is None:
        return []
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, list):
        return []

    facts: list[str] = []
    seen: set[str] = set()
    for x in raw:
        if not isinstance(x, str):
            continue
        fact = re.sub(r"^\s*[-*•]\s*", "", x)
        fact = re.sub(r"\s+", " ", fact).strip()
        if not fact:
            continue
        key = fact.lower()
        if key in seen:
            continue
        seen.add(key)
        facts.append(fact)
    return facts


def get_question(item: dict[str, Any]) -> str:
    return str(item.get("question") or item.get("query") or "").strip()


def get_generated_answer(item: dict[str, Any]) -> str:
    return str(
        item.get("format_answer")
        or item.get("generated_answer")
        or item.get("closed_book_answer")
        or item.get("gpt55_closed_book_answer")
        or item.get("gpt55_answer")
        or ""
    ).strip()


def get_gold_answer(item: dict[str, Any]) -> str:
    return str(
        item.get("gold_answer")
        or item.get("reference_answer")
        or item.get("answer")
        or ""
    ).strip()


def get_generated_facts(item: dict[str, Any]) -> list[str]:
    raw = (
        item.get("format_answer_atomic_facts")
        or item.get("generated_atomic_facts")
        or item.get("closed_book_atomic_facts")
        or item.get("gpt55_closed_book_atomic_facts")
        or item.get("gpt55_atomic_facts")
        or []
    )
    return normalize_fact_list(raw)


def get_gold_facts(item: dict[str, Any]) -> list[str]:
    raw = (
        item.get("gold_atomic_facts")
        or item.get("answer_atomic_facts")
        or item.get("gold_answer_atomic_facts")
        or item.get("atomic_facts")
        or []
    )
    return normalize_fact_list(raw)


# =============================================================================
# Token / context helpers
# =============================================================================

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
    return len(re.findall(r"\S+", text))


def chunk_text(text: str, chunk_size: int, chunk_overlap: int) -> list[str]:
    text = text or ""
    if not text.strip():
        return []
    if chunk_size <= 0 or len(text) <= chunk_size:
        return [text]

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + chunk_size, n)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= n:
            break
        start = max(0, end - chunk_overlap)
        if start >= end:
            start = end
    return chunks


def _extract_text_from_context_obj(obj: Any) -> str | None:
    if obj is None:
        return None
    if isinstance(obj, str):
        text = obj.strip()
        return text or None
    if not isinstance(obj, dict):
        return None

    for key in [
        "text", "content", "chunk", "chunk_text", "page_content",
        "document", "body", "evidence_text",
    ]:
        value = obj.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return None


def _walk_reference_for_contexts(obj: Any, out: list[str], seen: set[str]) -> None:
    if obj is None:
        return
    if isinstance(obj, str):
        text = obj.strip()
        if text and text[:500] not in seen:
            seen.add(text[:500])
            out.append(text)
        return
    if isinstance(obj, list):
        for x in obj:
            _walk_reference_for_contexts(x, out, seen)
        return
    if isinstance(obj, dict):
        text = _extract_text_from_context_obj(obj)
        if text and text[:500] not in seen:
            seen.add(text[:500])
            out.append(text)
        for key in [
            "chunks", "references", "reference", "documents", "docs",
            "results", "items", "data", "contexts", "retrieved_contexts",
        ]:
            value = obj.get(key)
            if isinstance(value, (list, dict)):
                _walk_reference_for_contexts(value, out, seen)


def extract_contexts(item: dict[str, Any]) -> list[str]:
    contexts: list[str] = []
    seen: set[str] = set()

    raw_contexts = item.get("retrieved_contexts") or item.get("contexts") or item.get("context") or []
    if isinstance(raw_contexts, str):
        raw_contexts = [raw_contexts]
    if isinstance(raw_contexts, list):
        for ctx in raw_contexts:
            text = _extract_text_from_context_obj(ctx)
            if text and text[:500] not in seen:
                seen.add(text[:500])
                contexts.append(text)

    raw_ref = item.get("reference")
    if raw_ref:
        if isinstance(raw_ref, str):
            try:
                parsed = json.loads(raw_ref)
                _walk_reference_for_contexts(parsed, contexts, seen)
            except Exception:
                text = raw_ref.strip()
                if text and text[:500] not in seen:
                    seen.add(text[:500])
                    contexts.append(text)
        else:
            _walk_reference_for_contexts(raw_ref, contexts, seen)

    raw_ragflow_ref = item.get("raw_ragflow_reference")
    if raw_ragflow_ref:
        _walk_reference_for_contexts(raw_ragflow_ref, contexts, seen)

    return contexts


def load_context_file(context_file: str | None, max_chars: int | None = None) -> list[str]:
    if not context_file:
        return []
    path = Path(context_file)
    if not path.exists():
        print(f"[Warning] context file not found: {context_file}", flush=True)
        return []
    text = path.read_text(encoding="utf-8", errors="ignore")
    if max_chars and max_chars > 0 and len(text) > max_chars:
        text = text[:max_chars] + "\n\n[CONTENT TRUNCATED]"
    return [text]


# =============================================================================
# SU8 structured client and schemas
# =============================================================================

def nli_verification_response_format(name: str = "nli_verification") -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": name,
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "verifications": {
                        "type": "array",
                        "items": {"type": "boolean"},
                        "description": "Boolean support labels, one per input statement.",
                    }
                },
                "required": ["verifications"],
                "additionalProperties": False,
            },
        },
    }


def fallback_claim_decomposition_response_format() -> dict[str, Any]:
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "fallback_claim_decomposition",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "claims": {
                        "type": "array",
                        "items": {"type": "string"},
                    }
                },
                "required": ["claims"],
                "additionalProperties": False,
            },
        },
    }


def relation_eval_schema() -> dict[str, Any]:
    pred_alignment = {
        "type": "object",
        "properties": {
            "pred_index": {"type": "integer"},
            "best_gold_index": {"type": "integer"},
            "relation": {"type": "string", "enum": RELATIONS},
            "reason": {"type": "string"},
        },
        "required": ["pred_index", "best_gold_index", "relation", "reason"],
        "additionalProperties": False,
    }
    gold_coverage = {
        "type": "object",
        "properties": {
            "gold_index": {"type": "integer"},
            "best_pred_index": {"type": "integer"},
            "relation": {"type": "string", "enum": RELATIONS},
            "reason": {"type": "string"},
        },
        "required": ["gold_index", "best_pred_index", "relation", "reason"],
        "additionalProperties": False,
    }
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "relation_aware_fact_evaluation",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "pred_alignments": {"type": "array", "items": pred_alignment},
                    "gold_coverage": {"type": "array", "items": gold_coverage},
                },
                "required": ["pred_alignments", "gold_coverage"],
                "additionalProperties": False,
            },
        },
    }


class SU8StructuredChatClient:
    def __init__(
        self,
        model: str,
        base_url: str,
        api_key: str | None = None,
        timeout: int = 120,
        strict_schema: bool = True,
        allow_json_object_fallback: bool = True,
    ) -> None:
        try:
            from openai import AsyncOpenAI
        except ImportError as exc:
            raise SystemExit("Missing dependency. Install with: pip install openai") from exc

        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key or get_su8_api_key(), timeout=timeout)
        self.model = model
        self.base_url = base_url
        self.strict_schema = strict_schema
        self.allow_json_object_fallback = allow_json_object_fallback

    async def close(self) -> None:
        try:
            await self.client.close()
        except Exception:
            pass

    async def call_json(
        self,
        system_prompt: str,
        user_prompt: str,
        response_format: dict[str, Any],
        max_tokens: int = 1600,
        max_retries: int = 3,
        sleep_between_retries: float = 2.0,
    ) -> dict[str, Any]:
        last_error: Exception | None = None
        for attempt in range(1, max_retries + 1):
            try:
                kwargs: dict[str, Any] = {
                    "model": self.model,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt},
                    ],
                    "temperature": 0,
                    "max_tokens": max_tokens,
                }
                if self.strict_schema:
                    kwargs["response_format"] = response_format
                else:
                    kwargs["response_format"] = {"type": "json_object"}

                try:
                    response = await self.client.chat.completions.create(**kwargs)
                except Exception as e:
                    msg = str(e)
                    if (
                        self.strict_schema
                        and self.allow_json_object_fallback
                        and ("response_format" in msg or "json_schema" in msg or "schema" in msg or "strict" in msg)
                    ):
                        kwargs["response_format"] = {"type": "json_object"}
                        response = await self.client.chat.completions.create(**kwargs)
                    else:
                        raise

                content = response.choices[0].message.content or ""
                return extract_json_object(content)

            except Exception as e:
                last_error = e
                if attempt < max_retries:
                    wait = sleep_between_retries * attempt
                    print(f"  [Retry:SU8] attempt {attempt}/{max_retries} failed: {e}. Retrying in {wait:.1f}s...", flush=True)
                    await asyncio.sleep(wait)

        raise RuntimeError(f"SU8 structured chat failed after {max_retries} attempts: {last_error}")


# =============================================================================
# NLI verification and fallback fact decomposition
# =============================================================================

class SU8NLIVerifier:
    SYSTEM_PROMPT = """You are a strict NLI support verifier for software-documentation QA evaluation.

Task:
Given a context and a list of statements, decide whether each statement is fully supported by the context.

Rules:
- Use ONLY the provided context.
- Mark true only if the statement is directly stated by, or strictly entailed by, the context.
- Do not use outside knowledge.
- For exact commands, endpoints, paths, configuration keys, protocols, ports, values, field names, and numbers, require exact support.
- If the context is insufficient, ambiguous, only topically related, or contradicts the statement, mark false.
- Return JSON only with this schema: {"verifications": [true, false, ...]}.
"""

    def __init__(self, chat_client: SU8StructuredChatClient, config: EvalConfig) -> None:
        self.chat_client = chat_client
        self.config = config

    @staticmethod
    def _create_prompt(context: str, statements: list[str]) -> str:
        lines = "\n".join(f"S{i}: {statement}" for i, statement in enumerate(statements))
        return f"""Context:
{context}

Statements:
{lines}

Return one boolean support label for every statement, in the same order."""

    async def _verify_batch(self, context: str, statements: list[str]) -> list[bool]:
        if not statements:
            return []
        for attempt in range(1, self.config.max_retries + 1):
            try:
                parsed = await self.chat_client.call_json(
                    system_prompt=self.SYSTEM_PROMPT,
                    user_prompt=self._create_prompt(context, statements),
                    response_format=nli_verification_response_format(),
                    max_tokens=800,
                    max_retries=1,
                    sleep_between_retries=self.config.sleep_between_retries,
                )
                raw = parsed.get("verifications", [])
                if not isinstance(raw, list):
                    raise ValueError(f"verifications is not list: {parsed}")
                verifications = [bool(x) for x in raw]
                if len(verifications) == len(statements):
                    return verifications
                print(f"    [Verifier Retry] count mismatch: expected={len(statements)}, got={len(verifications)}", flush=True)
            except Exception as e:
                print(f"    [Verifier Retry] attempt {attempt}/{self.config.max_retries} failed: {e}", flush=True)
            if attempt < self.config.max_retries:
                await asyncio.sleep(self.config.sleep_between_retries * attempt)
        print(f"    [Verifier Error] Defaulting {len(statements)} statements to False.", flush=True)
        return [False] * len(statements)

    async def verify(self, context: str, statements: list[str]) -> list[bool]:
        if not statements:
            return []
        results: list[bool] = []
        bs = max(1, int(self.config.nli_batch_size))
        for i in range(0, len(statements), bs):
            batch = statements[i:i + bs]
            results.extend(await self._verify_batch(context, batch))
        return results

    async def verify_against_contexts(self, contexts: list[str], statements: list[str]) -> list[bool]:
        if not statements:
            return []
        if not contexts:
            return [False] * len(statements)

        chunks: list[str] = []
        for ctx in contexts:
            chunks.extend(chunk_text(ctx, self.config.eval_chunk_size, self.config.eval_chunk_overlap))

        supported = [False] * len(statements)
        for chunk_idx, chunk in enumerate(chunks):
            pending_indices = [i for i, ok in enumerate(supported) if not ok]
            if not pending_indices:
                break
            pending_statements = [statements[i] for i in pending_indices]
            batch_result = await self.verify(chunk, pending_statements)
            for local_i, ok in enumerate(batch_result):
                if ok:
                    supported[pending_indices[local_i]] = True
            print(f"    [Context NLI] chunk {chunk_idx + 1}/{len(chunks)} supported={sum(supported)}/{len(statements)}", flush=True)
        return supported


class SU8FallbackClaimDecomposer:
    SYSTEM_PROMPT = """You decompose an input answer into independent factual claims.

Rules:
- Decompose ONLY the given input text.
- Do not use outside knowledge.
- Do not add facts that are not stated.
- Each claim must be independently checkable.
- Preserve exact technical terms, commands, API paths, configuration keys, parameters, model names, numbers, ports, endpoints, and code identifiers.
- Return JSON only with this schema: {"claims": ["...", "..."]}.
"""

    def __init__(self, chat_client: SU8StructuredChatClient, config: EvalConfig) -> None:
        self.chat_client = chat_client
        self.config = config

    async def decompose(self, text: str) -> list[str]:
        if not text or is_unknown_or_empty(text):
            return []
        prompt = f"""Input text:
{text}

Decompose the input text into independent factual claims. Return at most {self.config.max_fallback_claims} claims."""
        try:
            parsed = await self.chat_client.call_json(
                system_prompt=self.SYSTEM_PROMPT,
                user_prompt=prompt,
                response_format=fallback_claim_decomposition_response_format(),
                max_tokens=1200,
                max_retries=self.config.max_retries,
                sleep_between_retries=self.config.sleep_between_retries,
            )
            claims = normalize_fact_list(parsed.get("claims", []))
            return claims[: self.config.max_fallback_claims]
        except Exception as e:
            print(f"  [Fallback Decompose Error] {e}", flush=True)
            return []


# =============================================================================
# Similarity and scalar metrics
# =============================================================================

class OllamaSemanticSimilarityCalculator:
    def __init__(
        self,
        model: str,
        base_url: str,
        timeout: float = 180.0,
        max_retries: int = 5,
        retry_sleep: float = 3.0,
        allow_fallback: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.embeddings_url_old = f"{self.base_url}/api/embeddings"
        self.embeddings_url_new = f"{self.base_url}/api/embed"
        self.client = httpx.AsyncClient(timeout=timeout)
        self.max_retries = max_retries
        self.retry_sleep = retry_sleep
        self.allow_fallback = allow_fallback
        self.last_backend = "not_called"
        self.last_error: str | None = None

    async def close(self) -> None:
        await self.client.aclose()

    async def _get_embedding_old_api(self, text: str) -> list[float]:
        response = await self.client.post(self.embeddings_url_old, json={"model": self.model, "prompt": text})
        response.raise_for_status()
        emb = response.json().get("embedding")
        if not isinstance(emb, list):
            raise RuntimeError(f"No valid 'embedding' field from {self.embeddings_url_old}")
        return emb

    async def _get_embedding_new_api(self, text: str) -> list[float]:
        response = await self.client.post(self.embeddings_url_new, json={"model": self.model, "input": text})
        response.raise_for_status()
        payload = response.json()
        embeddings = payload.get("embeddings")
        if isinstance(embeddings, list) and embeddings and isinstance(embeddings[0], list):
            return embeddings[0]
        emb = payload.get("embedding")
        if isinstance(emb, list):
            return emb
        raise RuntimeError(f"No valid embedding field from {self.embeddings_url_new}")

    async def _get_embedding(self, text: str) -> list[float]:
        if not text:
            raise ValueError("Cannot embed empty text")

        last_error: Exception | None = None
        for attempt in range(1, self.max_retries + 1):
            # Try the stable legacy endpoint first, then the newer endpoint.
            for api_name, fn in [("/api/embeddings", self._get_embedding_old_api), ("/api/embed", self._get_embedding_new_api)]:
                try:
                    emb = await fn(text)
                    self.last_backend = f"ollama_embedding{api_name}"
                    self.last_error = None
                    return emb
                except Exception as e:
                    last_error = e

            if attempt < self.max_retries:
                wait = min(60.0, self.retry_sleep * attempt)
                print(
                    f"  [Retry:Ollama-embedding] attempt {attempt}/{self.max_retries} failed: {last_error}. "
                    f"Retrying in {wait:.1f}s...",
                    flush=True,
                )
                await asyncio.sleep(wait)

        self.last_backend = "failed"
        self.last_error = str(last_error)
        raise RuntimeError(f"Ollama embedding failed after {self.max_retries} attempts: {last_error}")

    @staticmethod
    def _cosine_similarity(v1: list[float], v2: list[float]) -> float:
        if not v1 or not v2 or len(v1) != len(v2):
            raise ValueError(f"Invalid embedding dimensions: {len(v1) if v1 else 0}, {len(v2) if v2 else 0}")
        dot = sum(a * b for a, b in zip(v1, v2))
        n1 = math.sqrt(sum(a * a for a in v1))
        n2 = math.sqrt(sum(b * b for b in v2))
        if n1 == 0 or n2 == 0:
            raise ValueError("Zero-norm embedding")
        return dot / (n1 * n2)

    @staticmethod
    def _simple_word_overlap(text1: str, text2: str) -> float:
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        return len(words1 & words2) / len(words1 | words2)

    async def calculate_similarity(self, text1: str, text2: str) -> float:
        if not text1 or not text2:
            self.last_backend = "empty_text"
            return 0.0
        try:
            emb1, emb2 = await asyncio.gather(self._get_embedding(text1), self._get_embedding(text2))
            cosine = self._cosine_similarity(emb1, emb2)
            self.last_backend = "ollama_embedding"
            self.last_error = None
            return float((cosine + 1.0) / 2.0)
        except Exception as e:
            self.last_error = str(e)
            if self.allow_fallback:
                self.last_backend = "word_overlap_fallback"
                return self._simple_word_overlap(text1, text2)
            self.last_backend = "failed"
            raise


def fbeta_count_score(tp: int, fp: int, fn: int, beta: float = 1.0) -> float:
    precision = 0.0 if tp + fp == 0 else tp / (tp + fp)
    recall = 0.0 if tp + fn == 0 else tp / (tp + fn)
    if precision == 0.0 and recall == 0.0:
        return 0.0
    beta2 = beta ** 2
    return ((1 + beta2) * precision * recall) / ((beta2 * precision) + recall)


def fbeta_pr_score(precision: float, recall: float, beta: float = 1.0) -> float:
    if precision <= 0 and recall <= 0:
        return 0.0
    beta2 = beta ** 2
    return ((1 + beta2) * precision * recall) / ((beta2 * precision) + recall)


# =============================================================================
# Answer correctness
# =============================================================================

class AnswerCorrectnessCalculator:
    def __init__(self, verifier: SU8NLIVerifier, decomposer: SU8FallbackClaimDecomposer, similarity: OllamaSemanticSimilarityCalculator | None, config: EvalConfig) -> None:
        self.verifier = verifier
        self.decomposer = decomposer
        self.similarity = similarity
        self.config = config

    async def _maybe_fallback(self, facts: list[str], text: str, role: str) -> tuple[list[str], bool]:
        # NOTE: with the outer-scope fallback in evaluate_file, generated_facts is
        # normally already populated by the time this runs, so this short-circuits.
        # It remains here as a safety net for direct calls to this calculator.
        if facts:
            return facts, False
        if not self.config.fallback_decompose_missing_facts or not text or is_unknown_or_empty(text):
            return facts, False
        print(f"  [Fallback] Missing {role}; decomposing text with SU8.", flush=True)
        return await self.decomposer.decompose(text), True

    async def calculate(self, answer: str, reference: str, generated_facts: list[str], gold_facts: list[str]) -> dict[str, Any]:
        if is_unknown_or_empty(answer):
            result = {"answer_correctness": 0.0, "factual_correctness": 0.0, "similarity_score": 0.0}
            if self.config.save_details:
                result["answer_correctness_details"] = {
                    "tp": 0, "fp": 0, "fn": 0,
                    "generated_facts": [], "gold_facts": gold_facts,
                    "generated_fact_supported_by_gold_answer": [],
                    "gold_fact_supported_by_generated_answer": [],
                    "reason": "empty_or_unknown_generated_answer",
                    "used_fallback_generated_facts": False,
                    "used_fallback_gold_facts": False,
                }
            return result

        if not reference:
            result = {"answer_correctness": None, "factual_correctness": None, "similarity_score": None}
            if self.config.save_details:
                result["answer_correctness_details"] = {"reason": "missing_gold_answer"}
            return result

        generated_facts, used_fallback_generated = await self._maybe_fallback(generated_facts, answer, "generated_facts")
        gold_facts, used_fallback_gold = await self._maybe_fallback(gold_facts, reference, "gold_facts")

        if not generated_facts:
            result = {"answer_correctness": None, "factual_correctness": None, "similarity_score": None}
            if self.config.save_details:
                result["answer_correctness_details"] = {
                    "reason": "missing_generated_atomic_facts",
                    "generated_facts": [], "gold_facts": gold_facts,
                    "used_fallback_generated_facts": used_fallback_generated,
                    "used_fallback_gold_facts": used_fallback_gold,
                }
            return result

        if not gold_facts:
            result = {"answer_correctness": None, "factual_correctness": None, "similarity_score": None}
            if self.config.save_details:
                result["answer_correctness_details"] = {
                    "reason": "missing_gold_atomic_facts",
                    "generated_facts": generated_facts, "gold_facts": [],
                    "used_fallback_generated_facts": used_fallback_generated,
                    "used_fallback_gold_facts": used_fallback_gold,
                }
            return result

        # TP / FP: generated fact supported by gold answer.
        generated_supported = await self.verifier.verify(reference, generated_facts)
        # FN: gold fact not supported by generated answer.
        gold_supported = await self.verifier.verify(answer, gold_facts)

        tp = sum(1 for x in generated_supported if x)
        fp = sum(1 for x in generated_supported if not x)
        fn = sum(1 for x in gold_supported if not x)

        factual_correctness = fbeta_count_score(tp, fp, fn, beta=self.config.beta)
        similarity_score = 0.0
        if self.config.weights[1] != 0 and self.similarity is not None:
            similarity_score = await self.similarity.calculate_similarity(answer, reference)

        denom = sum(self.config.weights)
        if denom == 0:
            final_score = factual_correctness
        else:
            final_score = (self.config.weights[0] * factual_correctness + self.config.weights[1] * similarity_score) / denom

        result = {
            "answer_correctness": safe_round(final_score, 3),
            "factual_correctness": safe_round(factual_correctness, 3),
            "similarity_score": safe_round(similarity_score, 3),
        }
        if self.config.save_details:
            result["answer_correctness_details"] = {
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "generated_facts": generated_facts,
                "gold_facts": gold_facts,
                "generated_fact_supported_by_gold_answer": [bool(x) for x in generated_supported],
                "gold_fact_supported_by_generated_answer": [bool(x) for x in gold_supported],
                "weights": self.config.weights,
                "beta": self.config.beta,
                "chat_backend": "su8_openai_compatible",
                "similarity_backend": getattr(self.similarity, "last_backend", "ollama_embedding") if self.similarity is not None else "disabled",
                "similarity_error": getattr(self.similarity, "last_error", None) if self.similarity is not None else None,
                "used_fallback_generated_facts": used_fallback_generated,
                "used_fallback_gold_facts": used_fallback_gold,
            }
        return result


# =============================================================================
# Context metrics: faithfulness + context recall sharing one context NLI pass
# =============================================================================

async def calculate_context_metrics_shared(
    verifier: SU8NLIVerifier,
    answer: str,
    generated_facts: list[str],
    gold_answer: str,
    gold_facts: list[str],
    contexts: list[str],
    config: EvalConfig,
) -> dict[str, Any]:
    context_token_count = sum(count_tokens(c) for c in contexts)

    if is_unknown_or_empty(answer):
        faithfulness_result = {
            "faithfulness": 0.0,
            "faithfulness_supported_count": 0,
            "faithfulness_statement_count": 0,
        }
        faith_details = {
            "reason": "empty_or_unknown_generated_answer",
            "statements": [],
            "supported": [],
            "context_count": len(contexts),
            "context_token_count": context_token_count,
        }
    elif not generated_facts:
        faithfulness_result = {
            "faithfulness": None,
            "faithfulness_supported_count": None,
            "faithfulness_statement_count": 0,
        }
        faith_details = {
            "reason": "missing_generated_atomic_facts",
            "statements": [],
            "supported": [],
            "context_count": len(contexts),
            "context_token_count": context_token_count,
        }
    elif not contexts:
        faithfulness_result = {
            "faithfulness": None,
            "faithfulness_supported_count": None,
            "faithfulness_statement_count": len(generated_facts),
        }
        faith_details = {
            "reason": "missing_retrieved_contexts",
            "statements": generated_facts,
            "supported": [False] * len(generated_facts),
            "context_count": 0,
            "context_token_count": 0,
        }
    else:
        faithfulness_result = {}
        faith_details = {}

    if not gold_facts:
        context_recall_result = {
            "context_recall": None,
            "context_recall_supported_count": None,
            "context_recall_statement_count": 0,
        }
        recall_details = {
            "reason": "missing_gold_atomic_facts",
            "gold_facts": [],
            "supported": [],
            "context_count": len(contexts),
            "context_token_count": context_token_count,
            "format_answer": answer,
            "format_answer_facts": generated_facts,
        }
    elif not contexts:
        context_recall_result = {
            "context_recall": None,
            "context_recall_supported_count": None,
            "context_recall_statement_count": len(gold_facts),
        }
        recall_details = {
            "reason": "missing_retrieved_contexts",
            "gold_facts": gold_facts,
            "supported": [False] * len(gold_facts),
            "context_count": 0,
            "context_token_count": 0,
            "format_answer": answer,
            "format_answer_facts": generated_facts,
        }
    else:
        context_recall_result = {}
        recall_details = {}

    # If both metrics already terminated due to missing answer/facts/contexts, no SU8 call is needed.
    need_gen = bool(generated_facts) and not is_unknown_or_empty(answer) and bool(contexts)
    need_gold = bool(gold_facts) and bool(contexts)
    if not need_gen and not need_gold:
        result = {}
        result.update(faithfulness_result)
        result.update(context_recall_result)
        if config.save_details:
            result["faithfulness_details"] = faith_details
            result["context_recall_details"] = recall_details
        return result

    # One shared context-NLI pass: generated_facts + gold_facts.
    combined_statements: list[str] = []
    gen_slice = slice(0, 0)
    gold_slice = slice(0, 0)
    if need_gen:
        gen_start = len(combined_statements)
        combined_statements.extend(generated_facts)
        gen_slice = slice(gen_start, len(combined_statements))
    if need_gold:
        gold_start = len(combined_statements)
        combined_statements.extend(gold_facts)
        gold_slice = slice(gold_start, len(combined_statements))

    combined_supported = await verifier.verify_against_contexts(contexts, combined_statements)

    if need_gen:
        gen_supported = combined_supported[gen_slice]
        supp_count = sum(1 for x in gen_supported if x)
        stmt_count = len(generated_facts)
        faithfulness_result = {
            "faithfulness": safe_round(supp_count / stmt_count if stmt_count else None, 4),
            "faithfulness_supported_count": supp_count,
            "faithfulness_statement_count": stmt_count,
        }
        faith_details = {
            "statements": generated_facts,
            "supported": [bool(x) for x in gen_supported],
            "context_count": len(contexts),
            "context_token_count": context_token_count,
            "shared_context_nli": True,
        }

    if need_gold:
        gold_supported = combined_supported[gold_slice]
        supp_count = sum(1 for x in gold_supported if x)
        stmt_count = len(gold_facts)
        context_recall_result = {
            "context_recall": safe_round(supp_count / stmt_count if stmt_count else None, 4),
            "context_recall_supported_count": supp_count,
            "context_recall_statement_count": stmt_count,
        }
        recall_details = {
            "gold_facts": gold_facts,
            "supported": [bool(x) for x in gold_supported],
            "context_count": len(contexts),
            "context_token_count": context_token_count,
            "format_answer": answer,
            "format_answer_facts": generated_facts,
            "shared_context_nli": True,
        }

    result = {}
    result.update(faithfulness_result)
    result.update(context_recall_result)
    if config.save_details:
        result["faithfulness_details"] = faith_details
        result["context_recall_details"] = recall_details
    return result


# =============================================================================
# Relation-aware factual correctness
# =============================================================================

RELATION_SYSTEM_PROMPT = """You are a strict evaluator for documentation-derived software QA.

You will compare predicted atomic facts against gold atomic facts.
The gold facts are the documentation-derived reference. Your task is NOT to judge whether a predicted fact is true in the real world. Your task is to classify the logical relation between each predicted fact and the most relevant gold fact.

Relation labels:
- equivalent: pred and gold express the same fact with the same meaning.
- pred_entails_gold: pred is more specific/stronger and logically entails the gold fact, but adds details beyond the gold fact.
- gold_entails_pred: pred is more general/weaker; the gold fact logically entails the pred fact, but pred misses gold-specific details.
- partial_overlap: pred and gold share meaningful content, but neither entails the other.
- coordinate: pred is same subject/category as a gold fact but states a different parallel capability/item/detail, with no entailment.
- contradiction: pred conflicts with a gold fact, such as a different command, port, endpoint, key binding, field name, protocol, or incompatible value.
- unrelated: pred has no meaningful relation to any gold fact.

Important rules:
- Use only the provided facts and question. Do not use external knowledge to upgrade a relation.
- If a prediction says a different concrete command, endpoint, port, key, field, or protocol value than gold, prefer contradiction.
- If a prediction is plausible but not stated or entailed by gold, classify it as coordinate or unrelated, not equivalent.
- For each pred fact, choose the best matching gold fact index, or -1 if none.
- For each gold fact, choose the best matching pred fact index, or -1 if none.
- Use relation labels exactly as defined.
- Return JSON only.
"""


def build_relation_prompt(question: str, gold_answer: str, pred_answer: str, gold_facts: list[str], pred_facts: list[str]) -> str:
    gold_lines = "\n".join(f"G{idx}: {fact}" for idx, fact in enumerate(gold_facts))
    pred_lines = "\n".join(f"P{idx}: {fact}" for idx, fact in enumerate(pred_facts))
    return f"""Question:
{question}

Gold answer:
{gold_answer}

Predicted answer:
{pred_answer}

Gold atomic facts:
{gold_lines}

Predicted atomic facts:
{pred_lines}

Return:
1. pred_alignments: one item for every predicted fact P_i, with its best matching gold fact G_j or -1.
2. gold_coverage: one item for every gold fact G_j, with its best matching predicted fact P_i or -1.
"""


def clean_relation(value: Any) -> str:
    if isinstance(value, str) and value in RELATIONS:
        return value
    return "unrelated"


def normalize_eval_output(raw: dict[str, Any], gold_facts: list[str], pred_facts: list[str]) -> dict[str, Any]:
    pred_by_idx: dict[int, dict[str, Any]] = {}
    for row in raw.get("pred_alignments", []):
        if not isinstance(row, dict):
            continue
        try:
            pred_idx = int(row.get("pred_index"))
        except Exception:
            continue
        if 0 <= pred_idx < len(pred_facts):
            try:
                best_gold = int(row.get("best_gold_index", -1))
            except Exception:
                best_gold = -1
            if best_gold < 0 or best_gold >= len(gold_facts):
                best_gold = -1
            pred_by_idx[pred_idx] = {
                "pred_index": pred_idx,
                "pred_fact": pred_facts[pred_idx],
                "best_gold_index": best_gold,
                "best_gold_fact": gold_facts[best_gold] if best_gold >= 0 else "",
                "relation": clean_relation(row.get("relation")),
                "reason": str(row.get("reason", "")),
            }

    pred_alignments = []
    for i, fact in enumerate(pred_facts):
        pred_alignments.append(pred_by_idx.get(i, {
            "pred_index": i,
            "pred_fact": fact,
            "best_gold_index": -1,
            "best_gold_fact": "",
            "relation": "unrelated",
            "reason": "Evaluator did not return an alignment for this predicted fact.",
        }))

    gold_by_idx: dict[int, dict[str, Any]] = {}
    for row in raw.get("gold_coverage", []):
        if not isinstance(row, dict):
            continue
        try:
            gold_idx = int(row.get("gold_index"))
        except Exception:
            continue
        if 0 <= gold_idx < len(gold_facts):
            try:
                best_pred = int(row.get("best_pred_index", -1))
            except Exception:
                best_pred = -1
            if best_pred < 0 or best_pred >= len(pred_facts):
                best_pred = -1
            gold_by_idx[gold_idx] = {
                "gold_index": gold_idx,
                "gold_fact": gold_facts[gold_idx],
                "best_pred_index": best_pred,
                "best_pred_fact": pred_facts[best_pred] if best_pred >= 0 else "",
                "relation": clean_relation(row.get("relation")),
                "reason": str(row.get("reason", "")),
            }

    gold_coverage = []
    for j, fact in enumerate(gold_facts):
        gold_coverage.append(gold_by_idx.get(j, {
            "gold_index": j,
            "gold_fact": fact,
            "best_pred_index": -1,
            "best_pred_fact": "",
            "relation": "unrelated",
            "reason": "Evaluator did not return a coverage row for this gold fact.",
        }))

    return {"pred_alignments": pred_alignments, "gold_coverage": gold_coverage}


def compute_relation_metrics(eval_output: dict[str, Any], gold_facts: list[str], pred_facts: list[str], beta: float = 1.0) -> dict[str, Any]:
    pred_alignments = eval_output["pred_alignments"]
    gold_coverage = eval_output["gold_coverage"]
    n_pred = len(pred_facts)
    n_gold = len(gold_facts)

    relation_counts = {rel: 0 for rel in RELATIONS}
    for row in pred_alignments:
        relation_counts[clean_relation(row.get("relation"))] += 1

    if n_pred == 0:
        relation_precision = 0.0
    else:
        relation_precision = sum(PRECISION_CREDIT[clean_relation(row.get("relation"))] for row in pred_alignments) / n_pred

    if n_gold == 0:
        relation_recall = 1.0 if n_pred == 0 else 0.0
    else:
        relation_recall = sum(RECALL_CREDIT[clean_relation(row.get("relation"))] for row in gold_coverage) / n_gold

    relation_f1 = fbeta_pr_score(relation_precision, relation_recall, beta=beta)
    denom = n_pred if n_pred > 0 else 1

    metrics = {
        "ra_relation_precision": safe_round(relation_precision, 4),
        "ra_relation_recall": safe_round(relation_recall, 4),
        "ra_relation_aware_factual_correctness": safe_round(relation_f1, 4),
        "ra_gold_fact_coverage": safe_round(relation_recall, 4),
        "ra_relation_counts": relation_counts,
        "ra_contradiction_rate": safe_round(relation_counts["contradiction"] / denom, 4),
        "ra_coordinate_extra_rate": safe_round(relation_counts["coordinate"] / denom, 4),
        "ra_partial_overlap_rate": safe_round(relation_counts["partial_overlap"] / denom, 4),
        "ra_over_specific_rate": safe_round(relation_counts["pred_entails_gold"] / denom, 4),
        "ra_under_specified_rate": safe_round(relation_counts["gold_entails_pred"] / denom, 4),
        "ra_unrelated_rate": safe_round(relation_counts["unrelated"] / denom, 4),
        "ra_unsupported_extra_rate": safe_round((relation_counts["coordinate"] + relation_counts["unrelated"] + relation_counts["contradiction"]) / denom, 4),
    }
    # Aliases for summary compatibility.
    metrics["ra_fc"] = metrics["ra_relation_aware_factual_correctness"]
    metrics["ra_fc_score"] = metrics["ra_relation_aware_factual_correctness"]
    metrics["relation_aware_factual_correctness"] = metrics["ra_relation_aware_factual_correctness"]
    return metrics


def empty_prediction_metrics(gold_facts: list[str], beta: float = 1.0) -> dict[str, Any]:
    pred_facts: list[str] = []
    eval_output = {
        "pred_alignments": [],
        "gold_coverage": [
            {
                "gold_index": i,
                "gold_fact": fact,
                "best_pred_index": -1,
                "best_pred_fact": "",
                "relation": "unrelated",
                "reason": "No predicted atomic facts were provided.",
            }
            for i, fact in enumerate(gold_facts)
        ],
    }
    metrics = compute_relation_metrics(eval_output, gold_facts, pred_facts, beta=beta)
    metrics["ra_pred_alignments"] = eval_output["pred_alignments"]
    metrics["ra_gold_coverage"] = eval_output["gold_coverage"]
    return metrics


class RelationAwareCalculator:
    def __init__(self, chat_client: SU8StructuredChatClient, config: EvalConfig) -> None:
        self.chat_client = chat_client
        self.config = config

    async def calculate(self, question: str, gold_answer: str, pred_answer: str, gold_facts: list[str], pred_facts: list[str]) -> dict[str, Any]:
        if not gold_facts:
            return {
                "ra_relation_precision": None,
                "ra_relation_recall": None,
                "ra_relation_aware_factual_correctness": None,
                "ra_fc": None,
                "relation_aware_factual_correctness": None,
                "relation_eval_status": "failed",
                "relation_eval_error": "missing_gold_atomic_facts",
                "relation_eval_gold_fact_count": 0,
                "relation_eval_pred_fact_count": len(pred_facts),
            }

        if not pred_facts or is_unknown_or_empty(pred_answer):
            metrics = empty_prediction_metrics(gold_facts, beta=self.config.ra_beta)
            metrics["relation_eval_status"] = "success"
            metrics["relation_eval_note"] = "empty_or_unknown_prediction"
        else:
            parsed = await self.chat_client.call_json(
                system_prompt=RELATION_SYSTEM_PROMPT,
                user_prompt=build_relation_prompt(question, gold_answer, pred_answer, gold_facts, pred_facts),
                response_format=relation_eval_schema(),
                max_tokens=1800,
                max_retries=self.config.max_retries,
                sleep_between_retries=self.config.sleep_between_retries,
            )
            normalized = normalize_eval_output(parsed, gold_facts, pred_facts)
            metrics = compute_relation_metrics(normalized, gold_facts, pred_facts, beta=self.config.ra_beta)
            metrics["ra_pred_alignments"] = normalized["pred_alignments"]
            metrics["ra_gold_coverage"] = normalized["gold_coverage"]
            metrics["relation_eval_status"] = "success"

        metrics["relation_eval_model"] = self.chat_client.model
        metrics["relation_eval_base_url"] = self.chat_client.base_url
        metrics["relation_eval_beta"] = self.config.ra_beta
        metrics["relation_eval_gold_fact_count"] = len(gold_facts)
        metrics["relation_eval_pred_fact_count"] = len(pred_facts)
        return metrics


# =============================================================================
# Summary
# =============================================================================

def build_summary(data: Any, result_path: Path, method: str | None = None, method_name: str | None = None, direct_summary: dict[str, Any] | None = None) -> dict[str, Any]:
    items = flatten_items(data)
    n = len(items)
    first = items[0] if items else {}

    ans_rate = safe_mean([1.0 if is_non_empty_answer(get_generated_answer(item)) else 0.0 for item in items])
    ctx_tok = safe_mean([item.get("retrieved_context_token_count") for item in items if item.get("retrieved_context_token_count") is not None])
    context_recall = safe_mean([item.get("context_recall") for item in items])
    faithfulness = safe_mean([item.get("faithfulness") for item in items])
    answer_correctness = safe_mean([item.get("answer_correctness") for item in items])
    factual_correctness = safe_mean([item.get("factual_correctness") for item in items])
    similarity_score = safe_mean([item.get("similarity_score") for item in items])
    ra_fc = safe_mean([item.get("ra_fc") or item.get("ra_relation_aware_factual_correctness") for item in items])

    summary = {
        "system": first.get("system_name"),
        "n": n,
        "generator": first.get("generator_model"),
        "provider": first.get("generator_provider"),
        "method": method_name or first.get("method_name") or first.get("method") or method,
        "method_raw": method or first.get("method"),
        "ctx_tok": safe_round(ctx_tok, 4) if ctx_tok is not None else None,
        "context_recall": safe_round(context_recall, 4) if context_recall is not None else None,
        "faithfulness": safe_round(faithfulness, 4) if faithfulness is not None else None,
        "answer_rate": safe_round(ans_rate, 4) if ans_rate is not None else None,
        "answer_correctness": safe_round(answer_correctness, 4) if answer_correctness is not None else None,
        "factual_correctness": safe_round(factual_correctness, 4) if factual_correctness is not None else None,
        "similarity_score": safe_round(similarity_score, 4) if similarity_score is not None else None,
        "ra_fc": safe_round(ra_fc, 4) if ra_fc is not None else None,
        "delta_ra": None,
        "result_path": str(result_path),
    }

    if direct_summary is not None:
        direct_ra = safe_float(direct_summary.get("ra_fc"))
        current_ra = safe_float(summary.get("ra_fc"))
        same_system = direct_summary.get("system") == summary.get("system")
        same_generator = direct_summary.get("generator") == summary.get("generator")
        is_direct = (str(summary.get("method") or "").lower() == "direct") or (str(summary.get("method_raw") or "").lower() == "direct")
        if direct_ra is not None and current_ra is not None and same_system and same_generator and not is_direct:
            summary["delta_ra"] = safe_round(current_ra - direct_ra, 4)

    return summary


# =============================================================================
# Main evaluation loop
# =============================================================================


def _field_present(item: dict[str, Any], key: str) -> bool:
    """True when a metric field exists and is not None.

    0.0 is a valid computed score, so do not use truthiness checks.
    """
    return key in item and item.get(key) is not None


def has_existing_answer_correctness(item: dict[str, Any]) -> bool:
    return (
        _field_present(item, "answer_correctness")
        and _field_present(item, "factual_correctness")
        and _field_present(item, "similarity_score")
    )


def has_existing_ra_fc(item: dict[str, Any]) -> bool:
    return any(
        _field_present(item, key)
        for key in [
            "ra_relation_aware_factual_correctness",
            "ra_fc",
            "ra_fc_score",
            "relation_aware_factual_correctness",
        ]
    )


def normalize_existing_ra_aliases(item: dict[str, Any]) -> None:
    """If RA-FC was produced by another script, fill aliases used by summaries."""
    value = None
    for key in [
        "ra_relation_aware_factual_correctness",
        "ra_fc",
        "ra_fc_score",
        "relation_aware_factual_correctness",
    ]:
        if _field_present(item, key):
            value = item.get(key)
            break
    if value is None:
        return
    item.setdefault("ra_relation_aware_factual_correctness", value)
    item.setdefault("ra_fc", value)
    item.setdefault("ra_fc_score", value)
    item.setdefault("relation_aware_factual_correctness", value)


def has_existing_context_metrics(item: dict[str, Any]) -> bool:
    return _field_present(item, "faithfulness") and _field_present(item, "context_recall")


async def ensure_generated_facts(
    item: dict[str, Any],
    answer: str,
    generated_facts: list[str],
    decomposer: SU8FallbackClaimDecomposer,
    config: EvalConfig,
) -> tuple[list[str], bool]:
    """Outer-scope fallback so ALL metrics share one generated-fact list.

    Previously the fallback lived inside AnswerCorrectnessCalculator, whose
    rebound local variable never reached RA-FC or the context-metric calls.
    Here we decompose once, write the result back into the item, and return it
    so every downstream calculator sees the same non-empty facts.
    """
    if generated_facts:
        return generated_facts, False
    if not config.fallback_decompose_missing_facts:
        return generated_facts, False
    if not is_non_empty_answer(answer):
        return generated_facts, False

    print("  [Fallback] Missing generated_facts; decomposing format_answer with SU8.", flush=True)
    decomposed = await decomposer.decompose(answer)
    if decomposed:
        # Persist so the output file is self-consistent and reusable on resume.
        item["format_answer_atomic_facts"] = decomposed
        item["format_answer_atomic_fact_count"] = len(decomposed)
        item["format_answer_atomic_fact_source"] = "evaluator_fallback_decompose"
        return decomposed, True

    # Decomposition was attempted but returned nothing; leave a breadcrumb.
    item["format_answer_atomic_fact_source"] = "evaluator_fallback_decompose_empty"
    return generated_facts, True


async def evaluate_file(args: argparse.Namespace) -> None:
    input_path = Path(args.input)
    output_path = Path(args.output) if args.output else input_path.with_name(input_path.stem + "_all_metrics.json")

    if output_path.exists() and not args.overwrite:
        print(f"[Resume] Loading existing output: {output_path}")
        data = load_json(output_path)
    else:
        print(f"[Read] {input_path}")
        data = load_json(input_path)

    config = EvalConfig(
        weights=args.weights,
        beta=args.beta,
        ra_beta=args.ra_beta,
        eval_chunk_size=args.eval_chunk_size,
        eval_chunk_overlap=args.eval_chunk_overlap,
        nli_batch_size=args.nli_batch_size,
        max_retries=args.max_retries,
        sleep_between_retries=args.sleep_between_retries,
        strict_schema=not args.disable_strict_schema,
        allow_json_object_fallback=not args.disable_schema_fallback,
        save_details=not args.no_details,
        fallback_decompose_missing_facts=args.fallback_decompose_missing_facts,
        max_fallback_claims=args.max_fallback_claims,
    )

    chat_client = SU8StructuredChatClient(
        model=args.su8_model,
        base_url=args.su8_base_url,
        api_key=get_su8_api_key(),
        strict_schema=config.strict_schema,
        allow_json_object_fallback=config.allow_json_object_fallback,
    )
    verifier = SU8NLIVerifier(chat_client, config)
    decomposer = SU8FallbackClaimDecomposer(chat_client, config)
    similarity = None if args.disable_similarity else OllamaSemanticSimilarityCalculator(
        model=args.embedding_model,
        base_url=args.ollama_base_url,
        timeout=args.similarity_timeout,
        max_retries=args.similarity_max_retries,
        retry_sleep=args.similarity_retry_sleep,
        allow_fallback=args.allow_similarity_fallback,
    )
    ac_calculator = AnswerCorrectnessCalculator(verifier, decomposer, similarity, config)
    ra_calculator = RelationAwareCalculator(chat_client, config)

    file_contexts = load_context_file(args.context_file, args.context_file_max_chars)

    method = (args.method or "").strip().lower()
    run_context_metrics = method != "direct" or args.run_context_metrics_for_direct

    print("========== Unified Metrics Config ==========")
    print("Input:", input_path)
    print("Output:", output_path)
    print("Method:", args.method)
    print("Run context metrics:", run_context_metrics)
    print("Fallback decompose missing facts:", config.fallback_decompose_missing_facts)
    print("Skip existing metric fields:", args.skip_existing_metrics)
    print("SU8 model:", args.su8_model)
    print("SU8 base URL:", args.su8_base_url)
    print("Ollama embedding:", "disabled" if args.disable_similarity else args.embedding_model)
    print("Similarity fallback:", args.allow_similarity_fallback)
    print("Similarity retries:", args.similarity_max_retries)
    print("===========================================")

    total = count_items(data)
    processed = 0
    skipped_existing = 0
    errors = 0
    fallback_used = 0
    skipped_metric_counts = {"answer_correctness": 0, "ra_fc": 0, "context_metrics": 0}

    try:
        try:
            from tqdm import tqdm
            iterator = tqdm(list(iter_items(data)), total=total)
        except ImportError:
            iterator = iter_items(data)

        for key, item in iterator:
            if not isinstance(item, dict):
                continue
            if args.limit is not None and processed >= args.limit:
                break
            if item.get("all_metrics_eval_status") == "success" and not args.overwrite:
                skipped_existing += 1
                continue

            question = get_question(item)
            answer = get_generated_answer(item)
            gold_answer = get_gold_answer(item)
            generated_facts = get_generated_facts(item)
            gold_facts = get_gold_facts(item)
            contexts = extract_contexts(item)
            if not contexts and file_contexts:
                contexts = file_contexts

            # -----------------------------------------------------------------
            # FIX: outer-scope fallback. Decompose format_answer into generated
            # facts ONCE, here, before any calculator runs, and write it back
            # into the item. This makes AC / RA-FC / context metrics all share
            # the same non-empty fact list, instead of the old behaviour where
            # the fallback inside AnswerCorrectnessCalculator was invisible to
            # RA-FC and faithfulness.
            # -----------------------------------------------------------------
            generated_facts, used_fallback_generated = await ensure_generated_facts(
                item=item,
                answer=answer,
                generated_facts=generated_facts,
                decomposer=decomposer,
                config=config,
            )
            if used_fallback_generated:
                fallback_used += 1
            item["used_fallback_generated_facts"] = used_fallback_generated

            item["answer_rate"] = 1.0 if is_non_empty_answer(answer) else 0.0
            item["ans_r"] = item["answer_rate"]
            item["answer_rate_status"] = "answered" if item["answer_rate"] == 1.0 else "empty_or_abstained"
            item["eval_generated_fact_count"] = len(generated_facts)
            item["eval_gold_fact_count"] = len(gold_facts)
            item["eval_context_count"] = len(contexts)
            if contexts:
                ctx_tok = sum(count_tokens(c) for c in contexts)
                item["retrieved_context_token_count"] = item.get("retrieved_context_token_count") or ctx_tok
                item["ctx_tok"] = item["retrieved_context_token_count"]

            print(f"\n[Item {key}] Ans.R={item['answer_rate']} gen_facts={len(generated_facts)} gold_facts={len(gold_facts)} contexts={len(contexts)}", flush=True)

            try:
                if not question:
                    raise ValueError("missing question")

                if args.skip_existing_metrics and has_existing_answer_correctness(item):
                    skipped_metric_counts["answer_correctness"] += 1
                    item.setdefault("answer_correctness_status", "reused_existing")
                else:
                    ac_scores = await ac_calculator.calculate(
                        answer=answer,
                        reference=gold_answer,
                        generated_facts=generated_facts,
                        gold_facts=gold_facts,
                    )
                    item.update(ac_scores)

                if args.skip_existing_metrics and has_existing_ra_fc(item):
                    skipped_metric_counts["ra_fc"] += 1
                    normalize_existing_ra_aliases(item)
                    item.setdefault("relation_eval_status", "reused_existing")
                else:
                    ra_scores = await ra_calculator.calculate(
                        question=question,
                        gold_answer=gold_answer,
                        pred_answer=answer,
                        gold_facts=gold_facts,
                        pred_facts=generated_facts,
                    )
                    item.update(ra_scores)

                if run_context_metrics:
                    if args.skip_existing_metrics and has_existing_context_metrics(item):
                        skipped_metric_counts["context_metrics"] += 1
                        item.setdefault("faithfulness_status", "reused_existing")
                        item.setdefault("context_recall_status", "reused_existing")
                    else:
                        context_scores = await calculate_context_metrics_shared(
                            verifier=verifier,
                            answer=answer,
                            generated_facts=generated_facts,
                            gold_answer=gold_answer,
                            gold_facts=gold_facts,
                            contexts=contexts,
                            config=config,
                        )
                        item.update(context_scores)
                else:
                    # Direct baseline: context-grounded metrics are intentionally undefined.
                    # Do not overwrite existing values when the user is explicitly reusing metrics.
                    if not (args.skip_existing_metrics and has_existing_context_metrics(item)):
                        item["faithfulness"] = None
                        item["context_recall"] = None
                        if config.save_details:
                            item["faithfulness_details"] = {"reason": "skipped_for_direct_method"}
                            item["context_recall_details"] = {"reason": "skipped_for_direct_method"}

                item["all_metrics_eval_status"] = "success"
                item["all_metrics_eval_model"] = args.su8_model
                item["all_metrics_eval_base_url"] = args.su8_base_url
                processed += 1

            except Exception as e:
                errors += 1
                item["all_metrics_eval_status"] = "failed"
                item["all_metrics_eval_error"] = str(e)
                print(f"  [Error] item={key}: {e}", flush=True)

            if args.save_every > 0 and processed > 0 and processed % args.save_every == 0:
                save_json(output_path, data)
                print(f"[Checkpoint] Saved {output_path}", flush=True)

            if args.sleep > 0:
                await asyncio.sleep(args.sleep)

    finally:
        if similarity is not None:
            await similarity.close()
        await chat_client.close()

    print(f"[Save] {output_path}")
    save_json(output_path, data)

    direct_summary = None
    if args.direct_summary_path and Path(args.direct_summary_path).exists():
        direct_summary = load_json(args.direct_summary_path)

    summary = build_summary(
        data=data,
        result_path=output_path,
        method=args.method,
        method_name=args.method_name,
        direct_summary=direct_summary,
    )

    summary_path = Path(args.summary_output) if args.summary_output else output_path.with_name(output_path.stem + "_summary.json")
    save_json(summary_path, summary)
    print(f"[Summary] Saved {summary_path}")

    print("[Done]")
    print(f"  total: {total}")
    print(f"  processed: {processed}")
    print(f"  fallback_decomposed_generated_facts: {fallback_used}")
    print(f"  skipped_existing_items: {skipped_existing}")
    print(f"  skipped_existing_metrics: {skipped_metric_counts}")
    print(f"  errors: {errors}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute Ans.R, Answer Correctness, RA-FC, and optional context metrics in one pass.")
    parser.add_argument("--input", required=True, help="Input JSON generated by generate_closed_book_answer_and_facts_su8_total.py.")
    parser.add_argument("--output", default=None, help="Output JSON with all metric fields. Default: *_all_metrics.json")
    parser.add_argument("--method", choices=["direct", "maxctx", "ragflow", "other"], default="direct")
    parser.add_argument("--method_name", default=None)

    parser.add_argument("--su8_model", default=DEFAULT_SU8_MODEL)
    parser.add_argument("--su8_base_url", default=DEFAULT_SU8_BASE_URL)
    parser.add_argument("--ollama_base_url", default=DEFAULT_OLLAMA_BASE_URL)
    parser.add_argument("--embedding_model", default=DEFAULT_EMBEDDING_MODEL)
    parser.add_argument("--disable_similarity", action="store_true", help="Disable Ollama embedding similarity and set similarity_score to 0.")
    parser.add_argument("--similarity_timeout", type=float, default=180.0, help="Timeout seconds for each Ollama embedding request.")
    parser.add_argument("--similarity_max_retries", type=int, default=5, help="Retry count for Ollama embedding before failing the item.")
    parser.add_argument("--similarity_retry_sleep", type=float, default=3.0, help="Base retry sleep seconds for Ollama embedding.")
    parser.add_argument("--allow_similarity_fallback", action="store_true", help="Allow fallback to word-overlap similarity if Ollama embedding fails. Not recommended for final experiments.")

    parser.add_argument("--weights", nargs=2, type=float, default=[0.75, 0.25], metavar=("FACTUAL", "SIMILARITY"))
    parser.add_argument("--beta", type=float, default=1.0, help="F-beta beta for answer correctness factual score.")
    parser.add_argument("--ra_beta", type=float, default=1.0, help="F-beta beta for RA-FC.")
    parser.add_argument("--eval_chunk_size", type=int, default=3000)
    parser.add_argument("--eval_chunk_overlap", type=int, default=300)
    parser.add_argument("--nli_batch_size", type=int, default=5)
    parser.add_argument("--max_retries", type=int, default=3)
    parser.add_argument("--sleep_between_retries", type=float, default=2.0)
    parser.add_argument("--disable_strict_schema", action="store_true")
    parser.add_argument("--disable_schema_fallback", action="store_true")
    parser.add_argument("--no_details", action="store_true")

    parser.add_argument("--fallback_decompose_missing_facts", action="store_true",
                        help="If format_answer_atomic_facts is empty but the answer is non-empty, decompose the answer with SU8 so ALL metrics (AC / RA-FC / faithfulness) get the facts.")
    parser.add_argument("--max_fallback_claims", type=int, default=12)

    parser.add_argument("--context_file", default=None, help="Fallback context file if item has no retrieved_contexts.")
    parser.add_argument("--context_file_max_chars", type=int, default=None)
    parser.add_argument("--run_context_metrics_for_direct", action="store_true")

    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--save_every", type=int, default=10)
    parser.add_argument("--sleep", type=float, default=0.0)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--skip_existing_metrics", action="store_true", help="Reuse per-item metric fields that already exist instead of recomputing them. Useful when RA-FC was already computed by relation_aware_factual_correctness_su8.py.")

    parser.add_argument("--summary_output", default=None)
    parser.add_argument("--direct_summary_path", default=None, help="Direct summary JSON used to compute DeltaRA for non-Direct methods.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    asyncio.run(evaluate_file(args))


if __name__ == "__main__":
    main()
