#!/usr/bin/env python3
"""Shared SU8 (gpt-5.5) client for T3.7 selective reevaluation and T3.8 reruns.

Frozen configuration (from e1_freeze_manifest.json model_endpoint_disclosure):
  model     = gpt-5.5
  endpoint  = https://www.su8.codes/v1
  api key   = $SU8_API_KEY
  temperature = 0.0
  generation: json_mode=True, max_tokens=1800  (DeepSeekJSONLLM frozen params)
  judge:       response_format=json_object, max_tokens=1400  (DeepSeekJudge frozen params)

Retry policy is identical to run_three_methods_clean_v1_en (inner retries =
$FCG_API_MAX_RETRIES default 6; transient 408/409/425/429/5xx; honors
Retry-After cap 120s; exponential 2^n cap 60s). Driver-level retry (3) wraps
generation in rerun_t3_8, mirroring error_repair_gpt55_su8.py.

This module does NOT mutate the frozen evaluator/generator source.
"""
from __future__ import annotations

import os
import re
import time
from typing import Any

import httpx
from openai import APIConnectionError, APITimeoutError, OpenAI
from urllib.parse import urlparse

# Reuse the frozen retry helpers verbatim (single source of truth).
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from run_three_methods_clean_v1_en import _is_retryable_api_error, _retry_after_seconds  # noqa: E402

ENDPOINT = "https://www.su8.codes/v1"
MODEL = os.environ.get("SU8_MODEL", "gpt-5.5")
GEN_MAX_TOKENS = 1800
JUDGE_MAX_TOKENS = 1400

# Read timeout: gpt-5.5 reasoning calls can exceed 60s at 1400-1800 max_tokens.
# Transport-level setting only — does not affect model, endpoint, auth,
# temperature, max_tokens, or prompts.
READ_TIMEOUT = float(os.environ.get("SU8_READ_TIMEOUT", "180"))


def _retryable(exc: Exception) -> bool:
    """Frozen retryable set plus client-side transport timeouts/disconnects.

    APITimeoutError/APIConnectionError carry no HTTP status, so the frozen
    status-code helper treats them as fatal; they are transient by nature.
    """
    return _is_retryable_api_error(exc) or isinstance(
        exc, (APITimeoutError, APIConnectionError)
    )


def _http_proxy() -> str | None:
    """Pick an http(s)-scheme proxy from env; ignore socks (httpx can't use it).

    Network path only — does not change model, endpoint, or auth.
    """
    for k in ("HTTPS_PROXY", "https_proxy", "HTTP_PROXY", "http_proxy"):
        v = os.environ.get(k)
        if v and urlparse(v).scheme in ("http", "https"):
            return v
    return None


def _client() -> OpenAI:
    api_key = os.environ.get("SU8_API_KEY")
    if not api_key:
        raise RuntimeError("SU8_API_KEY is required for the frozen gpt-5.5 endpoint")
    proxy = _http_proxy()
    # trust_env=False so httpx ignores socks ALL_PROXY (unsupported by httpx);
    # pass an explicit http proxy when one is configured (matches curl behavior).
    http_client = httpx.Client(
        proxy=proxy,
        trust_env=False,
        timeout=httpx.Timeout(READ_TIMEOUT, connect=30.0),
    )
    # Cloudflare WAF in front of SU8 blocks requests whose User-Agent contains
    # "OpenAI/Python" (the openai SDK default). Override the UA so the request
    # reaches the same frozen endpoint/model/auth. This is a transport-level
    # adaptation only; it does not change model, endpoint, auth, temperature,
    # max_tokens, or the prompt.
    return OpenAI(
        api_key=api_key,
        base_url=ENDPOINT,
        timeout=READ_TIMEOUT,
        http_client=http_client,
        default_headers={"User-Agent": "curl/8.5.0"},
    )


class SU8LLM:
    """Generator client (LLMClient protocol-compatible)."""

    def __init__(self, model: str = MODEL):
        self.client = _client()
        self.model = model
        self.last_request_id: str | None = None

    def complete(
        self, system: str, user: str, temperature: float = 0.0, json_mode: bool = False
    ) -> str:
        kwargs: dict[str, Any] = {}
        if json_mode:
            kwargs["response_format"] = {"type": "json_object"}
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        max_retries = int(os.environ.get("FCG_API_MAX_RETRIES", "6"))
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=GEN_MAX_TOKENS,
                    messages=messages,
                    **kwargs,
                )
                self.last_request_id = resp.id if hasattr(resp, "id") else None
                return resp.choices[0].message.content or ""
            except Exception as exc:
                if not _retryable(exc) or attempt >= max_retries:
                    raise
                wait = _retry_after_seconds(exc, attempt)
                print(
                    f"  [gen-retry] {self.model} attempt {attempt}/{max_retries} "
                    f"{type(exc).__name__} status={getattr(exc, 'status_code', '?')}: "
                    f"sleeping {wait:.0f}s",
                    flush=True,
                )
                time.sleep(wait)
        return ""  # unreachable


class SU8Judge:
    """Judge client (DeepSeekJudge-compatible)."""

    def __init__(self, model: str = MODEL):
        self.client = _client()
        self.model = model
        self.last_request_id: str | None = None

    def complete_json(self, system: str, user: str, temperature: float = 0.0) -> dict[str, Any]:
        import json

        max_retries = int(os.environ.get("FCG_API_MAX_RETRIES", "6"))
        for attempt in range(1, max_retries + 1):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=temperature,
                    max_tokens=JUDGE_MAX_TOKENS,
                    response_format={"type": "json_object"},
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                self.last_request_id = resp.id if hasattr(resp, "id") else None
                text = resp.choices[0].message.content or "{}"
                try:
                    return json.loads(text)
                except json.JSONDecodeError:
                    return {}
            except Exception as exc:
                if not _retryable(exc) or attempt >= max_retries:
                    raise
                wait = _retry_after_seconds(exc, attempt)
                print(
                    f"  [judge-retry] {self.model} attempt {attempt}/{max_retries} "
                    f"{type(exc).__name__} status={getattr(exc, 'status_code', '?')}: "
                    f"sleeping {wait:.0f}s",
                    flush=True,
                )
                time.sleep(wait)
        return {}
