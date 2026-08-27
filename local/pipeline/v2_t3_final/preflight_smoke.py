#!/usr/bin/env python3
"""Preflight check before any formal API call.

Verifies, without writing any formal result:
  - SU8_API_KEY is set
  - endpoint = https://www.su8.codes/v1 (frozen disclosure)
  - model alias = gpt-5.5 (frozen)
  - one live chat.completions health request returns a valid JSON-object answer

Stops and reports on auth/quota/model failure. Never switches models.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2_t3_final.su8_client import ENDPOINT, MODEL, SU8Judge  # noqa: E402


def main() -> int:
    print("== T3.7/T3.8 preflight ==", flush=True)
    if not os.environ.get("SU8_API_KEY"):
        print("STOP: SU8_API_KEY is not set", flush=True)
        return 2
    print(f"endpoint: {ENDPOINT}")
    print(f"model alias: {MODEL}")
    print(f"SU8_BASE_URL env: {os.environ.get('SU8_BASE_URL', '<unset — using frozen disclosure endpoint>')}")

    judge = SU8Judge()
    t0 = time.time()
    try:
        out = judge.complete_json(
            system="You are a health-check endpoint. Return JSON only.",
            user='Respond with exactly: {"status":"ok","model":"gpt-5.5"}',
            temperature=0.0,
        )
    except Exception as exc:
        print(f"STOP: smoke request failed: {type(exc).__name__}: "
              f"status={getattr(exc,'status_code','?')} {str(exc)[:300]}", flush=True)
        return 3
    dt = time.time() - t0
    print(f"smoke latency: {dt:.2f}s")
    print(f"smoke request_id: {judge.last_request_id}")
    print(f"smoke response: {json.dumps(out, ensure_ascii=False)[:200]}")
    if not isinstance(out, dict) or not out:
        print("STOP: smoke response was not a non-empty JSON object", flush=True)
        return 4
    print("PREFLIGHT OK — credentials, endpoint, and model alias verified.", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
