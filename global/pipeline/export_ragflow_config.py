#!/usr/bin/env python3
"""Export the RAGFlow retrieval configuration used by the global experiments.

Dumps (API keys are never written):
  ragflow_config/datasets.json   chunk/embedding/parser config of every dataset
  ragflow_config/documents.json  per-dataset document lists (name, chunks, status)
  ragflow_config/chats.json      chat-assistant configs (LLM, prompt, retrieval params)
  ragflow_config/EXPORT_META.json  export date, RAGFlow version, base URL host

Usage:
    export RAGFLOW_BASE_URL=http://localhost:9380/api/v1   # default
    export RAGFLOW_API_KEY=...                              # required
    python3 export_ragflow_config.py
"""
from __future__ import annotations

import json
import os
import sys
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

BASE = os.environ.get("RAGFLOW_BASE_URL", "http://localhost:9380/api/v1").rstrip("/")
KEY = os.environ.get("RAGFLOW_API_KEY", "")
OUT = Path(__file__).resolve().parent / "ragflow_config"


def get(path: str, retries: int = 3) -> dict:
    req = urllib.request.Request(
        f"{BASE}{path}", headers={"Authorization": f"Bearer {KEY}"})
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read().decode("utf-8"))
        except Exception as e:  # flaky gateway: retry
            last = e
    raise last


def get_paged(path: str, key: str) -> list:
    """Fetch all pages of a list endpoint (small pages; the server 502s on
    large page_size)."""
    items, page = [], 1
    while True:
        data = get(f"{path}{'&' if '?' in path else '?'}page={page}&page_size=20")
        chunk = data.get("data", [])
        if isinstance(chunk, dict):
            chunk = chunk.get(key, [])
        if not chunk:
            return items
        items.extend(chunk)
        if len(chunk) < 20:
            return items
        page += 1


def main() -> None:
    if not KEY:
        sys.exit("RAGFLOW_API_KEY is not set")
    OUT.mkdir(exist_ok=True)

    datasets = get_paged("/datasets", "datasets")
    (OUT / "datasets.json").write_text(
        json.dumps(datasets, ensure_ascii=False, indent=1), encoding="utf-8")

    documents = {}
    for ds in datasets:
        docs = get_paged(f"/datasets/{ds['id']}/documents", "docs")
        documents[ds["id"]] = {
            "dataset_name": ds.get("name"),
            "docs": [
                {k: d.get(k) for k in
                 ("id", "name", "chunk_count", "token_count", "status", "run", "progress")}
                for d in docs
            ],
        }
    (OUT / "documents.json").write_text(
        json.dumps(documents, ensure_ascii=False, indent=1), encoding="utf-8")

    chats = get_paged("/chats", "chats")
    (OUT / "chats.json").write_text(
        json.dumps(chats, ensure_ascii=False, indent=1), encoding="utf-8")

    meta = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "ragflow_base_url_host": BASE.split("/api")[0],
        "ragflow_image": "infiniflow/ragflow:v0.25.0 (recorded at export time)",
        "n_datasets": len(datasets),
        "n_chats": len(chats),
        "note": "API keys and secrets are excluded. Rebuild requires the same "
                "RAGFlow version and the embedding model named in datasets.json.",
    }
    (OUT / "EXPORT_META.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(meta, indent=1))


if __name__ == "__main__":
    main()
