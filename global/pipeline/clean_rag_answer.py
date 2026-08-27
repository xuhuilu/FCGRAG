#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Deterministic answer post-cleaning for RAG evaluation.

Fixes vs the previous version:
  1. CRITICAL: citation-bracket removal no longer destroys code answers.
     `argv[0]`, `list[0]`, `matrix[0, 1]`, `a[0][1]` are preserved; only
     free-standing [0] / [ID:0] / [ID:0, ID:7] citation markers are removed
     (guarded by a negative lookbehind on identifier chars, ')' and ']').
  2. Leading meta prefaces are stripped in a LOOP, so stacked prefaces like
     "Based on the retrieved context, according to the code, ..." are fully
     removed instead of only the first layer.
  3. Inline meta phrases are removed together with their preposition
     ("in the retrieved context") to avoid dangling "defined in ." grammar.
  4. Every cleaning rule is named; clean_with_audit() returns which rules
     fired, so you can report "cleaning modified X/N answers" in the paper
     appendix and eyeball-audit exactly the changed items.
  5. A CLI reprocesses EXISTING output JSONs from raw_ragflow_answer, so the
     same cleaner applies uniformly to Doc-RAG / Code-RAG / Report-RAG /
     Int-RAG without re-calling RAGFlow. Items whose format_answer changed
     are flagged with needs_fact_redecomposition=True; re-run fact
     decomposition and metrics only for those.

Usage (library):
    from clean_rag_answer import clean_rag_answer_for_eval
    cleaned = clean_rag_answer_for_eval(raw_answer)

Usage (retroactive reprocess of an existing run):
    python clean_rag_answer.py --input outputs/.../xxx.json \
        --output outputs/.../xxx_recleaned.json \
        --report outputs/.../xxx_cleaning_report.json
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

# =========================
# Abstention normalization
# =========================

IDK_SET = {
    "i don't know",
    "i do not know",
    "unknown",
    "not sure",
    "不知道",
    "无法确定",
    "不确定",
}


def _is_idk(text: str) -> bool:
    return text.strip().lower().rstrip(".。!").strip() in IDK_SET


# =========================
# Cleaning rules (named, ordered)
# =========================

# --- Rule: citation brackets -------------------------------------------------
# Negative lookbehind (?<![\w)\]]) ensures we do NOT match brackets attached to
# an identifier, a closing paren, or a closing bracket:
#   argv[0]      -> preserved ('v' is \w)
#   func()[0]    -> preserved (')')
#   a[0][1]      -> preserved (']')
#   "... ports. [0]"        -> removed
#   "... default. [ID:0, ID:7]" -> removed
_CITATION_RE = re.compile(
    r"(?<![\w)\]])\s*\[(?:ID:\s*)?\d+(?:\s*,\s*(?:ID:\s*)?\d+)*\]",
    flags=re.I,
)

# --- Rule: figure labels -----------------------------------------------------
_FIG_RE = re.compile(r"(?<![\w)\]])\bFig\.\s*\d+\b\.?", flags=re.I)

# --- Rule: leading meta prefaces (applied in a loop) -------------------------
_META_NOUN = (
    r"(?:retrieved|provided|given|above)?\s*"
    r"(?:context|contexts|evidence|documentation|docs?|report|reports|"
    r"community[- ]reports?|source[- ]code|code|knowledge(?:\s*base)?|dataset|graph)"
)

_LEADING_PREFACE_RES = [
    # "Based on the retrieved context, ..." / "According to the code: ..."
    re.compile(
        rf"^\s*(?:Based on|According to|From|Per|Drawing on|Looking at)\s+"
        rf"(?:the\s+|this\s+)?{_META_NOUN}\s*[,.:;-]?\s*",
        flags=re.I,
    ),
    # "The retrieved context shows that ..." / "The code indicates ..."
    re.compile(
        rf"^\s*The\s+{_META_NOUN}\s+"
        r"(?:shows?|states?|indicates?|suggests?|describes?|says?|reveals?|mentions?|confirms?)\s+"
        r"(?:that\s+)?",
        flags=re.I,
    ),
    # "In the provided context, ..." / "Within the retrieved documentation, ..."
    re.compile(
        rf"^\s*(?:In|Within|From)\s+(?:the\s+)?{_META_NOUN}\s*[,:;]\s*",
        flags=re.I,
    ),
]

# --- Rule: inline meta phrases (preposition + phrase removed together) -------
_INLINE_META_WITH_PREP_RE = re.compile(
    rf"\s*\b(?:in|from|within|per|of|by)\s+(?:the\s+)?"
    r"(?:retrieved|provided|given)\s+"
    r"(?:context|contexts|evidence|documentation|reports?)\b",
    flags=re.I,
)
# Bare leftovers ("the retrieved context") anywhere in the sentence.
_INLINE_META_BARE_RE = re.compile(
    r"\b(?:the\s+)?(?:retrieved|provided)\s+"
    r"(?:context|contexts|evidence|documentation)\b",
    flags=re.I,
)

# --- Rule: trailing source phrases -------------------------------------------
_TRAILING_META_RE = re.compile(
    rf"\s*[,;]?\s*(?:as\s+(?:shown|described|stated|indicated|documented))\s+"
    rf"(?:in|by)\s+(?:the\s+)?{_META_NOUN}\s*\.?\s*$",
    flags=re.I,
)

# --- Rule: leading bullet ------------------------------------------------------
_LEADING_BULLET_RE = re.compile(r"^\s*[-*•]\s+")


def clean_with_audit(text: str) -> tuple[str, list[str]]:
    """
    Clean formatting artifacts from a RAG answer.

    Returns (cleaned_text, fired_rule_names). Factual content is never
    "corrected"; only grounding/citation boilerplate is removed.
    """
    fired: list[str] = []
    text = str(text or "").strip()
    if not text:
        return "", fired

    original = text
    text = re.sub(r"\s+", " ", text).strip()
    if text != original:
        fired.append("whitespace_normalized")

    new = _CITATION_RE.sub("", text)
    if new != text:
        fired.append("citation_brackets")
    text = new

    new = _FIG_RE.sub("", text)
    if new != text:
        fired.append("figure_labels")
    text = new

    # Loop leading prefaces until stable (handles stacked prefaces).
    for _ in range(5):
        changed = False
        for i, pattern in enumerate(_LEADING_PREFACE_RES):
            new = pattern.sub("", text)
            if new != text:
                fired.append(f"leading_preface_{i}")
                text = new
                changed = True
        if not changed:
            break

    new = _TRAILING_META_RE.sub(".", text)
    if new != text:
        fired.append("trailing_meta")
    text = new

    new = _INLINE_META_WITH_PREP_RE.sub("", text)
    if new != text:
        fired.append("inline_meta_with_preposition")
    text = new

    new = _INLINE_META_BARE_RE.sub("", text)
    if new != text:
        fired.append("inline_meta_bare")
    text = new

    new = _LEADING_BULLET_RE.sub("", text)
    if new != text:
        fired.append("leading_bullet")
    text = new

    # Tidy punctuation artifacts left by removals: " ,", " .", double commas.
    text = re.sub(r"\s+([,.;:])", r"\1", text)
    text = re.sub(r"([,;:])\1+", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^[,.;:\s]+", "", text)

    # NOTE: we intentionally do NOT capitalize the first character after
    # stripping a preface — the answer may legitimately start with a code
    # identifier such as `create_agent`, and changing case would corrupt it.

    if not text or _is_idk(text):
        return "I don't know.", fired

    return text, fired


def clean_rag_answer_for_eval(text: str) -> str:
    """Drop-in replacement keeping the original str -> str signature."""
    cleaned, _ = clean_with_audit(text)
    return cleaned


# =========================
# Retroactive reprocessing CLI
# =========================

def _iter_items(data: Any):
    if isinstance(data, list):
        for idx, item in enumerate(data):
            yield idx, item
    elif isinstance(data, dict):
        items = data.get("items") if isinstance(data.get("items"), list) else None
        if items is not None:
            for idx, item in enumerate(items):
                yield idx, item
        else:
            for key, item in data.items():
                yield key, item


def reprocess_file(input_path: Path, output_path: Path, report_path: Path | None) -> None:
    data = json.loads(input_path.read_text(encoding="utf-8"))

    total = 0
    changed = 0
    rule_counter: dict[str, int] = {}
    changed_items: list[dict[str, Any]] = []

    for key, item in _iter_items(data):
        if not isinstance(item, dict):
            continue

        # Prefer the preserved raw answer; fall back to current format_answer.
        raw = item.get("raw_ragflow_answer")
        if raw is None:
            raw = item.get("format_answer") or item.get("generated_answer") or ""
        if not str(raw).strip():
            continue

        total += 1
        old_format = str(item.get("format_answer") or "")
        cleaned, fired = clean_with_audit(str(raw))

        for rule in fired:
            rule_counter[rule] = rule_counter.get(rule, 0) + 1

        if cleaned != old_format:
            changed += 1
            item["format_answer_pre_reclean"] = old_format
            item["format_answer"] = cleaned
            item["generated_answer"] = cleaned
            item["needs_fact_redecomposition"] = True
            item["cleaning_rules_fired"] = fired
            changed_items.append({
                "index": key,
                "question": (item.get("question") or "")[:120],
                "old": old_format[:300],
                "new": cleaned[:300],
                "rules": fired,
            })
        else:
            item["needs_fact_redecomposition"] = False

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    report = {
        "input": str(input_path),
        "output": str(output_path),
        "items_with_answers": total,
        "answers_changed_by_cleaning": changed,
        "changed_fraction": round(changed / total, 4) if total else None,
        "rule_fire_counts": rule_counter,
        "changed_items": changed_items,
    }
    print(json.dumps({k: v for k, v in report.items() if k != "changed_items"},
                     ensure_ascii=False, indent=2))

    if report_path:
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"Cleaning report saved: {report_path}")

    print(
        "\nNext step: re-run atomic fact decomposition and metrics ONLY for "
        "items with needs_fact_redecomposition=True, for every method, so the "
        "cleaner is applied uniformly across Doc-RAG / Code-RAG / Report-RAG / "
        "Int-RAG."
    )


# =========================
# Self-test
# =========================

_SELF_TESTS = [
    # (input, expected_substring_preserved_or_output)
    ("Based on the retrieved context, the default port is 11434.",
     "the default port is 11434."),
    ("Based on the retrieved context, according to the code, use create_agent.",
     "use create_agent."),
    ("The default model is stored in argv[0] and config[0, 1].",
     "The default model is stored in argv[0] and config[0, 1]."),
    ("Use a[0][1] to access it. [ID:0, ID:7]",
     "Use a[0][1] to access it."),
    ("The value is defined in the retrieved context and defaults to 8.",
     "The value is defined and defaults to 8."),
    ("The retrieved context shows that OLLAMA_HOST controls the bind address.",
     "OLLAMA_HOST controls the bind address."),
    ("i do not know", "I don't know."),
    ("The port is 11434, as shown in the documentation.",
     "The port is 11434."),
]


def _run_self_tests() -> None:
    failures = 0
    for src, expected in _SELF_TESTS:
        got = clean_rag_answer_for_eval(src)
        ok = got == expected
        if not ok:
            failures += 1
            print(f"FAIL\n  in : {src}\n  exp: {expected}\n  got: {got}")
    print(f"Self-tests: {len(_SELF_TESTS) - failures}/{len(_SELF_TESTS)} passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", help="Existing output JSON to reprocess")
    parser.add_argument("--output", help="Path for the recleaned JSON")
    parser.add_argument("--report", default=None, help="Optional cleaning report JSON")
    parser.add_argument("--self_test", action="store_true", help="Run built-in tests")
    args = parser.parse_args()

    if args.self_test or not args.input:
        _run_self_tests()
    if args.input:
        if not args.output:
            raise SystemExit("--output is required with --input")
        reprocess_file(Path(args.input), Path(args.output),
                       Path(args.report) if args.report else None)
