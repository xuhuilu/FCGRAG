"""Shared helpers for non-Python lightweight source analyzers.

These helpers intentionally avoid language-specific parser dependencies. They
are a first-pass structural extractor for benchmark generation when the call
graph already comes from an external tool.
"""
from __future__ import annotations

import re
from collections.abc import Iterable


IDENT = r"[A-Za-z_$][A-Za-z0-9_$]*"
QUAL_IDENT = rf"{IDENT}(?:\s*\.\s*{IDENT})*"


def line_no(source: str, pos: int) -> int:
    return source.count("\n", 0, max(0, pos)) + 1


def strip_comments(source: str) -> str:
    """Remove comments while preserving newlines enough for line numbers."""
    source = re.sub(r"/\*.*?\*/", lambda m: "\n" * m.group(0).count("\n"), source, flags=re.S)
    source = re.sub(r"//.*", "", source)
    return source


def short_call_name(name: str) -> str:
    return re.sub(r"\s+", "", name).split(".")[-1]


def collect_call_matches(source: str, skip: Iterable[str]) -> list[tuple[str, int, str]]:
    """Return (short_name, line, arg_text) for call-like tokens."""
    skip_set = set(skip)
    calls: list[tuple[str, int, str]] = []
    pattern = re.compile(rf"\b(?:new\s+)?({QUAL_IDENT})\s*\(([^()]*)\)", re.S)
    for match in pattern.finditer(source):
        raw = match.group(1)
        short = short_call_name(raw)
        if short in skip_set:
            continue
        prefix = source[max(0, match.start() - 32):match.start()]
        if re.search(r"\bfunc\s+$", prefix) or re.search(r"\bfunction\s+$", prefix):
            continue
        # A function/method declaration also looks like "name(...) {" in these
        # lightweight regexes. It is not an executable call site.
        if re.match(r"\s*\{", source[match.end():]):
            continue
        calls.append((short, line_no(source, match.start()), " ".join(match.group(2).split())))
    return calls


def outer_call(expr: str, skip: Iterable[str]) -> str:
    match = re.search(rf"\b(?:new\s+)?({QUAL_IDENT})\s*\(", expr)
    if not match:
        return ""
    short = short_call_name(match.group(1))
    return "" if short in set(skip) else short


def collect_block_calls(block: str, skip: Iterable[str]) -> list[str]:
    return [name for name, _, _ in collect_call_matches(block, skip)]


def first_braced_block(source: str, open_brace_pos: int) -> tuple[str, int] | None:
    """Return (body, close_pos) for the braced block starting at open_brace_pos."""
    if open_brace_pos < 0 or open_brace_pos >= len(source) or source[open_brace_pos] != "{":
        return None
    depth = 0
    for pos in range(open_brace_pos, len(source)):
        ch = source[pos]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                return source[open_brace_pos + 1:pos], pos
    return None


def split_top_level_args(text: str) -> list[str]:
    args: list[str] = []
    cur: list[str] = []
    depth = 0
    for ch in text:
        if ch in "([{":
            depth += 1
        elif ch in ")]}" and depth > 0:
            depth -= 1
        if ch == "," and depth == 0:
            arg = "".join(cur).strip()
            if arg:
                args.append(arg)
            cur = []
        else:
            cur.append(ch)
    arg = "".join(cur).strip()
    if arg:
        args.append(arg)
    return args
