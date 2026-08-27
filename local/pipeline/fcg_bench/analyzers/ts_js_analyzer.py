"""Lightweight TypeScript/JavaScript source analyzer."""
from __future__ import annotations

import re

from ..ast_analyzer import AssignInfo, BranchInfo, CallSite, FunctionAST, TryInfo
from .lightweight import (
    collect_block_calls,
    collect_call_matches,
    first_braced_block,
    line_no,
    outer_call,
    split_top_level_args,
    strip_comments,
)


_SKIP_CALLS = {
    "if", "for", "while", "switch", "catch", "function", "return", "typeof",
    "await", "async", "constructor", "super",
}
_LOG_NAMES = {"log", "debug", "trace", "warn", "error", "info"}


def _extract_args(source: str) -> list[str]:
    patterns = [
        r"\b(?:async\s+)?function\s+[A-Za-z_$][A-Za-z0-9_$]*\s*\(([^)]*)\)",
        r"\b[A-Za-z_$][A-Za-z0-9_$]*\s*\(([^)]*)\)\s*\{",
        r"\(([^)]*)\)\s*=>",
    ]
    for pattern in patterns:
        match = re.search(pattern, source, re.S)
        if match:
            args: list[str] = []
            for arg in split_top_level_args(match.group(1)):
                name = re.split(r"[:=?]", arg, maxsplit=1)[0].strip()
                name = name.lstrip("...").strip()
                if re.fullmatch(r"[A-Za-z_$][A-Za-z0-9_$]*", name):
                    args.append(name)
            return args
    return []


def _mark_call_context(source: str, info: FunctionAST) -> None:
    branch_spans = [(b.start_line, b.end_line) for b in info.branches if b.start_line and b.end_line]
    try_spans = [(t.start_line, t.end_line) for t in info.tries if t.start_line and t.end_line]
    for name, lineno, arg_text in collect_call_matches(source, _SKIP_CALLS):
        in_branch = any(start <= lineno <= end for start, end in branch_spans)
        in_except = any(start <= lineno <= end for start, end in try_spans)
        info.call_sites.append(CallSite(name, lineno, arg_text, in_branch, in_except))


def analyze_ts_js_function(qualname: str, source: str) -> FunctionAST:
    info = FunctionAST(qualname=qualname)
    source = strip_comments(source)
    if not source.strip():
        return info

    info.arg_names = _extract_args(source)

    for match in re.finditer(r"\bif\s*\((.*?)\)\s*\{", source, re.S):
        block = first_braced_block(source, match.end() - 1)
        if not block:
            continue
        body, close_pos = block
        calls = collect_block_calls(body, _SKIP_CALLS)
        if calls:
            start = line_no(source, match.start())
            end = line_no(source, close_pos)
            info.branches.append(
                BranchInfo(
                    condition=" ".join(match.group(1).split()),
                    branch_kind="if_true",
                    path_calls=calls,
                    excluded_branches=["if_false"],
                    start_line=start,
                    end_line=end,
                    true_calls=calls,
                )
            )
        else_match = re.match(r"\s*else\s*\{", source[close_pos + 1:], re.S)
        if else_match:
            else_open = close_pos + 1 + else_match.end() - 1
            else_block = first_braced_block(source, else_open)
            if else_block:
                else_body, else_close = else_block
                else_calls = collect_block_calls(else_body, _SKIP_CALLS)
                if else_calls:
                    info.branches.append(
                        BranchInfo(
                            condition=" ".join(match.group(1).split()),
                            branch_kind="if_false",
                            path_calls=else_calls,
                            excluded_branches=["if_true"],
                            start_line=line_no(source, else_open),
                            end_line=line_no(source, else_close),
                            false_calls=else_calls,
                        )
                    )

    for match in re.finditer(r"\btry\s*\{", source):
        block = first_braced_block(source, match.end() - 1)
        if not block:
            continue
        body, close_pos = block
        t = TryInfo(
            start_line=line_no(source, match.start()),
            end_line=line_no(source, close_pos),
            try_calls=collect_block_calls(body, _SKIP_CALLS),
        )
        catch_match = re.match(r"\s*catch(?:\s*\([^)]*\))?\s*\{", source[close_pos + 1:], re.S)
        if catch_match:
            catch_open = close_pos + 1 + catch_match.end() - 1
            catch_block = first_braced_block(source, catch_open)
            if catch_block:
                catch_body, catch_close = catch_block
                t.end_line = line_no(source, catch_close)
                t.except_calls = collect_block_calls(catch_body, _SKIP_CALLS)
                t.raises = [m.group(1).strip() for m in re.finditer(r"\bthrow\s+([^;\n]+)", catch_body)]
                t.except_returns = [m.group(1).strip() for m in re.finditer(r"\breturn\s+([^;\n]+)", catch_body)]
        info.tries.append(t)

    for match in re.finditer(r"\bthrow\s+([^;\n]+)", source):
        info.raises.append(match.group(1).strip())

    for match in re.finditer(
        r"\b(?:const|let|var)?\s*([A-Za-z_$][A-Za-z0-9_$]*)\s*=\s*(?:await\s+)?([A-Za-z_$][A-Za-z0-9_$.]*)\s*\(([^()]*)\)",
        source,
        re.S,
    ):
        src = match.group(2).split(".")[-1]
        if src not in _SKIP_CALLS:
            info.assigns.append(
                AssignInfo(match.group(1), src, line_no(source, match.start()), " ".join(match.group(3).split()))
            )

    for match in re.finditer(r"\breturn(?:\s+([^;\n}]+))?", source):
        expr = (match.group(1) or "").strip()
        if not expr:
            continue
        info.has_return_value = True
        info.return_lineno = line_no(source, match.start())
        info.return_expr = expr
        info.return_exprs.append(expr)
        src = outer_call(expr, _SKIP_CALLS)
        if src:
            info.return_source_func = src

    _mark_call_context(source, info)
    info.all_calls = [cs.func_name for cs in info.call_sites]
    info.has_logging_calls = any(c in _LOG_NAMES or c.startswith(("log", "debug", "trace")) for c in info.all_calls)
    return info
