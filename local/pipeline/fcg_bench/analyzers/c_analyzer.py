"""Lightweight C source analyzer."""
from __future__ import annotations

import re

from ..ast_analyzer import AssignInfo, BranchInfo, CallSite, FunctionAST
from .lightweight import collect_block_calls, collect_call_matches, first_braced_block, line_no, outer_call, split_top_level_args, strip_comments


_SKIP_CALLS = {"if", "for", "while", "switch", "return", "sizeof"}
_LOG_NAMES = {"printf", "fprintf", "perror", "puts", "fputs"}


def _extract_args(source: str) -> list[str]:
    match = re.search(r"\b[A-Za-z_][A-Za-z0-9_\s*]*\s+[A-Za-z_][A-Za-z0-9_]*\s*\(([^)]*)\)\s*\{", source, re.S)
    if not match:
        return []
    args: list[str] = []
    for arg in split_top_level_args(match.group(1)):
        arg = arg.strip()
        if arg == "void":
            continue
        name_match = re.search(r"([A-Za-z_][A-Za-z0-9_]*)\s*(?:\[[^\]]*\])?$", arg.replace("*", " "))
        if name_match:
            args.append(name_match.group(1))
    return args


def analyze_c_function(qualname: str, source: str) -> FunctionAST:
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
            info.branches.append(
                BranchInfo(
                    condition=" ".join(match.group(1).split()),
                    branch_kind="if_true",
                    path_calls=calls,
                    excluded_branches=["if_false"],
                    start_line=line_no(source, match.start()),
                    end_line=line_no(source, close_pos),
                    true_calls=calls,
                )
            )

    for match in re.finditer(r"\b(?:abort|exit)\s*\(([^)]*)\)", source, re.S):
        info.raises.append(f"{match.group(0).strip()}")

    for match in re.finditer(r"\b([A-Za-z_][A-Za-z0-9_]*)\s*=\s*([A-Za-z_][A-Za-z0-9_]*)\s*\(([^()]*)\)", source, re.S):
        src = match.group(2)
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

    branch_spans = [(b.start_line, b.end_line) for b in info.branches if b.start_line and b.end_line]
    for name, lineno, arg_text in collect_call_matches(source, _SKIP_CALLS):
        info.call_sites.append(
            CallSite(
                name,
                lineno,
                arg_text,
                any(start <= lineno <= end for start, end in branch_spans),
                False,
            )
        )
    info.all_calls = [cs.func_name for cs in info.call_sites]
    info.has_logging_calls = any(c in _LOG_NAMES or c.lower().startswith(("log", "debug", "trace")) for c in info.all_calls)
    return info
