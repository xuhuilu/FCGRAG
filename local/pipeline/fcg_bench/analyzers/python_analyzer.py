"""Python AST-backed analyzer."""
from __future__ import annotations

import ast
import textwrap

from ..ast_analyzer import AssignInfo, BranchInfo, CallSite, FunctionAST, TryInfo


_LOG_NAMES = {"log", "logger", "logging", "print"}
_AUX_PREFIXES = ("log_", "debug_", "print_", "helper_", "util_", "_log", "_debug")


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _collect_calls(node: ast.AST) -> list[str]:
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            n = _call_name(child.func)
            if n:
                names.append(n)
    return names


def _collect_executable_calls(node: ast.AST) -> list[str]:
    names: list[str] = []

    def visit(cur: ast.AST) -> None:
        if isinstance(cur, ast.Call):
            name = _call_name(cur.func)
            if name:
                names.append(name)
            for arg in cur.args:
                visit(arg)
            for kw in cur.keywords:
                visit(kw.value)
            return
        if isinstance(cur, ast.If):
            for stmt in cur.body:
                visit(stmt)
            for stmt in cur.orelse:
                visit(stmt)
            return
        if isinstance(cur, ast.Try):
            for stmt in cur.body:
                visit(stmt)
            for handler in cur.handlers:
                for stmt in handler.body:
                    visit(stmt)
            for stmt in cur.finalbody:
                visit(stmt)
            for stmt in cur.orelse:
                visit(stmt)
            return
        if isinstance(cur, (ast.For, ast.AsyncFor, ast.While, ast.With, ast.AsyncWith)):
            for stmt in cur.body:
                visit(stmt)
            for stmt in getattr(cur, "orelse", []):
                visit(stmt)
            for stmt in getattr(cur, "finalbody", []):
                visit(stmt)
            return
        for child in ast.iter_child_nodes(cur):
            visit(child)

    visit(node)
    return names


def _stmt_span(stmts: list[ast.stmt]) -> tuple[int, int]:
    if not stmts:
        return 0, 0
    start = getattr(stmts[0], "lineno", 0)
    end = max(getattr(stmt, "end_lineno", getattr(stmt, "lineno", 0)) for stmt in stmts)
    return start, end


def analyze_python_function(qualname: str, source: str) -> FunctionAST:
    info = FunctionAST(qualname=qualname)
    if not source.strip():
        return info
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return info

    fdef: ast.AST | None = None
    for n in tree.body:
        if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)):
            fdef = n
            break
    if fdef is None:
        fdef = tree

    if isinstance(fdef, (ast.FunctionDef, ast.AsyncFunctionDef)):
        info.arg_names = [a.arg for a in getattr(fdef.args, "posonlyargs", [])]
        info.arg_names.extend(a.arg for a in fdef.args.args)
        info.arg_names.extend(a.arg for a in fdef.args.kwonlyargs)
        if fdef.args.vararg is not None:
            info.arg_names.append(fdef.args.vararg.arg)
        if fdef.args.kwarg is not None:
            info.arg_names.append(fdef.args.kwarg.arg)

    for child in ast.walk(fdef):
        if isinstance(child, ast.If):
            try:
                cond = ast.unparse(child.test)
            except Exception:
                cond = "<cond>"
            true_calls: list[str] = []
            for b in child.body:
                true_calls.extend(_collect_executable_calls(b))
            true_start, true_end = _stmt_span(child.body)
            if true_calls:
                info.branches.append(
                    BranchInfo(
                        condition=cond,
                        branch_kind="if_true",
                        path_calls=true_calls,
                        excluded_branches=["if_false"] if child.orelse else [],
                        start_line=true_start,
                        end_line=true_end,
                        true_calls=true_calls,
                        false_calls=[],
                    )
                )
            has_nested_elif = bool(child.orelse) and isinstance(child.orelse[0], ast.If)
            if child.orelse and not has_nested_elif:
                false_calls: list[str] = []
                for b in child.orelse:
                    false_calls.extend(_collect_executable_calls(b))
                false_start, false_end = _stmt_span(child.orelse)
                if false_calls:
                    info.branches.append(
                        BranchInfo(
                            condition=cond,
                            branch_kind="if_false",
                            path_calls=false_calls,
                            excluded_branches=["if_true"],
                            start_line=false_start,
                            end_line=false_end,
                            true_calls=[],
                            false_calls=false_calls,
                        )
                    )
        elif isinstance(child, ast.Try):
            t = TryInfo(
                start_line=getattr(child, "lineno", 0),
                end_line=getattr(child, "end_lineno", getattr(child, "lineno", 0)),
            )
            for b in child.body:
                t.try_calls.extend(_collect_calls(b))
            for h in child.handlers:
                for b in h.body:
                    t.except_calls.extend(_collect_calls(b))
                    for sub in ast.walk(b):
                        if isinstance(sub, ast.Raise) and sub.exc is not None:
                            try:
                                t.raises.append(ast.unparse(sub.exc))
                            except Exception:
                                pass
                        elif isinstance(sub, ast.Return):
                            try:
                                t.except_returns.append(ast.unparse(sub.value) if sub.value is not None else "")
                            except Exception:
                                t.except_returns.append("")
                        elif isinstance(sub, ast.Call):
                            name = _call_name(sub.func)
                            if name == "warn":
                                try:
                                    t.warns.append(ast.unparse(sub))
                                except Exception:
                                    t.warns.append("warn(...)")
            info.tries.append(t)
        elif isinstance(child, ast.Raise) and child.exc is not None:
            try:
                info.raises.append(ast.unparse(child.exc))
            except Exception:
                pass

    def walk(stmts: list[ast.stmt], in_branch: bool, in_except: bool) -> None:
        for stmt in stmts:
            _visit(stmt, in_branch, in_except)

    def _visit(node: ast.AST, in_branch: bool, in_except: bool) -> None:
        if isinstance(node, ast.If):
            for b in node.body:
                _visit(b, True, in_except)
            for b in node.orelse:
                _visit(b, True, in_except)
            return
        if isinstance(node, ast.Try):
            for b in node.body:
                _visit(b, in_branch, in_except)
            for h in node.handlers:
                for b in h.body:
                    _visit(b, in_branch, True)
            for b in node.finalbody:
                _visit(b, in_branch, in_except)
            return
        if isinstance(node, (ast.For, ast.While, ast.With, ast.AsyncFor, ast.AsyncWith)):
            for b in node.body:
                _visit(b, in_branch, in_except)
            if getattr(node, "orelse", None):
                for b in node.orelse:
                    _visit(b, in_branch, in_except)
            return

        for sub in ast.walk(node):
            if isinstance(sub, ast.Call):
                n = _call_name(sub.func)
                if not n:
                    continue
                try:
                    args_text = ", ".join(ast.unparse(a) for a in sub.args)
                except Exception:
                    args_text = ""
                info.call_sites.append(
                    CallSite(
                        func_name=n,
                        lineno=getattr(sub, "lineno", 0),
                        arg_text=args_text,
                        in_branch=in_branch,
                        in_except=in_except,
                    )
                )

        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Call):
            src = _call_name(node.value.func)
            if src:
                try:
                    assign_args_text = ", ".join(ast.unparse(a) for a in node.value.args)
                except Exception:
                    assign_args_text = ""
                for t in node.targets:
                    try:
                        tgt = ast.unparse(t)
                    except Exception:
                        continue
                    info.assigns.append(
                        AssignInfo(
                            target=tgt,
                            source_func=src,
                            lineno=node.lineno,
                            arg_text=assign_args_text,
                            in_except=in_except,
                        )
                    )

        if isinstance(node, ast.Return) and node.value is not None:
            info.has_return_value = True
            info.return_lineno = getattr(node, "lineno", 0)
            try:
                info.return_expr = ast.unparse(node.value)
            except Exception:
                info.return_expr = ""
            if info.return_expr:
                info.return_exprs.append(info.return_expr)
            if isinstance(node.value, ast.Call):
                n = _call_name(node.value.func)
                if n:
                    info.return_source_func = n

    walk(getattr(fdef, "body", []), in_branch=False, in_except=False)
    info.all_calls = [cs.func_name for cs in info.call_sites]
    info.has_logging_calls = any(
        c in _LOG_NAMES or c.startswith(_AUX_PREFIXES) for c in info.all_calls
    )
    return info
