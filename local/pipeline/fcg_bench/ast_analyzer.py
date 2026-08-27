"""基于 ast 的源码分析:抽取分支、异常、返回、调用等结构。"""
from __future__ import annotations
import ast
import textwrap
from dataclasses import dataclass, field


_LOG_NAMES = {"log", "logger", "logging", "print"}
_AUX_PREFIXES = ("log_", "debug_", "print_", "helper_", "util_", "_log", "_debug")


@dataclass
class BranchInfo:
    """一个 if/elif/else 分支点。"""
    condition: str                         # 条件源码
    branch_kind: str = "if_true"          # if_true / if_false
    path_calls: list[str] = field(default_factory=list)
    excluded_branches: list[str] = field(default_factory=list)
    start_line: int = 0
    end_line: int = 0
    true_calls: list[str] = field(default_factory=list)
    false_calls: list[str] = field(default_factory=list)


@dataclass
class TryInfo:
    """一个 try/except 结构。"""
    start_line: int = 0
    end_line: int = 0
    try_calls: list[str] = field(default_factory=list)
    except_calls: list[str] = field(default_factory=list)
    raises: list[str] = field(default_factory=list)
    except_returns: list[str] = field(default_factory=list)
    warns: list[str] = field(default_factory=list)


@dataclass
class AssignInfo:
    """赋值:target = func(...)"""
    target: str
    source_func: str                       # 右侧最外层调用的函数名
    lineno: int
    arg_text: str = ""
    in_except: bool = False                # 是否位于 except 处理器内


@dataclass
class CallSite:
    """一次函数调用点。"""
    func_name: str
    lineno: int
    arg_text: str = ""                     # 所有参数拼起来的源码(用于查找变量引用)
    in_branch: bool = False                # 是否在 if/else 分支内(非顶层)
    in_except: bool = False                # 是否在 except 处理器内


@dataclass
class FunctionAST:
    """对单个函数源码的静态分析结果。"""
    qualname: str
    arg_names: list[str] = field(default_factory=list)
    has_return_value: bool = False
    return_expr: str = ""                  # 最后一个 return 的表达式源码
    return_source_func: str = ""           # 若 return foo(...) 则为 "foo"
    return_lineno: int = 0
    return_exprs: list[str] = field(default_factory=list)
    branches: list[BranchInfo] = field(default_factory=list)
    tries: list[TryInfo] = field(default_factory=list)
    raises: list[str] = field(default_factory=list)
    all_calls: list[str] = field(default_factory=list)
    call_sites: list[CallSite] = field(default_factory=list)
    assigns: list[AssignInfo] = field(default_factory=list)
    has_logging_calls: bool = False


def _call_name(node: ast.AST) -> str | None:
    """从一个 Call 的 func 节点中取可读名字。"""
    if isinstance(node, ast.Call):
        node = node.func
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _collect_calls(node: ast.AST) -> list[str]:
    """收集一个 AST 子树中所有函数调用的短名。"""
    names: list[str] = []
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            n = _call_name(child.func)
            if n:
                names.append(n)
    return names


def _collect_executable_calls(node: ast.AST) -> list[str]:
    """收集可执行语句体中的调用，跳过控制谓词中的调用。"""
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


def _analyze_python_function_legacy(qualname: str, source: str) -> FunctionAST:
    """解析一段函数源码,返回结构化事实。容错:解析失败返回空结果。"""
    info = FunctionAST(qualname=qualname)
    if not source.strip():
        return info
    try:
        tree = ast.parse(textwrap.dedent(source))
    except SyntaxError:
        return info

    # 找到第一个 FunctionDef / AsyncFunctionDef
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

    # 顶层分支 / 异常收集(用于出条件分支题、错误处理题)
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

    # 递归带上下文遍历,记录 call_sites / assigns / return,并标记 in_branch / in_except
    def walk(stmts: list[ast.stmt], in_branch: bool, in_except: bool) -> None:
        for stmt in stmts:
            _visit(stmt, in_branch, in_except)

    def _visit(node: ast.AST, in_branch: bool, in_except: bool) -> None:
        # 所有直接出现在这个 statement 中的 Call(不递归进子 statement)
        for sub in ast.walk(node):
            if isinstance(sub, (ast.If, ast.Try, ast.For, ast.While, ast.With,
                                ast.FunctionDef, ast.AsyncFunctionDef)):
                if sub is not node:
                    continue
            if isinstance(sub, ast.Call):
                # 跳过嵌套在子控制流里的 Call(后面递归处理)
                # 简化:这里把所有 Call 都算进来,靠后面递归时覆盖
                pass

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
        if isinstance(node, (ast.For, ast.While, ast.With,
                             ast.AsyncFor, ast.AsyncWith)):
            for b in node.body:
                _visit(b, in_branch, in_except)
            if getattr(node, "orelse", None):
                for b in node.orelse:
                    _visit(b, in_branch, in_except)
            return

        # 叶子 statement:收集它自身子树里的 Call / Assign / Return
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
                            target=tgt, source_func=src,
                            lineno=node.lineno, arg_text=assign_args_text, in_except=in_except,
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

    walk(fdef.body, in_branch=False, in_except=False)
    info.all_calls = [cs.func_name for cs in info.call_sites]
    info.has_logging_calls = any(
        c in _LOG_NAMES or c.startswith(_AUX_PREFIXES) for c in info.all_calls
    )
    return info


def analyze_function(qualname: str, source: str, language: str = "python") -> FunctionAST:
    """Analyze a function body using the language-specific frontend.

    The rest of the benchmark pipeline consumes only FunctionAST, so language
    differences stay isolated behind this dispatcher.
    """
    lang = (language or "python").strip().lower()
    try:
        if lang == "python":
            from .analyzers.python_analyzer import analyze_python_function

            return analyze_python_function(qualname, source)
        if lang in {"typescript", "javascript", "ts", "tsx", "js", "jsx", "mjs", "cjs"}:
            from .analyzers.ts_js_analyzer import analyze_ts_js_function

            return analyze_ts_js_function(qualname, source)
        if lang == "go":
            from .analyzers.go_analyzer import analyze_go_function

            return analyze_go_function(qualname, source)
        if lang in {"c", "h", "cc", "cpp", "cxx", "hpp"}:
            from .analyzers.c_analyzer import analyze_c_function

            return analyze_c_function(qualname, source)
    except Exception:
        return FunctionAST(qualname=qualname)
    return FunctionAST(qualname=qualname)
