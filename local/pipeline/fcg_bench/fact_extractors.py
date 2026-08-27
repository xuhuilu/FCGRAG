"""六个事实抽取器,分别服务于六类题型。

所有函数都接受:
  graph:  CallGraph
  ast_map: qualname -> FunctionAST
  root:   当前中心函数
  subgraph: 已剪枝后的局部节点集合

返回 Fact 或 None(抽取失败/不适合出题)。
"""
from __future__ import annotations
import ast
import re
from collections import defaultdict
from .schema import CallGraph, Fact
from .ast_analyzer import FunctionAST
from .features import is_aux_like_name, is_core_like_name
from .subgraph import extract_main_path


_TRIVIAL_CALLS = {
    "all", "any", "append", "bool", "cast", "dict", "enumerate", "extend",
    "get", "getattr", "hasattr", "info", "int", "isinstance", "items", "join",
    "issubclass", "keys", "len", "list", "pop", "print", "repr", "set", "setattr",
    "startswith", "str", "tuple", "type", "update", "values", "warn", "warning",
    "endswith",
    "map", "filter", "reduce", "forEach", "foreach", "then", "catch",
    "String", "Number", "Boolean", "Array", "Object", "Promise",
    "JSON.stringify", "JSON.parse",
    "cap", "make", "new", "copy", "delete", "panic", "recover",
    "malloc", "free", "calloc", "realloc", "memcpy", "memset", "strlen",
    "printf", "fprintf", "sizeof",
}
_HELPER_PREFIXES = ("_handle", "_validate", "_format", "_check", "_convert")
_C_FAMILY_NOISE_PREFIXES = {
    "std", "boost", "absl", "fmt", "pthread", "posix", "glibc",
}
_STAGE_ORDER = {
    "输入解析": 0,
    "合法性校验": 1,
    "核心执行": 2,
    "输出构造": 3,
}


def _is_semantic_call(name: str) -> bool:
    if not name or name in _TRIVIAL_CALLS:
        return False
    if name[0].isupper():
        return False
    if is_aux_like_name(name):
        return False
    if name.endswith(("Error", "Exception", "Warning")):
        return False
    return True


def _uniq_keep_order(names: list[str], limit: int | None = None) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for name in names:
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(name)
        if limit is not None and len(out) >= limit:
            break
    return out


def _semantic_calls(names: list[str], limit: int | None = None) -> list[str]:
    return _uniq_keep_order([n for n in names if _is_semantic_call(n)], limit=limit)


def _is_simple_identifier(expr: str) -> bool:
    import re
    return re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", expr or "") is not None


_PSEUDO_CALLERS = {
    "<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>", "<lambda>",
    "listcomp", "dictcomp", "setcomp", "genexpr", "lambda",
    "IMPORTED_SYMBOL_PSEUDO", "IMPORTED_MODULE_FUNCTION_PSEUDO",
}

_FLOW_GUARD_NAMES = {
    "env_var_is_set",
    "_tracing_v2_is_enabled",
    "_get_debug",
    "_get_tracer_project",
    "get_tracing_context",
    "logger",
    "warning",
    "warn",
    "debug",
    "info",
}


def _extract_return_dependency(expr: str) -> tuple[str, str, str]:
    """返回 (kind, identifier, wrapper)。"""
    if not expr.strip():
        return "", "", ""
    try:
        node = ast.parse(expr, mode="eval").body
    except SyntaxError:
        return "", "", ""

    if isinstance(node, ast.Name):
        return "direct_identifier", node.id, ""

    wrapper = ""
    names: set[str] = set()
    if isinstance(node, ast.Call):
        wrapper = _call_name(node.func) or "call"
        for arg in list(node.args) + [kw.value for kw in node.keywords]:
            for child in ast.walk(arg):
                if isinstance(child, ast.Name):
                    names.add(child.id)
    elif isinstance(node, ast.Dict):
        wrapper = "dict_literal"
        for value in node.values:
            for child in ast.walk(value):
                if isinstance(child, ast.Name):
                    names.add(child.id)
    elif isinstance(node, ast.List):
        wrapper = "list_literal"
        for elt in node.elts:
            for child in ast.walk(elt):
                if isinstance(child, ast.Name):
                    names.add(child.id)
    elif isinstance(node, ast.Tuple):
        wrapper = "tuple_literal"
        for elt in node.elts:
            for child in ast.walk(elt):
                if isinstance(child, ast.Name):
                    names.add(child.id)
    elif isinstance(node, ast.Set):
        wrapper = "set_literal"
        for elt in node.elts:
            for child in ast.walk(elt):
                if isinstance(child, ast.Name):
                    names.add(child.id)

    if not wrapper:
        return "", "", ""

    if len(names) != 1:
        return "", "", ""
    only_name = sorted(names)[0]
    kind = "wrapped_constructor" if isinstance(node, ast.Call) else "wrapped_literal"
    return kind, only_name, wrapper


def _call_name(node: ast.AST) -> str | None:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return None


def _assign_depends_on_previous_target(assign_arg_text: str, prior_targets: set[str]) -> bool:
    return any(_contains_identifier(assign_arg_text, target) for target in prior_targets if target)


def _is_valid_static_caller(name: str, graph: CallGraph, ast_map: dict[str, FunctionAST]) -> bool:
    node = graph.nodes.get(name)
    if node is None:
        return False
    short = node.short_name
    qualname = node.qualname or name
    node_kind = (node.node_kind or "").upper()
    if not short or short in _PSEUDO_CALLERS or qualname in _PSEUDO_CALLERS:
        return False
    lower_short = short.lower()
    lower_qual = qualname.lower()
    if lower_short in _PSEUDO_CALLERS or lower_qual in _PSEUDO_CALLERS:
        return False
    if short.startswith("<") and short.endswith(">"):
        return False
    if "IMPORTED_" in short or "IMPORTED_" in name or "IMPORTED_" in qualname:
        return False
    if any(
        token in {lower_short, lower_qual}
        or token in lower_qual.replace("<", "").replace(">", "").split(".")
        for token in {"listcomp", "dictcomp", "setcomp", "genexpr", "lambda"}
    ):
        return False
    if node_kind in {"CLASS", "MODULE"}:
        return False
    if node.start_line <= 0 or node.end_line <= 0:
        return False
    if not node.code.strip():
        return False
    return ast_map.get(name) is not None


def _is_warn_only_branch_call(name: str) -> bool:
    return name in {"warn", "warning"}


def _is_guard_like_flow_call(name: str) -> bool:
    low = (name or "").lower()
    if name in _FLOW_GUARD_NAMES:
        return True
    return any(token in low for token in ("debug", "trace", "tracing", "logger", "logging"))


def _score_flow_call(name: str, fa: FunctionAST) -> int:
    score = 0
    if _is_guard_like_flow_call(name):
        score -= 3
    if name == fa.return_source_func:
        score += 3
    if any(a.source_func == name and not a.in_except for a in fa.assigns):
        score += 2
    if _is_semantic_call(name):
        score += 1
    return score


# ---------- 1. 主调用链 ----------

def extract_main_path_fact(
    graph: CallGraph, ast_map: dict[str, FunctionAST],
    root: str, subgraph: set[str],
) -> Fact | None:
    path = extract_main_path(graph, root, subgraph, max_len=6, ast_map=ast_map)
    if len(path) < 3:
        return None
    fa = ast_map.get(root)
    short = [graph.nodes[p].short_name for p in path]
    if fa is not None:
        positive = [
            name for name in short[1:]
            if _score_flow_call(name, fa) > 0
        ]
        if len(positive) < 2:
            return None
    return Fact(
        kind="main_path",
        root=root,
        payload={
            "path_type": "representative_path",
            "path_condition": "",
            "path_nodes": short,
            "path_nodes_qual": path,
            "coverage_note": "This is one extracted path, not the unique full execution flow.",
        },
    )


# ---------- 2. 关键结果来源 ----------

def extract_result_origin_fact(
    graph: CallGraph, ast_map: dict[str, FunctionAST],
    root: str, subgraph: set[str],
) -> Fact | None:
    fa = ast_map.get(root)
    if fa is None or not fa.has_return_value:
        return None
    # 仅保留单返回、单路径且返回表达式简单的样本。
    unique_return_exprs = _uniq_keep_order([e.strip() for e in fa.return_exprs if e.strip()])
    if len(unique_return_exprs) != 1:
        return None
    ret_expr = unique_return_exprs[0]

    source_entity = ""
    target_object = ""
    source_lineno = 0
    return_dependency_kind = ""
    return_wrapper = ""

    # 情形 A: return foo(...) → source = foo, target = return_value(没有下游)
    if (
        fa.return_source_func
        and _is_semantic_call(fa.return_source_func)
        and ret_expr == fa.return_expr
    ):
        source_entity = fa.return_source_func
        target_object = "返回值"
        source_lineno = fa.return_lineno
        return_dependency_kind = "direct_call"

    dep_kind, dep_target, dep_wrapper = _extract_return_dependency(ret_expr)
    if not source_entity and fa.assigns and dep_target:
        # 优先选非 except 内的赋值
        normal_assigns = [a for a in fa.assigns if not a.in_except]
        pool = normal_assigns or fa.assigns
        # 同一个目标对象如果有多个来源，说明是多分支/多路径，跳过。
        target_sources: dict[str, set[str]] = {}
        for a in pool:
            target_sources.setdefault(a.target, set()).add(a.source_func)
        if len(target_sources.get(dep_target, set())) != 1:
            return None
        target_assigns = [a for a in pool if a.target == dep_target]
        if len(target_assigns) != 1:
            return None

        prior_targets: set[str] = set()
        for a in pool:
            if a.target != dep_target:
                if a.target:
                    prior_targets.add(a.target)
                continue
            if not _is_semantic_call(a.source_func):
                return None
            if _assign_depends_on_previous_target(a.arg_text, prior_targets):
                return None
            source_entity = a.source_func
            target_object = a.target
            source_lineno = a.lineno
            return_dependency_kind = dep_kind
            return_wrapper = dep_wrapper
            break
    if not source_entity:
        return None
    if target_object and target_object in set(fa.arg_names):
        return None

    # consumer: 在 source_lineno 之后、return 之前,源码参数里出现 target_object 的调用
    consumers: list[str] = []
    if target_object and target_object != "返回值":
        end = fa.return_lineno or 10**9
        for cs in fa.call_sites:
            if cs.lineno <= source_lineno or cs.lineno > end:
                continue
            if cs.func_name == source_entity:
                continue
            if cs.in_except:
                continue
            # 简单的标识符匹配
            if _contains_identifier(cs.arg_text, target_object):
                if _is_semantic_call(cs.func_name):
                    consumers.append(cs.func_name)

    return Fact(
        kind="result_origin",
        root=root,
        payload={
            "target_object": target_object,
            "source_entity": source_entity,
            "consumer_entities": _uniq_keep_order(consumers, limit=5),
            "return_dependency_kind": return_dependency_kind,
            "return_wrapper": return_wrapper,
        },
    )


def _contains_identifier(text: str, name: str) -> bool:
    """判断 text 中是否作为独立标识符出现 name。"""
    import re
    return re.search(rf"\b{re.escape(name)}\b", text) is not None


# ---------- 3. 条件分支路径 ----------

def extract_conditional_fact(
    graph: CallGraph, ast_map: dict[str, FunctionAST],
    root: str, subgraph: set[str],
) -> Fact | None:
    fa = ast_map.get(root)
    if fa is None or not fa.branches:
        return None
    # 只保留边界清楚的单个分支，不再把 elif/else 合并成粗糙的 false path。
    for b in fa.branches:
        raw_calls = b.path_calls or b.true_calls or b.false_calls
        branch_calls = [
            name for name in _semantic_calls(raw_calls)
            if not _is_warn_only_branch_call(name)
        ]
        branch_calls = _uniq_keep_order(branch_calls, limit=5)
        if branch_calls:
            return Fact(
                kind="conditional",
                root=root,
                payload={
                    "condition": b.condition,
                    "branch_kind": b.branch_kind,
                    "branch_span": {
                        "start_line": b.start_line,
                        "end_line": b.end_line,
                    },
                    "path_nodes": [graph.nodes[root].short_name] + branch_calls,
                    "excluded_branches": b.excluded_branches,
                },
            )
    return None


# ---------- 4. 错误处理链 ----------

def extract_error_fact(
    graph: CallGraph, ast_map: dict[str, FunctionAST],
    root: str, subgraph: set[str],
) -> Fact | None:
    fa = ast_map.get(root)
    if fa is None or not fa.tries:
        return None
    def _try_score(x: object) -> tuple[int, int, int]:
        return (
            len(getattr(x, "except_calls", [])),
            len(getattr(x, "raises", [])),
            len(getattr(x, "except_returns", [])),
        )

    t = max(fa.tries, key=_try_score)
    trigger_candidates = _semantic_calls(t.try_calls, limit=3)
    handler_calls = _semantic_calls(t.except_calls, limit=4)
    if not trigger_candidates:
        return None
    trigger = trigger_candidates[0]
    error_path = [graph.nodes[root].short_name, trigger] + handler_calls
    error_action = ""
    fallback_action = None
    if t.raises:
        error_action = f"raise {t.raises[0]}"
    elif t.except_returns:
        fallback_expr = t.except_returns[0] if t.except_returns[0] else "None"
        error_action = f"return {fallback_expr}"
        fallback_action = error_action
    elif t.warns:
        error_action = "warning"
    elif handler_calls:
        error_action = handler_calls[0]

    if len(error_path) < 3 and not error_action:
        return None
    return Fact(
        kind="error_handling",
        root=root,
        payload={
            "error_trigger": trigger,
            "error_path": error_path,
            "error_action": error_action,
            "fallback_action": fallback_action,
            "evidence_span": {
                "start_line": t.start_line,
                "end_line": t.end_line,
            },
        },
    )


# ---------- 5. 核心/辅助角色 ----------

def extract_role_fact(
    graph: CallGraph, ast_map: dict[str, FunctionAST],
    root: str, subgraph: set[str],
) -> Fact | None:
    main = set(extract_main_path(graph, root, subgraph, max_len=6, ast_map=ast_map))
    core: list[str] = []
    support: list[str] = []
    for n in subgraph:
        if n == root:
            continue
        node = graph.nodes[n]
        short = node.short_name
        if not _is_semantic_call(short):
            continue
        if n in main and not is_aux_like_name(short):
            core.append(short)
        elif is_aux_like_name(short) or (node.has_logging_calls and node.out_degree == 0):
            support.append(short)
        elif is_core_like_name(short):
            core.append(short)
        else:
            # 其他:按出度分
            if node.out_degree >= 1:
                core.append(short)
            else:
                support.append(short)

    core = _uniq_keep_order(sorted(set(core)))
    support = _uniq_keep_order(sorted(set(support)))
    support = [s for s in support if s not in set(core)]
    if not core or not support:
        return None
    return Fact(
        kind="role_classification",
        root=root,
        payload={"core_entities": core, "support_entities": support},
    )


# ---------- 6. 局部模块阶段 ----------

_STAGE_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("输入解析", (
        "parse", "load", "read", "fetch", "decode", "input", "payload",
        "prompt", "message", "request", "args", "kwargs", "params",
    )),
    ("合法性校验", (
        "validate", "check", "verify", "ensure", "normalize", "sanitize",
        "guard", "should_", "supports_", "prefers_", "filter",
    )),
    ("核心执行", (
        "run", "execute", "compute", "forward", "predict", "process", "update",
        "train", "step", "generate", "stream", "invoke", "call", "complete",
    )),
    ("输出构造", (
        "build", "format", "encode", "write", "send", "emit", "render",
        "return", "result", "response", "output", "chunk", "metadata",
        "create", "construct",
    )),
]


def _classify_stage(name: str) -> str | None:
    low = name.lower()
    for stage, kws in _STAGE_RULES:
        if any(k in low for k in kws):
            return stage
    return None


def _stage_hits_from_names(names: list[str]) -> dict[str, list[str]]:
    hits: dict[str, list[str]] = {s: [] for s, _ in _STAGE_RULES}
    for name in names:
        if not _is_semantic_call(name):
            continue
        stage = _classify_stage(name)
        if stage is None:
            continue
        hits[stage].append(name)
    return hits


def _pick_stage_representatives(
    stage: str,
    root_short: str,
    main_names: list[str],
    subgraph_names: list[str],
) -> list[str]:
    """为一个阶段挑最代表性的函数。

    排序优先级:
      1. root 本身若命中该阶段
      2. 主链中的命中函数
      3. 子图中的其他命中函数
      4. 名字更像 orchestrator / 更不像 helper
    """
    candidates = _uniq_keep_order(
        [root_short] + main_names + subgraph_names
    )

    def score(name: str) -> tuple[int, int, int, int, str]:
        stage_match = 1 if _classify_stage(name) == stage else 0
        is_root = 1 if name == root_short else 0
        on_main = 1 if name in main_names else 0
        orchestrator = 1 if any(
            k in name.lower()
            for k in ("run", "generate", "stream", "invoke", "process", "execute", "format")
        ) else 0
        helper_penalty = 0 if any(name.startswith(p) for p in _HELPER_PREFIXES) else 1
        return (stage_match, is_root, on_main, orchestrator + helper_penalty, name)

    picked = [n for n in candidates if _classify_stage(n) == stage]
    picked.sort(key=score, reverse=True)
    return picked[:1]


def extract_stages_fact(
    graph: CallGraph, ast_map: dict[str, FunctionAST],
    root: str, subgraph: set[str],
) -> Fact | None:
    main = extract_main_path(graph, root, subgraph, max_len=8, ast_map=ast_map)
    if len(main) < 3:
        return None

    root_short = graph.nodes[root].short_name
    fa = ast_map.get(root)
    if fa is None:
        return None

    main_names = [graph.nodes[q].short_name for q in main]
    # 只允许从 root 自身、一跳可见的 callee、以及 root 源码里显式出现过的调用中选核心函数。
    direct_callee_names = [
        graph.nodes[q].short_name
        for q in graph.callees(root)
        if q in graph.nodes
    ]
    root_callsite_names = [cs.func_name for cs in fa.call_sites if not cs.in_except]
    candidate_names = _uniq_keep_order(
        [root_short] + root_callsite_names + direct_callee_names
    )
    candidate_names = [
        n for n in candidate_names
        if n == root_short or _is_semantic_call(n)
    ]
    candidate_names.sort(key=lambda n: (_classify_stage(n) is None, n))

    stage_to_funcs: dict[str, list[str]] = {s: [] for s, _ in _STAGE_RULES}
    ordered: list[str] = []
    for stage, _ in _STAGE_RULES:
        picks = _pick_stage_representatives(
            stage,
            root_short=root_short,
            main_names=[n for n in main_names[1:] if n in candidate_names],
            subgraph_names=[n for n in candidate_names if n != root_short],
        )
        if picks:
            stage_to_funcs[stage].extend(picks)
            ordered.append(stage)

    # root 是 orchestration 函数时,默认存在核心执行阶段; 有返回值时默认存在输出构造阶段。
    root_low = root_short.lower()
    if "核心执行" not in ordered and any(
        k in root_low for k in ("run", "generate", "stream", "invoke", "process", "execute")
    ):
        ordered.append("核心执行")
        stage_to_funcs["核心执行"] = _uniq_keep_order([root_short] + stage_to_funcs["核心执行"], limit=1)
    if (
        fa is not None
        and fa.has_return_value
        and "输出构造" not in ordered
        and (stage_to_funcs["核心执行"] or stage_to_funcs["输入解析"] or stage_to_funcs["合法性校验"])
    ):
        ordered.append("输出构造")
        stage_to_funcs["输出构造"] = _uniq_keep_order(
            stage_to_funcs["输出构造"] + [root_short],
            limit=1,
        )

    ordered = sorted(set(ordered), key=lambda s: _STAGE_ORDER[s])
    if len(ordered) < 2 or "核心执行" not in ordered:
        return None

    core_entities = _uniq_keep_order(
        [root_short] + [f for s in ordered for f in stage_to_funcs[s] if _is_semantic_call(f)],
        limit=6,
    )
    if len(core_entities) < 2:
        return None
    return Fact(
        kind="stages",
        root=root,
        payload={"summary_core_entities": core_entities, "stages": ordered},
    )


# 汇总入口
# ---------- 7. 跨函数依赖追踪(纯调用图) ----------
#
# 设计动机:这一类题完全不依赖 AST,只靠调用图的多跳遍历回答。
# 这是为了让 benchmark 真正考验"跨函数推理"能力,弥补前 6 类题
# 都集中在单函数局部结构的不足。
#
# 题目示例:"从 X 出发,经过 2~3 跳能到达哪些以 Y 开头的函数?"
# Gold: 列出所有满足条件的目标函数名 + 一条代表性最短路径。

def _c_api_family_keys(qualname: str) -> list[str]:
    """Return stable namespace or snake-case API-family keys for C/C++ symbols."""
    symbol = re.sub(r"\([^)]*\)$", "", qualname or "").strip()
    symbol = re.sub(r"<[^<>]*>", "", symbol)
    if not symbol:
        return []

    keys: list[str] = []
    namespace_parts = [part for part in symbol.split("::") if part]
    if len(namespace_parts) >= 2:
        namespace = "::".join(namespace_parts[:-1])
        namespace_root = namespace.split("::", 1)[0].lower()
        if (
            namespace
            and not namespace_root.startswith("(")
            and namespace_root not in _C_FAMILY_NOISE_PREFIXES
        ):
            keys.append(namespace)

    leaf = namespace_parts[-1] if namespace_parts else symbol
    leaf = leaf.rsplit(".", 1)[-1].lstrip("_")
    tokens = [token for token in leaf.split("_") if token]
    if len(tokens) >= 2 and tokens[0].lower() not in _C_FAMILY_NOISE_PREFIXES:
        keys.append("_".join(tokens[:2]))
    elif tokens and len(tokens[0]) >= 3 and tokens[0].lower() not in _C_FAMILY_NOISE_PREFIXES:
        keys.append(tokens[0])
    return _uniq_keep_order(keys)


def _extract_c_dependency_trace_fact(
    graph: CallGraph,
    root: str,
    distances: dict[str, int],
) -> Fact | None:
    """Build a dependency trace around a repeated C/C++ API family.

    C and C++ projects rarely expose the Python-oriented target categories used
    by the generic extractor.  Their public and internal APIs instead tend to
    form stable namespaces (``ggml::cpu``) or snake-case families
    (``ggml_backend_*``).  Require three distinct, statically reachable targets
    at depth two or three so a direct-call list cannot become a dependency trace.
    """
    targets_by_family: dict[str, list[str]] = defaultdict(list)
    for qualname, distance in distances.items():
        if qualname == root or distance < 2 or distance > 3:
            continue
        node = graph.nodes.get(qualname)
        if node is None or node.is_test_related:
            continue
        for family in _c_api_family_keys(qualname):
            targets_by_family[family].append(qualname)

    candidates: list[tuple[int, int, str, list[str]]] = []
    for family, qualnames in targets_by_family.items():
        unique_by_short: dict[str, str] = {}
        for name in sorted(qualnames):
            if name in graph.nodes:
                unique_by_short.setdefault(graph.nodes[name].short_name, name)
        if len(unique_by_short) < 3:
            continue
        ordered = sorted(
            unique_by_short.values(),
            key=lambda name: (distances[name], graph.nodes[name].short_name, name),
        )
        specificity = family.count("::") + family.count("_")
        candidates.append((specificity, len(ordered), family, ordered))

    if not candidates:
        return None

    _, _, family, targets = max(candidates, key=lambda item: (item[0], item[1], item[2]))
    targets = targets[:8]
    sample_path = _shortest_path(graph, root, targets[0], max_depth=3)
    if sample_path is None or len(sample_path) < 3:
        return None

    return Fact(
        kind="dependency_trace",
        root=root,
        payload={
            "target_kind": f"{family} API-family functions",
            "target_description": f"{family} API-family functions",
            "target_entities": [graph.nodes[name].short_name for name in targets],
            "sample_path": [graph.nodes[name].short_name for name in sample_path],
            "max_hops": 3,
        },
    )


def extract_dependency_trace_fact(
    graph: CallGraph, ast_map: dict[str, FunctionAST],
    root: str, subgraph: set[str],
) -> Fact | None:
    """
    从 root 做 2~3 跳 BFS,收集所有"目标节点"。
    目标节点的定义:某种能命名出来的语义类别,比如:
      - 类实例化(短名是大写开头)
      - 包含特定关键词的函数(_handle、_validate、_format 等)

    为了让题目可读,我们挑出度最大的根的"主要目标类别",
    然后列出从 root 经过 ≤3 跳能到的所有该类别节点。
    """
    # 拿 2-hop 子图
    visited = {root}
    frontier = [root]
    distances = {root: 0}
    is_c_root = bool(graph.nodes.get(root) and graph.nodes[root].language == "c")
    for hop in range(1, 4):
        next_frontier: list[str] = []
        for u in frontier:
            callees = graph.callees(u)
            if is_c_root:
                callees = sorted(callees)
            for v in callees:
                if v not in visited:
                    visited.add(v)
                    distances[v] = hop
                    next_frontier.append(v)
        frontier = next_frontier
        if len(visited) > 80:
            break  # 太大了不出题

    if len(visited) < 6:
        return None  # 子图太小,出题没意思

    # C/C++: identify repeated API families rather than Python-style classes
    # and underscore-prefixed helpers.  `detect_language` intentionally maps
    # .c, .cc, .cpp, .cxx, and headers to "c".
    if is_c_root:
        return _extract_c_dependency_trace_fact(graph, root, distances)

    # 找候选目标类别:统计能识别的"语义类"
    # 类别 1:类实例化(大写开头,且不是异常类)
    ordered_visited = sorted(
        visited,
        key=lambda q: (distances.get(q, 99), graph.nodes[q].short_name, q),
    )

    class_targets: list[str] = []
    for q in ordered_visited:
        if q == root:
            continue
        short = graph.nodes[q].short_name
        if short and short[0].isupper() and not short.endswith(
            ("Error", "Exception", "Warning")
        ):
            class_targets.append(q)

    # 类别 2:helper 类的辅助函数(_handle / _validate / _format / _check 等)
    helper_targets: list[str] = []
    for q in ordered_visited:
        if q == root:
            continue
        short = graph.nodes[q].short_name
        if any(short.startswith(p) for p in _HELPER_PREFIXES):
            helper_targets.append(q)

    # 选数量更多的类别作为题目的目标
    if len(class_targets) >= 3 and len(class_targets) >= len(helper_targets):
        targets = class_targets
        target_kind = "class_instantiation"
        target_desc = "类实例化"
    elif len(helper_targets) >= 3:
        targets = helper_targets
        target_kind = "helper_function"
        target_desc = "辅助函数"
    else:
        return None

    targets = targets[:8]  # 截断,避免题目答案太长

    target_shorts = sorted(set(graph.nodes[q].short_name for q in targets))

    # 找一条代表性最短路径(从 root 到第一个目标)
    sample_target = targets[0]
    representative_path = _shortest_path(graph, root, sample_target, max_depth=4)
    if representative_path is None:
        return None
    rep_path_shorts = [graph.nodes[q].short_name for q in representative_path]

    return Fact(
        kind="dependency_trace",
        root=root,
        payload={
            "target_kind": target_kind,
            "target_description": target_desc,
            "target_entities": target_shorts,
            "sample_path": rep_path_shorts,
            "max_hops": 3,
        },
    )


def _shortest_path(
    graph: CallGraph, src: str, dst: str, max_depth: int = 4,
) -> list[str] | None:
    """BFS 找 src -> dst 的最短路径(无向意义上从 src 出发)。返回 qualname 列表。"""
    if src == dst:
        return [src]
    parent: dict[str, str] = {src: ""}
    frontier = [src]
    for _ in range(max_depth):
        next_frontier: list[str] = []
        for u in frontier:
            for v in graph.callees(u):
                if v in parent:
                    continue
                parent[v] = u
                if v == dst:
                    # 重建路径
                    path = [v]
                    cur = u
                    while cur:
                        path.append(cur)
                        cur = parent.get(cur, "")
                    return list(reversed(path))
                next_frontier.append(v)
        frontier = next_frontier
    return None


# ---------- 8. 影响面(反向调用关系) ----------
#
# 题目示例:"如果 X 的接口改变,有哪些函数会受影响?"
# Gold: 直接调用 X 的函数集合 + 间接(2 跳)调用者集合。

def extract_impact_scope_fact(
    graph: CallGraph, ast_map: dict[str, FunctionAST],
    root: str, subgraph: set[str],
) -> Fact | None:
    """
    用 rev_edges 做反向 BFS,收集 root 的直接和间接调用者。
    """
    direct = {
        q for q in graph.rev_edges.get(root, set())
        if q in graph.nodes
        and not graph.nodes[q].is_test_related
        and _is_valid_static_caller(q, graph, ast_map)
    }
    if len(direct) < 2:
        return None  # 调用者太少,这道题没区分度

    indirect: set[str] = set()
    for c in direct:
        indirect |= {
            q for q in graph.rev_edges.get(c, set())
            if q in graph.nodes
            and not graph.nodes[q].is_test_related
            and _is_valid_static_caller(q, graph, ast_map)
        }
    indirect -= direct
    indirect.discard(root)

    if len(direct) + len(indirect) < 4:
        return None
    if len(direct) > 15:
        return None  # 调用者太多反而不好出题(信息密度太高)

    direct_shorts = sorted(set(graph.nodes[q].short_name for q in direct if q in graph.nodes))
    indirect_shorts = sorted(set(
        graph.nodes[q].short_name for q in indirect if q in graph.nodes
    ))[:10]  # 限制 indirect 数量
    potential_shorts = _uniq_keep_order(direct_shorts + indirect_shorts, limit=12)
    direct_shorts = [name for name in direct_shorts if name and name not in _PSEUDO_CALLERS]
    potential_shorts = [name for name in potential_shorts if name and name not in _PSEUDO_CALLERS]
    if len(direct_shorts) < 2:
        return None
    if len(potential_shorts) < 4:
        return None

    return Fact(
        kind="impact_scope",
        root=root,
        payload={
            "static_direct_callers": direct_shorts,
            "potentially_affected_callers": potential_shorts,
        },
    )


EXTRACTORS = {
    "execution_flow": extract_main_path_fact,
    "result_origin": extract_result_origin_fact,
    "conditional_path": extract_conditional_fact,
    "error_handling_path": extract_error_fact,
    "core_support_classification": extract_role_fact,
    "local_subgraph_summary": extract_stages_fact,
    "dependency_trace": extract_dependency_trace_fact,
    "impact_scope": extract_impact_scope_fact,
}
