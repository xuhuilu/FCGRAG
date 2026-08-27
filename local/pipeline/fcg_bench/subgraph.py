"""子图剪枝(去掉 log/debug/util 等噪声)和主调用链的启发式抽取。"""
from __future__ import annotations
from collections import deque
from .schema import CallGraph
from .features import is_aux_like_name


_TRIVIAL_FLOW_CALLS = {
    "all", "any", "bool", "cast", "dict", "endswith", "get", "getattr",
    "hasattr", "info", "int", "isinstance", "issubclass", "len", "list",
    "print", "repr", "set", "setattr", "startswith", "str", "tuple", "type",
    "warn", "warning",
    "map", "filter", "reduce", "foreach", "then", "catch", "string", "number",
    "boolean", "array", "object", "promise", "json.stringify", "json.parse",
    "cap", "make", "new", "append", "copy", "delete", "panic", "recover",
    "malloc", "free", "calloc", "realloc", "memcpy", "memset", "strlen",
    "printf", "fprintf", "sizeof",
}
_FLOW_GUARD_KEYWORDS = ("debug", "trace", "tracing", "logger", "logging")
_FLOW_GUARD_NAMES = {
    "env_var_is_set",
    "_tracing_v2_is_enabled",
    "_get_debug",
    "_get_tracer_project",
    "get_tracing_context",
}


def _is_flow_noise_call(name: str) -> bool:
    low = (name or "").lower()
    if not name:
        return True
    if name in _TRIVIAL_FLOW_CALLS or name in _FLOW_GUARD_NAMES:
        return True
    if is_aux_like_name(name):
        return True
    return any(token in low for token in _FLOW_GUARD_KEYWORDS)


def prune_subgraph(
    graph: CallGraph,
    nodes: set[str],
    keep_some_aux: bool = True,
    max_size: int = 20,
) -> set[str]:
    """丢掉明显是辅助/噪声的节点,但保留一部分(用于核心/辅助区分题)。"""
    kept: set[str] = set()
    aux_kept = 0
    for n in nodes:
        node = graph.nodes.get(n)
        if node is None:
            continue
        aux = is_aux_like_name(node.short_name) or node.has_logging_calls and node.out_degree == 0
        if aux:
            if keep_some_aux and aux_kept < 3:
                kept.add(n)
                aux_kept += 1
            continue
        kept.add(n)

    if len(kept) <= max_size:
        return kept
    # 太大 → 保留入度/出度更高的
    ranked = sorted(
        kept,
        key=lambda x: graph.nodes[x].out_degree + graph.nodes[x].in_degree,
        reverse=True,
    )
    return set(ranked[:max_size])


def extract_main_path(
    graph: CallGraph,
    root: str,
    subgraph: set[str],
    max_len: int = 6,
    ast_map: "dict | None" = None,
) -> list[str]:
    """从 root 出发抽主链。

    策略:
    - 若提供 ast_map 且 root 的 call_sites 里有多个"非辅助"调用,
      则把它们按源码 lineno 顺序串成主链(这更像"X 是如何一步步完成功能"的答案)。
    - 否则退化为贪心 DFS:每步选一个得分最高的 callee 继续往下走。
    """
    root_node = graph.nodes.get(root)
    short_to_qual = {
        graph.nodes[q].short_name: q for q in subgraph if q in graph.nodes
    }

    if ast_map is not None and root in ast_map:
        fa = ast_map[root]
        seen: set[str] = set()
        ordered_calls: list[str] = []
        for cs in sorted(fa.call_sites, key=lambda c: c.lineno):
            # 跳过 if/else 分支内 和 except 内的调用(它们不属于"正常主流程")
            if cs.in_branch or cs.in_except:
                continue
            name = cs.func_name
            if name in seen:
                continue
            if _is_flow_noise_call(name):
                continue
            if name not in short_to_qual:
                continue
            seen.add(name)
            ordered_calls.append(short_to_qual[name])
        preferred_sources = {
            a.source_func for a in fa.assigns
            if not a.in_except and not _is_flow_noise_call(a.source_func)
        }
        if fa.return_source_func and not _is_flow_noise_call(fa.return_source_func):
            preferred_sources.add(fa.return_source_func)
        if preferred_sources:
            filtered_calls = [
                q for q in ordered_calls
                if graph.nodes[q].short_name in preferred_sources
            ]
            if len(filtered_calls) >= 2:
                return [root] + filtered_calls[: max_len - 1]
        if len(ordered_calls) >= 2:
            return [root] + ordered_calls[: max_len - 1]

    # 回退:贪心 DFS
    path = [root]
    visited = {root}
    current = root
    for _ in range(max_len - 1):
        candidates = [
            c for c in graph.callees(current)
            if c in subgraph and c not in visited
        ]
        if not candidates:
            break

        def score(c: str) -> float:
            node = graph.nodes[c]
            s = 0.0
            if not is_aux_like_name(node.short_name):
                s += 2.0
            if not _is_flow_noise_call(node.short_name):
                s += 1.5
            else:
                s -= 2.5
            if node.has_logging_calls and node.out_degree == 0:
                s -= 2.0
            s += min(node.out_degree, 5) * 0.2
            return s

        candidates.sort(key=score, reverse=True)
        nxt = candidates[0]
        path.append(nxt)
        visited.add(nxt)
        current = nxt
    return path


def bfs_path(graph: CallGraph, src: str, dst: str, subgraph: set[str]) -> list[str]:
    """在子图内找 src 到 dst 的最短路径(无则返回空)。"""
    if src == dst:
        return [src]
    prev: dict[str, str] = {src: ""}
    q = deque([src])
    while q:
        u = q.popleft()
        if u == dst:
            break
        for v in graph.callees(u):
            if v in subgraph and v not in prev:
                prev[v] = u
                q.append(v)
    if dst not in prev:
        return []
    path = [dst]
    while prev[path[-1]]:
        path.append(prev[path[-1]])
    return list(reversed(path))
