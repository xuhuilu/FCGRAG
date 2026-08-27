"""给每个函数打特征,并筛选出适合作为 root 的函数。"""
from __future__ import annotations
from .schema import CallGraph, FunctionNode
from .ast_analyzer import FunctionAST


_CORE_PREFIXES = (
    "run", "handle_", "predict", "forward", "execute", "train_step",
    "load_", "create_", "build_", "compute", "process_", "dispatch",
    "handle", "create", "build", "compute", "process", "dispatch",
    "serve", "generate", "parse", "render", "resolve", "convert",
)
_AUX_PREFIXES = (
    "log_", "debug_", "print_", "helper_", "util_", "_log", "_debug",
    "log", "debug", "trace", "assert", "printf", "fprintf",
)
_WRAPPER_BODY_LIMIT = 3   # 函数体调用 <=3 且只有一个非 log 调用 → 像 wrapper


def compute_features(
    graph: CallGraph,
    ast_map: dict[str, FunctionAST],
    k_hop: int = 2,
) -> None:
    """就地填充 graph.nodes[*] 上的特征字段。"""
    for name, node in graph.nodes.items():
        node.out_degree = len(graph.callees(name))
        node.in_degree = len(graph.callers(name))
        node.k_hop_size = len(graph.k_hop_subgraph(name, k=k_hop))
        node.cross_file_edges = sum(
            1 for c in graph.callees(name)
            if c in graph.nodes and graph.nodes[c].file_path != node.file_path
        )
        node.is_test_related = (
            node.is_test
            or "test" in name.lower().split(".")
            or node.short_name.startswith("test_")
        )

        fa = ast_map.get(name)
        if fa is None:
            node.callee_names = [graph.nodes[c].short_name for c in graph.callees(name)]
            continue

        node.branch_count = len(fa.branches)
        node.has_try_except = bool(fa.tries)
        node.has_return_value = fa.has_return_value
        node.has_raise = bool(fa.raises)
        node.has_logging_calls = fa.has_logging_calls
        node.callee_names = fa.all_calls or [
            graph.nodes[c].short_name for c in graph.callees(name)
        ]

        non_aux_calls = [c for c in fa.all_calls if not is_aux_like_name(c)]
        node.is_wrapper_like = (
            len(fa.all_calls) <= _WRAPPER_BODY_LIMIT and len(set(non_aux_calls)) <= 1
        )


def is_core_like_name(short_name: str) -> bool:
    return short_name.startswith(_CORE_PREFIXES)


def is_aux_like_name(short_name: str) -> bool:
    low = (short_name or "").lower()
    return low.startswith(_AUX_PREFIXES)


def pick_roots(
    graph: CallGraph,
    min_out: int = 2,
    max_out: int = 12,
    min_subgraph: int = 4,
) -> list[str]:
    """按文档里的启发式选 root:
    - 出度 2~8
    - 非 test
    - 非纯 wrapper
    - 子图规模够大
    - 名字像 orchestrator 会加分
    """
    scored: list[tuple[float, str]] = []
    for name, node in graph.nodes.items():
        if node.is_test_related:
            continue
        if node.is_wrapper_like:
            continue
        if not (min_out <= node.out_degree <= max_out):
            continue
        if node.k_hop_size < min_subgraph:
            continue

        score = 0.0
        score += node.k_hop_size * 0.5
        score += node.branch_count * 0.8
        score += 2.0 if node.has_try_except else 0.0
        score += 1.5 if is_core_like_name(node.short_name) else 0.0
        score += 1.0 if node.has_return_value else 0.0
        score -= 2.0 if is_aux_like_name(node.short_name) else 0.0
        scored.append((score, name))

    scored.sort(reverse=True)
    return [name for _, name in scored]
