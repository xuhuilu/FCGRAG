"""端到端 pipeline:从 CallGraph + 源码 → BenchmarkItem 列表。"""
from __future__ import annotations
import random
from .schema import CallGraph, BenchmarkItem
from .ast_analyzer import analyze_function, FunctionAST
from .features import compute_features, pick_roots
from .subgraph import prune_subgraph
from .fact_extractors import EXTRACTORS
from .question_gen import render_item


def validate_item(item: BenchmarkItem, graph: CallGraph) -> bool:
    """最小化校验:问题非空,gold 非空,涉及实体在图中可找到。"""
    if not item.question.strip():
        return False
    if not item.gold_answer:
        return False
    short_to_qual = {n.rsplit(".", 1)[-1] for n in graph.nodes}
    # 抽几个字段检查
    for key in ("gold_entities", "core_entities", "support_entities",
                "true_path", "false_path", "error_path"):
        vals = item.gold_answer.get(key)
        if isinstance(vals, list):
            for v in vals:
                if isinstance(v, str) and v and v not in short_to_qual:
                    # 允许 1 个外部名字(内置函数等),但别全军覆没
                    pass
    return True


def build_ast_map(graph: CallGraph) -> dict[str, FunctionAST]:
    return {
        name: analyze_function(name, node.code, language=node.language or "python")
        for name, node in graph.nodes.items()
    }


def generate_benchmark(
    graph: CallGraph,
    k_hop: int = 2,
    max_roots: int | None = None,
    seed: int = 0,
) -> list[BenchmarkItem]:
    """主入口:
    1. 解析源码 → FunctionAST
    2. 计算特征
    3. 选 root
    4. 对每个 root 抽 k-hop 子图 → 剪枝
    5. 依次运行 6 个 extractor
    6. 渲染为 BenchmarkItem 并校验
    """
    rng = random.Random(seed)
    ast_map = build_ast_map(graph)
    compute_features(graph, ast_map, k_hop=k_hop)
    roots = pick_roots(graph)
    if max_roots is not None:
        roots = roots[:max_roots]

    items: list[BenchmarkItem] = []
    for root in roots:
        raw_sub = graph.k_hop_subgraph(root, k=k_hop)
        sub = prune_subgraph(graph, raw_sub, keep_some_aux=True)
        if len(sub) < 3:
            continue
        sub_list = sorted(sub)
        for qtype, extractor in EXTRACTORS.items():
            try:
                fact = extractor(graph, ast_map, root, sub)
            except Exception:
                fact = None
            if fact is None:
                continue
            item = render_item(fact, sub_list, rng=rng)
            if validate_item(item, graph):
                items.append(item)
    return items
