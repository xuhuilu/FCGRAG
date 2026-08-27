"""
GraphRAG 离线阶段:为真实 RAG 系统预计算三样东西并缓存到 JSON。

1. main_paths:   每个 root 函数的主链(按源码顺序,跳过分支/异常调用)
2. annotated_trees: 每个 root 函数的带控制流标注的树形子图(LocAgent tree 格式的扩展)
3. role_hints:   每个函数的启发式角色标签(orchestrator/factory/helper/leaf)

输出一个 offline_cache.json,在线阶段加载即可。不引入任何新 dataclass。
"""
from __future__ import annotations
import json
from typing import Any

from .schema import CallGraph, FunctionNode
from .ast_analyzer import FunctionAST
from .features import pick_roots
from .subgraph import extract_main_path


# ============================================================
# 派生字段 1:role_hint —— 启发式函数角色标签
# ============================================================

_FACTORY_PREFIXES = ("create_", "make_", "build_", "_from_", "from_")
_HELPER_PREFIXES = (
    "_get_", "get_", "_format_", "format_", "_to_", "to_",
    "_is_", "is_", "_has_", "has_", "_check_", "check_",
    "_ensure_", "_parse_", "parse_", "_validate_", "validate_",
)
_LOG_PREFIXES = ("log_", "_log", "_debug", "warn_", "_warn")


def compute_role_hint(node: FunctionNode) -> str:
    """
    返回一个粗糙但有用的角色标签。喂给 LLM 作为 core/support 判断的先验。

    优先级(从高到低):
      - leaf:      出度 0,纯叶子
      - logger:    名字以 log/warn/debug 开头
      - factory:   名字含 create/make/build/from
      - orchestrator: 入度和出度都 >= 3(调度中枢)
      - helper:    get/format/parse/validate 等纯辅助动作
      - business:  以上都不是(多半是业务函数)
    """
    short = node.short_name

    if node.out_degree == 0:
        return "leaf"

    if any(short.startswith(p) for p in _LOG_PREFIXES):
        return "logger"

    if any(short.startswith(p) for p in _FACTORY_PREFIXES):
        return "factory"

    if node.in_degree >= 3 and node.out_degree >= 3:
        return "orchestrator"

    if any(short.startswith(p) for p in _HELPER_PREFIXES):
        return "helper"

    return "business"


# ============================================================
# 派生字段 2:annotated_tree —— 带控制流标注的树形子图
# ============================================================

def _clean_call_name(name: str) -> str | None:
    """过滤掉噪声调用名。"""
    NOISE = {
        "isinstance", "issubclass", "hasattr", "getattr", "setattr",
        "type", "cast", "super", "len", "print", "repr",
        "int", "str", "list", "dict", "set", "tuple", "bool",
    }
    if name in NOISE:
        return None
    return name


def build_annotated_tree(
    root_qn: str,
    cg: CallGraph,
    ast_map: dict[str, FunctionAST],
    main_path_shorts: list[str],
    max_hops: int = 2,
) -> str:
    """
    产出一个类似这样的文本:

    handle_request
    ├── calls ── parse_request        [main path, step 1]
    ├── branch[req.needs_auth]
    │   ├── true  → check_auth
    │   └── false → skip_auth
    ├── try-block
    │   ├── calls ── validate_request  [main path, step 2]
    │   └── calls ── execute_task      [main path, step 3]
    ├── except-block
    │   └── calls ── log_error
    └── post ── log_metrics

    它比 LocAgent 的 tree 多了三种标注:
      - [main path, step N]
      - branch[condition]
      - try-block / except-block
    """
    if root_qn not in cg.nodes:
        return f"<root {root_qn} not found>"

    root_node = cg.nodes[root_qn]
    fa = ast_map.get(root_qn)
    lines: list[str] = [root_node.short_name]

    # 建立 short_name → main_path 的 step 编号映射
    main_step: dict[str, int] = {}
    for i, short in enumerate(main_path_shorts):
        if i == 0:
            continue  # 第一个是 root 自己
        if short not in main_step:
            main_step[short] = i

    if fa is None or not (fa.branches or fa.tries or fa.call_sites):
        # 没有 AST 信息:退化成简单的邻居列表
        callees = sorted(cg.callees(root_qn))
        for i, c in enumerate(callees):
            prefix = "└──" if i == len(callees) - 1 else "├──"
            short = cg.nodes[c].short_name
            tag = f"  [main path, step {main_step[short]}]" if short in main_step else ""
            lines.append(f"{prefix} calls ── {short}{tag}")
        return "\n".join(lines)

    # ---- 1. 主执行顺序的 call_sites(非分支、非异常)----
    seen_main_calls: set[str] = set()
    main_calls_ordered: list[str] = []
    for cs in sorted(fa.call_sites, key=lambda c: c.lineno):
        if cs.in_branch or cs.in_except:
            continue
        cleaned = _clean_call_name(cs.func_name)
        if cleaned is None or cleaned in seen_main_calls:
            continue
        seen_main_calls.add(cleaned)
        main_calls_ordered.append(cleaned)

    # ---- 2. 分支块 ----
    branch_blocks: list[tuple[str, list[str], list[str]]] = []
    for b in fa.branches:
        tc = [c for c in (_clean_call_name(x) for x in b.true_calls) if c]
        fc = [c for c in (_clean_call_name(x) for x in b.false_calls) if c]
        if tc or fc:
            branch_blocks.append((b.condition, tc, fc))

    # ---- 3. try/except 块 ----
    try_blocks: list[tuple[list[str], list[str]]] = []
    for t in fa.tries:
        tc = [c for c in (_clean_call_name(x) for x in t.try_calls) if c]
        ec = [c for c in (_clean_call_name(x) for x in t.except_calls) if c]
        if tc or ec:
            try_blocks.append((tc, ec))

    # ---- 组装输出 ----
    # 用嵌套结构表示:每个 section 是 (head, [child_lines])
    sections: list[tuple[str, list[str]]] = []

    if main_calls_ordered:
        for c in main_calls_ordered[:8]:
            tag = f"  [main path, step {main_step[c]}]" if c in main_step else ""
            sections.append((f"calls ── {c}{tag}", []))

    for cond, tc, fc in branch_blocks[:3]:
        children = []
        if tc:
            children.append(f"true  → {' → '.join(tc[:4])}")
        if fc:
            children.append(f"false → {' → '.join(fc[:4])}")
        sections.append((f"branch[{cond[:60]}]", children))

    for tc, ec in try_blocks[:2]:
        children = []
        for c in tc[:4]:
            tag = f"  [main path, step {main_step[c]}]" if c in main_step else ""
            children.append(f"calls ── {c}{tag}")
        sections.append(("try-block", children))
        if ec:
            ex_children = [f"calls ── {c}" for c in ec[:4]]
            sections.append(("except-block", ex_children))

    # 渲染
    out_lines = [root_node.short_name]
    for i, (head, children) in enumerate(sections):
        is_last = (i == len(sections) - 1)
        branch_char = "└──" if is_last else "├──"
        out_lines.append(f"{branch_char} {head}")
        # child 缩进:最后一个 section 用空格,其他用 │
        indent = "    " if is_last else "│   "
        for j, child in enumerate(children):
            child_char = "└──" if j == len(children) - 1 else "├──"
            out_lines.append(f"{indent}{child_char} {child}")

    return "\n".join(out_lines)


# ============================================================
# 主入口:构建完整缓存
# ============================================================

def build_offline_cache(
    cg: CallGraph,
    ast_map: dict[str, FunctionAST],
    max_roots: int = 50,
    k_hop: int = 2,
) -> dict[str, Any]:
    """
    遍历 pick_roots 选中的 root,预计算三样东西。
    返回一个可 JSON 序列化的 dict。

    为了让 GraphRAG 能应对"题目里的 root 不在 pick_roots 的 top-50"的情况,
    我们实际上对所有入度+出度之和前 300 的节点都算一遍。
    """
    # 1. role_hints —— 对全图每个节点
    role_hints: dict[str, str] = {
        qn: compute_role_hint(node) for qn, node in cg.nodes.items()
    }

    # 2. 确定要预算 main_path 和 tree 的候选 root 集合
    # pick_roots 返回的是"合格 root"的列表,取前 max_roots 个
    all_picked = pick_roots(cg)
    picked = set(all_picked[:max_roots])
    # 再 union 上按出入度排前 300 的节点,应对"题目里的 root 不在 pick_roots 名单"
    by_degree = sorted(
        cg.nodes.values(),
        key=lambda n: n.out_degree + n.in_degree,
        reverse=True,
    )[:300]
    roots = picked | {n.qualname for n in by_degree}

    main_paths: dict[str, list[str]] = {}
    trees: dict[str, str] = {}

    for root in roots:
        try:
            subgraph = cg.k_hop_subgraph(root, k=k_hop)
            path_quals = extract_main_path(
                cg, root, subgraph, ast_map=ast_map, max_len=8
            )
            path_shorts = [cg.nodes[q].short_name for q in path_quals]
            main_paths[root] = path_shorts

            tree_str = build_annotated_tree(
                root, cg, ast_map, path_shorts, max_hops=k_hop
            )
            trees[root] = tree_str
        except Exception as e:
            # 单个节点失败不影响整体
            main_paths[root] = [cg.nodes[root].short_name]
            trees[root] = f"{cg.nodes[root].short_name}\n<tree build error: {e}>"

    return {
        "main_paths": main_paths,
        "trees": trees,
        "role_hints": role_hints,
        "meta": {
            "num_nodes": len(cg.nodes),
            "num_roots_cached": len(roots),
            "k_hop": k_hop,
        },
    }


def save_offline_cache(cache: dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cache, f, ensure_ascii=False, indent=2)


def load_offline_cache(path: str) -> dict[str, Any]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)
