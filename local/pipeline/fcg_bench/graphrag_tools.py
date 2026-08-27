"""
GraphRAG 在线工具:提供四个只读工具供主流程调用。

和 LocAgent 的三工具对比:
  - LocAgent: SearchEntity, TraverseGraph, RetrieveEntity (通用)
  - 我们:     get_function_card, get_local_subgraph(tree),
             get_branch_details, get_try_except_details
             (后两个是"结构专用"工具,LocAgent 没有)

LocalContext 是喂给 LLM 的上下文包,按题型可以切片。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any
import re

from .schema import CallGraph, FunctionNode
from .ast_analyzer import FunctionAST, BranchInfo, TryInfo


_PSEUDO_CALLERS = {
    "<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>", "<lambda>",
    "listcomp", "dictcomp", "setcomp", "genexpr", "lambda",
    "IMPORTED_SYMBOL_PSEUDO", "IMPORTED_MODULE_FUNCTION_PSEUDO",
}
_SYNTHETIC_LOCATION_RE = re.compile(r"^(?:[cm]?[jt]s|mjs|c|cc|cpp|go|py):\d+>$", re.I)
_TEST_NAME_RE = re.compile(r"(^|[._/-])(test|tests|spec|__tests__)([._/-]|$)|^test_|_test$", re.I)
_MAX_REVERSE_DIRECT = 15
_MAX_REVERSE_INDIRECT = 10
_MAX_REVERSE_POTENTIAL = 12


@dataclass
class LocalContext:
    """一道题的所有结构化上下文。按题型切片后送入 LLM。"""
    root_qualname: str
    root_short: str
    root_source: str                    # root 的完整源码
    root_docstring: str
    root_ast: FunctionAST | None        # 直接引用 FunctionAST
    main_path: list[str]                # 主链 short names(含 root)
    subgraph_tree: str                  # 带控制流标注的树形子图
    callees: list[dict[str, Any]]       # 每个 callee 的摘要信息
    return_expr: str
    return_source_func: str


class GraphRAGTools:
    """
    在线只读工具。所有方法返回纯文本或纯 dict,不返回对象引用,方便拼 prompt。
    """

    def __init__(
        self,
        cg: CallGraph,
        ast_map: dict[str, FunctionAST],
        offline_cache: dict[str, Any],
    ):
        self.cg = cg
        self.ast_map = ast_map
        self.main_paths: dict[str, list[str]] = offline_cache["main_paths"]
        self.trees: dict[str, str] = offline_cache["trees"]
        self.role_hints: dict[str, str] = offline_cache["role_hints"]

        # short_name → list[qualname] 索引(应对 root 用 short 名字或不完整路径)
        self._short_to_qn: dict[str, list[str]] = {}
        for qn in cg.nodes:
            short = qn.rsplit(".", 1)[-1]
            self._short_to_qn.setdefault(short, []).append(qn)

    # -------- 查找 --------
    def resolve_root(self, root_ref: str) -> str | None:
        """题目里的 root_function 可能是 short name 或完整 qualname,都要能找到。"""
        if root_ref in self.cg.nodes:
            return root_ref
        short = root_ref.rsplit(".", 1)[-1]
        candidates = self._short_to_qn.get(short, [])
        if not candidates:
            return None
        if len(candidates) == 1:
            return candidates[0]
        # 多候选:选和 root_ref 前缀最接近的
        for c in candidates:
            if root_ref in c or c in root_ref:
                return c
        return candidates[0]

    # -------- 工具 1:get_function_card --------
    def get_function_card(self, qualname: str) -> dict[str, Any] | None:
        """
        返回一个函数的完整"档案"。这不是新类,只是把现有字段打包成 dict。
        """
        node = self.cg.nodes.get(qualname)
        if node is None:
            return None
        fa = self.ast_map.get(qualname)
        return {
            "qualname": qualname,
            "short_name": node.short_name,
            "file_path": node.file_path,
            "source_code": node.code,
            "docstring": node.docstring or "",
            "role_hint": self.role_hints.get(qualname, "business"),
            "out_degree": node.out_degree,
            "in_degree": node.in_degree,
            "has_return_value": fa.has_return_value if fa else False,
            "return_expr": fa.return_expr if fa else "",
            "return_source_func": fa.return_source_func if fa else "",
            "num_branches": len(fa.branches) if fa else 0,
            "num_tries": len(fa.tries) if fa else 0,
        }

    # -------- 工具 2:get_local_subgraph (tree 格式) --------
    def get_local_subgraph(self, qualname: str, format: str = "tree") -> str:
        """
        返回带控制流标注的树形子图。
        离线已经预算了的直接查缓存;没预算的降级到邻接表。
        """
        if qualname in self.trees:
            return self.trees[qualname]

        # 降级:生成一个简单的邻接表
        node = self.cg.nodes.get(qualname)
        if node is None:
            return f"<not found: {qualname}>"
        callees = sorted(self.cg.callees(qualname))
        lines = [node.short_name]
        for i, c in enumerate(callees):
            prefix = "└──" if i == len(callees) - 1 else "├──"
            c_short = self.cg.nodes[c].short_name
            role = self.role_hints.get(c, "?")
            lines.append(f"{prefix} calls ── {c_short}  [{role}]")
        return "\n".join(lines)

    # -------- 工具 3:get_branch_details --------
    def get_branch_details(self, qualname: str) -> list[dict[str, Any]]:
        """
        返回这个函数里所有 if/else 分支的结构化信息。
        这是你相对 LocAgent 的特有工具 —— 针对 conditional_path 题型。
        """
        fa = self.ast_map.get(qualname)
        if fa is None:
            return []
        return [
            {
                "condition": b.condition,
                "branch_kind": b.branch_kind,
                "path_calls": b.path_calls,
                "excluded_branches": b.excluded_branches,
                "start_line": b.start_line,
                "end_line": b.end_line,
                "true_calls": b.true_calls,
                "false_calls": b.false_calls,
            }
            for b in fa.branches
        ]

    # -------- 工具 4:get_try_except_details --------
    def get_try_except_details(self, qualname: str) -> list[dict[str, Any]]:
        """
        返回这个函数里所有 try/except 的结构化信息。
        对应 error_handling_path 题型。
        """
        fa = self.ast_map.get(qualname)
        if fa is None:
            return []
        return [
            {
                "start_line": t.start_line,
                "end_line": t.end_line,
                "try_calls": t.try_calls,
                "except_calls": t.except_calls,
                "raises": t.raises,
                "except_returns": t.except_returns,
                "warns": t.warns,
            }
            for t in fa.tries
        ]

    # -------- 工具 5:k-hop forward BFS(用于 dependency_trace 题型) --------
    def get_khop_descendants(
        self, qualname: str, max_hops: int = 3, max_nodes: int = 80,
    ) -> list[dict[str, Any]]:
        """
        从 qualname 出发做 k 跳前向 BFS,返回每个可达节点的信息。
        每个节点带 hop 距离和短名,用于 LLM 判断"是否属于目标类别"。
        """
        if qualname not in self.cg.nodes:
            return []
        visited = {qualname: 0}
        frontier = [qualname]
        for hop in range(1, max_hops + 1):
            next_frontier: list[str] = []
            for u in frontier:
                for v in self.cg.callees(u):
                    if v not in visited:
                        visited[v] = hop
                        next_frontier.append(v)
                        if len(visited) >= max_nodes:
                            break
                if len(visited) >= max_nodes:
                    break
            frontier = next_frontier
            if len(visited) >= max_nodes:
                break

        results = []
        for qn, hop in visited.items():
            if qn == qualname:
                continue
            node = self.cg.nodes.get(qn)
            if node is None:
                continue
            results.append({
                "short_name": node.short_name,
                "hop": hop,
                "role_hint": self.role_hints.get(qn, "business"),
            })
        return sorted(results, key=lambda x: (x["hop"], x["short_name"]))

    # -------- 工具 6:reverse callers(用于 impact_scope 题型) --------
    def get_reverse_callers(
        self, qualname: str, max_hops: int = 2,
    ) -> dict[str, list[str]]:
        """
        反向 BFS,返回:
          {
            "direct":   [short_name, ...],   # 1 跳调用者
            "indirect": [short_name, ...],   # 2 跳调用者(不含 direct)
          }
        """
        if qualname not in self.cg.nodes:
            return {"direct": [], "indirect": []}

        direct_qns = {
            q for q in self.cg.rev_edges.get(qualname, set())
            if self._is_valid_static_caller(q)
        }
        indirect_qns: set[str] = set()
        for c in direct_qns:
            indirect_qns |= {
                q for q in self.cg.rev_edges.get(c, set())
                if self._is_valid_static_caller(q)
            }
        indirect_qns -= direct_qns
        indirect_qns.discard(qualname)

        direct = sorted(set(
            self.cg.nodes[q].short_name for q in direct_qns if q in self.cg.nodes
        ))[:_MAX_REVERSE_DIRECT]
        indirect = sorted(set(
            self.cg.nodes[q].short_name for q in indirect_qns if q in self.cg.nodes
        ))[:_MAX_REVERSE_INDIRECT]
        potential = []
        for name in direct + indirect:
            if name not in potential:
                potential.append(name)
        return {
            "direct": direct,
            "indirect": indirect,
            "potential": potential[:_MAX_REVERSE_POTENTIAL],
        }

    def _is_valid_static_caller(self, qualname: str) -> bool:
        node = self.cg.nodes.get(qualname)
        if node is None:
            return False
        short = node.short_name
        full = node.qualname or qualname
        kind = (node.node_kind or "").upper()
        if not short or short in _PSEUDO_CALLERS or full in _PSEUDO_CALLERS:
            return False
        if _SYNTHETIC_LOCATION_RE.match(short):
            return False
        if node.is_test_related or node.is_test or _TEST_NAME_RE.search(short) or _TEST_NAME_RE.search(full):
            return False
        if short.startswith("<") and short.endswith(">"):
            return False
        if "IMPORTED_" in short or "IMPORTED_" in qualname or "IMPORTED_" in full:
            return False
        if short.lower() in _PSEUDO_CALLERS:
            return False
        if any(part in _PSEUDO_CALLERS for part in full.lower().replace("<", "").replace(">", "").split(".")):
            return False
        if kind in {"CLASS", "MODULE"}:
            return False
        if node.start_line <= 0 or node.end_line <= 0:
            return False
        if not node.code.strip() and qualname not in self.ast_map:
            return False
        return True

    # -------- 组装 LocalContext --------
    def build_local_context(self, root_ref: str) -> LocalContext | None:
        root_qn = self.resolve_root(root_ref)
        if root_qn is None:
            return None
        node = self.cg.nodes[root_qn]
        fa = self.ast_map.get(root_qn)

        # 准备每个 callee 的摘要
        callee_summaries: list[dict[str, Any]] = []
        for c_qn in sorted(self.cg.callees(root_qn)):
            c_node = self.cg.nodes[c_qn]
            callee_summaries.append({
                "qualname": c_qn,
                "short_name": c_node.short_name,
                "role_hint": self.role_hints.get(c_qn, "business"),
                "docstring_1line": (c_node.docstring or "").split("\n")[0][:100],
                "out_degree": c_node.out_degree,
            })

        return LocalContext(
            root_qualname=root_qn,
            root_short=node.short_name,
            root_source=node.code,
            root_docstring=node.docstring or "",
            root_ast=fa,
            main_path=self.main_paths.get(root_qn, [node.short_name]),
            subgraph_tree=self.get_local_subgraph(root_qn, format="tree"),
            callees=callee_summaries,
            return_expr=fa.return_expr if fa else "",
            return_source_func=fa.return_source_func if fa else "",
        )
