"""数据结构定义:函数节点、调用图、事实、题目。"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any


@dataclass
class FunctionNode:
    """函数调用图中的一个节点。"""
    qualname: str                    # 唯一标识,例如 "pkg.mod.ClassA.predict"
    file_path: str = ""
    code: str = ""                   # 函数源码(含签名)
    start_line: int = 0
    end_line: int = 0
    docstring: str = ""
    language: str = ""               # python / typescript / javascript / go / c
    project: str = ""                # openclaw / ollama / ...
    is_test: bool = False
    is_public_api: bool = False
    node_kind: str = ""

    # 运行时填充的特征
    out_degree: int = 0
    in_degree: int = 0
    k_hop_size: int = 0
    cross_file_edges: int = 0
    branch_count: int = 0
    has_try_except: bool = False
    has_return_value: bool = False
    has_raise: bool = False
    has_logging_calls: bool = False
    callee_names: list[str] = field(default_factory=list)
    is_wrapper_like: bool = False
    is_test_related: bool = False

    @property
    def short_name(self) -> str:
        return self.qualname.rsplit(".", 1)[-1]


@dataclass
class CallGraph:
    """函数调用图。用邻接表表达 caller -> [callees]。"""
    nodes: dict[str, FunctionNode] = field(default_factory=dict)
    edges: dict[str, set[str]] = field(default_factory=dict)      # caller -> {callees}
    rev_edges: dict[str, set[str]] = field(default_factory=dict)  # callee -> {callers}

    def add_node(self, node: FunctionNode) -> None:
        self.nodes[node.qualname] = node
        self.edges.setdefault(node.qualname, set())
        self.rev_edges.setdefault(node.qualname, set())

    def add_edge(self, caller: str, callee: str) -> None:
        if caller not in self.nodes or callee not in self.nodes:
            return
        self.edges[caller].add(callee)
        self.rev_edges[callee].add(caller)

    def callees(self, name: str) -> set[str]:
        return self.edges.get(name, set())

    def callers(self, name: str) -> set[str]:
        return self.rev_edges.get(name, set())

    def k_hop_subgraph(self, root: str, k: int = 2) -> set[str]:
        """从 root 出发 BFS,取 k 跳内的后继节点(含 root)。"""
        visited = {root}
        frontier = {root}
        for _ in range(k):
            nxt: set[str] = set()
            for u in frontier:
                for v in self.callees(u):
                    if v not in visited:
                        nxt.add(v)
                        visited.add(v)
            frontier = nxt
            if not frontier:
                break
        return visited


@dataclass
class Fact:
    """从子图 + AST 中抽取的结构化事实。"""
    kind: str                        # main_path / result_origin / conditional / error / role / stages
    root: str
    payload: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class BenchmarkItem:
    """最终输出的一道题。"""
    question_type: str
    root_function: str
    question: str
    gold_answer: dict[str, Any]       # 结构化答案
    natural_answer: str = ""          # 自然语言答案(可由 Opus 生成)
    subgraph_nodes: list[str] = field(default_factory=list)
    structure_gold_hash: str = ""
    benchmark_tier: str = "main_experiment"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
