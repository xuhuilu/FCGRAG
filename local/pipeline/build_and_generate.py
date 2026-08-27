#!/usr/bin/env python3
"""
从 merged.csv(15 列格式)读取调用图,同时完成两件事:

1. 构造 networkx.Graph(无向),写出 .graphml —— 与原 build_graph.py 节点/边格式完全一致
2. 构造 fcg_bench.CallGraph(有向),送进 generate_benchmark 产出 6 类题目

两条路径独立填充,不做互相转换,保证 graphml 输出和原脚本字节级一致。

过滤规则:
    - 伪边 (ATTRIBUTE_FALLBACK / ATTR_PSEUDO) → 丢弃
    - callee / caller 行号为 0 → 丢弃
    - 空名字 → 丢弃
    - caller_kind 为 class 的边(装饰器/字段默认值等噪声)→ 丢弃 ← 新增
    - callee 是常见伪调用(cast、super、isinstance 等)→ 丢弃 ← 新增
    - 自环 → 丢弃
"""
from __future__ import annotations
import argparse
import csv
import itertools
import json
import sys
from pathlib import Path
import networkx as nx

from fcg_bench import CallGraph, FunctionNode, generate_benchmark


# =========================================================================
# 配置
# =========================================================================

INPUT_CSV      = "merged.csv"
OUTPUT_GRAPHML = "result_nx_standard_dify.graphml"
OUTPUT_JSONL   = "benchmark_output.jsonl"
SOURCE_ID      = "4a25dab6bbc7b764367e4d6baadd5a05"

SKIP_ZERO_CALLER  = True
SKIP_ZERO_CALLEE  = True
SKIP_PSEUDO_EDGES = True
SKIP_MODULE_CALLER = True
SKIP_CLASS_CALLER = True
SKIP_FAKE_CALLEE  = True

PSEUDO_EDGE_SOURCES = {"ATTRIBUTE_FALLBACK", "ATTR_PSEUDO"}

LANGUAGE_RULES = {
    "python": {
        "test_suffixes": ("_test.py",),
        "test_prefixes": ("test_",),
        "fake_callees": {
            "cast", "TYPE_CHECKING",
            "super",
            "isinstance", "issubclass", "hasattr", "getattr", "setattr", "delattr",
            "callable", "type", "id", "vars", "dir",
            "len", "bool", "int", "float", "tuple", "list", "dict", "set", "frozenset",
            "print", "repr", "format",
            "property", "staticmethod", "classmethod", "abstractmethod",
            "cached_property", "overload", "final", "override",
            "lru_cache", "cache", "wraps",
        },
    },
    "typescript": {
        "test_suffixes": (".test.ts", ".spec.ts", ".test.tsx", ".spec.tsx"),
        "test_prefixes": (),
        "fake_callees": {
            "map", "filter", "reduce", "forEach", "then", "catch",
            "JSON.stringify", "JSON.parse", "String", "Number", "Boolean", "Array",
            "Object", "Promise", "console.log", "console.warn", "console.error",
        },
    },
    "javascript": {
        "test_suffixes": (".test.js", ".spec.js", ".test.jsx", ".spec.jsx", ".test.mjs", ".spec.mjs"),
        "test_prefixes": (),
        "fake_callees": {
            "map", "filter", "reduce", "forEach", "then", "catch",
            "JSON.stringify", "JSON.parse", "String", "Number", "Boolean", "Array",
            "Object", "Promise", "console.log", "console.warn", "console.error",
        },
    },
    "go": {
        "test_suffixes": ("_test.go",),
        "test_prefixes": (),
        "fake_callees": {"len", "cap", "make", "new", "append", "copy", "delete", "panic", "recover"},
    },
    "c": {
        "test_suffixes": ("_test.c", "_test.h"),
        "test_prefixes": (),
        "fake_callees": {"malloc", "free", "calloc", "realloc", "memcpy", "memset", "strlen", "printf"},
    },
}

FIELD_ALIASES = {
    "caller_kind": ("caller_kind", "caller_type", "source_kind", "source_type"),
    "caller_name": ("caller_name", "caller", "source_name", "source"),
    "caller_start_line": ("caller_start_line", "caller_start", "caller_lineno", "source_start_line"),
    "caller_end_line": ("caller_end_line", "caller_end", "source_end_line"),
    "caller_file": ("caller_file", "caller_path", "source_file", "source_path"),
    "call_kind": ("call_kind", "relation_type", "edge_kind", "call_type"),
    "call_start_line": ("call_start_line", "call_start", "rel_start"),
    "call_end_line": ("call_end_line", "call_end", "rel_end"),
    "call_file": ("call_file", "call_path", "rel_path"),
    "callee_kind": ("callee_kind", "callee_type", "target_kind", "target_type"),
    "callee_name": ("callee_name", "callee", "target_name", "target"),
    "callee_start_line": ("callee_start_line", "callee_start", "target_start_line"),
    "callee_end_line": ("callee_end_line", "callee_end", "target_end_line"),
    "callee_file": ("callee_file", "callee_path", "target_file", "target_path"),
    "edge_source": ("edge_source", "resolved", "edge_resolved", "source"),
}

COL_CALLER_KIND, COL_CALLER_NAME, COL_CALLER_START, COL_CALLER_END, COL_CALLER_FILE = 0, 1, 2, 3, 4
COL_CALL_KIND, COL_CALL_START, COL_CALL_END, COL_CALL_FILE                          = 5, 6, 7, 8
COL_CALLEE_KIND, COL_CALLEE_NAME, COL_CALLEE_START, COL_CALLEE_END, COL_CALLEE_FILE = 9, 10, 11, 12, 13
COL_EDGE_SOURCE                                                                     = 14

EXPECTED_HEADER = [
    "caller_kind", "caller_name", "caller_start_line", "caller_end_line", "caller_file",
    "call_kind", "call_start_line", "call_end_line", "call_file",
    "callee_kind", "callee_name", "callee_start_line", "callee_end_line", "callee_file",
    "edge_source",
]

HEADER_ALIASES = {alias for aliases in FIELD_ALIASES.values() for alias in aliases}


# =========================================================================
# 辅助(和原 build_graph.py 完全一致)
# =========================================================================

def detect_language(path: str) -> str:
    p = (path or "").lower()
    if p.endswith((".ts", ".tsx")):
        return "typescript"
    if p.endswith((".js", ".jsx", ".mjs", ".cjs")):
        return "javascript"
    if p.endswith(".go"):
        return "go"
    if p.endswith((".c", ".h", ".cc", ".cpp", ".cxx", ".hpp")):
        return "c"
    if p.endswith(".py"):
        return "python"
    return ""


def _resolve_path(file_path: str, source_root: str = "") -> str:
    if not file_path:
        return ""
    p = Path(file_path)
    if p.is_absolute() or not source_root:
        return str(p)
    return str(Path(source_root) / p)


def _row_value(row: dict[str, str], canonical: str, default: str = "") -> str:
    for key in FIELD_ALIASES.get(canonical, (canonical,)):
        if key in row and row[key] is not None:
            return row[key]
    return default


def _looks_like_header(row: list[str] | None) -> bool:
    if not row:
        return False
    normalized = {str(col).strip() for col in row}
    return bool(normalized & HEADER_ALIASES)


def _list_row_to_dict(row: list[str]) -> dict[str, str]:
    return {
        name: row[idx] if idx < len(row) else ""
        for idx, name in enumerate(EXPECTED_HEADER)
    }


def _slice_c_like_function(lines: list[str], start: int, end: int) -> str:
    """Expand a C/C++ function declaration line to the matching braced body."""
    start_idx = max(0, start - 1)
    search_idx = start_idx
    while search_idx < len(lines) and "{" not in lines[search_idx]:
        search_idx += 1
        if search_idx - start_idx > 20:
            break
    if search_idx >= len(lines) or "{" not in lines[search_idx]:
        end_idx = min(len(lines), max(end, start))
        return "".join(lines[start_idx:end_idx])

    depth = 0
    seen_open = False
    for idx in range(search_idx, len(lines)):
        line = lines[idx]
        for ch in line:
            if ch == "{":
                depth += 1
                seen_open = True
            elif ch == "}" and seen_open:
                depth -= 1
                if depth == 0:
                    return "".join(lines[start_idx:idx + 1])
    end_idx = min(len(lines), max(end, start))
    return "".join(lines[start_idx:end_idx])


def get_source_code(file_path, start_line, end_line, source_root: str = "", language: str = ""):
    """Read source code lines from a file, robust against bad inputs."""
    try:
        full_path = _resolve_path(file_path, source_root)
        if not full_path:
            return "No source file."
        start = int(start_line)
        end = int(end_line)
        if start <= 0 or end <= 0:
            return "No valid line range."
        with open(full_path, "r", encoding="utf-8", errors="ignore") as f:
            lines = f.readlines()
        if (language or detect_language(file_path)) == "c":
            code = _slice_c_like_function(lines, start, end)
            return code.strip() if code else "No source code found."
        start_idx = max(0, start - 1)
        end_idx = min(len(lines), end)
        code = "".join(lines[start_idx:end_idx])
        return code.strip() if code else "No source code found."
    except Exception as e:
        return f"Error reading source: {e}"


def _is_zero(value):
    if value is None:
        return True
    s = str(value).strip()
    if not s:
        return True
    try:
        return int(s) == 0
    except ValueError:
        return True


def _safe_int(value, default=0):
    try:
        return int(str(value).strip())
    except (ValueError, TypeError):
        return default


def _is_class_caller(kind: str) -> bool:
    return kind.strip().upper() == "CLASS"


def _is_module_caller(kind: str) -> bool:
    return kind.strip().upper() == "MODULE"


def _is_fake_callee(callee_name: str, language: str = "") -> bool:
    short = callee_name.rsplit(".", 1)[-1]
    dotted = callee_name.rsplit(" ", 1)[-1]
    fake = set()
    if language in LANGUAGE_RULES:
        fake.update(LANGUAGE_RULES[language].get("fake_callees", set()))
    fake.update(LANGUAGE_RULES["python"].get("fake_callees", set()))
    return short in fake or dotted in fake


def _looks_like_test(file_path: str, language: str = "") -> bool:
    """只按路径判定测试代码,不看名字。"""
    p = file_path.replace("\\", "/")
    if "/tests/" in p or "/test/" in p:
        return True
    filename = p.rsplit("/", 1)[-1]
    rules = LANGUAGE_RULES.get(language or detect_language(file_path), LANGUAGE_RULES["python"])
    if any(filename.startswith(prefix) for prefix in rules.get("test_prefixes", ())):
        return True
    if any(filename.endswith(suffix) for suffix in rules.get("test_suffixes", ())):
        return True
    return False


def _is_real_source(text: str) -> bool:
    """判断 get_source_code 返回的是不是真的源码。"""
    if not text:
        return False
    if text in ("No source file.", "No valid line range.", "No source code found."):
        return False
    if text.startswith("Error reading source:"):
        return False
    return True


# =========================================================================
# 核心:一次遍历,两个图并行填充
# =========================================================================

def build_graphs(csv_path, project: str = "", source_root: str = ""):
    """同时构造 nx.Graph(graphml 用)和 fcg_bench.CallGraph(benchmark 用)。"""
    G = nx.Graph()
    cg = CallGraph()
    node_map = {}
    node_count = 1

    stats = {
        "total_rows":           0,
        "skipped_bad_row":      0,
        "skipped_pseudo_edge":  0,
        "skipped_zero_callee":  0,
        "skipped_zero_caller":  0,
        "skipped_empty_name":   0,
        "skipped_module_caller": 0,
        "skipped_class_caller": 0,
        "skipped_fake_callee":  0,
        "skipped_self_loop":    0,
        "edges_added":          0,
        "edges_merged":         0,
    }

    with open(csv_path, "r", encoding="utf-8") as f:
        raw_reader = csv.reader(f)
        first_row = next(raw_reader, None)
        has_header = _looks_like_header(first_row)
        if has_header:
            header = first_row or []
            if header != EXPECTED_HEADER:
                print(
                    f"Warning: CSV header differs from expected 15-column format; "
                    f"using alias-based field mapping.\n"
                    f"  Got:      {header}\n"
                    f"  Expected: {EXPECTED_HEADER}\n"
                    f"  Proceeding anyway.",
                    file=sys.stderr,
                )
            dict_rows = (dict(zip(header, row)) for row in raw_reader)
        else:
            if first_row is not None and len(first_row) != len(EXPECTED_HEADER):
                print(
                    f"Warning: CSV appears headerless and first row has {len(first_row)} columns; "
                    f"expected {len(EXPECTED_HEADER)} columns.",
                    file=sys.stderr,
                )
            pending_rows = [] if first_row is None else [first_row]
            dict_rows = (_list_row_to_dict(row) for row in itertools.chain(pending_rows, raw_reader))

        for row in dict_rows:
            stats["total_rows"] += 1

            if not row:
                stats["skipped_bad_row"] += 1
                continue

            caller_kind  = _row_value(row, "caller_kind")
            caller_name  = _row_value(row, "caller_name")
            caller_start = _row_value(row, "caller_start_line")
            caller_end   = _row_value(row, "caller_end_line")
            caller_file  = _row_value(row, "caller_file")
            call_kind    = _row_value(row, "call_kind")
            edge_source  = _row_value(row, "edge_source")
            callee_kind  = _row_value(row, "callee_kind")
            callee_name  = _row_value(row, "callee_name")
            callee_start = _row_value(row, "callee_start_line")
            callee_end   = _row_value(row, "callee_end_line")
            callee_file  = _row_value(row, "callee_file")
            caller_language = detect_language(caller_file)
            callee_language = detect_language(callee_file)

            # ---- 过滤层 ----
            if SKIP_PSEUDO_EDGES and edge_source in PSEUDO_EDGE_SOURCES:
                stats["skipped_pseudo_edge"] += 1
                continue
            if SKIP_ZERO_CALLEE and (_is_zero(callee_start) or _is_zero(callee_end)):
                stats["skipped_zero_callee"] += 1
                continue
            if SKIP_ZERO_CALLER and (_is_zero(caller_start) or _is_zero(caller_end)):
                stats["skipped_zero_caller"] += 1
                continue
            if not caller_name.strip() or not callee_name.strip():
                stats["skipped_empty_name"] += 1
                continue
            if SKIP_MODULE_CALLER and _is_module_caller(caller_kind):
                stats["skipped_module_caller"] += 1
                continue
            if SKIP_CLASS_CALLER and _is_class_caller(caller_kind):
                stats["skipped_class_caller"] += 1
                continue
            if SKIP_FAKE_CALLEE and _is_fake_callee(callee_name, callee_language):
                stats["skipped_fake_callee"] += 1
                continue

            # ---- 节点创建/去重 ----
            # 与原 build_graph.py 完全相同的节点 id / 属性填充逻辑
            for name, kind, s, e, path, language in [
                (caller_name, caller_kind, caller_start, caller_end, caller_file, caller_language),
                (callee_name, callee_kind, callee_start, callee_end, callee_file, callee_language),
            ]:
                if name in node_map:
                    continue

                node_id = f'"FUNCTIONID {node_count}"'
                node_map[name] = node_id
                node_count += 1

                full_path = _resolve_path(path, source_root)
                source_code = get_source_code(path, s, e, source_root=source_root, language=language)

                # 1) networkx Graph —— 节点格式和原脚本完全一致
                G.add_node(
                    node_id,
                    type=kind or "FUNCTION",
                    qualname=name,
                    file_path=full_path or path or "",
                    start_line=_safe_int(s),
                    end_line=_safe_int(e),
                    description=source_code,
                    source_id=SOURCE_ID,
                )

                # 2) fcg_bench CallGraph —— code 需要清洗占位字符串
                if kind.strip().upper() == "CLASS":
                    fcg_code = ""             # 类节点 → 哑节点
                elif _is_real_source(source_code):
                    fcg_code = source_code    # 真实源码
                else:
                    fcg_code = ""             # 占位字符串 → 空,让 ast_analyzer 跳过

                cg.add_node(FunctionNode(
                    qualname=name,
                    file_path=full_path or path or "",
                    code=fcg_code,
                    start_line=_safe_int(s),
                    end_line=_safe_int(e),
                    language=language,
                    project=project,
                    node_kind=kind or "",
                    is_test=_looks_like_test(path or "", language),
                ))

            # ---- 边创建/去重 ----
            u = node_map[caller_name]
            v = node_map[callee_name]
            if u == v:
                stats["skipped_self_loop"] += 1
                continue

            # 1) networkx Graph —— 和原脚本完全一致
            if G.has_edge(u, v):
                G[u][v]["weight"] += 1.0
                existing_kinds = G[u][v].get("call_kinds", "")
                if call_kind and call_kind not in existing_kinds.split(","):
                    if existing_kinds:
                        G[u][v]["call_kinds"] = existing_kinds + "," + call_kind
                    else:
                        G[u][v]["call_kinds"] = call_kind
                stats["edges_merged"] += 1
            else:
                G.add_edge(
                    u, v,
                    weight=1.0,
                    description="Function call",
                    call_kinds=call_kind or "",
                    source_id=SOURCE_ID,
                )
                stats["edges_added"] += 1

            # 2) fcg_bench CallGraph —— 有向,add_edge 内部去重
            cg.add_edge(caller_name, callee_name)

    return G, cg, stats


# =========================================================================
# 主流程
# =========================================================================

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build GraphML and FCG benchmark JSONL from a call-graph CSV."
    )
    parser.add_argument("--input-csv", default=INPUT_CSV, help="Call graph CSV path.")
    parser.add_argument("--project", default="", help="Project label stored on FunctionNode, e.g. openclaw or ollama.")
    parser.add_argument("--source-root", default="", help="Root used to resolve relative source paths in the CSV.")
    parser.add_argument("--output-jsonl", default=OUTPUT_JSONL, help="Benchmark JSONL output path.")
    parser.add_argument("--output-graphml", default=OUTPUT_GRAPHML, help="GraphML output path.")
    parser.add_argument("--max-roots", type=int, default=50, help="Maximum benchmark roots to sample.")
    parser.add_argument("--k-hop", type=int, default=2, help="K-hop subgraph size for benchmark generation.")
    parser.add_argument("--seed", type=int, default=0, help="Benchmark generation random seed.")
    return parser.parse_args()


def main():
    args = parse_args()
    try:
        G, cg, stats = build_graphs(args.input_csv, project=args.project, source_root=args.source_root)
    except FileNotFoundError:
        print(f"Error: input CSV {args.input_csv} not found", file=sys.stderr)
        sys.exit(1)

    # ---- 写 graphml(和原 build_graph.py 完全相同的方式) ----
    final_G = nx.Graph(G)
    nx.write_graphml(final_G, args.output_graphml, named_key_ids=True)

    # ---- 跑 fcg_bench 生成 benchmark ----
    print("\n正在生成 benchmark 题目...")
    items = generate_benchmark(cg, k_hop=args.k_hop, max_roots=args.max_roots, seed=args.seed)

    by_type: dict[str, list] = {}
    for it in items:
        by_type.setdefault(it.question_type, []).append(it)

    for qtype, lst in by_type.items():
        print(f"\n### {qtype}  ({len(lst)} 条)")
        for it in lst[:1]:
            print(f"  root    : {it.root_function}")
            print(f"  question: {it.question}")
            print(f"  gold    : {json.dumps(it.gold_answer, ensure_ascii=False)[:200]}")
            print(f"  natural : {it.natural_answer[:200]}")

    with open(args.output_jsonl, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it.to_dict(), ensure_ascii=False) + "\n")

    # ---- 报告 ----
    print("\n" + "=" * 60)
    print(" Build report")
    print("=" * 60)
    print(f"  Input CSV:              {args.input_csv}")
    print(f"  Project:                {args.project or '<unset>'}")
    print(f"  Source root:            {args.source_root or '<none>'}")
    print(f"  Total rows read:        {stats['total_rows']}")
    print(f"  Skipped (bad row):      {stats['skipped_bad_row']}")
    print(f"  Skipped (pseudo):       {stats['skipped_pseudo_edge']}")
    print(f"  Skipped (callee=0):     {stats['skipped_zero_callee']}")
    print(f"  Skipped (caller=0):     {stats['skipped_zero_caller']}")
    print(f"  Skipped (empty):        {stats['skipped_empty_name']}")
    print(f"  Skipped (module caller): {stats['skipped_module_caller']}")
    print(f"  Skipped (class caller): {stats['skipped_class_caller']}  ← new")
    print(f"  Skipped (fake callee):  {stats['skipped_fake_callee']}  ← new")
    print(f"  Skipped (self-loop):    {stats['skipped_self_loop']}")
    print(f"  Edges added:            {stats['edges_added']}")
    print(f"  Edges merged:           {stats['edges_merged']}")
    print("-" * 60)
    print(f"  GraphML nodes:          {final_G.number_of_nodes()}")
    print(f"  GraphML edges:          {final_G.number_of_edges()}")
    print(f"  fcg CallGraph nodes:    {len(cg.nodes)}")
    print(f"  fcg CallGraph edges:    {sum(len(e) for e in cg.edges.values())}")
    print(f"  Benchmark items:        {len(items)}")
    print("-" * 60)
    print(f"  GraphML output:         {args.output_graphml}")
    print(f"  Benchmark jsonl:        {args.output_jsonl}")
    print("=" * 60)


if __name__ == "__main__":
    main()
