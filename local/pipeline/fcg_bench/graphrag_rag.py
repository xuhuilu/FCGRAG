"""
GraphRAG 主流程:对一条 benchmark item,产出结构化预测。

流程:
  阶段 A: 定位 root (基本上是查索引,因为题目里已经给了)
  阶段 B: 用 tools.build_local_context 抽取局部上下文
  阶段 C: 按题型选 prompt → 调 LLM → 解析 JSON → 做字段后处理

可选:self-consistency,即对同一道题跑多次,做字段投票聚合。

LLM 调用被抽象成 `LLMClient` 接口,可以接 Claude / OpenAI / 本地 Qwen。
脚本里同时提供一个 EchoLLM 用来自测 pipeline。
"""
from __future__ import annotations
import json
import re
from collections import Counter
from dataclasses import dataclass
from typing import Any, Protocol

from .graphrag_tools import GraphRAGTools, LocalContext
from .graphrag_prompts import build_prompt, build_prompt_for_method


# ============================================================
# LLM 抽象
# ============================================================

class LLMClient(Protocol):
    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        ...


class EchoLLM:
    """
    一个假的 LLM,不做任何推理,直接基于 prompt 里的数据构造一个合法的 JSON。
    用于 pipeline 自测:验证 prompt 能组装、JSON 能解析、字段能送到 evaluator。
    """

    def complete(self, system: str, user: str, temperature: float = 0.0) -> str:
        # 从 user prompt 里抓 Root 函数名
        m = re.search(r"# Root 函数\n(\S+)", user)
        root = m.group(1) if m else "unknown"

        # 按 prompt 内容关键词选输出格式
        if "gold_paths" in user:
            return json.dumps({
                "path_type": "representative_path",
                "path_condition": "",
                "path_nodes": [root, "step1", "step2"],
                "coverage_note": "This is one extracted path, not the unique full execution flow.",
            })
        if "target_object" in user:
            return json.dumps({
                "target_object": "返回值", "source_entity": "unknown",
                "consumer_entities": []
            })
        if "branch_kind" in user and "path_nodes" in user:
            return json.dumps({
                "condition": "unknown",
                "branch_kind": "if_true",
                "branch_span": {"start_line": 0, "end_line": 0},
                "path_nodes": [root, "step1"],
                "excluded_branches": [],
            })
        if "error_trigger" in user:
            return json.dumps({
                "error_trigger": "unknown",
                "error_path": [root, "unknown"],
                "error_action": "",
                "fallback_action": None,
                "evidence_span": {"start_line": 0, "end_line": 0},
            })
        if "core_entities" in user and "support_entities" in user:
            return json.dumps({"core_entities": [], "support_entities": []})
        if "stages" in user:
            return json.dumps({"stages": ["核心执行"], "core_entities": [root]})
        if "potentially_affected_callers" in user:
            return json.dumps({
                "static_direct_callers": [],
                "potentially_affected_callers": [],
            })
        return "{}"


# ============================================================
# JSON 解析(容错)
# ============================================================

def parse_llm_json(text: str) -> dict[str, Any]:
    """
    从 LLM 输出里抽 JSON。容错处理:
    - 去掉可能的 ```json 包裹
    - 去掉前后的说明文字
    - 失败时返回空 dict
    """
    text = text.strip()
    # 去掉 markdown 代码块
    text = re.sub(r"^```(?:json)?\s*", "", text)
    text = re.sub(r"\s*```$", "", text)
    text = text.strip()

    # 直接尝试
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 抽最外层 {...}
    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        try:
            return json.loads(text[start:end + 1])
        except json.JSONDecodeError:
            return {}
    return {}


# ============================================================
# 主入口:单次回答
# ============================================================

@dataclass
class AnswerResult:
    prediction: dict[str, Any]
    raw_llm_output: str
    context_used: bool    # 是否成功组装了 context
    error: str | None = None


def answer_one(
    item: dict[str, Any],
    tools: GraphRAGTools,
    llm: LLMClient,
    temperature: float = 0.0,
    method: str = "structured",
) -> AnswerResult:
    """对一条 benchmark item 生成一次预测。

    method:
      - "structured": 完整的 GraphRAG (AST + 调用图 + tree + role_hint)
      - "code_only":  只给源码,不给 AST 派生和图结构
      - "direct":     完全不给上下文,只给问题
    """
    qtype = item["question_type"]
    question = item["question"]
    root_ref = item["root_function"]

    ctx = tools.build_local_context(root_ref)
    if ctx is None:
        return AnswerResult(
            prediction={}, raw_llm_output="",
            context_used=False,
            error=f"cannot resolve root: {root_ref}",
        )

    try:
        system_msg, user_msg = build_prompt_for_method(
            method, qtype, ctx, question, tools
        )
    except ValueError as e:
        return AnswerResult(
            prediction={}, raw_llm_output="", context_used=True, error=str(e)
        )

    try:
        raw = llm.complete(system_msg, user_msg, temperature=temperature)
    except Exception as e:
        return AnswerResult(
            prediction={}, raw_llm_output="", context_used=True, error=str(e)
        )
    pred = parse_llm_json(raw)

    # 后处理:保证字段齐全(防止 evaluator 报 KeyError)
    pred = _ensure_fields(qtype, pred, ctx)

    return AnswerResult(
        prediction=pred, raw_llm_output=raw, context_used=True,
    )


def _ensure_fields(qtype: str, pred: dict, ctx: LocalContext) -> dict:
    """给缺失的字段填默认值,让 evaluator 不会崩。"""
    pred.setdefault("explanation", "")
    if qtype == "execution_flow":
        pred.setdefault("path_type", "representative_path")
        pred.setdefault("path_condition", "")
        pred.setdefault("coverage_note", "This is one extracted path, not the unique full execution flow.")
        if "path_nodes" not in pred or not pred["path_nodes"]:
            pred["path_nodes"] = [ctx.root_short]
    elif qtype == "result_origin":
        pred.setdefault("target_object", "")
        pred.setdefault("source_entity", "")
        pred.setdefault("consumer_entities", [])
    elif qtype == "conditional_path":
        pred.setdefault("condition", "")
        pred.setdefault("branch_kind", "")
        pred.setdefault("branch_span", {"start_line": 0, "end_line": 0})
        pred.setdefault("path_nodes", [])
        pred.setdefault("excluded_branches", [])
    elif qtype == "error_handling_path":
        pred.setdefault("error_trigger", "")
        pred.setdefault("error_path", [])
        pred.setdefault("error_action", "")
        pred.setdefault("fallback_action", None)
        pred.setdefault("evidence_span", {"start_line": 0, "end_line": 0})
    elif qtype == "core_support_classification":
        pred.setdefault("core_entities", [])
        pred.setdefault("support_entities", [])
    elif qtype == "local_subgraph_summary":
        pred.setdefault("stages", [])
        pred.setdefault("core_entities", [])
    elif qtype == "dependency_trace":
        pred.setdefault("target_kind", "")
        pred.setdefault("target_entities", [])
        pred.setdefault("sample_path", [])
    elif qtype == "impact_scope":
        pred.setdefault("static_direct_callers", [])
        pred.setdefault("potentially_affected_callers", [])
    return pred


# ============================================================
# Self-consistency:多次采样后聚合
# ============================================================

def answer_with_voting(
    item: dict[str, Any],
    tools: GraphRAGTools,
    llm: LLMClient,
    n_samples: int = 3,
    temperature: float = 0.7,
    method: str = "structured",
) -> AnswerResult:
    """
    跑 n 次后聚合。不同字段用不同聚合策略:
      - 集合类(core_entities / support_entities / consumer_entities / stages):
        取出现次数 >= n/2 的元素
      - 标量类(source_entity / error_trigger / condition):
        取出现次数最多的
      - 顺序类(gold_paths / true_path / false_path / error_path):
        按元素在多次结果里的平均位置排序(只保留在过半样本里出现的)
    """
    if n_samples == 1:
        return answer_one(item, tools, llm, temperature=0.0, method=method)

    results: list[AnswerResult] = [
        answer_one(item, tools, llm, temperature=temperature, method=method)
        for _ in range(n_samples)
    ]

    qtype = item["question_type"]
    preds = [r.prediction for r in results if r.prediction]
    if not preds:
        return results[0]

    merged = _merge_predictions(qtype, preds, n_samples)

    # 还是用一个完整的 context 做字段补全
    ctx = tools.build_local_context(item["root_function"])
    if ctx is not None:
        merged = _ensure_fields(qtype, merged, ctx)

    return AnswerResult(
        prediction=merged,
        raw_llm_output=" || ".join(r.raw_llm_output for r in results),
        context_used=True,
    )


def _vote_set(values_list: list[list[str]], threshold: int) -> list[str]:
    """集合类字段投票:出现次数 >= threshold 的保留。"""
    cnt: Counter = Counter()
    for vs in values_list:
        for v in set(vs or []):
            cnt[v] += 1
    return sorted([v for v, c in cnt.items() if c >= threshold])


def _vote_scalar(values: list[str]) -> str:
    """标量类字段投票:取众数。"""
    cnt = Counter(v for v in values if v)
    if not cnt:
        return ""
    return cnt.most_common(1)[0][0]


def _vote_ordered(paths_list: list[list[str]], threshold: int) -> list[str]:
    """有序字段投票:先按出现次数过滤,再按平均位置排序。"""
    positions: dict[str, list[int]] = {}
    for p in paths_list:
        for i, v in enumerate(p or []):
            positions.setdefault(v, []).append(i)
    # 过滤出现次数不够的
    filtered = {v: pos for v, pos in positions.items() if len(pos) >= threshold}
    if not filtered:
        return []
    return sorted(filtered.keys(), key=lambda v: sum(filtered[v]) / len(filtered[v]))


def _merge_predictions(
    qtype: str, preds: list[dict], n_samples: int,
) -> dict:
    threshold = max(1, (n_samples + 1) // 2)  # 过半

    if qtype == "execution_flow":
        all_paths = [p.get("gold_paths", [[]])[0] for p in preds]
        merged = _vote_ordered(all_paths, threshold)
        return {"gold_paths": [merged]}

    elif qtype == "result_origin":
        return {
            "target_object": _vote_scalar([p.get("target_object", "") for p in preds]),
            "source_entity": _vote_scalar([p.get("source_entity", "") for p in preds]),
            "consumer_entities": _vote_set(
                [p.get("consumer_entities", []) for p in preds], threshold
            ),
        }

    elif qtype == "conditional_path":
        return {
            "condition": _vote_scalar([p.get("condition", "") for p in preds]),
            "true_path": _vote_ordered(
                [p.get("true_path", []) for p in preds], threshold
            ),
            "false_path": _vote_ordered(
                [p.get("false_path", []) for p in preds], threshold
            ),
        }

    elif qtype == "error_handling_path":
        return {
            "error_trigger": _vote_scalar([p.get("error_trigger", "") for p in preds]),
            "error_path": _vote_ordered(
                [p.get("error_path", []) for p in preds], threshold
            ),
            "shared_functions": _vote_set(
                [p.get("shared_functions", []) for p in preds], threshold
            ),
            "exception_strategy": _vote_scalar(
                [p.get("exception_strategy", "") for p in preds]
            ),
            "exception_return": _vote_scalar(
                [p.get("exception_return", "") for p in preds]
            ),
        }

    elif qtype == "core_support_classification":
        return {
            "core_entities": _vote_set(
                [p.get("core_entities", []) for p in preds], threshold
            ),
            "support_entities": _vote_set(
                [p.get("support_entities", []) for p in preds], threshold
            ),
        }

    elif qtype == "local_subgraph_summary":
        return {
            "stages": _vote_ordered([p.get("stages", []) for p in preds], threshold),
            "core_entities": _vote_set(
                [p.get("core_entities", []) for p in preds], threshold
            ),
        }

    elif qtype == "dependency_trace":
        return {
            "target_kind": _vote_scalar([p.get("target_kind", "") for p in preds]),
            "target_entities": _vote_set(
                [p.get("target_entities", []) for p in preds], threshold
            ),
            "sample_path": _vote_ordered(
                [p.get("sample_path", []) for p in preds], threshold
            ),
        }

    elif qtype == "impact_scope":
        return {
            "direct_callers": _vote_set(
                [p.get("direct_callers", []) for p in preds], threshold
            ),
            "indirect_callers": _vote_set(
                [p.get("indirect_callers", []) for p in preds], threshold
            ),
            "bridge_callers": _vote_set(
                [p.get("bridge_callers", []) for p in preds], threshold
            ),
            "unaffected_callees": _vote_set(
                [p.get("unaffected_callees", []) for p in preds], threshold
            ),
            "critical_bridge": _vote_scalar(
                [p.get("critical_bridge", "") for p in preds]
            ),
        }

    return preds[0]
