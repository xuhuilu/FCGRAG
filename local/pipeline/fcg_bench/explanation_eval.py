"""解释评估模块。

在结构评估（evaluator.py）之上，增加对自然语言解释质量的评估。

三层评估:
  1. Claim Coverage  — 从 gold_answer 自动抽取原子声明，用 LLM 判定候选回答是否覆盖
  2. Reasoning Quality — LLM-as-Judge 对因果推理、结构一致性等维度打分
  3. Hallucination Check — 检查解释中是否引用了 gold 中不存在的实体/关系

用法:
    from fcg_bench.explanation_eval import (
        extract_claims,
        generate_explanation_gold,
        judge_explanation,
        evaluate_explanation_dataset,
    )
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field, asdict
from typing import Any, Protocol

# 复用已有的 LLM 抽象和 JSON 解析
from .gold_v2 import is_execution_flow_gold_v21, is_gold_v2, legacy_gold_view
from .graphrag_rag import LLMClient, parse_llm_json


# ============================================================
# 数据结构
# ============================================================

@dataclass
class Claim:
    """一条可验证的原子声明。"""
    text: str
    claim_type: str   # origin / causation / path / comparison / classification / scope


@dataclass
class ExplanationGold:
    """强模型生成的解释性金标准。"""
    explanation: str           # 自然语言参考解释
    claims: list[Claim]        # 原子声明列表
    source: str = "auto"       # "rule" | "llm" | "manual"
    structure_gold_hash: str = ""


@dataclass
class ExplanationScore:
    """单条解释评估结果。"""
    claim_coverage: float       # [0, 1] 被覆盖的 claim 比例
    claim_details: list[dict]   # 每条 claim 的判定结果
    causal_reasoning: int       # 0-2
    structural_consistency: int # 0-2
    no_hallucination: int       # 0-2
    clarity: int                # 0-2
    judge_evidence: dict        # LLM-as-Judge 的原始证据
    total_score: float = 0.0    # 加权综合分 [0, 1]

    def __post_init__(self):
        # 综合分: claim_coverage 占 40%, 四个维度各占 15%
        dim_avg = (
            self.causal_reasoning
            + self.structural_consistency
            + self.no_hallucination
            + self.clarity
        ) / 8.0  # 四个维度满分各 2，总满分 8，归一化到 [0, 1]
        self.total_score = round(0.4 * self.claim_coverage + 0.6 * dim_avg, 4)


# ============================================================
# 第 1 层: 规则化 Claim 抽取（不需要 LLM）
# ============================================================

def _short(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def extract_claims(qtype: str, gold: dict, question: str = "") -> list[Claim]:
    """从结构化 gold_answer 自动抽取原子声明列表。

    纯规则，不调用 LLM，保证可重复。
    """
    if is_gold_v2(gold) and qtype != "execution_flow":
        gold = legacy_gold_view(qtype, gold)
    claims: list[Claim] = []

    if qtype == "execution_flow":
        if is_execution_flow_gold_v21(gold):
            shared_prefix = gold.get("shared_prefix", []) or []
            branch_workflows = gold.get("branch_workflows", []) or []
            key_gates = gold.get("key_gates", []) or []
            if shared_prefix:
                claims.append(Claim(
                    text=f"共享 workflow 前缀是 {' → '.join(_short(x) for x in shared_prefix)}",
                    claim_type="path",
                ))
            for path in branch_workflows[:2]:
                claims.append(Claim(
                    text=f"一条代表性分支 workflow 是 {' → '.join(_short(x) for x in path)}",
                    claim_type="path",
                ))
            if key_gates:
                claims.append(Claim(
                    text=f"关键 gate 包括 {', '.join(_short(x) for x in key_gates)}",
                    claim_type="scope",
                ))
            claims.append(Claim(
                text="当前结果描述的是一条 representative workflow，而不是唯一完整执行流",
                claim_type="scope",
            ))
            return claims
        if is_gold_v2(gold):
            hard = gold.get("hard_constraints", {}) or {}
            soft = gold.get("soft_constraints", {}) or {}
            required = hard.get("required_entities", []) or []
            optional = soft.get("optional_entities", []) or []
            order_constraints = hard.get("order_constraints", []) or []
            if required:
                claims.append(Claim(
                    text=f"当前抽取的代表性 workflow 从 {_short(required[0])} 开始",
                    claim_type="path",
                ))
            if len(required) > 1:
                claims.append(Claim(
                    text=f"关键 workflow 步骤包含 {', '.join(_short(x) for x in required[1:])}",
                    claim_type="path",
                ))
            for pair in order_constraints:
                if isinstance(pair, list) and len(pair) == 2:
                    claims.append(Claim(
                        text=f"{_short(pair[0])} 在 workflow 中先于 {_short(pair[1])}",
                        claim_type="path",
                    ))
            if optional:
                claims.append(Claim(
                    text=f"可选 workflow 步骤包括 {', '.join(_short(x) for x in optional)}",
                    claim_type="scope",
                ))
            claims.append(Claim(
                text="当前结果描述的是一条 representative workflow，而不是唯一完整执行流",
                claim_type="scope",
            ))
            return claims
        path = gold.get("path_nodes") or (gold.get("gold_paths", [[]])[0] if gold.get("gold_paths") else [])
        if path:
            claims.append(Claim(
                text=f"当前抽取的代表性路径从 {_short(path[0])} 开始",
                claim_type="path",
            ))
            for i in range(len(path) - 1):
                claims.append(Claim(
                    text=f"{_short(path[i])} 之后调用 {_short(path[i+1])}",
                    claim_type="path",
                ))
            if len(path) >= 3:
                claims.append(Claim(
                    text=f"当前抽取路径共经过 {len(path)} 个关键函数",
                    claim_type="scope",
                ))

    elif qtype == "result_origin":
        src = gold.get("source_entity", "")
        tgt = gold.get("target_object", "")
        consumers = gold.get("consumer_entities", [])
        if src:
            claims.append(Claim(
                text=f"{src} 是 {tgt or '返回值'} 的最早来源函数",
                claim_type="origin",
            ))
        for c in consumers:
            claims.append(Claim(
                text=f"{c} 在 {src} 之后消费了 {tgt or '返回值'}",
                claim_type="causation",
            ))
        if not consumers and src:
            claims.append(Claim(
                text=f"当前结构化结果未标出 {tgt or '返回值'} 的后续显式参数消费者",
                claim_type="scope",
            ))

    elif qtype == "conditional_path":
        cond = gold.get("condition", "")
        branch_kind = gold.get("branch_kind", "")
        path_nodes = gold.get("path_nodes", [])
        if cond:
            claims.append(Claim(
                text=f"分支条件是 {cond}",
                claim_type="comparison",
            ))
        if branch_kind:
            claims.append(Claim(
                text=f"当前抽取的是 {branch_kind} 分支",
                claim_type="classification",
            ))
        if path_nodes:
            claims.append(Claim(
                text=f"该分支调用链包含 {', '.join(_short(x) for x in path_nodes[1:])}",
                claim_type="path",
            ))
        if gold.get("excluded_branches"):
            claims.append(Claim(
                text=f"未覆盖的其他分支标签包括 {', '.join(gold.get('excluded_branches', []))}",
                claim_type="comparison",
            ))

    elif qtype == "error_handling_path":
        trigger = gold.get("trigger_condition", "") or gold.get("error_trigger", "")
        path = gold.get("error_path", [])
        error_action = gold.get("error_action", "")
        error_message = gold.get("error_message", "")
        warns = gold.get("warns")
        except_returns = gold.get("except_returns")
        fallback_action = gold.get("fallback_action", "")
        evidence_span = gold.get("evidence_span", {}) or {}
        if trigger:
            claims.append(Claim(
                text=f"异常由 {trigger} 触发",
                claim_type="origin",
            ))
        if len(path) >= 3:
            handlers = path[2:]  # 跳过 root 和 trigger
            claims.append(Claim(
                text=f"异常处理链经过 {', '.join(_short(x) for x in handlers)}",
                claim_type="path",
            ))
        if error_action:
            claims.append(Claim(
                text=f"错误相关动作是 {error_action}",
                claim_type="classification",
            ))
        if error_message:
            claims.append(Claim(
                text=f"错误相关消息是 {error_message}",
                claim_type="classification",
            ))
        if warns is not None:
            claims.append(Claim(
                text=f"该错误相关路径记录了 warns={warns}",
                claim_type="classification",
            ))
        if except_returns is not None:
            claims.append(Claim(
                text=f"该错误相关路径记录了 except_returns={except_returns}",
                claim_type="classification",
            ))
        if fallback_action:
            claims.append(Claim(
                text=f"异常路径的回退动作是 {fallback_action}",
                claim_type="causation",
            ))
        if evidence_span:
            claims.append(Claim(
                text=(
                    "该解释只覆盖当前 evidence_span "
                    f"{evidence_span.get('start_line', 0)}-{evidence_span.get('end_line', 0)}，"
                    "不代表函数的全部错误处理逻辑"
                ),
                claim_type="scope",
            ))

    elif qtype == "core_support_classification":
        core = gold.get("core_entities", [])
        support = gold.get("support_entities", [])
        for c in core:
            claims.append(Claim(
                text=f"{_short(c)} 属于核心业务函数",
                claim_type="classification",
            ))
        for s in support:
            claims.append(Claim(
                text=f"{_short(s)} 属于辅助支撑函数",
                claim_type="classification",
            ))

    elif qtype == "local_subgraph_summary":
        stages = gold.get("stages", [])
        core = gold.get("core_entities", [])
        if stages:
            claims.append(Claim(
                text=f"整体流程分为 {len(stages)} 个阶段: {' → '.join(stages)}",
                claim_type="path",
            ))
            for i, s in enumerate(stages):
                claims.append(Claim(
                    text=f"存在 {s} 阶段",
                    claim_type="classification",
                ))
        for c in core:
            claims.append(Claim(
                text=f"{_short(c)} 是该子图中的核心函数",
                claim_type="classification",
            ))

    elif qtype == "dependency_trace":
        targets = gold.get("target_entities", [])
        sample_path = gold.get("sample_path", [])
        if targets:
            claims.append(Claim(
                text=f"可达目标包含 {', '.join(_short(x) for x in targets)}",
                claim_type="scope",
            ))
        if sample_path and len(sample_path) >= 2:
            claims.append(Claim(
                text=f"一条代表性路径为 {' → '.join(_short(x) for x in sample_path)}",
                claim_type="path",
            ))

    elif qtype == "impact_scope":
        direct = gold.get("static_direct_callers", []) or gold.get("direct_callers", [])
        impacted = gold.get("potentially_affected_callers", []) or gold.get("indirect_callers", [])
        if direct:
            claims.append(Claim(
                text=f"静态直接调用者有 {', '.join(_short(x) for x in direct)}",
                claim_type="scope",
            ))
        if impacted:
            claims.append(Claim(
                text=f"潜在影响范围包含 {', '.join(_short(x) for x in impacted)}",
                claim_type="scope",
            ))
        claims.append(Claim(
            text=f"当前答案描述的是静态调用图中的潜在影响范围",
            claim_type="scope",
        ))

    return claims


# ============================================================
# 第 2 层: 强模型生成 Explanation Gold（含 claims 增强）
# ============================================================

_EXPLANATION_GOLD_SYSTEM = """你是代码结构分析专家。你的任务是基于给定的结构化事实，生成一段严格 closed-world 的自然语言解释。

要求:
1. 只能使用结构化标准答案中已经出现的实体、路径、span、标签和字段
2. 不能引入结构化标准答案之外的新函数名、新变量名、新路径、新因果关系
3. 不允许用源码常识、框架常识或推测来补全结构化事实中没有写出的内容
4. conditional_path 只能描述当前 extracted branch，不能推广到所有 false/else/elif 分支
5. execution_flow 只能描述 representative_path，不能说唯一主流程、完整流程或主链
6. impact_scope 只能描述 static direct callers / potential impact points，不能说 runtime guaranteed impact
7. dependency_trace 只能描述 static FCG reachability
8. error_handling_path 只能基于 evidence_span 中对应的 raise / except / warn / fallback 字段来解释
9. 严格输出 JSON，不要有任何解释或 markdown 标记
"""

# 每类题的解释生成 prompt
_EXPLANATION_TEMPLATES: dict[str, str] = {
    "execution_flow": """# 题目
{question}

# 结构化标准答案
代表性抽取路径: {path_str}

# 任务
请只根据上面的结构化标准答案生成一段自然语言解释，说明:
1. 这条抽取路径的整体目的是什么
2. 每一步调用的作用是什么（不能只列函数名，要说它做了什么）
3. 前后步骤之间的因果/数据流关系
4. 不能把它说成“唯一主流程”或“完整执行流”
5. explanation 中出现的函数名必须全部来自路径字段本身

同时提取 atomic claims（可独立验证的原子声明），每条 claim 必须对应上面的结构化事实中某个具体关系。

输出 JSON:
{{
  "explanation": "一段 2-4 句的自然语言解释",
  "claims": [
    {{"text": "具体声明", "type": "path|origin|causation|comparison|classification|scope"}}
  ]
}}""",

    "result_origin": """# 题目
{question}

# 结构化标准答案
- 目标对象: {target_object}
- 来源函数: {source_entity}
- 消费者: {consumers_str}

# 任务
请只根据上面的三个结构化字段生成解释，并遵守下面约束:
1. explanation 中出现的对象名和函数名必须来自 目标对象 / 来源函数 / 消费者 这三个字段
2. 如果 消费者 为 "(无)"，只能说“当前结构化结果未标出后续显式参数消费者”，不能扩写成“直接返回”或“没有其他用途”
3. 不得引入新的变量名、中间对象名、运算符来源或额外数据流
4. 不能增加结构化事实之外的新关系

输出 JSON:
{{
  "explanation": "一段 2-4 句的自然语言解释",
  "claims": [
    {{"text": "具体声明", "type": "path|origin|causation|comparison|classification|scope"}}
  ]
}}""",

    "conditional_path": """# 题目
{question}

# 结构化标准答案
- 分支条件: {condition}
- 当前分支类型: {branch_kind}
- 当前分支调用链: {path_nodes_str}
- 分支范围: {branch_span}
- 未覆盖分支: {excluded_branches_str}

# 任务
请只根据上面的结构化标准答案生成一段自然语言解释，说明:
1. 当前抽取分支在什么条件下会进入
2. 该分支中的关键函数分别做了什么
3. explanation 必须明确这只是一个抽取分支，不代表所有 false/else/elif 情况都等价
4. 不要把未覆盖分支说成已经观察到的执行结果

输出 JSON:
{{
  "explanation": "一段 2-4 句的自然语言解释",
  "claims": [
    {{"text": "具体声明", "type": "path|origin|causation|comparison|classification|scope"}}
  ]
}}""",

    "error_handling_path": """# 题目
{question}

# 结构化标准答案
- 异常触发者: {error_trigger}
- 错误处理链: {error_path_str}
- 错误相关动作: {error_action}
- 回退动作: {fallback_action}
- 证据范围: {evidence_span}

# 任务
请只根据上面的结构化标准答案生成一段自然语言解释，说明:
1. {error_trigger} 在什么情况下会抛出异常
2. except 分支中执行了哪些错误相关动作
3. 只能基于 evidence_span 对应的 raise / except / warn / fallback 字段解释，不要引入额外“重试/恢复/包装”策略

输出 JSON:
{{
  "explanation": "一段 2-4 句的自然语言解释",
  "claims": [
    {{"text": "具体声明", "type": "path|origin|causation|comparison|classification|scope"}}
  ]
}}""",

    "core_support_classification": """# 题目
{question}

# 结构化标准答案
- 核心函数: {core_str}
- 辅助函数: {support_str}

# 任务
请生成一段自然语言解释，说明:
1. 为什么这些函数被归为核心（它们在主业务流程中的角色）
2. 为什么那些被归为辅助（它们提供了什么支撑）
3. 核心和辅助之间的协作关系

输出 JSON:
{{
  "explanation": "一段 2-4 句的自然语言解释",
  "claims": [
    {{"text": "具体声明", "type": "path|origin|causation|comparison|classification|scope"}}
  ]
}}""",

    "local_subgraph_summary": """# 题目
{question}

# 结构化标准答案
- 阶段: {stages_str}
- 核心函数: {core_str}

# 任务
请只根据给定的 阶段 和 核心函数 生成解释，并遵守下面约束:
1. explanation 中出现的阶段名必须完全来自 {stages_str}，顺序保持一致
2. explanation 中点名的核心函数必须完全来自 {core_str}
3. 不得新增新的阶段、核心函数或子图职责
4. 不得补充结构化答案中没有出现的函数或关系

输出 JSON:
{{
  "explanation": "一段 2-4 句的自然语言解释",
  "claims": [
    {{"text": "具体声明", "type": "path|origin|causation|comparison|classification|scope"}}
  ]
}}""",

    "dependency_trace": """# 题目
{question}

# 结构化标准答案
- 目标类别: {target_kind}
- 可达目标: {targets_str}
- 代表性路径: {sample_path_str}

# 任务
请只根据上面的结构化标准答案生成一段自然语言解释，说明:
1. 为什么在静态 FCG 中能到达这些目标
2. 代表性路径上每一步的作用
3. 这些依赖关系对理解代码结构有什么意义
4. 不要把静态可达关系说成运行时一定发生

输出 JSON:
{{
  "explanation": "一段 2-4 句的自然语言解释",
  "claims": [
    {{"text": "具体声明", "type": "path|origin|causation|comparison|classification|scope"}}
  ]
}}""",

    "impact_scope": """# 题目
{question}

# 结构化标准答案
- 静态直接调用者: {direct_str}
- 潜在影响范围: {indirect_str}

# 任务
请只根据上面的结构化标准答案生成一段自然语言解释，说明:
1. 直接调用者与 root 存在静态调用关系
2. 潜在影响范围表示当前静态图里可能受波及的调用者集合
3. explanation 必须使用保守口径，不要写成“运行时一定受影响”

输出 JSON:
{{
  "explanation": "一段 2-4 句的自然语言解释",
  "claims": [
    {{"text": "具体声明", "type": "path|origin|causation|comparison|classification|scope"}}
  ]
}}""",
}


def _build_explanation_prompt(
    qtype: str, gold: dict, question: str, source_code: str = "",
) -> str:
    """拼装 explanation gold 的 user prompt。"""
    if is_gold_v2(gold) and qtype != "execution_flow":
        gold = legacy_gold_view(qtype, gold)
    template = _EXPLANATION_TEMPLATES.get(qtype)
    if template is None:
        raise ValueError(f"unsupported question type: {qtype}")

    fmt_args: dict[str, str] = {
        "question": question,
    }

    if qtype == "execution_flow":
        if is_gold_v2(gold):
            hard = gold.get("hard_constraints", {}) or {}
            soft = gold.get("soft_constraints", {}) or {}
            path = (soft.get("acceptable_paths", []) or [hard.get("required_entities", [])])[-1]
            fmt_args["path_str"] = " → ".join(path) if path else "(空)"
        else:
            path = gold.get("path_nodes") or (gold.get("gold_paths", [[]])[0] if gold.get("gold_paths") else [])
            fmt_args["path_str"] = " → ".join(path) if path else "(空)"

    elif qtype == "result_origin":
        fmt_args["target_object"] = gold.get("target_object", "")
        fmt_args["source_entity"] = gold.get("source_entity", "")
        consumers = gold.get("consumer_entities", [])
        fmt_args["consumers_str"] = ", ".join(consumers) if consumers else "(无)"

    elif qtype == "conditional_path":
        fmt_args["condition"] = gold.get("condition", "")
        fmt_args["branch_kind"] = gold.get("branch_kind", "")
        fmt_args["path_nodes_str"] = " → ".join(gold.get("path_nodes", []))
        span = gold.get("branch_span", {}) or {}
        fmt_args["branch_span"] = f"{span.get('start_line', 0)}-{span.get('end_line', 0)}"
        fmt_args["excluded_branches_str"] = ", ".join(gold.get("excluded_branches", [])) or "(无)"

    elif qtype == "error_handling_path":
        fmt_args["error_trigger"] = gold.get("error_trigger", "")
        fmt_args["error_path_str"] = " → ".join(gold.get("error_path", []))
        fmt_args["error_action"] = gold.get("error_action", "") or "(未标注)"
        fmt_args["fallback_action"] = str(gold.get("fallback_action", "") or "(未标注)")
        span = gold.get("evidence_span", {}) or {}
        fmt_args["evidence_span"] = f"{span.get('start_line', 0)}-{span.get('end_line', 0)}"

    elif qtype == "core_support_classification":
        fmt_args["core_str"] = ", ".join(gold.get("core_entities", []))
        fmt_args["support_str"] = ", ".join(gold.get("support_entities", []))

    elif qtype == "local_subgraph_summary":
        fmt_args["stages_str"] = " → ".join(gold.get("stages", []))
        fmt_args["core_str"] = ", ".join(gold.get("core_entities", []))

    elif qtype == "dependency_trace":
        fmt_args["target_kind"] = gold.get("target_kind", "")
        fmt_args["targets_str"] = ", ".join(gold.get("target_entities", []))
        fmt_args["sample_path_str"] = " → ".join(gold.get("sample_path", []))

    elif qtype == "impact_scope":
        fmt_args["direct_str"] = ", ".join(gold.get("static_direct_callers", []) or gold.get("direct_callers", []))
        fmt_args["indirect_str"] = ", ".join(gold.get("potentially_affected_callers", []) or gold.get("indirect_callers", []))

    return template.format(**fmt_args)


def _join_entities(values: list[str]) -> str:
    return "、".join(values) if values else "(无)"


def build_closed_world_explanation(item: dict) -> str:
    """基于结构化事实生成保守的 closed-world 解释。

    该模板只使用结构化字段中的实体、路径、span 和标签，不尝试补全缺失信息。
    """
    qtype = item["question_type"]
    gold = item["gold_answer"]
    if is_gold_v2(gold) and qtype != "execution_flow":
        gold = legacy_gold_view(qtype, gold)

    if qtype == "conditional_path":
        cond = gold.get("condition", "")
        branch_kind = gold.get("branch_kind", "")
        nodes = gold.get("path_nodes", [])
        span = gold.get("branch_span", {}) or {}
        path_text = " → ".join(nodes) if nodes else "(空)"
        span_text = f"{span.get('start_line', 0)}-{span.get('end_line', 0)}"
        return (
            f"当前结构化结果抽取的是条件 {cond} 对应的 {branch_kind} 分支，分支范围是 {span_text}。"
            f"在这条 extracted branch 中，记录到的调用路径是 {path_text}。"
            "这只是当前被抽取的一条分支路径，不代表所有 false、else 或 elif 情况都与此等价。"
        )

    if qtype == "execution_flow":
        if is_execution_flow_gold_v21(gold):
            shared_prefix = gold.get("shared_prefix", []) or []
            branch_workflows = gold.get("branch_workflows", []) or []
            key_gates = gold.get("key_gates", []) or []
            parts = []
            if shared_prefix:
                parts.append(
                    f"当前结构化结果给出了一条 branch-aware representative workflow，共享前缀是 {' → '.join(shared_prefix)}。"
                )
            if branch_workflows:
                parts.append(
                    "代表性分支 workflow 包括 "
                    + "；".join(" → ".join(path) for path in branch_workflows[:2])
                    + "。"
                )
            if key_gates:
                parts.append(f"关键 gate 包括 {'、'.join(key_gates)}。")
            parts.append("这是一条代表性 workflow，不表示唯一主流程或完整执行流。")
            return "".join(parts)
        if is_gold_v2(gold):
            hard = gold.get("hard_constraints", {}) or {}
            soft = gold.get("soft_constraints", {}) or {}
            required = hard.get("required_entities", []) or []
            optional = soft.get("optional_entities", []) or []
            parts = []
            if required:
                parts.append(
                    f"当前结构化结果给出了一条 representative workflow，起点是 {required[0]}。"
                )
            if len(required) > 1:
                parts.append(
                    f"该 workflow 的关键步骤包括 {' → '.join(required[1:])}。"
                )
            if optional:
                parts.append(
                    f"可选步骤还包括 {'、'.join(optional)}。"
                )
            parts.append("这是一条代表性 workflow，不表示唯一主流程或完整执行流。")
            return "".join(parts)
        nodes = gold.get("path_nodes") or []
        path_text = " → ".join(nodes) if nodes else "(空)"
        cond = gold.get("path_condition", "")
        coverage = gold.get("coverage_note", "")
        parts = [f"当前结构化结果给出了一条 representative_path: {path_text}。"]
        if cond:
            parts.append(f"这条路径对应的条件说明是 {cond}。")
        if coverage:
            parts.append(coverage)
        else:
            parts.append("这是一条代表性抽取路径，不表示唯一主流程或完整执行流。")
        return "".join(parts)

    if qtype == "impact_scope":
        direct = gold.get("static_direct_callers", []) or gold.get("direct_callers", [])
        impacted = gold.get("potentially_affected_callers", []) or gold.get("indirect_callers", [])
        return (
            f"当前结构化结果中的静态直接调用者包括 {_join_entities(direct)}。"
            f"潜在影响范围包括 {_join_entities(impacted)}。"
            "这些关系描述的是静态调用图中的 direct callers 和 potential impact points，不表示运行时一定会受到影响。"
        )

    if qtype == "dependency_trace":
        targets = gold.get("target_entities", [])
        sample_path = gold.get("sample_path", [])
        target_kind = gold.get("target_kind", "")
        path_text = " → ".join(sample_path) if sample_path else "(空)"
        return (
            f"当前结构化结果表明，在静态 FCG 中可达的 {target_kind} 包括 {_join_entities(targets)}。"
            f"其中一条代表性静态可达路径是 {path_text}。"
            "这里描述的是静态可达关系，不表示这些调用在运行时一定发生。"
        )

    if qtype == "error_handling_path":
        trigger = gold.get("error_trigger", "")
        path = gold.get("error_path", [])
        error_action = gold.get("error_action", "") or "(未标注)"
        fallback_action = gold.get("fallback_action", "") or "(未标注)"
        span = gold.get("evidence_span", {}) or {}
        span_text = f"{span.get('start_line', 0)}-{span.get('end_line', 0)}"
        path_text = " → ".join(path) if path else "(空)"
        text = (
            f"当前结构化结果显示，错误相关路径由 {trigger} 触发，处理链为 {path_text}。"
            f"在 evidence span {span_text} 对应的结构化事实中，错误相关动作是 {error_action}。"
        )
        if fallback_action != "(未标注)":
            text += f"同一条错误相关路径里还记录了回退动作 {fallback_action}。"
        return text

    if qtype == "result_origin":
        target = gold.get("target_object", "")
        src = gold.get("source_entity", "")
        consumers = gold.get("consumer_entities", [])
        if consumers:
            return (
                f"当前结构化结果中，{src} 是 {target} 的来源实体。"
                f"后续显式消费者包括 {_join_entities(consumers)}。"
                "这段解释只复述结构化数据中的来源和消费者关系，不补充额外数据流。"
            )
        return (
            f"当前结构化结果中，{src} 是 {target} 的来源实体。"
            "结构化字段里没有记录到后续显式参数消费者，因此这里只能确认已记录的来源关系。"
        )

    return json.dumps(gold, ensure_ascii=False)


def generate_explanation_gold(
    item: dict,
    llm: LLMClient,
    source_code: str = "",
    temperature: float = 0.0,
) -> ExplanationGold:
    """用强模型生成 explanation gold: 参考解释 + LLM 增强的 claims。

    流程:
      1. 先用规则抽取基础 claims（确定性，可重复）
      2. 用强模型基于结构化答案 + 源码生成自然语言解释和额外 claims
      3. 合并两组 claims，去重
    """
    qtype = item["question_type"]
    gold = item["gold_answer"]
    question = item["question"]

    # 规则 claims（保底）
    rule_claims = extract_claims(qtype, gold, question)

    # 强模型生成
    user_msg = _build_explanation_prompt(qtype, gold, question, source_code)
    raw = llm.complete(_EXPLANATION_GOLD_SYSTEM, user_msg, temperature=temperature)
    parsed = parse_llm_json(raw)

    explanation = parsed.get("explanation", "") or build_closed_world_explanation(item)
    llm_claims_raw = parsed.get("claims", [])

    # 解析 LLM 生成的 claims
    llm_claims: list[Claim] = []
    for c in llm_claims_raw:
        if isinstance(c, dict) and c.get("text"):
            llm_claims.append(Claim(
                text=c["text"],
                claim_type=c.get("type", "causation"),
            ))

    claim_inputs = rule_claims

    # 合并: 规则 claims 在前（确定性高），LLM claims 在后（补充因果解释）
    # 按 text 去重
    seen_texts: set[str] = set()
    merged: list[Claim] = []
    for c in claim_inputs:
        key = c.text.strip().lower()
        if key not in seen_texts:
            seen_texts.add(key)
            merged.append(c)

    return ExplanationGold(
        explanation=explanation,
        claims=merged,
        source="rule+llm",
        structure_gold_hash=item.get("structure_gold_hash", ""),
    )


# ============================================================
# 第 3 层: LLM-as-Judge 评估
# ============================================================

_CLAIM_JUDGE_SYSTEM = """你是一个严格的事实核查员。你需要判定候选回答是否覆盖了给定的声明。

对于每条声明，你需要判定:
- "covered": 候选回答明确包含了该声明的核心信息（可以是不同措辞）
- "partial": 候选回答提到了相关内容但不完整或不够明确
- "missing": 候选回答完全没有涉及该声明

严格输出 JSON，不要有任何解释。
"""

_QUALITY_JUDGE_SYSTEM = """你是代码分析解释的质量评估专家。你需要从多个维度评估一段关于代码结构的解释。

评分标准严格按 0-2 分:
- 0: 完全不满足
- 1: 部分满足，有明显不足
- 2: 完全满足

严格输出 JSON，不要有任何解释或 markdown 标记。
"""


def _judge_claims(
    claims: list[Claim],
    response: str,
    llm: LLMClient,
    temperature: float = 0.0,
) -> tuple[float, list[dict]]:
    """用 LLM 判定候选回答对 claims 的覆盖率。

    返回 (coverage_ratio, per_claim_details)。
    """
    if not claims:
        return 1.0, []

    claims_text = "\n".join(
        f"  {i+1}. [{c.claim_type}] {c.text}"
        for i, c in enumerate(claims)
    )

    user_msg = f"""# 候选回答
{response}

# 需要核查的声明列表
{claims_text}

# 任务
对每条声明，判定候选回答是否覆盖了它。

输出 JSON:
{{
  "judgments": [
    {{"claim_id": 1, "verdict": "covered|partial|missing", "evidence": "简短理由"}},
    ...
  ]
}}
"""
    raw = llm.complete(_CLAIM_JUDGE_SYSTEM, user_msg, temperature=temperature)
    parsed = parse_llm_json(raw)

    judgments = parsed.get("judgments", [])

    # 计算覆盖率: covered=1, partial=0.5, missing=0
    details: list[dict] = []
    total = 0.0
    for i, claim in enumerate(claims):
        j = judgments[i] if i < len(judgments) else {}
        verdict = j.get("verdict", "missing")
        evidence = j.get("evidence", "")

        if verdict == "covered":
            score = 1.0
        elif verdict == "partial":
            score = 0.5
        else:
            score = 0.0
            verdict = "missing"

        total += score
        details.append({
            "claim": claim.text,
            "claim_type": claim.claim_type,
            "verdict": verdict,
            "score": score,
            "evidence": evidence,
        })

    coverage = total / len(claims) if claims else 1.0
    return coverage, details


def _judge_quality(
    item: dict,
    response: str,
    gold: dict,
    source_code: str,
    llm: LLMClient,
    temperature: float = 0.0,
) -> dict:
    """用 LLM 评估解释的推理质量。

    返回 {causal_reasoning, structural_consistency, no_hallucination, clarity, evidence}。
    """
    gold_summary = json.dumps(gold, ensure_ascii=False, indent=2)

    user_msg = f"""# 题目
{item.get("question", "")}

# 结构化标准答案
{gold_summary}

# 函数源码
```python
{source_code[:2000] if source_code else "(无源码)"}
```

# 候选回答
{response}

# 评估维度（每项 0-2 分）

1. **causal_reasoning（因果推理）**:
   - 0: 只列举了函数名，没有解释因果关系或原因
   - 1: 给了一些原因但不够具体，或部分因果链缺失
   - 2: 明确说清了因果链（如"因为 X 的返回值被 Y 作为参数传入…"）

2. **structural_consistency（结构一致性）**:
   - 0: 解释与标准答案矛盾（路径说反、分类错误、遗漏关键实体）
   - 1: 基本一致但有小偏差（遗漏了非关键实体，或描述不够精确）
   - 2: 完全一致，所有关键结构事实都被正确反映

3. **no_hallucination（无幻觉）**:
   - 0: 引用了标准答案和源码中都不存在的实体或关系
   - 1: 有过度推断但核心事实正确（如推测了函数的具体实现细节）
   - 2: 所有陈述都有标准答案或源码的支撑

4. **clarity（清晰度）**:
   - 0: 表述模糊、过于笼统，对开发者没有帮助
   - 1: 基本能理解，但有些地方不够具体
   - 2: 清晰、具体、可操作，能帮助不了解代码的人快速理解

输出 JSON:
{{
  "causal_reasoning": {{"score": 0, "evidence": "简短理由"}},
  "structural_consistency": {{"score": 0, "evidence": "简短理由"}},
  "no_hallucination": {{"score": 0, "evidence": "简短理由"}},
  "clarity": {{"score": 0, "evidence": "简短理由"}}
}}
"""
    raw = llm.complete(_QUALITY_JUDGE_SYSTEM, user_msg, temperature=temperature)
    parsed = parse_llm_json(raw)

    def _get_dim(name: str) -> tuple[int, str]:
        dim = parsed.get(name, {})
        if isinstance(dim, dict):
            s = dim.get("score", 0)
            return max(0, min(2, int(s))), dim.get("evidence", "")
        return 0, ""

    cr_score, cr_ev = _get_dim("causal_reasoning")
    sc_score, sc_ev = _get_dim("structural_consistency")
    nh_score, nh_ev = _get_dim("no_hallucination")
    cl_score, cl_ev = _get_dim("clarity")

    return {
        "causal_reasoning": cr_score,
        "structural_consistency": sc_score,
        "no_hallucination": nh_score,
        "clarity": cl_score,
        "evidence": {
            "causal_reasoning": cr_ev,
            "structural_consistency": sc_ev,
            "no_hallucination": nh_ev,
            "clarity": cl_ev,
        },
    }


def judge_explanation(
    item: dict,
    response: str,
    llm: LLMClient,
    claims: list[Claim] | None = None,
    source_code: str = "",
    temperature: float = 0.0,
) -> ExplanationScore:
    """对单条候选回答做完整的解释评估。

    Args:
        item: benchmark item（含 question_type, gold_answer, question）
        response: 模型生成的自然语言解释
        llm: LLM 客户端（用于 judge）
        claims: 预生成的 claims；如果为 None 则自动从 gold 抽取
        source_code: root 函数源码（可选，用于辅助判断）
        temperature: LLM 温度
    """
    gold = item["gold_answer"]
    qtype = item["question_type"]

    # 1. Claims coverage
    if claims is None:
        claims = extract_claims(qtype, gold, item.get("question", ""))

    coverage, claim_details = _judge_claims(claims, response, llm, temperature)

    # 2. Quality dimensions
    quality = _judge_quality(item, response, gold, source_code, llm, temperature)

    return ExplanationScore(
        claim_coverage=round(coverage, 4),
        claim_details=claim_details,
        causal_reasoning=quality["causal_reasoning"],
        structural_consistency=quality["structural_consistency"],
        no_hallucination=quality["no_hallucination"],
        clarity=quality["clarity"],
        judge_evidence=quality["evidence"],
    )


# ============================================================
# 数据集级别评估
# ============================================================

def evaluate_explanation_dataset(
    items: list[dict],
    responses: list[str],
    llm: LLMClient,
    claims_map: dict[int, list[Claim]] | None = None,
    source_map: dict[str, str] | None = None,
    temperature: float = 0.0,
) -> tuple[dict, list[dict]]:
    """对整个数据集做解释评估。

    Args:
        items: benchmark items 列表
        responses: 每条 item 对应的自然语言回答
        llm: judge 用的 LLM
        claims_map: 可选的预计算 claims（key 为 item 索引）
        source_map: qualname -> source_code 映射
        temperature: judge LLM 温度

    Returns:
        (summary_dict, per_item_details)
    """
    assert len(items) == len(responses), "items 和 responses 长度不一致"

    per_type: dict[str, list[ExplanationScore]] = {}
    details: list[dict] = []

    for i, (item, resp) in enumerate(zip(items, responses)):
        claims = claims_map.get(i) if claims_map else None
        source_code = ""
        if source_map:
            root_qn = item.get("root_function", "")
            source_code = source_map.get(root_qn, "")

        score = judge_explanation(
            item, resp, llm,
            claims=claims,
            source_code=source_code,
            temperature=temperature,
        )

        qtype = item["question_type"]
        per_type.setdefault(qtype, []).append(score)

        details.append({
            "index": i,
            "root_function": item.get("root_function", ""),
            "question_type": qtype,
            "claim_coverage": score.claim_coverage,
            "causal_reasoning": score.causal_reasoning,
            "structural_consistency": score.structural_consistency,
            "no_hallucination": score.no_hallucination,
            "clarity": score.clarity,
            "total_score": score.total_score,
            "claim_details": score.claim_details,
            "judge_evidence": score.judge_evidence,
        })

    # 汇总
    summary: dict[str, Any] = {"per_type": {}}
    for qt, scores in per_type.items():
        n = len(scores)
        summary["per_type"][qt] = {
            "n_items": n,
            "claim_coverage": round(sum(s.claim_coverage for s in scores) / n, 4),
            "causal_reasoning": round(sum(s.causal_reasoning for s in scores) / n, 4),
            "structural_consistency": round(
                sum(s.structural_consistency for s in scores) / n, 4
            ),
            "no_hallucination": round(
                sum(s.no_hallucination for s in scores) / n, 4
            ),
            "clarity": round(sum(s.clarity for s in scores) / n, 4),
            "total_score": round(sum(s.total_score for s in scores) / n, 4),
        }

    all_scores = [s for ss in per_type.values() for s in ss]
    n_all = len(all_scores)
    if n_all:
        summary["overall"] = {
            "n_items": n_all,
            "claim_coverage": round(
                sum(s.claim_coverage for s in all_scores) / n_all, 4
            ),
            "causal_reasoning": round(
                sum(s.causal_reasoning for s in all_scores) / n_all, 4
            ),
            "structural_consistency": round(
                sum(s.structural_consistency for s in all_scores) / n_all, 4
            ),
            "no_hallucination": round(
                sum(s.no_hallucination for s in all_scores) / n_all, 4
            ),
            "clarity": round(
                sum(s.clarity for s in all_scores) / n_all, 4
            ),
            "total_score": round(
                sum(s.total_score for s in all_scores) / n_all, 4
            ),
        }

    return summary, details
