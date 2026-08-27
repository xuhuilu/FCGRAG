"""把 Fact 渲染成 BenchmarkItem(自然语言问题 + 结构化标准答案)。

自然语言答案部分只生成一个模板字符串 —— 真正的润色交给 Opus。
"""
from __future__ import annotations
import hashlib
import json
import random
from .schema import Fact, BenchmarkItem


_TEMPLATES = {
    "execution_flow": [
        "在当前抽取路径里,{root} 这条代表性执行路径依次经过哪些关键函数?",
        "从 {root} 开始,当前抽取到的一条代表性路径经过哪些关键步骤?",
        "请描述 {root} 当前被抽取出的代表性执行路径。",
    ],
    "result_origin": [
        "在 {root} 中,最终返回的 {target} 是在哪一步生成的?",
        "{target} 最早由哪个函数产生,并被谁继续使用?",
        "{root} 的返回结果 {target} 来自哪里?",
    ],
    "conditional_path": [
        "在 {root} 中,当前抽取到的 {branch_kind} 分支会经过哪些关键函数?",
        "围绕条件 {cond},{root} 当前抽取出的这条分支路径是什么?",
        "{root} 在与 {cond} 相关的当前分支片段中会调用哪些函数?",
    ],
    "error_handling_path": [
        "当 {trigger} 抛出异常时,{root} 会经过哪些关键函数?",
        "{root} 的错误处理流程是怎样的?",
        "如果 {trigger} 失败,{root} 会如何处理?",
    ],
    "core_support_classification": [
        "围绕 {root} 的局部调用子图中,哪些函数属于核心执行链,哪些属于辅助支撑函数?",
        "在 {root} 相关的函数里,请区分核心业务函数和辅助函数。",
    ],
    "local_subgraph_summary": [
        "以 {root} 为根的局部调用子图主要在完成什么任务?",
        "围绕 {root} 的局部调用网络可以分成哪几个阶段?",
        "请总结 {root} 局部子图的整体功能。",
    ],
    "dependency_trace": [
        "从 {root} 出发,经过最多 {hops} 跳调用,可以到达哪些{kind}?",
        "{root} 间接依赖了哪些{kind}?(最多 {hops} 跳)",
        "请追踪 {root} 通过调用链能到达的所有{kind}(深度不超过 {hops})。",
    ],
    "impact_scope": [
        "在当前有界静态反向调用图中,哪些函数直接调用了 {root},哪些函数属于保守的潜在影响点?",
        "请基于静态反向调用关系说明 {root} 的直接调用者和保守潜在影响点。",
        "围绕 {root},当前静态调用图能恢复出哪些直接调用者和有界潜在影响点?",
    ],
}


def _short(qualname: str) -> str:
    return qualname.rsplit(".", 1)[-1]


_QUALITATIVE_ONLY_QTYPES = {
    "core_support_classification",
    "local_subgraph_summary",
    "result_origin",
}


def _stable_gold_hash(gold: dict) -> str:
    payload = json.dumps(gold, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def render_item(fact: Fact, subgraph_nodes: list[str], rng: random.Random | None = None) -> BenchmarkItem:
    rng = rng or random.Random(0)
    root_short = _short(fact.root)
    p = fact.payload

    if fact.kind == "main_path":
        qtype = "execution_flow"
        q = rng.choice(_TEMPLATES[qtype]).format(root=root_short)
        gold = {
            "path_type": p["path_type"],
            "path_condition": p["path_condition"],
            "path_nodes": p["path_nodes"],
            "coverage_note": p["coverage_note"],
            "gold_entities": p["path_nodes"][1:],
            "gold_paths": [p["path_nodes"]],
        }
        natural = (
            f"{root_short} 当前抽取到的一条代表性路径依次经过:"
            + " → ".join(p["path_nodes"]) + "。"
        )

    elif fact.kind == "result_origin":
        qtype = "result_origin"
        q = rng.choice(_TEMPLATES[qtype]).format(root=root_short, target=p["target_object"])
        gold = {
            "target_object": p["target_object"],
            "source_entity": p["source_entity"],
            "consumer_entities": p["consumer_entities"],
            "return_dependency_kind": p.get("return_dependency_kind", ""),
            "return_wrapper": p.get("return_wrapper", ""),
        }
        natural = (
            f"当前抽取结果将 {p['target_object']} 的来源归因为 {p['source_entity']}"
            + (
                f",并标出后续显式消费者为 {', '.join(p['consumer_entities'])}。"
                if p["consumer_entities"]
                else "，当前结构化结果未标出后续显式消费者。"
            )
        )

    elif fact.kind == "conditional":
        qtype = "conditional_path"
        q = rng.choice(_TEMPLATES[qtype]).format(
            root=root_short,
            cond=p["condition"],
            branch_kind=p["branch_kind"],
        )
        gold = {
            "condition": p["condition"],
            "branch_kind": p["branch_kind"],
            "branch_span": p["branch_span"],
            "path_nodes": p["path_nodes"],
            "excluded_branches": p["excluded_branches"],
        }
        natural = (
            f"当前抽取的是 {p['branch_kind']} 分支,条件为 {p['condition']},"
            f"路径为 {' → '.join(p['path_nodes'])}。"
        )

    elif fact.kind == "error_handling":
        qtype = "error_handling_path"
        q = rng.choice(_TEMPLATES[qtype]).format(root=root_short, trigger=p["error_trigger"])
        gold = {
            "error_trigger": p["error_trigger"],
            "error_path": p["error_path"],
            "error_action": p.get("error_action", ""),
            "fallback_action": p.get("fallback_action"),
            "evidence_span": p.get("evidence_span", {}),
        }
        natural = (
            f"当 {p['error_trigger']} 出现异常时,{root_short} 会依次经过:"
            + " → ".join(p["error_path"]) + "。"
        )

    elif fact.kind == "role_classification":
        qtype = "core_support_classification"
        q = rng.choice(_TEMPLATES[qtype]).format(root=root_short)
        gold = {
            "core_entities": p["core_entities"],
            "support_entities": p["support_entities"],
        }
        natural = (
            f"核心函数:{', '.join(p['core_entities'])};"
            f"辅助函数:{', '.join(p['support_entities'])}。"
        )

    elif fact.kind == "stages":
        qtype = "local_subgraph_summary"
        q = rng.choice(_TEMPLATES[qtype]).format(root=root_short)
        gold = {
            "core_entities": p["summary_core_entities"],
            "stages": p["stages"],
        }
        natural = (
            f"当前抽取结果将该子图概括为 {' → '.join(p['stages'])} 几个阶段；"
            f"代表性核心函数为 {', '.join(p['summary_core_entities'])}。"
        )

    elif fact.kind == "dependency_trace":
        qtype = "dependency_trace"
        q = rng.choice(_TEMPLATES[qtype]).format(
            root=root_short,
            kind=p["target_description"],
            hops=p["max_hops"],
        )
        gold = {
            "target_kind": p["target_kind"],
            "target_entities": p["target_entities"],
            "sample_path": p["sample_path"],
        }
        natural = (
            f"从 {root_short} 出发,可以到达的{p['target_description']}有:"
            f"{', '.join(p['target_entities'])}。"
            f" 一条代表性路径:{' → '.join(p['sample_path'])}。"
        )

    elif fact.kind == "impact_scope":
        qtype = "impact_scope"
        q = rng.choice(_TEMPLATES[qtype]).format(root=root_short)
        gold = {
            "static_direct_callers": p["static_direct_callers"],
            "potentially_affected_callers": p["potentially_affected_callers"],
        }
        natural = (
            f"在当前有界静态反向调用图中,{root_short} 的直接调用者:"
            f"{', '.join(p['static_direct_callers'])};"
            f"保守潜在影响点:{', '.join(p['potentially_affected_callers'])}。"
        )

    else:
        raise ValueError(f"unknown fact kind: {fact.kind}")

    benchmark_tier = (
        "qualitative_only" if qtype in _QUALITATIVE_ONLY_QTYPES else "main_experiment"
    )
    return BenchmarkItem(
        question_type=qtype,
        root_function=fact.root,
        question=q,
        gold_answer=gold,
        natural_answer=natural,
        subgraph_nodes=subgraph_nodes,
        structure_gold_hash=_stable_gold_hash(gold),
        benchmark_tier=benchmark_tier,
    )
