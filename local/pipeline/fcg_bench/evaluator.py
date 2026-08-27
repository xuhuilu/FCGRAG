"""Benchmark 评估器。

用法:
    from fcg_bench.evaluator import evaluate_prediction, evaluate_dataset
    score = evaluate_prediction(gold_item, pred_dict)

每类题的指标(都落在 [0, 1]):

- execution_flow            : shared_prefix / branch_workflow / key_gates 三段式
- result_origin             : source 精确匹配 * 0.6 + consumer F1 * 0.3 + target 包含 * 0.1
- conditional_path          : true_path F1 与 false_path F1 的均值
- error_handling_path       : error_path F1 + trigger 精确匹配 * 0.2
- core_support_classification: 核心 F1 与 辅助 F1 的均值
- local_subgraph_summary    : stages 顺序 LCS 占比 + core F1,各占一半

pred_dict 的字段要求和 gold_answer 一致(大小写不敏感,短名或 qualname 都行)。
"""
from __future__ import annotations
from typing import Any
from .gold_v2 import is_execution_flow_gold_v21, is_gold_v2, legacy_gold_view
from .schema import BenchmarkItem

try:  # T3 remediation: unified generator/evaluator normalization (v2 rule)
    from fcg_benchmark.v2_t3_remediation.normalization import norm as _norm
except ImportError:  # fcg_benchmark/ itself is on sys.path
    from v2_t3_remediation.normalization import norm as _norm  # type: ignore


def _norm_v1(x: str) -> str:
    """Frozen v1 rule, kept for drift reporting (do not use for scoring)."""
    return x.rsplit(".", 1)[-1].strip().lower()


def _set(items: Any) -> set[str]:
    if not items:
        return set()
    if isinstance(items, str):
        items = [items]
    return {_norm(x) for x in items if isinstance(x, str) and x}


def _f1(gold: set[str], pred: set[str]) -> float:
    if not gold and not pred:
        return 1.0
    if not gold or not pred:
        return 0.0
    tp = len(gold & pred)
    if tp == 0:
        return 0.0
    p = tp / len(pred)
    r = tp / len(gold)
    return 2 * p * r / (p + r)


def _kendall_order(gold: list[str], pred: list[str]) -> float:
    """对 gold 里每一对有序 (a, b),检查 pred 里是否也保持 a 在 b 前。返回比例。"""
    g = [_norm(x) for x in gold]
    p = [_norm(x) for x in pred]
    pos = {x: i for i, x in enumerate(p)}
    pairs = 0
    correct = 0
    for i in range(len(g)):
        for j in range(i + 1, len(g)):
            if g[i] in pos and g[j] in pos:
                pairs += 1
                if pos[g[i]] < pos[g[j]]:
                    correct += 1
    return correct / pairs if pairs else (1.0 if g == p else 0.0)


def _lcs_ratio(gold: list[str], pred: list[str]) -> float:
    """gold 与 pred 的最长公共子序列长度 / len(gold)。"""
    g = [_norm(x) for x in gold]
    p = [_norm(x) for x in pred]
    if not g:
        return 1.0
    m, n = len(g), len(p)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m):
        for j in range(n):
            if g[i] == p[j]:
                dp[i + 1][j + 1] = dp[i][j] + 1
            else:
                dp[i + 1][j + 1] = max(dp[i][j + 1], dp[i + 1][j])
    return dp[m][n] / m


# ---------- 每种题型的打分函数 ----------

def _order_constraints_score(order_constraints: list[list[str]], pred_path: list[str]) -> float:
    if not order_constraints:
        return 1.0
    pos = {_norm(x): i for i, x in enumerate(pred_path)}
    total = 0
    correct = 0
    for pair in order_constraints:
        if not isinstance(pair, list) or len(pair) != 2:
            continue
        a, b = _norm(pair[0]), _norm(pair[1])
        total += 1
        if a in pos and b in pos and pos[a] < pos[b]:
            correct += 1
    return correct / total if total else 1.0


def _first_path(pred: dict) -> list[str]:
    path = pred.get("path_nodes") or pred.get("shared_prefix") or []
    if isinstance(path, list) and path:
        return path
    candidates = pred.get("branch_workflows") or pred.get("gold_paths") or pred.get("paths") or []
    if isinstance(candidates, list) and candidates:
        first = candidates[0]
        if isinstance(first, list):
            return first
    fallback = pred.get("path") or []
    return fallback if isinstance(fallback, list) else []


def _workflow_paths_from_pred(pred: dict) -> list[list[str]]:
    paths: list[list[str]] = []
    for key in ("branch_workflows", "gold_paths", "paths"):
        value = pred.get(key) or []
        if isinstance(value, list):
            for path in value:
                if isinstance(path, list) and path:
                    paths.append(path)
    first = _first_path(pred)
    if first:
        paths.append(first)
    deduped: list[list[str]] = []
    seen: set[tuple[str, ...]] = set()
    for path in paths:
        key = tuple(_norm(x) for x in path if isinstance(x, str) and x)
        if not key or key in seen:
            continue
        seen.add(key)
        deduped.append(path)
    return deduped


def _pred_root(pred: dict) -> str:
    for candidate in (
        pred.get("root_entity"),
        (_first_path(pred) or [""])[0],
    ):
        if isinstance(candidate, str) and candidate:
            return _norm(candidate)
    return ""


def _path_match_score(gold_path: list[str], pred_path: list[str]) -> float:
    if not gold_path:
        return 1.0
    if not pred_path:
        return 0.0
    set_f1 = _f1(_set(gold_path), _set(pred_path))
    order = _lcs_ratio(gold_path, pred_path)
    return 0.5 * set_f1 + 0.5 * order


def _prefix_match(pred: dict, gold_prefix: list[str]) -> float:
    pred_prefix = pred.get("shared_prefix")
    if not isinstance(pred_prefix, list) or not pred_prefix:
        pred_prefix = _first_path(pred)
    return _lcs_ratio(gold_prefix, pred_prefix or [])


def _best_branch_match(pred: dict, gold_branches: list[list[str]]) -> float:
    if not gold_branches:
        return 1.0
    pred_paths = _workflow_paths_from_pred(pred)
    if not pred_paths:
        return 0.0
    best = 0.0
    for gold_path in gold_branches:
        for pred_path in pred_paths:
            best = max(best, _path_match_score(gold_path, pred_path))
    return best


def _gate_match(pred: dict, gold_gates: list[str]) -> float:
    if not gold_gates:
        return 1.0
    pred_gates = pred.get("key_gates")
    if not isinstance(pred_gates, list):
        pred_gates = []
    if not pred_gates:
        pred_gates = [
            step
            for path in _workflow_paths_from_pred(pred)
            for step in path
            if isinstance(step, str) and _norm(step).startswith(("is_", "has_", "check_", "validate_", "verify_", "should_"))
        ]
    return _f1(_set(gold_gates), _set(pred_gates))


def _score_execution_flow(gold: dict, pred: dict) -> float:
    if is_execution_flow_gold_v21(gold):
        root = _norm(gold.get("root_entity", ""))
        if not root or _pred_root(pred) != root:
            return 0.0
        prefix_score = _prefix_match(pred, gold.get("shared_prefix", []) or [gold.get("root_entity", "")])
        if gold.get("singleton", False):
            return prefix_score
        branch_score = _best_branch_match(pred, gold.get("branch_workflows", []) or [])
        gate_score = _gate_match(pred, gold.get("key_gates", []) or [])
        return 0.4 * prefix_score + 0.4 * branch_score + 0.2 * gate_score

    if is_gold_v2(gold):
        hard = gold.get("hard_constraints", {}) or {}
        soft = gold.get("soft_constraints", {}) or {}
        required = hard.get("required_entities", []) or []
        acceptable_paths = soft.get("acceptable_paths", []) or []
        order_constraints = hard.get("order_constraints", []) or []
        p_path = pred.get("path_nodes") or (
            (pred.get("gold_paths") or pred.get("paths") or [pred.get("path") or []])[0]
            if isinstance(pred.get("gold_paths") or pred.get("paths"), list)
            else (pred.get("path") or [])
        )
        if not isinstance(p_path, list):
            p_path = []
        pred_steps = p_path[1:] if p_path else []
        required_steps = required[1:] if len(required) > 1 else []
        req_recall = (
            len(_set(required_steps) & _set(pred_steps)) / len(_set(required_steps))
            if _set(required_steps) else 1.0
        )
        best_path_score = 0.0
        for g_path in acceptable_paths or [required]:
            path_f1 = _f1(_set(g_path[1:]), _set(pred_steps))
            order = _kendall_order(g_path, p_path)
            best_path_score = max(best_path_score, 0.5 * path_f1 + 0.5 * order)
        constraint_score = _order_constraints_score(order_constraints, p_path)
        return 0.6 * best_path_score + 0.25 * req_recall + 0.15 * constraint_score

    g_path = gold.get("path_nodes") or (gold.get("gold_paths") or [[]])[0]
    p_path = pred.get("path_nodes") or (
        (pred.get("gold_paths") or pred.get("paths") or [pred.get("path") or []])[0]
        if isinstance(pred.get("gold_paths") or pred.get("paths"), list)
        else (pred.get("path") or [])
    )
    if not isinstance(p_path, list):
        p_path = []
    f1 = _f1(_set(g_path[1:]), _set(p_path[1:] if p_path else []))
    order = _kendall_order(g_path, p_path)
    return 0.5 * f1 + 0.5 * order


def _score_result_origin(gold: dict, pred: dict) -> float:
    src_ok = 1.0 if _norm(gold.get("source_entity", "")) == _norm(pred.get("source_entity", "")) else 0.0
    cons_f1 = _f1(_set(gold.get("consumer_entities")), _set(pred.get("consumer_entities")))
    tgt_ok = 1.0 if _norm(gold.get("target_object", "")) == _norm(pred.get("target_object", "")) else 0.0
    return 0.6 * src_ok + 0.3 * cons_f1 + 0.1 * tgt_ok


def _score_conditional_path(gold: dict, pred: dict) -> float:
    if is_gold_v2(gold):
        gold = legacy_gold_view("conditional_path", gold)
    g_path = gold.get("path_nodes") or gold.get("true_path") or []
    p_path = pred.get("path_nodes") or pred.get("true_path") or []
    path_f1 = _f1(_set(g_path), _set(p_path))
    branch_ok = 1.0 if _norm(gold.get("branch_kind", "")) == _norm(pred.get("branch_kind", "")) else 0.0
    cond_ok = 1.0 if _norm(gold.get("condition", "")) == _norm(pred.get("condition", "")) else 0.0
    return 0.6 * path_f1 + 0.2 * branch_ok + 0.2 * cond_ok


def _score_error_handling_path(gold: dict, pred: dict) -> float:
    if is_gold_v2(gold):
        gold = legacy_gold_view("error_handling_path", gold)
    path_f1 = _f1(_set(gold.get("error_path")), _set(pred.get("error_path")))
    trig_ok = 1.0 if _norm(gold.get("error_trigger", "")) == _norm(pred.get("error_trigger", "")) else 0.0
    return 0.8 * path_f1 + 0.2 * trig_ok


def _score_core_support(gold: dict, pred: dict) -> float:
    core_f1 = _f1(_set(gold.get("core_entities")), _set(pred.get("core_entities")))
    supp_f1 = _f1(_set(gold.get("support_entities")), _set(pred.get("support_entities")))
    return 0.5 * core_f1 + 0.5 * supp_f1


def _score_local_summary(gold: dict, pred: dict) -> float:
    stage_lcs = _lcs_ratio(gold.get("stages") or [], pred.get("stages") or [])
    core_f1 = _f1(_set(gold.get("core_entities")), _set(pred.get("core_entities")))
    return 0.5 * stage_lcs + 0.5 * core_f1


def _score_dependency_trace(gold: dict, pred: dict) -> float:
    """
    target_entities 的集合 F1 占 0.7,
    sample_path 的顺序对齐(LCS)占 0.3。
    """
    if is_gold_v2(gold):
        gold = legacy_gold_view("dependency_trace", gold)
    targets_f1 = _f1(
        _set(gold.get("target_entities")),
        _set(pred.get("target_entities")),
    )
    g_path = gold.get("sample_path") or []
    p_path = pred.get("sample_path") or []
    path_lcs = _lcs_ratio(g_path, p_path) if g_path else 1.0
    return 0.7 * targets_f1 + 0.3 * path_lcs


def _score_impact_scope(gold: dict, pred: dict) -> float:
    """
    直接调用者 F1 占 0.6(更重要),间接调用者 F1 占 0.4。
    直接调用者权重更高是因为它更精确,而 indirect 集合大、噪声多。
    """
    if is_gold_v2(gold):
        gold = legacy_gold_view("impact_scope", gold)
    direct_f1 = _f1(
        _set(gold.get("static_direct_callers") or gold.get("direct_callers")),
        _set(pred.get("static_direct_callers") or pred.get("direct_callers")),
    )
    impacted_f1 = _f1(
        _set(gold.get("potentially_affected_callers") or gold.get("indirect_callers")),
        _set(pred.get("potentially_affected_callers") or pred.get("indirect_callers")),
    )
    return 0.6 * direct_f1 + 0.4 * impacted_f1


_SCORERS = {
    "execution_flow": _score_execution_flow,
    "result_origin": _score_result_origin,
    "conditional_path": _score_conditional_path,
    "error_handling_path": _score_error_handling_path,
    "core_support_classification": _score_core_support,
    "local_subgraph_summary": _score_local_summary,
    "dependency_trace": _score_dependency_trace,
    "impact_scope": _score_impact_scope,
}


def evaluate_prediction(item: BenchmarkItem, pred: dict) -> float:
    scorer = _SCORERS.get(item.question_type)
    if scorer is None:
        return 0.0
    try:
        return float(scorer(item.gold_answer, pred or {}))
    except Exception:
        return 0.0


def evaluate_dataset(
    items: list[BenchmarkItem],
    preds: list[dict],
) -> dict[str, Any]:
    """返回每类题型的均分和整体均分。preds 长度必须与 items 相同。"""
    assert len(items) == len(preds), "items / preds 长度不一致"
    per_type: dict[str, list[float]] = {}
    for it, pr in zip(items, preds):
        s = evaluate_prediction(it, pr)
        per_type.setdefault(it.question_type, []).append(s)
    summary = {qt: sum(v) / len(v) for qt, v in per_type.items()}
    all_scores = [s for v in per_type.values() for s in v]
    summary["_overall"] = sum(all_scores) / len(all_scores) if all_scores else 0.0
    summary["_counts"] = {qt: len(v) for qt, v in per_type.items()}
    return summary
