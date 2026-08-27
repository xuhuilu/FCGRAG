"""English template generation for clean-v1 benchmark artifacts.

This module generates English questions, explanation references, and claims
directly from validated structure_gold fields. It does not translate Chinese
text and does not call any external model.
"""
from __future__ import annotations

import re
from typing import Any

from .gold_v2 import is_execution_flow_gold_v21, is_gold_v2, legacy_gold_view


def _short(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _join_names(values: list[str]) -> str:
    return ", ".join(values) if values else "(none)"


def _path_text(nodes: list[str]) -> str:
    return " -> ".join(nodes) if nodes else "(empty)"


def _span_text(span: dict[str, Any] | None) -> str:
    span = span or {}
    return f"{span.get('start_line', 0)}-{span.get('end_line', 0)}"


def build_question_en(item: dict) -> str:
    qtype = item["question_type"]
    root = _short(item["root_function"])
    gold = item["gold_answer"]
    if is_gold_v2(gold) and qtype != "execution_flow":
        gold = legacy_gold_view(qtype, gold)

    if qtype == "dependency_trace":
        return (
            f"Within the current static FCG sample, which {gold.get('target_kind', 'targets')} "
            f"are reachable from {root}?"
        )

    if qtype == "conditional_path":
        return (
            f"For the currently extracted {gold.get('branch_kind', 'branch')} around the condition "
            f"{gold.get('condition', '')}, what path is recorded in {root}?"
        )

    if qtype == "execution_flow":
        return f"What representative workflow is currently extracted for {root}?"

    if qtype == "impact_scope":
        return (
            f"In the bounded static reverse call graph, which functions are static direct callers of {root}, "
            "and which functions are conservative potential impact points?"
        )

    if qtype == "error_handling_path":
        trigger = gold.get("trigger_condition", "") or gold.get("error_trigger", "")
        return (
            f"Within the current evidence_span for {root}, what error-related path is recorded when "
            f"{trigger} is triggered?"
        )

    raise ValueError(f"unsupported question type: {qtype}")


def build_natural_answer_en(item: dict) -> str:
    qtype = item["question_type"]
    root = _short(item["root_function"])
    gold = item["gold_answer"]
    if is_gold_v2(gold) and qtype != "execution_flow":
        gold = legacy_gold_view(qtype, gold)

    if qtype == "dependency_trace":
        targets = gold.get("target_entities", [])
        sample_path = gold.get("sample_path", [])
        target_kind = gold.get("target_kind", "targets")
        parts = [
            f"From {root}, the statically reachable {target_kind} are {_join_names(targets)}."
        ]
        if sample_path:
            parts.append(f"One representative static path is {_path_text(sample_path)}.")
        return " ".join(parts)

    if qtype == "conditional_path":
        branch_kind = gold.get("branch_kind", "branch")
        condition = gold.get("condition", "")
        path_nodes = gold.get("path_nodes", [])
        return (
            f"The extracted {branch_kind} branch around condition {condition} records the path "
            f"{_path_text(path_nodes)}."
        )

    if qtype == "execution_flow":
        if is_execution_flow_gold_v21(gold):
            shared_prefix = gold.get("shared_prefix", []) or [root]
            branch_workflows = gold.get("branch_workflows", []) or []
            key_gates = gold.get("key_gates", []) or []
            if gold.get("singleton", False) or not branch_workflows:
                return f"One representative workflow for {root} is {_path_text(shared_prefix)}."
            parts = [
                f"The shared prefix for {root} is {_path_text(shared_prefix)}.",
                "Representative branch-aware workflows include "
                + "; ".join(_path_text(path) for path in branch_workflows[:2])
                + ".",
            ]
            if key_gates:
                parts.append(f"Key gates include {_join_names(key_gates)}.")
            return " ".join(parts)
        if is_gold_v2(gold):
            soft = gold.get("soft_constraints", {}) or {}
            hard = gold.get("hard_constraints", {}) or {}
            acceptable_paths = soft.get("acceptable_paths") or []
            path = acceptable_paths[-1] if acceptable_paths else []
            if not path:
                path = hard.get("required_entities", []) or [root]
            return f"One representative workflow for {root} is {_path_text(path)}."
        path_nodes = gold.get("path_nodes", [])
        return f"One representative workflow for {root} is {_path_text(path_nodes)}."

    if qtype == "impact_scope":
        direct = gold.get("static_direct_callers", []) or gold.get("direct_callers", [])
        impacted = gold.get("potentially_affected_callers", []) or gold.get("indirect_callers", [])
        return (
            f"In the bounded static reverse call graph, the direct callers of {root} are {_join_names(direct)}; "
            f"the conservative potential impact points are {_join_names(impacted)}."
        )

    if qtype == "error_handling_path":
        trigger = gold.get("trigger_condition", "") or gold.get("error_trigger", "")
        error_path = gold.get("error_path", [])
        return (
            f"When {trigger} is triggered, {root} follows the error-related path "
            f"{_path_text(error_path)}."
        )

    raise ValueError(f"unsupported question type: {qtype}")


def build_explanation_reference_en(item: dict) -> str:
    qtype = item["question_type"]
    root = _short(item["root_function"])
    gold = item["gold_answer"]
    if is_gold_v2(gold) and qtype != "execution_flow":
        gold = legacy_gold_view(qtype, gold)

    if qtype == "dependency_trace":
        targets = gold.get("target_entities", [])
        sample_path = gold.get("sample_path", [])
        target_kind = gold.get("target_kind", "")
        return (
            f"The structured result indicates that the following {target_kind} are statically reachable "
            f"in the FCG: {_join_names(targets)}. One representative static reachability path is "
            f"{_path_text(sample_path)}. This describes static FCG reachability and does not guarantee "
            "runtime execution."
        )

    if qtype == "conditional_path":
        branch_kind = gold.get("branch_kind", "")
        condition = gold.get("condition", "")
        branch_span = _span_text(gold.get("branch_span", {}))
        path_nodes = gold.get("path_nodes", [])
        return (
            f"The structured result records the extracted {branch_kind} branch for the condition "
            f"{condition}, with branch span {branch_span}. Within this extracted branch, the recorded "
            f"path is {_path_text(path_nodes)}. This explanation only covers the extracted branch and "
            "does not imply that other false/else/elif branches behave the same way."
        )

    if qtype == "execution_flow":
        if is_execution_flow_gold_v21(gold):
            shared_prefix = gold.get("shared_prefix", []) or []
            branch_workflows = gold.get("branch_workflows", []) or []
            key_gates = gold.get("key_gates", []) or []
            parts = [
                f"The structured result records a branch-aware representative workflow rooted at {root}.",
                f"The shared workflow prefix is {_path_text(shared_prefix)}.",
            ]
            if branch_workflows:
                parts.append(
                    "Representative branch workflows include "
                    + "; ".join(_path_text(path) for path in branch_workflows[:2])
                    + "."
                )
            if key_gates:
                parts.append(f"Key gates include {_join_names(key_gates)}.")
            parts.append(
                "This is one representative workflow rather than a complete or unique execution flow."
            )
            return " ".join(parts)
        if is_gold_v2(gold):
            hard = gold.get("hard_constraints", {}) or {}
            soft = gold.get("soft_constraints", {}) or {}
            required = hard.get("required_entities", []) or []
            optional = soft.get("optional_entities", []) or []
            parts = [
                f"The structured result records one representative workflow rooted at {_short(required[0]) if required else '(unknown)'}.",
            ]
            if len(required) > 1:
                parts.append(f"The required workflow steps are {_join_names(required[1:])}.")
            if optional:
                parts.append(f"Optional workflow steps include {_join_names(optional)}.")
            parts.append(
                "This is one representative workflow rather than a complete or unique execution flow."
            )
            return " ".join(parts)
        path_nodes = gold.get("path_nodes", [])
        path_condition = gold.get("path_condition", "")
        parts = [f"The structured result provides one representative path: {_path_text(path_nodes)}."]
        if path_condition:
            parts.append(f"The recorded path condition is {path_condition}.")
        parts.append(
            "This is one extracted representative path rather than a complete or unique execution flow."
        )
        return " ".join(parts)

    if qtype == "impact_scope":
        direct = gold.get("static_direct_callers", []) or gold.get("direct_callers", [])
        impacted = gold.get("potentially_affected_callers", []) or gold.get("indirect_callers", [])
        return (
            f"In the bounded static reverse call graph, the static direct callers are {_join_names(direct)}. "
            f"The conservative potential impact points are {_join_names(impacted)}. These relations describe "
            "static reverse reachability only; they are not guaranteed runtime effects or complete dynamic "
            "impact sets."
        )

    if qtype == "error_handling_path":
        trigger = gold.get("trigger_condition", "") or gold.get("error_trigger", "")
        error_path = gold.get("error_path", [])
        error_action = gold.get("error_action", "") or "(not recorded)"
        fallback_action = gold.get("fallback_action")
        evidence_span = _span_text(gold.get("evidence_span", {}))
        parts = [
            f"Within evidence_span {evidence_span}, the structured result records an error-related path "
            f"triggered by {trigger}: {_path_text(error_path)}.",
            f"The recorded error_action is {error_action}.",
        ]
        error_message = gold.get("error_message", "")
        warns = gold.get("warns")
        except_returns = gold.get("except_returns")
        if error_message:
            parts.append(f"The recorded error_message is {error_message}.")
        if warns is not None:
            parts.append(f"The recorded warns value is {warns}.")
        if except_returns is not None:
            parts.append(f"The recorded except_returns value is {except_returns}.")
        if fallback_action:
            parts.append(f"The recorded fallback_action is {fallback_action}.")
        parts.append(
            "This explanation only covers the current evidence_span and does not represent the complete "
            "error-handling logic of the function."
        )
        return " ".join(parts)

    raise ValueError(f"unsupported question type: {qtype}")


def build_claims_en(item: dict) -> list[dict[str, str]]:
    qtype = item["question_type"]
    gold = item["gold_answer"]
    if is_gold_v2(gold) and qtype != "execution_flow":
        gold = legacy_gold_view(qtype, gold)
    claims: list[dict[str, str]] = []

    if qtype == "dependency_trace":
        targets = gold.get("target_entities", [])
        sample_path = gold.get("sample_path", [])
        target_kind = gold.get("target_kind", "")
        if targets:
            claims.append({
                "text": f"The statically reachable {target_kind} include {_join_names(targets)}.",
                "claim_type": "scope",
            })
        if sample_path:
            claims.append({
                "text": f"One representative static reachability path is {_path_text(sample_path)}.",
                "claim_type": "path",
            })
        claims.append({
            "text": "This item describes static FCG reachability rather than guaranteed runtime execution.",
            "claim_type": "scope",
        })
        return claims

    if qtype == "conditional_path":
        if gold.get("condition", ""):
            claims.append({
                "text": f"The branch condition is {gold['condition']}.",
                "claim_type": "comparison",
            })
        if gold.get("branch_kind", ""):
            claims.append({
                "text": f"The extracted branch kind is {gold['branch_kind']}.",
                "claim_type": "classification",
            })
        if gold.get("path_nodes", []):
            claims.append({
                "text": f"The recorded branch path is {_path_text(gold['path_nodes'])}.",
                "claim_type": "path",
            })
        claims.append({
            "text": (
                f"This explanation only covers the extracted branch at branch span "
                f"{_span_text(gold.get('branch_span', {}))} and does not stand for all branches."
            ),
            "claim_type": "scope",
        })
        return claims

    if qtype == "execution_flow":
        if is_execution_flow_gold_v21(gold):
            shared_prefix = gold.get("shared_prefix", []) or []
            branch_workflows = gold.get("branch_workflows", []) or []
            key_gates = gold.get("key_gates", []) or []
            if shared_prefix:
                claims.append({
                    "text": f"The shared workflow prefix is {_path_text(shared_prefix)}.",
                    "claim_type": "path",
                })
            for path in branch_workflows[:2]:
                claims.append({
                    "text": f"One representative branch workflow is {_path_text(path)}.",
                    "claim_type": "path",
                })
            if key_gates:
                claims.append({
                    "text": f"Key workflow gates include {_join_names(key_gates)}.",
                    "claim_type": "scope",
                })
            claims.append({
                "text": "This item records one representative workflow rather than a complete or unique execution flow.",
                "claim_type": "scope",
            })
            return claims
        if is_gold_v2(gold):
            hard = gold.get("hard_constraints", {}) or {}
            soft = gold.get("soft_constraints", {}) or {}
            required = hard.get("required_entities", []) or []
            optional = soft.get("optional_entities", []) or []
            order_constraints = hard.get("order_constraints", []) or []
            if required:
                claims.append({
                    "text": f"The representative workflow starts at {_short(required[0])}.",
                    "claim_type": "path",
                })
            if len(required) > 1:
                claims.append({
                    "text": f"The required workflow steps include {_join_names(required[1:])}.",
                    "claim_type": "path",
                })
            for pair in order_constraints:
                if isinstance(pair, list) and len(pair) == 2:
                    claims.append({
                        "text": f"{_short(pair[0])} occurs before {_short(pair[1])} in the representative workflow.",
                        "claim_type": "path",
                    })
            if optional:
                claims.append({
                    "text": f"Optional workflow steps include {_join_names(optional)}.",
                    "claim_type": "scope",
                })
            claims.append({
                "text": "This item records one representative workflow rather than a complete or unique execution flow.",
                "claim_type": "scope",
            })
            return claims
        path_nodes = gold.get("path_nodes", [])
        if path_nodes:
            claims.append({
                "text": f"The representative path starts at {_short(path_nodes[0])}.",
                "claim_type": "path",
            })
            claims.append({
                "text": f"The recorded representative path is {_path_text(path_nodes)}.",
                "claim_type": "path",
            })
        claims.append({
            "text": "This item records one representative path rather than a complete or unique execution flow.",
            "claim_type": "scope",
        })
        return claims

    if qtype == "impact_scope":
        direct = gold.get("static_direct_callers", []) or gold.get("direct_callers", [])
        impacted = gold.get("potentially_affected_callers", []) or gold.get("indirect_callers", [])
        if direct:
            claims.append({
                "text": f"The static direct callers are {_join_names(direct)}.",
                "claim_type": "scope",
            })
        if impacted:
            claims.append({
                "text": f"The conservative potential impact points are {_join_names(impacted)}.",
                "claim_type": "scope",
            })
        claims.append({
            "text": "These items are bounded static reverse-reachability candidates, not guaranteed runtime effects or complete dynamic impact sets.",
            "claim_type": "scope",
        })
        return claims

    if qtype == "error_handling_path":
        trigger = gold.get("trigger_condition", "") or gold.get("error_trigger", "")
        error_path = gold.get("error_path", [])
        error_action = gold.get("error_action", "")
        error_message = gold.get("error_message", "")
        warns = gold.get("warns")
        except_returns = gold.get("except_returns")
        fallback_action = gold.get("fallback_action")
        evidence_span = gold.get("evidence_span", {}) or {}
        if trigger:
            claims.append({
                "text": f"The recorded error-related trigger is {trigger}.",
                "claim_type": "origin",
            })
        if len(error_path) >= 2:
            claims.append({
                "text": f"The recorded error-related path is {_path_text(error_path)}.",
                "claim_type": "path",
            })
        if error_action:
            claims.append({
                "text": f"The recorded error_action is {error_action}.",
                "claim_type": "classification",
            })
        if error_message:
            claims.append({
                "text": f"The recorded error_message is {error_message}.",
                "claim_type": "classification",
            })
        if warns is not None:
            claims.append({
                "text": f"The recorded warns value is {warns}.",
                "claim_type": "classification",
            })
        if except_returns is not None:
            claims.append({
                "text": f"The recorded except_returns value is {except_returns}.",
                "claim_type": "classification",
            })
        if fallback_action:
            claims.append({
                "text": f"The recorded fallback_action is {fallback_action}.",
                "claim_type": "causation",
            })
        claims.append({
            "text": (
                f"This explanation only covers evidence_span {_span_text(evidence_span)} and does not "
                "represent the complete error-handling logic of the function."
            ),
            "claim_type": "scope",
        })
        return claims

    raise ValueError(f"unsupported question type: {qtype}")


def collect_allowed_code_tokens(item: dict) -> set[str]:
    gold = item["gold_answer"]
    allowed = {_short(item["root_function"])}

    def _collect(value: Any) -> None:
        if isinstance(value, str):
            for tok in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", value):
                allowed.add(tok)
        elif isinstance(value, list):
            for x in value:
                _collect(x)
        elif isinstance(value, dict):
            for x in value.values():
                _collect(x)

    _collect(gold)
    return allowed
