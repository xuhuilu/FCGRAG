from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any

from .ast_analyzer import FunctionAST
from .schema import CallGraph


_HIGH_VALUE_PREFIXES = (
    "convert_",
    "_convert_",
    "format_",
    "_format_",
    "build_",
    "_build_",
    "construct_",
    "_construct_",
    "parse_",
    "_parse_",
    "prepare",
    "_prepare",
    "merge_",
    "_merge_",
    "_generate_helper",
)

_GATE_PREFIXES = (
    "is_",
    "_is_",
    "has_",
    "_has_",
    "should_",
    "_should_",
    "check_",
    "_check_",
    "supports_",
    "_supports_",
    "use_",
    "_use_",
)


@dataclass
class ValidationMessage:
    code: str
    level: str
    message: str
    evidence: dict[str, Any]


@dataclass
class ValidationResult:
    item_id: str
    root_function: str
    root_qualname: str | None
    errors: list[ValidationMessage]
    warnings: list[ValidationMessage]


def _short(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _norm_set(values: list[str]) -> set[str]:
    return {_short(v) for v in values if isinstance(v, str) and v}


def _flatten_gold_nodes(gold: dict[str, Any]) -> list[str]:
    nodes: list[str] = []
    nodes.extend(gold.get("shared_prefix", []) or [])
    nodes.extend(gold.get("key_gates", []) or [])
    nodes.extend(gold.get("optional_steps", []) or [])
    for path in gold.get("branch_workflows", []) or []:
        if isinstance(path, list):
            nodes.extend(path)
    return [node for node in nodes if isinstance(node, str) and node]


def _root_lookup(root_ref: str, graph: CallGraph) -> str | None:
    if root_ref in graph.nodes:
        return root_ref
    matches = [qn for qn in graph.nodes if _short(qn) == root_ref]
    if len(matches) == 1:
        return matches[0]
    return None


def _branch_ids_by_name(fa: FunctionAST) -> dict[str, set[str]]:
    mapping: dict[str, set[str]] = defaultdict(set)
    for idx, branch in enumerate(fa.branches):
        branch_id = f"{idx}:{branch.branch_kind}:{branch.condition}"
        for name in branch.path_calls:
            if isinstance(name, str) and name:
                mapping[name].add(branch_id)
    return mapping


def _mainline_high_value_candidates(fa: FunctionAST) -> list[str]:
    first_branch_line = min((b.start_line for b in fa.branches if b.start_line > 0), default=None)
    seen: set[str] = set()
    out: list[str] = []
    for cs in sorted(fa.call_sites, key=lambda c: (c.lineno, c.func_name)):
        name = cs.func_name
        if not name or cs.in_branch or cs.in_except:
            continue
        if first_branch_line is not None and cs.lineno >= first_branch_line:
            break
        if name in seen:
            continue
        seen.add(name)
        low = name.lower()
        if any(low.startswith(prefix) for prefix in _HIGH_VALUE_PREFIXES):
            out.append(name)
    return out


def _suspicious_gate(name: str) -> bool:
    low = (name or "").lower()
    return not any(low.startswith(prefix) for prefix in _GATE_PREFIXES)


def _validate_e001(gold: dict[str, Any], allowed_nodes: set[str]) -> ValidationMessage | None:
    offenders = sorted({node for node in _flatten_gold_nodes(gold) if _short(node) not in allowed_nodes})
    if offenders:
        return ValidationMessage(
            code="E001",
            level="error",
            message="NodeOutOfSubgraph",
            evidence={"offenders": offenders},
        )
    return None


def _validate_e002(gold: dict[str, Any], fa: FunctionAST | None) -> ValidationMessage | None:
    if fa is None:
        return None
    branch_ids = _branch_ids_by_name(fa)
    bad_paths: list[dict[str, Any]] = []
    for idx, workflow in enumerate(gold.get("branch_workflows", []) or []):
        if not isinstance(workflow, list):
            continue
        branch_specific = []
        for step in workflow[1:]:
            ids = branch_ids.get(step, set())
            if ids:
                branch_specific.append((step, ids))
        if len(branch_specific) < 2:
            continue
        common = set.intersection(*(ids for _, ids in branch_specific))
        if not common:
            bad_paths.append({
                "path_index": idx,
                "workflow": workflow,
                "branch_membership": {step: sorted(ids) for step, ids in branch_specific},
            })
    if bad_paths:
        return ValidationMessage(
            code="E002",
            level="error",
            message="MixedExclusiveBranches",
            evidence={"bad_paths": bad_paths},
        )
    return None


def _validate_e003(gold: dict[str, Any]) -> ValidationMessage | None:
    root = gold.get("root_entity", "")
    shared_prefix = gold.get("shared_prefix", []) or []
    branch_workflows = gold.get("branch_workflows", []) or []
    evidence: dict[str, Any] = {}
    bad = False
    if not shared_prefix or shared_prefix[0] != root:
        bad = True
        evidence["shared_prefix"] = shared_prefix
    bad_paths = [path for path in branch_workflows if not isinstance(path, list) or not path or path[0] != root]
    if bad_paths:
        bad = True
        evidence["branch_workflows"] = bad_paths
    if bad:
        return ValidationMessage(
            code="E003",
            level="error",
            message="BrokenRootPrefix",
            evidence=evidence,
        )
    return None


def _validate_e004(gold: dict[str, Any]) -> ValidationMessage | None:
    singleton = bool(gold.get("singleton", False))
    branch_workflows = gold.get("branch_workflows", []) or []
    if singleton and branch_workflows:
        return ValidationMessage(
            code="E004",
            level="error",
            message="SingletonBranchMismatch",
            evidence={"singleton": singleton, "branch_workflows": branch_workflows},
        )
    if not singleton and not branch_workflows:
        return ValidationMessage(
            code="E004",
            level="error",
            message="SingletonBranchMismatch",
            evidence={"singleton": singleton, "branch_workflows": branch_workflows},
        )
    return None


def _validate_w101(gold: dict[str, Any]) -> ValidationMessage | None:
    shared_prefix = gold.get("shared_prefix", []) or []
    if len(shared_prefix) == 1:
        return ValidationMessage(
            code="W101",
            level="warning",
            message="RootOnlyPrefix",
            evidence={"shared_prefix": shared_prefix},
        )
    return None


def _validate_w102(gold: dict[str, Any], fa: FunctionAST | None) -> ValidationMessage | None:
    if fa is None:
        return None
    shared_prefix = _norm_set(gold.get("shared_prefix", []) or [])
    candidates = [name for name in _mainline_high_value_candidates(fa) if _short(name) not in shared_prefix]
    if candidates:
        return ValidationMessage(
            code="W102",
            level="warning",
            message="MissingHighValuePrefixStep",
            evidence={"missing_candidates": candidates[:5]},
        )
    return None


def _validate_w103(gold: dict[str, Any]) -> ValidationMessage | None:
    short_paths = [path for path in gold.get("branch_workflows", []) or [] if isinstance(path, list) and len(path) <= 2]
    if short_paths:
        return ValidationMessage(
            code="W103",
            level="warning",
            message="TooShortBranch",
            evidence={"short_paths": short_paths[:5]},
        )
    return None


def _validate_w104(gold: dict[str, Any]) -> ValidationMessage | None:
    offenders = [name for name in gold.get("key_gates", []) or [] if _suspicious_gate(name)]
    if offenders:
        return ValidationMessage(
            code="W104",
            level="warning",
            message="SuspiciousGateName",
            evidence={"offenders": offenders},
        )
    return None


def validate_execution_flow_item(
    item: dict[str, Any],
    *,
    graph: CallGraph,
    ast_map: dict[str, FunctionAST],
) -> ValidationResult:
    gold = item.get("gold_answer", {}) or {}
    root_fn = item.get("root_function", "")
    root_qn = _root_lookup(root_fn, graph)
    fa = ast_map.get(root_qn) if root_qn else None
    root_entity = gold.get("root_entity", _short(root_fn))
    allowed_nodes = _norm_set(item.get("subgraph_nodes", []) or []) | {_short(root_entity)}

    errors: list[ValidationMessage] = []
    warnings: list[ValidationMessage] = []

    for validator in (
        lambda: _validate_e001(gold, allowed_nodes),
        lambda: _validate_e002(gold, fa),
        lambda: _validate_e003(gold),
        lambda: _validate_e004(gold),
    ):
        result = validator()
        if result is not None:
            errors.append(result)

    for validator in (
        lambda: _validate_w101(gold),
        lambda: _validate_w102(gold, fa),
        lambda: _validate_w103(gold),
        lambda: _validate_w104(gold),
    ):
        result = validator()
        if result is not None:
            warnings.append(result)

    return ValidationResult(
        item_id=item.get("item_id", ""),
        root_function=root_fn,
        root_qualname=root_qn,
        errors=errors,
        warnings=warnings,
    )


def collect_w201(items: list[dict[str, Any]]) -> list[ValidationResult]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        gold = item.get("gold_answer", {}) or {}
        root_entity = gold.get("root_entity", _short(item.get("root_function", "")))
        groups[_short(root_entity)].append(item)

    out: list[ValidationResult] = []
    for root_short, grouped_items in groups.items():
        host_functions = sorted({item.get("root_function", "") for item in grouped_items})
        if len(host_functions) <= 1:
            continue
        warning = ValidationMessage(
            code="W201",
            level="warning",
            message="AmbiguousRootShortName",
            evidence={
                "root_entity": root_short,
                "host_functions": host_functions,
                "item_ids": [item.get("item_id", "") for item in grouped_items],
            },
        )
        for item in grouped_items:
            out.append(
                ValidationResult(
                    item_id=item.get("item_id", ""),
                    root_function=item.get("root_function", ""),
                    root_qualname=None,
                    errors=[],
                    warnings=[warning],
                )
            )
    return out


def summarize_results(results: list[ValidationResult]) -> dict[str, Any]:
    error_counts: Counter[str] = Counter()
    warning_counts: Counter[str] = Counter()
    representative: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for result in results:
        for msg in result.errors + result.warnings:
            counter = error_counts if msg.level == "error" else warning_counts
            counter[msg.code] += 1
            if len(representative[msg.code]) < 3:
                representative[msg.code].append(
                    {
                        "item_id": result.item_id,
                        "root_function": result.root_function,
                        "message": msg.message,
                        "evidence": msg.evidence,
                    }
                )

    return {
        "total_items": len(results),
        "error_counts": dict(sorted(error_counts.items())),
        "warning_counts": dict(sorted(warning_counts.items())),
        "representative_failures": dict(sorted(representative.items())),
    }
