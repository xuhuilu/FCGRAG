"""Unified gold schema helpers.

Most question types still use the generic gold_v2 wrapper.
execution_flow now uses a dedicated gold_v2.1 branch-aware workflow
schema so it can model shared prefix stages, branch-specific workflows,
and key gates separately.
"""
from __future__ import annotations

import re
from typing import Any

from .ast_analyzer import FunctionAST
from .features import is_aux_like_name
from .schema import CallGraph


SCHEMA_VERSION = "gold_v2"
EXECUTION_FLOW_SCHEMA_VERSION = "gold_v2.1"
EXECUTION_FLOW_ANSWER_FORM = "branch_aware_workflow"

ANSWER_FORM_BY_QTYPE = {
    "execution_flow": EXECUTION_FLOW_ANSWER_FORM,
    "conditional_path": "branch_path",
    "error_handling_path": "error_path",
    "dependency_trace": "reachable_set",
    "impact_scope": "caller_sets",
}

_NOISE_CALLS = {
    "all",
    "any",
    "append",
    "bool",
    "cast",
    "dict",
    "enumerate",
    "extend",
    "get",
    "getattr",
    "hasattr",
    "info",
    "int",
    "isinstance",
    "issubclass",
    "items",
    "join",
    "keys",
    "len",
    "list",
    "pop",
    "print",
    "repr",
    "set",
    "setattr",
    "startswith",
    "str",
    "tuple",
    "type",
    "update",
    "values",
    "warn",
    "warning",
}
_LOGGER_TOKENS = ("debug", "trace", "tracing", "logger", "logging")
_PREDICATE_PREFIXES = (
    "is_",
    "_is_",
    "has_",
    "_has_",
    "check_",
    "_check_",
    "validate_",
    "_validate_",
    "verify_",
    "_verify_",
    "should_",
    "_should_",
)
_LIGHT_HELPER_PREFIXES = ("_get_", "get_", "_check_", "_validate_")
_ACTION_TOKENS = (
    "merge",
    "format",
    "convert",
    "build",
    "create",
    "process",
    "normalize",
    "parse",
    "render",
    "load",
    "prepare",
    "collect",
    "update",
    "write",
    "invoke",
    "stream",
    "generate",
)
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


def is_gold_v2(gold: dict[str, Any]) -> bool:
    return gold.get("schema_version") in {SCHEMA_VERSION, EXECUTION_FLOW_SCHEMA_VERSION}


def is_execution_flow_gold_v21(gold: dict[str, Any]) -> bool:
    return (
        gold.get("schema_version") == EXECUTION_FLOW_SCHEMA_VERSION
        and gold.get("answer_form") == EXECUTION_FLOW_ANSWER_FORM
    )


def make_gold_v2(
    qtype: str,
    *,
    root_short: str,
    hard_constraints: dict[str, Any],
    soft_constraints: dict[str, Any],
    scope_constraints: dict[str, Any] | None = None,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "answer_form": ANSWER_FORM_BY_QTYPE[qtype],
        "root_entity": root_short,
        "hard_constraints": hard_constraints,
        "soft_constraints": soft_constraints,
        "scope_constraints": scope_constraints or {
            "static_only": True,
            "branch_local_only": False,
            "error_local_only": False,
            "evidence_span": None,
        },
        "notes": notes or [],
    }


def _unique_keep_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _path_prefixes(path: list[str], min_len: int) -> list[list[str]]:
    if len(path) < min_len:
        return [path] if path else []
    return [path[:i] for i in range(min_len, len(path) + 1)]


def _workflow_note() -> list[str]:
    return [
        "This gold represents a representative workflow inside the root function rather than a strict call chain.",
    ]


def _is_noise_call(name: str) -> bool:
    low = (name or "").lower()
    if not name:
        return True
    if name in _NOISE_CALLS:
        return True
    if is_aux_like_name(name):
        return True
    return any(token in low for token in _LOGGER_TOKENS)


def _is_predicate_like(name: str) -> bool:
    low = (name or "").lower()
    return low.startswith(_PREDICATE_PREFIXES)


def _is_light_helper(name: str) -> bool:
    low = (name or "").lower()
    return low.startswith(_LIGHT_HELPER_PREFIXES)


def _has_action_token(name: str) -> bool:
    low = (name or "").lower()
    return any(token in low for token in _ACTION_TOKENS)


def _is_high_value_name(name: str) -> bool:
    low = (name or "").lower()
    return any(low.startswith(prefix) for prefix in _HIGH_VALUE_PREFIXES)


def _contains_identifier(text: str, name: str) -> bool:
    return re.search(rf"\b{re.escape(name)}\b", text or "") is not None


def _execution_flow_role(name: str) -> str:
    low = (name or "").lower()
    if not name:
        return "other"
    if "error" in low or "except" in low or "fallback" in low or name in {"warn", "warning"}:
        return "error_handler"
    if _is_predicate_like(name):
        return "gate"
    if any(token in low for token in ("prepare", "parse", "load")):
        return "preprocess"
    if "format" in low or "render" in low:
        return "format"
    if any(token in low for token in ("build", "create", "make", "construct")):
        return "build"
    if any(token in low for token in ("dispatch", "invoke", "execute", "run", "stream", "generate", "send")):
        return "dispatch"
    if any(
        token in low
        for token in (
            "merge",
            "convert",
            "process",
            "normalize",
            "collect",
            "update",
            "write",
            "translate",
        )
    ):
        return "transform"
    if _is_light_helper(name) or is_aux_like_name(name):
        return "low_value_helper"
    if any(token in low for token in ("wrap", "finalize", "dump")):
        return "postprocess"
    return "other"


def _is_core_stage_role(role: str) -> bool:
    return role in {"preprocess", "transform", "format", "build", "dispatch", "postprocess", "error_handler"}


def _allowed_execution_nodes(root_short: str, subgraph_nodes: list[str] | None) -> set[str]:
    allowed = {root_short}
    for name in subgraph_nodes or []:
        if isinstance(name, str) and name:
            allowed.add(name.rsplit(".", 1)[-1])
    return allowed


def _make_execution_flow_gold_v21(
    *,
    root_short: str,
    shared_prefix: list[str],
    branch_workflows: list[list[str]],
    key_gates: list[str],
    optional_steps: list[str] | None = None,
    singleton: bool = False,
    notes: list[str] | None = None,
) -> dict[str, Any]:
    return {
        "schema_version": EXECUTION_FLOW_SCHEMA_VERSION,
        "answer_form": EXECUTION_FLOW_ANSWER_FORM,
        "root_entity": root_short,
        "shared_prefix": _unique_keep_order(shared_prefix) or [root_short],
        "branch_workflows": _dedup_list_of_paths(branch_workflows),
        "key_gates": _unique_keep_order(key_gates),
        "optional_steps": _unique_keep_order(optional_steps or []),
        "singleton": bool(singleton),
        "notes": notes or _workflow_note(),
    }


def _dedup_list_of_paths(paths: list[list[str]]) -> list[list[str]]:
    seen: set[tuple[str, ...]] = set()
    out: list[list[str]] = []
    for path in paths:
        cleaned = _unique_keep_order(path)
        if not cleaned:
            continue
        key = tuple(cleaned)
        if key in seen:
            continue
        seen.add(key)
        out.append(cleaned)
    return out


_EXECUTION_FLOW_TARGETED_PREFIX_PATCHES: dict[str, dict[str, Any]] = {
    "BaseChatOpenAI.with_structured_output": {
        "shared_prefix": ["with_structured_output", "_is_pydantic_class"],
        "prepend_to_branches": "_is_pydantic_class",
        "singleton": False,
    },
    "ChatOllama.with_structured_output": {
        "shared_prefix": ["with_structured_output", "_is_pydantic_class"],
        "prepend_to_branches": "_is_pydantic_class",
        "singleton": False,
    },
    "render": {
        "shared_prefix": ["render", "tokenize"],
        "branch_workflows": [],
        "key_gates": [],
        "singleton": True,
    },
    "ChatFireworks.with_structured_output": {
        "shared_prefix": ["with_structured_output", "_is_pydantic_class"],
        "prepend_to_branches": "_is_pydantic_class",
        "singleton": False,
    },
    "ChatGroq.with_structured_output": {
        "shared_prefix": ["with_structured_output", "_is_pydantic_class"],
        "prepend_to_branches": "_is_pydantic_class",
        "singleton": False,
    },
}


def _apply_execution_flow_targeted_patch(
    *,
    root: str,
    gold: dict[str, Any],
    allowed_names: set[str],
) -> dict[str, Any]:
    patch = _EXECUTION_FLOW_TARGETED_PREFIX_PATCHES.get(root)
    if not patch:
        return gold

    shared_prefix = [
        name
        for name in patch.get("shared_prefix", gold.get("shared_prefix", []) or [])
        if name in allowed_names
    ]
    if not shared_prefix:
        shared_prefix = gold.get("shared_prefix", []) or [gold.get("root_entity", "")]

    branch_workflows = gold.get("branch_workflows", []) or []
    if "branch_workflows" in patch:
        branch_workflows = [
            [name for name in path if name in allowed_names]
            for path in patch["branch_workflows"]
        ]
    elif patch.get("prepend_to_branches"):
        step = patch["prepend_to_branches"]
        if step in allowed_names:
            patched_branches: list[list[str]] = []
            for path in branch_workflows:
                if not path:
                    continue
                root_entity = path[0]
                tail = path[1:]
                if step in tail:
                    patched = [root_entity] + tail
                else:
                    patched = [root_entity, step] + tail
                patched = [name for name in patched if name in allowed_names]
                if len(patched) > len(shared_prefix):
                    patched_branches.append(patched)
            branch_workflows = patched_branches

    key_gates = [name for name in (patch.get("key_gates", gold.get("key_gates", []) or [])) if name in allowed_names]
    singleton = bool(patch.get("singleton", gold.get("singleton", False)))
    if singleton:
        branch_workflows = []

    return {
        **gold,
        "shared_prefix": _unique_keep_order(shared_prefix) or [gold.get("root_entity", "")],
        "branch_workflows": _dedup_list_of_paths(branch_workflows),
        "key_gates": _unique_keep_order(key_gates),
        "singleton": singleton,
    }


def _extract_condition_gate_names(condition: str, allowed_names: set[str]) -> list[str]:
    names: list[str] = []
    for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", condition or ""):
        if allowed_names and token not in allowed_names:
            continue
        if _is_predicate_like(token):
            names.append(token)
    return _unique_keep_order(names)


def _build_execution_candidates(
    fa: FunctionAST,
    allowed_names: set[str],
    root_short: str,
) -> list[dict[str, Any]]:
    assign_targets_by_source: dict[str, list[str]] = {}
    for assign in fa.assigns:
        if assign.in_except or not assign.target or not assign.source_func:
            continue
        assign_targets_by_source.setdefault(assign.source_func, []).append(assign.target)

    seen_names: set[tuple[str, bool]] = set()
    entries: list[dict[str, Any]] = []
    for idx, cs in enumerate(sorted(fa.call_sites, key=lambda c: (c.lineno, c.func_name))):
        name = cs.func_name
        if cs.in_except or _is_noise_call(name):
            continue
        if name not in allowed_names or name == root_short:
            continue
        role = _execution_flow_role(name)
        seen_key = (name, cs.in_branch)
        if seen_key in seen_names:
            continue
        seen_names.add(seen_key)

        source_kind = "normal_call"
        if name in assign_targets_by_source:
            source_kind = "assignment"
        elif name == fa.return_source_func:
            source_kind = "return_related"

        score = 0.0
        if source_kind == "assignment":
            score += 2.5
        if source_kind == "return_related":
            score += 1.5
        if _has_action_token(name):
            score += 1.0
        if _is_high_value_name(name):
            score += 1.2
        if role == "gate":
            score += 0.2
        if role == "low_value_helper":
            score -= 1.5
        produced_targets = assign_targets_by_source.get(name, [])
        if produced_targets and any(
            _contains_identifier(later.arg_text, target)
            for later in fa.call_sites
            for target in produced_targets
            if later.lineno > cs.lineno and not later.in_except
        ):
            score += 1.2
        entries.append({
            "name": name,
            "lineno": cs.lineno,
            "control_region": "branch" if cs.in_branch else "mainline",
            "source_kind": source_kind,
            "score": score,
            "role": role,
            "is_core_stage": _is_core_stage_role(role),
            "index": idx,
        })
    return entries


def _extract_shared_prefix(
    root_short: str,
    mainline_steps: list[dict[str, Any]],
    first_major_branch_line: int | None,
    has_major_branches: bool,
) -> list[str]:
    prefix = [root_short]
    candidates = []
    for step in mainline_steps:
        if has_major_branches and first_major_branch_line is not None and step["lineno"] >= first_major_branch_line:
            break
        if step["role"] == "gate":
            continue
        if not (step["is_core_stage"] or step["score"] >= 1.5):
            continue
        candidates.append(step["name"])
        if len(candidates) >= 3:
            break
    if not candidates and not has_major_branches:
        for step in mainline_steps:
            if step["role"] == "gate":
                continue
            if step["is_core_stage"] or step["score"] >= 0.8:
                candidates.append(step["name"])
            if len(candidates) >= 3:
                break
    prefix.extend(candidates)
    return _unique_keep_order(prefix)


def _extract_branch_specific_steps(branch: dict[str, Any], shared_prefix: list[str]) -> list[str]:
    shared = set(shared_prefix)
    actions: list[str] = []
    for step in branch.get("steps", []):
        if step["name"] in shared:
            continue
        if step["role"] == "gate":
            continue
        if step["is_core_stage"] or step["score"] >= 1.0:
            actions.append(step["name"])
        if len(_unique_keep_order(actions)) >= 2:
            break
    return _unique_keep_order(actions)[:2]


def _extract_key_gates(branches: list[dict[str, Any]]) -> list[str]:
    gates: list[str] = []
    branch_count = len([branch for branch in branches if branch.get("is_major")])
    for branch in branches:
        if not branch.get("is_major"):
            continue
        gate_names = branch.get("gate_names", [])
        branch_actions = branch.get("actions", [])
        if gate_names and branch_actions:
            gates.extend(gate_names)
        elif gate_names and branch_count > 1:
            gates.extend(gate_names)
    return _unique_keep_order(gates)


def _build_execution_flow_v2(
    graph: CallGraph,
    root: str,
    fa: FunctionAST | None,
    subgraph_nodes: list[str] | None = None,
) -> dict[str, Any]:
    root_short = graph.nodes[root].short_name
    allowed_names = _allowed_execution_nodes(root_short, subgraph_nodes)
    if fa is None:
        return _make_execution_flow_gold_v21(
            root_short=root_short,
            shared_prefix=[root_short],
            branch_workflows=[],
            key_gates=[],
            singleton=True,
        )

    entries = _build_execution_candidates(fa, allowed_names, root_short)
    if not entries:
        return _make_execution_flow_gold_v21(
            root_short=root_short,
            shared_prefix=[root_short],
            branch_workflows=[],
            key_gates=[],
            singleton=True,
        )

    mainline_steps = sorted([e for e in entries if e["control_region"] == "mainline"], key=lambda e: e["lineno"])
    branches: list[dict[str, Any]] = []
    for branch in sorted(fa.branches, key=lambda b: (b.start_line, b.branch_kind, b.condition)):
        gate_names = _extract_condition_gate_names(branch.condition, allowed_names)
        seen_branch_names: set[str] = set()
        branch_steps: list[dict[str, Any]] = []
        for raw_name in branch.path_calls:
            if raw_name in seen_branch_names:
                continue
            seen_branch_names.add(raw_name)
            if _is_noise_call(raw_name):
                continue
            if raw_name not in allowed_names or raw_name == root_short:
                continue
            role = _execution_flow_role(raw_name)
            if role == "low_value_helper":
                continue
            score = 0.0
            if _is_high_value_name(raw_name):
                score += 1.0
            if _has_action_token(raw_name):
                score += 0.8
            if role == "gate":
                score += 0.2
            branch_steps.append({
                "name": raw_name,
                "role": role,
                "score": score,
                "is_core_stage": _is_core_stage_role(role),
            })
        branch_actions = [step for step in branch_steps if step["role"] != "gate" and (step["is_core_stage"] or step["score"] >= 1.0)]
        is_major = bool(branch_actions or gate_names)
        branches.append({
            "branch_kind": branch.branch_kind,
            "condition": branch.condition,
            "start_line": branch.start_line,
            "gate_names": gate_names,
            "steps": branch_steps,
            "actions": branch_actions,
            "is_major": is_major,
        })

    major_branches = [branch for branch in branches if branch["is_major"]]
    first_major_branch_line = min((branch["start_line"] for branch in major_branches if branch["start_line"] > 0), default=None)
    shared_prefix = _extract_shared_prefix(
        root_short,
        mainline_steps,
        first_major_branch_line,
        has_major_branches=bool(major_branches),
    )

    branch_workflows: list[list[str]] = []
    for branch in major_branches:
        branch_steps = _extract_branch_specific_steps(branch, shared_prefix)
        if len(shared_prefix) == 1 and len(branch_steps) == 1:
            # Weak [root, X] branches are suppressed; keep only decisive gates for this branch.
            continue
        if branch_steps:
            workflow = _unique_keep_order(shared_prefix + branch_steps)
            if len(workflow) > len(shared_prefix):
                branch_workflows.append(workflow)
    branch_workflows = _dedup_list_of_paths(branch_workflows)[:3]

    key_gates = _extract_key_gates(major_branches)
    singleton = not branch_workflows

    if singleton and len(shared_prefix) == 1:
        fallback_steps = [
            step["name"]
            for step in mainline_steps
            if step["is_core_stage"] or step["score"] >= 1.0
        ][:3]
        shared_prefix = _unique_keep_order([root_short] + fallback_steps)

    gold = _make_execution_flow_gold_v21(
        root_short=root_short,
        shared_prefix=shared_prefix,
        branch_workflows=branch_workflows,
        key_gates=key_gates,
        optional_steps=[],
        singleton=singleton,
    )
    return _apply_execution_flow_targeted_patch(root=root, gold=gold, allowed_names=allowed_names)


def convert_legacy_gold_to_v2(
    qtype: str,
    gold: dict[str, Any],
    *,
    root_short: str,
) -> dict[str, Any]:
    if is_gold_v2(gold):
        return gold

    if qtype == "execution_flow":
        path = gold.get("path_nodes") or (gold.get("gold_paths") or [[]])[0]
        path = path or [root_short]
        return _make_execution_flow_gold_v21(
            root_short=root_short,
            shared_prefix=path,
            branch_workflows=[],
            key_gates=[],
            optional_steps=[],
            singleton=True,
            notes=[gold.get("coverage_note", "")] if gold.get("coverage_note") else _workflow_note(),
        )

    if qtype == "conditional_path":
        path = gold.get("path_nodes", [])
        return make_gold_v2(
            qtype,
            root_short=root_short,
            hard_constraints={
                "required_entities": path,
                "required_labels": {
                    "condition": gold.get("condition", ""),
                    "branch_kind": gold.get("branch_kind", ""),
                },
                "order_constraints": [],
            },
            soft_constraints={
                "optional_entities": [],
                "acceptable_paths": [path] if path else [[root_short]],
                "acceptable_sets": [],
            },
            scope_constraints={
                "static_only": True,
                "branch_local_only": True,
                "error_local_only": False,
                "evidence_span": gold.get("branch_span", {}),
            },
            notes=[],
        )

    if qtype == "error_handling_path":
        path = gold.get("error_path", [])
        return make_gold_v2(
            qtype,
            root_short=root_short,
            hard_constraints={
                "required_entities": path,
                "required_labels": {
                    "error_trigger": gold.get("error_trigger", ""),
                    "error_action": gold.get("error_action", ""),
                },
                "order_constraints": [[path[i], path[i + 1]] for i in range(1, len(path) - 1)],
            },
            soft_constraints={
                "optional_entities": [],
                "acceptable_paths": [path] if path else [[root_short]],
                "acceptable_sets": [],
            },
            scope_constraints={
                "static_only": True,
                "branch_local_only": False,
                "error_local_only": True,
                "evidence_span": gold.get("evidence_span", {}),
            },
            notes=[],
        )

    if qtype == "dependency_trace":
        targets = gold.get("target_entities", [])
        sample_path = gold.get("sample_path", [])
        return make_gold_v2(
            qtype,
            root_short=root_short,
            hard_constraints={
                "required_entities": targets,
                "required_labels": {
                    "target_kind": gold.get("target_kind", ""),
                },
                "order_constraints": [],
            },
            soft_constraints={
                "optional_entities": [],
                "acceptable_paths": [sample_path] if sample_path else [],
                "acceptable_sets": [targets] if targets else [],
            },
            notes=["This item describes static FCG reachability rather than guaranteed runtime execution."],
        )

    if qtype == "impact_scope":
        direct = gold.get("static_direct_callers", []) or gold.get("direct_callers", [])
        impacted = gold.get("potentially_affected_callers", []) or gold.get("indirect_callers", [])
        return make_gold_v2(
            qtype,
            root_short=root_short,
            hard_constraints={
                "required_entities": direct,
                "required_labels": {},
                "order_constraints": [],
            },
            soft_constraints={
                "optional_entities": [],
                "acceptable_paths": [],
                "acceptable_sets": [impacted] if impacted else [],
            },
            notes=[
                "Potential impact points are bounded static reverse-reachability candidates, "
                "not guaranteed runtime effects or complete dynamic impact sets."
            ],
        )

    return gold


def migrate_gold_to_v2(
    qtype: str,
    gold: dict[str, Any],
    *,
    root_qualname: str,
    graph: CallGraph | None = None,
    ast_map: dict[str, FunctionAST] | None = None,
    subgraph_nodes: list[str] | None = None,
) -> dict[str, Any]:
    if is_gold_v2(gold):
        return gold
    root_short = root_qualname.rsplit(".", 1)[-1]
    if qtype == "execution_flow" and graph is not None and ast_map is not None and root_qualname in graph.nodes:
        return _build_execution_flow_v2(
            graph,
            root_qualname,
            ast_map.get(root_qualname),
            subgraph_nodes=subgraph_nodes,
        )
    return convert_legacy_gold_to_v2(qtype, gold, root_short=root_short)


def legacy_gold_view(qtype: str, gold: dict[str, Any]) -> dict[str, Any]:
    """Project gold-v2 back to the legacy field view for downstream reuse.

    Active compatibility projection (not deprecated): the structured evaluator
    and validators read gold through this view, so the field names it returns
    (e.g. ``path_nodes``, ``sample_path``, ``static_direct_callers``) are the
    canonical names predictions are scored against. ``legacy`` refers to the
    projected field vocabulary, not to this function's status.
    """
    if not is_gold_v2(gold):
        return gold

    hard = gold.get("hard_constraints", {}) or {}
    soft = gold.get("soft_constraints", {}) or {}
    labels = hard.get("required_labels", {}) or {}
    required_entities = hard.get("required_entities", []) or []
    acceptable_paths = soft.get("acceptable_paths", []) or []
    acceptable_sets = soft.get("acceptable_sets", []) or []
    optional_entities = soft.get("optional_entities", []) or []
    notes = gold.get("notes", []) or []

    if qtype == "execution_flow":
        if is_execution_flow_gold_v21(gold):
            shared_prefix = gold.get("shared_prefix", []) or []
            branch_workflows = gold.get("branch_workflows", []) or []
            path = branch_workflows[0] if branch_workflows else shared_prefix
            return {
                "path_type": "representative_workflow",
                "path_condition": "",
                "path_nodes": path,
                "coverage_note": notes[0] if notes else _workflow_note()[0],
                "gold_entities": path[1:] if len(path) > 1 else [],
                "gold_paths": branch_workflows or ([shared_prefix] if shared_prefix else []),
                "shared_prefix": shared_prefix,
                "branch_workflows": branch_workflows,
                "key_gates": gold.get("key_gates", []) or [],
                "optional_steps": gold.get("optional_steps", []) or [],
                "singleton": bool(gold.get("singleton", False)),
            }
        path = acceptable_paths[-1] if acceptable_paths else required_entities
        return {
            "path_type": "representative_workflow",
            "path_condition": "",
            "path_nodes": path,
            "coverage_note": notes[0] if notes else "This is one representative workflow, not the unique full execution flow.",
            "gold_entities": path[1:] if len(path) > 1 else [],
            "gold_paths": acceptable_paths or ([path] if path else []),
        }

    if qtype == "conditional_path":
        path = acceptable_paths[0] if acceptable_paths else required_entities
        return {
            "condition": labels.get("condition", ""),
            "branch_kind": labels.get("branch_kind", ""),
            "branch_span": (gold.get("scope_constraints", {}) or {}).get("evidence_span", {}) or {},
            "path_nodes": path,
            "excluded_branches": [],
        }

    if qtype == "error_handling_path":
        path = acceptable_paths[0] if acceptable_paths else required_entities
        return {
            "error_trigger": labels.get("error_trigger", ""),
            "error_path": path,
            "error_action": labels.get("error_action", ""),
            "fallback_action": None,
            "evidence_span": (gold.get("scope_constraints", {}) or {}).get("evidence_span", {}) or {},
        }

    if qtype == "dependency_trace":
        targets = acceptable_sets[0] if acceptable_sets else required_entities
        sample_path = acceptable_paths[0] if acceptable_paths else []
        return {
            "target_kind": labels.get("target_kind", ""),
            "target_entities": targets,
            "sample_path": sample_path,
        }

    if qtype == "impact_scope":
        impacted = acceptable_sets[0] if acceptable_sets else required_entities + optional_entities
        return {
            "static_direct_callers": required_entities,
            "potentially_affected_callers": impacted,
        }

    return gold
