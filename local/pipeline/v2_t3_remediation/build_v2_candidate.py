#!/usr/bin/env python3
"""Build the benchmark-v2 candidate (T3.6) from the canonical v1 benchmark.

Deterministic, replayable application of the author-approved
T3_FINAL_ACTION_MATRIX (289 items) over the 632-item v1 benchmark.  The 343
items not covered by the matrix are carried over untouched
(KEEP_UNAUDITED_V1).  No model is run; no v1 artifact is modified.

Usage:  python3 -m fcg_benchmark.v2_t3_remediation.build_v2_candidate
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone

from . import repair_patches as P
from .js_newcall_recovery import recover_all

# --------------------------------------------------------------------------
V1_PATH = ("/home/lxh/fcg_benchmark (8)/fcg_benchmark/experiments/multilingual_en/"
           "benchmarks_with_explanations_v21_balanced_clean_plus_dedup/"
           "all_multilingual_v21_balanced_clean_en_with_explanations.jsonl")
V1_SHA256 = "df0a53b496c8608299a2da323201675447eab7a3c47e1540a1322c08936c9a94"
AUDITS = "/home/lxh/fcg-audits"
MATRIX_CSV = f"{AUDITS}/t3-decision-consolidated/T3_FINAL_ACTION_MATRIX.csv"
IMPACT_AUDIT_CSV = f"{AUDITS}/t3-expanded/T3_IMPACT_SCOPE_AUDIT.csv"
UNVERIF_AUDIT_CSV = f"{AUDITS}/t3-expanded/T3_UNVERIFIABLE_ENTITY_AUDIT.csv"
OUT_DIR = ("fcg_benchmark/experiments/multilingual_en/"
           "benchmarks_with_explanations_v22_t3_remediated")
CANDIDATE_NAME = "all_multilingual_v22_t3_remediated_en_with_explanations.jsonl"

FLOOR = 590  # T3.6 section 7 procedural floor; rescinded by author override T3.6A (2026-07-30)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def text_hash(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()[:16]


def gold_hash(gold: dict) -> str:
    payload = json.dumps(gold, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def load_v1():
    raw = open(V1_PATH, "rb").read()
    assert sha256_bytes(raw) == V1_SHA256, "v1 benchmark SHA mismatch — abort"
    return [json.loads(l) for l in raw.decode().splitlines() if l.strip()], raw


def load_matrix():
    return {r["item_id"]: r for r in csv.DictReader(open(MATRIX_CSV))}


def load_impact_audit():
    return {r["item_id"]: r for r in csv.DictReader(open(IMPACT_AUDIT_CSV))}


def load_unverif_audit():
    return {r["item_id"]: r for r in csv.DictReader(open(UNVERIF_AUDIT_CSV))}


def split_pipe(s: str):
    return [x.strip() for x in s.split("|") if x.strip()]


def clean_entity(x: str) -> str:
    """Strip audit annotations like ' (impacted)' or ' (3 RESOLVED call sites)'."""
    return x.split(" (")[0].strip()


# --------------------------------------------------------------------------
# text surgery helpers
# --------------------------------------------------------------------------

def _join(lst):
    return ", ".join(sorted(lst))


def replace_enumeration(text: str, old_list, new_list, counter: dict):
    """Replace the sorted enumeration of old_list by new_list inside text."""
    old_s, new_s = _join(old_list), _join(new_list)
    if old_s == new_s:
        return text
    if old_s in text:
        counter["enumeration_replaced"] += 1
        return text.replace(old_s, new_s)
    counter["enumeration_not_found"] += 1
    return text


def _seg(req, acc):
    """v1 impact enumeration order: sorted(direct) ++ sorted(impacted - direct)."""
    return ", ".join(sorted(req) + sorted(set(acc) - set(req)))


def regenerate_impact_texts(item: dict, new_req, new_acc):
    """Regenerate natural_answer / explanation / claims for impact_scope items.

    All 80 v1 impact items share one exact EN template (verified); the zh
    natural_answer embeds the same two segmented enumerations.  Regeneration
    avoids the substring-surgery ambiguity when old direct == old impacted.
    """
    gold = item["gold_answer"]
    root = gold.get("root_entity") or item["root_function"].split(".")[-1]
    d, i = ", ".join(sorted(new_req)), _seg(new_req, new_acc)
    item["natural_answer"] = (
        f"In the bounded static reverse call graph, the direct callers of {root} are {d}; "
        f"the conservative potential impact points are {i}.")
    item["explanation_reference_en"] = (
        f"In the bounded static reverse call graph, the static direct callers are {d}. "
        f"The conservative potential impact points are {i}. These relations describe static "
        "reverse reachability only; they are not guaranteed runtime effects or complete "
        "dynamic impact sets.")
    item["claims_en"] = [
        {"text": f"The static direct callers are {d}.", "claim_type": "scope"},
        {"text": f"The conservative potential impact points are {i}.", "claim_type": "scope"},
        {"text": "These items are bounded static reverse-reachability candidates, not guaranteed runtime effects or complete dynamic impact sets.", "claim_type": "scope"},
    ]
    zh = item.get("natural_answer_zh")
    if zh:
        m = re.match(r"^在当前静态调用图中,(.+)的直接调用者:", zh)
        zroot = m.group(1) if m else root
        item["natural_answer_zh"] = (
            f"在当前静态调用图中,{zroot}的直接调用者:{d};潜在影响范围:{i}。")


def update_path_texts(item: dict, old_path, new_path, counter: dict):
    """Replace the path string in natural/explanation/claims (EN ' -> ', zh ' → ')."""
    for arrow, fields in ((" -> ", ("natural_answer", "explanation_reference_en")), (" → ", ("natural_answer_zh",))):
        old_s, new_s = arrow.join(old_path), arrow.join(new_path)
        if old_s == new_s:
            continue
        hit = False
        for f in fields:
            if old_s in item.get(f, ""):
                item[f] = item[f].replace(old_s, new_s)
                hit = True
        if arrow == " -> ":
            for c in item.get("claims_en", []):
                if old_s in c.get("text", ""):
                    c["text"] = c["text"].replace(old_s, new_s)
                    hit = True
        counter["path_text_replaced" if hit else "path_text_not_found"] += 1


def replace_substring(item: dict, old: str, new: str, counter: dict,
                      fields=("question", "question_en", "question_zh")):
    for f in fields:
        if f in item and isinstance(item[f], str) and old in item[f]:
            item[f] = item[f].replace(old, new)
            counter[f"{f}_replaced"] = counter.get(f"{f}_replaced", 0) + 1


def provenance(item: dict, action: str, reason: str, evidence: str, old_gold_h: str, old_q_h: str):
    item["v2_provenance"] = {
        "action": action,
        "reason": reason,
        "evidence": evidence,
        "old_structure_gold_hash": old_gold_h,
        "old_question_hash": old_q_h,
        "candidate": "benchmark-v2-candidate (not-final, not-yet-rerun, not-yet-reevaluated)",
    }


def impact_sets(gold: dict):
    hard = gold["hard_constraints"]
    soft = gold["soft_constraints"]
    req = list(hard.get("required_entities", []))
    acc = list((soft.get("acceptable_sets") or [[]])[0])
    return req, acc


def set_impact_sets(gold: dict, req, acc):
    gold["hard_constraints"]["required_entities"] = sorted(req)
    if gold["soft_constraints"].get("acceptable_sets"):
        gold["soft_constraints"]["acceptable_sets"][0] = sorted(acc)
    gold["notes"] = (gold.get("notes") or []) + [
        "v2 (T3 remediation): caller sets rebuilt under the fixed impact generator "
        "(no 8/12 caps, comprehension callers mapped to enclosing function, "
        "destructor-clique pseudo-edges filtered, template/qualified-name "
        "normalization repaired; dynamic/ambiguous callers flagged in v2_provenance)."
    ]


# --------------------------------------------------------------------------
# per-action appliers — each returns (item_or_None, record)
# --------------------------------------------------------------------------

def apply_impact(item, m, audit, counter):
    iid = item["item_id"]
    gold = item["gold_answer"]
    req, acc = impact_sets(gold)
    old_req, old_acc = list(req), list(acc)
    a = audit.get(iid, {})
    add_direct, add_impacted, drop = [], [], []
    note = ""
    if iid in P.IMPACT_DECLIQUE:
        drop = [clean_entity(x) for x in split_pipe(a.get("spurious_callers", ""))]
        note = "virtual-destructor sibling clique filtered (7 spurious sibling destructors dropped)"
    elif iid in P.IMPACT_ADD_BOTH:
        add_direct = [clean_entity(x) for x in split_pipe(a.get("missing_true_callers", ""))]
        add_impacted = list(add_direct)
        note = "truncation caps removed; missing source-verified callers re-added (direct+impacted)"
    elif iid in P.IMPACT_ADD_IMPACTED_ONLY:
        spec = P.IMPACT_ADD_IMPACTED_ONLY[iid]
        add_impacted = spec if spec is not None else [clean_entity(x) for x in split_pipe(a.get("missing_true_callers", ""))]
        note = "impacted-cap removed; missing same-depth/depth-2 callers re-added (impacted only)"
    elif iid in P.IMPACT_DROP:
        drop = P.IMPACT_DROP[iid]
        note = "spurious extern-stub caller removed"
    elif iid in P.IMPACT_ADD_ENCLOSING:
        add_direct = P.IMPACT_ADD_ENCLOSING[iid]
        add_impacted = list(add_direct)
        note = "unscorable anon-closure caller replaced by its enclosing nameable function"
    elif iid in P.IMPACT_NO_CHANGE:
        note = P.IMPACT_NO_CHANGE[iid]
        rec_note_only = True
    else:
        raise AssertionError(f"no impact repair rule for {iid}")
    if iid in P.IMPACT_NO_CHANGE:
        new_req, new_acc = old_req, old_acc
    else:
        new_req = sorted((set(old_req) | set(add_direct)) - set(drop))
        new_acc = sorted((set(old_acc) | set(add_impacted) | set(add_direct)) - set(drop))
    ambiguous = split_pipe(a.get("ambiguous_dynamic_callers", ""))
    evidence = (f"T3_IMPACT_SCOPE_AUDIT.csv row {iid}; missing+{len(add_direct)}/"
                f"impacted+{len(add_impacted)}/drop-{len(drop)}")
    if not new_req:
        return None, dict(item_id=iid, action="GATE_EXCLUDE", disposition="DROP_EXCLUDE",
                          reason="G1/G3: declique leaves an empty scored caller set; static-object "
                                 "destruction has no explicit call site — no valid impact item",
                          evidence=evidence)
    old_gh, old_qh = item["structure_gold_hash"], text_hash(item["question"])
    set_impact_sets(gold, new_req, new_acc)
    regenerate_impact_texts(item, new_req, new_acc)
    item["structure_gold_hash"] = gold_hash(gold)
    provenance(item, "REBUILD_GOLD", note + ("; ambiguous/dynamic callers flagged: " + "; ".join(ambiguous[:3]) if ambiguous else ""),
               evidence, old_gh, old_qh)
    return item, dict(item_id=iid, action="REBUILD_GOLD", disposition="REUSE_REEVALUATE",
                      reason=note, evidence=evidence,
                      old_gold_hash=old_gh, new_gold_hash=item["structure_gold_hash"],
                      old_required="|".join(old_req), new_required="|".join(new_req),
                      old_acceptable="|".join(old_acc), new_acceptable="|".join(new_acc))


def apply_expand(item, counter):
    iid = item["item_id"]
    spec = P.EXPAND_ACCEPTABLE[iid]
    gold = item["gold_answer"]
    req, acc = impact_sets(gold)
    old_req, old_acc = list(req), list(acc)
    new_req = sorted(set(old_req) | set(spec["add_direct"]))
    new_acc = sorted(set(old_acc) | set(spec["add_impacted"]) | set(spec["add_direct"]))
    old_gh, old_qh = item["structure_gold_hash"], text_hash(item["question"])
    set_impact_sets(gold, new_req, new_acc)
    regenerate_impact_texts(item, new_req, new_acc)
    item["structure_gold_hash"] = gold_hash(gold)
    provenance(item, "EXPAND_ACCEPTABLE", spec["reason"], "T3_IMPACT_SCOPE_AUDIT.csv + adjudication packet", old_gh, old_qh)
    return item, dict(item_id=iid, action="EXPAND_ACCEPTABLE", disposition="REUSE_REEVALUATE",
                      reason=spec["reason"], old_gold_hash=old_gh, new_gold_hash=item["structure_gold_hash"],
                      old_required="|".join(old_req), new_required="|".join(new_req),
                      old_acceptable="|".join(old_acc), new_acceptable="|".join(new_acc))


def apply_unverif_resolve(item, audit, counter):
    """C-UNVERIF-UNRESOLVED: per-item verdict from implementation-time
    frozen-source verification (KEEP_ONLY_SOURCE_VERIFIABLE_ENTITIES)."""
    iid = item["item_id"]
    verdict, evidence = P.UNVERIF_VERDICTS[iid]
    gold = item["gold_answer"]
    old_gh, old_qh = item["structure_gold_hash"], text_hash(item["question"])
    if verdict == "KEEP_SOURCE_VERIFIED":
        provenance(item, "KEEP", f"flagged entity verified in frozen source: {evidence}",
                   f"T3_UNVERIFIABLE_ENTITY_AUDIT.csv row {iid} + frozen source read", old_gh, old_qh)
        return item, dict(item_id=iid, action="KEEP", disposition="REUSE_UNCHANGED",
                          reason=f"source-verified at implementation: {evidence}")
    if verdict == "KEEP_EXTERNAL":
        item["explanation_reference_en"] += " " + P.EXTERNAL_ENTITY_NOTE_GENERIC
        provenance(item, "FIX_EXPLANATION", f"external/stdlib entity documented: {evidence}",
                   f"T3_UNVERIFIABLE_ENTITY_AUDIT.csv row {iid}", old_gh, old_qh)
        return item, dict(item_id=iid, action="FIX_EXPLANATION", disposition="REUSE_REEVALUATE",
                          reason=f"external/stdlib entity documented: {evidence}")
    if verdict == "GATE_G2":
        return None, dict(item_id=iid, action="GATE_EXCLUDE", disposition="DROP_EXCLUDE",
                          reason=f"G2: {evidence}",
                          evidence=f"T3_UNVERIFIABLE_ENTITY_AUDIT.csv row {iid} + frozen source reads")
    # DROP_ENTITY: remove from scored sets (REBUILD_GOLD)
    flags = audit[iid]["entity_flags"]
    drops = [f.split("(")[0] for f in flags.split("|") if "unverifiable" in f]
    hard = gold["hard_constraints"]
    soft = gold["soft_constraints"]
    old_req = list(hard.get("required_entities", []))
    new_req = [e for e in old_req if e not in drops]
    if len(new_req) < 2:
        return None, dict(item_id=iid, action="GATE_EXCLUDE", disposition="DROP_EXCLUDE",
                          reason=f"G1/G3: dropping unverifiable entities {drops} leaves <2 required entities",
                          evidence=f"T3_UNVERIFIABLE_ENTITY_AUDIT.csv row {iid}")
    old_paths = [list(p) for p in soft.get("acceptable_paths", [])]
    for path in soft.get("acceptable_paths", []):
        path[:] = [e for e in path if e not in drops]
    for s in soft.get("acceptable_sets", []):
        s[:] = sorted(e for e in s if e not in drops)
    hard["required_entities"] = new_req
    for op, np_ in zip(old_paths, soft.get("acceptable_paths", [])):
        update_path_texts(item, op, np_, counter)
    gold["notes"] = (gold.get("notes") or []) + [
        f"v2 (T3 remediation): unverifiable required entit{'ies' if len(drops)>1 else 'y'} "
        f"{', '.join(drops)} removed from the scored set (KEEP_ONLY_SOURCE_VERIFIABLE_ENTITIES)."
    ]
    item["structure_gold_hash"] = gold_hash(gold)
    provenance(item, "REBUILD_GOLD", f"dropped unverifiable scored entities: {drops}; {evidence}",
               f"T3_UNVERIFIABLE_ENTITY_AUDIT.csv row {iid}", old_gh, old_qh)
    return item, dict(item_id=iid, action="REBUILD_GOLD", disposition="REUSE_REEVALUATE",
                      reason=f"drop unverifiable entities {drops}: {evidence}", old_gold_hash=old_gh,
                      new_gold_hash=item["structure_gold_hash"],
                      old_required="|".join(old_req), new_required="|".join(new_req))


def apply_condition(item, counter):
    iid = item["item_id"]
    new_cond, src = P.CONDITION_REPAIRS[iid]
    gold = item["gold_answer"]
    labels = gold["hard_constraints"]["required_labels"]
    old_cond = labels.get("condition", "")
    old_gh, old_qh = item["structure_gold_hash"], text_hash(item["question"])
    normalization_only = old_cond == new_cond
    if not normalization_only:
        labels["condition"] = new_cond
        # answer-side texts quote the old condition verbatim (question texts
        # intentionally stay unchanged: gold-only repair, re-eval only)
        for f in ("natural_answer", "explanation_reference_en", "natural_answer_zh"):
            if old_cond in item.get(f, ""):
                item[f] = item[f].replace(old_cond, new_cond)
                counter["condition_text_replaced"] += 1
        for c in item.get("claims_en", []):
            if old_cond in c.get("text", ""):
                c["text"] = c["text"].replace(old_cond, new_cond)
    item["structure_gold_hash"] = gold_hash(gold)
    note = ("condition label verified against frozen source (whitespace-normalized identical; "
            "normalization-only confirmation)" if normalization_only
            else f"condition label repaired to source-verbatim guard ({src}); question text unchanged")
    provenance(item, "REBUILD_GOLD", note, f"frozen source read: {src}", old_gh, old_qh)
    return item, dict(item_id=iid, action="REBUILD_GOLD", disposition="REUSE_REEVALUATE",
                      reason=note, old_gold_hash=old_gh, new_gold_hash=item["structure_gold_hash"],
                      old_label=old_cond, new_label=new_cond)


def apply_fix_gold_0022(item, counter):
    spec = P.FIX_GOLD_ERROR_0022
    gold = item["gold_answer"]
    old_gh, old_qh = item["structure_gold_hash"], text_hash(item["question"])
    old_path = list(gold["soft_constraints"]["acceptable_paths"][0])
    hard = gold["hard_constraints"]
    hard["required_entities"] = list(spec["path"])
    hard["required_labels"]["error_trigger"] = spec["trigger"]
    hard["required_labels"]["error_action"] = spec["error_action"]
    gold["soft_constraints"]["acceptable_paths"][0] = list(spec["path"])
    item["question"] = spec["question"]
    if "question_en" in item:
        item["question_en"] = spec["question"]
    item["natural_answer"] = (
        f"When {spec['trigger']} is triggered, AgentExecutor follows the local failure-handling path "
        + " -> ".join(spec["path"]) + ".")
    item["explanation_reference_en"] = (
        "Within evidence span 302-350, the structured result records a local failure-handling path "
        f"triggered by {spec['trigger']}: " + " -> ".join(spec["path"]) +
        f". The recorded error_action is {spec['error_action']}. This explanation only covers the "
        "current evidence span and does not represent the complete runtime error behavior of the function.")
    item["claims_en"] = [
        {"text": f"The recorded failure trigger is {spec['trigger']}.", "claim_type": "origin"},
        {"text": "The recorded local failure-handling path is " + " -> ".join(spec["path"]) + ".", "claim_type": "path"},
        {"text": f"The recorded error_action is {spec['error_action']}.", "claim_type": "classification"},
        {"text": "This explanation only covers evidence span 302-350 and does not represent complete runtime error behavior.", "claim_type": "scope"},
    ]
    gold["notes"] = (gold.get("notes") or []) + [
        "v2 (T3 remediation): builtin 'callable' dropped from the scored path; real raising call "
        "'plan' (agent.py:1317) recorded as required error call and trigger; question reworded "
        "accordingly (rerun required). Dedup twin error_v21_0023 removed; no re-duplication."
    ]
    item["structure_gold_hash"] = gold_hash(gold)
    provenance(item, "FIX_GOLD", "drop builtin 'callable'; repair required error call, trigger and explanation; trigger is question-embedded -> RERUN",
               spec["evidence"], old_gh, old_qh)
    return item, dict(item_id=item["item_id"], action="REBUILD_GOLD", disposition="RERUN_ALL_4",
                      reason="FIX_GOLD with question-embedded trigger repair (rerun)",
                      old_gold_hash=old_gh, new_gold_hash=item["structure_gold_hash"],
                      old_question_hash=old_qh, new_question_hash=text_hash(item["question"]),
                      old_path="|".join(old_path), new_path="|".join(spec["path"]))


def apply_rewrite_question(item, counter):
    iid = item["item_id"]
    fq = P.SHARED_FQ_ROOT[iid]
    short = item["root_function"].split(".")[-1]
    old_qh = text_hash(item["question"])
    old_q = item["question"]
    replace_substring(item, short, fq, counter)
    assert item["question"] != old_q, f"question rewrite had no effect for {iid}"
    gold = item["gold_answer"]
    gold["notes"] = (gold.get("notes") or []) + [
        f"v2 (T3 remediation): question text now names the fully qualified root {fq}; "
        "shared structure gold across distinct bodies is legitimate equivalence (author decision REWRITE_QUESTION_KEEP_ALL_7)."
    ]
    old_gh = item["structure_gold_hash"]
    item["structure_gold_hash"] = gold_hash(gold)
    provenance(item, "REWRITE_QUESTION", f"fully qualified root {fq} added to question text",
               "T3_DUPLICATE_GROUPS.csv + frozen source def lines", old_gh, old_qh)
    return item, dict(item_id=iid, action="REWRITE_QUESTION", disposition="RERUN_ALL_4",
                      reason=f"FQ root {fq} in question", old_question_hash=old_qh,
                      new_question_hash=text_hash(item["question"]),
                      old_gold_hash=old_gh, new_gold_hash=item["structure_gold_hash"])


def _reword_kind_texts(item, old_phrase, new_phrase, counter):
    for f in ("natural_answer", "explanation_reference_en"):
        if old_phrase in item.get(f, ""):
            item[f] = item[f].replace(old_phrase, new_phrase)
            counter["kind_text_replaced"] += 1
    for c in item.get("claims_en", []):
        if old_phrase in c.get("text", ""):
            c["text"] = c["text"].replace(old_phrase, new_phrase)


def apply_go_rescope(item, counter):
    iid = item["item_id"]
    gold = item["gold_answer"]
    old_gh, old_qh = item["structure_gold_hash"], text_hash(item["question"])
    root_short = gold.get("root_entity") or item["root_function"].split(".")[-1]
    targets = list(gold["hard_constraints"]["required_entities"])
    if iid == "ollama_go_0010":
        spec = P.REBUILD_GO_0010
        gold["hard_constraints"]["required_entities"] = sorted(spec["targets"])
        gold["hard_constraints"]["required_labels"]["target_kind"] = spec["target_kind"]
        gold["soft_constraints"]["acceptable_sets"][0] = sorted(spec["targets"])
        gold["soft_constraints"]["acceptable_paths"][0] = list(spec["sample_path"])
        item["question"] = spec["question"]
        item["question_zh"] = "PullModel 中实例化了哪些结构体复合字面量?"
        item["natural_answer_zh"] = (
            f"PullModel 中实例化的结构体复合字面量有:{_join(spec['targets'])}。"
            "一条代表性路径:PullModel → downloadOpts。")
        for n in spec["add_subgraph"]:
            if n not in item["subgraph_nodes"]:
                item["subgraph_nodes"].append(n)
        kind_phrase = "struct composite literals"
        reason = "rebuilt per true Go semantics with the real composite literals (api.ProgressResponse, downloadOpts)"
        evidence = spec["evidence"]
    else:
        # rescope: gold targets are audit-verified genuine calls; only the
        # question/label framing changes (class_instantiation -> helper calls)
        gold["hard_constraints"]["required_labels"]["target_kind"] = P.GO_RESCOPE_KIND
        item["question"] = P.GO_RESCOPE_QUESTION.format(root=root_short)
        # declared-sample verifiability: add scored targets missing from subgraph_nodes
        missing = [t for t in targets if t not in item["subgraph_nodes"]
                   and not any(n.endswith("." + t) for n in item["subgraph_nodes"])]
        item["subgraph_nodes"].extend(missing)
        for f in ("question_zh", "natural_answer_zh"):
            if f in item and "类实例化" in item[f]:
                item[f] = item[f].replace("类实例化", "辅助函数或方法调用")
        kind_phrase = "helper functions and method calls"
        reason = "rescoped from class_instantiation to helper-function/method-call reachability; audit-verified genuine callees added to declared subgraph"
        evidence = "T3_CLASS_INSTANTIATION_AUDIT.csv row " + iid
    if "question_en" in item:
        item["question_en"] = item["question"]
    _reword_kind_texts(item, "class_instantiation", kind_phrase, counter)
    if iid == "ollama_go_0010":
        item["natural_answer"] = (
            f"From PullModel, the instantiated struct composite literals are {_join(spec['targets'])}. "
            "One representative static path is PullModel -> downloadOpts.")
        item["explanation_reference_en"] = (
            f"The structured result indicates that the following struct composite literals are "
            f"instantiated within PullModel: {_join(spec['targets'])}. One representative static "
            "reachability path is PullModel -> downloadOpts. This describes static FCG reachability "
            "and does not guarantee runtime execution.")
        item["claims_en"] = [
            {"text": f"The instantiated struct composite literals include {_join(spec['targets'])}.", "claim_type": "scope"},
            {"text": "One representative static reachability path is PullModel -> downloadOpts.", "claim_type": "path"},
            {"text": "This item describes static FCG reachability rather than guaranteed runtime execution.", "claim_type": "scope"},
        ]
    item["structure_gold_hash"] = gold_hash(gold)
    provenance(item, "REBUILD_ITEM", reason, evidence, old_gh, old_qh)
    return item, dict(item_id=iid, action="REBUILD_ITEM", disposition="RERUN_ALL_4",
                      reason=reason, old_gold_hash=old_gh, new_gold_hash=item["structure_gold_hash"],
                      old_question_hash=old_qh, new_question_hash=text_hash(item["question"]))


def apply_cpp_0088(item, counter):
    spec = P.REBUILD_CPP_0088
    gold = item["gold_answer"]
    old_gh, old_qh = item["structure_gold_hash"], text_hash(item["question"])
    old_cond = gold["hard_constraints"]["required_labels"].get("condition", "")
    old_path = list(gold["soft_constraints"]["acceptable_paths"][0])
    hard = gold["hard_constraints"]
    hard["required_entities"] = list(spec["path"])
    hard["required_labels"]["condition"] = spec["condition"]
    hard["required_labels"]["branch_kind"] = spec["branch_kind"]
    gold["soft_constraints"]["acceptable_paths"][0] = list(spec["path"])
    gold["soft_constraints"]["optional_entities"] = list(spec["optional"])
    item["question"] = spec["question"]
    # zh stays consistent with the rebuilt item
    if "question_zh" in item and old_cond in item["question_zh"]:
        item["question_zh"] = item["question_zh"].replace(old_cond, spec["condition"])
    if "natural_answer_zh" in item:
        zh = item["natural_answer_zh"]
        if old_cond in zh:
            zh = zh.replace(old_cond, spec["condition"])
        zh = zh.replace("if_true 分支", "loop_body 分支")
        old_zh_path, new_zh_path = " → ".join(old_path), " → ".join(spec["path"])
        if old_zh_path in zh:
            zh = zh.replace(old_zh_path, new_zh_path)
        item["natural_answer_zh"] = zh
    path_s = " -> ".join(spec["path"])
    item["natural_answer"] = (
        f"The extracted loop_body branch around condition {spec['condition']} records the path {path_s}.")
    item["explanation_reference_en"] = (
        f"The structured result records the extracted loop_body branch for the condition "
        f"{spec['condition']}, with branch span 8-53 (function-relative). Within this extracted "
        f"branch, the recorded path is {path_s}. This explanation only covers the extracted branch "
        "and does not imply that other branches behave the same way.")
    item["claims_en"] = [
        {"text": f"The branch condition is {spec['condition']}.", "claim_type": "comparison"},
        {"text": "The extracted branch kind is loop_body.", "claim_type": "classification"},
        {"text": f"The recorded branch path is {path_s}.", "claim_type": "path"},
        {"text": "This explanation only covers the extracted branch at function-relative span 8-53.", "claim_type": "scope"},
    ]
    gold["notes"] = (gold.get("notes") or []) + [
        "v2 (T3 remediation): rebuilt from the spliced guard/body/for-header condition. Real guard "
        "'end > tiles' (sgemm.cpp:865) has an empty if-true branch; item re-scoped to the succeeding "
        "tile loop (L867-910) with RESOLVED callees only. NOT_IN_EXTRACTION intrinsics "
        "(_mm_set_epi64x, _mm_mul_ps, _mm256_permute2f128_ps, _mm256_shuffle_ps) removed from the "
        "forced path; updot/_mm256_sign_epi8 kept optional."
    ]
    item["structure_gold_hash"] = gold_hash(gold)
    provenance(item, "REBUILD_ITEM", "rebuilt legal condition/branch/path; no spliced condition, no forced intrinsics",
               spec["evidence"], old_gh, old_qh)
    return item, dict(item_id=item["item_id"], action="REBUILD_ITEM", disposition="RERUN_ALL_4",
                      reason="condition splice removed; re-scoped to tile loop body (new branch_kind loop_body, documented)",
                      old_gold_hash=old_gh, new_gold_hash=item["structure_gold_hash"],
                      old_question_hash=old_qh, new_question_hash=text_hash(item["question"]))


def apply_ts_0090(item, counter):
    spec = P.REBUILD_TS_0090
    gold = item["gold_answer"]
    old_gh, old_qh = item["structure_gold_hash"], text_hash(item["question"])
    old_cond = gold["hard_constraints"]["required_labels"].get("condition", "")
    old_path = list(gold["soft_constraints"]["acceptable_paths"][0])
    hard = gold["hard_constraints"]
    hard["required_entities"] = list(spec["path"])
    hard["required_labels"]["condition"] = spec["condition"]
    hard["required_labels"]["branch_kind"] = spec["branch_kind"]
    gold["root_entity"] = spec["root_entity"]
    gold["soft_constraints"]["acceptable_paths"][0] = list(spec["path"])
    gold["scope_constraints"]["evidence_span"] = dict(spec["evidence_span"])
    item["question"] = spec["question"]
    # zh consistency: new guard, de-mangled root, rebuilt path
    if "question_zh" in item:
        qzh = item["question_zh"]
        if old_cond in qzh:
            qzh = qzh.replace(old_cond, spec["condition"])
        qzh = qzh.replace("ts:464>", spec["root_entity"])
        item["question_zh"] = qzh
    if "natural_answer_zh" in item:
        zh = item["natural_answer_zh"]
        if old_cond in zh:
            zh = zh.replace(old_cond, spec["condition"])
        zh = zh.replace("ts:464>", spec["root_entity"])
        old_zh_path, new_zh_path = " → ".join(old_path), " → ".join(spec["path"])
        if old_zh_path in zh:
            zh = zh.replace(old_zh_path, new_zh_path)
        item["natural_answer_zh"] = zh
    path_s = " -> ".join(spec["path"])
    item["natural_answer"] = (
        f"The extracted if_true branch around condition {spec['condition']} records the path {path_s}.")
    item["explanation_reference_en"] = (
        f"The structured result records the extracted if_true branch (token-expiry retry branch) for "
        f"the condition {spec['condition']}, with function-relative branch span 24-49. Within this "
        f"extracted branch, the recorded path is {path_s}. This explanation only covers the extracted "
        "branch and does not imply that other branches behave the same way.")
    item["claims_en"] = [
        {"text": f"The branch condition is {spec['condition']}.", "claim_type": "comparison"},
        {"text": "The extracted branch kind is if_true.", "claim_type": "classification"},
        {"text": f"The recorded branch path is {path_s}.", "claim_type": "path"},
        {"text": "This explanation only covers the extracted retry branch at function-relative span 24-49.", "claim_type": "scope"},
    ]
    gold["notes"] = (gold.get("notes") or []) + [
        "v2 (T3 remediation): ~900-char condition dump replaced by the real retry-branch guard "
        "(gateway.ts:490, verified in frozen source); root de-mangled from 'ts:464>' to "
        "<anon@gateway.ts:464>; 'includes' builtin pseudo node removed from the forced path; gold "
        "path is the retry branch in exact source order (adjudication errata)."
    ]
    item["structure_gold_hash"] = gold_hash(gold)
    provenance(item, "REBUILD_ITEM", "real guard restored; dump condition and builtin pseudo node removed; root de-mangled",
               spec["evidence"], old_gh, old_qh)
    return item, dict(item_id=item["item_id"], action="REBUILD_ITEM", disposition="RERUN_ALL_4",
                      reason="retry-branch guard restored; root de-mangled; pseudo node removed",
                      old_gold_hash=old_gh, new_gold_hash=item["structure_gold_hash"],
                      old_question_hash=old_qh, new_question_hash=text_hash(item["question"]))


def apply_js_rebuild(item, rec, counter):
    iid = item["item_id"]
    gold = item["gold_answer"]
    old_gh, old_qh = item["structure_gold_hash"], text_hash(item["question"])
    old_targets = list(gold["hard_constraints"]["required_entities"])
    old_path = list((gold["soft_constraints"].get("acceptable_paths") or [[]])[0])
    targets = sorted(rec["targets"])
    root = gold.get("root_entity") or rec["root"]
    first = min(rec["paths"].items(), key=lambda kv: (rec["targets"][kv[0]], kv[0]))[0]
    sample_path = list(rec["paths"][first])
    gold["hard_constraints"]["required_entities"] = targets
    gold["soft_constraints"]["acceptable_sets"][0] = targets
    gold["soft_constraints"]["acceptable_paths"][0] = sample_path
    item["subgraph_nodes"] = sorted({n for p in rec["paths"].values() for n in p})
    t_s, path_s = _join(targets), " -> ".join(sample_path)
    # zh natural answer embeds the same target enumeration and arrow path
    if "natural_answer_zh" in item:
        zh = item["natural_answer_zh"]
        old_t_zh, new_t_zh = _join(old_targets), t_s
        if old_t_zh in zh:
            zh = zh.replace(old_t_zh, new_t_zh)
        old_p_zh, new_p_zh = " → ".join(old_path), " → ".join(sample_path)
        if old_p_zh in zh:
            zh = zh.replace(old_p_zh, new_p_zh)
        item["natural_answer_zh"] = zh
    item["natural_answer"] = (
        f"From {root}, the statically reachable class_instantiation are {t_s}. "
        f"One representative static path is {path_s}.")
    item["explanation_reference_en"] = (
        f"The structured result indicates that the following class_instantiation are statically "
        f"reachable in the FCG: {t_s}. One representative static reachability path is {path_s}. "
        "This describes static FCG reachability and does not guarantee runtime execution.")
    item["claims_en"] = [
        {"text": f"The statically reachable class_instantiation include {t_s}.", "claim_type": "scope"},
        {"text": f"One representative static reachability path is {path_s}.", "claim_type": "path"},
        {"text": "This item describes static FCG reachability rather than guaranteed runtime execution.", "claim_type": "scope"},
    ]
    gold["notes"] = (gold.get("notes") or []) + [
        "v2 (T3 remediation): scored target set rebuilt from real NEW_CALL edges in the frozen "
        "extraction with file-anchored traversal (v1 name-pooling removed); hop-1 supplemented by a "
        "deterministic new-expression scan of the root def span. Question and root unchanged."
    ]
    item["structure_gold_hash"] = gold_hash(gold)
    anchor = rec["anchor"]
    provenance(item, "REBUILD_ITEM",
               "template-polluted gold rebuilt from file-anchored NEW_CALL recovery"
               + (" (" + rec["anchor_note"] + ")" if rec.get("anchor_note") else ""),
               f"call_openclaw.csv anchored at {anchor[0].rsplit('/', 1)[-1]}:{anchor[1]}-{anchor[2]}; cross-check {rec['cross_check']}",
               old_gh, old_qh)
    return item, dict(item_id=iid, action="REBUILD_ITEM", disposition="REUSE_REEVALUATE",
                      reason="gold rebuilt from NEW_CALL (question/root unchanged)",
                      old_gold_hash=old_gh, new_gold_hash=item["structure_gold_hash"],
                      new_targets="|".join(targets), sample_path=" -> ".join(sample_path))


def apply_fix_label(item):
    gold = item["gold_answer"]
    old = gold["hard_constraints"]["required_labels"].get("target_kind", "")
    gold["hard_constraints"]["required_labels"]["target_kind"] = P.GO_RESCOPE_KIND
    old_gh, old_qh = item["structure_gold_hash"], text_hash(item["question"])
    item["structure_gold_hash"] = gold_hash(gold)
    provenance(item, "FIX_LABEL", "Go has no class instantiation; target_kind relabeled to helper_function (score-neutral)",
               "T3_CLASS_INSTANTIATION_AUDIT.csv row " + item["item_id"], old_gh, old_qh)
    return item, dict(item_id=item["item_id"], action="FIX_LABEL", disposition="REUSE_REEVALUATE",
                      reason="target_kind class_instantiation -> helper_function (score-neutral)",
                      old_label=old, new_label=P.GO_RESCOPE_KIND,
                      old_gold_hash=old_gh, new_gold_hash=item["structure_gold_hash"])


def apply_fix_explanation(item, counter):
    iid = item["item_id"]
    old_gh, old_qh = item["structure_gold_hash"], text_hash(item["question"])
    if iid in P.EXTERNAL_ENTITY_NOTES:
        item["explanation_reference_en"] += " " + P.EXTERNAL_ENTITY_NOTES[iid]
        reason = "external/stdlib entity status documented"
    else:
        if iid == "ollama_cpp_0054":
            item["explanation_reference_en"] = item["explanation_reference_en"].replace(
                "with branch span 95-98", "with function-relative branch span 95-98 (offset from the def start)")
            counter["span_requote"] += 1
        item["explanation_reference_en"] += " " + P.SPAN_DECLARATION
        reason = "evidence_span declared function-relative; explanation re-quoted"
    provenance(item, "FIX_EXPLANATION", reason, "T3 remediation matrix + T3_PROTOCOL section 2.4", old_gh, old_qh)
    return item, dict(item_id=iid, action="FIX_EXPLANATION", disposition="REUSE_REEVALUATE",
                      reason=reason, old_gold_hash=old_gh, new_gold_hash=gold_hash(item["gold_answer"]))


# --------------------------------------------------------------------------

def main():
    # repo root = two levels above fcg_benchmark/v2_t3_remediation/
    os.chdir(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
    v1_items, v1_raw = load_v1()
    matrix = load_matrix()
    impact_audit = load_impact_audit()
    unverif_audit = load_unverif_audit()
    js_recovery = recover_all()
    counter = {k: 0 for k in ("enumeration_replaced", "enumeration_not_found", "question_replaced",
                              "question_en_replaced", "condition_text_replaced", "kind_text_replaced",
                              "span_requote", "path_text_replaced", "path_text_not_found")}

    out_items, records, dispositions = [], [], {}
    for it in v1_items:
        iid = it["item_id"]
        m = matrix.get(iid)
        action = m["final_action"] if m else "KEEP_UNAUDITED_V1"
        cluster = m["cluster_id"] if m else ""
        item = json.loads(json.dumps(it))  # deep copy; v1 dict never mutated
        rec = None
        if action in ("KEEP", "KEEP_UNAUDITED_V1"):
            disp = "REUSE_UNCHANGED"
            rec = dict(item_id=iid, action=action, disposition=disp, reason="")
            if iid == "ollama_go_0014":
                # dedup survivor carrying the audited C-CLASS-GO-MINOR label
                # defect (class audit: MINOR label_only_error).  Score-neutral
                # relabel under LANGUAGE_CORRECT_GO_REMEDIATION; documented
                # deviation from the matrix KEEP action (validation #11).
                item, rec = apply_fix_label(item)
                rec["action"] = "FIX_LABEL"
                rec["reason"] = ("dedup survivor (group 81bd0507) with audited MINOR label_only_error; "
                                 "relabeled like C-CLASS-GO-MINOR (score-neutral, documented deviation)")
                disp = rec["disposition"]
            out_items.append(item)
        elif action == "DEDUPLICATE":
            disp = "DROP_DEDUPLICATE"
            rec = dict(item_id=iid, action=action, disposition=disp,
                       reason=f"true duplicate; survivor={m['survivor_item_id']} (qualname-pinned)",
                       survivor_item_id=m["survivor_item_id"])
        elif action == "EXCLUDE":
            disp = "DROP_EXCLUDE"
            rec = dict(item_id=iid, action=action, disposition=disp,
                       reason="construction failure: template target pollution, mangled root, no NEW_CALL support, no recoverable anchor")
        elif action == "FIX_EXPLANATION":
            item, rec = apply_fix_explanation(item, counter)
            out_items.append(item); disp = rec["disposition"]
        elif action == "FIX_LABEL":
            item, rec = apply_fix_label(item)
            out_items.append(item); disp = rec["disposition"]
        elif action == "EXPAND_ACCEPTABLE":
            item, rec = apply_expand(item, counter)
            out_items.append(item); disp = rec["disposition"]
        elif action == "REWRITE_QUESTION":
            item, rec = apply_rewrite_question(item, counter)
            out_items.append(item); disp = rec["disposition"]
        elif action == "REBUILD_GOLD":
            if iid == "langchain_python_error_v21_0022":
                item, rec = apply_fix_gold_0022(item, counter)
            elif cluster.startswith("C-IMPACT"):
                item, rec = apply_impact(item, m, impact_audit, counter)
            elif cluster == "C-UNVERIF-UNRESOLVED":
                item, rec = apply_unverif_resolve(item, unverif_audit, counter)
            elif cluster == "C-UNVERIF-MAJOR":
                item, rec = apply_condition(item, counter)
            else:
                raise AssertionError(f"unexpected REBUILD_GOLD routing for {iid} ({cluster})")
            if item is not None:
                out_items.append(item)
            disp = rec["disposition"]
        elif action == "REBUILD_ITEM":
            if cluster == "C-CLASS-GO-MAJOR":
                item, rec = apply_go_rescope(item, counter)
                out_items.append(item)
            elif iid == "ollama_cpp_0088":
                item, rec = apply_cpp_0088(item, counter)
                out_items.append(item)
            elif iid == "openclaw_ts_js_0090":
                item, rec = apply_ts_0090(item, counter)
                out_items.append(item)
            elif cluster == "C-CLASS-JS-MAJOR":
                jrec = js_recovery[iid]
                if jrec["status"] == "REBUILD":
                    item, rec = apply_js_rebuild(item, jrec, counter)
                    out_items.append(item)
                else:
                    rec = dict(item_id=iid, action="GATE_EXCLUDE", disposition="DROP_EXCLUDE",
                               reason=jrec["gate_reason"],
                               evidence=f"js_newcall_recovery cross-check {jrec['cross_check']}; "
                                        f"anchor {jrec.get('anchor')}")
            else:
                raise AssertionError(f"unexpected REBUILD_ITEM routing for {iid} ({cluster})")
            disp = rec["disposition"]
        else:
            raise AssertionError(f"unhandled action {action} for {iid}")
        rec.setdefault("cluster_id", cluster)
        rec.setdefault("dataset", it["dataset"])
        rec.setdefault("qtype", it["question_type"])
        records.append(rec)
        dispositions[iid] = disp

    n_cand = len(out_items)
    print(f"candidate items: {n_cand} (floor {FLOOR})")
    from collections import Counter as C
    print("dispositions:", dict(C(dispositions.values())))
    print("counters:", counter)
    gate_fires = [r for r in records if r["action"].startswith("GATE")]
    print(f"gate fires: {len(gate_fires)}")

    # ---------------- write outputs ----------------
    os.makedirs(OUT_DIR, exist_ok=True)
    for d in ("manifest", "mappings", "reports", "validation"):
        os.makedirs(os.path.join(OUT_DIR, d), exist_ok=True)

    cand_path = os.path.join(OUT_DIR, CANDIDATE_NAME)
    with open(cand_path, "w") as f:
        for it in out_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    cand_sha = sha256_bytes(open(cand_path, "rb").read())
    print("candidate SHA-256:", cand_sha)

    def wcsv(rel, rows, fields):
        with open(os.path.join(OUT_DIR, rel), "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            w.writeheader()
            for r in rows:
                w.writerow(r)

    # mappings
    map_rows = []
    rec_by_id = {r["item_id"]: r for r in records}
    for it in v1_items:
        iid = it["item_id"]
        r = rec_by_id[iid]
        new_it = next((x for x in out_items if x["item_id"] == iid), None)
        map_rows.append({
            "original_item_id": iid,
            "v2_item_id": iid if new_it is not None else "",
            "action": r["action"], "source_system": it["dataset"], "qtype": it["question_type"],
            "old_question_hash": text_hash(it["question"]),
            "new_question_hash": text_hash(new_it["question"]) if new_it is not None else "",
            "old_gold_hash": it["structure_gold_hash"],
            "new_gold_hash": new_it["structure_gold_hash"] if new_it is not None else "",
            "source_evidence_hash": text_hash(json.dumps(it["gold_answer"], sort_keys=True, ensure_ascii=False)),
            "prediction_disposition": dispositions[iid],
            "survivor_item_id": r.get("survivor_item_id", ""),
            "exclusion_reason": r.get("reason", "") if dispositions[iid].startswith("DROP") else "",
            "rerun_required": "yes" if dispositions[iid] == "RERUN_ALL_4" else "no",
            "reevaluation_required": "yes" if dispositions[iid] == "REUSE_REEVALUATE" else "no",
        })
    wcsv("mappings/v1_to_v2_item_mapping.csv", map_rows, list(map_rows[0].keys()))
    dedup_rows = [r for r in records if r["action"] == "DEDUPLICATE"]
    wcsv("mappings/deduplicate_survivor_mapping.csv", dedup_rows,
         ["item_id", "survivor_item_id", "dataset", "qtype", "reason"])
    excl_rows = [r for r in records if dispositions[r["item_id"]] == "DROP_EXCLUDE"]
    wcsv("mappings/excluded_items.csv", excl_rows, ["item_id", "dataset", "qtype", "action", "reason", "evidence"])
    disp_rows = [dict(item_id=iid, dataset=rec_by_id[iid]["dataset"], qtype=rec_by_id[iid]["qtype"],
                      action=rec_by_id[iid]["action"], prediction_disposition=d)
                 for iid, d in sorted(dispositions.items())]
    wcsv("mappings/prediction_disposition.csv", disp_rows,
         ["item_id", "dataset", "qtype", "action", "prediction_disposition"])

    # reports
    wcsv("reports/action_application_report.csv", records,
         ["item_id", "dataset", "qtype", "cluster_id", "action", "disposition", "reason", "evidence"])
    impact_rows = [r for r in records if "old_required" in r and r.get("action") in ("REBUILD_GOLD", "EXPAND_ACCEPTABLE")
                   and r.get("qtype") == "impact_scope"]
    wcsv("reports/impact_scope_before_after.csv", impact_rows,
         ["item_id", "action", "old_required", "new_required", "old_acceptable", "new_acceptable", "reason"])
    gold_rows = [r for r in records if r.get("old_gold_hash") and r.get("old_gold_hash") != r.get("new_gold_hash")]
    wcsv("reports/gold_before_after.csv", gold_rows,
         ["item_id", "action", "old_gold_hash", "new_gold_hash", "reason"])
    q_rows = [r for r in records if r.get("old_question_hash") and r.get("old_question_hash") != r.get("new_question_hash")]
    wcsv("reports/question_before_after.csv", q_rows,
         ["item_id", "action", "old_question_hash", "new_question_hash", "reason"])
    label_rows = [r for r in records if r.get("old_label") is not None and "old_label" in r]
    wcsv("reports/label_before_after.csv", label_rows,
         ["item_id", "action", "old_label", "new_label", "reason"])
    expl_rows = [r for r in records if r.get("action") == "FIX_EXPLANATION"]
    wcsv("reports/explanation_before_after.csv", expl_rows, ["item_id", "action", "reason"])
    wcsv("reports/implementation_gate_failures.csv", gate_fires,
         ["item_id", "dataset", "qtype", "cluster_id", "action", "disposition", "reason", "evidence"])

    comp = {}
    for it in out_items:
        comp.setdefault("dataset:" + it["dataset"], 0)
        comp["dataset:" + it["dataset"]] += 1
        comp.setdefault("qtype:" + it["question_type"], 0)
        comp["qtype:" + it["question_type"]] += 1
    comp_rows = [dict(dimension=k.split(":", 1)[0], value=k.split(":", 1)[1], v2_count=v)
                 for k, v in sorted(comp.items())]
    wcsv("reports/composition_report.csv", comp_rows, ["dimension", "value", "v2_count"])

    e1_rows = [
        dict(item_id="clean_v1_0058", e1_disposition="DROP", note="true-duplicate non-survivor under qualname-pinned rule; E1 repair discarded with item"),
        dict(item_id="clean_v1_0112", e1_disposition="RETAIN", note="qualname-pinned survivor; E1 repair retained"),
        dict(item_id="clean_v1_0105", e1_disposition="RERUN", note="C-DUP-SHARED question rewrite; E1 result superseded by v2 rerun"),
        dict(item_id="__other_31_e1_items__", e1_disposition="RETAIN", note="20 never-flagged + 11 keep/relabel/rebuild-gold/expand (incl. former NEEDS_AUTHOR item resolved to REBUILD_GOLD re-eval)"),
    ]
    wcsv("reports/e1_item_disposition.csv", e1_rows, ["item_id", "e1_disposition", "note"])

    # manifest
    def file_sha(p):
        return sha256_bytes(open(p, "rb").read())
    rerun_n = sum(1 for d in dispositions.values() if d == "RERUN_ALL_4")
    reev_n = sum(1 for d in dispositions.values() if d == "REUSE_REEVALUATE")
    manifest = {
        "candidate_status": "benchmark-v2-candidate-accepted",
        "flags": ["not-yet-rerun", "not-yet-reevaluated", "not-final-scientific-freeze"],
        "author_count_override": {
            "original_floor": 590,
            "accepted_candidate_count": 572,
            "reason": "Validity and source recoverability prioritized over restoring nominal benchmark size.",
            "resampling_authorized": False,
            "minified_bundle_reextraction_authorized": False,
        },
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source_commit": "1ff68805cfc3a4b0b8147087f925ccb953edda46 (code-e1-repair-freeze-v1 baseline)",
        "v1_benchmark": {"path": V1_PATH, "sha256": V1_SHA256},
        "v2_candidate": {"file": CANDIDATE_NAME, "sha256": cand_sha, "item_count": n_cand},
        "inputs_sha256": {
            "T3_FINAL_ACTION_MATRIX.csv": file_sha(MATRIX_CSV),
            "T3_CLUSTER_DECISION_DRAFT.csv": file_sha(f"{AUDITS}/t3-author-decision/T3_CLUSTER_DECISION_DRAFT.csv"),
            "T3_HIGH_RISK_DECISION_DRAFT.csv": file_sha(f"{AUDITS}/t3-author-decision/T3_HIGH_RISK_DECISION_DRAFT.csv"),
            "T3_V2_IMPLEMENTATION_SPEC.md": file_sha(f"{AUDITS}/t3-decision-consolidated/T3_V2_IMPLEMENTATION_SPEC.md"),
            "T3_POLICY_CONFLICTS.csv": file_sha(f"{AUDITS}/t3-decision-consolidated/T3_POLICY_CONFLICTS.csv"),
        },
        "item_counts": {
            "v1": 632, "candidate": n_cand,
            "dropped_deduplicate": sum(1 for d in dispositions.values() if d == "DROP_DEDUPLICATE"),
            "dropped_exclude": sum(1 for d in dispositions.values() if d == "DROP_EXCLUDE"),
            "floor_590_breached": n_cand < FLOOR,
            "floor_590_breach_overridden_by_author": True,
        },
        "composition": {k: v for k, v in sorted(comp.items())},
        "action_counts": dict(C(r["action"] for r in records)),
        "prediction_dispositions": dict(C(dispositions.values())),
        "rerun_items": rerun_n, "rerun_runs_x4": rerun_n * 4,
        "reevaluation_only_items": reev_n,
        "gate_results": {
            "G1_question_change_required": [r["item_id"] for r in gate_fires if r["reason"].startswith("G1")],
            "G2_extraction_source_conflict": [r["item_id"] for r in gate_fires if r["reason"].startswith("G2")],
            "G3_not_uniquely_recoverable": [r["item_id"] for r in gate_fires if r["reason"].startswith("G3")],
            "G4_new_duplicates": "none detected (see validation/duplicate_validation.csv)",
            "G5_composition_imbalance": "see reports/composition_report.csv",
        },
        "planned_vs_actual": {
            "planned_candidate": 603, "actual_candidate": n_cand,
            "planned_rerun_items": 23, "actual_rerun_items": rerun_n,
            "planned_reeval_items": 138, "actual_reeval_items": reev_n,
            "planned_drops": 29, "actual_drops": 632 - n_cand,
            "deviation": "23 JS-MAJOR items + 7 impact declique items gate-excluded under G1/G3 (see reports/implementation_gate_failures.csv)",
        },
        "not_final": True,
    }
    with open(os.path.join(OUT_DIR, "manifest", "benchmark_v2_candidate_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)

    print("outputs written under", OUT_DIR)
    return dispositions, records, out_items


if __name__ == "__main__":
    main()
