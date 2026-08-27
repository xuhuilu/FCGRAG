"""Deterministic NEW_CALL recovery for the 37 openclaw JS-MAJOR items.

Rebuilds per-root class-instantiation target sets from the frozen extraction
(``call_openclaw.csv``) using *file-anchored* traversal — the v1 generator
pooled callees by bare name across files, which produced the shared templated
8-target set condemned by the T3 audit.  This module:

1. anchors each root to a unique def (name, file, span) in the frozen
   extraction (an audit-provided anchor file wins when several defs exist;
   a single non-test def wins over test/harness duplicates);
2. runs a hop<=3 BFS over RESOLVED edges keyed by (caller name, caller file),
   collecting NEW_CALL callees;
3. supplements hop-1 targets with a deterministic ``new X`` scan of the root
   def span in the frozen source (recovers builtins the extraction dropped,
   e.g. ``new Error`` in message.ts:53);
4. refuses (gate) when the anchor is not unique, the def is missing, the
   recovered set is empty, or the recovered closure is shared with other
   roots (line-collapsed minified-bundle defs — the extraction cannot
   distinguish those roots at all).

The 8 audit-verified group-A items serve as the cross-check
(``AUDIT_VERIFIED``); a mismatch marks the item G2 instead of REBUILD.
"""

from __future__ import annotations

import csv
import re
from collections import defaultdict

EXTRACTION_CSV = "/home/lxh/fcg_benchmark (8)/fcg_benchmark/call_openclaw.csv"
CSV_SRC_PREFIX = "/home/luxuhui/下载/openclaw-main"
FROZEN_SRC = "/home/lxh/fcg_benchmark (8)/remote_sources/openclaw-main"

MAX_HOP = 3
POOLED_NODES = {"constructor"}  # audit-flagged pooled pseudo node
DEF_KINDS = {"FUNCTION_DECL", "FUNCTION", "CONSTRUCTOR", "METHOD"}
CALL_EDGES = {"FUNCTION_CALL", "METHOD_CALL", "CALLABLE_CALL"}
TEST_MARKERS = (".test.", ".e2e.", "/e2e/", "harness")

_NEW_RE = re.compile(r"\bnew\s+([A-Za-z_$][\w$]*)")

# item_id -> (root, audit-anchored def-file suffix or None)
JS_MAJOR_ITEMS = {
    "openclaw_ts_js_0002": ("messageCommand", "src/commands/message.ts"),
    "openclaw_ts_js_0005": ("syncDocsTree", "scripts/docs-sync-publish.mjs"),
    "openclaw_ts_js_0007": ("evictIdleRuntimeHandles", "src/acp/control-plane/manager.core.ts"),
    "openclaw_ts_js_0011": ("withTimeout", None),  # audit: root pooled across 6 files
    "openclaw_ts_js_0014": ("withSessionActor", "src/acp/control-plane/manager.core.ts"),
    "openclaw_ts_js_0017": ("runWithRequiredActionPinnedMessageTarget", "extensions/msteams/src/channel.ts"),
    "openclaw_ts_js_0020": ("scheduleEvent", "src/agents/pi-embedded-subscribe.handlers.ts"),
    "openclaw_ts_js_0022": ("runFetch", "scripts/readability-basic-compare.ts"),
    "openclaw_ts_js_0037": ("_getCachedCompiledPatterns", None),
    "openclaw_ts_js_0040": ("probeEmbeddedRuntime", "extensions/acpx/src/health/probe.ts"),
    "openclaw_ts_js_0043": ("connectGateway", None),
    "openclaw_ts_js_0048": ("attachGatewayWsMessageHandler", None),
    "openclaw_ts_js_0050": ("withCronServiceForTest", None),
    "openclaw_ts_js_0053": ("_getCachedCompiledWhilePatterns", None),
    "openclaw_ts_js_0055": ("HC", None),
    "openclaw_ts_js_0058": ("compile", None),
    "openclaw_ts_js_0060": ("ua", None),
    "openclaw_ts_js_0062": ("runNodeHost", None),
    "openclaw_ts_js_0065": ("jC", None),
    "openclaw_ts_js_0068": ("insert", None),
    "openclaw_ts_js_0071": ("_doInsertHere", None),
    "openclaw_ts_js_0074": ("NC", None),
    "openclaw_ts_js_0077": ("serveAcpGateway", None),
    "openclaw_ts_js_0084": ("gw", None),
    "openclaw_ts_js_0100": ("createHunksRenderer", None),
    "openclaw_ts_js_0107": ("fromExtension", None),
    "openclaw_ts_js_0114": ("NB", None),
    "openclaw_ts_js_0123": ("$B", None),
    "openclaw_ts_js_0126": ("_loadGrammar", None),
    "openclaw_ts_js_0130": ("createRootAndLookUpScopeName", None),
    "openclaw_ts_js_0136": ("loadGrammarWithConfiguration", None),
    "openclaw_ts_js_0138": ("createIMessageRpcClient", None),
    "openclaw_ts_js_0141": ("startCronForStore", None),
    "openclaw_ts_js_0144": ("findNextMatchSync", None),
    "openclaw_ts_js_0147": ("kc", None),
    "openclaw_ts_js_0151": ("_pushAttributed", None),
    "openclaw_ts_js_0156": ("pushFrame", None),
}

# audit-verified expectations for the group-A cross-check:
#   value = (verified genuine targets [lower bound], verified-empty flag)
AUDIT_VERIFIED = {
    "openclaw_ts_js_0002": ({"Error"}, False),
    "openclaw_ts_js_0005": ({"Set"}, False),
    "openclaw_ts_js_0007": (set(), True),
    "openclaw_ts_js_0011": ({"Promise"}, False),  # anchor expected to gate anyway
    "openclaw_ts_js_0014": ({"Promise"}, False),
    "openclaw_ts_js_0017": (set(), True),
    "openclaw_ts_js_0020": (set(), True),
}


def _frozen(path: str) -> str:
    if path.startswith(CSV_SRC_PREFIX):
        return FROZEN_SRC + path[len(CSV_SRC_PREFIX):]
    return path


def _is_test_path(path: str) -> bool:
    low = path.lower()
    return any(m in low for m in TEST_MARKERS)


class ExtractionIndex:
    def __init__(self, csv_path: str = EXTRACTION_CSV):
        self.defs: dict[str, set[tuple[str, str, str]]] = defaultdict(set)
        self.edges: dict[tuple[str, str], list[list[str]]] = defaultdict(list)
        with open(csv_path, newline="") as f:
            for r in csv.reader(f):
                if len(r) != 15:
                    continue
                if r[14] == "RESOLVED" and r[9] in DEF_KINDS:
                    self.defs[r[10]].add((r[13], r[11], r[12]))
                if r[0] in ("FUNCTION", "FUNCTION_DECL"):
                    self.edges[(r[1], r[4])].append(r)

    def anchor(self, root: str, audit_file: str | None):
        """Resolve a root to (file, start, end); returns (anchor, gate, note)."""
        cands = self.defs.get(root, set())
        if audit_file:
            cands = {c for c in cands if c[0].endswith(audit_file)}
            if not cands:
                return None, (
                    f"G3: audit anchor {audit_file} not found among extraction defs "
                    "(root pooled across files)"
                ), ""
        files = {c[0] for c in cands}
        if len(files) == 1:
            f0 = next(iter(files))
            span = next(iter(cands))
            return (f0, span[1], span[2]), None, ""
        if not audit_file:
            non_test = {c for c in cands if not _is_test_path(c[0])}
            nt_files = {c[0] for c in non_test}
            if len(nt_files) == 1 and files - nt_files:
                f0 = next(iter(nt_files))
                span = next(iter(non_test))
                return (f0, span[1], span[2]), None, (
                    f"anchor chosen by non-test-def rule among {len(files)} candidates"
                )
        if files:
            return None, (
                f"G3: root anchor not unique ({len(files)} defs: "
                + ", ".join(sorted(f.rsplit('/', 1)[-1] for f in files)[:6]) + ")"
            ), ""
        return None, "G3: root def absent from frozen extraction", ""

    def source_span_news(self, file: str, start: str, end: str) -> set[str]:
        try:
            with open(_frozen(file), encoding="utf-8", errors="replace") as f:
                lines = f.read().splitlines()
        except OSError:
            return set()
        body = "\n".join(lines[int(start) - 1:int(end)])
        return set(_NEW_RE.findall(body))

    def recover(self, root: str, anchor):
        """Hop<=3 file-anchored BFS; returns {target: (hop, path)}."""
        f0, start, end = anchor
        targets: dict[str, tuple[int, list[str]]] = {}
        visited = {(root, f0)}
        frontier = [(root, f0, [root])]
        for hop in range(1, MAX_HOP + 1):
            nxt = []
            for name, f, path in frontier:
                for r in self.edges.get((name, f), []):
                    if r[14] != "RESOLVED":
                        continue
                    et, cn, cf = r[5], r[10], r[13]
                    if et == "NEW_CALL":
                        if cn not in POOLED_NODES and cn not in targets:
                            targets[cn] = (hop, path + [cn])
                    elif et in CALL_EDGES:
                        if (cn, cf) not in visited:
                            visited.add((cn, cf))
                            nxt.append((cn, cf, path + [cn]))
            frontier = nxt
        for n in self.source_span_news(f0, start, end):
            if n not in POOLED_NODES and n not in targets:
                targets[n] = (1, [root, n])
        return targets


def recover_all(idx: ExtractionIndex | None = None):
    """Recover all 37 JS-MAJOR items.

    Returns {item_id: {status, root, anchor, targets, paths, gate_reason,
    cross_check}} where status is REBUILD or GATE.
    """
    idx = idx or ExtractionIndex()
    out = {}
    for iid, (root, audit_file) in JS_MAJOR_ITEMS.items():
        anchor, gate, note = idx.anchor(root, audit_file)
        rec = {"item_id": iid, "root": root, "anchor": anchor,
               "targets": {}, "paths": {}, "gate_reason": gate or "",
               "anchor_note": note}
        if anchor is not None and gate is None:
            targets = idx.recover(root, anchor)
            rec["targets"] = {n: h for n, (h, p) in targets.items()}
            rec["paths"] = {n: p for n, (h, p) in targets.items()}
            if not targets:
                rec["gate_reason"] = "G3: recovered NEW_CALL target set is empty; no valid dependency_trace item"
        # cross-check against audit-verified expectations
        got = set(rec["targets"])
        cc = "n/a"
        if iid in AUDIT_VERIFIED:
            verified, expect_empty = AUDIT_VERIFIED[iid]
            if expect_empty:
                cc = "pass" if got == set() else f"FAIL: expected empty, got {sorted(got)}"
            elif rec["targets"]:
                cc = "pass" if verified <= got else f"FAIL: verified {sorted(verified - got)} missing"
            else:
                cc = "skip (gated before recovery)"
        rec["cross_check"] = cc
        if isinstance(cc, str) and cc.startswith("FAIL"):
            rec["status"] = "GATE"
            rec["gate_reason"] = "G2: mechanical recovery contradicts audit-verified evidence: " + cc
        elif rec["gate_reason"]:
            rec["status"] = "GATE"
        else:
            rec["status"] = "REBUILD"
        out[iid] = rec
    # shared-closure gate: identical recovered sets whose anchors collapse to
    # the same (file, start) mean the extraction cannot distinguish the roots
    # (minified-bundle line collapse) — the per-root set is not uniquely
    # recoverable.  Identical small sets across *distinct* anchors are
    # legitimate twins (cf. C-DUP-SHARED) and stay, with a provenance note.
    by_set: dict[tuple[str, ...], list[str]] = defaultdict(list)
    for iid, rec in out.items():
        if rec["status"] == "REBUILD":
            by_set[tuple(sorted(rec["targets"]))].append(iid)
    for key, members in by_set.items():
        if len(members) <= 1:
            continue
        anchors = {(out[m]["anchor"][0], out[m]["anchor"][1]) for m in members}
        if len(anchors) == 1:
            for iid in members:
                out[iid]["status"] = "GATE"
                out[iid]["gate_reason"] = (
                    "G3: recovered closure identical to sibling roots and all anchors "
                    f"collapse to {next(iter(anchors))[0].rsplit('/', 1)[-1]}:"
                    f"{next(iter(anchors))[1]}; extraction cannot distinguish these "
                    "roots, per-root target set not uniquely recoverable"
                )
                out[iid]["shared_with"] = [m for m in members if m != iid]
        else:
            for iid in members:
                out[iid]["anchor_note"] = (
                    (out[iid]["anchor_note"] + "; " if out[iid]["anchor_note"] else "")
                    + f"recovered set identical to {[m for m in members if m != iid]} "
                    "but anchors are distinct (legitimate twin)"
                )
    return out
