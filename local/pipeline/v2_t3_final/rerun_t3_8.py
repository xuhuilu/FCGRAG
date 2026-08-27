#!/usr/bin/env python3
"""T3.8 targeted reruns for the 37 remediated items (x4 methods = 148 records).

23 original T3.8 items (reports/T3_8_RERUN_ITEM_LIST.csv) plus 14 OpenClaw
dependency-trace items (pass --item-id) whose v2 gold was rebuilt from
file-anchored NEW_CALL evidence.

Uses the exact frozen generation pipeline:
  - same CSV + offline cache + AST + GraphRAGTools as balanced_clean_plus_gpt55_su8
  - build_prediction_prompt (HEAD 4-arg form) per method
  - gpt-5.5 via SU8, temperature=0.0, json_mode=True, max_tokens=1800
  - inner retries = FCG_API_MAX_RETRIES (6); driver retries = 3 with preflight
    prompt/context SHA-256 equality gating (mirrors error_repair_gpt55_su8)

Then scores each new prediction with the frozen v2 evaluator (Struct) and the
frozen judge prompts (explanation + consistency, gpt-5.5 via SU8).

Two-phase checkpoint so a successful generation is never re-billed:
  - rerun_gen_checkpoint.jsonl : generation provenance per (item, method)
  - rerun_judge_checkpoint.jsonl: struct + judges per (item, method)
Resume skips completed phases. No missing values reach the formal bundle.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # fcg_benchmark/

from build_and_generate import build_graphs  # noqa: E402
from fcg_bench import build_ast_map, compute_features  # noqa: E402
from fcg_bench.evaluator import evaluate_prediction  # noqa: E402
from fcg_bench.graphrag_offline import load_offline_cache  # noqa: E402
from fcg_bench.graphrag_rag import parse_llm_json  # noqa: E402
from fcg_bench.graphrag_tools import GraphRAGTools  # noqa: E402
from fcg_bench.schema import BenchmarkItem  # noqa: E402
from evaluate_three_methods_clean_v1_en import (  # noqa: E402
    evaluate_consistency_item,
    evaluate_explanation_item,
)
from run_three_methods_clean_v1_en import (  # noqa: E402
    build_prediction_prompt,
    run_prediction_for_item,
)

from v2_t3_final import (  # noqa: E402
    CHECKPOINT_DIR,
    DATASET_CSV,
    EXT_FCG,
    EXT_RUN_DIR,
    MANIFEST_DIR,
    METHODS,
    RERUN_LIST_CSV,
    V2_BENCH,
    ensure_out_dirs,
    load_jsonl,
    verify_frozen_inputs,
)
from v2_t3_final.su8_client import ENDPOINT, MODEL, SU8Judge, SU8LLM  # noqa: E402
from v2_t3_remediation.js_newcall_recovery import (  # noqa: E402
    CALL_EDGES,
    CSV_SRC_PREFIX,
    FROZEN_SRC,
    POOLED_NODES,
    ExtractionIndex,
    recover_all,
)

GEN_CKPT = CHECKPOINT_DIR / "rerun_gen_checkpoint.jsonl"
JUDGE_CKPT = CHECKPOINT_DIR / "rerun_judge_checkpoint.jsonl"
GEN_LOG = CHECKPOINT_DIR / "rerun_gen_error.log"
LOCK = threading.Lock()

_TOOLS: dict[str, GraphRAGTools] = {}


REMOTE_SOURCES = str(Path(FROZEN_SRC).parent)

# extraction-host source prefix -> local frozen mirror (all four datasets)
SOURCE_PREFIX_MAP = {
    CSV_SRC_PREFIX: FROZEN_SRC,  # openclaw_ts_js
    "/home/luxuhui/下载/ollama-main": REMOTE_SOURCES + "/ollama-main",  # ollama_go/ollama_cpp
    "/home/luxuhui/langchain-master": REMOTE_SOURCES + "/langchain-master",  # langchain_python
}


def _local_source_csv(csv_path: Path) -> Path:
    """Rewrite extraction-machine source prefixes to the local frozen mirror.

    The frozen CSVs carry absolute paths from the extraction host
    (``/home/luxuhui/...``), which do not exist here, so ``build_graphs``
    silently produced empty ``node.code`` — root source sections in prompts
    were blank.  Same prefix mapping as js_newcall_recovery._frozen.
    Derived artifact; lives in the (uncommitted) checkpoint dir.
    """
    src = csv_path.read_text(encoding="utf-8")
    mapped = src
    for old, new in SOURCE_PREFIX_MAP.items():
        mapped = mapped.replace(old, new)
    if mapped == src:
        return csv_path
    out = CHECKPOINT_DIR / (csv_path.stem + "_localsrc.csv")
    if not out.exists() or out.read_text(encoding="utf-8") != mapped:
        out.write_text(mapped, encoding="utf-8")
    return out


class OpenClawTraceTools(GraphRAGTools):
    """File-anchored retrieval for repaired OpenClaw dependency-trace items.

    Descendant lists come from a repaired-graph BFS (RESOLVED edges keyed by
    (caller name, caller file), pooled ``constructor`` excluded) and are
    presented as a *superset* — intermediate call nodes and NEW_CALL targets
    mixed, generic role hints, no sample paths — mirroring the standard
    pipeline's get_khop_descendants so the model performs the same selection
    task as on every other dependency_trace item.  Feeding the recovered
    target set directly (previous behavior) leaked the gold into the context.
    """

    def __init__(self, cg, ast_map, offline_cache, records, index):
        super().__init__(cg, ast_map, offline_cache)
        self._trace_records = {
            rec["root"]: rec for rec in records.values() if rec.get("status") == "REBUILD"
        }
        self._index = index

    def resolve_root(self, root_ref: str) -> str | None:
        root_short = root_ref.rsplit(".", 1)[-1]
        rec = self._trace_records.get(root_short)
        anchor = rec.get("anchor") if rec else None
        if anchor:
            file_path, start, end = anchor
            if file_path.startswith(CSV_SRC_PREFIX):
                file_path = FROZEN_SRC + file_path[len(CSV_SRC_PREFIX):]
            matches = [
                qn for qn, node in self.cg.nodes.items()
                if node.short_name == root_short
                and node.file_path == file_path
                and node.start_line == int(start)
                and node.end_line == int(end)
            ]
            if matches:
                return sorted(matches)[0]
        return super().resolve_root(root_ref)

    def get_khop_descendants(self, qualname: str, max_hops: int = 3, max_nodes: int = 80):
        node = self.cg.nodes.get(qualname)
        rec = self._trace_records.get(node.short_name) if node else None
        if not rec:
            return super().get_khop_descendants(qualname, max_hops=max_hops, max_nodes=max_nodes)
        f0 = rec["anchor"][0]
        root = rec["root"]
        entries: dict[str, int] = {}  # short name -> first-discovery hop
        visited = {(root, f0)}
        frontier = [(root, f0)]
        for hop in range(1, max_hops + 1):
            nxt = []
            for name, f in frontier:
                for r in self._index.edges.get((name, f), []):
                    if r[14] != "RESOLVED":
                        continue
                    et, cn, cf = r[5], r[10], r[13]
                    if et == "NEW_CALL":
                        if cn not in POOLED_NODES and cn not in entries:
                            entries[cn] = hop
                    elif et in CALL_EDGES:
                        if cn not in entries:
                            entries[cn] = hop
                        if (cn, cf) not in visited:
                            visited.add((cn, cf))
                            nxt.append((cn, cf))
            frontier = nxt
        return [
            {"short_name": name, "hop": hop, "role_hint": "business"}
            for name, hop in sorted(entries.items(), key=lambda x: (x[1], x[0]))
        ][:max_nodes]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def build_tools(dataset: str) -> GraphRAGTools:
    if dataset not in _TOOLS:
        csv_path = EXT_FCG / DATASET_CSV[dataset]
        cache_path = EXT_RUN_DIR / f"cache_{dataset}.json"
        # remap extraction-host source prefixes so node.code is populated
        csv_path = _local_source_csv(csv_path)
        _, cg, _ = build_graphs(str(csv_path))
        ast_map = build_ast_map(cg)
        compute_features(cg, ast_map)
        cache = load_offline_cache(str(cache_path))
        if dataset == "openclaw_ts_js":
            _TOOLS[dataset] = OpenClawTraceTools(
                cg, ast_map, cache, recover_all(), ExtractionIndex()
            )
        else:
            _TOOLS[dataset] = GraphRAGTools(cg, ast_map, cache)
    return _TOOLS[dataset]


def bench_item_of(item: dict) -> BenchmarkItem:
    return BenchmarkItem(
        question_type=item["question_type"],
        root_function=item["root_function"],
        question=item.get("question_en", item.get("question", "")),
        gold_answer=item["gold_answer"],
        natural_answer="",
        subgraph_nodes=item.get("subgraph_nodes", []),
        structure_gold_hash=item.get("structure_gold_hash", ""),
        benchmark_tier=item.get("benchmark_tier", "main_experiment"),
    )


class CaptureLLM:
    def __init__(self, inner: SU8LLM):
        self.inner = inner
        self.last_system = None
        self.last_user = None

    def complete(self, system, user, temperature=0.0, json_mode=False):
        self.last_system = system
        self.last_user = user
        return self.inner.complete(system, user, temperature=temperature, json_mode=json_mode)


def compute_prompt_hashes(method, item, tools):
    ctx = tools.build_local_context(item["root_function"])
    if ctx is None:
        return {"resolve_error": f"cannot resolve root: {item['root_function']}"}
    system, user = build_prediction_prompt(method, item, ctx, tools)
    return {
        "prompt_sha256": sha(system),
        "context_sha256": sha(user),
        "prompt_len": len(system),
        "context_len": len(user),
    }


def run_gen(task, item, tools, max_driver_retries):
    method = task["method"]
    rec = {
        "item_id": task["item_id"],
        "dataset": task["dataset"],
        "qtype": task["qtype"],
        "method": method,
        "model": MODEL,
        "endpoint": ENDPOINT,
        "temperature": 0.0,
        "inner_max_retries": int(__import__("os").environ.get("FCG_API_MAX_RETRIES", "6")),
        "started_at": utc_now(),
        "attempts": [],
    }
    hashes = compute_prompt_hashes(method, item, tools)
    rec.update(hashes)
    if "resolve_error" in hashes:
        rec["final_status"] = "failed_resolve"
        rec["completed_at"] = utc_now()
        return rec

    llm = SU8LLM()
    capture = CaptureLLM(llm)
    final_row = None
    for attempt in range(1, max_driver_retries + 1):
        att = {"attempt": attempt, "started_at": utc_now()}
        row = run_prediction_for_item(method, item, tools, capture, temperature=0.0)
        att["finished_at"] = utc_now()
        att["sent_prompt_sha256"] = sha(capture.last_system or "")
        att["sent_context_sha256"] = sha(capture.last_user or "")
        att["sent_hash_match"] = (
            att["sent_prompt_sha256"] == hashes["prompt_sha256"]
            and att["sent_context_sha256"] == hashes["context_sha256"]
        )
        att["error"] = row.error
        att["request_id"] = llm.last_request_id
        att["raw_output_len"] = len(row.raw_output or "")
        att["explanation_len"] = len((row.explanation_answer or "").strip())
        att["ok"] = (
            (not row.error)
            and bool(row.raw_output)
            and bool(parse_llm_json(row.raw_output))
            and bool((row.explanation_answer or "").strip())
        )
        rec["attempts"].append(att)
        if att["ok"]:
            final_row = row
            break
        time.sleep(min(5.0 * attempt, 30.0))

    rec["completed_at"] = utc_now()
    rec["retry_count"] = len(rec["attempts"]) - 1
    if final_row is None:
        rec["final_status"] = "failed_api"
        rec["raw_output"] = row.raw_output or ""
        rec["structure_answer"] = row.structure_answer
        rec["explanation_answer"] = row.explanation_answer
        with LOCK, GEN_LOG.open("a", encoding="utf-8") as f:
            f.write(f"{utc_now()} GEN-FAIL {task['item_id']}|{method}\n")
        return rec

    rec["final_status"] = "success"
    rec["raw_output"] = final_row.raw_output
    rec["raw_output_sha256"] = sha(final_row.raw_output)
    rec["structure_answer"] = final_row.structure_answer
    rec["explanation_answer"] = final_row.explanation_answer
    rec["structure_from_model"] = True
    rec["explanation_nonempty"] = bool(final_row.explanation_answer.strip())
    rec["request_id"] = llm.last_request_id
    rec["prediction_sha256"] = sha(
        json.dumps(
            {
                "structure_answer": final_row.structure_answer,
                "explanation_answer": final_row.explanation_answer,
                "raw_output": final_row.raw_output,
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    return rec


def run_judge(item, gen_rec):
    judge = SU8Judge()
    pred = {
        "structure_answer": gen_rec["structure_answer"],
        "explanation_answer": gen_rec["explanation_answer"],
    }
    bi = bench_item_of(item)
    struct = round(evaluate_prediction(bi, pred["structure_answer"]), 4)
    exp = evaluate_explanation_item(judge, item, item, pred)
    exp_rid = judge.last_request_id
    cons = evaluate_consistency_item(judge, item, pred)
    cons_rid = judge.last_request_id
    return {
        "item_id": gen_rec["item_id"],
        "dataset": gen_rec["dataset"],
        "qtype": gen_rec["qtype"],
        "method": gen_rec["method"],
        "structure": struct,
        "explanation": exp,
        "explanation_request_id": exp_rid,
        "consistency": cons,
        "consistency_request_id": cons_rid,
        "completed_at": utc_now(),
        "status": "success",
    }


def load_keys(path):
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                r = json.loads(line)
                out[(r["item_id"], r["method"])] = r
    return out


def append(path, rec):
    with LOCK, path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--max-driver-retries", type=int, default=3)
    ap.add_argument("--item-id", action="append", dest="item_ids")
    args = ap.parse_args()

    ensure_out_dirs()
    verify_frozen_inputs()

    v2_items = {it["item_id"]: it for it in load_jsonl(V2_BENCH)}
    rerun_items = (
        args.item_ids
        if args.item_ids
        else [r["item_id"] for r in csv.DictReader(RERUN_LIST_CSV.open())]
    )
    if args.item_ids:
        assert len(rerun_items) == len(set(rerun_items)), rerun_items
        assert all(i.startswith("openclaw_ts_js_") for i in rerun_items), rerun_items
        assert all(i in v2_items for i in rerun_items), rerun_items
    else:
        assert len(rerun_items) == 23, len(rerun_items)

    tasks = []
    for iid in rerun_items:
        item = v2_items[iid]
        for m in METHODS:
            tasks.append({"item_id": iid, "dataset": item["dataset"], "qtype": item["question_type"], "method": m})

    gen_done = load_keys(GEN_CKPT)
    gen_done = {k: v for k, v in gen_done.items() if v.get("final_status") == "success"}
    judge_done = load_keys(JUDGE_CKPT)
    if args.limit > 0:
        tasks = tasks[: args.limit]
    print(f"tasks={len(tasks)} gen_done={len(gen_done)} judge_done={len(judge_done)} "
          f"workers={args.workers}", flush=True)

    # prebuild tools per dataset
    for ds in {t["dataset"] for t in tasks}:
        build_tools(ds)

    def process(task):
        item = v2_items[task["item_id"]]
        tools = build_tools(task["dataset"])
        key = (task["item_id"], task["method"])
        if key not in gen_done:
            gen_rec = run_gen(task, item, tools, args.max_driver_retries)
            if gen_rec["final_status"] == "success":
                append(GEN_CKPT, gen_rec)
                with LOCK:
                    gen_done[key] = gen_rec
            else:
                append(GEN_CKPT, gen_rec)
                return ("gen_failed", task, gen_rec)
        else:
            gen_rec = gen_done[key]
        if key not in judge_done:
            jrec = run_judge(item, gen_rec)
            append(JUDGE_CKPT, jrec)
            with LOCK:
                judge_done[key] = jrec
        return ("ok", task, None)

    n_ok = n_fail = 0
    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(process, t): t for t in tasks}
        for i, fut in enumerate(as_completed(futs), 1):
            status, task, _ = fut.result()
            n_ok += status == "ok"
            n_fail += status != "ok"
            if i % 8 == 0 or i == len(tasks):
                print(f"  {i}/{len(tasks)} ok={n_ok} fail={n_fail} elapsed={time.time()-t0:.0f}s", flush=True)

    gen_done = {k: v for k, v in load_keys(GEN_CKPT).items() if v.get("final_status") == "success"}
    judge_done = load_keys(JUDGE_CKPT)
    missing_gen = [k for k in [(t["item_id"], t["method"]) for t in tasks] if k not in gen_done]
    missing_judge = [k for k in [(t["item_id"], t["method"]) for t in tasks] if k not in judge_done]
    print(f"gen_success={len(gen_done)}/{len(tasks)} judge_done={len(judge_done)}/{len(tasks)} "
          f"missing_gen={len(missing_gen)} missing_judge={len(missing_judge)}", flush=True)
    if missing_gen or missing_judge:
        (MANIFEST_DIR / "rerun_missing.json").write_text(
            json.dumps({"missing_gen": missing_gen, "missing_judge": missing_judge}, indent=2),
            encoding="utf-8",
        )
        print("STOP: not all 92 reruns complete; re-run to retry.", flush=True)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
