#!/usr/bin/env python3
"""Paper data pack for the post-rerun v2 results (no API calls).

Recomputes every quantity the paper's local Results section (RQ3/RQ4) binds to,
from the CURRENT bundles (after the 37-item rerun re-assembly):

  - per-system + overall method table (Struct/Expl/Qual/Consist, delta-Qual%
    relative to the strongest non-FCG baseline)   [tab:gpt55_by_language]
  - per-qtype method table                        [tab:gpt55_by_qtype]
  - question-type metric decomposition (FCG-RAG - Code+AST)
                                                    [tab:local_qtype_metric_decomposition]
  - system x question-type delta-Qual matrix + N  [fig:local_system_type_delta_qual]
  - paired per-item Qual distribution (FCG-RAG - Code+AST): improved/tied/
    degraded, mean/median/Q1/Q3 (type-7), min/max [fig:local_paired_qual_difference]
  - paper-protocol paired inference: overall Qual FCG-RAG vs each baseline with
    root-clustered bootstrap 95% CI (10,000 resamples, master seed 20260731),
    two-sided Pratt-Wilcoxon p (no continuity correction), matched-pairs
    rank-biserial correlation; RQ2 (per-system) and RQ3 (per-qtype) families
    with Holm adjustment                     [tab:local_pairwise_inference]
  - consistency diagnostic (high Consist, low Struct counts)

Protocol validated against the frozen binding values on the pre-rerun scores
(--validate-old): mean exact; clustered CI within 2.2e-4; Pratt p 8.38e-9 vs
bound 1.00e-8; rank-biserial -0.28357 vs bound -0.28195 (minor implementation
drift; the original script survives only as SHA 2f4d31686fb1...).

Writes:
  significance/paper_inference_v22.json   (machine-readable pack)
stdout: human-readable summary.
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

import numpy as np
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from v2_t3_final import SCORE_DIR, SIG_DIR, V2_BENCH, load_jsonl  # noqa: E402

METHODS = ("direct", "code_only", "source_ast", "fcg_rag")
DATASETS = ("langchain_python", "ollama_cpp", "ollama_go", "openclaw_ts_js")
QTYPES = ("conditional_path", "dependency_trace", "error_handling_path",
          "execution_flow", "impact_scope")
SEED = 20260731
N_BOOT = 10000


def qual(r) -> float:
    return (float(r["structure"]) + float(r["explanation"])) / 2.0


def load_scores(path: Path):
    by_item: dict[str, dict[str, dict]] = {}
    for line in path.read_text().splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        by_item.setdefault(r["item_id"], {})[r["method"]] = r
    return by_item


def clustered_ci(diffs: np.ndarray, cluster_ids: list) -> list[float]:
    """Percentile CI of the mean, resampling root clusters with replacement."""
    clusters: dict[object, list[int]] = {}
    for idx, c in enumerate(cluster_ids):
        clusters.setdefault(c, []).append(idx)
    vecs = [diffs[v] for v in clusters.values()]
    rng = np.random.default_rng(SEED)
    n = len(vecs)
    means = np.empty(N_BOOT)
    for b in range(N_BOOT):
        sel = rng.integers(0, n, n)
        means[b] = np.concatenate([vecs[j] for j in sel]).mean()
    lo, hi = np.percentile(means, [2.5, 97.5])
    return [float(lo), float(hi)]


def pratt_p(diffs: np.ndarray) -> float:
    if np.all(diffs == 0):
        return 1.0
    return float(stats.wilcoxon(diffs, zero_method="pratt", correction=False,
                                alternative="two-sided").pvalue)


def rank_biserial(diffs: np.ndarray) -> float:
    """Matched-pairs rank-biserial, Pratt ranks, normalized by W+ + W−."""
    absd = np.abs(diffs)
    ranks = stats.rankdata(absd)
    wp = ranks[diffs > 0].sum()
    wm = ranks[diffs < 0].sum()
    if wp + wm == 0:
        return 0.0
    return float((wp - wm) / (wp + wm))


def holm_adjust(pvals: list[float]) -> list[float]:
    order = np.argsort(pvals)
    m = len(pvals)
    adj = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adj[idx] = min(1.0, running)
    return adj


def quantile_type7(values, q):
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo, hi = int(np.floor(pos)), int(np.ceil(pos))
    if lo == hi:
        return values[lo]
    return values[lo] + (values[hi] - values[lo]) * (pos - lo)


def build_pack(by_item, bench):
    items = sorted(by_item)
    meta = {i: (bench[i]["dataset"], bench[i]["question_type"], bench[i]["root_function"])
            for i in items}
    cluster_ids = [(meta[i][0], meta[i][2]) for i in items]

    def scope_items(dataset=None, qtype=None):
        return [i for i in items
                if (dataset is None or meta[i][0] == dataset)
                and (qtype is None or meta[i][1] == qtype)]

    def method_means(scope):
        out = {}
        for m in METHODS:
            rows = [by_item[i][m] for i in scope]
            if not rows:
                out[m] = {"n": 0, "struct": None, "expl": None, "consist": None,
                          "qual": None, "delta_qual_rel_vs_best_nonfcg": None}
                continue
            out[m] = {
                "n": len(rows),
                "struct": statistics.fmean(float(r["structure"]) for r in rows),
                "expl": statistics.fmean(float(r["explanation"]) for r in rows),
                "consist": statistics.fmean(float(r["consistency"]) for r in rows),
                "qual": statistics.fmean(qual(r) for r in rows),
            }
        nonfcg = [out[m]["qual"] for m in METHODS[:3] if out[m]["qual"] is not None]
        best_nonfcg = max(nonfcg) if nonfcg else None
        for m in METHODS:
            if m == "fcg_rag":
                out[m]["delta_qual_rel_vs_best_nonfcg"] = (
                    (out[m]["qual"] - best_nonfcg) / best_nonfcg
                    if out[m]["qual"] is not None and best_nonfcg else None
                )
            else:
                out[m]["delta_qual_rel_vs_best_nonfcg"] = None
        return out

    pack: dict = {"qual_definition": "(structure + explanation) / 2",
                  "seed": SEED, "n_boot": N_BOOT}

    pack["by_system"] = {ds: method_means(scope_items(dataset=ds)) for ds in DATASETS}
    pack["overall"] = method_means(items)
    pack["by_qtype"] = {q: method_means(scope_items(qtype=q)) for q in QTYPES}

    # question-type metric decomposition (FCG-RAG - Code+AST)
    pack["qtype_decomposition"] = {
        q: {"n": pack["by_qtype"][q]["fcg_rag"]["n"],
            **{f"delta_{k}": pack["by_qtype"][q]["fcg_rag"][k] - pack["by_qtype"][q]["source_ast"][k]
               for k in ("struct", "expl", "qual", "consist")}}
        for q in QTYPES
    }

    # system x qtype delta-Qual matrix
    pack["system_type_delta_qual"] = {
        f"{ds}|{q}": (lambda sc: {"n": sc["fcg_rag"]["n"],
                                  "delta_qual": (sc["fcg_rag"]["qual"] - sc["source_ast"]["qual"])
                                  if sc["fcg_rag"]["n"] else None}
                      )(method_means(scope_items(dataset=ds, qtype=q)))
        for ds in DATASETS for q in QTYPES
    }

    # paired per-item distribution (FCG-RAG - Code+AST)
    deltas = np.array([qual(by_item[i]["fcg_rag"]) - qual(by_item[i]["source_ast"])
                       for i in items])
    pack["paired_distribution"] = {
        "n": len(deltas),
        "improved": int((deltas > 1e-12).sum()),
        "tied": int((np.abs(deltas) <= 1e-12).sum()),
        "degraded": int((deltas < -1e-12).sum()),
        "mean": float(deltas.mean()),
        "median": float(np.median(deltas)),
        "q1_type7": quantile_type7(list(deltas), 0.25),
        "q3_type7": quantile_type7(list(deltas), 0.75),
        "min": float(deltas.min()),
        "max": float(deltas.max()),
    }

    # paper-protocol inference: overall Qual, FCG-RAG vs each baseline
    overall_pairs = {}
    for base in METHODS[:3]:
        d = np.array([qual(by_item[i]["fcg_rag"]) - qual(by_item[i][base]) for i in items])
        overall_pairs[f"fcg_rag_vs_{base}"] = {
            "n": len(d),
            "mean_diff": float(d.mean()),
            "clustered_ci": clustered_ci(d, cluster_ids),
            "pratt_wilcoxon_p": pratt_p(d),
            "rank_biserial": rank_biserial(d),
            "wins": int((d > 1e-12).sum()),
            "ties": int((np.abs(d) <= 1e-12).sum()),
            "losses": int((d < -1e-12).sum()),
        }
    pack["overall_pairwise_inference"] = overall_pairs

    # RQ2 family: per-system FCG-RAG vs Code+AST, Holm-adjusted
    rq2 = {}
    ps = []
    for ds in DATASETS:
        sc = scope_items(dataset=ds)
        d = np.array([qual(by_item[i]["fcg_rag"]) - qual(by_item[i]["source_ast"]) for i in sc])
        rq2[ds] = {"n": len(d), "mean_diff": float(d.mean()), "pratt_wilcoxon_p": pratt_p(d)}
        ps.append(rq2[ds]["pratt_wilcoxon_p"])
    for ds, p_adj in zip(DATASETS, holm_adjust(ps)):
        rq2[ds]["holm_p"] = p_adj
    pack["rq2_system_family"] = rq2

    # RQ3 family: per-qtype FCG-RAG vs Code+AST, Holm-adjusted
    rq3 = {}
    ps = []
    for q in QTYPES:
        sc = scope_items(qtype=q)
        d = np.array([qual(by_item[i]["fcg_rag"]) - qual(by_item[i]["source_ast"]) for i in sc])
        rq3[q] = {"n": len(d), "mean_diff": float(d.mean()), "pratt_wilcoxon_p": pratt_p(d)}
        ps.append(rq3[q]["pratt_wilcoxon_p"])
    for q, p_adj in zip(QTYPES, holm_adjust(ps)):
        rq3[q]["holm_p"] = p_adj
    pack["rq3_qtype_family"] = rq3

    # consistency diagnostic: Consist >= 0.9 and Struct <= 0.2
    diag = {}
    for m in METHODS:
        sel = [i for i in items
               if float(by_item[i][m]["consistency"]) >= 0.9
               and float(by_item[i][m]["structure"]) <= 0.2]
        diag[m] = {"total": len(items), "high_consist_low_struct": len(sel),
                   "by_qtype": {q: sum(1 for i in sel if meta[i][1] == q) for q in QTYPES}}
    pack["consistency_diagnostic"] = diag
    return pack


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-old", type=Path, default=None,
                    help="pre-rerun scores jsonl for protocol validation")
    args = ap.parse_args()

    bench = {it["item_id"]: it for it in load_jsonl(V2_BENCH)}
    if args.validate_old:
        by_item = load_scores(args.validate_old)
        pack = build_pack(by_item, bench)
        ov = pack["overall_pairwise_inference"]["fcg_rag_vs_source_ast"]
        print("VALIDATION on pre-rerun scores (bound values in brackets):")
        print(f"  mean diff  {ov['mean_diff']:+.10f}  [+0.1008673077]")
        print(f"  clust CI   [{ov['clustered_ci'][0]:.10f}, {ov['clustered_ci'][1]:.10f}]"
              "  [[0.07928574844, 0.1238296015]]")
        print(f"  pratt p    {ov['pratt_wilcoxon_p']:.6e}  [1.00170413518e-8]")
        print(f"  rank-biser {ov['rank_biserial']:+.10f}  [+0.2819529652]")
        print("  RQ2 holm:", {k: f"{v['holm_p']:.6g}" for k, v in pack["rq2_system_family"].items()})
        print("   [bound: langchain 6.6e-7, ollama_cpp .4956, ollama_go .00083, openclaw .4956]")
        print("  RQ3 holm:", {k: f"{v['holm_p']:.6g}" for k, v in pack["rq3_qtype_family"].items()})
        print("   [bound: cond .0372, dep .00114, err .0372, exec .0372, impact 5.29e-13]")
        return 0

    by_item = load_scores(SCORE_DIR / "scores_v22_all_methods.jsonl")
    assert len(by_item) == 572
    pack = build_pack(by_item, bench)
    out = SIG_DIR / "paper_inference_v22.json"
    out.write_text(json.dumps(pack, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
                   encoding="utf-8")
    print("wrote", out)
    ov = pack["overall_pairwise_inference"]["fcg_rag_vs_source_ast"]
    print(f"overall Qual FCG-RAG - Code+AST: {ov['mean_diff']:+.4f} "
          f"CI [{ov['clustered_ci'][0]:.4f}, {ov['clustered_ci'][1]:.4f}] "
          f"p={ov['pratt_wilcoxon_p']:.2e} r_rb={ov['rank_biserial']:+.4f}")
    pd_ = pack["paired_distribution"]
    print(f"paired: improved={pd_['improved']} tied={pd_['tied']} degraded={pd_['degraded']} "
          f"mean={pd_['mean']:+.4f} median={pd_['median']:+.4f} "
          f"IQR=[{pd_['q1_type7']:+.4f}, {pd_['q3_type7']:+.4f}]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
