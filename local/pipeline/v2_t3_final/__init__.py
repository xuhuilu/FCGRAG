"""Shared constants and loaders for T3.7/T3.8 final local result assembly.

Reads frozen inputs only:
- v2 benchmark candidate (tracked in this repo, SHA-pinned)
- T3.7/T3.8 plan CSVs (reports/)
- v1 predictions and v1 hybrid per-item scores (external workspace, SHA-verified)
- v1 benchmark (external workspace, SHA-pinned by manifest)

Writes only under fcg_benchmark/experiments/multilingual_en/results_v22_t3_remediated/.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]  # local/

V2_BENCH_DIR = REPO_ROOT / "benchmark"
V2_BENCH = V2_BENCH_DIR / "all_multilingual_v22_t3_remediated_en_with_explanations.jsonl"
V2_BENCH_SHA256 = "cd321211ae5ede2e8d150fbedd41bb5dfeb7617e5ea77f172b9d7bf205478b60"
V2_ITEM_COUNT = 572

MAPPING_CSV = V2_BENCH_DIR / "mappings" / "v1_to_v2_item_mapping.csv"

# --- Historical remediation-only inputs (NOT needed for reproduction) ---
# The paths below were used by the one-off T3.7/T3.8 remediation drivers on
# the authors' machines. Override via environment variables if you ever need
# to re-run those drivers; paper_data_pack.py does not touch them.
PLAN_CSV = Path(os.environ.get("FCG_PLAN_CSV", "/nonexistent/T3_7_FINAL_INPUT_PLAN.csv"))
METRIC_PLAN_CSV = Path(os.environ.get("FCG_METRIC_PLAN_CSV", "/nonexistent/T3_7_METRIC_DEPENDENCY_PLAN.csv"))
RERUN_LIST_CSV = Path(os.environ.get("FCG_RERUN_LIST_CSV", "/nonexistent/T3_8_RERUN_ITEM_LIST.csv"))

EXT_ROOT = Path(os.environ.get("FCG_EXT_ROOT", "/nonexistent/fcg_benchmark_ext"))
EXT_FCG = EXT_ROOT / "fcg_benchmark"
EXT_RUN_DIR = EXT_FCG / "experiments" / "multilingual_en" / "runs" / "balanced_clean_plus_gpt55_su8"
V1_BENCH = (
    EXT_FCG / "experiments" / "multilingual_en"
    / "benchmarks_with_explanations_v21_balanced_clean_plus_dedup"
    / "all_multilingual_v21_balanced_clean_en_with_explanations.jsonl"
)
V1_BENCH_SHA256 = "df0a53b496c8608299a2da323201675447eab7a3c47e1540a1322c08936c9a94"

# v1 hybrid per-item scores (remediation drivers only; hash-verified on the
# authors' machines)
V1_SCORE_DIR = Path(os.environ.get("FCG_V1_SCORE_DIR", "/nonexistent/v1_scores"))

DATASETS = ["langchain_python", "ollama_cpp", "ollama_go", "openclaw_ts_js"]
METHODS = ["direct", "code_only", "source_ast", "fcg_rag"]
DATASET_CSV = {
    "langchain_python": "merged.csv",
    "ollama_cpp": "ollama-cpp.csv",
    "ollama_go": "ollama-go.csv",
    "openclaw_ts_js": "call_openclaw.csv",
}

# Frozen result bundle shipped in this repository under local/results/.
OUT_DIR = REPO_ROOT / "results"
PRED_DIR = OUT_DIR / "predictions"
SCORE_DIR = OUT_DIR / "scores"
AGG_DIR = OUT_DIR / "aggregates"
SIG_DIR = OUT_DIR / "significance"
MANIFEST_DIR = OUT_DIR / "manifests"
REPORT_DIR = OUT_DIR / "reports"
CHECKPOINT_DIR = OUT_DIR / "checkpoints"  # never committed

# Frozen model configuration (from e1_freeze_manifest / repair records)
MODEL_ALIAS = os.environ.get("SU8_MODEL", "gpt-5.5")
ENDPOINT = os.environ.get("SU8_BASE_URL", "https://www.su8.codes/v1")
GENERATION_TEMPERATURE = 0.0
GENERATION_MAX_TOKENS = 1800
JUDGE_TEMPERATURE = 0.0
JUDGE_MAX_TOKENS = 1400
INNER_MAX_RETRIES = int(os.environ.get("FCG_API_MAX_RETRIES", "6"))
DRIVER_MAX_RETRIES = 3  # same as error_repair_gpt55_su8.py --max-driver-retries default

# Planned API budget (no-retry)
PLANNED_GENERATION_CALLS = 148  # 37 rerun items x 4 methods
PLANNED_JUDGE_CALLS = 592 + 296  # T3.7 rejudge (296 recs x2) + T3.8 rerun judge (148 recs x2)


def sha256_bytes(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def sha256_text(s: str) -> str:
    return hashlib.sha256(s.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def load_jsonl(path) -> list[dict]:
    return [
        json.loads(line)
        for line in Path(path).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_jsonl(path, rows) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def ensure_out_dirs() -> None:
    for d in (PRED_DIR, SCORE_DIR, AGG_DIR, SIG_DIR, MANIFEST_DIR, REPORT_DIR, CHECKPOINT_DIR):
        d.mkdir(parents=True, exist_ok=True)


def verify_frozen_inputs() -> None:
    """Hard-stop checks before any result assembly or API call."""
    assert V2_BENCH.exists(), f"missing v2 benchmark: {V2_BENCH}"
    got = sha256_file(V2_BENCH)
    assert got == V2_BENCH_SHA256, f"v2 benchmark SHA drift: {got}"
    n = sum(1 for line in V2_BENCH.open("rb") if line.strip())
    assert n == V2_ITEM_COUNT, f"v2 item count {n} != {V2_ITEM_COUNT}"
    assert V1_BENCH.exists(), f"missing v1 benchmark: {V1_BENCH}"
    got1 = sha256_file(V1_BENCH)
    assert got1 == V1_BENCH_SHA256, f"v1 benchmark SHA drift: {got1}"
    # v1 prediction files must match the post-repair freeze hashes
    man = json.loads(
        (REPO_ROOT / "results" / "error_repair" / "prediction_hashes_before_after.json").read_text()
    )
    for name, h in man["after"].items():
        p = EXT_RUN_DIR / name
        assert p.exists(), f"missing v1 pred file: {p}"
        actual = sha256_file(p)
        assert actual == h, f"v1 pred file SHA drift: {name}"
    # v1 hybrid score files must match the hybrid freeze manifest
    hm = json.loads(
        (REPO_ROOT / "results" / "error_repair" / "hybrid_score_hashes_before_after.json").read_text()
    )
    for name, h in hm["after"].items():
        p = V1_SCORE_DIR / name
        assert p.exists(), f"missing v1 score file: {p}"
        actual = sha256_file(p)
        assert actual == h, f"v1 score file SHA drift: {name}"
    # evidence files needed for reruns
    for ds, csv_name in DATASET_CSV.items():
        assert (EXT_FCG / csv_name).exists(), f"missing evidence CSV for {ds}"
        assert (EXT_RUN_DIR / f"cache_{ds}.json").exists(), f"missing cache for {ds}"
