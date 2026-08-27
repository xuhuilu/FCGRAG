"""Curated repair manifests for the benchmark-v2 candidate.

Every entry is traceable to a T3 audit artifact (impact audit, class audit,
unverifiable-entity audit, duplicate groups, adjudication evidence packets)
or to a direct frozen-source read recorded in the build report.  These are
structured patch manifests: the builder applies them deterministically and
replays are reproducible — no JSONL is hand-edited.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# EXPAND_ACCEPTABLE (2 high-risk items)
# ---------------------------------------------------------------------------
EXPAND_ACCEPTABLE = {
    # T3_IMPACT_SCOPE_AUDIT: gold direct = 8 of 12 real direct callers; the 4
    # misses are comprehension-hidden (.listcomp pseudo callers), source-verified
    # at langchain_core/callbacks/base.py:5722-5843.
    "langchain_python_impact_v21_0017": {
        "add_direct": ["abatch", "abatch_as_completed", "batch", "batch_as_completed"],
        "add_impacted": ["abatch", "abatch_as_completed", "batch", "batch_as_completed"],
        "reason": "add 4 source-verified direct callers hidden inside list comprehensions (batch family, base.py:5722-5843)",
    },
    # Impacted set capped at 12: only 4 of 8 same-depth forward_mul_mat
    # instantiations kept despite 8 RESOLVED edges (repack.cpp).
    "ollama_cpp_impact_v21_0005": {
        "add_direct": [],
        "add_impacted": [
            "ggml::cpu::repack::tensor_traits<block_q4_0, 8L, 4L, GGML_TYPE_Q8_0>::forward_mul_mat",
            "ggml::cpu::repack::tensor_traits<block_q4_0, 8L, 8L, GGML_TYPE_Q8_0>::forward_mul_mat",
            "ggml::cpu::repack::tensor_traits<block_q4_K, 4L, 8L, GGML_TYPE_Q8_K>::forward_mul_mat",
            "ggml::cpu::repack::tensor_traits<block_q4_K, 8L, 8L, GGML_TYPE_Q8_K>::forward_mul_mat",
        ],
        "reason": "add the 4 forward_mul_mat instantiations cut by the impacted=12 cap (8 RESOLVED same-depth edges)",
    },
}

# ---------------------------------------------------------------------------
# Impact-scope gold repairs (REBUILD_GOLD, driven by T3_IMPACT_SCOPE_AUDIT.csv)
# add_direct/add_impacted entity names are read from the audit CSV at build
# time unless given explicitly here.
# ---------------------------------------------------------------------------
IMPACT_ADD_BOTH = [  # missing callers go to required_entities AND acceptable_sets[0]
    "clean_v1_0075",
    "langchain_python_impact_v21_0005",
    "langchain_python_impact_v21_0009",
    "langchain_python_impact_v21_0010",
    "langchain_python_impact_v21_0012",
    "langchain_python_impact_v21_0018",
    "openclaw_ts_js_0034",
    "openclaw_ts_js_impact_v21_0014_assert",
    "openclaw_ts_js_impact_v21_0014_captureEnv",
    "openclaw_ts_js_impact_v21_0015_connect",
    "openclaw_ts_js_impact_v21_0018_resolveBrowserConfig",
    "openclaw_ts_js_impact_v21_0019_resolveCommandSecretRefsViaGateway",
]
IMPACT_ADD_IMPACTED_ONLY = {  # missing callers go to acceptable_sets[0] only
    "langchain_python_impact_v21_0013": ["astream_events"],  # single RESOLVED depth-2 caller (cf. twin 0006)
    "ollama_cpp_impact_v21_0008": None,  # 4 compute_forward instantiations, read from audit CSV
    "ollama_go_impact_v21_0008": ["gatedDeltaFallback", "sampleTokensSerial"],  # marked "(impacted)" in audit
}
IMPACT_DROP = {  # documented spurious callers to remove from both sets
    "openclaw_ts_js_impact_v21_0014_assert": ["main"],  # extern stub, not a real caller of assert
    "openclaw_ts_js_0044": ["main"],  # extern stub (audit: spurious_callers=main)
}
# anon-closure callers are unscorable under mangling; the audit names the
# enclosing nameable function, which is added to both sets instead.
IMPACT_ADD_ENCLOSING = {
    "ollama_go_impact_v21_0003": ["newManifestWriter"],   # anon <create.go:333> closure returned by newManifestWriter
    "ollama_go_impact_v21_0005": ["LaunchCmd"],           # anon RunE closure inside LaunchCmd (launch.go:228)
    "ollama_go_impact_v21_0006": ["BoolWithDefault", "String", "Uint", "Uint64"],  # 4 anon closures (config.go)
    "ollama_go_impact_v21_0007": ["PushHandler"],         # anon goroutine inside PushHandler (routes.go:1003)
}
# audit-verified no-change items (provenance flag only; evaluator
# normalization repair restores scorability of template/qualified names).
IMPACT_NO_CHANGE = {
    "langchain_python_impact_v21_0011": "ambiguous same-named callers; no documented set change (flag only)",
    "ollama_cpp_impact_v21_0006": "normalization-only; ambiguous ggml_backend_cpu_repack_buffer_set_tensor flagged",
    "ollama_cpp_impact_v21_0009": "normalization-only; ambiguous specialization constructors flagged",
    "ollama_cpp_impact_v21_0010": "base destructor: 8 specialization destructors are TRUE derived->base callers; keep",
    "ollama_go_0075": "only CSV caller outside gold is the recursive self-edge (conventionally excluded); keep",
    "openclaw_ts_js_impact_v21_0018_getSession": "all 8 RESOLVED named non-test callers already in gold; 'ts:1150>' anon mangling flagged",
}
# virtual-destructor sibling clique: each item's gold mixes all 8 sibling
# destructors; the 7 spurious siblings (audit CSV spurious_callers) are
# dropped.  The declique leaves an empty scored set (static-object
# destruction has no explicit call site), so these items gate out (G1).
IMPACT_DECLIQUE = [
    "ollama_cpp_impact_v21_0013",
    "ollama_cpp_impact_v21_0014",
    "ollama_cpp_impact_v21_0015",
    "ollama_cpp_impact_v21_0016",
    "ollama_cpp_impact_v21_0017",
    "ollama_cpp_impact_v21_0018",
    "ollama_cpp_impact_v21_0019",
]

# ---------------------------------------------------------------------------
# C-UNVERIF-MAJOR condition repairs (frozen-source reads, see build report)
# ---------------------------------------------------------------------------
CONDITION_REPAIRS = {
    # chat_models.py:1959-1968 (elif): gold had single quotes + extra paren
    "clean_v1_0044": (
        'event.type == "content_block_start" and event.content_block is not None and '
        '("tool_result" in event.content_block.type or "tool_use" in event.content_block.type or '
        '"document" in event.content_block.type or "redacted_thinking" in event.content_block.type)',
        "langchain_anthropic/chat_models.py:1959-1968",
    ),
    # base.py:3817: if schema := payload.pop("response_format", None):
    "clean_v1_0097": (
        'schema := payload.pop("response_format", None)',
        "langchain_openai/chat_models/base.py:3817",
    ),
    # base.py:382: if raw_tool_calls := _dict.get("tool_calls"):
    "clean_v1_0125": (
        'raw_tool_calls := _dict.get("tool_calls")',
        "langchain_openai/chat_models/base.py:382",
    ),
    # repack.cpp:2275-2280 multi-line if, flattened; verified identical to the
    # existing gold string modulo whitespace (normalization-only confirmation)
    "ollama_cpp_0026": (
        "op->op == GGML_OP_MUL_MAT && op->src[0]->buffer && (ggml_n_dims(op->src[0]) == 2) && "
        "op->src[0]->buffer->buft == ggml_backend_cpu_repack_buffer_type() && "
        "ggml_repack_get_optimal_repack_type(op->src[0])",
        "ggml-cpu/repack.cpp:2275-2280",
    ),
}

# ---------------------------------------------------------------------------
# High-risk REBUILD_ITEM repairs (adjudication evidence packets)
# ---------------------------------------------------------------------------
REBUILD_CPP_0088 = {
    # sgemm.cpp:865-866: if-true branch is a bare assignment (zero calls); the
    # recorded path lives in the succeeding tile loop (867-910).  Rebuilt as a
    # loop-body item over RESOLVED callees only (packet section 3); intrinsics
    # (_mm_*) are real but NOT_IN_EXTRACTION and must not be forced path
    # entities (author directive).
    "question": "For the currently extracted loop_body around the condition `job < end`, what path is recorded in gemmMx4?",
    "condition": "job < end",
    "branch_kind": "loop_body",
    "path": ["gemmMx4", "load", "unhalf", "madd", "hsum"],
    "optional": ["updot", "_mm256_sign_epi8"],
    "evidence": "sgemm.cpp:865-910 (guard L865, loop L867-910; RESOLVED callees load/unhalf/madd/hsum)",
}
REBUILD_TS_0090 = {
    # gateway.ts:488-513 retry branch; inner if at :490 verified in frozen
    # source; gold path = retry branch in exact source order (packet errata).
    # Root de-mangled to <anon@gateway.ts:464>; ~900-char condition dump and
    # the 'includes' builtin pseudo node removed (author directive).
    "question": (
        'For the currently extracted if_true around the condition `errMsg.includes("token") || '
        'errMsg.includes("401") || errMsg.includes("11244")`, what path is recorded in <anon@gateway.ts:464>?'
    ),
    "root_entity": "<anon@gateway.ts:464>",
    "condition": 'errMsg.includes("token") || errMsg.includes("401") || errMsg.includes("11244")',
    "branch_kind": "if_true",
    "path": ["<anon@gateway.ts:464>", "clearTokenCache", "getAccessToken", "sendC2CInputNotify", "start"],
    "evidence_span": {"start_line": 24, "end_line": 49},  # retry branch 488-513 relative to def 464
    "evidence": "gateway.ts:488-513 (retry branch; guard L490 verified in frozen source)",
}
FIX_GOLD_ERROR_0022 = {
    # agent.py:1314-1362: drop builtin 'callable' (a check, not a call step);
    # insert the real raising call 'plan' (L1317); repair the scored trigger
    # to 'plan' (question-embedded -> question changes -> RERUN, author-approved).
    "question": "Within the current evidence span for AgentExecutor, what local failure-handling path is recorded when plan is triggered?",
    "trigger": "plan",
    "path": ["AgentExecutor", "_prepare_intermediate_steps", "plan", "handle_parsing_errors", "on_agent_action", "tool_run_logging_kwargs"],
    "error_action": "raise ValueError(msg)",  # verbatim at agent.py:1334
    "evidence": "agent.py:1314 (_prepare_intermediate_steps), :1317 (plan raises OutputParserException), :1322 (except), :1334 (error_action), :1345 (handle_parsing_errors), :1351-1352 (on_agent_action, tool_run_logging_kwargs)",
}
REBUILD_GO_0010 = {
    # images.go:596-708: real composite literals are api.ProgressResponse{...}
    # x7 (L619-701) and downloadOpts{...} (L643-648).  All four methods
    # independently predicted exactly these (strongest audit evidence).
    "question": "Within the current static FCG sample, which struct composite literals are instantiated within PullModel?",
    "target_kind": "struct_literal",
    "targets": ["ProgressResponse", "downloadOpts"],
    "sample_path": ["PullModel", "downloadOpts"],
    "add_subgraph": ["github.com/ollama/ollama/api.ProgressResponse", "github.com/ollama/ollama/server.downloadOpts"],
    "evidence": "server/images.go:619-701 (api.ProgressResponse{...} x7), :643-648 (downloadOpts{...}); api/types.go:781",
}

# ---------------------------------------------------------------------------
# C-DUP-SHARED question rewrites (fully qualified roots, frozen-source verified)
# ---------------------------------------------------------------------------
SHARED_FQ_ROOT = {
    "clean_v1_0037": "libs.partners.ollama.langchain_ollama.chat_models.ChatOllama.with_structured_output",
    "clean_v1_0107": "libs.partners.fireworks.langchain_fireworks.chat_models.ChatFireworks.with_structured_output",
    "clean_v1_0124": "libs.partners.groq.langchain_groq.chat_models.ChatGroq.with_structured_output",
    "clean_v1_0105": "libs.partners.fireworks.langchain_fireworks.chat_models.ChatFireworks.with_structured_output",
    "clean_v1_0122": "libs.partners.groq.langchain_groq.chat_models.ChatGroq.with_structured_output",
    "langchain_python_impact_v21_0007": "libs.core.langchain_core.tracers.base.AsyncBaseTracer._end_trace",
    "langchain_python_impact_v21_0014": "libs.core.langchain_core.tracers.base.BaseTracer._end_trace",
}

# ---------------------------------------------------------------------------
# Explanation repairs
# ---------------------------------------------------------------------------
SPAN_DECLARATION = (
    "Note: evidence_span values are function-relative offsets from the function "
    "definition start line, not absolute source line numbers."
)
SPAN_ITEMS = ["clean_v1_0026", "clean_v1_0028", "ollama_cpp_0054", "ollama_go_0006", "ollama_go_0113", "openclaw_ts_js_0003"]
EXTERNAL_ENTITY_NOTES = {
    "langchain_python_error_v21_0016": "Note: 'callable' is a Python builtin (external to the extraction graph); it is source-verifiable as a builtin check, not a project call.",
    "ollama_go_0078": "Note: Printf, cancel and byte are Go standard-library / predeclared entities (external to the extraction graph); they are semantically legitimate path elements.",
}

# ---------------------------------------------------------------------------
# C-UNVERIF-UNRESOLVED per-item verdicts (KEEP_ONLY_SOURCE_VERIFIABLE_ENTITIES)
# Implementation-time verification of each flagged entity against the frozen
# sources (T3.6 read-only inputs); verdicts:
#   KEEP_SOURCE_VERIFIED  entity/condition located verbatim in frozen source
#   KEEP_EXTERNAL         stdlib/builtin/predeclared entity (documented)
#   DROP_ENTITY           not verifiable -> removed from scored set (REBUILD_GOLD)
#   GATE_G2               extraction/source conflict, cannot auto-adjudicate
# ---------------------------------------------------------------------------
UNVERIF_VERDICTS = {
    "clean_v1_0052": ("KEEP_SOURCE_VERIFIED", "tools/base.py: 'if error_to_raise:' guard verbatim (error_to_raise assigned in except arm)"),
    "clean_v1_0064": ("KEEP_SOURCE_VERIFIED", "tools/base.py: 'if error_to_raise:' guard verbatim (async twin)"),
    "clean_v1_0100": ("KEEP_SOURCE_VERIFIED", "chat_models.py:939/955: 'if run_managers:' verbatim in BaseChatModel.generate"),
    "clean_v1_0114": ("KEEP_SOURCE_VERIFIED", "tools/base.py: validate_arguments_v1 present in frozen source"),
    "langchain_python_error_v21_0009": ("KEEP_SOURCE_VERIFIED", "agent.py:216 def tool_run_logging_kwargs + use at :1352 (source-real, extraction gap)"),
    "langchain_python_error_v21_0019": ("KEEP_EXTERNAL", "numpy np.linalg.norm (chroma vectorstores.py:100-101); external numeric library"),
    "ollama_cpp_0017": ("GATE_G2", "condition 'src1_cont' absent from frozen source (cf. is_src1_contiguous in binary-ops.cpp, a different function); candidate guards at span (ggml-cpu.c:1271-1289) are 'if (!llamafile_sgemm(...))' and 'if (src1->type != vec_dot_type)' — branch identity not auto-adjudicable"),
    "ollama_cpp_0086": ("KEEP_SOURCE_VERIFIED", "ggml-quants.c:3686: 'if (quant_weights)' verbatim inside quantize_row_iq3_xxs_impl (def :3624)"),
    "ollama_go_0001": ("KEEP_SOURCE_VERIFIED", "server/create.go:136: 'ctx, cancel := context.WithCancel(...); defer cancel()'"),
    "ollama_go_0004": ("KEEP_SOURCE_VERIFIED", "server/create.go:136: cancel verbatim"),
    "ollama_go_0034": ("KEEP_SOURCE_VERIFIED", "runner/ollamarunner/runner.go:589: 'if i+1 == len(seq.inputs) || seq.embeddingOnly {' verbatim"),
    "ollama_go_0062": ("KEEP_SOURCE_VERIFIED", "fs/ggml/gguf.go:314: 'for i := range a.size' — size present in Decode region"),
    "ollama_go_0071": ("KEEP_SOURCE_VERIFIED", "runner.go:961-962: 'case resp, ok := <-seq.responses: if ok {' verbatim"),
    "ollama_go_0085": ("KEEP_SOURCE_VERIFIED", "registry.go:818: 'return func(yield func(chunksum, error) bool) {' — yield is the iterator parameter"),
    "ollama_go_0088": ("KEEP_SOURCE_VERIFIED", "convert/convert_nemotron_h.go:41,43,79,82: hidden_size/num_attention_heads JSON tags and guards"),
    "ollama_go_0114": ("KEEP_SOURCE_VERIFIED", "x/create/client/create.go:284-293: 'hasQuantize := false' ... 'if hasQuantize {' (mechanical screen looked in the wrong create.go)"),
    "ollama_go_error_v21_0001": ("KEEP_EXTERNAL", "fmt.Printf/Println are Go stdlib (external to extraction graph)"),
    "ollama_go_error_v21_0003": ("KEEP_EXTERNAL", "context.Context.Err() stdlib method (cloud_proxy.go:247 ctxErr)"),
    "ollama_go_error_v21_0007": ("KEEP_EXTERNAL", "context .Err() stdlib method; byte is a Go predeclared type"),
    "ollama_go_error_v21_0008": ("KEEP_EXTERNAL", "fmt.Println Go stdlib"),
    "ollama_go_error_v21_0009": ("KEEP_EXTERNAL", "fmt.Printf Go stdlib"),
    "ollama_go_error_v21_0018": ("KEEP_EXTERNAL", "fmt.Printf Go stdlib"),
    "ollama_go_error_v21_0022": ("KEEP_EXTERNAL", "fmt.Printf Go stdlib"),
    "ollama_go_error_v21_0023": ("KEEP_EXTERNAL", "fmt.Println Go stdlib"),
    "openclaw_ts_js_error_v21_0012": ("DROP_ENTITY", "'back' not verifiable as a callable entity in state-migrations.ts (only 'rolled back'/'Rollback' prose)"),
    "openclaw_ts_js_error_v21_0020": ("KEEP_SOURCE_VERIFIED", "handleDeployRequest present in extensions/discord/src/monitor/provider.ts"),
    "openclaw_ts_js_error_v21_0021": ("KEEP_SOURCE_VERIFIED", "command-queue.ts:187: lane.startsWith(...) verbatim in drainLane region"),
}
EXTERNAL_ENTITY_NOTE_GENERIC = (
    "Note: one or more path entities are external standard-library/builtin/predeclared "
    "entities (external to the extraction graph); they are semantically legitimate, "
    "source-verifiable path elements."
)

# Go MAJOR rescope question template (12 of 13; ollama_go_0010 is separate)
GO_RESCOPE_QUESTION = "Within the current static FCG sample, which helper functions or method calls are reachable from {root}?"
GO_RESCOPE_KIND = "helper_function"
