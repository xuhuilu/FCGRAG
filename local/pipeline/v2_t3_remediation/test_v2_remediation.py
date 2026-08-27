#!/usr/bin/env python3
"""Deterministic unit tests for the T3 v2 remediation pipeline.

Run:  python3 -m pytest fcg_benchmark/v2_t3_remediation/test_v2_remediation.py -q
or:   python3 fcg_benchmark/v2_t3_remediation/test_v2_remediation.py
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from fcg_benchmark.v2_t3_remediation.normalization import norm, norm_v1, drift
from fcg_benchmark.v2_t3_remediation.js_newcall_recovery import ExtractionIndex, JS_MAJOR_ITEMS, recover_all
from fcg_benchmark.v2_t3_remediation import repair_patches as P

V1_PATH = ("/home/lxh/fcg_benchmark (8)/fcg_benchmark/experiments/multilingual_en/"
           "benchmarks_with_explanations_v21_balanced_clean_plus_dedup/"
           "all_multilingual_v21_balanced_clean_en_with_explanations.jsonl")
OUT_DIR = ("fcg_benchmark/experiments/multilingual_en/"
           "benchmarks_with_explanations_v22_t3_remediated")
CANDIDATE = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
                         OUT_DIR, "all_multilingual_v22_t3_remediated_en_with_explanations.jsonl")


def gold_hash(gold):
    return hashlib.sha256(json.dumps(gold, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()[:16]


class TestNormalization(unittest.TestCase):
    def test_v1_parity_for_plain_names(self):
        for x in ["ParseName", "libs.core.langchain_core.callbacks.manager._configure",
                  "github.com/ollama/ollama/server.Server.CreateHandler", "ChatOllama.with_structured_output",
                  "_merge_configs", "forward_mul_mat"]:
            self.assertEqual(norm(x), norm_v1(x), x)

    def test_template_and_colon_repair(self):
        self.assertEqual(norm("ggml::cpu::repack::gemm"), "gemm")
        self.assertEqual(norm("ggml::cpu::repack::tensor_traits<block_q4_0, 8L, 4L, GGML_TYPE_Q8_0>::forward_mul_mat"),
                         "forward_mul_mat")
        self.assertEqual(norm("ggml::cpu::repack::extra_buffer_type::supports_op"), "supports_op")

    def test_anon_names_preserved(self):
        self.assertEqual(norm("<anon@config.go:178>"), norm_v1("<anon@config.go:178>"))
        self.assertEqual(norm("<anon@gateway.ts:464>"), "ts:464>")

    def test_drift_only_for_template_or_colon(self):
        self.assertFalse(drift("some.plain.Name"))
        self.assertTrue(drift("a::b::c"))
        self.assertTrue(drift("x<y>::z"))


class TestJsRecovery(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.res = recover_all()

    def test_all_37_items_decided(self):
        self.assertEqual(len(self.res), 37)
        for rec in self.res.values():
            self.assertIn(rec["status"], ("REBUILD", "GATE"))

    def test_audit_verified_cross_checks_pass(self):
        for iid, rec in self.res.items():
            self.assertFalse(str(rec["cross_check"]).startswith("FAIL"),
                             f"{iid}: {rec['cross_check']}")
        # exact matches where the audit pinned the full set
        self.assertEqual(set(self.res["openclaw_ts_js_0005"]["targets"]), {"Set"})
        self.assertEqual(set(self.res["openclaw_ts_js_0014"]["targets"]), {"Promise"})
        self.assertEqual(set(self.res["openclaw_ts_js_0007"]["targets"]), set())
        # verified lower bounds
        self.assertIn("Error", self.res["openclaw_ts_js_0002"]["targets"])

    def test_no_template_pool_in_recovered(self):
        pool = {"Aa", "CSSStyleSheet", "CustomEvent", "HTMLElement"}
        for iid, rec in self.res.items():
            if rec["status"] == "REBUILD":
                self.assertFalse(pool & set(rec["targets"]), iid)

    def test_line_collapsed_roots_gated(self):
        for iid in ("openclaw_ts_js_0037", "openclaw_ts_js_0055", "openclaw_ts_js_0107",
                    "openclaw_ts_js_0130", "openclaw_ts_js_0147"):
            self.assertEqual(self.res[iid]["status"], "GATE", iid)
            self.assertIn("G3", self.res[iid]["gate_reason"])


class TestCandidate(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not os.path.exists(CANDIDATE):
            raise unittest.SkipTest("candidate not built yet")
        cls.cand = {json.loads(l)["item_id"]: json.loads(l) for l in open(CANDIDATE)}
        cls.v1 = {json.loads(l)["item_id"]: json.loads(l) for l in open(V1_PATH)}

    def test_keep_items_byte_identical(self):
        matrix = {r["item_id"]: r for r in csv.DictReader(open(
            "/home/lxh/fcg-audits/t3-decision-consolidated/T3_FINAL_ACTION_MATRIX.csv"))}
        n = 0
        for iid, it in self.cand.items():
            m = matrix.get(iid)
            untouched = (m is None) or (m["final_action"] == "KEEP" and "v2_provenance" not in it)
            if untouched:
                self.assertEqual(json.dumps(it, ensure_ascii=False, sort_keys=True),
                                 json.dumps(self.v1[iid], ensure_ascii=False, sort_keys=True), iid)
                n += 1
        self.assertGreater(n, 400)

    def test_gold_hash_function_stable(self):
        for iid, it in self.cand.items():
            self.assertEqual(gold_hash(it["gold_answer"]), it["structure_gold_hash"], iid)

    def test_survivors_present_drops_absent(self):
        surv = list(csv.DictReader(open(os.path.join(
            os.path.dirname(CANDIDATE), "mappings", "deduplicate_survivor_mapping.csv"))))
        self.assertEqual(len(surv), 16)
        for r in surv:
            self.assertNotIn(r["item_id"], self.cand)
            self.assertIn(r["survivor_item_id"], self.cand)
        self.assertIn("clean_v1_0112", self.cand)
        self.assertNotIn("clean_v1_0058", self.cand)
        self.assertNotIn("langchain_python_error_v21_0023", self.cand)
        self.assertIn("langchain_python_error_v21_0022", self.cand)

    def test_high_risk_repairs(self):
        c = self.cand
        g22 = c["langchain_python_error_v21_0022"]["gold_answer"]
        self.assertNotIn("callable", g22["hard_constraints"]["required_entities"])
        self.assertIn("plan", g22["hard_constraints"]["required_entities"])
        self.assertEqual(g22["hard_constraints"]["required_labels"]["error_trigger"], "plan")
        g10 = c["ollama_go_0010"]["gold_answer"]
        self.assertEqual(g10["hard_constraints"]["required_entities"], ["ProgressResponse", "downloadOpts"])
        self.assertEqual(g10["hard_constraints"]["required_labels"]["target_kind"], "struct_literal")
        g88 = c["ollama_cpp_0088"]["gold_answer"]
        self.assertEqual(g88["hard_constraints"]["required_labels"]["condition"], "job < end")
        self.assertNotIn("_mm_set_epi64x", g88["hard_constraints"]["required_entities"])
        g90 = c["openclaw_ts_js_0090"]["gold_answer"]
        self.assertEqual(g90["root_entity"], "<anon@gateway.ts:464>")
        self.assertLess(len(g90["hard_constraints"]["required_labels"]["condition"]), 100)
        self.assertNotIn("includes", g90["hard_constraints"]["required_entities"])
        g17 = c["langchain_python_impact_v21_0017"]["gold_answer"]
        self.assertEqual(len(g17["hard_constraints"]["required_entities"]), 12)

    def test_rewrite_questions_have_fq_root(self):
        for iid, fq in P.SHARED_FQ_ROOT.items():
            if iid in self.cand:
                self.assertIn(fq, self.cand[iid]["question"], iid)

    def test_go_labels(self):
        for it in self.cand.values():
            if it["dataset"] == "ollama_go" and it["question_type"] == "dependency_trace":
                self.assertNotEqual(
                    it["gold_answer"]["hard_constraints"]["required_labels"].get("target_kind"),
                    "class_instantiation", it["item_id"])

    def test_excluded_items_absent(self):
        excl = list(csv.DictReader(open(os.path.join(
            os.path.dirname(CANDIDATE), "mappings", "excluded_items.csv"))))
        self.assertEqual(len(excl), 44)
        for r in excl:
            self.assertNotIn(r["item_id"], self.cand)

    def test_dispositions_cover_632(self):
        disp = list(csv.DictReader(open(os.path.join(
            os.path.dirname(CANDIDATE), "mappings", "prediction_disposition.csv"))))
        self.assertEqual(len(disp), 632)
        self.assertEqual({r["item_id"] for r in disp}, set(self.v1))


if __name__ == "__main__":
    unittest.main(verbosity=2)
