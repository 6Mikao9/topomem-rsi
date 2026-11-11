import hashlib
import json
from pathlib import Path
import re
import sqlite3
import tempfile
import unittest

from rsi_memory.topology_report import _fingerprint, read_run, render


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


class TopologyReportTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.directory = Path(temp.name)
        self.manifest = {"schema": 1, "config": {"rounds": 6, "max_output_tokens": 65536},
                         "contract": {"scope": "cuda_text_derived_fixed_block_sparse", "private": "not embedded"}}

    def test_empty_is_a_standalone_report_with_no_scores(self):
        path = self.directory / "report.html"
        data = render(path)
        self.assertEqual(data["runs"], [])
        self.assertEqual(data["receipts"], [])
        self.assertEqual(len(data["scenario_labels"]), 4)
        page = path.read_text(encoding="utf-8")
        self.assertIn("尚未导入拓扑 RSI 运行", page)
        self.assertNotRegex(page, r"<(?:script|link)\b[^>]+(?:src|href)=")
        with self.assertRaises(FileExistsError):
            render(path)

    def test_read_only_journal_has_priority_and_keeps_failed_and_pending_calls(self):
        journal = self.directory / "journal.sqlite"
        with sqlite3.connect(journal) as db:
            db.execute("CREATE TABLE records(key TEXT PRIMARY KEY, data TEXT NOT NULL)")
            rows = {"manifest": self.manifest, "round/r0000": {"round": 0, "decision": "rejected", "reason": "timeout"},
                    "api/r0000/parent/00": {"status": "failed", "failure": "APIError",
                        "input": {"payload": {"model": "a", "max_tokens": 42, "messages": ["private provider request"]}},
                        "result": {"content": "private provider reply", "usage": {"prompt_tokens": 11}}},
                    "api/r0001/adaptive-meta": {"status": "pending", "input": {}},
                    "plan/r0001/adaptive": {"status": "recorded_before_evaluation", "arm": "adaptive",
                        "plan": {"observed_problem": "QA hypothesis", "training_evidence_ids": ["native"],
                                 "kind": "pre_evaluation_hypothesis", "private": "omit this field"}},
                    "pair/r0000/validation": {"status": "completed", "input": {"split": "validation", "block_ids": ["v1"]},
                        "result": {"native": {"score": 11, "feasible": True, "samples": [{"block_id": "v1", "score": 11,
                            "receipt_id": "actual-job"}], "metrics": {"receipts": ["actual-job"], "score_unit": "ms",
                            "execution_order": [{"block_id": "v1", "order": ["native", "child"]}]}}}}}
            db.executemany("INSERT INTO records VALUES (?, ?)", [(key, json.dumps(value)) for key, value in rows.items()])
        db.close()
        save(self.directory / "report.json", {"manifest": {"config": {"rounds": 1}}, "rounds": []})
        original = journal.read_bytes()
        run = read_run(self.directory)
        self.assertEqual(journal.read_bytes(), original)
        self.assertEqual(run["origin"], "journal.sqlite")
        self.assertEqual(run["manifest"]["config"]["rounds"], 6)
        self.assertEqual(run["rounds"][0]["reason"], "timeout")
        self.assertEqual([row["status"] for row in run["calls"]], ["failed", "pending"])
        self.assertEqual(run["pairs"][0]["arms"]["native"]["samples"][0]["receipt_id"], "actual-job")
        self.assertEqual(run["pairs"][0]["arms"]["native"]["metrics"]["execution_order"][0]["order"], ["native", "child"])
        self.assertEqual(run["meta_plans"][0]["plan"]["observed_problem"], "QA hypothesis")
        self.assertEqual(run["meta_plans"][0]["id"], "plan/r0001/adaptive")
        self.assertNotIn("private", json.dumps(run))

    def test_json_export_source_and_frozen_hashes_and_script_data_escaping(self):
        source = "# </script><script>malicious()</script>\n"
        sha = hashlib.sha256(source.encode()).hexdigest()
        self.manifest["seed_sha256"] = sha
        files = self.directory / "harnesses"
        files.mkdir()
        (files / f"{sha}.py").write_bytes(source.encode())
        rows = [{"round": 0, "decision": "rejected", "active_before": sha, "source_diff": [source]}]
        save(self.directory / "report.json", {"manifest": self.manifest, "rounds": rows, "frozen": {
            "manifest_sha256": _fingerprint(self.manifest), "rounds_sha256": _fingerprint(rows)}})
        path = self.directory / "view.html"
        data = render(path, runs=[("scenario view", self.directory)])
        run = data["runs"][0]
        self.assertTrue(run["frozen"]["hashes_match"])
        self.assertEqual(run["sources"][sha]["source"], source)
        page = path.read_text(encoding="utf-8")
        self.assertNotIn("<script>malicious()", page)
        embedded = re.search(r'id="data">(.*?)</script>', page, re.S).group(1)
        self.assertEqual(json.loads(embedded), data)

    def test_receipt_scope_actions_and_tamper_are_separate_from_run_claims(self):
        result = {"feasible": True, "scope": "cuda_text_derived_fixed_block_sparse", "score": 4,
                  "score_unit": "sum_case_median_ms", "cases": [{"name": "chat", "median_elapsed_ns": 1000000,
                  "diagnostic": {"fixture_kind": "chatbot_growth", "arm": "generated", "actions": [
                      {"action": "place_owner", "node": "gpu1"}, {"action": "evict", "node": "gpu0"}]}}]}
        valid = self.directory / "good.json"
        bad = self.directory / "bad.json"
        receipt = {"schema": 2, "job_id": "good", "status": "completed", "result": result,
                   "result_canonical_sha256": _fingerprint(result)}
        save(valid, receipt)
        save(bad, {**receipt, "job_id": "bad", "result_canonical_sha256": "0" * 64})
        data = render(self.directory / "view.html", receipts=[valid, bad], max_actions=1)
        self.assertEqual(data["runs"], [])
        correct, broken = data["receipts"]
        self.assertEqual(correct["result"]["scope"], "cuda_text_derived_fixed_block_sparse")
        diag = correct["result"]["cases"][0]["diagnostic"]
        self.assertEqual(diag["action_count"], 2)
        self.assertEqual(len(diag["actions"]), 1)
        self.assertFalse(broken["canonical_result_matches"])
        self.assertIsNone(broken["result"])

    def test_scenario_binding_is_distinct_from_authored_fixture(self):
        binding = self.directory / "binding.json"
        manifest = {"rank_placement": [{"rank": 0, "node": "gpu0"}], "attention": {"top_k": 16},
                    "target_sha256": "a" * 64, "workload_sha256": "b" * 64, "schedule_sha256": "c" * 64}
        save(binding, manifest)
        data = render(self.directory / "view.html", scenarios=[(binding, "agent_tools")])
        self.assertEqual(data["scenarios"][0]["binding_sha256"], _fingerprint(manifest))
        self.assertEqual(data["receipts"], [])

    def test_strict_schema2_shows_selected_outputs_and_rejects_missing_effective_digest(self):
        program = {"schema": 1, "kind": "kv_memory_program", "rules": []}
        digest = _fingerprint(program)
        incumbent = {"id": "bootstrap/initial-incumbent", "origin": "initial_program_bootstrap", "score": 8,
                     "program": program, "program_sha256": digest, "status": "feasible"}
        choice = {"origin": "incumbent", "trial": incumbent}
        row = {"round": 0, "decision": "rejected", "reason": "child_retained_incumbent",
               "task_output_selection": {"shared_incumbent": incumbent, "fixed_incumbent": incumbent,
                   "selected": dict.fromkeys(("parent", "child", "fixed"), choice),
                   "fresh": dict.fromkeys(("parent", "child", "fixed", "fixed_a", "fixed_b")),
                   "parent_incumbent_aa": True, "effective_program_sha256_by_raw_program": {digest: "a" * 64}},
               "incumbent_validation": {"score": 8, "status": "feasible", "trial": incumbent}}
        manifest = {**self.manifest, "schema": 2, "task_output_protocol": "shared_measured_train_incumbent_v1",
                    "effective_program_fingerprint": "adapter_training_only_v1"}
        save(self.directory / "report.json", {"manifest": manifest, "rounds": [row]})
        run = read_run(self.directory)
        self.assertEqual(run["schema"], 2)
        view = run["rounds"][0]["task_output_selection"]
        self.assertEqual(view["selected"]["child"]["trial"]["effective_program_sha256"], "a" * 64)
        self.assertEqual(run["rounds"][0]["reason"], "child_retained_incumbent")
        self.assertTrue(view["parent_incumbent_aa"])
        row["task_output_selection"]["effective_program_sha256_by_raw_program"] = {}
        save(self.directory / "report.json", {"manifest": manifest, "rounds": [row]})
        with self.assertRaisesRegex(ValueError, "effective fingerprint"):
            read_run(self.directory)

    def test_sglang_scope_keeps_observed_tiers_and_budget_ledgers_only_for_valid_receipt(self):
        result = {"schema": 1, "scope": "real_model_sglang_hisparse_configurable_memory", "phase": "parity",
                  "arm": "generated", "case": {"id": "explicit-mock-case"}, "feasible": True, "score": None,
                  "memory_compilation": {"base_topology_sha256": "a" * 64, "cold_plan_sha256": "b" * 64,
                     "startup": {"hot_slots": 8192, "metadata_bytes_reserved_by_node": {"gpu0": 581632}},
                     "total_incremental_bytes_by_node": {"gpu0": 1048576}},
                  "metrics": {"ranks": [{"rank": 0, "layer_tiers": ["host", "local_hbm"], "hot_slots": 8192,
                                         "selected_miss_payload_bytes_by_tier": {"host": 2048}}]},
                  "feedback": {"cold_miss_rows_by_rank_layer": [[3, 0]]}, "private_response": "unused"}
        good = self.directory / "mock-parity.json"
        save(good, {"schema": 2, "job_id": "mock-sglang", "status": "completed", "result": result,
                    "result_canonical_sha256": _fingerprint(result)})
        report = self.directory / "view.html"
        serving = render(report, receipts=[good])["receipts"][0]["result"]["serving"]
        self.assertEqual(serving["metrics"]["ranks"][0]["layer_tiers"], ["host", "local_hbm"])
        self.assertEqual(serving["memory_compilation"]["startup"]["hot_slots"], 8192)
        self.assertEqual(serving["memory_compilation"]["total_incremental_bytes_by_node"]["gpu0"], 1048576)
        self.assertEqual(serving["feedback"]["cold_miss_rows_by_rank_layer"], [[3, 0]])
        html = report.read_text(encoding="utf-8")
        self.assertNotIn("private_response", html)
        self.assertIn("Tiny GLM · 架构与集成测试范围", html)


if __name__ == "__main__":
    unittest.main()
