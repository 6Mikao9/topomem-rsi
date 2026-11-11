from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from rsi_memory.topology_report import SGLANG_SCOPE, _fingerprint, render
from rsi_memory.topology_report_ablation import ADAPTIVE, BLOCKS, FORMAT, LABELS, MATCH, NATIVE


def save(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return sha256(raw).hexdigest()


class AblationReportTest(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.formal = self.root / "formal" / "controller"
        self.formal.mkdir(parents=True)
        self.study = self.root / "ablation"
        (self.study / "receipts").mkdir(parents=True)
        self.program = {"rules": ["mock program"]}
        self.margin = 11.49229000000014
        self.program_sha = _fingerprint(self.program)
        self.harness = "a" * 64
        self.source, self.profile = "b" * 64, "c" * 64
        self.cases = [{"id": f"mock-test-{i}", **dict.fromkeys(("manifest_sha256", "target_sha256", "topology_sha256",
                       "scenario_sha256", "workload_sha256", "schedule_sha256"), str(i) * 64)} for i in (1, 2)]
        self.contract = {"kind": SGLANG_SCOPE, "cases": {"test": self.cases}, "source_manifest_sha256": self.source,
                         "metadata_profile_sha256": self.profile, "launch_sha256": "d" * 64, "tokenization_sha256": "e" * 64,
                         "generator_program_protocol": {"kind": "backend_program_contract_v3"}, "manual_selection": {"sweep_sha256": "f" * 64}}
        manifest = {"config": {"rounds": 2, "min_improvement": self.margin}, "contract": self.contract}
        rounds = [{"round": 0}, {"round": 1}]
        trial = {"id": "mock-selected-trial", "program": self.program, "program_sha256": self.program_sha, "harness_sha256": self.harness}
        frozen = {"adaptive": {"trial": trial}, "partial": False, "rounds": 2, "admitted_harness_sha256s": [self.harness],
                  "manifest_sha256": _fingerprint(manifest), "rounds_sha256": _fingerprint(rounds)}
        self.frozen_sha = save(self.formal / "frozen.json", frozen)
        report = {"status": "test_settled", "manifest": manifest, "frozen": frozen, "rounds": rounds, "paired_evaluations": {
                  "pair/frozen/test": {"status": "failed", "input": {"split": "test", "programs": {"adaptive": self.program}}}}}
        report_sha = save(self.formal / "report.json", report)
        pipeline_sha = save(self.formal.parent / "pipeline-status.json", [{"phase": phase, "returncode": 0}
                            for phase in ("run", "freeze", "test")])
        archive = self.formal.parent / "frozen-source.zip"
        with ZipFile(archive, "w") as package:
            package.writestr("LOCAL_SOURCE_MANIFEST.json", b"{}")
        self.spec = {"schema": 1, "kind": FORMAT, "blocks": BLOCKS, "cases": self.cases, "driver_sha256": "9" * 64,
                     "worker_source_manifest_sha256": self.source, "metadata_profile_sha256": self.profile, "timed_job_count": 32,
                     "launch_sha256": self.contract["launch_sha256"], "tokenization_sha256": self.contract["tokenization_sha256"],
                     "margin_hand_sweep_canonical_sha256": self.contract["manual_selection"]["sweep_sha256"],
                     "controller_source_archive_sha256": sha256(archive.read_bytes()).hexdigest(),
                     "controller_source_manifest_sha256": sha256(b"{}").hexdigest(),
                     "decision": {"positive_in_every_block": True, "median_strictly_greater_than_ms": self.margin},
                     "native": {"arm": "native", "native_hot": 6144, "program": None, "backing": "all_host"},
                     "adaptive": {"arm": "generated", "program_source": "frozen.adaptive.trial.program"}}
        self.prereg = save(self.study / "preregistration.json", self.spec)
        save(self.study / "manifest.json", {"schema": 1, "kind": FORMAT, "planned_jobs": 32, "bound": {
            "preregistration_sha256": self.prereg, "formal_report_sha256": report_sha, "formal_frozen_sha256": self.frozen_sha,
            "formal_manifest_canonical_sha256": frozen["manifest_sha256"], "formal_pipeline_status_sha256": pipeline_sha,
            "adaptive_program_sha256": self.program_sha, "adaptive_trial_id": trial["id"], "adaptive_harness_sha256": self.harness,
            "adapter_config_field_diff": {"receipt_directory": {}}}})
        save(self.study / "frozen_adaptive.program.json", self.program)

    def receipt(self, case, label, phase, slots, score=None, suffix=""):
        job = "sg-" + _fingerprint([case["id"], label, phase, suffix])[:48]
        digest = _fingerprint(job)
        native = label == NATIVE
        program = None if native else self.program_sha
        mode = "tiered" if native or slots != 6144 else "resident"
        ranks = [{"rank": rank, "hot_slots": slots, "layer_tiers": ["resident" if mode == "resident" else "host"] * 6,
                  "hot_metadata_bytes": 879616 if slots == 6144 else 588800, "packed_device_payload_bytes": 64739328,
                  "device_physical_rows": 16448, "device_logical_row_capacity": 16384, "request_metadata_slots_including_pad": 5,
                  "pinned_host_bytes": 0 if mode == "resident" else 129226752, "local_hbm_bytes": 0, "peer_hbm_bytes": 0}
                 for rank in (0, 1)]
        summary = {"case": case, "scope": SGLANG_SCOPE, "arm": "native" if native else "generated", "phase": phase,
                   "feasible": True, "score": score, "program_sha256": program, "raw_result_sha256": digest,
                   "cross_capacity_native_parity": {"matched": True} if native and phase == "parity" else {},
                   "memory_compilation": {"startup": {"hot_slots": slots, "backing_mode": mode, "native_reference_override": 6144 if native else None},
                     "full_program_sha256": program, "metadata_profile_sha256": self.profile, "base_target_sha256": case["target_sha256"],
                     "base_topology_sha256": case["topology_sha256"], "external_scheduler_sha256": "d" * 64, "ingress_contract_sha256": "e" * 64},
                   "metrics": {**dict.fromkeys(MATCH, "8" * 64), "backing_mode": mode, "ranks": ranks,
                      "packed_kv_sha256_by_rank": ["1" * 64, "2" * 64] if phase == "parity" else [],
                      "timing": {"window_seconds": score / 1000} if score is not None else {}}}
        save(self.study / "receipts" / f"{job}.json", {"schema": 2, "job_id": job, "status": "completed", "returncode": 0,
             "process_group_remaining": False, "process_group_cleanup_required": False, "remaining_compute_pids": "",
             "request_sha256": "7" * 64, "source_manifest_sha256": self.source, "gpu_indices": [0, 1], "gpu_uuids": ["gpu-a", "gpu-b"],
             "result": summary, "result_canonical_sha256": _fingerprint(summary)})
        return {"job_id": job, "raw_result_sha256": digest}

    def complete(self, adaptive_slots=(6144, 6144)):
        startup, proof, rows = {}, {}, []
        for case, slots in zip(self.cases, adaptive_slots, strict=True):
            key = case["id"]
            proof[key] = {ADAPTIVE: self.receipt(case, ADAPTIVE, "parity", slots), NATIVE: self.receipt(case, NATIVE, "parity", 6144)}
            repeat = self.receipt(case, NATIVE, "parity", 6144, suffix="repeat")
            rows.append({"phase": "qualification", "case": key, "native_same_capacity_repeat_job_id": repeat["job_id"], "proofs": proof[key]})
            startup[key] = {ADAPTIVE: {"hot_slots": slots, "backing_mode": "resident" if slots == 6144 else "tiered"},
                            NATIVE: {"hot_slots": 6144, "backing_mode": "tiered"}, "same_hot_capacity": slots == 6144}
        blocks = []
        for block in BLOCKS:
            windows, ids = ({label: {} for label in LABELS} for _ in range(2))
            for case, slots in zip(self.cases, adaptive_slots, strict=True):
                key = case["id"]
                for label in block["order"]:
                    score = 100 if label == ADAPTIVE else 120
                    timed = self.receipt(case, label, "performance", slots if label == ADAPTIVE else 6144, score, block["block_id"])
                    rows.append({**timed, "phase": "performance", "identity": {"kind": FORMAT, "preregistration_sha256": self.prereg,
                                "driver_sha256": self.spec["driver_sha256"], "formal_frozen_sha256": self.frozen_sha,
                                "block_id": block["block_id"], "case_id": key, "label": label},
                                "window_ms": score, "qualification": proof[key][label]})
                    windows[label][key], ids[label][key] = score, timed["job_id"]
            blocks.append({**block, "window_ms_by_arm_case": windows, "job_id_by_arm_case": ids,
                           "window_ms_sum_by_arm": {ADAPTIVE: 200, NATIVE: 240}, "native_minus_adaptive_ms": 40})
        (self.study / "jobs.jsonl").write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")
        save(self.study / "result.json", {"status": "completed", "timed_jobs_completed": 32, "complete_blocks": blocks, "median_ms": 40,
             "startup_by_case": startup, "practical_margin_ms": self.margin, "positive_blocks": 8,
             "performance_gate": "passed", "preregistration_sha256": self.prereg})
        return rows

    def view(self, name="view", *, run=True):
        data = render(self.root / f"{name}.html", runs=[(None, self.formal)] if run else [], ablations=[self.study])
        return data["ablations"][0]

    def test_complete_joins_preregistration_formal_freeze_and_every_receipt(self):
        self.complete()
        view = self.view()
        self.assertEqual((view["status"], view["wins"], view["median_gain_ms"], view["passed"]), ("complete", 8, 40, True))
        self.assertTrue(view["same_hot_capacity"])
        self.assertEqual(self.view("no-formal-run", run=False)["status"], "unverified")

    def test_mixed_capacity_is_reported_and_missing_timing_withholds_all_results(self):
        rows = self.complete((6144, 4096))
        view = self.view()
        self.assertFalse(view["same_hot_capacity"])
        self.assertEqual(view["adaptive_hot_slots"], [4096, 6144])
        (self.study / "receipts" / f"{rows[-1]['job_id']}.json").unlink()
        self.assertEqual(self.view("missing-receipt"), {"status": "unverified", "completed_blocks": None})

    def test_skip_and_partial_show_only_status_and_counts(self):
        path = self.study / "result.json"
        save(path, {"status": "skipped_budget", "preregistration_sha256": self.prereg, "timed_jobs_completed": 0, "performance_gate": "not_evaluated"})
        self.assertEqual(self.view(run=False)["status"], "skipped")
        save(path, {"status": "budget_incomplete", "preregistration_sha256": self.prereg, "timed_jobs_completed": 4,
                    "performance_gate": "not_evaluated", "complete_blocks": BLOCKS[:1], "median_ms": 900})
        partial = self.view("partial")
        self.assertEqual((partial["status"], partial["completed_blocks"]), ("partial", 1))
        self.assertNotIn("passed", partial)
        self.assertNotIn("median_gain_ms", partial)


if __name__ == "__main__":
    unittest.main()
