import json
from pathlib import Path
from statistics import median
import tempfile
import unittest

from rsi_memory.topology_report import SGLANG_SCOPE, _fingerprint, read_receipt, read_run, render


class SGLangConfigurationTest(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.paths = {}
        self.blocks = [f"t{i}" for i in range(1, 13)]
        self.source, self.profile = "a" * 64, "b" * 64
        self.programs = {label: {"rules": [label]} for label in ("adaptive", "fixed", "manual")}
        self.programs["native"] = None
        self.cases = [{"id": f"mock-test-{i}", **dict.fromkeys(("manifest_sha256", "target_sha256", "topology_sha256",
                          "scenario_sha256", "workload_sha256", "schedule_sha256"), str(i) * 64)} for i in (1, 2)]
        arms = {}
        for label, program in self.programs.items():
            mode = "tiered" if label == "native" else "resident"
            slots = 4096 if mode == "tiered" else 6144
            arm = label if label in ("native", "manual") else "generated"
            digest = _fingerprint(program) if program is not None else None
            scores, refs = [], []
            for block in self.blocks:
                windows = {}
                for case in self.cases:
                    key = f"{label}-{case['id']}"
                    parity = self.receipt(key + "-parity", case, digest, arm, "parity", slots, mode)
                    repeat = self.receipt(key + "-repeat", case, None, "native", "parity", slots, "tiered")
                    score = 100 if label == "adaptive" else 120
                    timed = self.receipt(key + "-" + block, case, digest, arm, "performance", slots, mode, score=score)
                    refs.extend((parity, repeat, timed))
                    windows[case["id"]] = score
                scores.append({"block_id": block, "score": sum(windows.values()), "case_scores_ms": windows})
            arms[label] = {"feasible": True, "metrics": {"receipts": refs}, "samples": scores}
        pair = {"status": "completed", "input": {"split": "test", "block_ids": self.blocks, "programs": self.programs}, "result": arms}
        comparisons = {}
        for label in ("native", "fixed", "manual"):
            gains = [right["score"] - left["score"] for left, right in zip(arms["adaptive"]["samples"], arms[label]["samples"], strict=True)]
            comparisons[f"adaptive_vs_{label}"] = {"median_improvement": median(gains), "passed": True,
                 "deltas": [{"block_id": block, "reference_minus_candidate": gain} for block, gain in zip(self.blocks, gains, strict=True)]}
        manifest = {"config": {"rounds": 2, "min_improvement": 11.49229}, "contract": {
            "kind": SGLANG_SCOPE, "cases": {"test": self.cases}, "paired_block_ids": {"test": self.blocks},
            "source_manifest_sha256": self.source, "metadata_profile_sha256": self.profile,
            "manual_program_sha256": _fingerprint(self.programs["manual"])}}
        frozen = {"manifest_sha256": _fingerprint(manifest), "rounds_sha256": _fingerprint([]),
                  **{label: {"trial": {"program": self.programs[label], "program_sha256": _fingerprint(self.programs[label])}}
                     for label in ("adaptive", "fixed")}}
        self.report = {"manifest": manifest, "frozen": frozen, "paired_evaluations": {"pair/frozen/test": pair}, "test_comparisons": comparisons}
        self.store_report()

    def receipt(self, job, case, program, arm, phase, slots, mode, *, score=None):
        ranks = [{"rank": rank, "hot_slots": slots, "layer_tiers": ["resident" if mode == "resident" else "host"] * 6,
                  "hot_metadata_bytes": 879616 if slots == 6144 else 588800, "packed_device_payload_bytes": 64739328,
                  "device_physical_rows": 16448, "device_logical_row_capacity": 16384, "request_metadata_slots_including_pad": 5,
                  "pinned_host_bytes": 0 if mode == "resident" else 129226752, "local_hbm_bytes": 0, "peer_hbm_bytes": 0}
                 for rank in (0, 1)]
        summary = {"case": case, "scope": SGLANG_SCOPE, "arm": arm, "phase": phase, "feasible": True, "score": score,
                   "program_sha256": program, "raw_result_sha256": "d" * 64,
                   "memory_compilation": {"startup": {"hot_slots": slots, "backing_mode": mode}, "full_program_sha256": program,
                                          "metadata_profile_sha256": self.profile, "base_target_sha256": case["target_sha256"],
                                          "base_topology_sha256": case["topology_sha256"]},
                   "metrics": {"backing_mode": mode, "ranks": ranks}}
        path = self.root / (job + ".json")
        path.write_text(json.dumps({"schema": 2, "job_id": job, "status": "completed", "returncode": 0,
                                   "process_group_remaining": False, "process_group_cleanup_required": False,
                                   "remaining_compute_pids": "", "request_sha256": "e" * 64, "source_manifest_sha256": self.source,
                                   "gpu_indices": [0, 1], "gpu_uuids": ["gpu0", "gpu1"], "result": summary,
                                   "result_canonical_sha256": _fingerprint(summary)}), encoding="utf-8")
        self.paths[job] = path
        return job

    def store_report(self):
        (self.root / "report.json").write_text(json.dumps(self.report), encoding="utf-8")

    def test_layouts_are_bound_over_all_observations_without_changing_the_verdict(self):
        output = self.root / "view.html"
        view = render(output, runs=[(None, self.root)], receipts=self.paths.values())["runs"][0]["heldout_outcome"]
        configs = view["recorded_configurations"]
        self.assertEqual((configs["fixed"]["status"], configs["fixed"]["matched"], configs["fixed"]["observations"]), ("same", 24, 24))
        self.assertEqual((configs["manual"]["status"], configs["native"]["status"]), ("same", "different"))
        self.assertEqual(configs["fixed"]["cases"][1]["reference"]["ranks"][1]["gpu_uuid"], "gpu1")
        self.assertEqual(view["comparisons"]["fixed"], {"status": "passed", "wins": 12, "median_gain": 40})

    def test_missing_corrupt_wrong_case_or_wrong_program_receipts_leave_comparison_unverified(self):
        key = "manual-mock-test-2-t12"
        index = {job: read_receipt(path, 0) for job, path in self.paths.items() if job != key}
        def status():
            return read_run(self.root, receipt_index=index)["heldout_outcome"]["recorded_configurations"]["manual"]["status"]
        self.assertEqual(status(), "unverified")
        path = self.paths[key]
        receipt = json.loads(path.read_bytes())
        result = receipt["result"]
        result["score"] = 999
        path.write_text(json.dumps(receipt), encoding="utf-8")
        index[key] = read_receipt(path, 0)
        self.assertIs(index[key]["canonical_result_matches"], False)
        self.assertEqual(status(), "unverified")
        result["score"] = 120
        for field, replacement in (("case", self.cases[0]), ("program_sha256", _fingerprint(self.programs["fixed"])), ("phase", "parity")):
            changed = {**result, field: replacement}
            path.write_text(json.dumps({**receipt, "result": changed, "result_canonical_sha256": _fingerprint(changed)}), encoding="utf-8")
            index[key] = read_receipt(path, 0)
            self.assertIs(index[key]["canonical_result_matches"], True)
            self.assertEqual(status(), "unverified")

    def test_reused_timing_job_and_moved_rank_do_not_support_same_configuration(self):
        index = {job: read_receipt(path, 0) for job, path in self.paths.items()}
        for job, receipt in index.items():
            if job.startswith("fixed-"):
                receipt["gpu_uuids"] = ["gpu1", "gpu0"]
        outcome = read_run(self.root, receipt_index=index)["heldout_outcome"]
        self.assertEqual(outcome["recorded_configurations"]["fixed"]["status"], "different")
        refs = self.report["paired_evaluations"]["pair/frozen/test"]["result"]["fixed"]["metrics"]["receipts"]
        refs[-1] = refs[-7]
        self.store_report()
        outcome = read_run(self.root, receipt_index=index)["heldout_outcome"]
        self.assertEqual(outcome["recorded_configurations"]["fixed"]["status"], "unverified")


if __name__ == "__main__":
    unittest.main()
