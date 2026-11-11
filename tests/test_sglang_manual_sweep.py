import json
from pathlib import Path
import tempfile
import unittest

from integrations.topology_memory.generate_dp2_case import COMPUTE, HOST
from rsi_memory.topology import fingerprint
from tests.test_configurable_sglang import fake_observed_profile, fixture_case
from topomem_sglang_rsi.domain import SGLangDomain
from topomem_sglang_rsi.manual_sweep import DEFAULT, REFERENCE, RESIDENT, execute, prepare


class MockSweepAdapter:
    source, launch = "1" * 64, "2" * 64
    contract = {"adapter_sources_sha256": {"domain.py": "3" * 64}}
    _compile = SGLangDomain._compile
    _proof = staticmethod(SGLangDomain._proof)
    effective_program_fingerprint = SGLangDomain.effective_program_fingerprint

    def __init__(self, root, stable_win):
        case, inputs, self.preflight = fixture_case(root, "train")
        self.profile = fake_observed_profile(case)
        self.cases = {"train": [case], "validation": ["sentinel-val"], "test": ["sentinel-test"]}
        self.token_inputs = {case["id"]: inputs}
        self.geometry_by_case = {case["id"]: self.preflight}
        self.native_repeats = {}
        self.failed_program = None
        self.calls = []
        self.stable_win = stable_win

    def _parity(self, case, arm, program, *, slots=None):
        self.calls.append(("parity", case["id"], arm, slots))
        job_id = f"job-{len(self.calls)}"
        result = {"feasible": True, "raw_result_sha256": fingerprint(job_id)}
        if arm == "native":
            self.native_repeats[case["id"], slots] = job_id + "-repeat"
        elif program.fingerprint() == self.failed_program:
            result.update(feasible=False, failure="test gate")
        return job_id, result

    def _job(self, case, arm, phase, program, identity, qualification, *, native_hot):
        label, block = identity["label"], identity["block"]
        self.calls.append((phase, case["id"], arm, native_hot, label, block, qualification))
        job_id = f"job-{len(self.calls)}"
        score = 96 if label == "local-layer-2" and (self.stable_win or block != 3) else 100
        return job_id, {"feasible": True, "score": score, "raw_result_sha256": fingerprint(job_id)}


class ManualSweepTests(unittest.TestCase):
    def test_preflight_binds_nine_physical_policies_with_common_budgets(self):
        with tempfile.TemporaryDirectory() as temp:
            adapter = MockSweepAdapter(Path(temp), stable_win=True)
            manifest = prepare(adapter, "test-sweep", 8, 1.0)
            self.assertEqual(len(manifest["policies"]), 9)
            self.assertEqual(manifest["max_gpu_jobs"], 83)
            self.assertEqual(manifest["complete_order_cycle"], 18)
            self.assertEqual(adapter.calls, [])
            labels = [DEFAULT, REFERENCE, *(f"local-layer-{index}" for index in range(6)), RESIDENT]
            self.assertEqual(manifest["declared_policy_order"], labels)
            for row in manifest["order"]:
                self.assertCountEqual(row, labels)
            for column in zip(*manifest["order"], strict=True):
                self.assertEqual(len(set(column)), len(column))
            self.assertGreater(len({row.index(RESIDENT) for row in manifest["order"]}), 1)
            for label, row in manifest["policies"].items():
                audit = row["compilation"]
                layer = row["local_layer_on_each_rank"]
                self.assertEqual(audit["startup"]["hot_slots"], 4096 if label == DEFAULT else 6144)
                if label == RESIDENT:
                    self.assertEqual(row["layer_tiers_by_rank"], [["resident"] * 6] * 2)
                    self.assertEqual(row["arm"], "manual")
                    self.assertEqual(audit["total_incremental_bytes_by_node"][HOST], 0)
                    continue
                expected = ["local_hbm" if index == layer else "host" for index in range(6)]
                self.assertEqual(row["layer_tiers_by_rank"], [expected, expected])
                self.assertEqual(audit["total_incremental_bytes_by_node"][HOST], (10 if layer is not None else 12) * 21537792)
                if layer is not None:
                    self.assertEqual([audit["total_incremental_bytes_by_node"][node] for node in COMPUTE], [21828608] * 2)

    def test_correctness_before_timing_and_selection_uses_only_train(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for stable_win in (False, True):
                output = root / str(stable_win)
                output.mkdir()
                adapter = MockSweepAdapter(output, stable_win=stable_win)
                manifest = prepare(adapter, "test-sweep", 3, 1.0)
                adapter.failed_program = manifest["policies"]["local-layer-3"]["program_sha256"]
                result = execute(adapter, manifest, output)
                self.assertEqual(result["selected"], "local-layer-2" if stable_win else REFERENCE)
                self.assertEqual(set(result["parity_failures"]), {"local-layer-3"})
                self.assertNotIn("local-layer-3", result["sum_window_ms_by_block"])
                self.assertEqual({row[1] for row in adapter.calls}, {"train"})
                self.assertEqual([row[0] for row in adapter.calls], ["parity"] * 9 + ["performance"] * 24)
                timed = adapter.calls[9:]
                self.assertTrue(all(row[2:4] == ("native", 6144) for row in timed if row[4] == REFERENCE))
                self.assertTrue(all(row[2:4] == ("native", None) for row in timed if row[4] == DEFAULT))
                self.assertTrue(all(row[2:4] == ("manual", None) for row in timed if row[4].startswith("local-")))
                self.assertEqual(len([json.loads(line) for line in (output / "jobs.jsonl").read_text().splitlines()]), 33)


if __name__ == "__main__":
    unittest.main()
