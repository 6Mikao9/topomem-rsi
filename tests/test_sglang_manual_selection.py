from contextlib import redirect_stdout
from copy import deepcopy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from integrations.topology_memory.sglang_observe import save
from rsi_memory.execution_worker import digest
from rsi_memory.memory_program import MemoryProgram
from rsi_memory.topology import fingerprint
from tests.test_configurable_sglang import UUIDS, fake_observed_profile, fixture_case
from topomem_sglang_rsi.contract import SCOPE
from topomem_sglang_rsi.domain import SGLangDomain
from topomem_sglang_rsi.hot_config import PARITY_KEYS, compile_memory, stock_control_program
from topomem_sglang_rsi.manual_selection import main, pin_selection
from topomem_sglang_rsi.manual_sweep import execute, prepare


def evidence_fixture(root):
    cases = {name: fixture_case(root, name) for name in ("train", "validation", "test-a", "test-b")}
    for name in ("validation", "test-a", "test-b"):
        (root / "cases" / name / "fixture.json").unlink()
    profile = fake_observed_profile(cases["train"][0])
    receipts = root / "sweep-receipts"
    receipts.mkdir()
    old = {"schema": 2, "experiment_id": "training-hand-sweep", "source_manifest_sha256": "f" * 64,
           "launch_sha256": "e" * 64, "receipt_directory": receipts.name, "local_cases": "cases",
           "cases": {"train": [cases["train"][0]], "validation": [cases["validation"][0]],
                     "test": [cases[name][0] for name in ("test-a", "test-b")]},
           "token_inputs": {name: value[1] for name, value in cases.items()}, "deadline": 10, "job_timeout": 30,
           "paired_block_ids": {"validation": ["v1"], "test": ["t1"]}, "initial_program": stock_control_program().to_dict(),
           "executor": {"host": "mock", "worker": "mock", "root": "mock", "python": "mock", "source": "mock",
                        "gpu_indices": [0, 1], "gpu_uuids": UUIDS}, "hot_profile": profile}

    class MockExecutor:
        def __init__(self, source, source_manifest_sha256, gpu_indices, gpu_uuids, **unused):
            self.config = {"source": source, "source_manifest_sha256": source_manifest_sha256,
                           "gpu_indices": gpu_indices, "gpu_uuids": gpu_uuids, "environment": {}}
            self.default_timing_calls = 0

        def status(self, job_id):
            receipt = receipts / f"{job_id}.json"
            return json.loads(receipt.read_bytes()) if receipt.exists() else {"status": "not_found"}

        def run(self, job_id, module, payload, *, timeout):
            if payload["case"]["id"] != "train":
                raise AssertionError("sweep accessed a protected workload")
            program = None if payload["program"] is None else MemoryProgram.from_dict(payload["program"])
            compiled = compile_memory(*cases["train"][2], profile, program, native_hot=payload["native_hot"])
            audit = compiled.audit
            result = {"scope": SCOPE, "case": payload["case"], "arm": payload["arm"], "phase": payload["phase"],
                      "program_sha256": audit["full_program_sha256"], "memory_compilation": audit, "feasible": True,
                      "raw_result_sha256": fingerprint(payload), "score": None,
                      "metrics": {**{key: "9" * 64 for key in PARITY_KEYS},
                                  "input_tokenization_sha256": fingerprint(payload["expected_inputs"])}}
            if payload["cross_cap_reference"] is not None:
                result["cross_capacity_native_parity"] = {"matched": True}
            if payload["phase"] == "performance":
                if program is None and payload["native_hot"] is None:
                    result["score"] = [20, 24, 26][self.default_timing_calls]
                    self.default_timing_calls += 1
                else:
                    result["score"] = 15 if compiled.workers[0].layer_tiers[2] == "local_hbm" else 31
            request = digest({"operation": "run", "job_id": job_id, "module": module, "payload": payload,
                              "timeout": timeout, **self.config})
            receipt = {"schema": 2, "job_id": job_id, "status": "completed", "request_sha256": request,
                       "returncode": 0, "remaining_compute_pids": "", "process_group_remaining": False,
                       "process_group_cleanup_required": False, "result_canonical_sha256": digest(result),
                       "result_sha256": digest(result), "result": result,
                       **{key: self.config[key] for key in ("source_manifest_sha256", "gpu_indices", "gpu_uuids")}}
            save(receipts / f"{job_id}.json", receipt)
            return receipt

        def close(self):
            pass

    source = root / "schema2.json"
    save(source, old)
    adapter = SGLangDomain(old, root, executor_factory=MockExecutor)
    assert adapter.manual == stock_control_program()
    sweep_dir = root / "sweep"
    sweep_dir.mkdir()
    manifest = prepare(adapter, "hand-control-v1", 3, 5.0)
    try:
        result = execute(adapter, manifest, sweep_dir)
    finally:
        adapter.close()
    save(sweep_dir / "manifest.json", manifest)
    save(sweep_dir / "result.json", result)
    save(sweep_dir / "selected_program.json", manifest["policies"][result["selected"]]["program"])
    current = {**old, "schema": 3, "experiment_id": "formal-run", "receipt_directory": "formal-receipts",
               "paired_block_ids": {"validation": [f"v{i}" for i in range(6)], "test": [f"t{i}" for i in range(12)]},
               **pin_selection(source, sweep_dir)}
    return current, source, sweep_dir, receipts


class ManualSelectionTests(unittest.TestCase):
    def test_replays_training_receipts_and_binds_manual_controller_budget(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, _, _, receipts = evidence_fixture(root)
            adapter = SGLangDomain(config, root, executor_factory=lambda **_: self.fail("validation launched an executor"))
            domain = adapter.domain()
            self.assertEqual(domain.contract["manual_selection"]["selected"], "local-layer-2")
            self.assertEqual(domain.contract["manual_selection"]["receipt_count"], 38)
            self.assertEqual(len(list(receipts.glob("*.json"))), 38)
            self.assertEqual(domain.manual_program, MemoryProgram.from_dict(config["manual_program"]))
            self.assertEqual(domain.training_context["minimum_paired_improvement_ms"], 12)
            self.assertEqual(domain.training_context["manual_control"]["layer_tiers_by_rank"][1][2], "local_hbm")
            self.assertNotIn(str(root), json.dumps(domain.training_context))
            self.assertNotIn('"test"', json.dumps(domain.training_context))
            self.assertFalse((root / "formal-receipts").exists())

            config_path = root / "schema3.json"
            save(config_path, config)
            args = ["manual_selection", "freeze", "--adapter-config", str(config_path), "--output", str(root / "formal")]
            with (patch("sys.argv", args), redirect_stdout(StringIO()),
                  patch("topomem_sglang_rsi.manual_selection.topology_rsi.TopologyRSI") as controller):
                controller.return_value.freeze.return_value = {"rounds": 2}
                main()
                observed = controller.call_args.args[1]
                self.assertEqual((observed.rounds, observed.slots, observed.max_output_tokens), (2, 2, 65536))
                self.assertEqual((observed.wall_seconds, observed.min_paired_blocks, observed.min_improvement), (14400, 6, 12))
                controller.return_value.freeze.assert_called_once_with()

    def test_rejects_changed_program_threshold_and_forged_selection(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, source, sweep_dir, _ = evidence_fixture(root)
            wrong = {**config, "manual_program": stock_control_program().to_dict()}
            with self.assertRaisesRegex(ValueError, "differs from the verified training selection"):
                SGLangDomain(wrong, root)
            wrong = deepcopy(config)
            wrong["controller_preregistration"]["min_improvement_ms"] = 5
            with self.assertRaisesRegex(ValueError, "threshold differs"):
                SGLangDomain(wrong, root)
            path = sweep_dir / "result.json"
            result = json.loads(path.read_bytes())
            result["selected"] = "stock-6144-host"
            path.write_text(json.dumps(result), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "artifact or source config digest changed"):
                SGLangDomain(config, root)
            repinned = {**config, **pin_selection(source, sweep_dir)}
            with self.assertRaisesRegex(ValueError, "disagrees with replay of leased receipts"):
                SGLangDomain(repinned, root)

    def test_rejects_missing_changed_and_rehashed_wrong_request_receipts(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            config, source, sweep_dir, receipts = evidence_fixture(root)
            receipt = next(receipts.glob("*.json"))
            raw = receipt.read_bytes()
            receipt.unlink()
            with self.assertRaisesRegex(ValueError, "missing or symlinked"):
                SGLangDomain(config, root)
            updated = json.loads(raw)
            updated["request_sha256"] = "0" * 64
            save(receipt, updated)
            with self.assertRaisesRegex(ValueError, "receipt inventory changed"):
                SGLangDomain(config, root)
            repinned = {**config, **pin_selection(source, sweep_dir)}
            with self.assertRaisesRegex(RuntimeError, "failed or is unresolved"):
                SGLangDomain(repinned, root)


if __name__ == "__main__":
    unittest.main()
