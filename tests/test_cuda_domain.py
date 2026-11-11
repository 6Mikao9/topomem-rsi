from collections import Counter
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from integrations.topology_memory import torch_eval
from rsi_memory.cuda_domain import CudaDomain
from rsi_memory.execution_client import ExecutionError
from rsi_memory.experiment_cases import VARIANTS, build_configs
from rsi_memory.topology import fingerprint


ROOT = Path(__file__).resolve().parents[1]


def policy(name):
    return json.loads((ROOT / "examples" / "policies" / f"torch-{name}.json").read_text(encoding="utf-8"))


class _Executor:
    def __init__(self):
        self.calls = []
        self.drift = False
        self.work_counts = {}

    def run(self, job, module, payload, *, timeout):
        self.calls.append((job, module, deepcopy(payload), timeout))
        results = []
        for case in payload["cases"]:
            identity = "1"*64 if self.drift and payload["arm"] == "generated" else "0"*64
            results.append({"name": case["name"], "case_sha256": fingerprint(case), "median_elapsed_ns": 123,
                            "diagnostic": {"status": "verified", "fixture_kind": case["workload"]["kind"],
                                           "selected_and_output_sha256": identity, "total_fetch_misses": 0,
                                           "page_activity": [], "link_bytes": {}, "managed_peak_bytes_by_node": {},
                                           "residual_before_teardown_bytes_by_node": {}, "actions": [],
                                           "work_counts": self.work_counts},
                            "timings": [{"transfer_wall_ns": 14}]})
        return {"status": "completed", "result": {"scope": "cuda_text_derived_fixed_block_sparse", "feasible": True,
                                                      "score": 0.000123, "cases": results}}

    def close(self):
        pass


def _mock_config(output_directory):
    cases = {split: [{"name": split, "workload": {"kind": "rollout_cow", "content": f"PRIVATE_{split.upper()}"},
                     "compile": {"head_dim": 16}, "topology": {}, "target": {},
                     "compute_node": "gpu0", "device_map": {"gpu0": 0}}]
             for split in ("train", "validation", "test")}
    return {"output_directory": output_directory, "experiment_id": "cpu-mock",
            "source_manifest_sha256": "0"*64, "executor": {"gpu_uuids": ["TEST-DEVICE"]}, "cases": cases,
            "paired_block_ids": {split: [f"{split}-{index:02d}" for index in range(12)] for split in ("validation", "test")},
            "initial_program": policy("native-lru-equivalent"), "manual_program": policy("manual-topology-aware")}


def _measured_profile(directory):
    # Synthetic calibration records are test doubles only. Production uses measured raw evidence.
    uuids = ["TEST-DEVICE-A", "TEST-DEVICE-B"]
    physical = 1_000_000_000
    fits = [{"gpu_ordinal": ordinal, "direction": direction, "latency_ns": 100 + ordinal,
             "bandwidth_bytes_per_s": 10**9 + ordinal} for ordinal in range(2) for direction in ("h2d", "d2h")]
    raw = json.dumps({"schema": 1, "scope": "synchronized_directed_pinned_host_copy_not_attention_latency",
                      "devices": [{"ordinal": ordinal, "uuid": uid, "total_memory": physical}
                                  for ordinal, uid in enumerate(uuids)], "affine_link_fits": fits,
                      "host_cpu_affinity": [0], "torch_version": "test"}).encode()
    calibration = directory / "calibration.json"
    calibration.write_bytes(raw)
    links = []
    for row in fits:
        ordinal, direction = row["gpu_ordinal"], row["direction"]
        gpu = f"gpu{ordinal}"
        src, dst = ("host", gpu) if direction == "h2d" else (gpu, "host")
        links.append({"id": f"{gpu}-{direction}", "src": src, "dst": dst, "transport": "pinned_host", "bottlenecks": [],
                      "bandwidth_bytes_per_s": row["bandwidth_bytes_per_s"], "latency_ns": row["latency_ns"],
                      "evidence_sha256": sha256(raw).hexdigest()})
    topology = {"schema": 1, "nodes": [{"id": "gpu0", "kind": "gpu", "physical_bytes": physical, "capacity_bytes": 65536},
                                       {"id": "gpu1", "kind": "gpu", "physical_bytes": physical, "capacity_bytes": 4194304},
                                       {"id": "host", "kind": "host", "capacity_bytes": 4194304}],
                "links": links, "bottlenecks": []}
    profile = {"topology": topology, "target": {}, "compute_node": "gpu0", "device_map": {"gpu0": 0, "gpu1": 1},
               "compile": {"block_rows": 16, "head_dim": 16, "top_k": 4, "layers": 1, "text_bytes_per_row": 2}}
    return {"calibration": calibration, "executor": {"gpu_uuids": uuids},
            "profiles": {kind: deepcopy(profile) for kind in VARIANTS}}


class CudaDomainTests(unittest.TestCase):
    def test_worker_collects_diagnostic_counts_outside_timing(self):
        gpu = SimpleNamespace(device_count=lambda: 2, get_device_name=lambda ordinal: "mock",
                              get_device_properties=lambda ordinal: SimpleNamespace(uuid=f"GPU-{ordinal}"),
                              get_device_capability=lambda ordinal: (12, 0))
        fake_torch = SimpleNamespace(cuda=gpu, __version__="mock")
        case = _mock_config("unused")["cases"]["train"][0]
        payload = {"schema": 1, "arm": "native_lru", "program": None, "cases": [case], "warmup": 1, "samples": 2,
                   "expected_gpu_uuids": ["GPU-0", "GPU-1"]}
        prepared = SimpleNamespace(bind_scenario=lambda *args: object())
        calls = []
        leak_counts = False

        def trial(*args, verify, **kwargs):
            calls.append((verify, kwargs.get("collect_work", False)))
            result = {"status": "verified" if verify else "timed", "actions": [], "semantics": [], "accesses": [],
                      "link_bytes": {}, "program_sha256": None, "topology_sha256": "1"*64, "scenario_sha256": "2"*64,
                      "workload_sha256": "3"*64, "schedule_sha256": "4"*64, "managed_reserved_bytes_by_node": {},
                      "live_references_at_end": 0, "residual_before_teardown_bytes_by_node": {}, "elapsed_ns": 1234}
            if kwargs.get("collect_work") or leak_counts:
                result["work_counts"] = {"allocate/place": {"static_route_choices": 3}}
            return result

        with patch.dict(sys.modules, {"torch": fake_torch}), \
                patch.object(torch_eval.HardwareTopology, "from_dict"), patch.object(torch_eval.ScenarioWorkload, "from_dict"), \
                patch.object(torch_eval, "compile_workload", return_value=prepared), patch.object(torch_eval, "run_trial", new=trial):
            result = torch_eval.evaluate(payload)
            self.assertEqual(calls, [(True, True), (False, False), (False, False), (False, False)])
            self.assertEqual(result["score"], 1234/1e6)
            self.assertIn("work_counts", result["cases"][0]["diagnostic"])
            self.assertTrue(all("work_counts" not in row for row in result["cases"][0]["timings"]))
            leak_counts = True
            with self.assertRaisesRegex(RuntimeError, "timed CUDA trial emitted diagnostic work counts"):
                torch_eval.evaluate(payload)

    def test_williams_pair_order_balances_position_and_first_order_carryover(self):
        with TemporaryDirectory() as tmp:
            domain = CudaDomain(_mock_config("receipts"), Path(tmp))
            executor = domain.executor = _Executor()
            checked = domain.domain()
            for width in (2, 3, 4, 6):
                labels = [f"arm-{index}" for index in range(width)]
                variants = {label: None if index == 0 else checked.initial_program for index, label in enumerate(labels)}
                result = checked.evaluate_pair("validation", variants, f"pair-{width}")
                recorded = result[labels[0]]["metrics"]["execution_order"]
                orders = [row["order"] for row in recorded]
                self.assertEqual([row["block_id"] for row in recorded], domain.contract["paired_block_ids"]["validation"])
                for position in range(width):
                    self.assertEqual(Counter(order[position] for order in orders), dict.fromkeys(labels, 12//width))
                neighbors = Counter(pair for order in orders for pair in zip(order, order[1:]))
                self.assertEqual(neighbors, {(a, b): 12//width for a in labels for b in labels if a != b})
            self.assertEqual(len({call[0] for call in executor.calls}), len(executor.calls))
            domain.close()

    def test_training_context_and_feedback_do_not_export_heldout_content(self):
        with TemporaryDirectory() as tmp:
            domain = CudaDomain(_mock_config("receipts"), Path(tmp))
            executor = domain.executor = _Executor()
            checked = domain.domain()
            context = json.dumps(checked.training_context)
            self.assertNotIn("PRIVATE_", context)
            self.assertNotIn('"validation"', context)
            self.assertNotIn('"test"', context)
            self.assertTrue(checked.evaluate("train", None, "train")["feedback"])
            for split in ("validation", "test"):
                self.assertEqual(checked.evaluate(split, None, split)["feedback"], {})
                self.assertEqual(executor.calls[-1][2]["cases"], domain.cases[split])
            self.assertNotIn("write_route_hops", checked.training_context["feature_units"])
            domain.close()

    def test_pair_rejects_verified_but_different_selected_output_hash(self):
        with TemporaryDirectory() as tmp:
            domain = CudaDomain(_mock_config("receipts"), Path(tmp))
            executor = domain.executor = _Executor()
            executor.drift = True
            checked = domain.domain()
            with self.assertRaisesRegex(ExecutionError, "different selected bytes"):
                checked.evaluate_pair("test", {"native": None, "candidate": checked.initial_program}, "drift")
            self.assertEqual(len(executor.calls), 2)
            domain.close()

    def test_training_projects_only_bounded_diagnostic_work_counts(self):
        with TemporaryDirectory() as tmp:
            domain = CudaDomain(_mock_config("receipts"), Path(tmp))
            executor = domain.executor = _Executor()
            executor.work_counts = {"allocate/place": {"static_route_choices": 19, "elapsed_ns": 900},
                                    "access/evict": {"indexed_choices": 4, "rows_scanned": 12, "predicate_evals": -1},
                                    "access/replicate": {"choice_calls": True, "score_evals": 2**64},
                                    "unknown/private": {"rows_scanned": 18}}
            checked = domain.domain()
            train = checked.evaluate("train", checked.initial_program, "diagnostics")
            self.assertEqual(train["score"], 0.000123)
            self.assertEqual(train["feedback"]["cases"][0]["diagnostic_work_counts"],
                             {"allocate/place": {"static_route_choices": 19},
                              "access/evict": {"indexed_choices": 4, "rows_scanned": 12}})
            self.assertEqual(checked.evaluate("validation", checked.initial_program, "sealed")["feedback"], {})
            domain.close()

    def test_builder_pins_four_content_splits_and_rejects_unmeasured_bandwidth(self):
        with TemporaryDirectory() as tmp:
            directory = Path(tmp)
            inputs = {**_measured_profile(directory), "experiment_id": "test-rsi", "source_manifest_sha256": "0"*64,
                      "initial_program": policy("native-lru-equivalent"), "manual_program": policy("manual-topology-aware"),
                      "workload_directory": ROOT / "examples" / "workloads", "output_root": directory / "runs"}
            result = build_configs(**inputs)
            self.assertEqual(result, build_configs(**inputs))
            self.assertEqual(set(result), set(VARIANTS))
            for kind, config in result.items():
                fingerprints = {split: {row["workload"]["source"]["content_sha256"] for row in cases}
                                for split, cases in config["cases"].items()}
                self.assertEqual([len(fingerprints[split]) for split in ("train", "validation", "test")], [1, 2, 2])
                self.assertEqual(len(set.union(*fingerprints.values())), 5)
                self.assertEqual(len({row["compile"]["seed"] for cases in config["cases"].values() for row in cases}), 6)
                self.assertTrue(all(row["workload"]["kind"] == kind for cases in config["cases"].values() for row in cases))
            inputs["profiles"]["agent_tools"]["topology"]["links"][0]["bandwidth_bytes_per_s"] += 1
            with self.assertRaisesRegex(ValueError, "measured directed fit"):
                build_configs(**inputs)
