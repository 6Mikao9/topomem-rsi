from hashlib import sha256
from pathlib import Path
import unittest
from unittest.mock import patch

from integrations.topology_memory.torch_eval import execution_signature
from integrations.topology_memory import torch_runtime
from integrations.topology_memory.torch_runtime import SUPPORTED_FEATURES, _Runtime
from integrations.topology_memory.torch_workload import compile_workload
from rsi_memory.compiled_memory import CompiledProgram, compile_scalar
from rsi_memory.memory_program import BASE_FEATURES, FEATURE_UNITS, MemoryProgram
from rsi_memory.scalar_program import evaluate_scalar, validate_program
from rsi_memory.scenarios import ScenarioWorkload
from rsi_memory.topology import HardwareTopology


WORKLOADS = Path(__file__).resolve().parents[1] / "examples" / "workloads"


def fixture(name, **geometry):
    return compile_workload(ScenarioWorkload.read(WORKLOADS / f"{name}.json"), **geometry)


def topology(block_bytes):
    return HardwareTopology.from_dict({"schema": 1, "nodes": [
        {"id": "gpu0", "kind": "gpu", "capacity_bytes": block_bytes*8},
        {"id": "gpu1", "kind": "gpu", "capacity_bytes": block_bytes*64},
        {"id": "host", "kind": "host", "capacity_bytes": block_bytes*64},
    ], "bottlenecks": [{"id": "pcie", "bandwidth_bytes_per_s": 10**10}], "links": [
        {"id": f"{src}-{dst}", "src": src, "dst": dst, "bandwidth_bytes_per_s": 10**10,
         "latency_ns": 100, "bottlenecks": ["pcie"], "transport": "pinned_host"}
        for src, dst in (("gpu0", "host"), ("host", "gpu0"), ("gpu1", "host"), ("host", "gpu1"))]})


def generated():
    always = {"op": "const", "value": 1}
    return MemoryProgram.from_dict({"schema": 1, "kind": "kv_memory_program", "rules": [
        {"event": "allocate", "action": "place", "when": {"op": "feature", "name": "route_hops"},
         "score": {"op": "feature", "name": "target_index"}},
        {"event": "access", "action": "replicate", "when": {"op": "feature", "name": "target_is_local"}, "score": always},
        {"event": "hint", "action": "prefetch", "when": always, "score": always},
        *({"event": event, "action": "evict", "when": always, "score": always} for event in ("allocate", "access")),
        {"event": "release", "action": "reclaim", "when": always, "score": always},
    ]})


def generated_lru():
    always = {"op": "const", "value": 1}
    local = {"op": "feature", "name": "target_is_local"}
    path_cost = {"op": "feature", "name": "route_transfer_ns"}
    recency = {"op": "feature", "name": "last_use_step"}
    return MemoryProgram.from_dict({"schema": 1, "kind": "kv_memory_program", "rules": [
        {"event": "allocate", "action": "place", "when": {"op": "feature", "name": "target_is_host"}, "score": path_cost},
        {"event": "allocate", "action": "replicate", "when": local, "score": always},
        {"event": "access", "action": "replicate", "when": local, "score": path_cost},
        *({"event": event, "action": "evict", "when": always, "score": recency} for event in ("allocate", "access")),
        {"event": "release", "action": "reclaim", "when": always, "score": always},
    ]})


class TorchWorkloadTests(unittest.TestCase):
    def test_all_workloads_pin_attention_queries_and_actual_payload(self):
        for path in sorted(WORKLOADS.glob("*.json")):
            source = ScenarioWorkload.read(path)
            case = compile_workload(source, layers=2)
            self.assertEqual(case, compile_workload(source, layers=2))
            self.assertNotEqual(case.workload_sha256, compile_workload(source, layers=2, seed=1).workload_sha256)
            for block in case.blocks:
                self.assertEqual(len(block.payload), case.block_bytes)
                self.assertEqual(sha256(block.payload).hexdigest(), block.digest)
            for request in (r for step in case.steps for r in step.attention):
                self.assertEqual(len(request.query), 4*case.head_dim)
                self.assertEqual(len(set(request.selected)), min(case.top_k, len(request.blocks)))
                self.assertTrue(set(request.selected) <= set(request.blocks))
            requested = sum(event["max_new_tokens"] for event in source.to_dict()["events"] if event["op"] == "infer")
            self.assertEqual(sum(len(step.attention) for step in case.steps), requested * case.layers)
            self.assertTrue(all(step.writes for step in case.steps if step.op in ("decode", "commit")))

    def test_rollout_compiler_rejects_no_tail_cow_geometry(self):
        with self.assertRaisesRegex(ValueError, "partial tail"):
            fixture("rollout-bridge-options", block_rows=49)

    def test_compiled_program_preserves_first_rule_ties_lazy_branch_and_domain(self):
        policy = generated_lru()
        policy.validate_features(SUPPORTED_FEATURES)
        with self.assertRaisesRegex(ValueError, "last_use_step"):
            policy.validate_features(BASE_FEATURES)
        candidates = [("a", {"last_use_step": 3}), ("b", {"last_use_step": 1}), ("c", {"last_use_step": 1})]
        self.assertEqual(CompiledProgram(policy).choose("allocate", "evict", iter(candidates)),
                         policy.choose("allocate", "evict", candidates))
        tree = {"op": "if_lt", "args": [{"op": "const", "value": 0}, {"op": "const", "value": 1},
                {"op": "div", "args": [{"op": "const", "value": 2}, {"op": "const", "value": -0.0}]},
                {"op": "feature", "name": "route_hops"}]}
        self.assertEqual(compile_scalar(tree)({}), evaluate_scalar(validate_program(tree, FEATURE_UNITS), {}))

    def test_actual_torch_cpu_storage_exercises_runtime_without_cuda_claim(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch is unavailable")
        case = fixture("rollout-bridge-options", block_rows=8, layers=2)
        graph = topology(case.block_bytes)
        scenario = case.bind_scenario("1"*64, "gpu0")
        devices = {node.id: torch.device("cpu") for node in graph.nodes}
        with torch.inference_mode():
            runs = [_Runtime(torch, graph, scenario, case, program, arm, devices, True, pin_host=False).run()
                    for arm, program in (("native_lru", None), ("manual_topology", None), ("generated", generated()),
                                         ("generated", generated_lru()))]
        identity = [[(access["selected_sha256"], access["output_sha256"]) for access in run["accesses"]] for run in runs]
        self.assertEqual(identity[0], identity[1])
        self.assertEqual(identity[0], identity[2])
        self.assertEqual(identity[0], identity[3])
        for run in runs:
            self.assertEqual(run["live_references_at_end"], 0)
            self.assertEqual({row["session"] for row in run["semantics"] if row["action"] == "copy_on_write"}, {"north", "south"})
            self.assertEqual(run["managed_reserved_bytes_by_node"], runs[0]["managed_reserved_bytes_by_node"])
        self.assertGreater(runs[2]["link_bytes"]["host-gpu1"], 0)
        self.assertGreater(runs[2]["link_bytes"]["gpu1-host"], 0)
        self.assertTrue(any(row["links"] == ["gpu1-host", "host-gpu0"] for row in runs[2]["actions"] if "links" in row))
        self.assertEqual(runs[0]["link_bytes"], runs[3]["link_bytes"])
        self.assertEqual(runs[0]["total_fetch_misses"], runs[3]["total_fetch_misses"])

    def test_deferred_teardown_is_timed_and_preserves_verified_execution(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch is unavailable")
        case = fixture("rollout-bridge-options")
        raw = topology(case.block_bytes).to_dict()
        for node in raw["nodes"]:
            node["capacity_bytes"] = case.block_bytes * (len(case.blocks)+2)
        graph = HardwareTopology.from_dict(raw)
        scenario = case.bind_scenario("1"*64, "gpu0")
        devices = {node.id: torch.device("cpu") for node in graph.nodes}
        policy = generated_lru().to_dict()
        policy["rules"] = [row for row in policy["rules"] if row["event"] != "release"]
        program = MemoryProgram.from_dict(policy)
        clock, drop_cost = [0], 7
        original_drop = _Runtime._drop

        def charge_teardown(runtime, state, node, action, rule=None):
            if action == "teardown":
                clock[0] += drop_cost
            return original_drop(runtime, state, node, action, rule)

        with torch.inference_mode():
            verified = _Runtime(torch, graph, scenario, case, program, "generated", devices, True, pin_host=False).run()
            with patch.object(torch_runtime, "perf_counter_ns", side_effect=lambda: clock[0]), \
                    patch.object(_Runtime, "_drop", new=charge_teardown):
                timed = _Runtime(torch, graph, scenario, case, program, "generated", devices, False, pin_host=False).run()
        copies = sum(timed["residual_before_teardown_bytes_by_node"].values()) // case.block_bytes
        self.assertGreater(copies, 0)
        self.assertEqual(sum(row["action"] == "teardown" for row in timed["actions"]), copies)
        self.assertEqual(timed["elapsed_ns"], copies*drop_cost)
        binding = {"program_sha256": program.fingerprint(), "topology_sha256": graph.fingerprint(),
                   "scenario_sha256": scenario.fingerprint()}
        self.assertEqual(execution_signature(verified | binding), execution_signature(timed | binding))
        timed["actions"][-1] = {**timed["actions"][-1], "node": "unexpected"}
        self.assertNotEqual(execution_signature(verified | binding), execution_signature(timed | binding))
