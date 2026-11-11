import unittest
from unittest.mock import patch

from integrations.topology_memory.torch_runtime import InfeasiblePlan, _Runtime
from rsi_memory.memory_program import MemoryProgram
from test_torch_workload import fixture, generated_lru, topology


def policy(*, demotion=True, existing_copy=False, peer=False):
    f = lambda name: {"op": "feature", "name": name}
    c = lambda value: {"op": "const", "value": value}
    binary = lambda op, left, right: {"op": op, "args": [left, right]}
    local = f("target_is_local")
    target = binary("mul", f("target_is_gpu"), binary("sub", c(1), local)) if peer else f("target_is_host")
    direction = binary("mul", f("source_is_local"), target)
    rules = [{"event": "allocate", "action": "place", "when": local, "score": c(0)},
             {"event": "access", "action": "replicate", "when": local, "score": f("route_transfer_ns")},
             {"event": "release", "action": "reclaim", "when": c(1), "score": c(0)}]
    if existing_copy:
        rules.append({"event": "allocate", "action": "replicate", "when": target, "score": f("route_transfer_ns")})
    for event in ("allocate", "access"):
        rules.append({"event": event, "action": "evict", "when": local, "score": f("last_use_step")})
        if demotion:
            rules.append({"event": event, "action": "demote", "when": direction, "score": f("last_use_step")})
    return MemoryProgram.from_dict({"schema": 1, "kind": "kv_memory_program", "rules": rules})


class PressureDemotionTests(unittest.TestCase):
    def test_live_owner_spill_and_handoff_preserve_sparse_attention(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch is unavailable")
        packed = fixture("rollout-bridge-options", block_rows=8, layers=2)
        graph = topology(packed.block_bytes)
        scenario = packed.bind_scenario("1"*64, "gpu0")
        devices = {node.id: torch.device("cpu") for node in graph.nodes}

        def run(program, arm="generated", *, collect_work=False):
            return _Runtime(torch, graph, scenario, packed, program, arm, devices, True,
                            pin_host=False, collect_work=collect_work).run()

        with torch.inference_mode():
            with self.assertRaisesRegex(InfeasiblePlan, "compute producer"):
                run(policy(demotion=False))
            native, manual = run(None, "native_lru"), run(None, "manual_topology")
            spilled, handed = run(policy(peer=True)), run(policy(existing_copy=True))
            counted = run(policy(peer=True), collect_work=True)
            with self.assertRaisesRegex(ValueError, "only during verified diagnostics"):
                _Runtime(torch, graph, scenario, packed, policy(), "generated", devices, False,
                         pin_host=False, collect_work=True)

            original = _Runtime._demote_owner

            def corrupt_existing_copy(runtime, state, route, rule, event):
                if not route.links:
                    runtime._tensor(state, route.nodes[-1]).flatten()[0] += 1
                return original(runtime, state, route, rule, event)

            with patch.object(_Runtime, "_demote_owner", new=corrupt_existing_copy), \
                    self.assertRaisesRegex(RuntimeError, "corrupt replica bytes"):
                run(policy(existing_copy=True))
        identity = lambda report: [(row["selected_sha256"], row["output_sha256"]) for row in report["accesses"]]
        for report in (manual, spilled, handed):
            self.assertEqual(identity(report), identity(native))
            self.assertEqual(report["live_references_at_end"], 0)
        demotions = [row for row in spilled["actions"] if row["action"] == "demote"]
        self.assertEqual({row["pressure_event"] for row in demotions}, {"allocate", "access"})
        self.assertTrue(all(row["links"] == ["gpu0-host", "host-gpu1"] for row in demotions))
        handoffs = [row for row in handed["actions"] if row["action"] == "demote_owner_handoff"]
        self.assertTrue(handoffs)
        self.assertTrue(all(not row["links"] for row in handoffs))
        self.assertFalse(any(row["action"] == "demote" for row in handed["actions"]))
        self.assertEqual(counted["actions"], spilled["actions"])
        self.assertEqual(identity(counted), identity(spilled))
        self.assertNotIn("work_counts", spilled)
        work = counted["work_counts"]["allocate/demote"]
        self.assertGreater(work["rows_scanned"], 0)
        self.assertGreater(work["indexed_choices"], 0)
        self.assertNotIn("predicate_evals", work)
        self.assertNotIn("score_evals", work)

    def test_lowered_affine_keys_keep_rounding_ties_and_error_fallback(self):
        try:
            import torch
        except ImportError:
            self.skipTest("optional Torch is unavailable")
        packed = fixture("rollout-bridge-options", block_rows=8, layers=2)
        graph = topology(packed.block_bytes)
        scenario = packed.bind_scenario("1"*64, "gpu0")
        devices = {node.id: torch.device("cpu") for node in graph.nodes}

        def runner(data):
            return _Runtime(torch, graph, scenario, packed, MemoryProgram.from_dict(data), "generated",
                            devices, True, pin_host=False, collect_work=True)

        affine = policy(existing_copy=True).to_dict()
        generic = policy(existing_copy=True).to_dict()
        for indexed, oracle in zip(affine["rules"], generic["rules"], strict=True):
            if indexed["action"] in ("demote", "evict"):
                indexed["score"] = {"op": "add", "args": [{"op": "const", "value": 2**53},
                                                             {"op": "feature", "name": "page_accesses"}]}
                oracle["score"] = {"op": "div", "args": [indexed["score"], {"op": "const", "value": 1}]}
        with torch.inference_mode():
            fast, slow = runner(affine), runner(generic)
            self.assertIn(("allocate", "demote"), fast.pressure_keys)
            self.assertNotIn(("allocate", "demote"), slow.pressure_keys)
            fast_result, slow_result = fast.run(), slow.run()
            self.assertEqual(fast_result["actions"], slow_result["actions"])
            self.assertEqual(fast_result["link_bytes"], slow_result["link_bytes"])
            self.assertEqual([a["output_sha256"] for a in fast_result["accesses"]],
                             [a["output_sha256"] for a in slow_result["accesses"]])

            invalid = generated_lru().to_dict()
            for row in invalid["rules"]:
                if (row["event"], row["action"]) == ("allocate", "evict"):
                    row["when"] = {"op": "mul", "args": [{"op": "feature", "name": "target_capacity_bytes"},
                                                            {"op": "const", "value": 1e308}]}
            bad = runner(invalid)
            self.assertNotIn(("allocate", "evict"), bad.pressure_keys)
            with self.assertRaisesRegex(ValueError, "finite real"):
                bad.run()
