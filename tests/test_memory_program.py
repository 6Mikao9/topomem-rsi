from hashlib import sha256
import unittest

from rsi_memory.memory_program import MemoryProgram, MemoryScenario, generator_context, plan_backing
from rsi_memory.memory_program_reference import ReferenceExecutor
from rsi_memory.topology import HardwareTopology, fingerprint


def _hardware(host_cap=32):
    def link(src, dst, name):
        return {"id": name, "src": src, "dst": dst, "bandwidth_bytes_per_s": 1_000_000_000,
                "latency_ns": 10, "bottlenecks": ["pcie"], "transport": "host-dma"}

    return {"schema": 1, "nodes": [{"id": "gpu0", "kind": "gpu", "capacity_bytes": 16, "physical_bytes": 64},
                                   {"id": "gpu1", "kind": "gpu", "capacity_bytes": 16, "physical_bytes": 64},
                                   {"id": "host", "kind": "host", "capacity_bytes": host_cap, "physical_bytes": 64}],
            "links": [link("gpu0", "host", "d2h0"), link("host", "gpu0", "h2d0"),
                      link("gpu1", "host", "d2h1"), link("host", "gpu1", "h2d1")],
            "bottlenecks": [{"id": "pcie", "bandwidth_bytes_per_s": 100_000_000}]}


def _scenario(events=None):
    return MemoryScenario.from_dict({"schema": 1, "target_sha256": "a" * 64, "schedule_sha256": "b" * 64,
                                     "workload_sha256": fingerprint(events) if events is not None else "c" * 64,
                                     "rank_placement": [{"rank": 0, "node": "gpu0"}, {"rank": 1, "node": "gpu1"}],
                                     "attention": {"family": "dsa", "selector": "torch-unfused", "selection_unit": "token",
                                                   "top_k": 2, "kv_row_bytes": 4}})


def _const(value):
    return {"op": "const", "value": value}


def _feature(name):
    return {"op": "feature", "name": name}


def _rule(event, action, score, when=None):
    return {"event": event, "action": action, "score": score, "when": _const(1) if when is None else when}


def _program(*rules):
    return MemoryProgram.from_dict({"schema": 1, "kind": "kv_memory_program", "rules": list(rules)})


def _key(rank=0, block=0):
    return {"rank": rank, "request": "rsi-r0", "generation": 1, "layer": 0, "block": block}


def _alloc(key, raw, when=0, allowed=None):
    return {"event": "allocate", "time_ns": when, "key": key, "payload_hex": raw.hex(),
            "allowed_nodes": allowed if allowed is not None else [f"gpu{key['rank']}", "host"]}


def _access(rank, rows, when, hold=0):
    return {"event": "access", "time_ns": when, "rank": rank, "hold_ns": hold,
            "selected": [{"key": key, "sha256": sha256(raw).hexdigest()} for key, raw in rows]}


def _lifecycle(kind, keys, when):
    return {"event": kind, "time_ns": when, "keys": keys}


class HardwareTopologyTest(unittest.TestCase):
    def test_routes_are_directional_and_bind_shared_capacity(self):
        topology = HardwareTopology.from_dict(_hardware())
        (forward,) = topology.routes("gpu0", "gpu1")
        (reverse,) = topology.routes("gpu1", "gpu0")
        self.assertEqual(forward.ids, ("d2h0", "h2d1"))
        self.assertEqual(reverse.ids, ("d2h1", "h2d0"))
        self.assertEqual(topology.link_bandwidth(forward.links[0]), 100_000_000)
        self.assertFalse(topology.routes("gpu0", "gpu1", transports={"cuda-p2p"}))
        reordered = _hardware()
        reordered["links"].reverse()
        self.assertEqual(HardwareTopology.from_dict(reordered).fingerprint(), topology.fingerprint())

    def test_invalid_capacity_and_missing_bottleneck_fail(self):
        broken = _hardware(65)
        with self.assertRaisesRegex(ValueError, "exceeds its physical capacity"):
            HardwareTopology.from_dict(broken)
        broken = _hardware()
        broken["links"][0]["bottlenecks"] = ["invented"]
        with self.assertRaisesRegex(ValueError, "invalid hardware resources"):
            HardwareTopology.from_dict(broken)


class MemoryProgramTest(unittest.TestCase):
    def test_generated_slab_policy_changes_rank_tiers_under_shared_capacity(self):
        topology, scenario = HardwareTopology.from_dict(_hardware(12)), _scenario()
        slabs = [{"key": f"rank{rank}-layer0", "rank": rank, "layer": 0, "bytes": 8,
                  "allowed_nodes": [f"gpu{rank}", "host"]} for rank in range(2)]
        host_first = _program(_rule("allocate", "place", _feature("target_is_gpu")))
        rank_priority = {"op": "if_lt", "args": [_feature("rank"), _const(1),
                                                      _feature("target_is_host"), _feature("target_is_gpu")]}
        mixed = _program(_rule("allocate", "place", rank_priority))
        first = plan_backing(topology, scenario, host_first, slabs)
        second = plan_backing(topology, scenario, mixed, slabs)
        self.assertEqual([action["node"] for action in first["actions"]], ["host", "gpu1"])
        self.assertEqual([action["node"] for action in second["actions"]], ["gpu0", "host"])
        self.assertEqual(second["actions"][1]["read_links"], ["h2d1"])
        self.assertEqual(second["actions"][1]["write_links"], ["d2h1"])
        self.assertEqual(second["used_bytes_by_node"], {"gpu0": 8, "gpu1": 0, "host": 8})
        feedback = [{"metric": "gpu0_peak_bytes", "value": 8, "units": "bytes"}]
        context = generator_context(topology, scenario, feedback, supported_actions={('allocate', 'place')})
        self.assertEqual(context["measured_feedback"], feedback)
        self.assertEqual(context["program"]["events_and_actions"]["access"], [])

    def test_adapter_refuses_unevaluated_generated_action_and_scheduler_features(self):
        topology, scenario = HardwareTopology.from_dict(_hardware()), _scenario()
        dynamic = _program(_rule("allocate", "place", _const(0)), _rule("idle", "reclaim", _const(0)))
        with self.assertRaisesRegex(ValueError, "adapter cannot execute"):
            plan_backing(topology, scenario, dynamic, [])
        with self.assertRaisesRegex(ValueError, "unsupported program feature"):
            _program(_rule("allocate", "place", _feature("batch_admission")))
        with self.assertRaisesRegex(ValueError, "unsupported event/action"):
            _program(_rule("allocate", "place", _const(0)), _rule("access", "reorder_requests", _const(0)))

    def test_async_prefetch_changes_wait_and_preserves_ordered_bytes(self):
        first, second, left, right = _key(0), _key(1), b"rank", b"RANK"
        events = [_alloc(first, left), _alloc(second, right), _lifecycle("hint", [first, second], 200),
                  _access(0, [(first, left)], 210), _access(1, [(second, right)], 210),
                  _lifecycle("release", [first, second], 400)]
        topology, scenario = HardwareTopology.from_dict(_hardware()), _scenario(events)
        base = (_rule("allocate", "place", _feature("target_is_gpu")), _rule("access", "replicate", _const(0)),
                _rule("release", "reclaim", _feature("age_ns")))
        demand = ReferenceExecutor(topology, scenario, _program(*base)).run(events)
        prefetch = ReferenceExecutor(topology, scenario, _program(*base, _rule("hint", "prefetch", _const(0)))).run(events)
        self.assertEqual(prefetch["ordered_selected_sha256"], demand["ordered_selected_sha256"])
        self.assertEqual((prefetch["demand_misses"], demand["demand_misses"]), (0, 2))
        self.assertLess(prefetch["simulated_access_wait_ns"], demand["simulated_access_wait_ns"])
        copies = [action for action in prefetch["actions"] if action.get("mode") == "prefetch"]
        self.assertEqual(copies[0]["stages"][0]["finish_ns"], copies[1]["stages"][0]["start_ns"])
        self.assertEqual(prefetch["shared_bottleneck_bytes"], {"pcie": 16})
        self.assertEqual(prefetch["residual_before_teardown_bytes_by_node"], {"gpu0": 0, "gpu1": 0, "host": 0})

    def test_staged_route_reserves_host_and_copy_corruption_fails(self):
        key, raw = _key(), b"row0"
        events = [_alloc(key, raw, allowed=["gpu1"]), _access(0, [(key, raw)], 200), _lifecycle("release", [key], 400)]
        scenario = _scenario(events)
        program = _program(_rule("allocate", "place", _const(0)), _rule("access", "replicate", _const(0)))
        topology = HardwareTopology.from_dict(_hardware(4))
        result = ReferenceExecutor(topology, scenario, program).run(events)
        copies = [action for action in result["actions"] if action["action"] == "copy"]
        self.assertEqual(copies[0]["route_links"], ["d2h0", "h2d1"])
        self.assertEqual(copies[0]["staging_nodes"], ["host"])
        self.assertEqual(result["peak_bytes_by_node"]["host"], 4)
        with self.assertRaisesRegex(MemoryError, "owner placement"):
            ReferenceExecutor(HardwareTopology.from_dict(_hardware(3)), scenario, program).run(events)
        with self.assertRaisesRegex(RuntimeError, "copied payload disagrees"):
            ReferenceExecutor(topology, scenario, program, copy_payload=lambda action, row: row[::-1]).run(events)

    def test_retire_waits_for_reader_and_generation_cannot_be_reused(self):
        key, raw = _key(), b"row0"
        events = [_alloc(key, raw), _access(0, [(key, raw)], 1, hold=100), _lifecycle("release", [key], 2),
                  {"event": "idle", "time_ns": 3}, {"event": "idle", "time_ns": 102}]
        topology, scenario = HardwareTopology.from_dict(_hardware()), _scenario(events)
        program = _program(_rule("allocate", "place", _feature("target_is_host")), _rule("release", "reclaim", _const(0)),
                           _rule("idle", "reclaim", _const(0)), _rule("idle", "evict", _const(0)))
        result = ReferenceExecutor(topology, scenario, program).run(events)
        reclaims = [action for action in result["actions"] if action["action"] == "reclaim"]
        self.assertEqual([action["time_ns"] for action in reclaims], [102])
        duplicate = [*events, _alloc(key, raw, 103)]
        with self.assertRaisesRegex(ValueError, "published more than once"):
            ReferenceExecutor(topology, _scenario(duplicate), program).run(duplicate)
        with self.assertRaisesRegex(ValueError, "does not match"):
            ReferenceExecutor(topology, scenario, program).run(duplicate)

    def test_wrong_rank_is_rejected_even_if_payload_matches(self):
        key, raw = _key(), b"row0"
        events = [_alloc(key, raw), _access(1, [(key, raw)], 1), _lifecycle("release", [key], 2)]
        program = _program(_rule("allocate", "place", _feature("target_is_host")))
        with self.assertRaisesRegex(ValueError, "rank-local KV identity"):
            ReferenceExecutor(HardwareTopology.from_dict(_hardware()), _scenario(events), program).run(events)
