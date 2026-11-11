import copy
from dataclasses import replace
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from integrations.topology_memory import sglang_backing as backing
from rsi_memory.memory_program import MemoryProgram, MemoryScenario, plan_backing
from rsi_memory.topology import HardwareTopology


ROWS = 8192
LAYER_BYTES = ROWS * 656
MEASUREMENT = "a" * 64


def _fixture():
    links = []
    for name, source, destination, transport in (
        ("h2d", "host", "gpu-a", "pinned_h2d"), ("d2h", "gpu-a", "host", "pinned_d2h"),
        ("p2p-a", "gpu-a", "gpu-b", "cuda_p2p"), ("p2p-b", "gpu-b", "gpu-a", "cuda_p2p"),
    ):
        links.append({"id": name, "src": source, "dst": destination, "transport": transport,
                      "bandwidth_bytes_per_s": 1, "latency_ns": 1, "bottlenecks": [], "evidence_sha256": MEASUREMENT})
    topology = HardwareTopology.from_dict({"schema": 1, "nodes": [
        {"id": node, "kind": kind, "capacity_bytes": LAYER_BYTES}
        for node, kind in (("gpu-a", "gpu"), ("gpu-b", "gpu"), ("host", "host"))], "links": links, "bottlenecks": []})
    scenario = MemoryScenario.from_dict({"schema": 1, "rank_placement": [{"rank": 0, "node": "gpu-a"}],
        "attention": {"family": "dsa", "selector": "torch-unfused", "selection_unit": "token", "top_k": 2048, "kv_row_bytes": 656},
        "target_sha256": MEASUREMENT, "workload_sha256": MEASUREMENT, "schedule_sha256": MEASUREMENT})
    program = MemoryProgram.from_dict({"schema": 1, "kind": "kv_memory_program", "rules": [
        {"event": "allocate", "action": "place", "when": {"op": "const", "value": 1}, "score": {"op": "const", "value": 0}}]})
    slabs = [{"key": f"layer-{layer}", "rank": 0, "layer": layer, "bytes": LAYER_BYTES, "allowed_nodes": [node]}
             for layer, node in enumerate(("gpu-b", "host", "gpu-a"))]
    plan = plan_backing(topology, scenario, program, slabs, transports=backing.NATIVE_TRANSPORTS)
    return topology, scenario, plan


def _bind(topology, scenario, plan):
    return backing.bind_plans(topology, scenario, plan, gpu_ordinals={"gpu-a": 0, "gpu-b": 1},
                              rows=ROWS, page_size=64, layers=3, hot_slots=4096)


class TieredBackingTests(unittest.TestCase):
    def test_generated_reservations_bind_actual_tiers_and_bytes(self):
        (worker,) = _bind(*_fixture())
        self.assertEqual(worker.layer_tiers, (backing.PEER_HBM, backing.HOST, backing.LOCAL_HBM))
        self.assertEqual((worker.compute_device, worker.peer_device), (0, 1))
        self.assertEqual((worker.pinned_host_bytes, worker.local_hbm_bytes, worker.peer_hbm_bytes), (LAYER_BYTES,) * 3)
        with self.assertRaisesRegex(ValueError, "tier cap"):
            replace(worker, peer_hbm_cap_bytes=LAYER_BYTES - 1)

    def test_global_rank_one_can_have_a_local_attention_tp_group_of_one(self):
        (worker,) = _bind(*_fixture())
        worker = replace(worker, world_size=2, rank=1, attention_tp_size=1, compute_device=1, peer_device=0)
        coordinator = SimpleNamespace(tp_world_size=1, is_dsv4_hisparse=False, enable_prefetch=False, skip_io=False,
                                      compress_ratio=1, item_size_bytes=656, page_size=64, top_k=2048, device_buffer_size=4096,
                                      mem_pool_host=SimpleNamespace(layout="layer_first", layer_num=3, size=ROWS))
        backing._check_worker_layout(coordinator, worker)
        with self.assertRaisesRegex(ValueError, "synchronous ordinary MLA"):
            backing._check_worker_layout(coordinator, replace(worker, attention_tp_size=2))

    def test_wrong_link_node_or_bytes_are_rejected(self):
        topology, scenario, plan = _fixture()
        for change, message in (({"read_links": ["p2p-a"]}, "measured topology link"),
                                ({"write_links": ["d2h", "p2p-a"]}, "two measured"),
                                ({"layer": 1}, "duplicate"), ({"bytes": LAYER_BYTES - 1}, "slab")):
            with self.subTest(change=change):
                bad = copy.deepcopy(plan)
                bad["actions"][0].update(change)
                with self.assertRaisesRegex(ValueError, message):
                    _bind(topology, scenario, bad)
        bad = {**plan, "used_bytes_by_node": {}}
        with self.assertRaisesRegex(ValueError, "totals"):
            _bind(topology, scenario, bad)

    def test_staged_peer_routes_and_scratch_are_bound_to_both_devices_and_host(self):
        topology, scenario, plan = _fixture()
        graph = topology.to_dict()
        graph["links"] = [link for link in graph["links"] if link["transport"] != "cuda_p2p"]
        for edge in (dict(graph["links"][0], id="d2h-b", src="gpu-b"), dict(graph["links"][1], id="h2d-b", dst="gpu-b")):
            graph["links"].append(edge)
        for node in graph["nodes"]:
            node["capacity_bytes"] += backing.workspace_bytes(2048, 656, 2)[{"gpu-a": "compute", "gpu-b": "peer", "host": "host"}[node["id"]]]
        staged = HardwareTopology.from_dict(graph)
        plan["topology_sha256"] = staged.fingerprint()
        plan["actions"][0].update(read_links=["d2h-b", "h2d"], write_links=["d2h", "h2d-b"])
        (worker,) = _bind(staged, scenario, plan)
        self.assertEqual(worker.layer_tiers, (backing.PEER_RELAY, backing.HOST, backing.LOCAL_HBM))
        self.assertGreater(worker.relay_workspace_bytes["peer"], 0)
        self.assertGreater(worker.relay_workspace_bytes["host"], 0)
        with self.assertRaisesRegex(ValueError, "tier cap"):
            replace(worker, peer_hbm_cap_bytes=LAYER_BYTES)

    def test_pool_keeps_original_pointer_builder_outside_startup_and_allocates_real_destinations(self):
        (plan,) = _bind(*_fixture())
        torch = Mock()
        torch.device.side_effect = lambda kind, index=None: SimpleNamespace(type=kind, index=index)
        torch.cuda.current_device.return_value = 0
        torch.uint8 = object()
        torch.empty.side_effect = lambda shape, **kwargs: SimpleNamespace(shape=shape, **kwargs)
        pointer_table = Mock()

        class Native:
            def __init__(self):
                self.layout, self.size, self.layer_num, self.kv_cache_dim = "layer_first", ROWS, 3, 656
                self.dtype, self.dcp_size, self.mtp_draft_device_pools = torch.uint8, 1, ()
                self._is_dummy, self.pin_memory = False, True
                self.device_pool = SimpleNamespace(kv_buffer=[SimpleNamespace(device=torch.device("cuda", 0))], layer_shard_enabled=False)
                self.kv_buffer = self.init_kv_buffer()
                self.data_refs = list(self.kv_buffer)
                self.data_ptrs = module.make_kernel_ptr_table(self.data_refs, "cuda:0", host_memory_registered=True)

        module = SimpleNamespace(MLATokenToKVPoolHost=Native, make_kernel_ptr_table=pointer_table)
        with (patch.object(backing, "enable_direct_peer") as enable,
              patch.object(backing, "_pointer_table", return_value="mixed-pointers") as pointers):
            pool = backing._tiered_pool_type(module, torch, plan)()
        self.assertIs(module.make_kernel_ptr_table, pointer_table)
        self.assertEqual(pool.data_ptrs, "mixed-pointers")
        self.assertEqual([(t.device.type, t.device.index, t.pin_memory) for t in pool.kv_buffer],
                         [("cuda", 1, False), ("cpu", None, True), ("cuda", 0, False)])
        enable.assert_called_once_with(torch, 0, 1)
        pointers.assert_called_once_with(torch, pointer_table, pool.data_refs, "cuda:0")
        pool.destroy()
        pool.destroy()
        torch.cuda.synchronize.assert_called_once()
        self.assertIsNone(pool.kv_buffer)


if __name__ == "__main__":
    unittest.main()
