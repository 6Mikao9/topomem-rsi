from array import array
from copy import deepcopy
from enum import IntEnum
import json
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from integrations.topology_memory import sglang_radix_probe as probe
from integrations.topology_memory.sglang_radix_probe import _proof, _read_radix_mapping, _validate, _worker
from rsi_memory.topology import fingerprint


class _ComponentType(IntEnum):
    FULL = 0
    SWA = 1


class _Slots(list):
    def __getitem__(self, key):
        result = super().__getitem__(key)
        return _Slots(result) if isinstance(key, slice) else result

    def tolist(self):
        return list(self)


class NativeRadixProofTests(unittest.TestCase):
    def test_worker_hashes_native_array_and_publishes_only_complete_rank_receipt(self):
        requests = [SimpleNamespace(rid=rid, kv=SimpleNamespace(req_pool_idx=0), extend_range=SimpleNamespace(end=2),
                                    origin_input_ids=array("q", [1, token]), prefix_indices=[], finished=lambda: True,
                                    extra_key=None, cache_salt=None)
                    for rid, token in (("root", 2), ("left", 3), ("right", 4))]
        batch = SimpleNamespace(reqs=requests, forward_mode=SimpleNamespace(is_extend=lambda: True, name="EXTEND"))

        class UnifiedTreeCore:
            is_eagle = enable_hicache = False
            page_size = 2
            root_node = None

            def match_full_device_prefix(self, key):
                return 0, 0, 0

            def node_by_id(self, node_id):
                return self.root_node

        class FullComponent:
            pass

        class UnifiedRadixCache:
            disable = enable_session_radix_cache = False
            page_size = 2
            tree_core = UnifiedTreeCore()
            linker = cache_controller = None
            tree_components = (_ComponentType.FULL,)
            components = {_ComponentType.FULL: FullComponent()}

        class Scheduler:
            tree_cache = UnifiedRadixCache()
            enable_hisparse = False
            req_to_token_pool = SimpleNamespace(req_to_token=MagicMock())
            req_to_token_pool.req_to_token.__getitem__.return_value = _Slots([20, 21])

            def run_batch(self, batch):
                pass

            def process_batch_result(self, batch):
                pass

        def run_scheduler_process(tp_rank, gpu_id):
            scheduler = Scheduler()
            scheduler.run_batch(batch)
            scheduler.process_batch_result(batch)

        modules = {
            "sglang.srt.managers.scheduler": SimpleNamespace(Scheduler=Scheduler, run_scheduler_process=run_scheduler_process),
            "sglang.srt.mem_cache.radix_cache": SimpleNamespace(RadixKey=lambda tokens, *args, **kwargs: tokens),
            "sglang.srt.mem_cache.unified_cache.component_type": SimpleNamespace(ComponentType=_ComponentType),
            "sglang.srt.mem_cache.unified_cache.components.full": SimpleNamespace(FullComponent=FullComponent),
            "sglang.srt.mem_cache.unified_cache.unified_tree_core": SimpleNamespace(UnifiedTreeCore=UnifiedTreeCore),
            "sglang.srt.mem_cache.unified_radix_cache": SimpleNamespace(UnifiedRadixCache=UnifiedRadixCache),
        }
        save = probe.save

        def observed_save(path, data):
            self.assertEqual(path.name, "rank-0.pending")
            self.assertFalse(path.with_suffix(".json").exists())
            save(path, data)
            self.assertFalse(path.with_suffix(".json").exists())

        with tempfile.TemporaryDirectory() as directory, patch.dict(sys.modules, modules), patch.dict(
                os.environ, {"RSI_NATIVE_SESSION_OUTPUT": directory}), patch.object(probe, "save", side_effect=observed_save):
            output = Path(directory)
            (output / "worker_input.json").write_text(json.dumps({"request_ids": [row.rid for row in requests]}), encoding="utf-8")
            _worker(tp_rank=0, gpu_id=0)
            record = json.loads((output / "rank-0.json").read_bytes())
            self.assertFalse((output / "rank-0.pending").exists())
            self.assertEqual((record["cache_class"], record["tree_core_class"], record["component"]),
                             ("UnifiedRadixCache", "UnifiedTreeCore", "FULL"))
            launched = record["launched"]
        for request in requests:
            self.assertEqual(launched[request.rid][0]["input_ids_sha256"], fingerprint(list(request.origin_input_ids)))

    def test_read_only_full_device_match_trims_partially_matched_leaf(self):
        root = SimpleNamespace(last_access_time=1)
        parent = SimpleNamespace(id=2, key=(1, 2), parent=root, last_access_time=2)
        leaf = SimpleNamespace(id=3, key=(3, 4, 5, 6), parent=parent, last_access_time=3)
        values = {(2, _ComponentType.FULL): _Slots([20, 21]), (3, _ComponentType.FULL): _Slots([22, 23, 24, 25])}
        core = SimpleNamespace(root_node=root, page_size=2, match_full_device_prefix=MagicMock(return_value=(4, 3, 6)),
                               node_by_id={3: leaf}.__getitem__, get_component_device_value=lambda node, component: values[node, component])
        key = (1, 2, 3, 4, 9, 9)
        self.assertEqual(_read_radix_mapping(core, key, _ComponentType.FULL), [20, 21, 22, 23])
        core.match_full_device_prefix.assert_called_once_with(key)
        self.assertEqual((root.last_access_time, parent.last_access_time, leaf.last_access_time), (1, 2, 3))
        values[3, _ComponentType.FULL] = None
        with self.assertRaisesRegex(RuntimeError, "missing its FULL device rows"):
            _read_radix_mapping(core, key, _ComponentType.FULL)

    def test_branch_evidence_checks_consumed_prefix_and_rejects_suffix_alias(self):
        physical = {"root": [10, 11], "left": [10, 11, 12, 13, 14, 15], "right": [10, 11, 12, 13, 16, 17]}
        record = {"page_size": 2, "retained_mappings_after_both_branches": physical,
                  "launched": {name: [{"matched_prefix_rows": 2, "physical_slots": physical[name], "input_ids_sha256": name}]
                               for name in ("left", "right")}}
        expected = {"root_length": 2, "common_length": 4, "input_hashes": {"left": "left", "right": "right"}}
        proof = _proof(record, "root", ["left", "right"], **expected)
        self.assertEqual((proof["shared_root_prefix_rows"], proof["shared_prefix_rows"], proof["retained_suffix_rows"]), (2, 4, [2, 2]))
        bad = deepcopy(record)
        bad["retained_mappings_after_both_branches"]["left"][4] = 16
        with self.assertRaisesRegex(ValueError, "share physical suffix"):
            _proof(bad, "root", ["left", "right"], **expected)
        bad = deepcopy(record)
        bad["launched"]["right"][0]["matched_prefix_rows"] = 0
        with self.assertRaisesRegex(ValueError, "did not consume"):
            _proof(bad, "root", ["left", "right"], **expected)

    def test_fixture_demands_explicit_session_branch_semantics(self):
        fixture = {"schema": 1, "name": "radix-branch", "max_new_tokens": 8, "prefix": {"rid": "root", "input_ids": [1]},
                   "branches": [{"rid": name, "input_ids": [2], "drop_parent_output": True} for name in ("left", "right")]}
        self.assertEqual(_validate(fixture), fixture)
        fixture["branches"][0].pop("drop_parent_output")
        with self.assertRaises(ValueError):
            _validate(fixture)


if __name__ == "__main__":
    unittest.main()
