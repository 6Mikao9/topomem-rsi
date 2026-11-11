from copy import deepcopy
from pathlib import Path
import tempfile
import unittest

from integrations.topology_memory import sglang_ingress_gate as ingress
from integrations.topology_memory.generate_dp2_case import COMPUTE, HOST
from integrations.topology_memory.torch_runtime import SUPPORTED_ACTIONS as CUDA_ACTIONS
from rsi_memory.memory_program import MemoryProgram, MemoryScenario
from rsi_memory.topology import HardwareTopology, fingerprint
from tests.test_configurable_sglang import configure_choice, fake_observed_profile, fixture_case
from topomem_sglang_rsi.domain import SGLangDomain
from topomem_sglang_rsi.hot_config import (RESIDENT_ALLOCATOR_CHARGE_BYTES, RESIDENT_SENTINEL_LOGICAL_BYTES, compile_memory,
                                         resident_control_program, stock_control_program)


def geometry_with_tokens(geometry, input_tokens):
    target, graph, scenario, external = geometry
    changed = deepcopy(external)
    for group in changed[ingress.EXTERNAL_FIELD]["groups"]:
        for row in group["members"]:
            row["input_tokens"] = input_tokens
    scenario = MemoryScenario.from_dict({**scenario.to_dict(), "schedule_sha256": fingerprint(changed)})
    return target, graph, scenario, changed


class ResidentCompileTests(unittest.TestCase):
    def test_proves_case_bounds_and_separately_charges_sentinel_and_hot_metadata(self):
        with tempfile.TemporaryDirectory() as temp:
            case, inputs, geometry = fixture_case(Path(temp), "train")
            profile = fake_observed_profile(case)
            resident = resident_control_program()
            compiled = compile_memory(*geometry, profile, resident)
            startup = compiled.audit["startup"]
            self.assertEqual(startup["backing_mode"], "resident")
            self.assertEqual(startup["hot_slots"], 6144)
            self.assertEqual(startup["case_max_request_tokens"], 5064)
            self.assertEqual(startup["resident"]["reserved_rows_by_rank"], [10240, 10240])
            self.assertEqual({row["rid"]: row["prompt_sha256"] for row in startup["resident"]["requests"]},
                             {rid: item["input_ids_sha256"] for rid, item in inputs.items()})
            self.assertEqual(startup["resident_sentinel_logical_bytes_by_node"], dict.fromkeys(COMPUTE, RESIDENT_SENTINEL_LOGICAL_BYTES))
            self.assertEqual(startup["resident_sentinel_allocator_charge_bytes_by_node"], dict.fromkeys(COMPUTE, RESIDENT_ALLOCATOR_CHARGE_BYTES))
            self.assertEqual(compiled.audit["total_incremental_bytes_by_node"], {HOST: 0, **{
                node: RESIDENT_ALLOCATOR_CHARGE_BYTES + startup["metadata_bytes_reserved_by_node"][node] for node in COMPUTE}})
            self.assertIsNone(compiled.cold_program)
            self.assertFalse(compiled.audit["equivalent_to_strong_stock_control"])
            stock = compile_memory(*geometry, profile, None, native_hot=6144)
            self.assertEqual(compiled.scenario, stock.scenario)
            self.assertEqual(compiled.plan, stock.plan)
            fake = object.__new__(SGLangDomain)
            fake.profile, fake.preflight = profile, geometry
            fake.cases, fake.geometry_by_case = {"train": [case]}, {case["id"]: geometry}
            self.assertNotEqual(fake.effective_program_fingerprint(resident),
                                fake.effective_program_fingerprint(stock_control_program()))

    def test_rejects_overlong_requests_unsupported_capacity_mixed_tiers_and_underbudget(self):
        with tempfile.TemporaryDirectory() as temp:
            case, _, geometry = fixture_case(Path(temp), "train")
            profile = fake_observed_profile(case)
            resident = resident_control_program()
            boundary = compile_memory(*geometry_with_tokens(geometry, 6080), profile, resident)
            self.assertEqual(boundary.audit["startup"]["resident"]["reserved_rows_by_rank"], [12416, 12416])
            with self.assertRaisesRegex(MemoryError, "outgrow"):
                compile_memory(*geometry_with_tokens(geometry, 6081), profile, resident)
            backing_only = {**resident.to_dict(), "rules": resident.to_dict()["rules"][1:]}
            with self.assertRaisesRegex(ValueError, "qualified 6144"):
                compile_memory(*geometry, profile, configure_choice(8192, backing_only))
            mixed = resident.to_dict()
            mixed["rules"].append(stock_control_program().to_dict()["rules"][-1])
            with self.assertRaisesRegex(ValueError, "cannot include cold"):
                compile_memory(*geometry, profile, MemoryProgram.from_dict(mixed))
            target, graph, scenario, external = geometry
            graph_dict = graph.to_dict()
            for node in graph_dict["nodes"]:
                if node["kind"] == "gpu":
                    node["capacity_bytes"] = RESIDENT_ALLOCATOR_CHARGE_BYTES
            limited = HardwareTopology.from_dict(graph_dict)
            profile = {**profile, "base_topology_sha256": limited.fingerprint()}
            with self.assertRaisesRegex(MemoryError, "common budget"):
                compile_memory(target, limited, scenario, external, profile, resident)
            with self.assertRaisesRegex(ValueError, "adapter cannot execute"):
                resident.validate_actions(CUDA_ACTIONS)

    def test_infeasible_heldout_geometry_blocks_whole_arm_before_executor(self):
        with tempfile.TemporaryDirectory() as temp:
            case, _, geometry = fixture_case(Path(temp), "train")
            adapter = object.__new__(SGLangDomain)
            adapter.profile, adapter.preflight = fake_observed_profile(case), geometry
            adapter.cases = {"test": [{"id": "fits"}, {"id": "too-long"}]}
            adapter.geometry_by_case = {"fits": geometry, "too-long": geometry_with_tokens(geometry, 6100)}
            adapter._parity = lambda *_args, **_kwargs: self.fail("GPU qualification started before all case bounds were proved")
            result = adapter._trial("test", resident_control_program(), "candidate", {})
            self.assertFalse(result["feasible"])
            self.assertEqual(result["metrics"]["receipts"], [])
            self.assertEqual(result["feedback"], {})


if __name__ == "__main__":
    unittest.main()
