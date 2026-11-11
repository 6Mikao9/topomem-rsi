from dataclasses import replace
import json
from pathlib import Path
import tempfile
import unittest

from rsi_memory.topology_rsi import Config, Domain
from rsi_memory.topology_rsi_contract import ContractDomain, ContractTopologyRSI, ProgramFamily, build_program_contract
from tests.test_configurable_sglang import fake_observed_profile, fixture_case
from topomem_sglang_rsi import contract_v3
from topomem_sglang_rsi.hot_config import ACTIONS, compile_memory, resident_control_program, stock_control_program


class ContractV3Tests(unittest.TestCase):
    def test_actual_task_envelope_and_resident_compilation_use_same_grammar(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            case, _, geometry = fixture_case(root, "train")
            profile = fake_observed_profile(case)
            resident = resident_control_program().to_dict()
            mixed = {**resident, "rules": [*resident["rules"], stock_control_program().to_dict()["rules"][-1]]}
            outputs = iter((resident, mixed))
            seen = []
            evaluated = []

            def request(payload):
                seen.append(json.loads(payload["messages"][1]["content"]))
                return {"choices": [{"finish_reason": "stop", "message": {"content": json.dumps(next(outputs))}}]}

            def evaluate(split, program, _key):
                self.assertEqual(split, "train")
                evaluated.append(program.to_dict())
                try:
                    compiled = compile_memory(*geometry, profile, program)
                except (ValueError, MemoryError) as error:
                    return {"feasible": False, "score": None, "feedback": {"failure": str(error)}}
                self.assertEqual(compiled.audit["startup"]["backing_mode"], "resident")
                self.assertIsNone(compiled.cold_program)
                return {"feasible": True, "score": 1.0}

            backend = Domain({}, {"runtime_semantics": "test SGLang target"}, ACTIONS, evaluate)
            controller = ContractTopologyRSI(root / "run", Config(), contract_v3.contract_domain(backend), request,
                                             backend_source=Path(contract_v3.__file__))
            try:
                ctx = controller._context(0, 0, controller.seed_source, {}, {})
                good = controller._trial("r0000/parent/00", ctx, controller.seed_source, {})
                bad = controller._trial("r0000/parent/01", ctx, controller.seed_source, {})
                saved = json.loads((root / "run/manifest.json").read_bytes())["contract"]["generator_program_protocol"]
            finally:
                controller.close()

            self.assertEqual(good["status"], "feasible")
            self.assertEqual(good["program"], resident)
            self.assertEqual(bad["status"], "infeasible")
            self.assertIn("cannot include cold placement", bad["feedback"]["failure"])
            self.assertEqual(evaluated, [resident, mixed])
            for envelope in seen:
                actual = envelope["trusted_program_contract"]
                self.assertEqual(actual, saved["trusted_program_contract"])
                self.assertNotIn("required_rule", actual)
                self.assertEqual(actual["required_any_rule"], [{"event": "allocate", "action": "place"},
                                                               {"event": "configure", "action": "backing_mode"}])
                families = {row["name"]: row for row in actual["program_families"]}
                self.assertEqual(families["resident"]["forbidden_rules"], [{"event": "allocate", "action": "place"}])
                self.assertIn("can replace only one existing rule", actual["rule_requirement_scope"])
            self.assertEqual(set(saved["source_sha256"]), {"rsi_memory/topology_rsi_contract.py", "topomem_sglang_rsi/contract_v3.py"})

    def test_cuda_keeps_original_required_rule_and_invalid_family_is_rejected(self):
        domain = ContractDomain({}, {}, (("allocate", "place"), ("access", "evict")), lambda *_: None)
        result = build_program_contract(domain)
        self.assertEqual(result["required_rule"], {"event": "allocate", "action": "place"})
        self.assertNotIn("required_any_rule", result)
        invalid = replace(domain, program_families=(ProgramFamily("unsupported", (("allocate", "place"),),
                                                                 (("configure", "backing_mode"),), "bad"),))
        with self.assertRaisesRegex(ValueError, "contradicts"):
            build_program_contract(invalid)


if __name__ == "__main__":
    unittest.main()
