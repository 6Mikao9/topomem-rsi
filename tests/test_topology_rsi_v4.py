from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from rsi_memory import cuda_v4
from rsi_memory.cuda_v4 import checked_preregistration, contract_domain
from rsi_memory.memory_program import MemoryProgram
from rsi_memory.topology import fingerprint
from rsi_memory.topology_rsi import META_SYSTEM
from rsi_memory.topology_rsi_contract import ContractDomain
from rsi_memory.topology_rsi_v4 import Config, TopologyRSIV4, structural_fields


PLAN = {"observed_problem": "Selected candidates may be slower than the anchor.",
        "mechanism_change": "Change search and revision decisions.",
        "expected_effect": "Lower repeated training and paired validation time.",
        "falsification": "Retain the parent when the paired comparison fails."}
BLOCKS = {"validation": ["v1", "v2", "v3"], "test": ["t1", "t2", "t3"]}


def rule(value):
    return {"event": "allocate", "action": "place", "when": {"op": "const", "value": 1},
            "score": {"op": "const", "value": value}}


def program(value):
    return {"schema": 1, "kind": "kv_memory_program", "rules": [rule(value)]}


def source(value, mode):
    return f'''
def build_prompt(context):
    parents = context["parent_ids"][:1]
    return str({{"operator": "mutate" if parents else "construct", "parents": parents,
                "evidence": [], "rule": 0, "guidance": "{value}"}})

def build_meta_prompt(context):
    return str({{"mode": "{mode}", "evidence": [], "guidance": "test the mechanism"}})
'''


def value(candidate):
    return candidate.rules[0].score["value"] if candidate else 20


def effect(candidate):
    item = value(candidate)
    return fingerprint({"physical": 7 if item in (7, 70) else item})


class V4IntegrationTests(unittest.TestCase):
    def setUp(self):
        scratch = tempfile.TemporaryDirectory()
        self.addCleanup(scratch.cleanup)
        self.run_dir = Path(scratch.name) / "run"
        self.api = []
        self.evals = []
        self.pairs = []
        self.interrupt_matrix = False
        self.reuse_matrix_receipt = False
        self.default_reply = 70
        self.config = Config(rounds=2, slots=1, min_improvement=0.5, training_margin_ms=0.5)

    def evaluate(self, split, candidate, key):
        self.evals.append((split, key, value(candidate)))
        if self.interrupt_matrix and "/training-matrix/" in key:
            self.interrupt_matrix = False
            raise KeyboardInterrupt
        physical = 7 if value(candidate) in (7, 70) else value(candidate)
        receipt_key = "duplicate" if self.reuse_matrix_receipt and "/training-matrix/" in key else key
        return {"feasible": True, "score": physical + 0.1,
                "metrics": {"receipts": [fingerprint({"key": receipt_key})]},
                "feedback": {"split": split + "-feedback", "origin": key}}

    def pair(self, split, variants, key):
        self.pairs.append((split, key, {label: value(candidate) for label, candidate in variants.items()}))
        if split == "test":
            self.assertTrue((self.run_dir / "frozen.json").is_file())
        return {label: {"feasible": True,
                        "samples": [{"block_id": block, "score": (7 if value(candidate) == 70 else value(candidate)) + i / 10}
                                    for i, block in enumerate(BLOCKS[split])],
                        "metrics": {"private_fixture": split + "-paired-private"}}
                for label, candidate in variants.items()}

    def request(self, payload):
        envelope = json.loads(payload["messages"][1]["content"])
        self.api.append(envelope)
        decision = envelope["generator_decision"]
        if "mode" in decision:
            new_value = 7 if envelope["round"] == 0 or self.default_reply == 70 else 5
            result = {"source": source(new_value, "repair" if new_value == 7 else "rethink"), "plan": PLAN}
        else:
            guide = decision["guidance"]
            item = int(guide) if guide.isdigit() else self.default_reply
            result = program(item) if decision["operator"] == "construct" else {"rule": rule(item)}
        return {"choices": [{"message": {"content": json.dumps(result)}}]}

    def controller(self, config=None, *, request=None):
        domain = ContractDomain({"paired_block_ids": BLOCKS, "test": "sealed-test-commitment"},
                                {"target": "training only"}, (("allocate", "place"),), self.evaluate,
                                MemoryProgram.from_dict(program(12)), self.pair,
                                MemoryProgram.from_dict(program(8)), effect)
        return TopologyRSIV4(self.run_dir, config or self.config, domain, request or self.request, backend_source=Path(__file__))

    def test_provider_payload_and_manifest_share_v41_selection_contract(self):
        payloads = []

        def capture(payload):
            payloads.append(payload)
            return self.request(payload)

        config = Config(rounds=1, slots=1, min_improvement=0.5, training_margin_ms=0.5)
        c = self.controller(config, request=capture)
        try:
            report = c.run()
        finally:
            c.close()
        self.assertEqual(report["status"], "rounds_completed")
        protocol = report["manifest"]["contract"]["selection_protocol"]
        self.assertEqual(protocol["version"], "4.1")
        contract = protocol["trusted_selection_contract"]
        self.assertEqual(set(contract["finalists"]), {"manager_best", "admitted_output", "fixed"})
        self.assertEqual(contract["training_blocks"], 3)
        self.assertEqual(contract["training_margin_ms"], 0.5)
        self.assertIn("Both sides may retain", META_SYSTEM)
        self.assertEqual(len(payloads), 6)
        meta = []
        for payload in payloads:
            system, user = payload["messages"]
            envelope = json.loads(user["content"])
            self.assertEqual(envelope["trusted_selection_contract"], contract)
            self.assertNotIn("Both sides may retain", system["content"])
            if "mode" in envelope["generator_decision"]:
                meta.append(system["content"])
        self.assertEqual(len(meta), 2)
        for text in meta:
            self.assertIn("fresh program", text)
            self.assertIn("separate paired validation reference", text)
            self.assertIn("shared round-start archive", text)

    def test_repeated_training_dedup_preserves_raw_provenance_and_separate_holdout(self):
        c = self.controller()
        try:
            with self.assertRaisesRegex(ValueError, "sealed"):
                c.test()
            initial = c.run(1)
            r0 = initial["rounds"][0]
            selected = r0["v4_selection"]
            self.assertEqual(r0["decision"], "rejected")
            self.assertEqual(r0["reason"], "identical_training_effect_child_and_reference")
            self.assertEqual(selected["manager_best"]["proposal_id"], "r0000/child/00")
            self.assertEqual(selected["admitted_output"]["proposal_id"], "r0000/parent/00")
            self.assertEqual(selected["manager_best"]["effect_sha256"], selected["admitted_output"]["effect_sha256"])
            self.assertEqual(sum("/training-matrix/" in key for _, key, _ in self.evals), 6)
            self.assertEqual(len(r0["v4_training"]["cohort_proposals"]), 4)
            with self.assertRaisesRegex(ValueError, "incomplete"):
                c.freeze()
        finally:
            c.close()
        c = self.controller()
        try:
            report = c.run()
            self.assertEqual(report["status"], "rounds_completed")
            self.assertEqual(len(self.api), 12)
            r1 = report["rounds"][1]
            self.assertTrue(r1["v4_selection"]["parent"]["proposal_id"].startswith("r0001/parent/"))
            self.assertTrue(r1["v4_selection"]["child"]["proposal_id"].startswith("r0001/child/"))
            self.assertEqual(sum("/training-matrix/" in key for _, key, _ in self.evals), 9)
            self.assertEqual(r1["v4_selection"]["manager_best"]["proposal_id"], "r0000/child/00")
            self.assertEqual(r1["v4_selection"]["admitted_output"]["proposal_id"], "r0000/parent/00")
            self.assertEqual(sum(split == "test" for split, _, _ in self.pairs), 0)
            for key, call in report["calls"].items():
                envelope = json.loads(call["input"]["payload"]["messages"][1]["content"])
                evidence = envelope["selected_training_evidence"]
                if "/fixed" in key:
                    self.assertFalse(any("/parent/" in name or "/child/" in name for name in evidence))
                else:
                    self.assertFalse(any("/fixed" in name for name in evidence))
            messages = json.dumps(self.api)
            for marker in ("sealed-test-commitment", "validation-paired-private", "test-paired-private", "validation-feedback"):
                self.assertNotIn(marker, messages)
            frozen = c.freeze()
            self.assertEqual(value(MemoryProgram.from_dict(frozen["manager_best"]["trial"]["program"])), 7)
            self.assertEqual(value(MemoryProgram.from_dict(frozen["admitted_output"]["trial"]["program"])), 70)
            final = c.test()
            c.test()
            self.assertEqual(final["status"], "test_settled")
            self.assertEqual(sum(split == "test" for split, _, _ in self.pairs), 1)
            self.assertEqual(self.pairs[-1][2], {"native": 20, "manager_best": 7, "admitted_output": 70, "fixed": 70, "manual": 8})
            self.assertIn("selection_protocol", final["manifest"]["contract"])
        finally:
            c.close()

    def test_multiple_slots_reuse_only_their_own_prior_rough_feedback(self):
        c = self.controller(Config(rounds=1, slots=2, min_improvement=0.5, training_margin_ms=0.5))
        try:
            result = c.run()
            self.assertEqual(result["status"], "rounds_completed")
            row = result["rounds"][0]
            for label in ("parent", "fixed-a", "fixed-b"):
                first_id = f"r0000/{label}/00"
                call = result["calls"][f"api/r0000/{label}/01"]
                envelope = json.loads(call["input"]["payload"]["messages"][1]["content"])
                self.assertEqual(envelope["selected_training_evidence"][first_id]["measurement_kind"], "single_rough_evaluation")
                self.assertFalse(any(key.startswith("r0000/") and key != first_id
                                     for key in envelope["selected_training_evidence"]))
                trial = next(trial for trial in row["parent"] + row["fixed"] if trial["id"] == f"r0000/{label}/01")
                self.assertEqual(trial["parent_proposal_ids"], [first_id])
            self.assertEqual(row["child"][1]["parent_proposal_ids"], ["r0000/child/00"])
        finally:
            c.close()

    def test_holdout_requires_completed_feasible_pair_and_never_retries(self):
        root, original_pair = self.run_dir, self.pair
        config = Config(rounds=1, slots=1, min_improvement=0.5, training_margin_ms=0.5)
        for defect, effect_status, report_status in (
                ("failed", "failed", "test_incomplete"),
                ("unknown", "interrupted_unknown", "test_incomplete"),
                ("infeasible", "completed", "test_infeasible")):
            with self.subTest(defect=defect):
                self.run_dir = root / defect
                attempts = []

                def pair(split, variants, key):
                    if split != "test":
                        return original_pair(split, variants, key)
                    attempts.append(key)
                    if defect == "failed":
                        raise OSError("injected paired evaluation failure")
                    if defect == "unknown":
                        raise KeyboardInterrupt
                    result = original_pair(split, variants, key)
                    result["fixed"] = {"feasible": False, "samples": [], "metrics": {}}
                    return result

                with patch.object(self, "pair", side_effect=pair):
                    c = self.controller(config)
                    try:
                        self.assertEqual(c.run()["status"], "rounds_completed")
                        frozen = c.freeze()
                        if defect == "unknown":
                            with self.assertRaises(KeyboardInterrupt):
                                c.test()
                        else:
                            self.assertEqual(c.test()["status"], report_status)
                    finally:
                        c.close()
                    c = self.controller(config)
                    try:
                        report = c.test()
                        self.assertEqual(report["status"], report_status)
                        self.assertEqual(report["frozen"], frozen)
                        self.assertEqual(report["paired_evaluations"]["pair/frozen/test"]["status"], effect_status)
                        self.assertIsNone(report["test_comparisons"])
                        self.assertIsNone(report["test_program_relations"])
                        self.assertEqual(attempts, ["frozen/test"])
                    finally:
                        c.close()

    def test_cuda_cli_test_exit_code_requires_settled_report(self):
        path = self.run_dir.parent / "adapter.json"
        prereg = {"rounds": 1, "slots": 1, "training_blocks": 3, "validation_blocks": 3, "test_blocks": 3,
                  "max_output_tokens": 65536, "training_margin_ms": 2.0, "min_improvement_ms": 2.0,
                  "wall_seconds": 6000, "model": "deepseek-v4.1-flash"}
        path.write_text(json.dumps({"paired_block_ids": BLOCKS, "controller_preregistration": prereg}), encoding="utf-8")
        args = ["test", "--adapter-config", str(path), "--output", str(self.run_dir)]
        for status, expected in (("test_settled", 0), ("test_incomplete", 1), ("test_infeasible", 1)):
            with (self.subTest(status=status), patch.object(cuda_v4, "CudaDomain"),
                  patch.object(cuda_v4, "contract_domain"), patch.object(cuda_v4, "TopologyRSIV4") as controller,
                  patch.object(cuda_v4, "load_api_key") as read_key, redirect_stdout(io.StringIO()) as stdout):
                controller.return_value.test.return_value = {"status": status}
                self.assertEqual(cuda_v4.main(args), expected)
                self.assertEqual(json.loads(stdout.getvalue()), {"status": status})
                controller.return_value.close.assert_called_once()
                read_key.assert_not_called()

    def test_unknown_matrix_effect_is_preserved_and_never_repeated(self):
        self.interrupt_matrix = True
        c = self.controller()
        try:
            with self.assertRaises(KeyboardInterrupt):
                c.run()
        finally:
            c.close()
        before = (len(self.evals), len(self.api))
        c = self.controller()
        try:
            report = c.run()
            self.assertEqual(report["status"], "training_matrix_incomplete")
            self.assertEqual((len(self.evals), len(self.api)), before)
            self.assertTrue(any(row["status"] == "interrupted_unknown" for row in report["evaluations"].values()))
            self.assertEqual(c.run()["status"], "training_matrix_incomplete")
            self.assertEqual((len(self.evals), len(self.api)), before)
        finally:
            c.close()

    def test_one_receipt_cannot_supply_two_training_measurements(self):
        self.reuse_matrix_receipt = True
        c = self.controller()
        try:
            result = c.run()
            self.assertEqual(result["status"], "training_matrix_incomplete")
            matrix = result["training_matrices"]["training_matrix/r0000/result"]
            self.assertTrue(any(obs.get("measurement_error") == "reused_timing_receipt_bundle"
                                for row in matrix["measurements"].values() for obs in row["evaluations"]))
            self.assertFalse(self.pairs)
        finally:
            c.close()

    def test_seed_bytes_and_training_configuration_bind_resume(self):
        c = self.controller()
        try:
            seed = self.run_dir / "harnesses" / (c.manifest["seed_sha256"] + ".py")
            original = seed.read_bytes()
        finally:
            c.close()
        with self.assertRaisesRegex(ValueError, "resume manifest"):
            self.controller(Config(rounds=2, slots=1, min_improvement=0.5, training_margin_ms=0.6))
        seed.write_bytes(original + b"\n")
        with self.assertRaisesRegex(ValueError, "existing archived generator bytes differ"):
            self.controller()

    def test_cuda_wrapper_checks_declared_blocks_and_uses_raw_identity(self):
        raw = {"rounds": 3, "slots": 1, "training_blocks": 3, "validation_blocks": 3, "test_blocks": 3,
               "max_output_tokens": 65536, "training_margin_ms": 2.0, "min_improvement_ms": 2.0,
               "wall_seconds": 6000, "model": "deepseek-v4.1-flash"}
        checked = checked_preregistration(raw, BLOCKS)
        self.assertEqual((checked.rounds, checked.training_blocks, checked.min_paired_blocks), (3, 3, 3))
        with self.assertRaisesRegex(ValueError, "paired adapter blocks"):
            checked_preregistration({**raw, "test_blocks": 6}, BLOCKS)
        base = ContractDomain({}, {}, (("allocate", "place"),), self.evaluate, effective_program_fingerprint=effect)
        actual = contract_domain(base, raw)
        a, b = (MemoryProgram.from_dict(program(number)) for number in (7, 70))
        self.assertEqual(effect(a), effect(b))
        self.assertNotEqual(actual.effective_program_fingerprint(a), actual.effective_program_fingerprint(b))
        self.assertEqual(actual.contract["controller_preregistration_sha256"], fingerprint(raw))
        self.assertFalse(actual.program_families)

    def test_admission_requires_actual_next_round_structural_meta_reuse(self):
        self.default_reply = 9
        c = self.controller()
        try:
            report = c.run()
            self.assertEqual([row["decision"] for row in report["rounds"]], ["admitted", "rejected"])
            self.assertEqual(report["rounds"][1]["reason"], "both_entries_need_structural_behavior_change")
            reuse = report["rounds"][1]["previous_admission_reuse"]
            self.assertTrue(reuse["recursive_behavior_observed"])
            self.assertIn("mode", reuse["changed_structural_fields"])
            self.assertEqual(reuse["actual_meta_decision"]["mode"], "repair")
            self.assertEqual(reuse["previous_source_same_input"]["mode"], "revise")
            self.assertEqual(c.freeze()["reused_harness_sha256s"], [report["rounds"][0]["active_after"]])
        finally:
            c.close()
        self.assertEqual(structural_fields({"mode": "revise", "evidence": ["a", "b"], "guidance": "one"},
                                           {"mode": "revise", "evidence": ["b", "a"], "guidance": "two"}), [])


if __name__ == "__main__":
    unittest.main()
