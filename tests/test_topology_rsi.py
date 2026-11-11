from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest

from rsi_memory.memory_program import MemoryProgram
from rsi_memory.topology_rsi import Config, Domain, SEED_SOURCE, TopologyRSI


META_PLAN = {"observed_problem": "Selected training trials have high measured cost.",
             "mechanism_change": "Change parent selection and reconsider evidence.",
             "expected_effect": "Lower paired validation cost.",
             "falsification": "Reject the hypothesis if paired cost stays flat or rises."}


def rule(score):
    return {"event": "allocate", "action": "place", "when": {"op": "const", "value": 1},
            "score": {"op": "const", "value": score}}


def program(score):
    return {"schema": 1, "kind": "kv_memory_program", "rules": [rule(score)]}


def variant(operator, mode, score):
    parent_count = {"construct": 0, "mutate": 1, "crossover": 2}[operator]
    return f'''
def build_prompt(context):
    ids = context["parent_ids"]
    if len(ids) < {parent_count}:
        return str({{"operator": "construct", "parents": [], "evidence": [], "rule": 0, "guidance": "{score}"}})
    return str({{"operator": "{operator}", "parents": ids[:{parent_count}], "evidence": [],
                "rule": 0, "guidance": "{score}"}})

def build_meta_prompt(context):
    return str({{"mode": "{mode}", "evidence": context["evidence_ids"][:1], "guidance": "study failures"}})
'''


def reply(value):
    return {"choices": [{"message": {"content": json.dumps(value), "reasoning_content": "private scratch"}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18,
                      "completion_tokens_details": {"reasoning_tokens": 2}}}


class TopologyRSITest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name)
        self.eval_calls = []
        self.api_calls = []
        self.sources = [variant("mutate", "repair", 7), variant("crossover", "rethink", 5),
                        variant("mutate", "repair", 3)]

    def domain(self):
        def evaluate(split, candidate, key):
            self.eval_calls.append((split, key))
            if split == "train" and candidate is not None and key.startswith("r"):
                prefix = key.split("/")[0]
                connection = sqlite3.connect(self.path / "journal.sqlite")
                try:
                    for arm in ("adaptive", "fixed"):
                        plan = connection.execute("SELECT data FROM records WHERE key = ?", (f"plan/{prefix}/{arm}",)).fetchone()
                        self.assertIsNotNone(plan)
                finally:
                    connection.close()
            if split == "test":
                self.assertTrue((self.path / "frozen.json").is_file())
            score = candidate.rules[0].score["value"] if candidate else 11
            return {"feasible": True, "score": score, "metrics": {"private": split + "_metric"},
                    "feedback": {"fixture": split + "_feedback"}}

        return Domain({"target": "t", "train": "safe", "validation": "validation_commitment",
                       "test": "test_commitment"}, {"target_name": "two ranks"}, (("allocate", "place"),),
                      evaluate, MemoryProgram.from_dict(program(10)))

    def request(self, payload):
        user = json.loads(payload["messages"][1]["content"])
        self.api_calls.append(user)
        decision = user["generator_decision"]
        if "mode" in decision:
            return reply({"source": self.sources[user["round"]], "plan": META_PLAN})
        guide = decision["guidance"]
        score = int(guide) if guide.isdigit() else 9
        return reply(program(score) if decision["operator"] == "construct" else {"rule": rule(score)})

    def controller(self, config=None, request=None, domain=None):
        controller = TopologyRSI(self.path, config or Config(rounds=3, slots=1), domain or self.domain(),
                                 request or self.request)
        self.addCleanup(controller.close)
        return controller

    def test_same_start_then_real_reuse_across_resume_and_frozen_test(self):
        controller = self.controller()
        with self.assertRaisesRegex(ValueError, "sealed"):
            controller.test()
        first = controller.run(1)
        self.assertEqual(first["status"], "checkpointed")
        self.assertEqual(first["rounds"][0]["decision"], "admitted")
        self.assertEqual(len(self.api_calls), 6)
        initial_round = first["rounds"][0]
        self.assertEqual(initial_round["adaptive_meta_plan"]["training_evidence_ids"], ["bootstrap/initial-incumbent", "native"])
        self.assertEqual(initial_round["adaptive_meta_plan"]["kind"], "pre_evaluation_hypothesis")
        self.assertEqual(first["meta_plans"][initial_round["fixed_meta_plan_id"]]["plan"]["falsification"], META_PLAN["falsification"])
        with self.assertRaisesRegex(ValueError, "incomplete"):
            controller.freeze()

        resumed = self.controller()
        report = resumed.run()
        self.assertEqual(report["status"], "rounds_completed")
        self.assertEqual(len(self.api_calls), 18)
        self.assertEqual(self.eval_calls.count(("train", "bootstrap/initial-incumbent")), 1)
        self.assertEqual([row["decision"] for row in report["rounds"]], ["admitted"] * 3)
        self.assertEqual([row["child_validation"]["score"] for row in report["rounds"]], [7, 5, 3])
        for row in report["rounds"][1:]:
            reuse = row["previous_admission_reuse"]
            self.assertEqual(reuse["source_sha256"], row["active_before"])
            self.assertEqual(reuse["task_api_completed"], 1)
            self.assertTrue(reuse["meta_api_completed"])
            for call_id in [*reuse["task_api_ids"], reuse["meta_api_id"]]:
                self.assertEqual(report["calls"][call_id]["input"]["harness_sha256"], row["active_before"])
            self.assertEqual(reuse["task_program_trial_ids"], [row["parent"][0]["id"]])
            self.assertTrue(row["source_diff"])
        middle = report["rounds"][1]
        self.assertEqual(len(middle["child"][0]["program"]["rules"]), 2)
        for arm in ("adaptive", "fixed"):
            self.assertEqual(report["usage"][arm]["calls"], 9)
            self.assertEqual(report["usage"][arm]["reported_tokens"]["completion_tokens_details"], {"reasoning_tokens": 18})
        model_inputs = json.dumps(self.api_calls)
        for unavailable in ("validation_commitment", "test_commitment", "validation_feedback", "test_feedback", "private scratch"):
            self.assertNotIn(unavailable, model_inputs)
        self.assertNotIn("private scratch", (self.path / "report.json").read_text(encoding="utf-8"))
        self.assertTrue(all(split != "test" for split, _ in self.eval_calls))

        frozen = resumed.freeze()
        self.assertEqual(frozen["adaptive"]["score"], 3)
        self.assertEqual(frozen["fixed"]["score"], 9)
        self.assertEqual(frozen["active_harness_sha256"], report["rounds"][-1]["proposed_child_sha256"])
        self.assertEqual(len(frozen["admitted_harness_sha256s"]), 4)
        self.assertEqual(set(frozen["reused_harness_sha256s"]), {row["active_after"] for row in report["rounds"][:2]})
        with self.assertRaisesRegex(ValueError, "frozen"):
            resumed.run()
        final = resumed.test()
        resumed.test()
        self.assertEqual(final["status"], "test_settled")
        self.assertEqual(sum(split == "test" for split, _ in self.eval_calls), 3)

    def test_resume_counts_unknown_attempt_without_duplicate_provider_call(self):
        interrupted = []

        def request(payload):
            if not interrupted:
                interrupted.append(True)
                raise KeyboardInterrupt
            return self.request(payload)

        config = Config(rounds=1, slots=1)
        controller = self.controller(config, request)
        with self.assertRaises(KeyboardInterrupt):
            controller.run()
        resumed = self.controller(config, request)
        report = resumed.run()
        row = report["rounds"][0]
        self.assertEqual(len(self.api_calls), 5)
        self.assertEqual(row["reason"], "meta_api_interrupted_unknown")
        self.assertTrue(row["child_fallback_to_parent"])
        self.assertEqual(row["active_before"], row["active_after"])
        self.assertEqual(report["usage"]["adaptive"]["interrupted_unknown"], 1)
        self.assertEqual(report["usage"]["adaptive"]["calls"], 3)
        self.assertEqual(report["meta_plans"][row["adaptive_meta_plan_id"]]["status"], "unavailable")

    def test_no_admission_on_unchanged_mechanism_or_unmatched_gain(self):
        self.sources[0] = variant("construct", "repair", 6).replace('"evidence": []',
            '"evidence": context["evidence_ids"][:3]')
        controller = self.controller(Config(rounds=1, slots=1))
        report = controller.run()
        row = report["rounds"][0]
        self.assertEqual(row["child_validation"]["score"], 6)
        self.assertEqual(row["reason"], "both_entries_need_structural_behavior_change")
        self.assertEqual(row["active_before"], row["active_after"])
        frozen = controller.freeze()
        self.assertEqual(frozen["adaptive"]["score"], 9)
        self.assertEqual(frozen["adaptive_archive_best"]["score"], 6)
        self.assertEqual(frozen["adaptive_archive_best"]["trial"]["harness_sha256"], row["proposed_child_sha256"])
        self.assertNotIn(row["proposed_child_sha256"], frozen["admitted_harness_sha256s"])
        settled = controller.test()
        self.assertEqual(settled["evaluations"]["eval/frozen/adaptive"]["result"]["score"], 9)

    def test_mismatching_resume_config_fails_closed(self):
        self.controller(Config(rounds=1, slots=1))
        with self.assertRaisesRegex(ValueError, "manifest differs"):
            self.controller(Config(rounds=2, slots=1))

    def test_missing_or_unbounded_meta_plan_cannot_activate_child(self):
        def request(payload):
            user = json.loads(payload["messages"][1]["content"])
            if "mode" in user["generator_decision"]:
                return reply({"source": self.sources[0], "plan": {**META_PLAN, "expected_effect": "x" * 1001}})
            return self.request(payload)

        controller = self.controller(Config(rounds=1, slots=1), request)
        report = controller.run()
        row = report["rounds"][0]
        self.assertEqual(row["reason"], "invalid_meta_ValueError")
        self.assertIsNone(row["adaptive_meta_plan"])
        self.assertIsNone(row["proposed_child_sha256"])
        self.assertTrue(row["child_fallback_to_parent"])

    def test_trusted_vocabulary_and_same_start_witness_carry_sanitized_failure_forward(self):
        injected = []
        requests = []

        def request(payload):
            user = json.loads(payload["messages"][1]["content"])
            requests.append(user)
            contract = user["trusted_program_contract"]
            self.assertEqual(contract["events_and_actions"], {"allocate": ["place"]})
            self.assertIn("read-only", contract["execution_labels"])
            if "mode" in user["generator_decision"]:
                behavior = user["trusted_parent_behavior"]
                self.assertEqual(behavior["candidate_slot_count"], 1)
                self.assertEqual(behavior["task_decisions"][0]["slot"], 0)
                self.assertNotIn("guidance", behavior["task_decisions"][0]["decision"])
                self.assertNotIn("guidance", behavior["meta_decision"])
                if user["round"] == 0:
                    return reply({"source": SEED_SOURCE, "plan": META_PLAN})
            elif not injected:
                injected.append(True)
                invalid = program(1)
                invalid["rules"][0]["action"] = "baseline_place_model_supplied_text"
                return reply(invalid)
            return self.request(payload)

        controller = self.controller(Config(rounds=2, slots=1), request)
        report = controller.run()
        trial = report["rounds"][0]["parent"][0]
        self.assertEqual(trial["status"], "invalid_candidate")
        self.assertEqual(trial["feedback"], {"failure_category": "unsupported_event_action"})
        self.assertNotIn(("train", trial["id"]), self.eval_calls)
        witnesses = [row["selected_training_evidence"][trial["id"]] for row in requests
                     if row["round"] == 1 and trial["id"] in row["selected_training_evidence"]]
        self.assertTrue(witnesses)
        self.assertTrue(all(row["feedback"] == trial["feedback"] for row in witnesses))
        self.assertNotIn("baseline_place_model_supplied_text", json.dumps(requests))

    def paired_domain(self, flaw=None):
        original = self.domain()
        block_ids = {"validation": ["v0", "v1", "v2"], "test": ["t0", "t1", "t2"]}

        def paired(split, variants, key):
            self.eval_calls.append((split, key))
            self.assertIsNone(variants["native"])
            self.assertEqual(variants["manual"].rules[0].score["value"], 8)
            if split == "test":
                self.assertTrue((self.path / "frozen.json").is_file())
            arms = {}
            for label, candidate in variants.items():
                score = candidate.rules[0].score["value"] if candidate else 11
                samples = [{"block_id": block, "score": score + i / 10, "wire_bytes": 32}
                           for i, block in enumerate(block_ids[split])]
                if label == "child" and flaw == "reordered":
                    samples.reverse()
                if label == "child" and flaw == "losing_block":
                    samples[0]["score"] = 20
                if label == "incumbent" and flaw == "incumbent_stronger":
                    samples = [{**sample, "score": 6.0 + i / 10} for i, sample in enumerate(samples)]
                arms[label] = {"feasible": True, "score": -12345, "samples": samples,
                               "metrics": {"private": split + "_paired_raw"}}
            return arms

        return replace(original, contract={**original.contract, "paired_block_ids": block_ids},
                       evaluate_pair=paired, manual_program=MemoryProgram.from_dict(program(8)))

    def test_strict_pairing_computes_raw_deltas_and_seals_test(self):
        controller = self.controller(Config(rounds=1, slots=1, min_paired_blocks=3), domain=self.paired_domain())
        result = controller.run()
        row = result["rounds"][0]
        self.assertEqual(row["decision"], "admitted")
        self.assertEqual(row["child_validation"]["score"], 7.1)
        self.assertTrue(row["paired_validation"]["comparisons"]["child_vs_manual"]["passed"])
        self.assertTrue(row["paired_validation"]["comparisons"]["child_vs_parent"]["passed"])
        self.assertTrue(row["paired_validation"]["comparisons"]["child_vs_incumbent"]["passed"])
        self.assertEqual(set(result["paired_evaluations"]["pair/r0000/validation"]["input"]["programs"]),
                         {"native", "manual", "parent", "child", "fixed", "incumbent"})
        self.assertNotIn("validation_paired_raw", json.dumps(self.api_calls))
        frozen = controller.freeze()
        final = controller.test()
        self.assertEqual(final["frozen"], frozen)
        self.assertTrue(final["test_comparisons"]["adaptive_vs_native"]["passed"])
        self.assertTrue(final["test_comparisons"]["adaptive_vs_manual"]["passed"])
        self.assertEqual(final["paired_evaluations"]["pair/frozen/test"]["input"]["block_ids"], ["t0", "t1", "t2"])

    def test_pair_order_mismatch_rejects_and_prevents_freeze(self):
        controller = self.controller(Config(rounds=1, slots=1, min_paired_blocks=3),
                                     domain=self.paired_domain("reordered"))
        result = controller.run()
        self.assertEqual(result["rounds"][0]["reason"], "missing_feasible_matched_validation")
        self.assertEqual(result["paired_evaluations"]["pair/r0000/validation"]["status"], "failed")
        with self.assertRaisesRegex(ValueError, "feasible validation finalist"):
            controller.freeze()

    def test_median_gain_with_losing_block_cannot_admit(self):
        controller = self.controller(Config(rounds=1, slots=1, min_paired_blocks=3),
                                     domain=self.paired_domain("losing_block"))
        result = controller.run()
        row = result["rounds"][0]
        self.assertEqual(row["reason"], "insufficient_stable_paired_improvement")
        comparison = row["paired_validation"]["comparisons"]["child_vs_parent"]
        self.assertGreater(comparison["median_improvement"], 0)
        self.assertFalse(comparison["passed"])

    def test_child_must_beat_the_incumbent_even_if_it_beats_fresh_parent(self):
        controller = self.controller(Config(rounds=1, slots=1, min_paired_blocks=3),
                                     domain=self.paired_domain("incumbent_stronger"))
        result = controller.run()
        row = result["rounds"][0]
        self.assertEqual(row["reason"], "insufficient_stable_paired_incumbent_improvement")
        self.assertTrue(row["paired_validation"]["comparisons"]["child_vs_parent"]["passed"])
        self.assertFalse(row["paired_validation"]["comparisons"]["child_vs_incumbent"]["passed"])

    def test_shadowed_crossover_cannot_turn_historical_recovery_into_admission(self):
        domain = self.paired_domain()
        original_evaluate = domain.evaluate
        original_paired = domain.evaluate_pair
        proposal_counter = [0, 0]

        def request(payload):
            self.api_calls.append(payload)
            user = json.loads(payload["messages"][1]["content"])
            number = user["round"]
            decision = user["generator_decision"]
            if "mode" in decision:
                source = SEED_SOURCE if number == 0 else variant("crossover", "repair", 6)
                return reply({"source": source, "plan": META_PLAN})
            ordinal = proposal_counter[number]
            proposal_counter[number] += 1
            score = ([6, 7, 9, 9] if number == 0 else [6, 20, 9, 9])[ordinal]
            return reply(program(score) if decision["operator"] == "construct" else {"rule": rule(score)})

        def evaluate(split, candidate, key):
            result = original_evaluate(split, candidate, key)
            if key == "r0001/child/00":
                self.assertEqual(len(candidate.rules), 2)
                self.assertEqual(candidate.rules[0].score["value"], 6)
                result["score"] = 5
            return result

        def paired(split, variants, key):
            result = original_paired(split, variants, key)
            if key == "r0001/validation":
                self.assertEqual(variants["parent"].fingerprint(), variants["incumbent"].fingerprint())
                self.assertEqual(variants["child"].fingerprint(), variants["incumbent"].fingerprint())
                for sample in result["child"]["samples"]:
                    sample["score"] -= 1
            return result

        controller = self.controller(Config(rounds=2, slots=1, min_paired_blocks=3), request,
                                     replace(domain, evaluate=evaluate, evaluate_pair=paired))
        result = controller.run()
        row = result["rounds"][1]
        selection = row["task_output_selection"]
        self.assertEqual(len(self.api_calls), 12)
        self.assertEqual(row["reason"], "child_retained_incumbent")
        self.assertEqual(selection["shared_incumbent"]["id"], "r0000/parent/00")
        self.assertEqual(selection["fresh"]["child"]["score"], 5)
        self.assertEqual(selection["fresh"]["parent"]["score"], 20)
        self.assertEqual(selection["selected"]["child"]["origin"], "incumbent")
        self.assertTrue(selection["parent_incumbent_aa"])
        self.assertTrue(row["paired_validation"]["comparisons"]["child_vs_incumbent"]["passed"])
        self.assertIsNone(selection["fresh_paired_comparison"])

    def test_without_an_initial_program_promotion_waits_for_a_measured_incumbent(self):
        domain = replace(self.paired_domain(), initial_program=None)
        controller = self.controller(Config(rounds=1, slots=1, min_paired_blocks=3), domain=domain)
        result = controller.run()
        row = result["rounds"][0]
        self.assertEqual(row["reason"], "missing_measured_incumbent")
        self.assertTrue(row["paired_validation"]["comparisons"]["child_vs_parent"]["passed"])
        self.assertIsNone(row["task_output_selection"]["shared_incumbent"])
        self.assertNotIn(("train", "bootstrap/initial-incumbent"), self.eval_calls)

    def test_adapter_fingerprint_blocks_distinct_ir_with_identical_effective_output(self):
        seen = []

        def effective(candidate):
            self.assertIsInstance(candidate, MemoryProgram)
            score = candidate.rules[0].score["value"]
            seen.append(score)
            return "a" * 64 if score in (7, 10) else "b" * 64

        domain = replace(self.paired_domain(), effective_program_fingerprint=effective)
        controller = self.controller(Config(rounds=1, slots=1, min_paired_blocks=3), domain=domain)
        result = controller.run()
        row = result["rounds"][0]
        selection = row["task_output_selection"]
        different_ir = selection["fresh"]["child"]["program_sha256"]
        incumbent_ir = selection["shared_incumbent"]["program_sha256"]
        self.assertNotEqual(different_ir, incumbent_ir)
        digests = selection["effective_program_sha256_by_raw_program"]
        self.assertEqual(digests[different_ir], digests[incumbent_ir])
        self.assertEqual(row["reason"], "child_retained_incumbent")
        self.assertEqual(result["manifest"]["effective_program_fingerprint"], "adapter_training_only_v1")
        self.assertEqual(sorted(seen), [7, 9, 10])
        self.assertEqual(len(self.api_calls), 6)
        self.assertFalse(any(split == "test" for split, _ in self.eval_calls))


if __name__ == "__main__":
    unittest.main()
