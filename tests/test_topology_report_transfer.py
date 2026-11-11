import hashlib
import json
from pathlib import Path
import sqlite3
from tempfile import TemporaryDirectory
import unittest

from rsi_memory.topology_report_transfer import read_transfer


SCENARIOS = ("agent_tools", "chatbot_growth", "long_document_qa", "rollout_cow")


def save(path, value):
    raw = json.dumps(value).encode()
    path.write_bytes(raw)
    return hashlib.sha256(raw).hexdigest()


class TransferReportTest(unittest.TestCase):
    def test_sealed_lineage_and_parity(self):
        with TemporaryDirectory() as path:
            path = Path(path)
            blocks = [str(i) for i in range(12)]
            order = ["native", "manual-direct", "seed", "admitted"]
            plan = {"schema": 1, "kind": "frozen_v7_generators_transfer_on_v11", "generators": {"seed": "a", "admitted": "b"},
                    "scenarios": {name: {"block_ids": blocks, "test_case_sha256": ["k"]} for name in SCENARIOS},
                    "margins_ms": dict.fromkeys(SCENARIOS, 0.5), "operator_profiles": {name: {"gpu_indices": [0, 1]} for name in SCENARIOS},
                    "budget": {"family_comparisons": 12}, "generation": {}, "runtime_sha256": "r", "driver_sha256": "d",
                    "scope": {"agent_tools": "same-domain-engine-transfer", "other": "cross-scenario-transfer"},
                    "test_schedule_by_eligible_source_set": {"both": [order] * 12}}
            plan_sha = save(path / "preregistration.json", plan)
            controls = read_transfer(path)
            self.assertFalse(controls["journal_verified"])
            self.assertIsNone(controls["scenarios"]["agent_tools"]["heldout"])
            generated_sha = save(path / "generated.json", {"context_sha256": "c", "slots": {}})
            program = {"rules": []}
            winner = {"program": program, "program_sha256": hashlib.sha256(b'{"rules":[]}').hexdigest(), "slot_id": "0"}
            selected_sha = save(path / "selected.json", {"generated_sha256": generated_sha,
                               "winners": {name: {"seed": winner, "admitted": winner} for name in SCENARIOS}})
            evidence = {"name": "mock", "case_sha256": "k", "attention_sha256": "a"}
            arms = {label: {"feasible": True, "cases": [dict(evidence)], "score": 50} for label in order}
            aggregate = {"blocks": [{"block_id": i, "candidate_ms": 50, "reference_ms": 50, "gain_ms": 0} for i in blocks],
                         "preregistered_minimum_gain_ms": 0.5,
                         "paired_bootstrap_interval_ms": [0, 0]}
            outcome = {"source_eligibility": "both", "blocks": [{"block_id": i, "order": order, "arms": arms} for i in blocks],
                       "analysis": {"comparisons": dict.fromkeys(("seed", "native", "manual-direct"), {"complete": True,
                                    "strict_all_blocks_above_margin": False, "scopes": {"aggregate": aggregate}})}}
            heldout = {"selection_sha256": selected_sha, "primary_comparisons": 12, "scenarios": dict.fromkeys(SCENARIOS, outcome)}
            def seal():
                digest = save(path / "heldout.json", heldout)
                with sqlite3.connect(path / "journal.sqlite") as db:
                    db.execute("CREATE TABLE IF NOT EXISTS records(key TEXT PRIMARY KEY,data TEXT)")
                    rows = {"identity": {"preregistration_sha256": plan_sha}, "phase/contexts": "c",
                            "phase/generated": generated_sha, "phase/selected": selected_sha, "phase/heldout": digest}
                    db.executemany("INSERT OR REPLACE INTO records VALUES (?,?)", [(k, json.dumps(v)) for k, v in rows.items()])
            seal()
            view = read_transfer(path)
            self.assertTrue(view["journal_verified"])
            self.assertEqual(len(view["scenarios"]["agent_tools"]["heldout"]["blocks"]), 12)
            self.assertTrue(view["scenarios"]["agent_tools"]["winners"]["admitted"]["program_hash_matches"])
            arms["admitted"]["cases"][0]["attention_sha256"] = "wrong"
            seal()
            with self.assertRaisesRegex(ValueError, "selected-KV/output hashes"):
                read_transfer(path)
            (path / "heldout.json").write_bytes(b"{}")
            with self.assertRaisesRegex(ValueError, "seal does not match"):
                read_transfer(path)


if __name__ == "__main__":
    unittest.main()
