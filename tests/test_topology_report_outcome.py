import json
from pathlib import Path
from statistics import median
import tempfile
import unittest

from rsi_memory.topology_report import CUDA_SCOPE, _fingerprint, read_run


class TopologyOutcomeTest(unittest.TestCase):
    def setUp(self):
        work = tempfile.TemporaryDirectory()
        self.addCleanup(work.cleanup)
        self.directory = Path(work.name)
        self.programs = {label: {"rules": [label]} for label in ("adaptive", "fixed", "manual")}
        self.programs["native"] = None
        self.blocks = ["block-1", "block-2", "block-3"]
        self.scores = {"adaptive": [10, 10, 12], "native": [9, 10, 11], "fixed": [11, 9, 12], "manual": [13, 13, 13]}
        manifest = {"config": {"rounds": 1, "min_improvement": 0.5}, "contract": {"kind": CUDA_SCOPE,
                    "paired_block_ids": {"test": self.blocks}, "manual_program_sha256": _fingerprint(self.programs["manual"])}}
        frozen = {"manifest_sha256": _fingerprint(manifest), "rounds_sha256": _fingerprint([]),
                  **{label: {"trial": {"program": self.programs[label], "program_sha256": _fingerprint(self.programs[label])}}
                     for label in ("adaptive", "fixed")}}
        self.pair = {"status": "completed", "input": {"split": "test", "block_ids": self.blocks, "programs": self.programs.copy()},
                     "result": {label: {"feasible": True, "samples": [{"block_id": block, "score": score}
                                for block, score in zip(self.blocks, scores, strict=True)]} for label, scores in self.scores.items()}}
        comparisons = {}
        for label in ("native", "fixed", "manual"):
            values = [b - a for a, b in zip(self.scores["adaptive"], self.scores[label], strict=True)]
            comparisons[f"adaptive_vs_{label}"] = {"median_improvement": median(values), "passed": label == "manual",
                "deltas": [{"block_id": block, "reference_minus_candidate": gain}
                           for block, gain in zip(self.blocks, values, strict=True)]}
        self.report = {"manifest": manifest, "frozen": frozen, "paired_evaluations": {"pair/frozen/test": self.pair},
                       "test_comparisons": comparisons}

    def read(self):
        (self.directory / "report.json").write_text(json.dumps(self.report), encoding="utf-8")
        return read_run(self.directory)["heldout_outcome"]

    def test_negative_native_result_does_not_inherit_manual_win(self):
        result = self.read()
        self.assertEqual((result["status"], result["blocks"], result["unit"]), ("settled", 3, "ms"))
        self.assertEqual(result["comparisons"], {
            "native": {"status": "rejected", "wins": 0, "median_gain": -1},
            "fixed": {"status": "rejected", "wins": 1, "median_gain": 0},
            "manual": {"status": "passed", "wins": 3, "median_gain": 3}})

    def test_pending_and_infeasible_arms_have_no_score(self):
        self.pair["status"] = "pending"
        self.assertEqual(self.read(), {"status": "pending", "pair_status": "pending"})
        self.pair["status"] = "completed"
        self.pair["result"]["native"]["feasible"] = False
        self.report["test_comparisons"]["adaptive_vs_native"] = {"passed": False, "deltas": []}
        self.assertEqual(self.read()["comparisons"]["native"], {"status": "infeasible"})

    def test_changed_pairing_or_unfrozen_program_suppresses_outcome(self):
        self.pair["input"]["programs"]["adaptive"] = self.programs["fixed"]
        self.assertEqual(self.read(), {"status": "unverified"})
        self.pair["input"]["programs"]["adaptive"] = self.programs["adaptive"]
        self.report["test_comparisons"]["adaptive_vs_native"]["deltas"][0]["reference_minus_candidate"] = 20
        self.assertEqual(self.read(), {"status": "unverified"})

    def test_completed_unpaired_test_does_not_fabricate_a_win_count(self):
        self.report["paired_evaluations"] = {}
        self.report["test_comparisons"] = None
        self.report["evaluations"] = {key: {"status": "completed", "input": {"split": "test"}}
            for key in ("eval/native/test", "eval/frozen/adaptive", "eval/frozen/fixed")}
        self.assertEqual(self.read(), {"status": "unpaired"})


if __name__ == "__main__":
    unittest.main()
