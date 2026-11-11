from copy import deepcopy
import unittest

from rsi_memory.statistics import summarize_paired
from rsi_memory.topology import fingerprint


BLOCKS = [f"block-{index:02d}" for index in range(12)]


def plan():
    return {"schema": 1, "kind": "paired_latency_preregistration", "block_ids": BLOCKS,
            "case_names": ["short", "long"], "candidate": "adaptive",
            "comparators": ["native", "manual", "fixed"],
            "threshold_ms": {"short": 0.1, "long": 0.2, "aggregate": 0.3},
            "bootstrap": {"seed": 423, "draws": 2000, "family_alpha": 0.05}}


def result():
    def arm(short, long):
        return {"feasible": True, "samples": [{"block_id": block, "score": short+long+index,
                                                "case_scores_ms": {"short": short, "long": long+index}}
                                               for index, block in enumerate(BLOCKS)]}

    return {"adaptive": arm(3, 7), "native": arm(2, 10), "manual": arm(4, 9), "fixed": arm(3, 8)}


class PairedStatisticsTests(unittest.TestCase):
    def test_paired_bootstrap_reports_regression_even_when_aggregate_wins(self):
        prereg, data = plan(), result()
        summary = summarize_paired(prereg, data, expected_plan_sha256=fingerprint(prereg))
        self.assertEqual(summary, summarize_paired(prereg, data, expected_plan_sha256=fingerprint(prereg)))
        self.assertEqual(summary["bootstrap"]["contrast_count"], 9)
        self.assertAlmostEqual(summary["bootstrap"]["marginal_interval_confidence"], 1-0.05/9)
        scopes = summary["comparators"]["native"]["scopes"]
        self.assertEqual(scopes["aggregate"]["paired_bootstrap_interval_ms"], [2, 2])
        self.assertTrue(scopes["aggregate"]["interval_above_minimum"])
        self.assertEqual(scopes["short"]["median_gain_ms"], -1)
        self.assertFalse(scopes["short"]["all_blocks_positive"])
        self.assertEqual(list(summary["comparators"]), prereg["comparators"])

    def test_missing_or_infeasible_comparator_is_reported_without_partial_interval(self):
        prereg, data = plan(), result()
        data["native"]["samples"].pop()
        data["native"]["feasible"] = False
        del data["fixed"]
        summary = summarize_paired(prereg, data, expected_plan_sha256=fingerprint(prereg))
        native, fixed = (summary["comparators"][arm] for arm in ("native", "fixed"))
        self.assertEqual(native["missing_block_ids"], [BLOCKS[-1]])
        self.assertEqual(native["comparator_status"], "infeasible")
        self.assertEqual(len(native["scopes"]["aggregate"]["blocks"]), 11)
        self.assertIsNone(native["scopes"]["aggregate"]["paired_bootstrap_interval_ms"])
        self.assertEqual(fixed["missing_block_ids"], BLOCKS)
        self.assertTrue(summary["comparators"]["manual"]["complete"])

    def test_rejects_changed_preregistration_duplicate_and_aggregate_drift(self):
        prereg, data = plan(), result()
        digest = fingerprint(prereg)
        changed = deepcopy(prereg)
        changed["threshold_ms"]["short"] = 0.5
        with self.assertRaisesRegex(ValueError, "preregistered fingerprint"):
            summarize_paired(changed, data, expected_plan_sha256=digest)
        data["native"]["samples"][0]["score"] += 1
        with self.assertRaisesRegex(ValueError, "aggregate score"):
            summarize_paired(prereg, data, expected_plan_sha256=digest)
        data = result()
        data["native"]["samples"][-1]["block_id"] = BLOCKS[0]
        with self.assertRaisesRegex(ValueError, "duplicate paired block"):
            summarize_paired(prereg, data, expected_plan_sha256=digest)
