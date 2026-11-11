import json
from pathlib import Path
import tempfile
import unittest

from rsi_memory.topology_report import RESIDENT_ENGINEERING, SGLANG_SCOPE, _engineering, _fingerprint, render


def save(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


class ResidentReportTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)

    def test_resident_receipt_keeps_proof_and_separates_all_memory_ledgers(self):
        program = {"rules": []}
        proof = {"device_rows": 16384, "reserved_rows_by_rank": [4096, 4096], "requests": [
            {"rank": 0, "phase": "requests", "rid": "case-r0", "prompt_length": 4032, "max_new_tokens": 64,
             "prompt_sha256": "b" * 64, "prompt_text": "private mock prompt"}]}
        result = {"scope": SGLANG_SCOPE, "arm": "generated", "phase": "parity", "feasible": True, "case": {"id": "case"},
                  "memory_compilation": {"startup": {"hot_slots": 6144, "backing_mode": "resident", "resident": proof,
                                                    "resident_sentinel_allocator_charge_bytes_by_node": {"gpu0": 12582912}},
                                         "cold_plan_role": "native_comparison_geometry_only"},
                  "metrics": {"ranks": [{"rank": 0, "packed_device_payload_bytes": 64739328, "hot_metadata_bytes": 879616,
                                        "pinned_host_bytes": 0, "resident_sentinel_logical_bytes": 3936,
                                        "resident_sentinel_allocated_delta_bytes": 4608,
                                        "resident_sentinel_reserved_delta_bytes": 2097152}]}}
        receipt = self.directory / "receipt.json"
        save(receipt, {"schema": 2, "job_id": "mock-resident", "result_canonical_sha256": _fingerprint(result), "result": result})
        selection = {"selected": "resident-6144", "backing_mode": "resident", "layer_tiers_by_rank": [["resident"] * 6],
                     "selected_program_sha256": _fingerprint(program)}
        save(self.directory / "manifest.json", {"config": {"rounds": 2}, "contract": {"kind": SGLANG_SCOPE,
             "native_comparators": {"default": {"hot_slots": 4096, "backing": "all_host"}}, "manual_selection": selection}})
        html = self.directory / "resident.html"
        data = render(html, runs=[(None, self.directory)], receipts=[receipt])
        self.assertEqual(data["runs"][0]["sglang"]["manual_selection"], selection)
        serving = data["receipts"][0]["result"]["serving"]
        start = serving["memory_compilation"]["startup"]
        self.assertEqual(start["backing_mode"], "resident")
        self.assertEqual(start["resident"]["requests"][0]["prompt_length"], 4032)
        self.assertEqual(start["resident"]["proof_sha256"], _fingerprint(proof))
        self.assertEqual(serving["metrics"]["ranks"][0], result["metrics"]["ranks"][0])
        page = html.read_text(encoding="utf-8")
        self.assertNotIn("private mock prompt", page)
        self.assertIn("人工受信原语", page)

    def test_preflight_is_pending_and_training_winner_is_reconstructed(self):
        program = {"rules": []}
        sha = _fingerprint(program)
        reference, resident = "stock-6144-host", "resident-6144"
        order = [reference, resident]
        manifest = {"schema": 1, "scope": "train-only independent hand-policy sweep", "reference": reference,
                    "minimum_paired_median_gain_ms": 5, "cases": ["train-1"], "order": [order, order[::-1]],
                    "declared_policy_order": order, "policies": {
                        label: {"status": "preflight_passed", "program_sha256": sha, "backing_mode": mode}
                        for label, mode in zip(order, ("tiered", "resident"), strict=True)}}
        save(self.directory / "manifest.json", manifest)
        pending = render(self.directory / "preflight.html", manual_sweeps=[self.directory])["manual_sweeps"][0]
        self.assertEqual(pending["status"], "preflight_only")
        self.assertIsNone(pending["policies"][1]["window_ms_by_block"])
        outcome = {"status": "training_selection_only", "sweep_sha256": _fingerprint(manifest), "parity_failures": {},
                   "sum_window_ms_by_block": {reference: [110, 120], resident: [100, 111]},
                   "paired_gain_over_stock_6144_ms_by_block": {reference: [0, 0], resident: [10, 9]},
                   "selected": resident, "selected_program_sha256": sha, "eligible_to_displace_stock_6144": [resident]}
        save(self.directory / "result.json", outcome)
        save(self.directory / "selected_program.json", program)
        settled = render(self.directory / "settled.html", manual_sweeps=[self.directory])["manual_sweeps"][0]
        self.assertEqual(settled["selected"], resident)
        self.assertEqual(settled["policies"][1]["gain_ms_by_block"], [10, 9])
        outcome["selected"] = reference
        save(self.directory / "result.json", outcome)
        with self.assertRaisesRegex(ValueError, "winner differs"):
            render(self.directory / "tampered.html", manual_sweeps=[self.directory])

    def test_engineering_timings_cannot_be_promoted_to_heldout_evidence(self):
        audit = self.directory / "audit.json"
        phases, members = {}, {}
        for phase in ("parity", "performance"):
            trials = []
            for index, (mode, window) in enumerate(zip(("stock", "resident", "resident", "stock"), (20, 14, 12, 19)), 1):
                digest = f"{index:064x}"
                members[f"resident-{phase}-v3/{index:02d}-{mode}/result.json"] = {"sha256": digest}
                trials.append({"trial": index, "mode": mode, "raw_result_sha256": digest,
                               **({"window_ms": window} if phase == "performance" else {}),
                               "ranks": [{"rank": rank, "mode": mode, "oracle": {} if phase == "parity" else None}
                                         for rank in (0, 1)]})
            phases[phase] = trials
        value = {"schema": 1, "kind": RESIDENT_ENGINEERING, "phases": phases, "members": members, "paired_gain_ms": [6, 7],
                 "generated_program": False, "recursive_improvement": False, "heldout": False, "posthoc_stability_claim": False}
        save(audit, value)
        data = render(self.directory / "engineering.html", engineering=[audit])
        self.assertEqual(data["receipts"], [])
        self.assertEqual([row["gain_ms"] for row in data["engineering"][0]["pairs"]], [6, 7])
        value["heldout"] = True
        save(audit, value)
        with self.assertRaisesRegex(ValueError, "engineering audit"):
            _engineering(audit)


if __name__ == "__main__":
    unittest.main()
