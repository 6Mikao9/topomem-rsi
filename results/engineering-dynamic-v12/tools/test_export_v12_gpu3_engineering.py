"""Exercise the v12 archive against a fully simulated run with real receipt shapes."""
import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import export_v12_gpu3_engineering as exporter
from test_qualify_v12_paired_gpu3 import TrialProtocol,trial


class ExportTest(unittest.TestCase):
    def test_export_and_replay_all_24_synthetic_jobs(self):
        original_analyze=trial.analyze
        checked=[]
        def analyze_and_archive(args):
            original_analyze(args)
            output=args.output.parent/"published"
            summary=exporter.export(SimpleNamespace(output=output,prototype_root=ROOT,analysis=args.output,
                run_directory=args.audit_directory,expected_analysis_sha256=exporter.digest(args.output.read_bytes()),
                expected_ledger_sha256=exporter.digest((args.audit_directory/"jobs.jsonl").read_bytes())))
            self.assertFalse(summary["old_gap_recovery_preregistered"])
            self.assertEqual(summary["receipt_count"],24)
            self.assertEqual(summary["statistics"]["aggregate"]["v11_minus_v12"]["positive_blocks"],8)
            for case in summary["statistics"]["cases"].values():
                self.assertEqual(case["v11_minus_v12"]["positive_blocks"],8)
                self.assertAlmostEqual(case["v11_minus_v12"]["median_ms"],0.001)
            self.assertEqual(exporter.verify(output,replay=True)["receipts"],24)
            checked.append(True)
        with patch.object(trial,"analyze",analyze_and_archive):
            TrialProtocol("test_reference_receipt_shapes_and_serial_new_study").test_reference_receipt_shapes_and_serial_new_study()
        self.assertEqual(checked,[True])

    def test_descriptive_recovery_handles_positive_and_absent_gap(self):
        row={"native":10_000_000,"writeback-v11":18_000_000,"writeback-v12":14_000_000}
        values=exporter.scope_stats([row]*8)
        self.assertEqual(values["old_gap_recovery_ratio"],0.5)
        self.assertTrue(values["recovered_at_least_half_of_positive_old_gap"])
        self.assertIsNone(exporter.scope_stats([{**row,"native":20_000_000}]*8)["old_gap_recovery_ratio"])

if __name__=="__main__":
    unittest.main()
