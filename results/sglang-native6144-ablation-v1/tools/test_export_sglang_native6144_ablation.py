import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
import unittest
from zipfile import ZipFile
import sys

import export_sglang_native6144_ablation as exporter

ROOT=Path(r"D:\ws\01_研究\实验环境\topomem-rsi")
sys.path.insert(0,str(ROOT))
view_path=ROOT/"tests/test_topology_report_ablation.py"
module=importlib.util.module_from_spec(spec:=importlib.util.spec_from_file_location("mock_ablation_viewer_fixture",view_path))
spec.loader.exec_module(module)
save=module.save

class PublicationTests(unittest.TestCase):
    def setUp(self):
        self.fixture=fixture=module.AblationReportTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.study=fixture.study
        self.formal=fixture.formal.parent
        self.control=fixture.root/"control"
        self.control.mkdir()
        self.prereg=fixture.root/"sealed-prereg.json"
        self.driver=fixture.root/"driver.py"
        self.waiter=fixture.root/"waiter.py"
        self.driver.write_bytes(b"# fake frozen driver\n")
        self.waiter.write_bytes(b"# fake frozen waiter\n")
        fixture.spec["driver_sha256"]=exporter.digest(self.driver.read_bytes())
        self.source=self.formal/"frozen-source.zip"
        files={"rsi_memory/topology_rsi.py":b"# fake core","topomem_sglang_rsi/domain.py":b"# fake domain",
               "topomem_sglang_rsi/contract_v3.py":b"# fake v3"}
        inventory={name:exporter.digest(value) for name,value in files.items()}
        manifest_bytes=exporter.json_bytes({"worker_source_manifest_sha256":fixture.source,"files":inventory})
        with ZipFile(self.source,"w") as archive:
            archive.writestr("LOCAL_SOURCE_MANIFEST.json",manifest_bytes)
            for name,raw in files.items():
                archive.writestr(name,raw)
        fixture.spec.update(controller_source_archive_sha256=exporter.digest(self.source.read_bytes()),
                            controller_source_manifest_sha256=exporter.digest(manifest_bytes),
                            cutoffs={"latest_start_utc":"2026-11-11T00:00:00Z","no_new_submissions_at_or_after_utc":"2026-11-11T00:20:00Z"})
        fixture.prereg=save(fixture.study/"preregistration.json",fixture.spec)
        self.prereg.write_bytes((fixture.study/"preregistration.json").read_bytes())
        self.fixed={"preregistration":fixture.prereg,"driver":fixture.spec["driver_sha256"],
                    "waiter":exporter.digest(self.waiter.read_bytes())}
        report_path=fixture.formal/"report.json"
        report=json.loads(report_path.read_bytes())
        report["manifest"]["source_sha256"]={"topology_rsi.py":inventory["rsi_memory/topology_rsi.py"]}
        report["manifest"]["contract"]["adapter_sources_sha256"]={"topomem_sglang_rsi/domain.py":inventory["topomem_sglang_rsi/domain.py"]}
        report["manifest"]["contract"]["generator_program_protocol"]["source_sha256"]={
            "topomem_sglang_rsi/contract_v3.py":inventory["topomem_sglang_rsi/contract_v3.py"]}
        report["frozen"]["manifest_sha256"]=exporter.canonical(report["manifest"])
        fixture.frozen_sha=save(fixture.formal/"frozen.json",report["frozen"])
        self.report_sha=save(report_path,report)
        phases=[{"phase":phase,"returncode":0,"finished_utc":f"2026-11-10T23:0{i+1}:00Z"}
                for i,phase in enumerate(("run","freeze","test"))]
        pipeline_sha=save(self.formal/"pipeline-status.json",phases)
        archive_sha=fixture.spec["controller_source_archive_sha256"]
        self.private="a"*64
        save(self.formal/"artifact_manifest.json",{"private_schema3_bytes_sha256":self.private,"files_sha256":{
            "controller/report.json":self.report_sha,"controller/frozen.json":fixture.frozen_sha,
            "pipeline-status.json":pipeline_sha,"frozen-source.zip":archive_sha}})
        bound_path=fixture.study/"manifest.json"
        manifest=json.loads(bound_path.read_bytes())
        manifest["bound"].update(formal_report_sha256=self.report_sha,formal_frozen_sha256=fixture.frozen_sha,
            formal_manifest_canonical_sha256=report["frozen"]["manifest_sha256"],formal_pipeline_status_sha256=pipeline_sha,
            preregistration_sha256=fixture.prereg,formal_private_config_sha256=self.private,derived_private_config_sha256="b"*64)
        save(bound_path,manifest)
        save(self.control/"wait-plan.json",{"schema":1,"created_utc":"2026-11-10T22:59:59Z",
            "driver_sha256":self.fixed["driver"],"preregistration_sha256":fixture.prereg,
            "read_formal_report_only_after_all_phases_exit_zero":True,**fixture.spec["cutoffs"]})
        save(self.control/"source-preflight.log",{"status":"source_and_preregistration_verified","preregistration_sha256":fixture.prereg})
        (self.control/"driver.log").write_bytes(b"mock no API/no CUDA\n")

    def controls(self,result,*,no_report=False,returncode=0):
        phases=[["run",0],["freeze",0],["test",0]]
        binding={"formal_phases":phases,"utc":"2026-11-10T23:05:00Z"}
        if no_report:
            binding={"formal_phases":[["run",0]],"utc":"2026-11-11T00:00:01Z"}
        else:
            binding["formal_report_sha256"]=self.report_sha
        save(self.control/"launch-binding.json",binding)
        done={"returncode":returncode,"finished_utc":"2026-11-11T00:24:00Z"}
        if result:
            done.update(status=result["status"],timed_jobs_completed=result["timed_jobs_completed"])
        return save(self.control/"completion.json",done)

    def normalize_receipts(self):
        for path in (self.study/"receipts").glob("sg-*.json"):
            row=json.loads(path.read_bytes())
            row["result_sha256"]=exporter.digest(exporter.json_bytes(row["result"]))
            save(path,row)

    def options(self,completion,result,*,output="out",no_formal=False):
        return SimpleNamespace(output=self.fixture.root/output,control=self.control,study=self.study,repo_root=ROOT,
            preregistration=self.prereg,driver=self.driver,waiter=self.waiter,source_archive=self.source,
            formal_export=None if no_formal else self.formal,expected_formal_report_sha256=None if no_formal else self.report_sha,
            expected_completion_sha256=completion,expected_result_sha256=result)

    def test_complete_byte_preservation_and_direct_viewer(self):
        self.fixture.complete((6144,4096))
        self.normalize_receipts()
        raw=(self.study/"result.json").read_bytes()
        completion=self.controls(json.loads(raw))
        opts=self.options(completion,exporter.digest(raw))
        audit=exporter.export(opts,self.fixed)
        self.assertEqual(audit["status"],"completed")
        self.assertEqual(audit["receipt_count"],38)
        self.assertEqual(len(audit["ledger"]["timed_job_ids_in_order"]),32)
        self.assertEqual(audit["viewer_verification"]["median_gain_ms"],40)
        self.assertFalse(audit["viewer_verification"]["same_hot_capacity"])
        self.assertEqual((opts.output/"result.json").read_bytes(),raw)
        self.assertEqual((opts.output/"waiter/driver.log").read_bytes(),(self.control/"driver.log").read_bytes())
        with ZipFile(opts.output/"leased-receipts.zip") as archive:
            self.assertEqual(len(archive.namelist()),38)
            for name in archive.namelist():
                self.assertEqual(archive.read(name),(self.study/name).read_bytes())
                self.assertEqual(archive.read(name),(opts.output/name).read_bytes())
        self.assertNotIn("schema3.json",[p.name for p in opts.output.rglob("*")])
        manifest=json.loads((opts.output/"artifact_manifest.json").read_bytes())
        self.assertTrue(all(exporter.digest((opts.output/path).read_bytes())==sha for path,sha in manifest["files_sha256"].items()))

    def test_tampered_seal_rejected_before_publication(self):
        self.fixture.complete()
        self.normalize_receipts()
        raw=(self.study/"result.json").read_bytes()
        completion=self.controls(json.loads(raw))
        for field in ("expected_completion_sha256","expected_formal_report_sha256","expected_result_sha256"):
            with self.subTest(field=field):
                opts=self.options(completion,exporter.digest(raw))
                setattr(opts,field,"0"*64)
                with self.assertRaises(ValueError):
                    exporter.export(opts,self.fixed)
                self.assertFalse(opts.output.exists())

    def test_partial_stores_records_without_performance_outcome(self):
        rows=self.fixture.complete()
        self.normalize_receipts()
        data=json.loads((self.study/"result.json").read_bytes())
        raw_rows=rows[:7]
        (self.study/"jobs.jsonl").write_text("\n".join(json.dumps(row) for row in raw_rows)+"\n",encoding="utf-8")
        for row in rows[7:]:
            (self.study/"receipts"/f"{row['job_id']}.json").unlink()
        result={"status":"budget_incomplete","performance_gate":"not_evaluated","timed_jobs_completed":5,
                "preregistration_sha256":self.fixed["preregistration"],"complete_blocks":data["complete_blocks"][:1]}
        result_sha=save(self.study/"result.json",result)
        opts=self.options(self.controls(result),result_sha)
        audit=exporter.export(opts,self.fixed)
        self.assertEqual(audit["status"],"budget_incomplete")
        self.assertEqual(audit["viewer_verification"]["status"],"partial")
        self.assertNotIn("median_gain_ms",audit["viewer_verification"])
        self.assertEqual(audit["receipt_count"],11)
        self.assertEqual(audit["ledger"]["observed_timed_jobs"],5)

    def test_budget_skip_without_formal_report(self):
        (self.study/"manifest.json").unlink()
        (self.study/"frozen_adaptive.program.json").unlink()
        result={"status":"skipped_budget","performance_gate":"not_evaluated","timed_jobs_completed":0,
                "preregistration_sha256":self.fixed["preregistration"]}
        result_sha=save(self.study/"result.json",result)
        opts=self.options(self.controls(result,no_report=True),result_sha,no_formal=True)
        audit=exporter.export(opts,self.fixed)
        self.assertEqual(audit["status"],"skipped_budget")
        self.assertIsNone(audit["formal_anchor"])
        self.assertEqual(audit["receipt_count"],0)

    def test_nonzero_driver_no_synthetic_result(self):
        (self.study/"manifest.json").unlink()
        (self.study/"frozen_adaptive.program.json").unlink()
        opts=self.options(self.controls(None,no_report=True,returncode=1),None,no_formal=True)
        audit=exporter.export(opts,self.fixed)
        self.assertEqual(audit["status"],"driver_failed_without_result")
        self.assertFalse((opts.output/"result.json").exists())

if __name__=="__main__":
    unittest.main()
