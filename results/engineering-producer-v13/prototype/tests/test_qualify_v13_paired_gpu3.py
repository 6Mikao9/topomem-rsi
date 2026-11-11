"""Dry-run the paired controller using historical receipt shapes and a fake lease client."""
from copy import deepcopy
from datetime import datetime,timedelta,timezone
import io
import json
from pathlib import Path
from contextlib import redirect_stdout
import sys
import tempfile
from types import SimpleNamespace
import unittest

BASE=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(BASE))
import qualify_v13_paired_gpu3 as trial

PLAN=BASE/"gpu3-paired-study-r1/preregistration.json"
HISTORICAL=BASE.parent/"topomem-qualification-v11-paired-gpu3"


class TrialProtocol(unittest.TestCase):
    def test_reference_receipt_shapes_and_serial_new_study(self):
        plan=json.loads(PLAN.read_bytes())
        seal=trial.digest(PLAN.read_bytes())
        old={label:[json.loads((HISTORICAL/f"qualification-v11-paired-gpu3-b{i}-h64-{label}.json").read_bytes())
             for i in range(8)] for label in ("native","lru-equivalent")}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)
            cfg={"host":"mock.invalid","worker":"mock-worker","root":"mock-root","python":"mock-python",
                 "source":"mock-v11","gpu_indices":[3],"gpu_uuids":plan["payload"]["expected_gpu_uuids"],
                 "environment":{"OMP_NUM_THREADS":"2"}}
            (path/"mock.json").write_bytes(trial.nice(cfg))
            submitted=[]
            active=set()
            class MockExecutor:
                def __init__(self,*,source,source_manifest_sha256,gpu_indices,gpu_uuids,environment,audit_directory,**ignored):
                    self.config=dict(source=source,source_manifest_sha256=source_manifest_sha256,
                                     gpu_indices=gpu_indices,gpu_uuids=gpu_uuids,environment=environment)
                    self.path=audit_directory
                    self.path.mkdir()
                def __enter__(self):
                    active.add(self.config["source"])
                    return self
                def __exit__(self,*exc):
                    active.remove(self.config["source"])
                def status(self,job):
                    self_outer.assertNotIn(job,submitted)
                    return {"status":"not_found"}
                def run(self,job,module,payload,*,timeout):
                    self_outer.assertEqual(active,{"mock-v11","mock-v13"})
                    clock=datetime(2026,1,1,tzinfo=timezone.utc)+timedelta(minutes=len(submitted))
                    submitted.append(job)
                    block,label=job.rsplit("-b",1)[1].split("-",1)
                    receipt=deepcopy(old["native" if label=="native" else "lru-equivalent"][int(block)])
                    result=receipt["result"]
                    if label=="lru-equivalent-v13":
                        self_outer.assertEqual(self.config["source"],"mock-v13")
                        for case in result["cases"]:
                            diagnostic=case["diagnostic"]
                            group=diagnostic["work_counts"]["allocate/replicate"]
                            diagnostic["work_counts"]["allocate/replicate"]={"static_producer_retain_choices":group["choice_calls"]}
                            case["median_elapsed_ns"]-=1000
                            for observation in case["timings"]:
                                observation["elapsed_ns"]-=1000
                        result["score"]=sum(case["median_elapsed_ns"] for case in result["cases"])/1e6
                    request={"operation":"run","job_id":job,"module":module,"payload":payload,"timeout":timeout,**self.config}
                    receipt.update(job_id=job,source_manifest_sha256=self.config["source_manifest_sha256"],request_sha256=trial.canonical(request),
                                   pid=1000+len(submitted),proc_starttime=str(len(submitted)),utc_start=clock.isoformat(),
                                   utc_end=(clock+timedelta(seconds=1)).isoformat(),result_canonical_sha256=trial.canonical(result))
                    (self.path/(job+".json")).write_bytes(trial.nice(receipt))
                    return receipt
            self_outer=self
            args=SimpleNamespace(plan=PLAN,expected_plan_sha256=seal,audit_directory=path/"audit",
                                 executor_config=path/"mock.json",v13_source="mock-v13",output=path/"summary.json")
            with redirect_stdout(io.StringIO()):
                trial.run(args,executor_type=MockExecutor)
                trial.analyze(args)
            self.assertFalse(active)
            self.assertEqual(len(submitted),24)
            summary=json.loads(args.output.read_bytes())
            self.assertTrue(summary["same_ir_physical_trace_verified"])
            self.assertTrue(summary["expected_producer_bypass_verified"])
            self.assertAlmostEqual(summary["old_minus_new_paired_median_ms"],0.004)
            self.assertEqual(summary["old_minus_new_positive_blocks"],8)
            ledger=args.audit_directory/"jobs.jsonl"
            rows=[json.loads(line) for line in ledger.read_bytes().splitlines()]
            bad=next(row for row in rows if row["source"]=="v13")
            path=args.audit_directory/bad["source"]/(bad["job_id"]+".json")
            raw=json.loads(path.read_bytes())
            raw["result"]["cases"][0]["diagnostic"]["actions"][0]["step"]=100000
            raw["result_canonical_sha256"]=trial.canonical(raw["result"])
            path.write_bytes(trial.nice(raw))
            bad["receipt_sha256"]=trial.digest(path.read_bytes())
            ledger.write_bytes(b"".join(trial.nice(row).replace(b"\n",b"")+b"\n" for row in rows))
            with self.assertRaisesRegex(ValueError,"action signature"):
                trial.analyze(args)

if __name__=="__main__":
    unittest.main()
