import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from sglang_post_freeze_native6144 import (
    ADAPTIVE, NATIVE, MARGIN, MATCH, START_BY, STOP_SUBMIT, ExecutorGate, BudgetExpired,
    fixed_blocks, canonical, digest, time_utc, run_study, phase_status, frozen_selection,
)

BEFORE = time_utc("2026-11-10T23:55:00Z")
AFTER_START = time_utc("2026-11-11T00:00:01Z")
AT_STOP = time_utc(STOP_SUBMIT)
CASES = [{"id":"case-a"},{"id":"case-b"}]
SPEC = {"cases":CASES,"blocks":fixed_blocks(),"driver_sha256":"a"*64}
PROGRAM = {"schema":1,"kind":"kv_memory_program","rules":[]}
BINDING = {"preregistration_sha256":"b"*64,"formal_frozen_sha256":"c"*64,"adaptive_program_sha256":canonical(PROGRAM)}

def job_id(value):
    return "sg-"+canonical(value)[:48]

class FakeExecutor:
    config = {}

    def __init__(self, existing=False):
        self.existing = existing
        self.submissions = []

    def status(self, job):
        return {"status":"completed" if self.existing else "not_found"}

    def run(self, job, module, payload, *, timeout):
        self.submissions.append((job,module,payload,timeout))

class FakeAdapter:
    def __init__(self, cutoff_after=None, same_slots=False, bad_kv=False, existing_timing=False, adaptive_score=900.0):
        self.native_repeats = {}
        self.parity_calls = []
        self.timed_calls = []
        self.current = BEFORE
        self.executor = FakeExecutor(existing_timing)
        self.client = ExecutorGate(self.executor,lambda:self.current)
        self.cutoff_after = cutoff_after
        self.same_slots = same_slots
        self.bad_kv = bad_kv
        self.adaptive_score = adaptive_score

    def _client(self):
        return self.client

    def _proof(self, job, result):
        assert result["feasible"]
        return {"job_id":job,"raw_result_sha256":result["raw_result_sha256"]}

    def _parity(self, case, arm, program, *, slots=None):
        self.parity_calls.append((case["id"],arm,program,slots))
        assert (slots==6144 and program is None) if arm=="native" else (slots is None and program==PROGRAM)
        if arm=="native":
            self.native_repeats[case["id"],6144]=job_id([case,"native-repeat"])
        job=job_id([case,arm,"qualification"])
        return job,self.result(job,case,arm,"parity",program)

    def _job(self, case, arm, phase, program, identity, proof, *, native_hot=None):
        assert phase=="performance"
        assert (native_hot==6144 and program is None) if arm=="native" else (native_hot is None and program==PROGRAM)
        assert self.client.require_new_timing
        assert proof["job_id"]==job_id([case,arm,"qualification"])
        job=job_id([identity,program])
        self.client.status(job)
        self.client.run(job,"sglang_evaluator",{"deadline":300},timeout=780)
        self.timed_calls.append((identity,proof))
        if self.cutoff_after==len(self.timed_calls):
            self.current=AT_STOP
        return job,self.result(job,case,arm,phase,program)

    def result(self, job, case, arm, phase, program):
        native=arm=="native"
        slots=6144 if native or self.same_slots else 4096
        mode="tiered" if native or not self.same_slots else "resident"
        packed=[canonical([case,rank]) for rank in range(2)] if phase=="parity" else []
        if phase=="parity" and self.bad_kv and not native:
            packed[1]="d"*64
        metadata={key:canonical([case,key]) for key in MATCH}
        ingress=canonical([case,"ingress"])
        seconds=(1000.0 if native else self.adaptive_score)/1000
        metrics={**metadata,"global_ticks":{"global_ticks_sha256":metadata["global_ticks_sha256"]},
                 "packed_kv_sha256_by_rank":packed,"backing_mode":mode,"ingress":{"contract_sha256":ingress},
                 "ranks":[{"rank":rank,"hot_slots":slots,"layer_tiers":["host"]*6,"local_hbm_bytes":0,
                           "peer_hbm_bytes":0,"pinned_host_bytes":129226752} for rank in range(2)],
                 "timing":{"window_seconds":seconds}}
        return {"case":case,"arm":arm,"phase":phase,"program_sha256":canonical(program) if program is not None else None,
                "feasible":True,"raw_result_sha256":canonical([job,"serving"]),"metrics":metrics,
                "memory_compilation":{"external_scheduler_sha256":canonical([case,"schedule"]),
                  "ingress_contract_sha256":ingress,
                  "startup":{"hot_slots":slots,"backing_mode":mode,"native_reference_override":6144 if native else None}},
                "cross_capacity_native_parity":{"matched":True},"score":seconds*1000 if phase=="performance" else None}

class TestSupplement(unittest.TestCase):
    def execute(self, fake, now=None):
        with tempfile.TemporaryDirectory() as d:
            path = Path(d)
            result = run_study(fake,PROGRAM,SPEC,BINDING,path,clock=now or (lambda:fake.current))
            jobs=path/"jobs.jsonl"
            self.journal = [json.loads(l) for l in jobs.read_bytes().splitlines()] if jobs.is_file() else []
            return result

    def test_actual_native_and_balanced_fresh_jobs(self):
        fake=FakeAdapter()
        result=self.execute(fake)
        self.assertEqual(result["status"],"completed")
        self.assertEqual(result["timed_jobs_completed"],32)
        self.assertEqual(result["performance_gate"],"passed")
        self.assertEqual(result["median_ms"],200.0)
        self.assertFalse(any(v["same_hot_capacity"] for v in result["startup_by_case"].values()))
        actual=[(i["block_id"],i["case_id"],i["label"]) for i,_ in fake.timed_calls]
        expected=[(b["block_id"],c["id"],l) for b in fixed_blocks() for c in CASES for l in b["order"]]
        self.assertEqual(actual,expected)
        self.assertEqual(len({job for job,*_ in fake.executor.submissions}),32)
        self.assertTrue(all(i["preregistration_sha256"]=="b"*64 and i["driver_sha256"]=="a"*64
                            and i["formal_frozen_sha256"]=="c"*64 for i,_ in fake.timed_calls))
        self.assertEqual(len(self.journal),34)
        self.assertEqual([a for a in fake.parity_calls if a[1]=="native"],[(c["id"],"native",None,6144) for c in CASES])

    def test_nonpassing_and_same_hot_capacity_are_distinct(self):
        fake=FakeAdapter(same_slots=True,adaptive_score=995.0)
        result=self.execute(fake)
        self.assertEqual(result["status"],"completed")
        self.assertEqual(result["positive_blocks"],8)
        self.assertEqual(result["performance_gate"],"failed")
        self.assertLess(result["median_ms"],MARGIN)
        self.assertTrue(all(v["same_hot_capacity"] for v in result["startup_by_case"].values()))

    def test_budget_incomplete_keeps_partial_originals(self):
        fake=FakeAdapter(cutoff_after=5)
        result=self.execute(fake)
        self.assertEqual(result["status"],"budget_incomplete")
        self.assertEqual(result["performance_gate"],"not_evaluated")
        self.assertEqual(result["timed_jobs_completed"],5)
        self.assertEqual(len(result["complete_blocks"]),1)
        self.assertEqual(len(self.journal),7)
        self.assertEqual(len(fake.executor.submissions),5)

    def test_recursive_submission_guard_at_cutoff(self):
        fake=FakeAdapter()
        fake.current=AT_STOP
        with self.assertRaises(BudgetExpired):
            fake.client.run(job_id("recursive-native-qualification"),"sglang_evaluator",{"deadline":300},timeout=780)
        self.assertFalse(fake.executor.submissions)

    def test_refuses_cross_arm_packed_change_and_timing_cache(self):
        for fake in (FakeAdapter(bad_kv=True),FakeAdapter(existing_timing=True)):
            with self.subTest(fake=fake):
                result=self.execute(fake)
                self.assertEqual(result["status"],"execution_incomplete")
                self.assertEqual(result["performance_gate"],"not_evaluated")
                self.assertFalse(fake.executor.submissions)

    def test_after_start_cutoff_has_no_access(self):
        fake=FakeAdapter()
        result=self.execute(fake,lambda:AFTER_START)
        self.assertEqual(result["status"],"skipped_budget")
        self.assertFalse(fake.parity_calls)
        self.assertFalse(self.journal)

    def test_frozen_binding_does_not_choose_from_test_scores(self):
        source = {"rsi_memory/topology_rsi.py":"a"*64, "v3.py":"b"*64}
        spec = {**SPEC,"worker_source_manifest_sha256":"c"*64,"launch_sha256":"d"*64,
                "tokenization_sha256":"e"*64,"metadata_profile_sha256":"f"*64,"margin_hand_sweep_canonical_sha256":"1"*64}
        contract = {"cases":{"test":CASES},"source_manifest_sha256":spec["worker_source_manifest_sha256"],
                    "launch_sha256":spec["launch_sha256"],"tokenization_sha256":spec["tokenization_sha256"],
                    "metadata_profile_sha256":spec["metadata_profile_sha256"],
                    "manual_selection":{"sweep_sha256":spec["margin_hand_sweep_canonical_sha256"]},
                    "adapter_sources_sha256":{},"generator_program_protocol":{"kind":"backend_program_contract_v3",
                                                                              "source_sha256":{"v3.py":"b"*64}}}
        manifest={"config":{"rounds":2,"min_improvement":MARGIN},"contract":contract,
                  "source_sha256":{"topology_rsi.py":"a"*64}}
        rounds=[{"round":0},{"round":1}]
        frozen={"partial":False,"rounds":2,"manifest_sha256":canonical(manifest),"rounds_sha256":canonical(rounds),
                "admitted_harness_sha256s":["2"*64],"adaptive":{"trial":{
                    "program":PROGRAM,"program_sha256":canonical(PROGRAM),"harness_sha256":"2"*64,"id":"r0001/child/00"}}}
        report={"status":"test_settled","frozen":frozen,"manifest":manifest,"rounds":rounds,
                "paired_evaluations":{"pair/frozen/test":{"status":"completed","input":{"split":"test","programs":{"adaptive":PROGRAM}},
                    "result":{"adaptive":{"feasible":False},"fixed":{"feasible":True,"score":1}}}}}
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            ctrl=root/"controller"
            ctrl.mkdir()
            for filename,value in (("manifest.json",manifest),("frozen.json",frozen),("report.json",report)):
                (ctrl/filename).write_text(json.dumps(value),encoding="utf-8")
            seal=digest((ctrl/"report.json").read_bytes())
            program,base,selected=frozen_selection(root,seal,spec,source)
            self.assertEqual(program,PROGRAM)
            self.assertEqual(selected["adaptive_trial_id"],"r0001/child/00")
            self.assertNotIn("generator_program_protocol",base)
            with self.assertRaisesRegex(ValueError,"authorized"):
                frozen_selection(root,"0"*64,spec,source)

    def test_waiter_only_uses_stage_status_and_clock(self):
        def row(phase,rc,start,end):
            return {"phase":phase,"returncode":rc,"started_utc":start,"finished_utc":end}
        with tempfile.TemporaryDirectory() as d:
            path=Path(d)
            self.assertIsNone(phase_status(path,BEFORE))
            marker=path/"pipeline-status.json"
            entries=[row("run",0,"2026-11-10T23:20:00Z","2026-11-10T23:30:00Z"),
                     row("freeze",0,"2026-11-10T23:30:01Z","2026-11-10T23:31:00Z")]
            marker.write_text(json.dumps(entries),encoding="utf-8")
            self.assertIsNone(phase_status(path,BEFORE))
            late=row("test",0,"2026-11-10T23:31:01Z","2026-11-11T00:00:01Z")
            marker.write_text(json.dumps(entries+[late]),encoding="utf-8")
            self.assertEqual(phase_status(path,AFTER_START),"formal_pipeline_finished_after_cutoff")
            on_time={**late,"finished_utc":"2026-11-10T23:50:01Z"}
            marker.write_text(json.dumps(entries+[on_time]),encoding="utf-8")
            self.assertEqual(phase_status(path,BEFORE),entries+[on_time])
            marker.write_text(json.dumps(entries+[{**on_time,"returncode":1}]),encoding="utf-8")
            self.assertEqual(phase_status(path,BEFORE),"formal_pipeline_failed")

if __name__=="__main__":
    unittest.main()
