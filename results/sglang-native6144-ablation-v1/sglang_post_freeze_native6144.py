"""Run a preregistered post-freeze SGLang comparison with real native 6144 all-host."""
import argparse
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
from statistics import median
import sys
from zipfile import ZipFile

NATIVE = "native6144_all_host"
ADAPTIVE = "frozen_adaptive"
FORMAT = "sglang_post_freeze_native6144_v1"
MATCH = ("effective_schedule_sha256", "dispatch_sha256", "output_ids_sha256", "runtime_sha256",
         "global_ticks_sha256", "input_tokenization_sha256")
PARITY = (*MATCH, "packed_kv_sha256_by_rank")
SHA = re.compile(r"[a-f0-9]{64}")
JOB = re.compile(r"sg-[a-f0-9]{48}")
MARGIN = 11.49229000000014
START_BY = "2026-11-11T00:00:00Z"
STOP_SUBMIT = "2026-11-11T00:20:00Z"


class BudgetExpired(RuntimeError):
    pass


def require(predicate, message):
    if not predicate:
        raise ValueError(message)


def utc_now():
    return datetime.now(timezone.utc)


def time_utc(value):
    stamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    require(stamp.utcoffset() == timezone.utc.utcoffset(None), "expected timestamp in UTC")
    return stamp


def digest(data):
    return sha256(data).hexdigest()


def nice(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)+"\n").encode()


def canonical(value):
    return digest(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode())


def read(path):
    require(path.is_file() and not path.is_symlink(), "missing or symlinked " + path.name)
    return path.read_bytes()


def output_json(path, value):
    pending = path.with_name(path.name + ".pending")
    with pending.open("xb") as file:
        file.write(nice(value))
        file.flush()
        os.fsync(file.fileno())
    require(not path.exists(), "output already exists: " + path.name)
    pending.rename(path)


def append(path, event):
    with path.open("ab") as file:
        file.write(json.dumps(event,sort_keys=True,allow_nan=False).encode()+b"\n")
        file.flush()
        os.fsync(file.fileno())


def require_sha(value, what):
    require(isinstance(value,str) and SHA.fullmatch(value), "missing SHA-256 for " + what)


def fixed_blocks():
    return [{"block_id":f"n6144-b{number:02d}", "order":[ADAPTIVE,NATIVE] if number%2 else [NATIVE,ADAPTIVE]}
            for number in range(1,9)]


def preregistration(path, expected_sha, driver):
    raw = read(path)
    require_sha(expected_sha,"preregistration")
    require(digest(raw)==expected_sha,"preregistration bytes differ")
    spec = json.loads(raw)
    require(spec["schema"]==1 and spec["kind"]==FORMAT and spec["blocks"]==fixed_blocks()
            and spec["cutoffs"]=={"latest_start_utc":START_BY,"no_new_submissions_at_or_after_utc":STOP_SUBMIT}
            and spec["decision"]=={"positive_in_every_block":True,"median_strictly_greater_than_ms":MARGIN}
            and spec["timed_job_count"]==32 and len(spec["cases"])==2 and len({case["id"] for case in spec["cases"]})==2
            and spec["native"]=={"arm":"native","native_hot":6144,"program":None,"backing":"all_host"}
            and spec["adaptive"]=={"arm":"generated","program_source":"frozen.adaptive.trial.program"}
            and spec["lease_timeouts"]=={"serving_deadline_seconds":300,"lease_timeout_seconds":780},
            "preregistration differs from the fixed design")
    require(digest(read(driver))==spec["driver_sha256"],"driver does not match preregistration")
    return raw,spec


def check_source(root, manifest_file, archive_file, spec):
    manifest_raw, archive_raw = read(manifest_file), read(archive_file)
    require(digest(manifest_raw)==spec["controller_source_manifest_sha256"]
            and digest(archive_raw)==spec["controller_source_archive_sha256"],"frozen source commitment differs")
    manifest = json.loads(manifest_raw)
    require(manifest["worker_source_manifest_sha256"]==spec["worker_source_manifest_sha256"],"worker source differs")
    inventory = manifest["files"]
    with ZipFile(BytesIO(archive_raw)) as archive:
        names = archive.namelist()
        require(len(set(names))==len(names) and set(names)=={"LOCAL_SOURCE_MANIFEST.json",*inventory}
                and archive.read("LOCAL_SOURCE_MANIFEST.json")==manifest_raw,"controller archive roster differs")
        for name, expected in inventory.items():
            parts = PurePosixPath(name)
            require(not parts.is_absolute() and ".." not in parts.parts, "unsafe source path")
            require(digest(archive.read(name))==digest(read(root/parts))==expected, "frozen source differs: " + name)
    return inventory


def phase_status(root, now):
    path = root / "pipeline-status.json"
    if not path.exists():
        return None
    rows = json.loads(read(path))
    require(isinstance(rows,list) and [r["phase"] for r in rows]==["run","freeze","test"][:len(rows)]
            and all(type(r["returncode"]) is int for r in rows), "invalid formal pipeline status")
    if any(row["returncode"]!=0 for row in rows):
        return "formal_pipeline_failed"
    if len(rows)<3:
        return None
    starts = [time_utc(row["started_utc"]) for row in rows]
    ends = [time_utc(row["finished_utc"]) for row in rows]
    require(all(a<=b for a,b in zip(starts,ends,strict=True)) and all(a<=b for a,b in zip(ends,starts[1:]))
            and ends[-1]<=now, "formal pipeline phase timestamps overlap or extend into the future")
    if ends[-1]>time_utc(START_BY):
        return "formal_pipeline_finished_after_cutoff"
    return rows


def frozen_selection(root, expected_report_sha, spec, source_inventory):
    require_sha(expected_report_sha,"settled formal report")
    report_bytes = read(root/"controller/report.json")
    require(digest(report_bytes)==expected_report_sha,"formal report differs from the authorized settled bytes")
    report = json.loads(report_bytes)
    frozen_bytes = read(root/"controller/frozen.json")
    frozen = json.loads(frozen_bytes)
    manifest = json.loads(read(root/"controller/manifest.json"))
    require(report["status"]=="test_settled" and report["frozen"]==frozen and report["manifest"]==manifest
            and not frozen["partial"] and frozen["rounds"]==manifest["config"]["rounds"]==2
            and canonical(manifest)==frozen["manifest_sha256"] and canonical(report["rounds"])==frozen["rounds_sha256"],
            "formal pipeline did not seal the complete selected run")
    contract = manifest["contract"]
    protocol = contract["generator_program_protocol"]
    binding_sets = ({f"rsi_memory/{name}":value for name,value in manifest["source_sha256"].items()},
                    contract["adapter_sources_sha256"],protocol["source_sha256"])
    require(protocol["kind"]=="backend_program_contract_v3"
            and all(source_inventory.get(name)==value for binding in binding_sets for name,value in binding.items()),
            "formal run differs from the preregistered controller and program contract source")
    require(contract["cases"]["test"]==spec["cases"]
            and contract["source_manifest_sha256"]==spec["worker_source_manifest_sha256"]
            and contract["launch_sha256"]==spec["launch_sha256"]
            and contract["tokenization_sha256"]==spec["tokenization_sha256"]
            and contract["metadata_profile_sha256"]==spec["metadata_profile_sha256"]
            and manifest["config"]["min_improvement"]==MARGIN
            and contract["manual_selection"]["sweep_sha256"]==spec["margin_hand_sweep_canonical_sha256"],
            "formal test cases, evaluator, or training margin differ")
    trial = frozen["adaptive"]["trial"]
    program = trial["program"]
    program_sha = canonical(program)
    require(trial["program_sha256"]==program_sha and trial["harness_sha256"] in frozen["admitted_harness_sha256s"],
            "selected adaptive program lacks frozen admitted provenance")
    terminal = report["paired_evaluations"]["pair/frozen/test"]
    require(terminal["status"] in {"completed","failed","interrupted_unknown"}
            and terminal["input"]["split"]=="test"
            and terminal["input"]["programs"]["adaptive"]==program,
            "formal heldout effect did not settle on the frozen adaptive program")
    adapter_contract = {name:value for name,value in contract.items() if name!="generator_program_protocol"}
    return program,adapter_contract,{"formal_report_sha256":expected_report_sha,"formal_frozen_sha256":digest(frozen_bytes),
        "formal_manifest_canonical_sha256":frozen["manifest_sha256"],"adaptive_program_sha256":program_sha,
        "adaptive_trial_id":trial["id"],"adaptive_harness_sha256":trial["harness_sha256"]}


class ExecutorGate:
    """Check the UTC budget at the actual submission point, including recursive qualification."""

    def __init__(self, inner, clock):
        self.inner = inner
        self.config = inner.config
        self.clock = clock
        self.require_new_timing = False

    def status(self, job_id):
        receipt = self.inner.status(job_id)
        if self.require_new_timing:
            require(receipt["status"]=="not_found","timed receipt already exists for " + job_id)
        return receipt

    def run(self, job_id, module, payload, *, timeout):
        if self.clock()>=time_utc(STOP_SUBMIT):
            raise BudgetExpired("preregistered GPU submission cutoff reached")
        require(timeout==780 and payload["deadline"]==300,"frozen lease or serving timeout changed")
        return self.inner.run(job_id,module,payload,timeout=timeout)

    def close(self):
        self.inner.close()


def validate_result(job, result, label, phase, case, program_sha):
    require(isinstance(job,str) and JOB.fullmatch(job),"unbound leased job ID")
    require(result["case"]==case and result["feasible"] is True and result["phase"]==phase
            and result["arm"]==("native" if label==NATIVE else "generated")
            and result["program_sha256"]==(None if label==NATIVE else program_sha), "infeasible or mismatched " + phase + " result")
    require_sha(result["raw_result_sha256"],"raw serving result")
    metrics = result["metrics"]
    for field in MATCH:
        require_sha(metrics.get(field),field)
    require(metrics["global_ticks"]["global_ticks_sha256"]==metrics["global_ticks_sha256"]
            and metrics["ingress"]["contract_sha256"]==result["memory_compilation"]["ingress_contract_sha256"],
            "actual global tick or ingress commitment differs")
    if label==NATIVE:
        startup = result["memory_compilation"]["startup"]
        require(startup["hot_slots"]==startup["native_reference_override"]==6144 and startup["backing_mode"]=="tiered"
                and metrics["backing_mode"]=="tiered" and len(metrics["ranks"])==2
                and [row["rank"] for row in metrics["ranks"]]==[0,1]
                and all(row["hot_slots"]==6144 and row["layer_tiers"]==["host"]*6
                        and row["local_hbm_bytes"]==row["peer_hbm_bytes"]==0 and row["pinned_host_bytes"]>0
                        for row in metrics["ranks"]),"comparison arm is not native 6144 all-host")
        if phase=="parity":
            require(result["cross_capacity_native_parity"]["matched"] is True,"native 6144 did not match the stock cross-capacity oracle")
    packed = metrics["packed_kv_sha256_by_rank"]
    if phase=="parity":
        require(len(packed)==2,"missing packed KV parity for both ranks")
        for value in packed:
            require_sha(value,"packed KV digest")
    else:
        require(packed==[] and all(row.get("resident_producer_oracle") is None
                and row.get("selected_miss_payload_bytes_by_tier") is None for row in metrics["ranks"]),
                "timing contains parity instrumentation")
        require(type(result["score"]) in (int,float) and math.isfinite(result["score"]) and result["score"]>0
                and result["score"]==metrics["timing"]["window_seconds"]*1000, "invalid replay window")


def compare(case, left, right, fields):
    require(all(left["metrics"][name]==right["metrics"][name] for name in fields)
            and all(left["memory_compilation"][name]==right["memory_compilation"][name]
                    for name in ("external_scheduler_sha256","ingress_contract_sha256")),
            "paired qualification or timed gate mismatch for " + case["id"])


def run_study(adapter, program, spec, binding, output, clock=utc_now):
    proof_by_case = {}
    startup_by_case = {}
    completed = []
    start = clock()
    if start>time_utc(START_BY):
        return {"status":"skipped_budget","reason":"latest_start_passed","timed_jobs_completed":0,
                "performance_gate":"not_evaluated"}
    timings = set()
    pairs = []
    try:
        for case in spec["cases"]:
            native_id,native = adapter._parity(case,"native",None,slots=6144)
            adaptive_id,adaptive = adapter._parity(case,"generated",program)
            for label,job,value in ((NATIVE,native_id,native),(ADAPTIVE,adaptive_id,adaptive)):
                validate_result(job,value,label,"parity",case,binding["adaptive_program_sha256"])
            compare(case,adaptive,native,PARITY)
            repeat = adapter.native_repeats[case["id"],6144]
            proof_by_case[case["id"]] = {ADAPTIVE:(adaptive_id,adaptive),NATIVE:(native_id,native)}
            actual = {label:{field:value["memory_compilation"]["startup"][field] for field in ("hot_slots","backing_mode")}
                      for label,(_,value) in proof_by_case[case["id"]].items()}
            actual["same_hot_capacity"] = actual[ADAPTIVE]["hot_slots"]==actual[NATIVE]["hot_slots"]
            startup_by_case[case["id"]] = actual
            append(output/"jobs.jsonl",{"phase":"qualification","case":case["id"],
                "native_same_capacity_repeat_job_id":repeat,
                "proofs":{label:adapter._proof(job,result) for label,(job,result) in proof_by_case[case["id"]].items()}})
        for block in spec["blocks"]:
            windows = {label:{} for label in (ADAPTIVE,NATIVE)}
            ids = {label:{} for label in (ADAPTIVE,NATIVE)}
            for case in spec["cases"]:
                measured = {}
                for label in block["order"]:
                    if clock()>=time_utc(STOP_SUBMIT):
                        raise BudgetExpired("preregistered GPU submission cutoff reached")
                    is_native = label==NATIVE
                    identity = {"kind":FORMAT,"preregistration_sha256":binding["preregistration_sha256"],
                                "driver_sha256":spec["driver_sha256"],"formal_frozen_sha256":binding["formal_frozen_sha256"],
                                "block_id":block["block_id"],"case_id":case["id"],"label":label}
                    proof = adapter._proof(*proof_by_case[case["id"]][label])
                    client = adapter._client()
                    client.require_new_timing = True
                    try:
                        job,value = adapter._job(case,"native" if is_native else "generated","performance",
                            None if is_native else program,identity,proof,native_hot=6144 if is_native else None)
                    finally:
                        client.require_new_timing = False
                    require(job not in timings,"timed job reused inside supplement")
                    timings.add(job)
                    append(output/"jobs.jsonl",{"phase":"performance","identity":identity,"job_id":job,
                        "raw_result_sha256":value["raw_result_sha256"],"window_ms":value["score"],"qualification":proof})
                    validate_result(job,value,label,"performance",case,binding["adaptive_program_sha256"])
                    compare(case,value,proof_by_case[case["id"]][label][1],MATCH)
                    measured[label] = value
                    windows[label][case["id"]] = value["score"]
                    ids[label][case["id"]] = job
                    completed.append(job)
                compare(case,measured[ADAPTIVE],measured[NATIVE],MATCH)
            case_order = [case["id"] for case in spec["cases"]]
            totals = {label:sum(values[case] for case in case_order) for label,values in windows.items()}
            pairs.append({"block_id":block["block_id"],"order":block["order"],"window_ms_by_arm_case":windows,
                          "job_id_by_arm_case":ids,"window_ms_sum_by_arm":totals,
                          "native_minus_adaptive_ms":totals[NATIVE]-totals[ADAPTIVE]})
    except BudgetExpired as error:
        return {"status":"budget_incomplete","reason":str(error),"timed_jobs_completed":len(completed),"complete_blocks":pairs,
                "startup_by_case":startup_by_case,"started_utc":start.isoformat(),"performance_gate":"not_evaluated"}
    except Exception as error:
        return {"status":"execution_incomplete","failure_type":type(error).__name__,"reason":str(error),
                "timed_jobs_completed":len(completed),"complete_blocks":pairs,"startup_by_case":startup_by_case,
                "started_utc":start.isoformat(),"performance_gate":"not_evaluated"}
    require(len(pairs)==8 and len(completed)==32,"paired study does not contain all declared trials")
    deltas = [row["native_minus_adaptive_ms"] for row in pairs]
    return {"status":"completed","timed_jobs_completed":len(completed),"complete_blocks":pairs,"median_ms":median(deltas),
            "started_utc":start.isoformat(),"startup_by_case":startup_by_case,
            "practical_margin_ms":MARGIN,"positive_blocks":sum(value>0 for value in deltas),
            "performance_gate":"passed" if min(deltas)>0 and median(deltas)>MARGIN else "failed"}


def create_root(path, formal, source):
    path = path.resolve()
    require(not path.is_relative_to(formal.resolve()) and not path.is_relative_to(source.resolve()),
            "supplement output must be separate from formal run and frozen source")
    path.mkdir(parents=True,exist_ok=False)
    return path


def run(args, clock=utc_now):
    prereg_raw,spec = preregistration(args.preregistration,args.expected_prereg_sha256,Path(__file__))
    source = check_source(args.source_root,args.source_manifest,args.source_archive,spec)
    if not args.execute:
        return {"status":"source_and_preregistration_verified","preregistration_sha256":digest(prereg_raw)}
    require(args.formal_root and args.private_adapter_config and args.output,"execution requires formal run, adapter and output paths")
    now = clock()
    state = phase_status(args.formal_root,now) if now<=time_utc(START_BY) else "latest_start_passed"
    if state is None:
        return {"status":"waiting_for_formal_pipeline"}
    if isinstance(state,str):
        destination = create_root(args.output,args.formal_root,args.source_root)
        result = {"status":"skipped_budget" if state!="formal_pipeline_failed" else "skipped_formal_pipeline_failed",
                  "reason":state,"preregistration_sha256":digest(prereg_raw),"timed_jobs_completed":0,
                  "performance_gate":"not_evaluated"}
        (destination/"preregistration.json").write_bytes(prereg_raw)
        output_json(destination/"result.json",result)
        return result
    program_dict,contract,binding = frozen_selection(args.formal_root,args.expected_formal_report_sha256,spec,source)
    private_raw = read(args.private_adapter_config)
    base = json.loads(private_raw)
    pipeline_plan = json.loads(read(args.formal_root/"pipeline-plan.json"))
    require(pipeline_plan["config_sha256"]==digest(private_raw) and base["schema"]==3 and base["cases"]["test"]==spec["cases"]
            and (base["deadline"],base["job_timeout"])==(300,780),"private config differs from the preregistered formal run")
    destination = create_root(args.output,args.formal_root,args.source_root)
    path = destination/"receipts"
    old_path = (args.private_adapter_config.parent/base["receipt_directory"]).resolve()
    require(path!=old_path and not path.is_relative_to(old_path),"supplement receipts overlap formal originals")
    derived = {**base,"receipt_directory":str(path)}
    require({key for key,value in derived.items() if value!=base[key]}=={"receipt_directory"},"adapter config may change receipt directory only")
    binding.update(preregistration_sha256=digest(prereg_raw),formal_private_config_sha256=digest(private_raw),
        derived_private_config_sha256=digest(nice(derived)),formal_pipeline_status_sha256=digest(read(args.formal_root/"pipeline-status.json")),
        adapter_config_field_diff={"receipt_directory":{"original_value_sha256":canonical(base["receipt_directory"]),
          "replacement_value_sha256":canonical(derived["receipt_directory"]),"public_supplement_path":"receipts"}})
    (destination/"preregistration.json").write_bytes(prereg_raw)
    output_json(destination/"manifest.json",{"schema":1,"kind":FORMAT,"bound":binding,"planned_jobs":32,
                "qualification_reuse":"completed same worker source, unchanged adapter identity, exact request and program hashes"})
    (destination/"frozen_adaptive.program.json").write_bytes(nice(program_dict))
    # The source is imported only after a terminal, on-time formal pipeline is verified.
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(args.source_root.resolve()))
    from rsi_memory.execution_client import PersistentExecutor
    from rsi_memory.memory_program import MemoryProgram
    from topomem_sglang_rsi.domain import SGLangDomain

    def gated_executor(**kwargs):
        require(kwargs["audit_directory"]==path,"executor receipt destination differs")
        return ExecutorGate(PersistentExecutor(**kwargs),clock)

    try:
        adapter = SGLangDomain(derived,args.private_adapter_config.parent,executor_factory=gated_executor)
        try:
            require(adapter.contract==contract and adapter.identity==canonical(contract),"supplement changed formal adapter identity")
            program = MemoryProgram.from_dict(program_dict)
            require(program.fingerprint()==binding["adaptive_program_sha256"],"compiler changed frozen adaptive program")
            result = run_study(adapter,program,spec,binding,destination,clock=clock)
        finally:
            adapter.close()
    except Exception as error:
        result = {"status":"execution_incomplete","failure_type":type(error).__name__,"reason":str(error),
                  "timed_jobs_completed":0,"performance_gate":"not_evaluated"}
    result["preregistration_sha256"] = binding["preregistration_sha256"]
    result["finished_utc"] = clock().isoformat()
    output_json(destination/"result.json",result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("preregistration","source-root","source-manifest","source-archive"):
        parser.add_argument("--"+key,required=True,type=Path)
    parser.add_argument("--expected-prereg-sha256",required=True)
    for key in ("formal-root","private-adapter-config","output"):
        parser.add_argument("--"+key,type=Path)
    parser.add_argument("--expected-formal-report-sha256")
    parser.add_argument("--execute",action="store_true")
    args = parser.parse_args()
    result = run(args)
    print(json.dumps(result,sort_keys=True,allow_nan=False),flush=True)


if __name__=="__main__":
    main()
