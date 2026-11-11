"""Freeze, run, and verify a same-IR v11/v13 CUDA engineering comparison."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
import math
from pathlib import Path, PurePosixPath
import re
from statistics import median
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

PARENT_PLAN = "8da4c6d303f22133e5084f47dd39a0cc970d26f3faaa4b8375a1f93c63764390"
V11_ARCHIVE = "afca095a53ddd69f038ff7a9806e76da72f8b7411a54b8448ad3494dfedb6aac"
V11_MANIFEST = "944183b276b231c2dd306dbfc9ddbbf9823d94875e21a94b76008ef2b91ba0a8"
HISTORICAL_RECEIPT = "e7d457e9d8fd972a34e5dc8ffe91c79f93efc09a6c50ce90b56076ac40bd1e0f"
V13_PROTOTYPE = "8c3798f8feab6e281d650f025a247c416662eec1c49f304a587b3493ae85a7fb"
HISTORICAL_JOB = "qualification-v11-paired-gpu3-b0-h64-lru-equivalent.json"
MODULE = "integrations.topology_memory.torch_eval"
RUNTIME = "integrations/topology_memory/torch_runtime.py"
LABELS = ("native","lru-equivalent-v11","lru-equivalent-v13")
SCOPE = "v13_single_gpu3_same_producer_retain_ir_engineering"
HEX = re.compile(r"[0-9a-f]{64}")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(value):
    return digest(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode())


def nice(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=True,allow_nan=False)+"\n").encode()


def sealed(path,expected):
    raw=path.read_bytes()
    require(digest(raw)==expected,"frozen bytes differ for "+path.name)
    return raw


def write(path,raw):
    with path.open("xb") as stream:
        stream.write(raw)


def schedule():
    # Six Williams rows for three arms, then the first row and its reverse.
    orders=((0,1,2),(1,2,0),(2,0,1),(2,1,0),(0,2,1),(1,0,2),(0,1,2),(2,1,0))
    return [{"block":block,"order":[LABELS[i] for i in order]} for block,order in enumerate(orders)]


def read_source(raw, expected_manifest=None):
    with ZipFile(BytesIO(raw)) as package:
        manifest=package.read("SOURCE_MANIFEST.json")
        require(expected_manifest is None or digest(manifest)==expected_manifest,"source manifest bytes differ")
        expected=json.loads(manifest)["files"]
        names=package.namelist()
        require(len(names)==len(set(names)) and set(names)=={"SOURCE_MANIFEST.json",*expected},"source roster differs")
        entries={}
        for name,value in expected.items():
            path=PurePosixPath(name)
            require(not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name,"unsafe source member")
            entries[name]=package.read(name)
            require(digest(entries[name])==value,"source file SHA differs: "+name)
    return manifest,entries


def make_source(entries):
    manifest=nice({"files":{name:digest(raw) for name,raw in entries.items()}})
    result=BytesIO()
    with ZipFile(result,"w",compression=ZIP_DEFLATED,compresslevel=9) as archive:
        for name,raw in sorted({**entries,"SOURCE_MANIFEST.json":manifest}.items()):
            item=ZipInfo(name,date_time=(2020,1,1,0,0,0))
            item.external_attr=0o644<<16
            archive.writestr(item,raw,compress_type=ZIP_DEFLATED,compresslevel=9)
    return result.getvalue(),manifest


def producer_work(row):
    return row["work_counts"].get("allocate/replicate", {})


def producer_actions(row):
    return sum(action["action"]=="retain_producer_replica" for action in row["actions"])


def observed_producer_ir(raw, parent):
    receipt=json.loads(raw)
    result=receipt["result"]
    program=parent["arms"]["lru-equivalent"]["program"]
    require(receipt["job_id"]+".json"==HISTORICAL_JOB and receipt["status"]=="completed"
            and receipt["source_manifest_sha256"]==V11_MANIFEST
            and canonical(result)==receipt["result_canonical_sha256"],"historical diagnostic receipt differs")
    counts={}
    for case in result["cases"]:
        diag=case["diagnostic"]
        require(case["case_sha256"]==parent["case_sha256"][case["name"]]
                and diag["program_sha256"]==canonical(program),"historical diagnostic case/IR differs")
        count=producer_actions(diag)
        require(count>0 and producer_work(diag)==dict.fromkeys(("choice_calls","candidate_records","predicate_evals","score_evals"),count),
                "LRU IR did not exercise only single-candidate producer retention")
        counts[case["name"]]=count
    require(counts.keys()==parent["case_sha256"].keys(),"missing historical retain coverage")
    return {"receipt":HISTORICAL_JOB,"receipt_sha256":digest(raw),"generic_producer_choices_by_case":counts}


def prepare(args):
    parent=json.loads(sealed(args.parent_plan,PARENT_PLAN))
    payload=parent["payload"]
    require(parent["source_manifest_sha256"]==V11_MANIFEST and parent["gpu_indices"]==[3]
            and parent["host_cpu_affinity"]==[64] and payload["warmup"]==3 and payload["samples"]==5
            and parent["public_training_only"] and not parent["heldout"]
            and len(payload["cases"])==4 and {case["name"]:canonical(case) for case in payload["cases"]}==parent["case_sha256"],
            "public training geometry differs")
    witness=observed_producer_ir(sealed(args.historical_receipt,HISTORICAL_RECEIPT),parent)
    v11=sealed(args.v11_source,V11_ARCHIVE)
    m11,files=read_source(v11,V11_MANIFEST)
    patched=sealed(args.prototype_root/RUNTIME,V13_PROTOTYPE)
    old_gate=b"or len(expected) < 1"
    require(files[RUNTIME].count(old_gate)==patched.count(old_gate)==1,"single GPU gate differs")
    v13,m13=make_source({**files,RUNTIME:patched})
    _,check=read_source(v13,digest(m13))
    require(set(check)==set(files) and {name for name in files if files[name]!=check[name]}=={RUNTIME},
            "unexpected v13 source changes")
    args.output.mkdir(exist_ok=False,parents=True)
    program=parent["arms"]["lru-equivalent"]["program"]
    plan={"schema":1,"kind":SCOPE,"script_sha256":digest(Path(__file__).read_bytes()),"parent_plan_sha256":PARENT_PLAN,
          "scope":"engineering on four historical public training fixtures, single GPU3, CPU64",
          "selection_reason":"The historical LRU-equivalent arm has exactly one generic allocate/replicate candidate per producer retain. The case roster and IR are fixed from those diagnostics before timing.",
          "producer_retain_witness":witness,"gpu_indices":[3],"host_cpu_affinity":[64],"timeout_seconds":parent["job_timeout_seconds"],
          "payload":payload,"case_sha256":parent["case_sha256"],"schedule":schedule(),
          "arms":{"native":{"source":"v11","arm":"native_lru","program":None},
                  "lru-equivalent-v11":{"source":"v11","arm":"generated","program":program},
                  "lru-equivalent-v13":{"source":"v13","arm":"generated","program":program}},
          "program_sha256":canonical(program),"single_gpu_gate_preserved":True,
          "source_versions":{"v11":{"archive_sha256":digest(v11),"manifest_sha256":digest(m11),"runtime_sha256":digest(files[RUNTIME])},
                             "v13":{"archive_sha256":digest(v13),"manifest_sha256":digest(m13),"runtime_sha256":digest(patched)}},
          "analysis":"Compare per-block case-median sums and per-case medians. Same-IR v11/v13 requires ordered actions, lifecycle, bytes, capacity, selected bytes, outputs and timed signatures to match. Diagnostic producer work must change from generic single-candidate evaluation to the proved fast path; all other policy counts must match. Native is an independent reference."}
    for name,raw in (("v11.source.zip",v11),("v11.source-manifest.json",m11),("v13.source.zip",v13),
                     ("v13.source-manifest.json",m13),("preregistration.json",nice(plan))):
        write(args.output/name,raw)
    print(json.dumps({"plan_sha256":digest(nice(plan)),"jobs":24,"sources":plan["source_versions"]},sort_keys=True))


def plan_for(args):
    raw=sealed(args.plan,args.expected_plan_sha256)
    plan=json.loads(raw)
    require(plan["kind"]==SCOPE and plan["script_sha256"]==digest(Path(__file__).read_bytes())
            and plan["schedule"]==schedule() and plan["payload"]["warmup"]==3 and plan["payload"]["samples"]==5
            and plan["arms"]["lru-equivalent-v11"]["program"]==plan["arms"]["lru-equivalent-v13"]["program"]
            and canonical(plan["arms"]["lru-equivalent-v11"]["program"])==plan["program_sha256"],"frozen plan differs")
    source={}
    for version,desc in plan["source_versions"].items():
        raw_source=sealed(args.plan.parent/(version+".source.zip"),desc["archive_sha256"])
        manifest,entries=read_source(raw_source,desc["manifest_sha256"])
        require(manifest==sealed(args.plan.parent/(version+".source-manifest.json"),desc["manifest_sha256"])
                and digest(entries[RUNTIME])==desc["runtime_sha256"],"staged source bytes differ")
        source[version]=entries
    return plan,source


def entries(plan,plan_sha):
    for block in plan["schedule"]:
        for label in block["order"]:
            arm=plan["arms"][label]
            job=f"qualification-v13-{plan_sha[:12]}-b{block['block']}-{label}"
            yield block["block"],label,arm["source"],job,{**plan["payload"],"arm":arm["arm"],"program":arm["program"]}


def run(args,executor_type=None):
    plan,source=plan_for(args)
    config=json.loads(args.executor_config.read_bytes())
    require(set(config)=={"host","worker","root","python","source","gpu_indices","gpu_uuids","environment"}
            and config["gpu_indices"]==plan["gpu_indices"] and config["gpu_uuids"]==plan["payload"]["expected_gpu_uuids"]
            and config["environment"]=={"OMP_NUM_THREADS":"2"} and args.v13_source!=config["source"],
            "operator config differs from frozen GPU3 and distinct source bindings")
    if executor_type is None:
        root=args.controller_root.resolve(strict=True)
        for name in ("rsi_memory/execution_client.py","rsi_memory/execution_worker.py"):
            require((root/name).read_bytes()==source["v11"][name],"local executor source differs")
        sys.dont_write_bytecode=True
        sys.path.insert(0,str(root))
        from rsi_memory.execution_client import PersistentExecutor
        executor_type=PersistentExecutor
    args.audit_directory.mkdir(exist_ok=False,parents=True)
    with ExitStack() as stack:
        executors={}
        for version in ("v11","v13"):
            operator=config if version=="v11" else {**config,"source":args.v13_source}
            executors[version]=stack.enter_context(executor_type(**operator,audit_directory=args.audit_directory/version,
                source_manifest_sha256=plan["source_versions"][version]["manifest_sha256"]))
        binding={"schema":1,"plan_sha256":args.expected_plan_sha256,
                 "request_fields_by_source":{version:executor.config for version,executor in executors.items()}}
        write(args.audit_directory/"run-binding.json",nice(binding))
        for block,label,version,job,payload in entries(plan,args.expected_plan_sha256):
            client=executors[version]
            require(client.status(job)["status"]=="not_found","fresh timed job already exists: "+job)
            receipt=client.run(job,MODULE,payload,timeout=plan["timeout_seconds"])
            path=args.audit_directory/version/(job+".json")
            event={"block":block,"arm":label,"source":version,"job_id":job,"receipt_sha256":digest(path.read_bytes())}
            with (args.audit_directory/"jobs.jsonl").open("ab") as log:
                log.write(json.dumps(event,sort_keys=True).encode()+b"\n")
            print(json.dumps({**event,"status":receipt["status"]}),flush=True)
            require(receipt["status"]=="completed" and receipt["result"].get("feasible") is True,"inspect immutable failed receipt "+job)


def signature(row):
    # The runtime sorts integer fork ownership keys before JSON serialization.
    semantics = [{**event,"owners_after":{int(key):count for key,count in event["owners_after"].items()}}
                 if "owners_after" in event else event for event in row["semantics"]]
    return canonical({"actions":row["actions"],"semantics":semantics,
                      "selections":[[a["step"],a["session"],a["layer"],a["selected"]] for a in row["accesses"]],
                      "link_bytes":row["link_bytes"],"program":row["program_sha256"],"topology":row["topology_sha256"],
                      "scenario":row["scenario_sha256"],"reserved":row["managed_reserved_bytes_by_node"],
                      "live_references_at_end":row["live_references_at_end"],"residual":row["residual_before_teardown_bytes_by_node"]})


def selected(row):
    return canonical([{k:a[k] for k in ("step","layer","selected","selected_sha256","output_sha256")} for a in row["accesses"]])


def physical(row):
    fields=("actions","semantics","link_bytes","shared_bottleneck_bytes","managed_peak_bytes_by_node",
            "managed_reserved_bytes_by_node","residual_before_teardown_bytes_by_node","live_references_at_end",
            "page_activity","total_fetch_misses")
    return {k:row[k] for k in fields}


def check_receipt(receipt,request_fields,plan,job,payload):
    source=request_fields["source_manifest_sha256"]
    request={"operation":"run","job_id":job,"module":MODULE,"payload":payload,"timeout":plan["timeout_seconds"],**request_fields}
    result=receipt["result"]
    require(receipt["schema"]==2 and receipt["job_id"]==job and receipt["status"]=="completed"
            and receipt["source_manifest_sha256"]==source and receipt["request_sha256"]==canonical(request)
            and receipt["gpu_indices"]==request_fields["gpu_indices"] and receipt["gpu_uuids"]==request_fields["gpu_uuids"]
            and receipt["returncode"]==0 and not any(receipt[k] for k in ("timeout","failure","environment_conflict",
              "process_group_cleanup_required","process_group_remaining","remaining_compute_pids"))
            and receipt.get("replayed_receipt") is not True and canonical(result)==receipt["result_canonical_sha256"]
            and HEX.fullmatch(receipt["result_sha256"]),"unbound or failed receipt "+job)
    require(result["scope"]=="cuda_text_derived_fixed_block_sparse" and result["feasible"] is True
            and result["score_unit"]=="sum_case_median_ms","unbound evaluator score "+job)
    hardware = result["hardware"]
    expected_uuid = request_fields["gpu_uuids"][0].lower().removeprefix("gpu-")
    require(len(hardware)==1 and hardware[0]["ordinal"]==0
            and hardware[0]["uuid"].lower().removeprefix("gpu-")==expected_uuid,"live CUDA device differs from leased GPU3")
    case_map={}
    program=None if payload["program"] is None else plan["program_sha256"]
    for case in result["cases"]:
        name,diag=case["name"],case["diagnostic"]
        require(name not in case_map and case["case_sha256"]==plan["case_sha256"][name] and diag["program_sha256"]==program
                and diag["status"]=="verified" and diag["arm"]==payload["arm"] and diag["live_references_at_end"]==0
                and selected(diag)==diag["selected_and_output_sha256"] and signature(diag)==case["execution_signature"],
                "case, selected KV, output or action signature differs: "+job)
        rows=case["timings"]
        require(len(rows)==5 and all(row["status"]=="timed" and row["program_sha256"]==program and "work_counts" not in row
               and row["execution_signature"]==case["execution_signature"] and row["schedule_sha256"]==diag["schedule_sha256"]
               and row["link_bytes"]==diag["link_bytes"] and row["managed_peak_bytes_by_node"]==diag["managed_peak_bytes_by_node"]
               and row["total_fetch_misses"]==diag["total_fetch_misses"] and type(row["elapsed_ns"]) is int and row["elapsed_ns"]>0
               for row in rows),"timings do not match verified execution: "+job)
        require(median(row["elapsed_ns"] for row in rows)==case["median_elapsed_ns"],"case median differs "+job)
        case_map[name]=case
    require(case_map.keys()==plan["case_sha256"].keys()
            and result["score"]==sum(case["median_elapsed_ns"] for case in case_map.values())/1e6,"job score differs "+job)
    return result,case_map


def analyze(args):
    plan,_=plan_for(args)
    directory=args.audit_directory
    binding=json.loads((directory/"run-binding.json").read_bytes())
    fields=binding["request_fields_by_source"]
    require(binding["plan_sha256"]==args.expected_plan_sha256 and set(fields)=={"v11","v13"},"run binding differs")
    for version,value in fields.items():
        require(set(value)=={"source","source_manifest_sha256","gpu_indices","gpu_uuids","environment"}
                and value["source_manifest_sha256"]==plan["source_versions"][version]["manifest_sha256"]
                and value["gpu_indices"]==[3] and value["gpu_uuids"]==plan["payload"]["expected_gpu_uuids"]
                and value["environment"]=={"OMP_NUM_THREADS":"2"},"executor binding differs")
    ledger=[json.loads(line) for line in (directory/"jobs.jsonl").read_bytes().splitlines()]
    planned=list(entries(plan,args.expected_plan_sha256))
    require(len(ledger)==len(planned)==24,"paired study incomplete")
    results={}
    current_end=None
    processes=set()
    hardware=None
    for logged,(block,label,version,job,payload) in zip(ledger,planned,strict=True):
        require({k:logged[k] for k in ("block","arm","source","job_id")}==
                {"block":block,"arm":label,"source":version,"job_id":job},"actual order differs")
        receipt=json.loads(sealed(directory/version/(job+".json"),logged["receipt_sha256"]))
        result,cases=check_receipt(receipt,fields[version],plan,job,payload)
        start,end=(datetime.fromisoformat(receipt[k]) for k in ("utc_start","utc_end"))
        proc=receipt["pid"],receipt["proc_starttime"]
        require(start<end and (current_end is None or current_end<=start) and proc not in processes,"worker leases overlapped or replayed")
        current_end=end
        processes.add(proc)
        if hardware is None:
            hardware=result["hardware"]
        require(result["hardware"]==hardware,"GPU identity changed")
        results[block,label]=result,cases
    blocks=[]
    for row in plan["schedule"]:
        block=row["block"]
        values={label:results[block,label] for label in LABELS}
        case_values={}
        for name in plan["case_sha256"]:
            c={label:pair[1][name] for label,pair in values.items()}
            a,b=c["lru-equivalent-v11"],c["lru-equivalent-v13"]
            native=c["native"]["diagnostic"]
            d11,d13=a["diagnostic"],b["diagnostic"]
            require(a["execution_signature"]==b["execution_signature"] and physical(d11)==physical(d13)
                    and d11["selected_and_output_sha256"]==d13["selected_and_output_sha256"]==native["selected_and_output_sha256"]
                    and all(d11[k]==d13[k]==native[k] for k in ("schedule_sha256","workload_sha256","topology_sha256",
                                                                "scenario_sha256","managed_reserved_bytes_by_node")),
                    "same-IR physical execution or native correctness differs")
            expected=plan["producer_retain_witness"]["generic_producer_choices_by_case"][name]
            require(producer_actions(d11)==producer_actions(d13)==expected
                    and producer_work(d11)==dict.fromkeys(("choice_calls","candidate_records","predicate_evals","score_evals"),expected)
                    and producer_work(d13)=={"static_producer_retain_choices":expected}
                    and {k:v for k,v in d11["work_counts"].items() if k!="allocate/replicate"}==
                        {k:v for k,v in d13["work_counts"].items() if k!="allocate/replicate"},
                    "producer retention bypass or other work differs")
            case_values[name]={label:{"median_ms":case["median_elapsed_ns"]/1e6,
                                     "samples_ns":[t["elapsed_ns"] for t in case["timings"]]} for label,case in c.items()}
        scores={label:pair[0]["score"] for label,pair in values.items()}
        blocks.append({**row,"case_scores":case_values,"sum_ms_by_arm":scores,
                       "old_minus_new_ms":scores["lru-equivalent-v11"]-scores["lru-equivalent-v13"],
                       "native_minus_new_ms":scores["native"]-scores["lru-equivalent-v13"]})
    gains=[b["old_minus_new_ms"] for b in blocks]
    native=[b["native_minus_new_ms"] for b in blocks]
    per_case={}
    for name in plan["case_sha256"]:
        times=[b["case_scores"][name] for b in blocks]
        old_deltas=[row["lru-equivalent-v11"]["median_ms"]-row["lru-equivalent-v13"]["median_ms"] for row in times]
        native_deltas=[row["native"]["median_ms"]-row["lru-equivalent-v13"]["median_ms"] for row in times]
        per_case[name]={"old_minus_new_paired_median_ms":median(old_deltas),
                        "old_minus_new_positive_blocks":sum(value>0 for value in old_deltas),
                        "native_minus_new_paired_median_ms":median(native_deltas),
                        "native_minus_new_positive_blocks":sum(value>0 for value in native_deltas)}
    answer={"schema":1,"scope":SCOPE,"status":"completed","plan_sha256":args.expected_plan_sha256,"jobs":24,
            "blocks":blocks,"old_minus_new_paired_median_ms":median(gains),"old_minus_new_positive_blocks":sum(x>0 for x in gains),
            "native_minus_new_paired_median_ms":median(native),"native_minus_new_positive_blocks":sum(x>0 for x in native),
            "case_comparisons":per_case,"same_ir_physical_trace_verified":True,
            "expected_producer_bypass_verified":True,"hardware":hardware}
    write(args.output,nice(answer))
    print(json.dumps({k:v for k,v in answer.items() if k!="blocks"},sort_keys=True))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    phases=parser.add_subparsers(dest="phase",required=True)
    prep=phases.add_parser("prepare")
    for flag in ("parent-plan","historical-receipt","v11-source","prototype-root","output"):
        prep.add_argument("--"+flag,required=True,type=Path)
    execute=phases.add_parser("run")
    analyze_parser=phases.add_parser("analyze")
    for sub in (execute,analyze_parser):
        sub.add_argument("--plan",required=True,type=Path)
        sub.add_argument("--expected-plan-sha256",required=True)
        sub.add_argument("--audit-directory",required=True,type=Path)
    for flag in ("executor-config","controller-root"):
        execute.add_argument("--"+flag,required=True,type=Path)
    execute.add_argument("--v13-source",required=True)
    analyze_parser.add_argument("--output",required=True,type=Path)
    args=parser.parse_args()
    {"prepare":prepare,"run":run,"analyze":analyze}[args.phase](args)


if __name__=="__main__":
    main()
