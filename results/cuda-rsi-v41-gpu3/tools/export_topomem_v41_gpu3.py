"""Audit and publish the settled v4.1 CUDA engineering experiment offline."""
import argparse
from collections import Counter, defaultdict
from contextlib import closing
from datetime import datetime
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from statistics import median
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

CONTROLLER_SHA = "6fc56c59daa4ed0deea1dfd2ff11ea0efe4b9670fd83b6270eb27a195463f650"
CONTROLLER_ZIP = "1a2c88fe08cff40bd2ab3cf654646ce5c6e3312ae1fc0f891dfb201098273b3d"
WORKER_SHA = "e66bf6796bfec6e14a5980bd8effcab96dd5ba4da7a2ba339103ae9f40e93f5d"
WORKER_ZIP = "8d3a98e1791fdc4fb40564e1341686de466b6e8105d033c7fd7f97459eee52f6"
FORMAL_PLAN = "caea3b9dd4b82b9b2e0412f158e755077f3837cb18e2e98bbe7c1e6b2b143bad"
MODULE = "integrations.topology_memory.torch_eval"
ARMS = ("manager_best","admitted_output","fixed")
ROUND_COUNT = 2
PROVIDER_BUDGET = 20
PRIVATE_CONFIG = "79c1cd08835fc39873c4e512221e55ec17fa1c7d2bbf5b70d4ba9e2f18c413ac"
PIPELINE_SCRIPT = "e214ce9513f67f9938cf002be88c7062a48e0612f3422f106ac87eab4262d08a"
ENGINEERING_PLAN = "19789ad0cd6c55cd3faeed4aab151854f3c0c1f8608c9ccd173ccdfe90dc9e29"
ENGINEERING_RESULT = "10ba4b701f15630f9bf82a1a93d266c852cdb669f42cfcd07983ce74ea81e182"
SCENARIOS = ("agent_tools","chatbot_growth","long_document_qa","rollout_cow")
SHA = re.compile(r"[a-f0-9]{64}")
JOB = re.compile(r"eval-[a-f0-9]{48}")


def need(predicate, message):
    if not predicate:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def canon(value):
    return digest(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode())


def nice(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()


def read(path):
    need(path.is_file() and not path.is_symlink(),"missing or symlinked file "+path.name)
    return path.read_bytes()


def seal(path, expected):
    need(isinstance(expected,str) and SHA.fullmatch(expected),"missing independent SHA-256 for "+path.name)
    raw=read(path)
    need(digest(raw)==expected,"frozen SHA-256 differs for "+path.name)
    return raw


def write(path,raw):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)


def safe_path(name):
    path=PurePosixPath(name)
    need(path.parts and not path.is_absolute() and ".." not in path.parts and "\\" not in name and ":" not in name,
         "unsafe archive path")
    return path


def verify_source(raw,expected,extract=None):
    with ZipFile(BytesIO(raw)) as package:
        manifest=package.read("SOURCE_MANIFEST.json")
        need(digest(manifest)==expected,"source manifest SHA differs")
        files=json.loads(manifest)["files"]
        names=package.namelist()
        need(len(names)==len(set(names)) and set(names)=={"SOURCE_MANIFEST.json",*files},"source archive roster differs")
        for name,commitment in files.items():
            member=package.read(name)
            need(digest(member)==commitment,"frozen source member differs "+name)
            if extract is not None:
                write(extract/safe_path(name),member)
    return manifest,files


def terminal(root,report_sha,pipeline_sha):
    original={"controller/report.json":seal(root/"controller/report.json",report_sha),
              "pipeline-status.json":seal(root/"pipeline-status.json",pipeline_sha)}
    pipeline=json.loads(original["pipeline-status.json"])
    need([row["phase"] for row in pipeline]==["run","freeze","test"]
         and all(type(row["returncode"]) is int and row["returncode"]==0 for row in pipeline),"pipeline did not complete all phases")
    times=[(datetime.fromisoformat(row["started_utc"]),datetime.fromisoformat(row["finished_utc"])) for row in pipeline]
    need(all(start<=finish for start,finish in times) and all(left[1]<=right[0] for left,right in zip(times,times[1:])),
         "pipeline phases overlapped or reversed")
    report=json.loads(original["controller/report.json"])
    need(report["status"]=="test_settled" and report["frozen"]["partial"] is False
         and len(report["rounds"])==report["frozen"]["rounds"]==ROUND_COUNT,"controller report did not settle both rounds")
    original["controller/frozen.json"]=read(root/"controller/frozen.json")
    original["controller/manifest.json"]=read(root/"controller/manifest.json")
    need(json.loads(original["controller/frozen.json"])==report["frozen"]
         and json.loads(original["controller/manifest.json"])==report["manifest"]
         and canon(report["rounds"])==report["frozen"]["rounds_sha256"]
         and canon(report["manifest"])==report["frozen"]["manifest_sha256"],"frozen export differs from terminal report")
    return original,report


def journal(root,original,*,settled=True):
    path=root/"controller/journal.sqlite"
    raw=read(path)
    need(not any(path.with_name(path.name+suffix).exists() for suffix in ("-journal","-wal","-shm")),"journal has an unsettled sidecar")
    original["controller/journal.sqlite"]=raw
    with tempfile.TemporaryDirectory(prefix="topomem-v4-journal-") as tmp:
        copied=Path(tmp)/"journal.sqlite"
        copied.write_bytes(raw)
        with closing(sqlite3.connect(copied.as_uri()+"?mode=ro&immutable=1",uri=True)) as db:
            need(db.execute("PRAGMA integrity_check").fetchall()==[("ok",)],"SQLite integrity check failed")
            rows=db.execute("SELECT key,data FROM records ORDER BY rowid").fetchall()
    values={key:json.loads(value) for key,value in rows}
    order=list(values)
    test="pair/frozen/test"
    if settled:
        need(test in values and order.index("frozen")<order.index(test)
             and [key for key,row in values.items() if key.startswith(("eval/","pair/")) and row["input"]["split"]=="test"]==[test]
             and all(order.index(key)<order.index("frozen") for key in order if key.startswith("api/")),"journal opened test or called provider out of sequence")
    return values,{key:value.encode() for key,value in rows}


def sanitize_config(config):
    private=config["executor"]
    executor={key:"REPLACE_WITH_OPERATOR_"+key.upper() for key in private}
    executor.update(gpu_indices=private["gpu_indices"],gpu_uuids=["REPLACE_WITH_GPU_UUID_"+str(i) for i in private["gpu_indices"]])
    if "environment" in private:
        executor["environment"]={key:value if key=="OMP_NUM_THREADS" else "REPLACE_WITH_OPERATOR_VALUE"
                                 for key,value in private["environment"].items()}
    return {**config,"executor":executor,"output_directory":"REPLACE_WITH_NEW_AUDIT_DIRECTORY"}


def check_geometry(config,case_plan,report):
    contract=report["manifest"]["contract"]
    pre=config["controller_preregistration"]
    limits=case_plan["controller_config"]
    need(pre==contract["controller_preregistration"] and canon(pre)==contract["controller_preregistration_sha256"]
         and all(pre[key]==value for key,value in limits.items())
         and (pre["rounds"],pre["slots"],pre["training_blocks"],pre["validation_blocks"],pre["test_blocks"])==(2,2,3,3,6)
         and pre["training_margin_ms"]==pre["min_improvement_ms"]==2.0 and pre["wall_seconds"]==3600
         and config["source_manifest_sha256"]==case_plan["worker_manifest_sha256"]==contract["source_manifest_sha256"]==WORKER_SHA
         and config["executor"]["gpu_indices"]==[3] and len(config["executor"]["gpu_uuids"])==1
         and (config["warmup"],config["samples"])==(3,5),
         "case plan, GPU lease or controller budget differs")
    need(config["paired_block_ids"]==contract["paired_block_ids"],"paired block IDs changed")
    policies=case_plan["baseline_provenance"]
    need(all(canon(config[label+"_program"])==policies[label+"_program_sha256"] for label in ("initial","manual")),
         "fixed initial or hand-written manager changed")
    provenance=case_plan["origins"]["cases"]
    need(set(config["cases"])==set(provenance)=={"train","validation","test"},"wrong case roster")
    for split,cases in config["cases"].items():
        need(len(cases)==len(provenance[split])==4 and [canon(case) for case in cases]==contract["split_case_sha256"][split],
             "split cases changed")
        for index,(kind,case) in enumerate(zip(SCENARIOS,cases,strict=True)):
            origin=provenance[split][index]
            target=case["target"]
            capacities={node["id"]:node["capacity_bytes"] for node in case["topology"]["nodes"]}
            need(case["workload"]["kind"]==kind and case["compute_node"]=="gpu0" and case["device_map"]=={"gpu0":0}
                 and case["compile"]["head_dim"]==target["head_dim"]==64
                 and target["gpu_uuid"]==config["executor"]["gpu_uuids"][0]
                 and target["host_cpu_affinity"]==[64] and target["profile_sha256"]==case_plan["origins"]["target_profile_sha256"]
                 and capacities=={"gpu0":131072,"host0":33554432}
                 and canon(case)==origin["case_sha256"] and canon(case["compile"])==origin["compile_sha256"]
                 and canon(case["workload"])==origin["workload_sha256"],
                 "case geometry or original roster differs")
    return contract


def verify_sources(report,controller_files):
    manifest=report["manifest"]
    contract=manifest["contract"]
    expected={f"rsi_memory/{n}":v for n,v in manifest["source_sha256"].items()}
    expected["rsi_memory/cuda_domain.py"]=contract["adapter_source_sha256"]
    expected.update(contract["selection_protocol"]["source_sha256"])
    expected.update(contract["generator_program_protocol"]["source_sha256"])
    need(all(controller_files.get(name)==value for name,value in expected.items()),"manifest disagrees with frozen controller code")
    return len(expected)


def replay_controller(source,config,archived,original,report,temporary):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(source))
    from rsi_memory.cuda_domain import CudaDomain
    from rsi_memory.cuda_v4 import checked_preregistration,contract_domain
    from rsi_memory.topology_rsi_v4 import TopologyRSIV4
    from rsi_memory import cuda_v4

    adapter=CudaDomain({**config,"output_directory":str(temporary/"inert-adapter")},temporary)
    pre=config["controller_preregistration"]
    domain=contract_domain(adapter.domain(),pre)
    checked=checked_preregistration(pre,config["paired_block_ids"])
    consumed=set()

    class OfflineController(TopologyRSIV4):
        def _effect(self,key,details,operation,*,time_budget=True):
            need(key in archived and key not in consumed and archived[key]["input"]==details,
                 "offline controller effect changed or repeated "+key)
            consumed.add(key)
            return self._save(key,archived[key])

    def no_remote(_):
        raise AssertionError("offline replay cannot call the provider")

    offline=OfflineController(temporary/"controller-replay",checked,domain,no_remote,backend_source=Path(cuda_v4.__file__))
    try:
        baseline=offline._get("manifest")
        need({k:v for k,v in baseline.items() if k!="started_unix"}==
             {k:v for k,v in archived["manifest"].items() if k!="started_unix"},"replay controller contract changed")
        offline._save("manifest",archived["manifest"])
        need(offline.run()["status"]=="rounds_completed","offline round replay did not complete")
        need(offline.freeze()==report["frozen"],"offline freeze did not reproduce all three finalists")
        need(offline.test()==report,"offline replay differs from the terminal report")
        replayed={key:json.loads(value) for key,value in offline.db.execute("SELECT key,data FROM records")}
        need(replayed==archived,"offline replay did not reproduce every journal record")
        need(consumed=={key for key in archived if key.startswith(("api/","eval/","pair/"))},"offline effects roster differs")
        for filename,raw in original.items():
            if filename.startswith(("controller/harnesses/","controller/rounds/")):
                need(read(temporary/"controller-replay"/filename.removeprefix("controller/"))==raw,
                     "offline replay differs from original generated source or round "+filename)
        return adapter.contract,len(consumed)
    finally:
        offline.close()
        adapter.close()


def execution_signature(diag):
    semantics=[{**event,"owners_after":{int(key):value for key,value in event["owners_after"].items()}}
               if "owners_after" in event else event for event in diag["semantics"]]
    return canon({"actions":diag["actions"],"semantics":semantics,
                 "selections":[[a["step"],a["session"],a["layer"],a["selected"]] for a in diag["accesses"]],
                 "link_bytes":diag["link_bytes"],"program":diag["program_sha256"],"topology":diag["topology_sha256"],
                 "scenario":diag["scenario_sha256"],"reserved":diag["managed_reserved_bytes_by_node"],
                 "live_references_at_end":diag["live_references_at_end"],"residual":diag["residual_before_teardown_bytes_by_node"]})


def selected_digest(diag):
    return canon([{key:a[key] for key in ("step","layer","selected","selected_sha256","output_sha256")} for a in diag["accesses"]])


def audit_result(result,cases,program,arm,samples):
    need(result["scope"]=="cuda_text_derived_fixed_block_sparse" and type(result["feasible"]) is bool,
         "unknown CUDA evaluator result scope")
    observed=result["cases"]
    need(len(observed)<=len(cases),"unexpected case report count")
    expected_program=None if program is None else canon(program)
    facts={}
    for row,case in zip(observed,cases):
        diag=row["diagnostic"]
        sig=execution_signature(diag)
        need(row["case_sha256"]==canon(case) and row["name"]==case["name"] and diag["arm"]==arm
             and diag["program_sha256"]==expected_program and diag["status"]=="verified" and diag["live_references_at_end"]==0
             and diag["accesses"] and selected_digest(diag)==diag["selected_and_output_sha256"] and sig==row["execution_signature"],
             "case, action signature, selected KV or output differs")
        timings=row["timings"]
        need(len(timings)==samples and all(t["status"]=="timed" and "work_counts" not in t
             and t["execution_signature"]==sig and all(t[k]==diag[k] for k in ("program_sha256","schedule_sha256",
                 "workload_sha256","link_bytes","total_fetch_misses","managed_peak_bytes_by_node","managed_reserved_bytes_by_node"))
             and type(t["elapsed_ns"]) is int and t["elapsed_ns"]>0 for t in timings),"timed execution differs from diagnostic")
        need(median(t["elapsed_ns"] for t in timings)==row["median_elapsed_ns"],"sample median differs")
        compute=case["compute_node"]
        peers={n["id"] for n in case["topology"]["nodes"] if n["kind"]=="gpu" and n["id"]!=compute}
        peer_links={link["id"] for link in case["topology"]["links"] if {link["src"],link["dst"]}&peers}
        facts[row["name"]]={"samples_ns":[t["elapsed_ns"] for t in timings],"median_ns":row["median_elapsed_ns"],
                           "selection_sha256":diag["selected_and_output_sha256"],"execution_signature":sig,
                           "managed_peak_bytes_by_node":diag["managed_peak_bytes_by_node"],
                           "cuda_peak_allocated_delta_bytes_by_node":diag["cuda_peak_allocated_bytes_delta_by_node"],
                           "link_bytes":diag["link_bytes"],
                           "remote_gpu_peak_bytes":sum(diag["managed_peak_bytes_by_node"][n] for n in peers),
                           "remote_gpu_link_bytes":sum(diag["link_bytes"].get(link,0) for link in peer_links),
                           "physical_sha256":canon({"actions":[{k:v for k,v in a.items() if k!="rule"} for a in diag["actions"]],
                               **{k:diag[k] for k in ("semantics","link_bytes","total_fetch_misses","managed_peak_bytes_by_node")}})}
    if result["feasible"]:
        need(len(observed)==len(cases) and result["score"]==sum(x["median_elapsed_ns"] for x in observed)/1e6,
             "observed per-case medians do not reproduce aggregate")
    else:
        need(result["score"] is None,"infeasible trial supplied performance score")
    return facts


def paired_orders(labels,key,block_ids):
    labels=sorted(labels)
    count=len(labels)
    offset=int(digest(key.encode())[:8],16)%count
    base=[0]+[(i+1)//2 if i%2 else count-i//2 for i in range(1,count)]
    orders=[[labels[(item+shift+offset)%count] for item in base] for shift in range(count)]
    if count%2:
        orders+= [list(reversed(order)) for order in orders.copy()]
    return {block:orders[index%len(orders)] for index,block in enumerate(block_ids)}


def audit_receipts(root,config,base,records,qualifications):
    executor=config["executor"]
    directory=root/config["output_directory"]/"receipts"
    raw_files,items,paired={},{},{}
    chronological=[]
    expected_hardware=None

    def job(split,program,effect,label,block,expected_id):
        nonlocal expected_hardware
        cases=config["cases"][split]
        arm="native_lru" if program is None else "generated"
        payload={"schema":1,"arm":arm,"program":program,"cases":cases,"warmup":base["warmup"],"samples":base["samples"],
                 "expected_gpu_uuids":executor["gpu_uuids"]}
        identity={"experiment":config["experiment_id"],"contract":base,"evaluation":effect,"label":label,"block":block,"payload":payload}
        job_id="eval-"+canon(identity)[:48]
        need(job_id==expected_id and JOB.fullmatch(job_id) and job_id not in items,"duplicate or changed worker request "+job_id)
        raw=read(directory/(job_id+".json"))
        receipt=json.loads(raw)
        result=receipt["result"]
        request={"operation":"run","job_id":job_id,"module":MODULE,"payload":payload,"timeout":config.get("timeout",900),
                 "source":executor["source"],"source_manifest_sha256":WORKER_SHA,"gpu_indices":executor["gpu_indices"],
                 "gpu_uuids":executor["gpu_uuids"],"environment":executor.get("environment") or {}}
        need(receipt["schema"]==2 and receipt["job_id"]==job_id and receipt["source_manifest_sha256"]==WORKER_SHA
             and receipt["request_sha256"]==canon(request) and receipt["result_canonical_sha256"]==canon(result)
             and SHA.fullmatch(receipt["result_sha256"]) and receipt["returncode"]==0 and receipt["status"]=="completed"
             and receipt["gpu_indices"]==executor["gpu_indices"] and receipt["gpu_uuids"]==executor["gpu_uuids"]
             and not any(receipt[k] for k in ("timeout","failure","environment_conflict","remaining_compute_pids",
                         "process_group_remaining","process_group_cleanup_required")) and receipt.get("replayed_receipt") is not True,
             "raw leased request, outcome, source or cleanup differs "+job_id)
        hardware=result["hardware"]
        uuids=lambda values:[v.lower().removeprefix("gpu-") for v in values]
        need(len(hardware)==1 and hardware[0]["ordinal"]==0
             and uuids([h["uuid"] for h in hardware])==uuids(executor["gpu_uuids"]),"observed GPU UUID binding differs")
        if expected_hardware is None:
            expected_hardware=hardware
        need(hardware==expected_hardware,"GPU hardware changed")
        facts=audit_result(result,cases,program,arm,base["samples"])
        raw_files["receipts/"+job_id+".json"]=raw
        row={"job_id":job_id,"split":split,"effect":effect,"label":label,"block_id":block,"score_ms":result["score"],
             "feasible":result["feasible"],"receipt_sha256":digest(raw),"request_sha256":receipt["request_sha256"],
             "result_sha256":receipt["result_sha256"],"cases":facts,
             "utc_start":receipt["utc_start"],"utc_end":receipt["utc_end"]}
        items[job_id]=row
        chronological.append((datetime.fromisoformat(receipt["utc_start"]),datetime.fromisoformat(receipt["utc_end"]),
                              receipt["pid"],receipt["proc_starttime"]))
        return row

    def direct(effect,split,program,record):
        ids=record["metrics"]["receipts"]
        if not ids:
            need(not record["feasible"] and record["metrics"].get("preflight_rejection") is True,
                 "feasible result missing a worker receipt")
            return
        need(len(ids)==1,"direct evaluation has more than one raw job")
        row=job(split,program,effect,"native" if program is None else "candidate","train",ids[0])
        need(row["feasible"]==record["feasible"] and row["score_ms"]==record["score"],"direct evaluation summary differs")

    for name,wrapper in qualifications.items():
        program=None if name.startswith("native-") else config[name+"_program"]
        direct("qualification-"+name,"train",program,wrapper["result"])
    for key,record in records.items():
        if key.startswith("eval/"):
            need(record["status"]=="completed","direct effect did not settle")
            direct(key.removeprefix("eval/"),record["input"]["split"],record["input"]["program"],record["result"])
        elif key.startswith("pair/"):
            need(record["status"]=="completed","paired effect did not settle")
            split,programs,blocks=(record["input"][k] for k in ("split","programs","block_ids"))
            effect=key.removeprefix("pair/")
            result=record["result"]
            orders=paired_orders(programs,effect,blocks)
            need(blocks==base["paired_block_ids"][split] and programs.keys()==result.keys(),"paired case plan differs")
            by_label={label:[] for label in programs}
            paired[effect]=[]
            for i,block in enumerate(blocks):
                paired_row={"block_id":block,"order":orders[block],"arms":{}}
                for label in orders[block]:
                    arm=result[label]
                    metrics=arm["metrics"]
                    need(len(metrics["receipts"])==len(metrics["execution_order"])==len(blocks)
                         and metrics["execution_order"][i]=={"block_id":block,"order":orders[block]},"recorded arm order differs")
                    row=job(split,programs[label],effect,label,block,metrics["receipts"][i])
                    paired_row["arms"][label]=row
                    by_label[label].append(row)
                observed=[tuple((name,c["selection_sha256"]) for name,c in row["cases"].items())
                          for row in paired_row["arms"].values() if row["feasible"]]
                need(len(set(observed))<=1,"paired arms differ on ordered selected bytes/output")
                paired[effect].append(paired_row)
            for label,rows in by_label.items():
                arm=result[label]
                ok=all(row["feasible"] for row in rows)
                samples=[{"block_id":row["block_id"],"score":row["score_ms"],"receipt_id":row["job_id"],
                          "case_scores_ms":{n:c["median_ns"]/1e6 for n,c in row["cases"].items()}} for row in rows if row["feasible"]]
                need(samples==arm["samples"] and arm["feasible"]==ok
                     and arm["score"]==(median(s["score"] for s in samples) if ok else None),"paired summary differs from receipts")
    need({p.stem for p in directory.glob("eval-*.json")}==set(items),"unused or missing local leases")
    need(all(a<b for a,b,_,_ in chronological) and
         all(a[1]<=b[0] for a,b in zip(chronological,chronological[1:]))
         and len({row[2:] for row in chronological})==len(chronological),"worker timing overlapped or reused a process")
    return raw_files,items,paired,expected_hardware

def audit_training(report,records,receipt_rows):
    summaries=[]
    used=set()
    admitted={report["manifest"]["seed_sha256"]}
    for row in report["rounds"]:
        number=row["round"]
        prefix=f"r{number:04d}"
        matrix=row["v4_training"]
        plan=matrix["plan"]
        scope=plan["scope_sha256"]
        declared={k:v for k,v in plan.items() if k!="scope_sha256"}
        need(canon({"contract":report["manifest"]["contract"],"plan":declared})==scope
             and plan==records[f"training_matrix/{prefix}/plan"]
             and matrix==records[f"training_matrix/{prefix}/result"]
             and len(plan["block_ids"])==3,"training matrix or declared scope differs")
        measures={}
        for effect,measure in matrix["measurements"].items():
            evaluations=measure["evaluations"]
            need(canon({"scope_sha256":scope,"effect_sha256":effect,"evaluations":evaluations})
                 ==measure["qualification_record_sha256"],"training qualification digest differs")
            windows=[]
            for item in evaluations:
                key=item["evaluation_key"]
                ev=records[key]
                result=ev["result"]
                need(key not in used and ev["input"]["split"]=="train" and canon(ev["input"]["program"])==effect
                     and item["evaluation_sha256"]==canon({"evaluation_key":key.removeprefix("eval/"),"record":ev})
                     and item["status"]==ev["status"]=="completed" and item["feasible"]==result["feasible"]
                     and item["reported_receipts"]==result["metrics"]["receipts"],"repeated training evaluation identity changed")
                used.add(key)
                if result["feasible"]:
                    [job]=item["reported_receipts"]
                    raw=receipt_rows[job]
                    need(raw["effect"]==key.removeprefix("eval/") and raw["split"]=="train"
                         and raw["score_ms"]==result["score"]==item["window_ms"]
                         and item["receipt_bundle_sha256"]==canon([job]),"repeated training sample differs")
                    windows.append({"block_id":item["block_id"],"receipt_id":job,"window_ms":raw["score_ms"],
                                    "case_median_ns":{n:c["median_ns"] for n,c in raw["cases"].items()}})
            if measure["status"]=="complete":
                need([x["block_id"] for x in windows]==plan["block_ids"]
                     and median(x["window_ms"] for x in windows)==measure["median_ms"],"training median differs")
            else:
                need(measure["status"]=="correctness_failed" and measure["median_ms"] is None,
                     "unsettled or unknown training measurement")
            measures[effect]={"status":measure["status"],"median_ms":measure["median_ms"],"windows":windows}
        lineage=row["active_before"]
        if row["decision"]=="admitted":
            need(row["active_after"]==row["proposed_child_sha256"],"admitted source did not become active")
            admitted.add(row["active_after"])
        else:
            need(row["decision"]=="rejected" and row["active_after"]==lineage,"rejected source changed active lineage")
        choices=row["v4_selection"]
        for label in (*ARMS,"parent","child"):
            choice=choices[label]
            if not choice["proposal_id"]:
                continue
            selected=matrix["cohort_proposals"][choice["proposal_id"]]
            need(choice["effect_sha256"]==selected["training_effect_sha256"]==selected["program_sha256"]==canon(selected["program"])
                 and choice["source_sha256"]==selected["harness_sha256"] and measures[choice["effect_sha256"]]["status"]=="complete",
                 "selected proposal, measurement or generating source differs")
            if label=="admitted_output":
                need(choice["source_sha256"] in admitted,"admitted finalist has unadmitted source")
            if label=="fixed":
                need(choice["source_sha256"]==report["manifest"]["seed_sha256"],"fixed arm changed source")
        reuse=row.get("previous_admission_reuse")
        if reuse:
            task_ids=reuse["task_api_ids"]
            need(len(task_ids)==reuse["task_api_completed"]
                 and all(records[k]["status"]=="completed" and records[k]["input"]["harness_sha256"]==reuse["source_sha256"] for k in task_ids)
                 and (not reuse["meta_api_completed"] or records[reuse["meta_api_id"]]["input"]["harness_sha256"]==reuse["source_sha256"]),
                 "recorded source reuse differs from actual calls")
        summaries.append({"round":number,"decision":row["decision"],"reason":row["reason"],"active_before":lineage,
                          "active_after":row["active_after"],"proposed_source":row["proposed_child_sha256"],
                          "controlled_behavior":row["controlled_behavior"],"reuse":reuse,
                          "training_scope_sha256":scope,"training_measurements":measures,
                          "selections":choices,"validation":row["paired_validation"]})
    need(used=={k for k in records if k.startswith("eval/") and "/training-matrix/" in k},"unused training matrix effect")
    return summaries


def delta(values):
    return {"wins":sum(x>0 for x in values),"ties":sum(x==0 for x in values),"losses":sum(x<0 for x in values),
            "median_ms":median(values),"block_deltas_ms":values}


def test_summary(report,records,paired,config):
    test=records["pair/frozen/test"]
    frozen=report["frozen"]
    refs=("native","fixed","manual")
    need(test["status"]=="completed" and set(test["input"]["programs"])=={*ARMS,"native","manual"}
         and test["input"]["programs"]["native"] is None and test["input"]["programs"]["manual"]==config["manual_program"]
         and all(test["input"]["programs"][label]==frozen[label]["trial"]["program"] for label in ARMS),
         "test used different program from the freeze")
    block_rows=paired["frozen/test"]
    by_kind={}
    for kind,case in zip(SCENARIOS,config["cases"]["test"],strict=True):
        name=case["name"]
        scopes={}
        for label in test["result"]:
            available=[row["arms"][label]["cases"].get(name) if row["arms"][label]["feasible"] else None for row in block_rows]
            scopes[label]={"fully_feasible":all(x is not None for x in available),
                           "per_block":[None if item is None else {"block_id":row["block_id"],**item}
                                        for row,item in zip(block_rows,available,strict=True)]}
            if scopes[label]["fully_feasible"]:
                scopes[label]["median_ms"]=median(item["median_ns"] for item in available)/1e6
                scopes[label]["managed_peak_bytes_by_node_max"]={node:max(item["managed_peak_bytes_by_node"][node] for item in available)
                    for node in available[0]["managed_peak_bytes_by_node"]}
                scopes[label]["cuda_peak_allocated_delta_bytes_by_node_max"]={node:max(item["cuda_peak_allocated_delta_bytes_by_node"][node]
                    for item in available) for node in available[0]["cuda_peak_allocated_delta_bytes_by_node"]}
                scopes[label]["link_bytes_max_by_link"]={link:max(item["link_bytes"].get(link,0) for item in available)
                    for link in sorted({link for item in available for link in item["link_bytes"]})}
        comparisons={}
        for candidate in ARMS:
            for ref in ("native","manual","fixed","admitted_output"):
                if ref==candidate:
                    continue
                left,right=scopes[candidate],scopes[ref]
                if left["fully_feasible"] and right["fully_feasible"]:
                    vals=[(r["median_ns"]-l["median_ns"])/1e6 for l,r in zip(left["per_block"],right["per_block"],strict=True)]
                    comparisons[f"{candidate}_vs_{ref}"]={**delta(vals),"same_observed_physical_trace":all(
                         l["physical_sha256"]==r["physical_sha256"] for l,r in zip(left["per_block"],right["per_block"],strict=True))}
                else:
                    comparisons[f"{candidate}_vs_{ref}"]={"status":"missing_feasible_pair"}
        by_kind[kind]={"case":name,"compute_node":case["compute_node"],"arms":scopes,"comparisons":comparisons}
    margin=report["manifest"]["config"]["min_improvement"]
    aggregate={}
    for candidate in ("manager_best","admitted_output"):
        for ref in refs:
            key=f"{candidate}_vs_{ref}"
            claim=report["test_comparisons"][key]
            same=canon(test["input"]["programs"][candidate])==canon(test["input"]["programs"][ref]) if ref!="native" else False
            need(report["test_program_relations"][key]=={"same_raw_program":same,"same_training_effect":same},
                 "same raw identity relation differs")
            after,before=test["result"][candidate],test["result"][ref]
            if not (after["feasible"] and before["feasible"]):
                need(not claim["passed"] and claim["deltas"]==[],"infeasible comparison claims a gain")
                aggregate[key]={"status":"missing_feasible_pair","same_raw_program":same}
                continue
            vals=[r["score"]-l["score"] for l,r in zip(after["samples"],before["samples"],strict=True)]
            measured=delta(vals)
            recorded=[{"block_id":row["block_id"],"reference_minus_candidate":x} for row,x in zip(block_rows,vals,strict=True)]
            passed=min(vals)>0 and measured["median_ms"]>margin
            need(claim["deltas"]==recorded and claim["median_improvement"]==measured["median_ms"] and claim["passed"] is passed,
                 "formal test comparison differs from raw samples")
            aggregate[key]={**measured,"passed":passed,"same_raw_program":same}
    arm_scores={label:{"fully_feasible":arm["feasible"],"median_ms":arm["score"],
                       "block_samples_ms":{row["block_id"]:row["score"] for row in arm["samples"]}}
                for label,arm in test["result"].items()}
    return {"block_ids":[row["block_id"] for row in block_rows],"margin_ms":margin,"aggregate":aggregate,
            "arm_scores":arm_scores,"scenarios":by_kind}


def receipt_zip(path,members):
    with ZipFile(path,"x",compression=ZIP_DEFLATED,compresslevel=9) as package:
        for name,raw in sorted(members.items()):
            entry=ZipInfo(name,date_time=(2020,1,1,0,0,0))
            entry.external_attr=0o644 << 16
            package.writestr(entry,raw,compress_type=ZIP_DEFLATED,compresslevel=9)
    with ZipFile(path) as package:
        need(set(package.namelist())==set(members) and all(package.read(name)==raw for name,raw in members.items()),
             "ZIP member bytes differ")


def verify_publication(root):
    manifest=json.loads(read(root/"artifact_manifest.json"))
    for name,sha in manifest["files_sha256"].items():
        seal(root/safe_path(name),sha)
    archives={"leased-receipts.zip":manifest["leased_receipt_member_sha256"],
              "original-api-and-plans.zip":manifest["original_api_and_plan_member_sha256"]}
    for archive,members in archives.items():
        with ZipFile(root/archive) as package:
            need(len(package.namelist())==len(set(package.namelist())) and set(package.namelist())==set(members),
                 "publication ZIP roster differs")
            for name,sha in members.items():
                safe_path(name)
                need(digest(package.read(name))==sha,"publication ZIP member bytes changed")
    return {"files":len(manifest["files_sha256"]),"receipt_members":len(archives["leased-receipts.zip"])}


def write_readme(audit):
    roles=audit["finalists"]
    lines=["# TopoMem CUDA v4.1 工程复评","",
           "四个已公开文本场景均在同一张物理 GPU3 上执行 H64 注意力，逻辑设备名为 gpu0。KV 预算为 128 KiB GPU 和 32 MiB pinned host。该工程复评与此前中断的双 GPU 试验分别归档。","",
           "两轮各使用两个任务槽。控制器先按训练测量封存程序，再在六个配对块上评估。三个封存角色分别记录全局训练管理器、已晋级来源的输出及固定生成器输出。","",
           "| 封存角色 | 试验 | 生成器 SHA-256 | 重复训练中位数 ms |","|---|---|---|---:|"]
    for label,entry in roles.items():
        lines.append(f"| {label} | {entry['trial_id']} | {entry['harness_sha256']} | {entry['training_score_ms']:.6f} |")
    lines.extend(["","| 正式对照 | 缩短窗口的区块 | 配对中位差 ms | 通过门槛 |","|---|---:|---:|:---:|"])
    for key,entry in audit["test"]["aggregate"].items():
        if "wins" in entry:
            lines.append(f"| {key} | {entry['wins']}/6 | {entry['median_ms']:+.6f} | {'是' if entry['passed'] else '否'} |")
        else:
            lines.append(f"| {key} | — | — | — |")
    counts=Counter(row["decision"] for row in audit["rounds"])
    lines.extend(["",f"两轮中晋级 {counts['admitted']} 次。完整审计列出每轮计划、决定、实际源复用及所有训练重复和每场景内存指标。"
                  f"核对了 {audit['receipt_count']} 份租约回执和 {audit['provider_calls']} 个实际模型请求。","",
                  "v4.1 提示修订与 v13 零跳保留优化由操作方提供。原生、人工管理器、初始程序及生成程序使用同一 v13 执行器。此前 24 份工程资格回执属于独立对照。"
                  "实际发给模型的 v4.1 提示说明父、子各自从新槽选出候选，历史 incumbent 独立参与配对验证。","",
                  "[原终局报告](controller/report.json) · [独立审计](audit.json) · [原始回执 ZIP](leased-receipts.zip) · [公开配置模板](inputs/adapter-config.template.json) · [工程资格绑定](qualification/binding.json) · [工件清单](artifact_manifest.json)",
                  "","离线校验命令。","",chr(96)*3+"sh",
                  "python -B tools/export_topomem_v41_gpu3.py verify --publication .",chr(96)*3])
    return ("\n".join(lines)+"\n").encode()


def audit_provider_envelopes(report,config):
    protocol=report["manifest"]["contract"]["selection_protocol"]
    need(protocol["kind"]=="training_paired_declared_identity_v41" and protocol["version"]=="4.1",
         "not the corrected v4.1 selection protocol")
    training_names={case["name"] for case in config["cases"]["train"]}
    excluded={case["name"] for split in ("validation","test") for case in config["cases"][split]}-training_names
    for key,call in report["calls"].items():
        payload=call["input"]["payload"]
        messages=payload["messages"]
        envelope=json.loads(messages[-1]["content"])
        need(envelope["trusted_selection_contract"]==protocol["trusted_selection_contract"]
             and not any(name in messages[-1]["content"] for name in excluded),"model selection facts changed or nontraining case supplied")
        if key.endswith("-meta"):
            need("The parent and child each select a fresh program" in messages[0]["content"]
                 and "Both sides may retain the best measured training program" not in messages[0]["content"],
                 "legacy meta instruction appears in v4.1 call")


def verify_engineering_qualification(root,publication,config,case_plan,original):
    need(case_plan["qualified_engineering_plan_sha256"]==ENGINEERING_PLAN
         and case_plan["qualified_engineering_result_sha256"]==ENGINEERING_RESULT,"engineering qualification binding differs")
    result=seal(root/"engineering-qualification.json",ENGINEERING_RESULT)
    need(result==seal(publication/"run/analysis.json",ENGINEERING_RESULT),"engineering analysis bytes changed")
    plan=seal(publication/"inputs/preregistration.json",ENGINEERING_PLAN)
    analysis,prespec=json.loads(result),json.loads(plan)
    need(analysis["status"]=="completed" and analysis["jobs"]==24 and analysis["plan_sha256"]==ENGINEERING_PLAN
         and analysis["same_ir_physical_trace_verified"] is True and analysis["expected_producer_bypass_verified"] is True
         and prespec["source_versions"]["v13"]["archive_sha256"]==WORKER_ZIP
         and prespec["source_versions"]["v13"]["manifest_sha256"]==WORKER_SHA
         and prespec["arms"]["lru-equivalent-v13"]["program"]==config["initial_program"]
         and config["cases"]["train"]==prespec["payload"]["cases"]
         and config["executor"]["gpu_uuids"]==prespec["payload"]["expected_gpu_uuids"]
         and prespec["gpu_indices"]==config["executor"]["gpu_indices"]==[3]
         and prespec["host_cpu_affinity"]==[64],"engineering qualification hardware or training geometry differs")
    raw_manifest=read(publication/"artifact_manifest.json")
    manifest=json.loads(raw_manifest)
    commitments=manifest["leased_receipt_member_sha256"]
    need(len(commitments)==24,"engineering qualification has wrong receipt roster")
    ledger=read(publication/"run/jobs.jsonl")
    need(digest(ledger)==manifest["files_sha256"]["run/jobs.jsonl"],"engineering ledger differs")
    ids=[]
    with ZipFile(publication/"leased-receipts.zip") as archive:
        names=archive.namelist()
        need(len(names)==len(set(names))==24 and set(names)==set(commitments),"engineering receipt ZIP roster differs")
        for line in ledger.splitlines():
            job=json.loads(line)
            name=f"{job['source']}/{job['job_id']}.json"
            safe_path(name)
            raw=archive.read(name)
            need(digest(raw)==job["receipt_sha256"]==commitments[name],"engineering receipt differs")
            receipt=json.loads(raw)
            need(receipt["job_id"]==job["job_id"] and receipt["status"]=="completed"
                 and receipt["source_manifest_sha256"]==prespec["source_versions"][job["source"]]["manifest_sha256"]
                 and receipt["result_canonical_sha256"]==canon(receipt["result"]),"engineering receipt binding differs")
            ids.append(job["job_id"])
    need(len(ids)==len(set(ids))==24,"engineering ledger job roster differs")
    original["qualification/engineering-analysis.json"]=result
    original["qualification/engineering-preregistration.json"]=plan
    original["qualification/engineering-artifact-manifest.json"]=raw_manifest
    original["qualification/engineering-jobs.jsonl"]=ledger
    original["qualification/binding.json"]=nice({"schema":1,"historical_job_ids":ids,
         "historical_artifact_manifest_sha256":digest(raw_manifest),"source_manifest_sha256":WORKER_SHA,
         "scope":"Independent same-IR lowering experiment; none of these jobs are current controller effects."})
    return ids


def export(args):
    root=args.run_root
    need(not args.output.exists() and not args.output.resolve().is_relative_to(root.resolve()),"new independent publication directory required")
    original,report=terminal(root,args.expected_report_sha256,args.expected_pipeline_sha256)
    records,original_records=journal(root,original)
    private=read(root/"private-config.json")
    config=json.loads(private)
    case_plan_raw=seal(root/"case-plan.json",FORMAL_PLAN)
    case_plan=json.loads(case_plan_raw)
    pipeline=read(root/"pipeline-plan.json")
    pipeline_plan=json.loads(pipeline)
    need(digest(private)==PRIVATE_CONFIG==case_plan["private_config_sha256"]==pipeline_plan["private_config_sha256"]
         and pipeline_plan["case_plan_sha256"]==FORMAL_PLAN and pipeline_plan["source_manifest_sha256"]==CONTROLLER_SHA
         and pipeline_plan["script_sha256"]==PIPELINE_SCRIPT and pipeline_plan["stages"]==["run","freeze","test"],
         "private adapter or source binding changed")
    contract=check_geometry(config,case_plan,report)
    need(records["manifest"]==report["manifest"] and records["frozen"]==report["frozen"],"report differs from the journal")
    for prefix,field in (("api/","calls"),("eval/","evaluations"),("pair/","paired_evaluations"),
                         ("plan/","meta_plans"),("training_matrix/","training_matrices"),("selection_context/","selection_contexts")):
        need({k:v for k,v in records.items() if k.startswith(prefix)}==report[field],"report records differ "+field)
    call_count=len(report["calls"])
    need(call_count<=PROVIDER_BUDGET and all(sum(key.split("/")[2].startswith("fixed")==is_fixed
         for key in report["calls"] if key.startswith(f"api/r{number:04d}/"))<=5
         for number in range(ROUND_COUNT) for is_fixed in (False,True)),"recorded provider request count exceeds frozen budget")
    controller=seal(root/"controller-source.zip",CONTROLLER_ZIP)
    worker=seal(args.worker_source_archive,WORKER_ZIP)
    original["pipeline-plan.json"]=pipeline
    original["source/pipeline-launcher.py"]=seal(args.pipeline_source,PIPELINE_SCRIPT)
    original["inputs/case-plan.json"]=case_plan_raw
    original["source/controller-source.zip"]=controller
    original["source/worker-source.zip"]=worker
    original["source/binding.json"]=nice({"controller_source_archive_sha256":CONTROLLER_ZIP,
        "controller_source_manifest_sha256":CONTROLLER_SHA,"worker_source_archive_sha256":WORKER_ZIP,
        "worker_source_manifest_sha256":WORKER_SHA})
    original["inputs/adapter-config.template.json"]=nice(sanitize_config(config))
    original["inputs/adapter-contract.json"]=nice(contract)
    original["inputs/cases.json"]=nice(config["cases"])
    for name in ("lease.sqlite","lease.sqlite-journal"):
        path=root/"controller"/name
        if path.is_file():
            original[f"controller/{name}"]=read(path)
    for sub,suffix in (("rounds","*.json"),("harnesses","*.py")):
        path=root/"controller"/sub
        for member in path.glob(suffix):
            raw=read(member)
            need(sub!="harnesses" or digest(raw)==member.stem,"generator source filename digest changed")
            original[f"controller/{sub}/{member.name}"]=raw
    need({k for k in original if k.startswith("controller/rounds/")}==
         {f"controller/rounds/r{i:04d}.json" for i in range(ROUND_COUNT)},"archived round files are incomplete")
    for member in root.glob("pipeline-*.log"):
        original[member.name]=read(member)
    historical_ids=verify_engineering_qualification(root,args.qualification_publication,config,case_plan,original)
    audit_provider_envelopes(report,config)
    with tempfile.TemporaryDirectory(prefix="topomem-v4-audit-") as directory:
        temp=Path(directory)
        m,files=verify_source(controller,CONTROLLER_SHA,temp/"source")
        w,worker_files=verify_source(worker,WORKER_SHA)
        need(m==read(root/"controller-source-manifest.json"),"separate controller manifest differs")
        original["source/controller-source-manifest.json"]=m
        original["source/worker-source-manifest.json"]=w
        source_bindings=verify_sources(report,files)
        base,effects=replay_controller(temp/"source",config,records,original,report,temp)
    receipts,receipt_rows,paired,hardware=audit_receipts(root,config,base,records,{})
    need(not (set(historical_ids)&set(receipt_rows)),"v4.1 formal leases overlap the separate engineering study")
    training=audit_training(report,records,receipt_rows)
    final=test_summary(report,records,paired,config)
    finalists={}
    for label in ARMS:
        selected=report["frozen"][label]
        trial=selected["trial"]
        sha=trial["harness_sha256"]
        source_path=f"controller/harnesses/{sha}.py"
        provider_id="api/"+trial["id"]
        need(sha and records[provider_id]["input"]["harness_sha256"]==sha and records[provider_id]["status"]=="completed"
             and trial["program_sha256"]==canon(trial["program"]) and source_path in original,
             "selected manager missing actual provider/source provenance")
        original[f"selected/{label}.program.json"]=nice(trial["program"])
        original[f"selected/{label}.generator.py"]=original[source_path]
        finalists[label]={"trial_id":trial["id"],"program_sha256":trial["program_sha256"],"harness_sha256":sha,
                          "provider_call_id":provider_id,"training_score_ms":selected["training_score"],"selection":selected["selection"]}
    original["selected/selection.json"]=nice(finalists)
    audit={"schema":1,"scope":"public-corpus CUDA engineering reassessment; single compute device per case",
           "terminal_report_sha256":args.expected_report_sha256,"pipeline_status_sha256":args.expected_pipeline_sha256,
           "private_config_sha256":digest(private),"case_plan_sha256":FORMAL_PLAN,
           "controller_source_sha256":CONTROLLER_SHA,"worker_source_sha256":WORKER_SHA,
           "controller_source_files":len(files),"worker_source_files":len(worker_files),"report_source_bindings":source_bindings,
           "exact_offline_controller_replay":True,"archived_effects_replayed":effects,"provider_calls":call_count,
           "provider_usage":report["usage"],"receipt_count":len(receipts),"receipt_counts_by_split":dict(Counter(r["split"] for r in receipt_rows.values())),
           "separate_engineering_job_count":len(historical_ids),"separate_engineering_job_ids":historical_ids,
           "engineering_qualification_sha256":ENGINEERING_RESULT,
           "finalists":finalists,"rounds":training,"test":final,"hardware":hardware,
           "reported_reused_sources":report["frozen"]["reused_harness_sha256s"]}
    for filename,raw in original.items():
        if filename.endswith((".log",".json",".py")):
            need(not re.search(rb"sk-[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----",raw),"credential-like value in "+filename)
    api={f"{k}.json":v for k,v in original_records.items() if k.startswith(("api/","plan/"))}
    for filename,raw in api.items():
        need(not re.search(rb"sk-[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----",raw),"credential-like value in "+filename)
    args.output.mkdir(exist_ok=False,parents=True)
    for name,raw in original.items():
        write(args.output/safe_path(name),raw)
    receipt_zip(args.output/"leased-receipts.zip",receipts)
    receipt_zip(args.output/"original-api-and-plans.zip",api)
    write(args.output/"audit.json",nice(audit))
    write(args.output/"server_job_index.json",nice({"schema":1,"jobs":receipt_rows}))
    write(args.output/"README_ZH.md",write_readme(audit))
    write(args.output/"tools/export_topomem_v41_gpu3.py",read(Path(__file__)))
    test_source=Path(__file__).with_name("test_export_topomem_v41_gpu3.py")
    if test_source.is_file():
        write(args.output/"tools"/test_source.name,read(test_source))
    files_sha={p.relative_to(args.output).as_posix():digest(read(p)) for p in sorted(args.output.rglob("*")) if p.is_file()}
    artifacts={"schema":1,"files_sha256":files_sha,"leased_receipt_member_sha256":{k:digest(v) for k,v in receipts.items()},
               "original_api_and_plan_member_sha256":{k:digest(v) for k,v in api.items()},
               "private_config_sha256":digest(private)}
    write(args.output/"artifact_manifest.json",nice(artifacts))
    verify_publication(args.output)
    return {key:audit[key] for key in ("scope","receipt_count","receipt_counts_by_split","provider_calls","finalists")}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest="phase",required=True)
    final=sub.add_parser("export")
    for key in ("run-root","worker-source-archive","qualification-publication","pipeline-source","output"):
        final.add_argument("--"+key,type=Path,required=True)
    final.add_argument("--expected-report-sha256",required=True)
    final.add_argument("--expected-pipeline-sha256",required=True)
    verify=sub.add_parser("verify")
    verify.add_argument("--publication",type=Path,required=True)
    args=parser.parse_args()
    result=export(args) if args.phase=="export" else verify_publication(args.publication)
    print(json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":
    main()
