"""Publish settled native-6144 supplement evidence without accessing its private adapter config."""
import argparse
from collections import Counter
from datetime import datetime, timezone
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import sys
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

FORMAT = "sglang_post_freeze_native6144_v1"
JOB = re.compile(r"sg-[0-9a-f]{48}")
HASH = re.compile(r"[0-9a-f]{64}")
FIXED = {
    "preregistration":"47fb544e2c326d388d221101d652fb59073af1899ff69c949ffdf8bc07e7442d",
    "waiter":"d3b21173bb0967be744a9d49f4b1d8f6c1a8449a83456601f5a7311c1ef1ff15",
    "driver":"9a4eef769bc864ff45f123c4cbf7cfcb0a81881ca222abbf26ff412802247994",
}
CONTROL = ("wait-plan.json","source-preflight.log","launch-binding.json","driver.log","completion.json")
RAW_STUDY = ("preregistration.json","manifest.json","result.json","jobs.jsonl","frozen_adaptive.program.json")
STATES = {"completed","budget_incomplete","execution_incomplete","skipped_budget","skipped_formal_pipeline_failed"}
LABELS = ("frozen_adaptive","native6144_all_host")


def need(test, message):
    if not test:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def json_bytes(obj, *, ascii=True):
    return (json.dumps(obj,sort_keys=True,indent=2,ensure_ascii=ascii,allow_nan=False)+"\n").encode()


def canonical(obj):
    return digest(json.dumps(obj,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode())


def time_utc(value):
    stamp = datetime.fromisoformat(value.replace("Z","+00:00"))
    need(stamp.utcoffset()==timezone.utc.utcoffset(None),"expected a UTC timestamp")
    return stamp


def read(path):
    need(path.is_file() and not path.is_symlink(),"missing or symlinked " + path.name)
    return path.read_bytes()


def sealed(path, expected):
    need(isinstance(expected,str) and HASH.fullmatch(expected),"missing independently supplied seal for " + path.name)
    raw = read(path)
    need(digest(raw)==expected,"sealed artifact SHA-256 differs for " + path.name)
    return raw


def check_source(path, spec):
    raw = sealed(path,spec["controller_source_archive_sha256"])
    with ZipFile(BytesIO(raw)) as archive:
        manifest_raw = archive.read("LOCAL_SOURCE_MANIFEST.json")
        need(digest(manifest_raw)==spec["controller_source_manifest_sha256"],"frozen source manifest SHA differs")
        manifest = json.loads(manifest_raw)
        need(manifest["worker_source_manifest_sha256"]==spec["worker_source_manifest_sha256"],"frozen worker source differs")
        files = manifest["files"]
        names = archive.namelist()
        need(len(names)==len(set(names)) and set(names)=={"LOCAL_SOURCE_MANIFEST.json",*files},"source archive roster differs")
        for name,expected in files.items():
            path = PurePosixPath(name)
            need(not path.is_absolute() and ".." not in path.parts and digest(archive.read(name))==expected,
                 "frozen source file changed " + name)
    return manifest["files"]


def check_controls(original, spec, fixed):
    plan = json.loads(original["waiter/wait-plan.json"])
    binding = json.loads(original["waiter/launch-binding.json"])
    completion = json.loads(original["waiter/completion.json"])
    preflight = json.loads(original["waiter/source-preflight.log"])
    need(plan["schema"]==1 and plan["driver_sha256"]==spec["driver_sha256"]==fixed["driver"]
         and plan["preregistration_sha256"]==fixed["preregistration"]
         and plan["read_formal_report_only_after_all_phases_exit_zero"] is True
         and all(plan[name]==value for name,value in spec["cutoffs"].items())
         and preflight=={"status":"source_and_preregistration_verified","preregistration_sha256":fixed["preregistration"]},
         "wait plan or successful source preflight differs from sealed study")
    need(time_utc(plan["created_utc"])<=time_utc(binding["utc"])<=time_utc(completion["finished_utc"]),
         "waiter timestamps are out of order")
    phases = binding["formal_phases"]
    need(isinstance(phases,list) and [row[0] for row in phases]==["run","freeze","test"][:len(phases)]
         and all(len(row)==2 and type(row[1]) is int for row in phases),"invalid recorded formal phase state")
    if binding.get("formal_report_sha256"):
        need(phases==[["run",0],["freeze",0],["test",0]]
             and time_utc(binding["utc"])<=time_utc(spec["cutoffs"]["latest_start_utc"]),
             "report opened outside authorized terminal phase and time gate")
    return binding,completion


def formal_anchor(root, expected, binding, spec, inventory, viewer):
    need(root is not None and binding.get("formal_report_sha256")==expected,"expected formal report SHA differs from waiter's launch binding")
    directory = root / "controller"
    report_raw = sealed(directory/"report.json",expected)
    frozen_raw, pipeline_raw = (read(p) for p in (directory/"frozen.json",root/"pipeline-status.json"))
    source_raw = sealed(root/"frozen-source.zip",spec["controller_source_archive_sha256"])
    artifact = json.loads(read(root/"artifact_manifest.json"))
    records = {"controller/report.json":report_raw,"controller/frozen.json":frozen_raw,
               "pipeline-status.json":pipeline_raw,"frozen-source.zip":source_raw}
    need(all(artifact["files_sha256"].get(name)==digest(data) for name,data in records.items()),"formal publication file inventory differs")
    report = json.loads(report_raw)
    manifest, frozen, pipeline = report["manifest"],json.loads(frozen_raw),json.loads(pipeline_raw)
    need([[row["phase"],row["returncode"]] for row in pipeline]==binding["formal_phases"]
         and time_utc(pipeline[-1]["finished_utc"])<=time_utc(binding["utc"]),"formal phase completion or timestamps differ")
    contract = manifest["contract"]
    protocol = contract["generator_program_protocol"]
    bindings = ({f"rsi_memory/{name}":value for name,value in manifest["source_sha256"].items()},
                contract["adapter_sources_sha256"],protocol["source_sha256"])
    need(all(inventory.get(name)==value for group in bindings for name,value in group.items()),"formal controller source binding differs")
    frozen_view = {"hashes_match": canonical(manifest)==frozen["manifest_sha256"]
                   and canonical(report["rounds"])==frozen["rounds_sha256"]}
    anchor = viewer._formal_anchor(directory,manifest,frozen,frozen_view)
    need(anchor is not None and anchor["formal_report_sha256"]==expected,"viewer could not establish a settled formal anchor")
    compare = ("cases","worker_source_manifest_sha256","metadata_profile_sha256","launch_sha256","tokenization_sha256",
               "margin_hand_sweep_canonical_sha256","controller_source_archive_sha256","controller_source_manifest_sha256")
    need(all(anchor[key]==spec[key] for key in compare)
         and anchor["margin_ms"]==spec["decision"]["median_strictly_greater_than_ms"],"formal test split or practical margin differs")
    return anchor,artifact["private_schema3_bytes_sha256"]


def study_receipts(study, spec, original):
    directory = study/"receipts"
    jobs = {}
    if not directory.exists():
        return jobs
    need(directory.is_dir() and not directory.is_symlink(),"invalid supplement receipts directory")
    for path in sorted(directory.glob("sg-*.json")):
        name = path.stem
        raw = read(path)
        value = json.loads(raw)
        need(JOB.fullmatch(name) and value["job_id"]==name and value["schema"]==2
             and value["source_manifest_sha256"]==spec["worker_source_manifest_sha256"]
             and HASH.fullmatch(value["request_sha256"]) and value["status"] in {
                 "completed","failed","timed_out","environment_conflict","started_unknown"},"invalid leased receipt " + name)
        result = value.get("result")
        if result is not None:
            need(canonical(result)==value["result_canonical_sha256"] and digest(json_bytes(result))==value["result_sha256"]
                 and result["case"] in spec["cases"] and result["memory_compilation"]["metadata_profile_sha256"]==spec["metadata_profile_sha256"]
                 and HASH.fullmatch(result["raw_result_sha256"]),"leased evaluator summary changed " + name)
        if value["status"]=="completed":
            need(value["returncode"]==0 and not value["remaining_compute_pids"]
                 and value["process_group_remaining"] is False and result is not None,"unclean successful lease " + name)
        jobs[name] = {"receipt_sha256":digest(raw),"status":value["status"],"request_sha256":value["request_sha256"],
                      **({k:result[k] for k in ("raw_result_sha256","phase","arm","program_sha256","score")} if result else {}),
                      "case_id":result["case"]["id"] if result is not None else None}
        original[f"receipts/{path.name}"]=raw
    connection = directory/"connection.log"
    if connection.exists():
        original["receipts/connection.log"]=read(connection)
    return jobs


def job_rows(original, jobs, spec, result, bound):
    rows = [json.loads(line) for line in original.get("jobs.jsonl",b"").splitlines()]
    cases = [row["id"] for row in spec["cases"]]
    expected = [(block["block_id"],case,label) for block in spec["blocks"] for case in cases for label in block["order"]]
    seen, timed, qualifiers = set(), [], []
    for row in rows:
        if row["phase"]=="qualification":
            need(not timed and len(qualifiers)<len(cases) and row["case"]==cases[len(qualifiers)]
                 and set(row["proofs"])==set(LABELS),"qualification ledger differs")
            for label,proof in row["proofs"].items():
                job = proof["job_id"]
                got = jobs.get(job,{})
                need(job not in seen and got.get("status")=="completed" and got.get("case_id")==row["case"]
                     and got.get("phase")=="parity" and got.get("arm")==("native" if label==LABELS[1] else "generated")
                     and got.get("program_sha256")== (None if label==LABELS[1] else bound["adaptive_program_sha256"])
                     and got.get("raw_result_sha256")==proof["raw_result_sha256"],"missing qualification proof")
                seen.add(job)
            repeat = row["native_same_capacity_repeat_job_id"]
            repeat_row = jobs.get(repeat,{})
            need(repeat not in seen and repeat_row.get("status")=="completed" and repeat_row.get("phase")=="parity"
                 and repeat_row.get("arm")=="native" and repeat_row.get("case_id")==row["case"],
                 "missing or reused native repeat qualification")
            seen.add(repeat)
            qualifiers.append(row)
        else:
            need(row["phase"]=="performance" and len(qualifiers)==len(cases) and len(timed)<len(expected),
                 "timing appeared before qualification or outside the plan")
            job, identity = row["job_id"], row["identity"]
            block, case, label = expected[len(timed)]
            got = jobs.get(job,{})
            need(job not in seen and got.get("status")=="completed" and got.get("case_id")==case
                 and got.get("phase")=="performance" and got.get("arm")==("native" if label==LABELS[1] else "generated")
                 and got.get("program_sha256")==(None if label==LABELS[1] else bound["adaptive_program_sha256"])
                 and got.get("score")==row["window_ms"] and got.get("raw_result_sha256")==row["raw_result_sha256"]
                 and identity=={"kind":FORMAT,"preregistration_sha256":result["preregistration_sha256"],
                     "driver_sha256":spec["driver_sha256"],"formal_frozen_sha256":bound["formal_frozen_sha256"],
                     "block_id":block,"case_id":case,"label":label}
                 and row["qualification"]==qualifiers[cases.index(case)]["proofs"][label],"timed ledger binding or order differs")
            timed.append(job)
            seen.add(job)
    completed = result.get("timed_jobs_completed",0)
    need(type(completed) is int and 0<=completed<=len(timed)<=32,"recorded timing count differs from lease ledger")
    need(len(timed)-completed <= 1,"too many measured windows after the last qualified observation")
    if result["status"] in {"completed","budget_incomplete"}:
        need(completed==len(timed),"terminal complete or budget stop count differs from the raw ledger")
    if result["status"]=="completed":
        need(completed==32 and len(qualifiers)==2,"complete experiment has missing jobs")
    if result["status"].startswith("skipped"):
        need(not rows and not jobs and completed==0,"skipped experiment contains GPU observations")
    return {"directly_referenced_job_ids":sorted(seen),"supporting_or_unreferenced_job_ids":sorted(set(jobs)-seen),
            "timed_job_ids_in_order":timed,"observed_timed_jobs":len(timed)}


def receipt_zip(path, entries):
    with ZipFile(path,"x",compression=ZIP_DEFLATED,compresslevel=9) as archive:
        for name,raw in sorted(entries.items()):
            member = ZipInfo(name,date_time=(2020,1,1,0,0,0))
            member.external_attr = 0o644 << 16
            archive.writestr(member,raw,compress_type=ZIP_DEFLATED,compresslevel=9)
    with ZipFile(path) as archive:
        need(set(archive.namelist())==set(entries) and all(archive.read(k)==v for k,v in entries.items()),
             "leased receipt archive round trip failed")


def report_md(result, audit):
    status = audit["status"]
    lines = ["# SGLang 冻结后补充对照","",
        "使用正式测试已经使用的两个 case，对照冻结的 adaptive 程序与原生 6144 全 host。这个补充测量保持已选程序，证据范围是重复使用的测试 case。",
        "",f"运行状态 **{status}**。已核对 {audit['receipt_count']} 份本地租约回执。",
        "",f"等待器终态 SHA-256 为 {audit['waiter_completion_sha256']}。"]
    view = audit["viewer_verification"]
    if status=="completed":
        scope = "两个 case 的实际热容量均匹配" if view["same_hot_capacity"] else "两臂至少一个 case 的实际热容量不同"
        lines.extend(["",f"8 个完整区块中有 {view['wins']} 个区块窗口缩短，中位差为 {view['median_gain_ms']:.6f} ms。"
                      f"预登记门槛判定为 **{'通过' if view['passed'] else '未通过'}**。{scope}。实际 backing 与热容量见原结果。"])
    else:
        lines.extend(["",f"已验证 {len(result.get('complete_blocks',[]))} 个完整区块，性能门槛未评估。"])
    lines.extend(["","[预注册](preregistration.json) · [等待器终态](waiter/completion.json) · [原租约回执 ZIP](leased-receipts.zip) · [工件清单](artifact_manifest.json)"])
    if audit.get("supplement_result_sha256"):
        lines.extend(["","[补充运行原始结果](result.json)"])
    if audit.get("formal_anchor"):
        lines.extend(["","本地查看器用正式公开目录中的 controller 和本目录一起渲染。","",
                      chr(96)*3+"sh",
                      "python -m rsi_memory.topology_report --run FORMAL_EXPORT/controller --ablation THIS_EXPORT --output NEW_REPORT.html",
                      chr(96)*3])
    return ("\n".join(lines)+"\n").encode()


def export(args, fixed=FIXED):
    need(not args.output.exists() and not args.output.is_symlink(),"new output directory required")
    for source in (args.control,args.study):
        need(not args.output.resolve().is_relative_to(source.resolve()),"publication output must be separate from live evidence")
    completion_raw = sealed(args.control/"completion.json",args.expected_completion_sha256)
    original = {f"waiter/{name}":completion_raw if name=="completion.json" else read(args.control/name) for name in CONTROL}
    prereg = sealed(args.preregistration,fixed["preregistration"])
    spec = json.loads(prereg)
    need(spec["schema"]==1 and spec["kind"]==FORMAT,"unknown supplement preregistration")
    original.update({"preregistration.json":prereg,"sglang_post_freeze_native6144.py":sealed(args.driver,fixed["driver"]),
                     "waiter/wait_and_run_sglang_native6144_v1.py":sealed(args.waiter,fixed["waiter"])})
    binding,completion = check_controls(original,spec,fixed)
    inventory = check_source(args.source_archive,spec)
    result_path = args.study/"result.json"
    if result_path.exists():
        raw = sealed(result_path,args.expected_result_sha256)
        result = json.loads(raw)
        need(completion["returncode"]==0 and completion["status"]==result["status"] and result["status"] in STATES
             and completion["timed_jobs_completed"]==result["timed_jobs_completed"]
             and result["preregistration_sha256"]==fixed["preregistration"],"waiter and driver terminal artifacts disagree")
        original["result.json"]=raw
    else:
        need(args.expected_result_sha256 is None and type(completion["returncode"]) is int and completion["returncode"]!=0
             and "status" not in completion,"nonzero terminal driver is required to archive missing result")
        result={"status":"driver_failed_without_result","performance_gate":"not_evaluated","timed_jobs_completed":0,
                "preregistration_sha256":fixed["preregistration"]}
    if args.study.exists():
        need(args.study.is_dir() and not args.study.is_symlink(),"invalid supplement directory")
        for name in RAW_STUDY:
            path = args.study/name
            if path.exists():
                original[name]=read(path)
        need(original.get("preregistration.json")==prereg,"original run preregistration differs")
        if args.expected_result_sha256:
            need(digest(original["result.json"])==args.expected_result_sha256,"result changed during export")
    sys.dont_write_bytecode = True
    sys.path.insert(0,str(args.repo_root.resolve()))
    from rsi_memory import topology_report as viewer, topology_report_ablation as ablation
    anchor, private_sha = None,None
    if binding.get("formal_report_sha256"):
        anchor,private_sha = formal_anchor(args.formal_export,args.expected_formal_report_sha256,binding,spec,inventory,viewer)
    else:
        need(args.expected_formal_report_sha256 is None and result["status"] in {
             "skipped_budget","skipped_formal_pipeline_failed","driver_failed_without_result"},
             "unlaunched case unexpectedly references formal evidence")
    bound = None
    if "manifest.json" in original:
        manifest = json.loads(original["manifest.json"])
        bound = manifest["bound"]
        fields = ("formal_report_sha256","formal_frozen_sha256","formal_manifest_canonical_sha256","formal_pipeline_status_sha256",
                  "adaptive_program_sha256","adaptive_trial_id","adaptive_harness_sha256")
        need(anchor is not None and all(bound[key]==anchor[key] for key in fields)
             and bound["formal_private_config_sha256"]==private_sha and HASH.fullmatch(bound["derived_private_config_sha256"])
             and set(bound["adapter_config_field_diff"])=={"receipt_directory"}
             and bound["preregistration_sha256"]==fixed["preregistration"],"supplement differs from formal sealed binding")
        if "frozen_adaptive.program.json" in original:
            need(canonical(json.loads(original["frozen_adaptive.program.json"]))==anchor["adaptive_program_sha256"],"frozen program changed")
        else:
            need(result["status"]=="driver_failed_without_result","supplement bound without frozen program")
    jobs = study_receipts(args.study,spec,original)
    if result["status"]=="driver_failed_without_result":
        need(not any(job["status"]=="completed" for job in jobs.values()) or bound is not None,"successful lease without formal binding")
        view = {"status":"no_terminal_driver_result"}
        ledger={"directly_referenced_job_ids":[],"supporting_or_unreferenced_job_ids":sorted(jobs)}
    else:
        need(bound is not None or not jobs,"missing formal binding for existing leases")
        ledger = job_rows(original,jobs,spec,result,bound)
        view = ablation.read_ablation(args.study,[anchor] if anchor else [],viewer.read_receipt,canonical)
        expected = {"completed":"complete","skipped_budget":"skipped","skipped_formal_pipeline_failed":"skipped",
                    "budget_incomplete":"partial","execution_incomplete":"partial"}[result["status"]]
        need(view["status"]==expected,"receipt and viewer verification could not reproduce the terminal state")
        if "started_utc" in result:
            need(time_utc(result["started_utc"])<=time_utc(spec["cutoffs"]["latest_start_utc"])
                 and time_utc(binding["utc"])<=time_utc(result["started_utc"])<=time_utc(completion["finished_utc"]),
                 "supplement started outside the recorded time gate")
    for name, raw in original.items():
        if name.endswith(".log"):
            need(not re.search(rb"sk-[A-Za-z0-9]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----",raw),"credential-like value in log " + name)
    audit={"schema":1,"scope":"settled post-freeze supplementary replay of the two formal heldout cases",
           "status":result["status"],"preregistration_sha256":fixed["preregistration"],
           "supplement_result_sha256":args.expected_result_sha256,"waiter_completion_sha256":args.expected_completion_sha256,
           "formal_anchor":anchor,"receipt_count":len(jobs),"receipt_status_counts":dict(Counter(row["status"] for row in jobs.values())),
           "leases":jobs,"ledger":ledger,"viewer_verification":view}
    args.output.mkdir(parents=True,exist_ok=False)
    for name,raw in original.items():
        p=args.output/name
        p.parent.mkdir(parents=True,exist_ok=True)
        p.write_bytes(raw)
    entries={k:v for k,v in original.items() if k.startswith("receipts/") and k.endswith(".json")}
    receipt_zip(args.output/"leased-receipts.zip",entries)
    (args.output/"audit.json").write_bytes(json_bytes(audit,ascii=False))
    (args.output/"README_ZH.md").write_bytes(report_md(result,audit))
    manifest={"schema":1,"files_sha256":{p.relative_to(args.output).as_posix():digest(p.read_bytes())
               for p in sorted(args.output.rglob("*")) if p.is_file()},
              "leased_receipt_member_sha256":{k:digest(v) for k,v in entries.items()},
              "frozen_source_archive_sha256":spec["controller_source_archive_sha256"],
              "frozen_source_manifest_sha256":spec["controller_source_manifest_sha256"],
              "viewer_verifier_sha256":{Path(m.__file__).name:digest(read(Path(m.__file__))) for m in (viewer,ablation)}}
    (args.output/"artifact_manifest.json").write_bytes(json_bytes(manifest))
    return audit


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("repo-root","study","control","preregistration","driver","waiter","source-archive","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--formal-export",type=Path,help="already-published formal directory, not the live formal directory")
    parser.add_argument("--expected-completion-sha256",required=True)
    parser.add_argument("--expected-result-sha256")
    parser.add_argument("--expected-formal-report-sha256")
    args=parser.parse_args()
    audit=export(args)
    print(json.dumps({"status":audit["status"],"receipt_count":audit["receipt_count"],"output":str(args.output)},sort_keys=True))


if __name__=="__main__":
    main()
