"""Preserve the interrupted CUDA v4.1 run and verify its observed effects offline."""
from argparse import ArgumentParser
from collections import Counter
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory

import export_topomem_v41_cuda_engineering as common

REPORT = "fd7ef104c9da2c61de8b438eeb762aab6f57be49dd088b1a7d9ac078908b906d"
PIPELINE = "60d936e860183d5dcaa6b443c5701b8448ea906479fba9e8fc3f91357f0da33a"
PIPELINE_SOURCE = "ec8aeba8b30faf809253cb7eeb4f4686666c7a24bd9bdd5d94eb0b9d0aef6eef"
PRIVATE = "fe8b2f8cc554f6072c57f84d619d3a26931c1481b0523352e819babcf6c7f336"
UNKNOWN = "eval-99fd473a153c7b215cfbd43dabcd82562b13e01114982104"
RECOVERY_NAMES = {"input.json","process.log","request.json","started.json"}
SECRETS = re.compile(rb"sk-[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----")


def verify_interruption(root,original):
    original["controller/report.json"]=common.seal(root/"controller/report.json",REPORT)
    original["pipeline-status.json"]=common.seal(root/"pipeline-status.json",PIPELINE)
    report=json.loads(original["controller/report.json"])
    phases=json.loads(original["pipeline-status.json"])
    common.need([(row["phase"],row["returncode"]) for row in phases]==[("run",0),("freeze",1)]
                and report["status"]=="training_matrix_incomplete" and report["rounds"]==[]
                and report["frozen"] is None and report["paired_evaluations"]=={}
                and report["test_comparisons"] is None and not (root/"controller/frozen.json").exists(),
                "the original run is no longer the declared incomplete experiment")
    records,record_bytes=common.journal(root,original,settled=False)
    original["controller/manifest.json"]=common.read(root/"controller/manifest.json")
    common.need(json.loads(original["controller/manifest.json"])==records["manifest"]==report["manifest"],
                "controller manifest and journal disagree")
    for prefix,field in (("api/","calls"),("eval/","evaluations"),("pair/","paired_evaluations"),
                         ("plan/","meta_plans"),("training_matrix/","training_matrices"),
                         ("selection_context/","selection_contexts")):
        common.need({k:v for k,v in records.items() if k.startswith(prefix)}==report[field],"report records differ "+field)
    common.need(not any(k.startswith(("round/","pair/","frozen","test")) for k in records)
                and all(r["input"]["split"]!="test" for k,r in records.items() if k.startswith("eval/")),
                "unexpected selection or test journal record")
    calls=report["calls"]
    common.need(len(calls)==10 and all(k.startswith("api/r0000/") and v["status"]=="completed" for k,v in calls.items()),
                "API roster differs from the interrupted first round")
    failures={k:v for k,v in report["evaluations"].items() if v["status"]!="completed"}
    common.need(len(failures)==1,"expected exactly one unresolved controller effect")
    unknown,record=next(iter(failures.items()))
    common.need(unknown.startswith("eval/r0000/training-matrix/r0000-train-02/")
                and record["status"]=="failed" and record["failure"]=="ExecutionError" and "result" not in record,
                "unresolved effect changed")
    return report,records,record_bytes,unknown


def replay(source,config,records,original,report,temp):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(source))
    from rsi_memory.cuda_domain import CudaDomain
    from rsi_memory.cuda_v4 import checked_preregistration,contract_domain
    from rsi_memory.topology_rsi_v4 import TopologyRSIV4
    from rsi_memory import cuda_v4

    adapter=CudaDomain({**config,"output_directory":str(temp/"inert-adapter")},temp)
    pre=config["controller_preregistration"]
    consumed=set()

    class ArchivedEffectsController(TopologyRSIV4):
        def _effect(self,key,details,operation,*,time_budget=True):
            common.need(key in records and key not in consumed and records[key]["input"]==details,
                        "offline effect input or order differs "+key)
            consumed.add(key)
            return self._save(key,records[key])

    def no_provider(_):
        raise AssertionError("offline replay cannot access the model")

    output=temp/"controller-replay"
    controller=ArchivedEffectsController(output,checked_preregistration(pre,config["paired_block_ids"]),
                   contract_domain(adapter.domain(),pre),no_provider,backend_source=Path(cuda_v4.__file__))
    try:
        old=controller._get("manifest")
        common.need({k:v for k,v in old.items() if k!="started_unix"}==
                    {k:v for k,v in records["manifest"].items() if k!="started_unix"},"replay contract differs")
        controller._save("manifest",records["manifest"])
        common.need(controller.run()==report,"frozen source did not reproduce the interrupted report")
        try:
            controller.freeze()
        except ValueError as error:
            common.need(str(error)=="configured rounds are incomplete; explicitly allow a partial freeze",
                        "freeze replay failed for a different reason")
        else:
            raise ValueError("offline freeze unexpectedly sealed a finalist")
        replayed={key:json.loads(value) for key,value in controller.db.execute("SELECT key,data FROM records")}
        common.need(replayed==records and consumed=={k for k in records if k.startswith(("api/","eval/","pair/"))},
                    "offline replay changed archived effects or decisions")
        for name,raw in original.items():
            if name.startswith("controller/harnesses/"):
                common.need(common.read(output/name.removeprefix("controller/"))==raw,
                            "offline generated source differs "+name)
        return adapter.contract,len(consumed)
    finally:
        controller.close()
        adapter.close()


def copy_evidence(source,original,prefix):
    manifest=common.read(source/"evidence-manifest.json")
    original[prefix+"/evidence-manifest.json"]=manifest
    for name,digest in json.loads(manifest)["files"].items():
        original[prefix+"/"+name]=common.seal(source/common.safe_path(name),digest)


def recovery(root,original):
    inspection=common.read(root/"recovery-inspection.json")
    sources=common.read(root/"post-reboot-source-check.json")
    data=json.loads(inspection)
    bindings=json.loads(sources)
    job=root/UNKNOWN
    common.need(data["job_id"]==UNKNOWN and data["copy_returncode"]==0 and set(data["files"])==RECOVERY_NAMES
                and {path.name for path in job.iterdir()}==RECOVERY_NAMES and data["boot_id"]==bindings["boot_id"]
                and any(s["manifest_sha256"]==common.WORKER_SHA for s in bindings["verified_old_sources"]),
                "recovery or post-reboot source binding differs")
    for name,meta in data["files"].items():
        raw=common.seal(job/name,meta["sha256"])
        common.need(meta["bytes"]==len(raw)==0,"recovered file is no longer empty")
        original[f"recovery/{UNKNOWN}/{name}"]=raw
    original["recovery/recovery-inspection.json"]=inspection
    original["recovery/post-reboot-source-check.json"]=sources
    return data["container_state"]


def verify_unknown(unknown,records,base,config):
    path=unknown.removeprefix("eval/")
    program=records[unknown]["input"]["program"]
    payload={"schema":1,"arm":"generated","program":program,"cases":config["cases"]["train"],
             "warmup":base["warmup"],"samples":base["samples"],"expected_gpu_uuids":config["executor"]["gpu_uuids"]}
    identity={"experiment":config["experiment_id"],"contract":base,"evaluation":path,
              "label":"candidate","block":"train","payload":payload}
    common.need("eval-"+common.canon(identity)[:48]==UNKNOWN,"recovered job differs from unresolved controller effect")
    return {"effect":path,"job_id":UNKNOWN,"status":"transport_unknown","local_receipt":False,
            "recovered_empty_files":sorted(RECOVERY_NAMES)}


def export(args):
    root=args.run_root
    common.need(not args.output.exists() and not args.output.resolve().is_relative_to(root.resolve()),
                "new independent publication required")
    original={}
    report,records,raw_records,unknown=verify_interruption(root,original)
    private=common.seal(root/"private-config.json",PRIVATE)
    config=json.loads(private)
    original["inputs/adapter-config.template.json"]=common.nice(common.sanitize_config(config))
    original["inputs/case-plan.json"]=common.seal(root/"case-plan.json",common.FORMAL_PLAN)
    case_plan=json.loads(original["inputs/case-plan.json"])
    original["inputs/adapter-contract.json"]=common.nice(common.check_geometry(config,case_plan,report))
    original["pipeline-plan.json"]=common.read(root/"pipeline-plan.json")
    pipeline=json.loads(original["pipeline-plan.json"])
    common.need(pipeline["case_plan_sha256"]==common.FORMAL_PLAN and pipeline["private_config_sha256"]==PRIVATE
                and pipeline["script_sha256"]==PIPELINE_SOURCE and pipeline["source_manifest_sha256"]==common.CONTROLLER_SHA,
                "run plan binding differs")
    original["source/pipeline-launcher.py"]=common.seal(args.pipeline_source,PIPELINE_SOURCE)
    original["source/controller-source.zip"]=common.seal(root/"controller-source.zip",common.CONTROLLER_ZIP)
    original["source/worker-source.zip"]=common.seal(args.worker_source_archive,common.WORKER_ZIP)
    for name in ("pipeline-run.log","pipeline-freeze.log","controller/lease.sqlite",
                 "gpu-evidence/receipts/connection.log"):
        original[name]=common.read(root/name)
    common.need(b"Connection reset" in original["gpu-evidence/receipts/connection.log"],"transport message differs")
    for path in (root/"controller/harnesses").glob("*.py"):
        original[f"controller/harnesses/{path.name}"]=common.seal(path,path.stem)
    copy_evidence(args.independent_root/"v41-incomplete-public",original,"independent/v41-incomplete-public")
    copy_evidence(args.independent_root/"v41-recovery-followup",original,"independent/v41-recovery-followup")
    container=recovery(args.recovery_root,original)
    reused,reused_ids=common.verify_reused_qualification(root,args.qualification_publication,config,case_plan,original)
    common.audit_provider_envelopes(report,config)
    with TemporaryDirectory(prefix="topomem-v41-reboot-replay-") as directory:
        temp=Path(directory)
        source=temp/"source"
        manifest,controller_files=common.verify_source(original["source/controller-source.zip"],common.CONTROLLER_SHA,source)
        worker,worker_files=common.verify_source(original["source/worker-source.zip"],common.WORKER_SHA)
        common.need(manifest==common.read(root/"controller-source-manifest.json"),"controller source manifest differs")
        original["source/controller-source-manifest.json"]=manifest
        original["source/worker-source-manifest.json"]=worker
        common.verify_sources(report,controller_files)
        base,effects=replay(source,config,records,original,report,temp)
    receipts,rows,paired,hardware=common.audit_receipts(root,config,base,{k:v for k,v in records.items() if k!=unknown},{})
    unknown_fact=verify_unknown(unknown,records,base,config)
    common.need(not paired and UNKNOWN not in rows and len(receipts)==26 and len(reused)==4
                and not (set(rows)&set(reused_ids)),"local or historic receipt roster differs")
    matrix=report["training_matrices"]["training_matrix/r0000/result"]
    observations=Counter(item["block_id"] for value in matrix["measurements"].values() for item in value["evaluations"])
    blocks=matrix["plan"]["block_ids"]
    common.need([observations[block] for block in blocks]==[9,7,0] and len(matrix["measurements"])==9,
                "partial matrix observation shape differs")
    matrix_jobs={job for value in matrix["measurements"].values() for item in value["evaluations"]
                 for job in item.get("reported_receipts", ())}
    common.need(len(matrix_jobs)==15 and matrix_jobs<=rows.keys(),"partial matrix receipt roster differs")
    audit={"schema":1,"status":"training_matrix_incomplete","report_sha256":REPORT,"pipeline_status_sha256":PIPELINE,
           "controller_source_sha256":common.CONTROLLER_SHA,"worker_source_sha256":common.WORKER_SHA,
           "controller_source_files":len(controller_files),"worker_source_files":len(worker_files),
           "exact_offline_controller_replay":True,"archived_effects_replayed":effects,
           "provider_calls":len(report["calls"]),"provider_usage":report["usage"],"rounds_completed":0,"finalists":None,
           "validation_submitted":False,"native_validation_baseline_observed":True,"test_submitted":False,
           "local_receipts":len(receipts),"local_receipts_by_split":dict(Counter(row["split"] for row in rows.values())),
           "local_receipts_feasible":sum(row["feasible"] for row in rows.values()),
           "local_receipts_by_role":{"training_matrix":len(matrix_jobs),"pre_matrix":len(receipts)-len(matrix_jobs)},
           "reused_qualification_receipts":len(reused),"reused_qualification_job_ids":reused_ids,
           "matrix_identity_count":len(matrix["measurements"]),"matrix_observations":dict(zip(blocks,[observations[b] for b in blocks])),
           "unknown":unknown_fact,"recovery_container_state":container,"hardware":hardware}
    api={f"{key}.json":raw for key,raw in raw_records.items() if key.startswith(("api/","plan/"))}
    for name,raw in {**original,**api}.items():
        if name.endswith((".log",".json",".py")):
            common.need(not SECRETS.search(raw),"credential-like value in "+name)
    args.output.mkdir(exist_ok=False,parents=True)
    for name,raw in original.items():
        common.write(args.output/common.safe_path(name),raw)
    for name,members in (("leased-receipts.zip",receipts),("original-api-and-plans.zip",api),
                         ("qualification/reused-receipts.zip",reused)):
        common.receipt_zip(args.output/name,members)
    common.write(args.output/"audit.json",common.nice(audit))
    common.write(args.output/"server_job_index.json",common.nice({"schema":1,"jobs":rows,"unknown":unknown_fact}))
    common.write(args.output/"README_ZH.md",readme(audit))
    for tool in (Path(__file__),Path(common.__file__)):
        common.write(args.output/"tools"/tool.name,common.read(tool))
    files={p.relative_to(args.output).as_posix():common.digest(common.read(p)) for p in sorted(args.output.rglob("*")) if p.is_file()}
    artifact={"schema":1,"files_sha256":files,"private_config_sha256":PRIVATE,
              "leased_receipt_member_sha256":{name:common.digest(raw) for name,raw in receipts.items()},
              "original_api_and_plan_member_sha256":{name:common.digest(raw) for name,raw in api.items()},
              "reused_qualification_receipt_member_sha256":{name:common.digest(raw) for name,raw in reused.items()}}
    common.write(args.output/"artifact_manifest.json",common.nice(artifact))
    common.verify_publication(args.output)
    return audit


def readme(audit):
    return (f"""# CUDA v4.1 双卡工程复评中断

四个已公开场景采用 H256。每案在预置的单张计算卡执行注意力，四案计算设备依次为 GPU0、GPU1、GPU0、GPU1。独立实验共计划三轮，每臂两槽。第一轮在训练矩阵中遇到连接重置，保留 9、7、0 条区块观测。10 次模型调用均返回。流程记录 `run` 返回 0 并报告矩阵不完整，`freeze` 返回 1。原生验证基准已有回执。候选验证、提交轮次、来源晋级、程序冻结及留出执行均未发生。

本次保存 {audit['local_receipts']} 份本地原始租约回执，其中 15 份属于部分训练矩阵。另有四份启动资格回执来自 [此前提示中断试验](../prompt-v4-abort/README_ZH.md)，均为复用。训练回执只支持逐次观测，不形成完整配对比较。冻结源码用封存的外部效果离线重放，重现原报告、账本和 `freeze` 拒绝。

未知作业的标识由原效果输入与请求合同重算，与恢复检查完全一致。宿主机重启后、恢复容器服务前复制了其目录。四个原文件均为 0 字节。该作业没有本地回执，也没有远端完成结果。保留 `transport_unknown`，没有补测或拼接矩阵。

[原报告](controller/report.json) · [流程状态](pipeline-status.json) · [复核结果](audit.json) · [独立矩阵审计](independent/v41-incomplete-public/README.md) · [恢复检查](recovery/recovery-inspection.json) · [原租约回执](leased-receipts.zip) · [公开配置模板](inputs/adapter-config.template.json) · [工件清单](artifact_manifest.json)

原私有配置只保留 SHA-256。完整源码重放需要原配置；公开配置模板供新实验填写本机连接。公开文件与归档成员可离线核验。

```sh
python -B tools/export_topomem_v41_reboot_interruption.py verify --publication .
```
""").encode()


def main():
    parser=ArgumentParser(description=__doc__)
    commands=parser.add_subparsers(dest="phase",required=True)
    package=commands.add_parser("export")
    for name in ("run-root","worker-source-archive","pipeline-source","qualification-publication",
                 "independent-root","recovery-root","output"):
        package.add_argument("--"+name,type=Path,required=True)
    check=commands.add_parser("verify")
    check.add_argument("--publication",type=Path,required=True)
    args=parser.parse_args()
    result=export(args) if args.phase=="export" else common.verify_publication(args.publication)
    print(json.dumps(result,sort_keys=True,allow_nan=False))


if __name__=="__main__":
    main()
