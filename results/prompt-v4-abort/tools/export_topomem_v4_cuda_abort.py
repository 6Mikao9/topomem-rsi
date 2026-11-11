"""Preserve the stopped v4.0 pilot and replay its local controller prefix."""
import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import tempfile

from export_topomem_v4_cuda_engineering import (
    CONTROLLER_SHA,CONTROLLER_ZIP,WORKER_SHA,WORKER_ZIP,FORMAL_PLAN,
    audit_receipts,check_geometry,digest,journal,need,nice,read,receipt_zip,safe_path,sanitize_config,seal,verify_publication,
    verify_source,verify_sources,write,
)

STOP="08ee325ab05e262cfd9123315603a70ae235a5dd91b796a25875e7536fe19563"
LEDGER="3949c006950105ae820d67f2d9c0cb437df138d64acf64435214946eb0abacd8"
PIPELINE="eb02236491586789a8f1f037ba11d37a0ea540ce05986bf634ec0eef951221cf"


def replay_prefix(source,temp,config,records,original,pending):
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(source))
    from rsi_memory.cuda_domain import CudaDomain
    from rsi_memory.cuda_v4 import checked_preregistration,contract_domain
    from rsi_memory.topology_rsi_v4 import TopologyRSIV4
    from rsi_memory import cuda_v4

    adapter=CudaDomain({**config,"output_directory":str(temp/"inert-adapter")},temp)
    pre=config["controller_preregistration"]
    consumed=set()

    class PendingBoundary(BaseException):
        pass

    class Offline(TopologyRSIV4):
        def _effect(self,key,details,operation,*,time_budget=True):
            need(key in records and key not in consumed and records[key]["input"]==details,
                 "recorded effect input did not replay "+key)
            consumed.add(key)
            row=self._save(key,records[key])
            if key==pending:
                raise PendingBoundary
            return row

    def no_api(_):
        raise AssertionError("offline audit cannot call the provider")

    out=temp/"replayed"
    controller=Offline(out,checked_preregistration(pre,config["paired_block_ids"]),contract_domain(adapter.domain(),pre),
                       no_api,backend_source=Path(cuda_v4.__file__))
    try:
        initial=controller._get("manifest")
        need({k:v for k,v in initial.items() if k!="started_unix"}==
             {k:v for k,v in records["manifest"].items() if k!="started_unix"},"offline manifest differs")
        controller._save("manifest",records["manifest"])
        try:
            controller.run()
        except PendingBoundary:
            pass
        else:
            raise ValueError("offline replay did not reach the original pending request")
        replayed={key:json.loads(value) for key,value in controller.db.execute("SELECT key,data FROM records")}
        need(replayed==records,"offline prefix differs from SQLite")
        for name,raw in original.items():
            if name.startswith("controller/harnesses/"):
                need(read(out/name.removeprefix("controller/"))==raw,"generator source did not replay")
        return adapter.contract,len(consumed)
    finally:
        controller.close()
        adapter.close()


def export(args):
    root=args.run_root
    stop=seal(root/"operator-stop.json",STOP)
    pipeline=seal(root/"pipeline-status.json",PIPELINE)
    original={"operator-stop.json":stop,"pipeline-status.json":pipeline,
              "ledger-export.json":seal(root/"ledger-export.json",LEDGER),
              "pipeline-plan.json":read(root/"pipeline-plan.json"),"pipeline-run.log":read(root/"pipeline-run.log")}
    records,record_bytes=journal(root,original,settled=False)
    marker=json.loads(stop)
    need(json.loads(original["ledger-export.json"])==records and marker["completed_rounds"]==0
         and marker["journal_sha256"]==digest(original["controller/journal.sqlite"])
         and json.loads(pipeline)[0]["returncode"]==marker["pipeline_exitcode"]!=0,
         "operator stop marker or journal changed")
    state=[{"key":key,"status":row["status"]} for key,row in records.items() if key.startswith(("api/","eval/","pair/"))]
    need(state==marker["states"] and not any(k=="frozen" or k.startswith(("round/","pair/","training_matrix/")) for k in records)
         and all(row["input"]["split"]!="test" for key,row in records.items() if key.startswith("eval/")),
         "abort phase does not match journal chronology")
    calls={k:v for k,v in records.items() if k.startswith("api/")}
    completed={k:v for k,v in calls.items() if v["status"]=="completed"}
    pending=[k for k,v in calls.items() if v["status"]=="pending"]
    need(len(calls)==7 and len(completed)==6 and pending==["api/r0000/fixed-a/00"],"abort provider roster differs")
    sent=records["api/r0000/adaptive-meta"]["input"]["payload"]["messages"][0]["content"]
    need("Both sides may retain the best measured training program in the common starting archive." in sent,
         "expected historical meta clause absent")
    private=read(root/"private-config-r2.json")
    config=json.loads(private)
    plan=seal(root/"case-plan-r2.json",FORMAL_PLAN)
    case_plan=json.loads(plan)
    need(digest(private)==case_plan["private_config_sha256"]==json.loads(original["pipeline-plan.json"])["private_config_sha256"],
         "private configuration fingerprint differs")
    report={"manifest":records["manifest"]}
    contract=check_geometry(config,case_plan,report)
    controller_raw=seal(root/"controller-source.zip",CONTROLLER_ZIP)
    worker_raw=seal(args.worker_source_archive,WORKER_ZIP)
    original.update({"inputs/case-plan-r2.json":plan,"inputs/case-plan-parent.json":seal(root/"case-plan.json",case_plan["supersedes_case_plan_sha256"]),
        "inputs/adapter-contract.json":nice(contract),"inputs/adapter-config.template.json":nice(sanitize_config(config)),
        "source/controller-source.zip":controller_raw,"source/worker-source.zip":worker_raw,"source/source-freeze.json":read(root/"source-freeze.json"),
        "controller/manifest.json":read(root/"controller/manifest.json")})
    for name in ("lease.sqlite","lease.sqlite-journal"):
        path=root/"controller"/name
        if path.is_file():
            original[f"controller/{name}"]=read(path)
    for p in (root/"controller/harnesses").glob("*.py"):
        raw=read(p)
        need(digest(raw)==p.stem,"generator source filename digest differs")
        original[f"controller/harnesses/{p.name}"]=raw
    qualifications={}
    for name in case_plan["qualification_order"]:
        raw=read(root/(name+"-qualification.json"))
        qualifications[name]=json.loads(raw)
        original[f"qualification/{name}.json"]=raw
    with tempfile.TemporaryDirectory(prefix="topomem-v4-abort-replay-") as tmp:
        temp=Path(tmp)
        m,files=verify_source(controller_raw,CONTROLLER_SHA,temp/"source")
        w,worker=verify_source(worker_raw,WORKER_SHA)
        need(m==read(root/"controller-source-manifest.json"),"separate controller source manifest differs")
        original["source/controller-source-manifest.json"]=m
        original["source/worker-source-manifest.json"]=w
        verify_sources(report,files)
        base,effect_count=replay_prefix(temp/"source",temp,config,records,original,pending[0])
    receipts,receipt_rows,paired,hardware=audit_receipts(root,config,base,records,qualifications)
    need(len(receipts)==11 and not paired,"abort worker evidence roster differs")
    totals={arm:{"calls":0,"total_tokens":0,"seconds":0.0} for arm in ("adaptive","fixed")}
    for key,row in completed.items():
        stats=totals["fixed" if key.split("/")[2].startswith("fixed") else "adaptive"]
        stats["calls"]+=1
        stats["total_tokens"]+=row["result"]["usage"].get("total_tokens",0)
        stats["seconds"]+=row["seconds"]
    provider={"completed":6,"pending_unknown":pending[0],"by_arm":totals,
              "total_reported_tokens":sum(row["total_tokens"] for row in totals.values())}
    audit={"schema":1,"scope":"operator-stopped v4.0 prompt-contract pilot on public CUDA fixtures",
           "operator_stop_sha256":STOP,"pipeline_status_sha256":PIPELINE,"private_config_sha256":digest(private),
           "controller_source_sha256":CONTROLLER_SHA,"worker_source_sha256":WORKER_SHA,
           "exact_offline_replay_through_pending_api":True,"effect_count":effect_count,"journal_record_count":len(records),
           "provider":provider,"receipt_count":len(receipts),"receipt_counts_by_split":dict(Counter(r["split"] for r in receipt_rows.values())),
           "qualification_jobs":4,"controller_jobs":7,"completed_rounds":0,"native_validation_baseline_measured":True,
           "candidate_validation_effects":0,"freeze_effects":0,"test_effects":0,"hardware":hardware}
    original["audit.json"]=nice(audit)
    original["server_job_index.json"]=nice({"schema":1,"jobs":receipt_rows})
    original["tools/export_topomem_v4_cuda_abort.py"]=read(Path(__file__))
    original["tools/export_topomem_v4_cuda_engineering.py"]=read(Path(__file__).with_name("export_topomem_v4_cuda_engineering.py"))
    api={f"{k}.json":v for k,v in record_bytes.items() if k.startswith(("api/","plan/"))}
    docs=f"""# CUDA v4.0 提示合同中止记录

操作方在 2026-11-11 05:52:15 UTC 停止运行。实际 meta 请求写明双方可以保留共同起始档案中的最佳程序。冻结 v4.0 控制器从父、子各自的新候选槽选择对应输出，另将 incumbent 纳入独立验证。中止发生在第一轮固定臂首个请求仍为 pending 时。提示与裁判的差异由请求记录及冻结源码直接核对。

完成 6 个模型请求，提供方报告 {provider['total_reported_tokens']:,} tokens。第 7 个请求的提供方结果与用量未知。11 份已完成 GPU 租约回执覆盖 4 项启动资格、7 项控制器评估，其中一项为原生验证基线。运行完成 0 轮，候选配对验证、冻结与测试均未发生。

本地重放冻结控制器，在原 pending 请求边界停止，逐项核对 provider 输入、训练评估身份、原始源与 SQLite 全部记录。GPU 回执重算了请求、动作和选中输出指纹。公开模板保留固定输入与摘要，部署连接由操作方提供。

[原中止记录](operator-stop.json) · [原 journal](controller/journal.sqlite) · [审计](audit.json) · [原 API 与方案](original-api-and-plans.zip) · [11 份原回执](leased-receipts.zip) · [工件清单](artifact_manifest.json)
"""
    original["README_ZH.md"]=docs.encode()
    for name,data in {**original,**api}.items():
        if name.endswith((".log",".json",".py")):
            need(not re.search(rb"sk-[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----",data),"credential-like value in "+name)
    args.output.mkdir(exist_ok=False,parents=True)
    for name,raw in original.items():
        write(args.output/safe_path(name),raw)
    receipt_zip(args.output/"leased-receipts.zip",receipts)
    receipt_zip(args.output/"original-api-and-plans.zip",api)
    files={p.relative_to(args.output).as_posix():digest(read(p)) for p in sorted(args.output.rglob("*")) if p.is_file()}
    write(args.output/"artifact_manifest.json",nice({"schema":1,"files_sha256":files,"private_config_sha256":digest(private),
        "leased_receipt_member_sha256":{n:digest(b) for n,b in receipts.items()},
        "original_api_and_plan_member_sha256":{n:digest(b) for n,b in api.items()}}))
    return {"verified":verify_publication(args.output),"provider":provider,"replayed_effects":effect_count}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for flag in ("run-root","worker-source-archive","output"):
        parser.add_argument("--"+flag,required=True,type=Path)
    print(json.dumps(export(parser.parse_args()),sort_keys=True,ensure_ascii=False))


if __name__=="__main__":
    main()
