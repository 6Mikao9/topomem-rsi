"""Publish completed v11/v12 GPU3 engineering receipts and paired statistics."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path, PurePosixPath
import re
from statistics import median
import tempfile
from types import SimpleNamespace
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

ROOT = Path(__file__).resolve().parent
PLAN = "59d19cb8a0ca127fe7ac5b982e0b9b9f24d45dff7b1e33946f1ce734f0af2774"
DRIVER = "249d330f3323006c43a26b5313688e815c15ea6b4adba8b8fc8ce55c5f854a70"
HEX = re.compile(r"[0-9a-f]{64}")
RECEIPT = re.compile(r"v1[12]/qualification-v12-[a-f0-9]{12}-b[0-7]-(?:native|writeback-v11|writeback-v12)\.json")
ARTIFACTS = {
    "benchmark_dynamic_feature_lowering.py":"a097547a6950fc1d9fbfaa9291d9e8a62253a2d3948c8be61a67a7ef92518422",
    "cpu-benchmark-v12.json":"8554ed9398cd1f3d21715c3ed25925d97cc3668432c31940f6dca6b22553d83c",
    "cpu-benchmark-v12-repeat.json":"499d2e9c286bc4d14723af35fe556e4ce3dcee876a4eb5117582c7df25ca6b0a",
    "integrations/topology_memory/torch_runtime.py":"2b50f89c6541cd4d47af774a5c9ac7d841c03c7e3ab9c81afbed6d1a0c9c97ab",
    "integrations/topology_memory/torch_dynamic_features.py":"e5b636b177c2cceae9328a48bb51e1ba46de585312630aa28285dd7509b3a445",
    "tests/test_dynamic_feature_lowering.py":"1134cd34c3854f3e1d841d4248a5d70f126215112e8fe8d7750feede7296e231",
    "tests/test_qualify_v12_paired_gpu3.py":"df49bb25a69448112d5a78b60741385720d0af3c965501b703410581d1eb3e83",
}
ARMS = ("native","writeback-v11","writeback-v12")


def need(condition,message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def read(path):
    need(path.is_file() and not path.is_symlink(),"missing or symlinked file "+path.name)
    return path.read_bytes()


def sealed(path,expected):
    need(isinstance(expected,str) and HEX.fullmatch(expected),"missing SHA-256 seal for "+path.name)
    raw=read(path)
    need(digest(raw)==expected,"SHA-256 differs for "+path.name)
    return raw


def nice(value):
    return (json.dumps(value,indent=2,sort_keys=True,ensure_ascii=False,allow_nan=False)+"\n").encode()


def write(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open("xb") as stream:
        stream.write(data)


def pair(values,reference,candidate):
    deltas=[(row[reference]-row[candidate])/1e6 for row in values]
    return {"positive_blocks":sum(x>0 for x in deltas),"tied_blocks":sum(x==0 for x in deltas),
            "negative_blocks":sum(x<0 for x in deltas),"median_ms":median(deltas),
            "minimum_ms":min(deltas),"maximum_ms":max(deltas),"deltas_ms":deltas}


def scope_stats(values):
    old_to_new=pair(values,"writeback-v11","writeback-v12")
    native_to_old=pair(values,"native","writeback-v11")
    native_to_new=pair(values,"native","writeback-v12")
    old_gap=-native_to_old["median_ms"]
    recovered=old_to_new["median_ms"]/old_gap if old_gap>0 else None
    return {"block_median_windows_ms_by_arm":[{label:ns/1e6 for label,ns in row.items()} for row in values],
            "median_window_ms_by_arm":{label:median(row[label] for row in values)/1e6 for label in ARMS},
            "v11_minus_v12":old_to_new,"native_minus_v11":native_to_old,"native_minus_v12":native_to_new,
            "v11_gap_to_native_paired_median_ms":old_gap,"old_gap_recovery_ratio":recovered,
            "recovered_at_least_half_of_positive_old_gap":None if recovered is None else recovered>=0.5}


def raw_receipts(run_directory,plan,ledger_raw,analysis):
    ledger=[json.loads(line) for line in ledger_raw.splitlines()]
    schedule=[(b["block"],arm) for b in plan["schedule"] for arm in b["order"]]
    need([(row["block"],row["arm"]) for row in ledger]==schedule,"recorded order differs from plan")
    members={}
    raw_results={}
    for row in ledger:
        member=f"{row['source']}/{row['job_id']}.json"
        need(RECEIPT.fullmatch(member) and member not in members,"invalid or repeated receipt member")
        members[member]=sealed(run_directory/member,row["receipt_sha256"])
        result=json.loads(members[member])["result"]
        raw_results[row["block"],row["arm"]]=result
    observed={f"{version}/{p.name}" for version in ("v11","v12") for p in (run_directory/version).glob("*.json")}
    need(set(members)==observed and len(members)==24,"receipt roster differs from all 24 scheduled jobs")
    values={name:[] for name in plan["case_sha256"]}
    totals=[]
    for block in range(8):
        reported=analysis["blocks"][block]
        need(reported["block"]==block,"analysis block order differs")
        counts={label:0 for label in ARMS}
        rows={label:raw_results[block,label] for label in ARMS}
        for case_name in values:
            row={}
            for arm,result in rows.items():
                case=next(c for c in result["cases"] if c["name"]==case_name)
                samples=[timing["elapsed_ns"] for timing in case["timings"]]
                elapsed=median(samples)
                audited=reported["case_scores"][case_name][arm]
                need(len(samples)==5 and elapsed==case["median_elapsed_ns"]
                     and audited=={"samples_ns":samples,"median_ms":elapsed/1e6},"case timing or median differs")
                row[arm]=elapsed
                counts[arm]+=elapsed
            values[case_name].append(row)
        need({arm:counts[arm]/1e6 for arm in ARMS}==reported["sum_ms_by_arm"],"sum of case medians differs")
        totals.append(counts)
    stats={"aggregate":scope_stats(totals),"cases":{case:scope_stats(rows) for case,rows in values.items()}}
    # Raw nanosecond deltas preserve precision. The frozen analyzer subtracts
    # two separately converted millisecond values.
    for field,key in (("old_minus_new_paired_median_ms","v11_minus_v12"),("native_minus_new_paired_median_ms","native_minus_v12")):
        need(abs(analysis[field]-stats["aggregate"][key]["median_ms"])<1e-9,"analysis aggregate comparison differs")
    return members,stats


def receipt_zip(path,members):
    with ZipFile(path,"x",compression=ZIP_DEFLATED,compresslevel=9) as package:
        for name,data in sorted(members.items()):
            info=ZipInfo(name,date_time=(2020,1,1,0,0,0))
            info.external_attr=0o644<<16
            package.writestr(info,data,compress_type=ZIP_DEFLATED,compresslevel=9)


def build_readme(stats,plan):
    aggregate=stats["aggregate"]
    lines=["# CUDA GPU3 动态特征 lowering 工程对照","",
           "八个交错区块比较原生 LRU、v11 writeback 和同一 writeback IR 的 v12。"
           "四个已有公开训练场景共享 GPU3、CPU64、三次预热和五次计时。"
           "该结果估计执行器差异，属于固定公开输入的工程对照。","",
           "| 场景 | v12 较 v11 更快的块 | v11 − v12 配对中位差 ms | v12 较原生更快的块 | 原生 − v12 配对中位差 ms | 旧差距收回比例 |",
           "|---|---:|---:|---:|---:|---:|"]
    for name,row in [("四案汇总",aggregate),*stats["cases"].items()]:
        before,baseline=row["v11_minus_v12"],row["native_minus_v12"]
        recovery=row["old_gap_recovery_ratio"]
        ratio=f"{recovery:+.1%}" if recovery is not None else "—"
        lines.append(f"| {name} | {before['positive_blocks']}/8 | {before['median_ms']:+.6f} | "
                     f"{baseline['positive_blocks']}/8 | {baseline['median_ms']:+.6f} | {ratio} |")
    lines.extend(["","正值表示 v12 窗口缩短。旧差距收回比例是事后描述量，分子为配对中位差 v11 − v12，"
                  "分母为正的配对中位差 v11 − 原生。旧方案没有正差距的场景用横线表示。"
                  "预注册没有规定一半为验收门槛。","",
                  "两版 writeback 的有序动作、生命周期、选中 KV、注意力输出、缺页、链路字节和容量逐案相同。"
                  "原生对照匹配固定输入和选中输出。","",
                  "[预注册](inputs/preregistration.json) · [配对统计](paired-summary.json) · [原始分析](run/analysis.json) · [回执 ZIP](leased-receipts.zip) · [工件清单](artifact_manifest.json)","",
                  "本地复核可以从公开目录执行。","",chr(96)*3+"sh",
                  "python -B tools/export_v12_gpu3_engineering.py verify --publication . --replay-analysis",chr(96)*3])
    return ("\n".join(lines)+"\n").encode()


def replay_analysis(driver,plan,directory,expected):
    sealed(driver,DRIVER)
    spec=importlib.util.spec_from_file_location("_frozen_v12_qualifier",driver)
    qualifier=importlib.util.module_from_spec(spec)
    spec.loader.exec_module(qualifier)
    with tempfile.TemporaryDirectory(prefix="topomem-v12-analysis-") as tmp:
        rebuilt=Path(tmp)/"analysis.json"
        qualifier.analyze(SimpleNamespace(plan=plan,expected_plan_sha256=PLAN,audit_directory=directory,output=rebuilt))
        need(rebuilt.read_bytes()==expected,"frozen analysis did not replay byte-for-byte")


def export(args):
    need(not args.output.exists(),"new output directory required")
    prototype=args.prototype_root
    plan_dir=prototype/"gpu3-paired-study-r2"
    raw_plan=sealed(plan_dir/"preregistration.json",PLAN)
    plan=json.loads(raw_plan)
    raw_analysis=sealed(args.analysis,args.expected_analysis_sha256)
    ledger=sealed(args.run_directory/"jobs.jsonl",args.expected_ledger_sha256)
    analysis=json.loads(raw_analysis)
    need(analysis["status"]=="completed" and analysis["jobs"]==24
         and analysis["plan_sha256"]==PLAN and analysis["same_ir_physical_trace_verified"] is True,"study not settled")
    driver=prototype/"qualify_v12_paired_gpu3.py"
    source=sealed(driver,DRIVER)
    replay_analysis(driver,plan_dir/"preregistration.json",args.run_directory,raw_analysis)
    members,stats=raw_receipts(args.run_directory,plan,ledger,analysis)
    artifacts={"inputs/preregistration.json":raw_plan,"run/analysis.json":raw_analysis,"run/jobs.jsonl":ledger,
               "run/run-binding.json":read(args.run_directory/"run-binding.json"),
               "tools/qualify_v12_paired_gpu3.py":source,
               "tools/export_v12_gpu3_engineering.py":read(Path(__file__))}
    for version,binding in plan["source_versions"].items():
        for filename,key in ((version+".source.zip","archive_sha256"),(version+".source-manifest.json","manifest_sha256")):
            artifacts["inputs/"+filename]=sealed(plan_dir/filename,binding[key])
        log=args.run_directory/version/"connection.log"
        if log.is_file():
            artifacts[f"run/{version}.connection.log"]=read(log)
    for filename,expected in ARTIFACTS.items():
        artifacts["prototype/"+filename]=sealed(prototype/filename,expected)
    if Path(__file__).with_name("tests").joinpath("test_export_v12_gpu3_engineering.py").is_file():
        artifacts["tools/test_export_v12_gpu3_engineering.py"]=read(Path(__file__).with_name("tests")/"test_export_v12_gpu3_engineering.py")
    summary={"schema":1,"scope":"same writeback IR engineering comparison on public training cases",
             "plan_sha256":PLAN,"analysis_sha256":args.expected_analysis_sha256,"ledger_sha256":args.expected_ledger_sha256,
             "correctness_and_same_ir_physical_trace_verified":True,"receipt_count":24,"statistics":stats,
             "old_gap_recovery_preregistered":False,"old_gap_recovery_definition":"median paired(v11 - v12) / median paired(v11 - native), only when denominator is positive"}
    args.output.mkdir(parents=True,exist_ok=False)
    for filename,raw in artifacts.items():
        write(args.output/filename,raw)
    receipt_zip(args.output/"leased-receipts.zip",members)
    write(args.output/"paired-summary.json",nice(summary))
    write(args.output/"README_ZH.md",build_readme(stats,plan))
    files={p.relative_to(args.output).as_posix():digest(read(p)) for p in sorted(args.output.rglob("*")) if p.is_file()}
    manifest={"schema":1,"files_sha256":files,"leased_receipt_member_sha256":{name:digest(raw) for name,raw in members.items()}}
    write(args.output/"artifact_manifest.json",nice(manifest))
    verify(args.output)
    return summary


def verify(root,replay=False):
    manifest=json.loads(read(root/"artifact_manifest.json"))
    for filename,expected in manifest["files_sha256"].items():
        relative=PurePosixPath(filename)
        need(not relative.is_absolute() and ".." not in relative.parts and "\\" not in filename and ":" not in filename,"invalid artifact path")
        sealed(root/relative,expected)
    expected=manifest["leased_receipt_member_sha256"]
    with ZipFile(root/"leased-receipts.zip") as package:
        names=package.namelist()
        need(len(names)==len(set(names))==len(expected)==24 and set(names)==set(expected),"published receipt archive roster differs")
        for name,commitment in expected.items():
            need(RECEIPT.fullmatch(name) and digest(package.read(name))==commitment,"receipt archive member differs")
    if replay:
        with tempfile.TemporaryDirectory(prefix="topomem-v12-published-receipts-") as tmp:
            directory=Path(tmp)
            for name in ("jobs.jsonl","run-binding.json"):
                write(directory/name,read(root/"run"/name))
            with ZipFile(root/"leased-receipts.zip") as package:
                for name in expected:
                    write(directory/name,package.read(name))
            replay_analysis(root/"tools/qualify_v12_paired_gpu3.py",root/"inputs/preregistration.json",directory,
                            read(root/"run/analysis.json"))
    return {"files":len(manifest["files_sha256"]),"receipts":len(expected),"analysis_replayed":replay}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    phase=parser.add_subparsers(dest="phase",required=True)
    prepare=phase.add_parser("export")
    for name in ("run-directory","analysis","output"):
        prepare.add_argument("--"+name,required=True,type=Path)
    for name in ("expected-analysis-sha256","expected-ledger-sha256"):
        prepare.add_argument("--"+name,required=True)
    prepare.add_argument("--prototype-root",type=Path,default=ROOT)
    view=phase.add_parser("verify")
    view.add_argument("--publication",required=True,type=Path)
    view.add_argument("--replay-analysis",action="store_true")
    args=parser.parse_args()
    result=export(args) if args.phase=="export" else verify(args.publication,args.replay_analysis)
    print(json.dumps(result,sort_keys=True,ensure_ascii=False,allow_nan=False))


if __name__=="__main__":
    main()
