"""Publish the byte-preserved formal four-case SGLang replay inputs after the run settles."""
import argparse
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import sys
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

FIXED = {
    "original_case_archive_sha256":"921f665566bd6b3e1223954b1a0b5c2ef43711b47ef46dbab836c8f5ca6a4e65",
    "original_case_transfer_manifest_sha256":"e0c710bf6e4e54a337706cea4caf15713b06dfd81931c24ce550989cdc6e1ee5",
    "formal_case_set_manifest_sha256":"f7fe4ad994bd82f70ead6a631960a877b62d836d87697342b79830b15698b6c7",
    "parent_case_set_manifest_sha256":"01e136f07395f4a0f5dc20b594a4336bf889cebd613e470fe076db0181aba462",
    "formal_tokenization_receipt_sha256":"c9a50f89146421ef0a039dab5d0974e8a1ea3f7a0c28ebbda60f3ee0e6f19bd5",
    "frozen_controller_archive_sha256":"ab8479bbb8b017a50673c07485ddb0fd88252dc24ec9a7467e4b214f9dfdb7b9",
    "frozen_controller_source_manifest_sha256":"f452f06968cb2b1df687eba72e6dabcd3e5e419fe15bd205b555dce1f0485c9b",
}
CASE_IDS = {"train":["archive-train-reproduction"],"validation":["archive-validation-support"],
            "test":["archive-test-delayed-review","archive-test-new-observation"]}
ROOTS = ("case_set_manifest.json","formal_case_set_manifest.json","formal_tokenization_receipt.json")
CASE_FILES = {name+".json" for name in ("external_schedule","fixture","manifest","manual_plan_preview","manual_program",
                                      "native_plan_preview","scenario","target","topology")}
SHA = re.compile(r"[0-9a-f]{64}")
SECRET = re.compile(rb"sk-[A-Za-z0-9_-]{24,}|-----BEGIN [A-Z ]*PRIVATE KEY-----|authorization:[ ]*bearer[ ]+[A-Za-z0-9_.-]{16,}",re.I)
PROMPT_PRIVATE = re.compile(r"[A-Z]:[\\/](?:Users|ws)[\\/]|/home/[\w.-]+/|\b[\w.%+-]+@[\w.-]+\.\w+\b",re.I)


def need(predicate,message):
    if not predicate:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(value):
    return digest(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode())


def json_bytes(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()


def read(path):
    need(path.is_file() and not path.is_symlink(),"missing or symlinked " + path.name)
    return path.read_bytes()


def sealed(path,expected):
    need(isinstance(expected,str) and SHA.fullmatch(expected),"missing independent seal for " + path.name)
    raw=read(path)
    need(digest(raw)==expected,"SHA-256 differs for " + path.name)
    return raw


def safe_name(name):
    p=PurePosixPath(name)
    need(p.parts and not p.is_absolute() and ".." not in p.parts and "\\" not in name and ":" not in name,
         "unsafe archive member path")
    return p


def terminal(root,report_sha,pipeline_sha):
    """Authorize the public historical input release before touching any original case body."""
    report_raw=sealed(root/"controller/report.json",report_sha)
    pipeline_raw=sealed(root/"pipeline-status.json",pipeline_sha)
    frozen_raw=read(root/"controller/frozen.json")
    archive_manifest=json.loads(read(root/"artifact_manifest.json"))
    files=archive_manifest["files_sha256"]
    need(all(files.get(name)==digest(raw) for name,raw in (("controller/report.json",report_raw),
        ("pipeline-status.json",pipeline_raw),("controller/frozen.json",frozen_raw))),"formal publication artifact manifest differs")
    report,pipeline,frozen=map(json.loads,(report_raw,pipeline_raw,frozen_raw))
    need([row["phase"] for row in pipeline]==["run","freeze","test"]
         and all(type(row["returncode"]) is int and row["returncode"]==0 for row in pipeline),
         "formal run, freeze, and test must all exit successfully")
    manifest=report["manifest"]
    pair=report["paired_evaluations"]["pair/frozen/test"]
    need(report["status"]=="test_settled" and report["frozen"]==frozen and frozen["partial"] is False
         and frozen["rounds"]==manifest["config"]["rounds"]==2
         and canonical(manifest)==frozen["manifest_sha256"] and canonical(report["rounds"])==frozen["rounds_sha256"]
         and pair["input"]["split"]=="test" and pair["status"] in {"completed","failed","interrupted_unknown"},
         "formal report is not a settled, frozen two-round run")
    contract=manifest["contract"]
    protocol=contract["generator_program_protocol"]
    need(protocol["kind"]=="backend_program_contract_v3","formal run was not the pinned v3 experiment")
    return manifest,files


def extract_source(path,dest,fixed,manifest,formal_files):
    raw=sealed(path,fixed["frozen_controller_archive_sha256"])
    need(formal_files["frozen-source.zip"]==digest(raw),"formal publication points to a different source archive")
    with ZipFile(BytesIO(raw)) as package:
        source_raw=package.read("LOCAL_SOURCE_MANIFEST.json")
        need(digest(source_raw)==fixed["frozen_controller_source_manifest_sha256"],"controller source manifest changed")
        source=json.loads(source_raw)
        roster=source["files"]
        contract=manifest["contract"]
        protocol=contract["generator_program_protocol"]
        names=package.namelist()
        need(len(names)==len(set(names)) and set(names)=={"LOCAL_SOURCE_MANIFEST.json",*roster}
             and source["worker_source_manifest_sha256"]==contract["source_manifest_sha256"],"source archive roster or worker source differs")
        groups=({f"rsi_memory/{n}":s for n,s in manifest["source_sha256"].items()},
                contract["adapter_sources_sha256"],protocol["source_sha256"])
        need(all(roster.get(n)==h for group in groups for n,h in group.items()),"formal controller source bindings differ")
        for name,expected in roster.items():
            path=dest/safe_name(name)
            member=package.read(name)
            need(digest(member)==expected,"frozen source bytes differ " + name)
            path.parent.mkdir(parents=True,exist_ok=True)
            path.write_bytes(member)
    return len(roster)


def original_cases(path,stage,fixed,contract):
    raw=sealed(path,fixed["original_case_archive_sha256"])
    original={}
    with ZipFile(BytesIO(raw)) as package:
        transfer_raw=package.read("CASE_TRANSFER_MANIFEST.json")
        need(digest(transfer_raw)==fixed["original_case_transfer_manifest_sha256"],"six-case transfer manifest differs")
        transfer=json.loads(transfer_raw)
        roster=transfer["files"]
        names=package.namelist()
        need(len(names)==len(set(names)) and set(names)=={"CASE_TRANSFER_MANIFEST.json",*roster}
             and transfer["worker_source_manifest_sha256"]==contract["source_manifest_sha256"],"original transfer inventory or worker source differs")
        need(len({safe_name(name).parts[0] for name in roster if len(safe_name(name).parts)>1})==6,
             "original transfer must describe six case directories")

        def take(name):
            safe_name(name)
            data=package.read(name)
            need(digest(data)==roster[name] and not SECRET.search(data),"case member hash differs or contains credential-like bytes: " + name)
            original[name]=data
            return json.loads(data) if name.endswith(".json") else None

        parent,formal,tokens=(take(name) for name in ROOTS)
        need(digest(original[ROOTS[0]])==fixed["parent_case_set_manifest_sha256"]==formal["parent_case_set_manifest_sha256"]
             and digest(original[ROOTS[1]])==fixed["formal_case_set_manifest_sha256"]==transfer["formal_case_set_sha256"]
             and digest(original[ROOTS[2]])==fixed["formal_tokenization_receipt_sha256"]==transfer["formal_tokenization_receipt_sha256"]
             and tokens["case_set_manifest_sha256"]==fixed["formal_case_set_manifest_sha256"]
             and tokens["status"]=="tokenized" and canonical(tokens["tokens_by_case"])==contract["tokenization_sha256"]
             and formal["cases"]==contract["cases"] and len(parent["cases"]["train"])+len(parent["cases"]["validation"])+len(parent["cases"]["test"])==6,
             "formal case/token commitments differ")
        ids={split:[row["id"] for row in entries] for split,entries in formal["cases"].items()}
        need(ids==CASE_IDS and {case for values in ids.values() for case in values}==set(tokens["tokens_by_case"]),
             "formal four-case roster differs")
        for entries in formal["cases"].values():
            for entry in entries:
                case=entry["id"]
                case_manifest=take(case+"/manifest.json")
                need(digest(original[case+"/manifest.json"])==entry["manifest_sha256"],"case manifest differs " + case)
                evidence=set(case_manifest["link_evidence"].values())
                need(all(safe_name(n).parts[0]=="link_evidence" and len(safe_name(n).parts)==2 for n in evidence),"invalid link evidence path")
                files={case+"/"+name for name in CASE_FILES|evidence}
                need(files=={name for name in roster if name.startswith(case+"/")},"case subtree is not the allowed frozen roster")
                for name in sorted(files-{case+"/manifest.json"}):
                    data=take(name)
                    if name==case+"/fixture.json":
                        prompts=[row["prompt"] for key in ("warmup","requests") for row in data[key]]
                        need(len(prompts)==4 and not any(PROMPT_PRIVATE.search(text) for text in prompts),
                             "authored fixture needs private-locator review: " + case)
    for name,data in original.items():
        target=stage/safe_name(name)
        target.parent.mkdir(parents=True,exist_ok=True)
        target.write_bytes(data)
    return original,transfer_raw,formal,tokens


def verify_cases(source,stage,formal,tokens,tokenizer_model=None):
    """Invoke the exact historical loader; retokenization is optional and local-only."""
    sys.dont_write_bytecode=True
    sys.path.insert(0,str(source))
    try:
        from topomem_sglang_rsi.contract import load_case,load_external
        from topomem_sglang_rsi.rebind_cases import formal_subset
        need(formal_subset(stage/"formal_case_set_manifest.json")==formal,"frozen formal subset verification failed")
        tokenizer=None
        if tokenizer_model is not None:
            need(tokenizer_model.is_dir(),"local tokenizer model directory required")
            from transformers import AutoTokenizer
            tokenizer=AutoTokenizer.from_pretrained(str(tokenizer_model),local_files_only=True)
        for split,entries in formal["cases"].items():
            for entry in entries:
                case=entry["id"]
                path=stage/case
                load_case(path,entry,fixture=True)
                expected=tokens["tokens_by_case"][case]
                load_external(path,entry,expected_inputs=expected,require_gate=True)
                if tokenizer is not None:
                    fixture=json.loads(read(path/"fixture.json"))
                    actual={}
                    for group in ("warmup","requests"):
                        for row in fixture[group]:
                            ids=tokenizer.encode(row["prompt"],add_special_tokens=False)
                            actual[row["rid"]]={"input_tokens":len(ids),"input_ids_sha256":canonical(ids)}
                    need(actual==expected,"local tokenizer did not reproduce the frozen commitments for " + case)
    finally:
        sys.path.pop(0)
    return {"frozen_case_and_schedule_loaders_verified":True,"retokenized_locally":tokenizer is not None}


def write_zip(path,entries):
    with ZipFile(path,"x",compression=ZIP_DEFLATED,compresslevel=9) as package:
        for name,data in sorted(entries.items()):
            info=ZipInfo(name,date_time=(2020,1,1,0,0,0))
            info.external_attr=0o644 << 16
            package.writestr(info,data,compress_type=ZIP_DEFLATED,compresslevel=9)
    with ZipFile(path) as package:
        need(set(package.namelist())==set(entries) and all(package.read(n)==b for n,b in entries.items()),"four-case archive round trip differs")


def export(args,fixed=FIXED,verify=verify_cases):
    need(not args.output.exists() and not args.output.is_symlink(),"new output directory required")
    manifest,files=terminal(args.formal_export,args.expected_formal_report_sha256,args.expected_pipeline_status_sha256)
    with tempfile.TemporaryDirectory(prefix="topomem-case-publication-") as tmp:
        source,stage=Path(tmp)/"source",Path(tmp)/"cases"
        count=extract_source(args.formal_export/"frozen-source.zip",source,fixed,manifest,files)
        original,upstream,formal,tokens=original_cases(args.original_case_archive,stage,fixed,manifest["contract"])
        verification=verify(source,stage,formal,tokens,args.local_tokenizer_model)
    args.output.mkdir(parents=True,exist_ok=False)
    for name,data in original.items():
        path=args.output/"cases"/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(data)
    (args.output/"provenance").mkdir()
    (args.output/"provenance/CASE_TRANSFER_MANIFEST.json").write_bytes(upstream)
    bundle=args.output/"formal-four-cases.zip"
    write_zip(bundle,original)
    overlay={"schema":1,"scope":"byte-preserved four-case subset of the six-case historical transfer",
        "formal_report_sha256":args.expected_formal_report_sha256,"formal_pipeline_status_sha256":args.expected_pipeline_status_sha256,
        "formal_controller_manifest_canonical_sha256":canonical(manifest),
        "formal_adapter_contract_canonical_sha256":canonical(manifest["contract"]),
        "input_tokenization_canonical_sha256":manifest["contract"]["tokenization_sha256"],
        "worker_source_manifest_sha256":manifest["contract"]["source_manifest_sha256"],
        **fixed,"new_public_four_case_zip_sha256":digest(read(bundle)),"published_zip_member_sha256":{n:digest(b) for n,b in original.items()},
        "subset_indices":{"train":[0],"validation":[0],"test":[0,1]},
        "original_transfer_manifest_role":"upstream six-case provenance; not an inventory of this four-case archive",
        "historical_hardware_binding":"Opaque GPU UUID/ordinal/name/capability and link observations preserved as evidence. Site connection and launch are separate.",
        "tokenization_commitment_kind":"one input-token count and canonical token-ID digest per warmup or measured request",
        "formal_case_ids":CASE_IDS,"frozen_source_file_count":count,"verification":verification}
    (args.output/"PUBLIC_CASE_OVERLAY.json").write_bytes(json_bytes(overlay))
    docs="""# SGLang 正式四个 replay case

归档保留一训练、一验证、两测试 case 的原始请求、固定调度、拓扑、目标和链路证据。每案含两条预热请求和两条测量请求。正式分词回执保存输入长度及 token ID 摘要。

解压四案 ZIP，或将本目录的 cases 作为本地 case 根。部署方在远端实验根使用同样的 cases 子树。冻结适配器用 formal_case_set_manifest.json 与同目录的 formal_tokenization_receipt.json。父 case_set_manifest.json 保留六案的描述符，供冻结代码复核预登记索引。发布 ZIP 的成员 SHA 见公开 overlay。provenance 中的原传输清单覆盖六案，用于核对来源。

历史目标保留原 GPU UUID、序号、型号、能力与链路证据。原运行的私有 schema、启动配置和连接信息由操作方分别持有。迁移到其他硬件或模型时，保持请求与调度语义，重新绑定目标、分词、链路和资格，生成新的摘要。逐案重新分词可用同一历史模型的本地 tokenizer 核对，具体重分词状态见 overlay。

[四案 ZIP](formal-four-cases.zip) · [公开绑定](PUBLIC_CASE_OVERLAY.json) · [原六案传输清单](provenance/CASE_TRANSFER_MANIFEST.json) · [正式分词回执](cases/formal_tokenization_receipt.json)
"""
    (args.output/"README_ZH.md").write_bytes(docs.encode())
    inventory={p.relative_to(args.output).as_posix():digest(read(p)) for p in sorted(args.output.rglob("*")) if p.is_file()}
    (args.output/"artifact_manifest.json").write_bytes(json_bytes({"schema":1,"files_sha256":inventory}))
    return overlay


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ("formal-export","original-case-archive","output"):
        parser.add_argument("--"+name,required=True,type=Path)
    parser.add_argument("--expected-formal-report-sha256",required=True)
    parser.add_argument("--expected-pipeline-status-sha256",required=True)
    parser.add_argument("--local-tokenizer-model",type=Path)
    args=parser.parse_args()
    overlay=export(args)
    print(json.dumps({"output":str(args.output),"member_count":len(overlay["published_zip_member_sha256"]),
                      "four_case_zip_sha256":overlay["new_public_four_case_zip_sha256"]},sort_keys=True))


if __name__=="__main__":
    main()
