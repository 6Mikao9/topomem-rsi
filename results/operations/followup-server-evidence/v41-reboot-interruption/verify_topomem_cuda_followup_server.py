"""Verify and publish named CUDA server evidence against local leased receipts."""
from argparse import ArgumentParser
from collections import defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
import shutil
from zipfile import ZipFile

from list_topomem_cuda_followup_jobs import FINAL, FOLLOWUPS, JOB, completed_group

DESTINATIONS = {label: row[0] for label, row in {**FOLLOWUPS, "final-gpu3": FINAL}.items()}
REQUIRED = {"input.json","process.log","request.json","started.json","result.json","receipt.json"}
REBOOT_PARTIAL = "eval-960140543890d7588593c2bc332367896d8a1183b41a540e"


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(raw):
    return sha256(raw).hexdigest()


def canonical(value):
    return digest(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode())


def safe_relative(name):
    path=PurePosixPath(name)
    require(not path.is_absolute() and path.parts and not {"..","."}&set(path.parts)
            and "\\" not in name and ":" not in name and path.as_posix()==name,"invalid archive path")
    return path


def group_members(server,group,label,expected):
    archives,contents,manifests={},defaultdict(dict),{}
    jobs=group["job_ids"]
    require(len(jobs)==len(set(jobs)) and set(jobs)==set(expected),"unexpected job roster: "+label)
    for info in group["archives"]:
        name=info["name"]
        require(re.fullmatch(re.escape(label)+r"-server-[0-9]{2}\.zip",name) and name not in archives,
                "unexpected archive file: "+name)
        path=server/name
        data=path.read_bytes()
        require(len(data)==info["bytes"] and digest(data)==info["sha256"],"archive digest differs: "+name)
        archives[name]=info
        with ZipFile(path) as archive:
            names=archive.namelist()
            manifest=json.loads(archive.read("EVIDENCE_MANIFEST.json"))
            require(len(names)==len(set(names)) and set(names)=={"EVIDENCE_MANIFEST.json",*manifest}
                    and len(manifest)==info["file_count"]
                    and sum(x["bytes"] for x in manifest.values())==info["uncompressed_bytes"],
                    "archive member roster differs: "+name)
            for member,meta in manifest.items():
                rel=safe_relative(member)
                require(len(rel.parts)==2 and JOB.fullmatch(rel.parts[0]) and rel.parts[0] in jobs
                        and rel.parts[1] in REQUIRED and member not in manifests,"unexpected archive member: "+member)
                raw=archive.read(member)
                require(len(raw)==meta["bytes"] and digest(raw)==meta["sha256"],"member digest differs: "+member)
                contents[rel.parts[0]][rel.parts[1]]=raw
                manifests[member]={**meta,"archive":name}
    require(set(contents)==set(jobs),"missing server job evidence: "+label)
    return contents,manifests,archives


def local_leases(publication,expected):
    manifest=json.loads((publication/"artifact_manifest.json").read_bytes())
    members=manifest["leased_receipt_member_sha256"]
    values={}
    with ZipFile(publication/"leased-receipts.zip") as archive:
        names=archive.namelist()
        require(len(names)==len(set(names)) and set(names)==set(members),"local lease roster differs")
        for name,commitment in members.items():
            job=safe_relative(name).stem
            raw=archive.read(name)
            require(JOB.fullmatch(job) and job not in values and digest(raw)==commitment,"invalid local receipt: "+name)
            values[job]=json.loads(raw)
    require(set(values)==set(expected),"local and server job rosters differ")
    return values


def request_evidence(job,files,local):
    request=json.loads(files["request.json"])
    request_data=request["request"]
    started=json.loads(files["started.json"])
    require(canonical(request_data)==request["sha256"]==local["request_sha256"]==started["request_sha256"]
            and request_data["job_id"]==started["job_id"]==job
            and json.loads(files["input.json"])==request_data["payload"]
            and started["source_manifest_sha256"]==local["source_manifest_sha256"]==request_data["source_manifest_sha256"]
            and (started["pid"],started["proc_starttime"])==(local["pid"],local["proc_starttime"]),
            "server request or started record differs: "+job)
    require(digest(files["process.log"])==local["log_sha256"],"server process log differs: "+job)
    return {"request_sha256":request["sha256"],"process_log_sha256":local["log_sha256"]}


def job_evidence(job,files,local,*,allow_known_reboot_loss=False):
    require(set(files)==REQUIRED,"missing server files for "+job)
    require(local["job_id"]==job and local["status"]=="completed","local lease differs: "+job)
    checks=request_evidence(job,files,local)
    if allow_known_reboot_loss and job==REBOOT_PARTIAL and files["receipt.json"]==files["result.json"]==b"":
        return {**checks,"status":"partial_server_evidence","verified_files":["input.json","process.log","request.json","started.json"],
                "empty_server_files":["receipt.json","result.json"],
                "expected_result_sha256_from_local_receipt":local["result_sha256"],
                "local_feasible":local["result"]["feasible"]}
    receipt=json.loads(files["receipt.json"])
    require(receipt=={k:v for k,v in local.items() if k!="client_elapsed_seconds"},
            "server and local receipt differ: "+job)
    result=json.loads(files["result.json"])
    require(result==receipt["result"] and digest(files["result.json"])==receipt["result_sha256"]
            and canonical(result)==receipt["result_canonical_sha256"],"server result differs: "+job)
    return {**checks,"status":"verified","result_sha256":receipt["result_sha256"],"feasible":result["feasible"]}


def main():
    parser=ArgumentParser(description=__doc__)
    parser.add_argument("--server-export",type=Path,required=True)
    parser.add_argument("--results",type=Path,required=True)
    parser.add_argument("--group",choices=tuple(DESTINATIONS),required=True)
    parser.add_argument("--publish",action="store_true")
    args=parser.parse_args()
    index_raw=(args.server_export/"evidence-archive-index.json").read_bytes()
    label=args.group
    index=json.loads(index_raw)
    require(index["schema"]==1 and label in index["groups"],"server index differs")
    group=index["groups"][label]
    destination=args.results/DESTINATIONS[label]
    expected=completed_group(args.results, *(FINAL if label=="final-gpu3" else FOLLOWUPS[label]))
    files,members,archives=group_members(args.server_export,group,label,expected)
    local=local_leases(destination,expected)
    audited={job:job_evidence(job,files[job],local[job],allow_known_reboot_loss=label=="v41-reboot-interruption")
             for job in expected}
    partial=[job for job,row in audited.items() if row["status"]!="verified"]
    original_manifest=digest((destination/"artifact_manifest.json").read_bytes())
    report={"schema":1,"group":label,"original_combined_index_sha256":digest(index_raw),
            "original_result_directory":DESTINATIONS[label],"original_artifact_manifest_sha256":original_manifest,
            "group_from_original_index":group,"members":members,"job_checks":audited,
            "fully_verified_jobs":len(audited)-len(partial),"partial_server_evidence_jobs":partial}
    if args.publish:
        out=args.results/"operations"/"followup-server-evidence"/label
        out.mkdir(parents=True,exist_ok=False)
        for name in archives:
            shutil.copyfile(args.server_export/name,out/name)
            require(digest((out/name).read_bytes())==archives[name]["sha256"],"copied archive digest differs")
        (out/"evidence-archive-index.json").write_bytes(index_raw)
        (out/"index.json").write_bytes((json.dumps(report,sort_keys=True,indent=2)+"\n").encode())
        for name in (Path(__file__).name,"list_topomem_cuda_followup_jobs.py"):
            shutil.copyfile(Path(__file__).with_name(name),out/name)
        (out/"README.md").write_text(
            f"# CUDA server evidence for {label}\n\n"
            f"The server archive roster covers all {len(audited)} completed local leases in "
            f"[the original result directory](../../../{DESTINATIONS[label]}/). "
            f"Its artifact manifest SHA-256 is `{original_manifest}`. "
            "[The audit](index.json) records per-member hashes and request, result and process log checks.\n\n"
            f"Fully matched server records: {len(audited)-len(partial)}.\n"
            + ("\nThe reboot export contains one completed local lease with empty server result and receipt files. "
               "The original files are preserved. Its server input, request, started record and process log match the local lease.\n" if partial else "")
            + "\nRecheck from this directory.\n\n"
              "```sh\npython -B verify_topomem_cuda_followup_server.py --server-export . --results ../../.. --group "
              f"{label}\n```\n",
            encoding="utf-8",newline="\n",
        )
    print(json.dumps({"group":label,"jobs":len(audited),"fully_verified_jobs":len(audited)-len(partial),
                      "partial_server_evidence_jobs":partial,"members":len(members),"archives":len(archives)},sort_keys=True))


if __name__=="__main__":
    main()
