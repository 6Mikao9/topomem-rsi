import argparse
from collections import Counter, defaultdict
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tempfile
from zipfile import ZipFile

def digest(data):
    return sha256(data).hexdigest()

def fingerprint(value):
    return digest(json.dumps(value,sort_keys=True,separators=(",",":"),ensure_ascii=True,allow_nan=False).encode())

def nice(value):
    return (json.dumps(value,sort_keys=True,indent=2,ensure_ascii=False,allow_nan=False)+"\n").encode()

def lines(data):
    return [json.loads(line) for line in data.splitlines()]

def attest(base, source_index, index_bytes, label, destination, verify_ticks, ingress, verify_only):
    group = source_index["groups"][label]
    inventory, members, seen_jobs = {}, defaultdict(dict), set()
    for arc_row in group["archives"]:
        path = base / arc_row["name"]
        assert path.stat().st_size == arc_row["bytes"] and digest(path.read_bytes()) == arc_row["sha256"], path.name
        with ZipFile(path) as arc:
            names = arc.namelist()
            manifest_bytes = arc.read("EVIDENCE_MANIFEST.json")
            manifest = json.loads(manifest_bytes)
            assert len(set(names)) == len(names) and set(names) == {*manifest,"EVIDENCE_MANIFEST.json"}
            assert len(manifest)==arc_row["file_count"] and sum(row["bytes"] for row in manifest.values())==arc_row["uncompressed_bytes"]
            for member, row in manifest.items():
                p = PurePosixPath(member)
                assert not p.is_absolute() and ".." not in p.parts and "\\" not in member and ":" not in member and p.suffix in {".json",".jsonl",".log",".txt"}
                job, *parts = p.parts
                assert job in group["job_ids"] and member not in inventory
                data = arc.read(member)
                assert len(data) == row["bytes"] and digest(data) == row["sha256"], member
                inventory[member] = {**row,"archive":path.name}
                members[job]["/".join(parts)] = data
                seen_jobs.add(job)
    assert set(group["job_ids"]) == seen_jobs and len(group["job_ids"])==len(seen_jobs)
    with ZipFile(destination/"leased-receipts.zip") as arc:
        leased = {PurePosixPath(n).stem:arc.read(n) for n in arc.namelist()}
        assert len(leased)==len(arc.namelist())
    local = {job:json.loads(data) for job,data in leased.items()}
    assert set(local) == seen_jobs
    formal_index=destination/"server_job_index.json"
    if formal_index.is_file():
        authoritative=json.loads(formal_index.read_bytes())["formal_jobs"]
    else:
        authoritative=json.loads((destination/"audit.json").read_bytes())["leases"]
    assert set(authoritative)==seen_jobs
    for job,receipt in local.items():
        entry=authoritative[job]
        assert entry.get("receipt_sha256",entry.get("file_sha256"))==digest(leased[job])
        loose=destination/"receipts"/(job+".json")
        if loose.is_file():
            assert loose.read_bytes()==leased[job]
        assert entry.get("serving_result_sha256",entry.get("raw_result_sha256"))==receipt["result"]["raw_result_sha256"]
    detailed = {}
    for job, evidence in members.items():
        receipt = local[job]
        server_receipt = json.loads(evidence["receipt.json"])
        assert server_receipt == {key:receipt[key] for key in server_receipt} and server_receipt["status"] == "completed"
        assert not any(server_receipt[k] for k in ("remaining_compute_pids","process_group_remaining","environment_conflict","timeout"))
        request = json.loads(evidence["request.json"])
        assert fingerprint(request["request"]) == request["sha256"] == receipt["request_sha256"]
        assert json.loads(evidence["input.json"]) == request["request"]["payload"]
        assert digest(evidence["process.log"]) == receipt["log_sha256"]
        assert digest(evidence["result.json"]) == receipt["result_sha256"]
        summary = json.loads(evidence["result.json"])
        assert summary == receipt["result"] and summary["feasible"]
        assert digest(evidence["serving/result.json"]) == summary["raw_result_sha256"]
        result = json.loads(evidence["serving/result.json"])
        assert result["status"]=="completed" and result["phase"] == summary["phase"]
        schedule = json.loads(evidence["serving/external_schedule.json"])
        gate = schedule[ingress.EXTERNAL_FIELD]
        ticks = [json.loads(evidence[f"serving/global-ticks-rank-{rank}.json"]) for rank in range(2)]
        releases = [lines(evidence[f"serving/ingress-rank-{rank}.jsonl"]) for rank in range(2)]
        assert [row["rank"] for row in ticks] == [0,1] and all(row["pid"] == release[0]["pid"] for row,release in zip(ticks,releases,strict=True))
        actual = verify_ticks([row["ticks"] for row in ticks],gate)
        metrics = summary["metrics"]
        assert actual == result["global_ticks"] == metrics["global_ticks"]
        assert ingress.verify_ingress_events(releases,gate) == result["ingress"] == metrics["ingress"]
        tick_files = [digest(evidence[f"serving/global-ticks-rank-{r}.json"]) for r in range(2)]
        assert tick_files == result["global_tick_receipt_sha256_by_rank"] == metrics["global_tick_receipt_sha256_by_rank"]
        schedules = [json.loads(evidence[f"serving/schedule-rank-{r}.json"]) for r in range(2)]
        dispatch = [json.loads(evidence[f"serving/dispatch-rank-{r}.json"]) for r in range(2)]
        assert fingerprint([row["effective"] for row in schedules]) == result["effective_schedule_sha256"] == metrics["effective_schedule_sha256"]
        assert fingerprint([row["batches"] for row in dispatch]) == result["dispatch_sha256"] == metrics["dispatch_sha256"]
        observed = [(r["rid"],r["input_ids_sha256"],r["output_ids"]) for r in result["warmup"]+result["requests"]]
        assert fingerprint(observed) == result["output_ids_sha256"] == metrics["output_ids_sha256"]
        assert fingerprint(result["runtime"])==metrics["runtime_sha256"]
        parity_hashes=[]
        for rank in range(2):
            parity_file=f"serving/kv-rank-{rank}.jsonl"
            if summary["phase"] == "parity":
                blob=evidence[parity_file]
                assert digest(blob) == summary["feedback"]["kv_log_sha256_by_rank"][rank]
                comparable, counts = [], Counter()
                assigned = [r for r in result["warmup"]+result["requests"] if r["routed_dp_rank"] == rank]
                total = sum(len(row["output_ids"])-1 for row in assigned)
                for event in lines(blob):
                    assert event["event"] == "kv_selected_exact" and event["rank"] == rank and 0 <= event["layer"] < 6
                    counts[event["layer"]]+=len(event["rows"])
                    comparable.append({**event, "rows":[{k:v for k,v in row.items() if k != "miss_rows"} for row in event["rows"]]})
                assert counts == {layer:total for layer in range(6)}
                parity_hashes.append(fingerprint(comparable))
            else:
                assert parity_file not in evidence and dispatch[rank].get("resident_producer_oracle") is None
        assert parity_hashes == result["packed_kv_sha256_by_rank"] == metrics["packed_kv_sha256_by_rank"]
        for proof in (request["request"]["payload"][key] for key in ("qualification","cross_cap_reference")):
            if proof is not None:
                assert proof["job_id"] in local
                assert proof["raw_result_sha256"] == local[proof["job_id"]]["result"]["raw_result_sha256"]
        detailed[job]={"summary_sha256":receipt["result_sha256"],"serving_result_sha256":summary["raw_result_sha256"],
                       "phase":summary["phase"],"global_ticks_sha256":actual["global_ticks_sha256"],
                       "kv_packed_sha256_by_rank":parity_hashes}
    del members
    output = destination / "server-evidence"
    if not verify_only:
        output.mkdir(exist_ok=False)
        (output/"evidence-archive-index.json").write_bytes(index_bytes)
        for row in group["archives"]:
            copy = output / row["name"]
            shutil.copyfile(base/row["name"],copy)
            assert digest(copy.read_bytes())==row["sha256"]
    doc={"schema":1,"scope":source_index["scope"],"group":label,"original_combined_index_sha256":digest(index_bytes),
         "group_from_original_index":group,"members":inventory,"job_checks":detailed,
         "recomputed_global_tick_results":len(detailed),"recomputed_packed_kv_parity_jobs":sum(row["phase"]=="parity" for row in detailed.values())}
    audit=nice(doc)
    if verify_only:
        assert (output/"index.json").read_bytes()==audit
    else:
        (output/"index.json").write_bytes(audit)
    print(label,len(detailed),len(inventory),digest(audit),flush=True)

def main():
    parser=argparse.ArgumentParser(description="Verify byte-exact named SGLang server job archives.")
    parser.add_argument("--server-export",type=Path,required=True)
    parser.add_argument("--source-archive",type=Path,required=True)
    parser.add_argument("--destination",type=Path,required=True)
    parser.add_argument("--group",choices=("formal-v3","native6144-supplement-v1"),required=True)
    parser.add_argument("--verify-only",action="store_true")
    args=parser.parse_args()
    if not __debug__:
        raise RuntimeError("archive verification requires Python assertions enabled")
    base,source=args.server_export,args.source_archive
    assert digest(source.read_bytes())=="ab8479bbb8b017a50673c07485ddb0fd88252dc24ec9a7467e4b214f9dfdb7b9"
    index_bytes=(base/"evidence-archive-index.json").read_bytes()
    source_index=json.loads(index_bytes)
    assert source_index["schema"]==1
    with tempfile.TemporaryDirectory(prefix="topomem-raw-audit-") as temporary:
        p=Path(temporary)
        with ZipFile(source) as archive:
            inv=json.loads(archive.read("LOCAL_SOURCE_MANIFEST.json"))["files"]
            for name,sha in inv.items():
                path=PurePosixPath(name)
                assert not path.is_absolute() and ".." not in path.parts
                raw=archive.read(name)
                assert digest(raw)==sha
                out=p/path
                out.parent.mkdir(parents=True,exist_ok=True)
                out.write_bytes(raw)
        sys.path.insert(0,str(p))
        from integrations.topology_memory.sglang_global_ticks import verify_global_ticks
        from integrations.topology_memory import sglang_ingress_gate
        attest(base,source_index,index_bytes,args.group,args.destination,verify_global_ticks,sglang_ingress_gate,args.verify_only)
    print("source combined index SHA",digest(index_bytes),flush=True)

if __name__=="__main__":
    main()
