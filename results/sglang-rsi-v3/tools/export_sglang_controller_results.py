"""Export a settled controller or an explicitly recorded abort; never launch a job."""
import argparse
from collections import Counter
from contextlib import closing
from hashlib import sha256
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import sqlite3
from statistics import median
import tempfile
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo

JOB = re.compile(r"sg-[0-9a-f]{48}")
SHA = re.compile(r"[0-9a-f]{64}")
LOGICAL_EVIDENCE = {
    "job_required_when_available": ["request.json", "receipt.json", "result.json", "process.log"],
    "job_optional": ["started.json", "qualification.json", "cross_cap_reference.json"],
    "compiled_case": [name + ".json" for name in ("memory_compilation", "cold_plan", "full_program",
        "cold_program", "resident_proof", "external_schedule", "target", "topology", "scenario")],
    "serving": ["result.json", "compiled_plan.json", "worker_input.json", "external_schedule.json"]
        + [f"{name}-rank-{rank}.json" for rank in range(2) for name in ("schedule", "dispatch", "global-ticks")]
        + [f"{name}-rank-{rank}.jsonl" for rank in range(2) for name in ("kv", "ingress", "backing", "backup")],
}

def check(predicate, reason):
    if not predicate:
        raise ValueError(reason)

def digest(raw):
    return sha256(raw).hexdigest()

def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode())

def nice(value):
    return (json.dumps(value, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False) + "\n").encode()

def raw(path):
    check(path.is_file() and not path.is_symlink(), "missing or symlinked file " + path.name)
    return path.read_bytes()

def records(controller):
    database = controller / "journal.sqlite"
    original = raw(database)
    for suffix in ("-journal", "-wal", "-shm"):
        check(not database.with_name(database.name + suffix).exists(), "journal has unsettled sidecar " + suffix)
    with tempfile.TemporaryDirectory(prefix="topomem-readonly-") as temporary:
        snapshot = Path(temporary) / database.name
        snapshot.write_bytes(original)
        with closing(sqlite3.connect(snapshot.as_uri() + "?immutable=1&mode=ro", uri=True)) as connection:
            check(connection.execute("PRAGMA integrity_check").fetchall() == [("ok",)], "SQLite integrity check failed")
            rows = connection.execute("SELECT key,data FROM records ORDER BY rowid").fetchall()
    return original, {key: text.encode() for key,text in rows}, {key:json.loads(text) for key,text in rows}

def check_source(source_bytes, controller_manifest):
    with ZipFile(BytesIO(source_bytes)) as archive:
        manifest_raw = archive.read("LOCAL_SOURCE_MANIFEST.json")
        manifest = json.loads(manifest_raw)
        inventory = manifest["files"]
        check(len(set(archive.namelist())) == len(archive.namelist()), "duplicate archived source name")
        check(set(archive.namelist()) == {"LOCAL_SOURCE_MANIFEST.json", *inventory}, "archived source roster mismatch")
        for name, expected in inventory.items():
            p = PurePosixPath(name)
            check(not p.is_absolute() and ".." not in p.parts and digest(archive.read(name)) == expected,
                  "frozen source hash or path mismatch " + name)
    binding = {f"rsi_memory/{k}":v for k,v in controller_manifest["source_sha256"].items()}
    contract = controller_manifest["contract"]
    protocol = contract.get("generator_program_protocol", {})
    for group in (contract["adapter_sources_sha256"], protocol.get("source_sha256", {})):
        check(all(binding.get(name, value) == value for name, value in group.items()), "controller source bindings disagree")
        binding.update(group)
    check(all(inventory.get(name) == value for name,value in binding.items()), "source archive differs from the run")
    check(manifest["worker_source_manifest_sha256"] == controller_manifest["contract"]["source_manifest_sha256"],
          "source archive binds a different worker manifest")
    return manifest_raw, len(inventory), len(binding)

def receipt_ids(value):
    if isinstance(value, dict):
        for key, node in value.items():
            if key == "receipts":
                check(isinstance(node,list) and all(isinstance(j,str) and JOB.fullmatch(j) for j in node),
                      "controller mentions an invalid leased receipt ID")
                yield from node
            else:
                yield from receipt_ids(node)
    elif isinstance(value,list):
        for node in value:
            yield from receipt_ids(node)

def receipt_index(directory, manifest):
    check(directory.is_dir() and not directory.is_symlink(), "invalid receipts directory")
    contract = manifest["contract"]
    cases = {case["id"]:(split,case) for split, specs in contract["cases"].items() for case in specs}
    original, index = {}, {}
    for path in sorted(directory.glob("sg-*.json")):
        job = path.stem
        check(JOB.fullmatch(job), "unexpected leased file name")
        data = raw(path)
        receipt = json.loads(data)
        check(receipt["job_id"] == job and receipt["source_manifest_sha256"] == contract["source_manifest_sha256"],
              "leased job ID/source mismatch " + job)
        check(SHA.fullmatch(receipt["request_sha256"]), "invalid leased request SHA " + job)
        result = receipt.get("result")
        row = {"receipt_sha256":digest(data),"worker_request_sha256":receipt["request_sha256"],
               "status":receipt["status"],"worker_log_sha256":receipt.get("log_sha256"),"utc_start":receipt.get("utc_start"),
               "utc_end":receipt.get("utc_end"),"remaining_compute_pids":receipt.get("remaining_compute_pids")}
        if result is not None:
            check(fingerprint(result) == receipt["result_canonical_sha256"], "leased summary canonical hash differs " + job)
            exact = (json.dumps(result,sort_keys=True,indent=2,allow_nan=False)+"\n").encode()
            check(digest(exact) == receipt["result_sha256"], "leased summary file hash differs " + job)
            split, spec = cases[result["case"]["id"]]
            check(result["case"] == spec and result["raw_result_relative"] == "serving/result.json"
                  and SHA.fullmatch(result["raw_result_sha256"]), "leased serving identity differs " + job)
            row.update(split=split,case=spec["id"],phase=result["phase"],arm=result["arm"],score=result["score"],
                       feasible=result["feasible"],evaluator_summary_sha256=receipt["result_sha256"],
                       serving_result_sha256=result["raw_result_sha256"])
        original[f"receipts/{path.name}"] = data
        index[job] = row
    return original, index

def verify_api(rows, manifest):
    contract = manifest["contract"]
    declared_training = {case["id"] for case in contract["cases"]["train"]}
    excluded = {case[key] for split in ("validation", "test") for case in contract["cases"][split]
                for key in ("id", "workload_sha256", "schedule_sha256", "scenario_sha256")}
    calls = {}
    for key, row in rows.items():
        if not key.startswith("api/"):
            continue
        inp = row["input"]
        payload = inp["payload"]
        text = json.dumps(payload, ensure_ascii=False)
        check(not any(value in text for value in excluded), "non-training case commitment in provider payload " + key)
        check(not any(name in payload for name in ("api_key", "authorization", "headers")), "unexpected provider credentials in record")
        envelope = json.loads(payload["messages"][-1]["content"])
        trusted = envelope["trusted_domain"]
        protocol = contract.get("generator_program_protocol")
        if protocol:
            check(envelope["trusted_program_contract"] == protocol["trusted_program_contract"],
                  "actual provider envelope differs from the manifested program grammar " + key)
        check(fingerprint(trusted) == manifest["training_context_sha256"]
              and {case["case"] for case in trusted["cases"]} == declared_training, "provider domain not bound to training only")
        result = row.get("result", {})
        check(set(result) <= {"usage", "content", "finish_reason"}, "unexpected provider response fields " + key)
        calls[key] = {"status":row["status"],"harness_sha256":inp["harness_sha256"],
                      "seconds":row.get("seconds"),"usage":result.get("usage", {})}
    return calls

def public_config(config, manifest):
    contract = manifest["contract"]
    check(config["schema"] == 3 and all(config[k] == contract[k] for k in (
        "cases", "experiment_id", "paired_block_ids", "source_manifest_sha256", "launch_sha256", "controller_preregistration")),
        "private adapter config differs from controller manifest")
    check(fingerprint(config["token_inputs"]) == contract["tokenization_sha256"]
          and fingerprint(config["hot_profile"]) == contract["metadata_profile_sha256"]
          and config["manual_program"] == manifest["manual_program"] and config["initial_program"] == manifest["initial_program"],
          "private config differs from frozen programs, profile, or token commitments")
    proof = config["manual_selection"]
    check(all(proof[k] == contract["manual_selection"][k] for k in ("source_config_sha256", "artifact_sha256", "receipts_sha256")),
          "hand-sweep provenance differs")
    executor = {key: "REPLACE_WITH_OPERATOR_" + key.upper() for key in config["executor"]}
    executor.update(gpu_indices=config["executor"]["gpu_indices"],
                    gpu_uuids=["REPLACE_WITH_TARGET_GPU_UUID_0", "REPLACE_WITH_TARGET_GPU_UUID_1"])
    if "environment" in executor:
        executor["environment"] = {}
    template = {**config, "executor":executor, "receipt_directory":"REPLACE_WITH_NEW_RECEIPT_DIRECTORY",
                "local_cases":"REPLACE_WITH_EXTRACTED_CASE_ROOT",
                "manual_selection":{**proof, "sweep_dir":"REPLACE_WITH_ORIGINAL_HAND_SWEEP_DIRECTORY",
                    "source_config":"REPLACE_WITH_ORIGINAL_PRIVATE_SCHEMA2_PATH"}}
    sensitive = {config["executor"]["host"],config["executor"]["worker"],config["executor"]["python"],config["executor"]["root"],
                 config["local_cases"],proof["sweep_dir"],proof["source_config"],*config["executor"]["gpu_uuids"]}
    def strings(value):
        if isinstance(value, dict):
            for item in value.values():
                yield from strings(item)
        elif isinstance(value, list):
            for item in value:
                yield from strings(item)
        elif isinstance(value, str):
            yield value
    for payload in (template, contract):
        check(not any(secret in value for value in strings(payload) for secret in sensitive if len(secret)>4),
              "private deployment binding in public contract")
    return template

def write_zip(path, entries):
    with ZipFile(path, "x", compression=ZIP_DEFLATED, compresslevel=9) as archive:
        for name, value in sorted(entries.items()):
            entry = ZipInfo(name, date_time=(2020,1,1,0,0,0))
            entry.external_attr = 0o644 << 16
            archive.writestr(entry, value, compress_type=ZIP_DEFLATED, compresslevel=9)
    with ZipFile(path) as archive:
        check(set(archive.namelist()) == set(entries) and all(archive.read(k)==v for k,v in entries.items()),
              "ZIP round trip differs")

def read_named(controller, original, paths):
    for name in paths:
        path = controller / name
        if path.exists():
            original[name] = raw(path)

def check_abort(rows, marker, index):
    effects = [row for key,row in rows.items() if key.startswith(("eval/","pair/"))]
    check(not any(row["input"]["split"] == "test" for row in effects), "this abort path is limited to pre-heldout runs")
    check(not any(item.get("split") == "test" for item in index.values()), "abort receipts include heldout")
    check(all(rows[key]["status"] == status for key,status in marker["record_statuses"].items()), "abort status inventory differs")
    if "failed_evaluation" in marker:
        check(marker["failed_evaluation"] in effects, "saved abort failure differs from original journal")

def check_settled(rows, original, index):
    report = json.loads(original["controller/report.json"])
    check(report["status"] == "test_settled", "controller report is not terminal test_settled")
    for key, field in (("api/","calls"),("eval/","evaluations"),("plan/","meta_plans"),("pair/","paired_evaluations")):
        check({k:v for k,v in rows.items() if k.startswith(key)} == report[field], "report/journal disagree for " + field)
    frozen = rows["frozen"]
    rounds = [v for k,v in sorted(rows.items()) if k.startswith("round/")]
    check(frozen == report["frozen"] == json.loads(original["controller/frozen.json"])
          and rounds == report["rounds"] and report["manifest"] == rows["manifest"]
          and fingerprint(rows["manifest"]) == frozen["manifest_sha256"]
          and fingerprint(rounds) == frozen["rounds_sha256"], "report/journal freeze or manifest mismatch")
    check(len(rounds)==frozen["rounds"]==rows["manifest"]["config"]["rounds"] and not frozen["partial"],
          "configured rounds were incomplete")
    order = list(rows)
    heldout = rows["pair/frozen/test"]
    check(heldout["status"] in ("completed","failed","interrupted_unknown") and heldout["input"]["split"]=="test"
          and order.index("frozen") < order.index("pair/frozen/test")
          and all(order.index(k)<order.index("frozen") for k in rows if k.startswith("api/"))
          and rows["test_comparisons"] == report["test_comparisons"], "test order, terminal status, or comparisons disagree")
    check(set(k for k,v in rows.items() if k.startswith(("eval/","pair/")) and v["input"]["split"] == "test")=={"pair/frozen/test"},
          "more than one heldout controller effect")
    programs = heldout["input"]["programs"]
    contract = rows["manifest"]["contract"]
    block_ids = contract["paired_block_ids"]["test"]
    check(heldout["input"]["block_ids"] == block_ids, "heldout changed the predeclared block IDs")
    check(set(programs)=={"adaptive","fixed","manual","native"} and programs["native"] is None
          and programs["manual"]==rows["manifest"]["manual_program"]
          and all(programs[arm]==frozen[arm]["trial"]["program"] for arm in ("adaptive","fixed")), "test programs differ from frozen arms")
    compared = heldout.get("result", {})
    minimum = rows["manifest"]["config"]["min_improvement"]
    expected_cases = [spec["id"] for spec in contract["cases"]["test"]]
    for arm, entry in compared.items():
        if entry["feasible"]:
            samples = entry["samples"]
            check([sample["block_id"] for sample in samples] == block_ids, "heldout changed sample order " + arm)
            for sample in samples:
                scores = sample["case_scores_ms"]
                check(set(scores) == set(expected_cases) and sum(scores[case] for case in expected_cases) == sample["score"],
                      "heldout case-score sum differs " + arm)
            check(entry["score"] == median(sample["score"] for sample in samples), "heldout median score differs " + arm)
    comparisons = {}
    for reference in ("native","fixed","manual"):
        adaptive, baseline = compared.get("adaptive"), compared.get(reference)
        if not adaptive or not baseline or not adaptive["feasible"] or not baseline["feasible"]:
            comparison = {"passed":False,"reason":"missing_feasible_pair","deltas":[]}
        else:
            deltas = [{"block_id":left["block_id"],"reference_minus_candidate":right["score"]-left["score"]}
                      for left,right in zip(adaptive["samples"],baseline["samples"],strict=True)]
            values = [item["reference_minus_candidate"] for item in deltas]
            middle = median(values)
            passed = min(values) > 0 and middle > minimum
            comparison = {"passed":passed,"reason":"stable_gain" if passed else "insufficient_stable_gain",
                          "median_improvement":middle,"deltas":deltas}
        comparisons["adaptive_vs_"+reference] = comparison
    check(comparisons == report["test_comparisons"], "heldout comparison decisions differ from frozen thresholds")
    return {"report":report,"frozen":frozen,"heldout":heldout,"rounds":rounds}

def export(root, private, source, hand, dest, seal, aborted):
    check(not dest.exists() and not dest.is_symlink(), "archive destination already exists")
    for path in (root,root/"controller"):
        check(path.is_dir() and not path.is_symlink(), "invalid controller directory")
    gate_file = root / ("abort-audit.json" if aborted else "controller/report.json")
    gate_bytes = raw(gate_file)
    check(SHA.fullmatch(seal) and digest(gate_bytes)==seal, "operator-confirmed terminal marker SHA does not match")
    private_bytes = raw(private)
    config = json.loads(private_bytes)
    original = {}
    original["controller/journal.sqlite"], row_bytes, rows = records(root / "controller")
    read_named(root,original,["controller/manifest.json","controller/lease.sqlite","controller/lease.sqlite-journal",
        "controller/report.json","controller/frozen.json","pipeline-plan.json","pipeline-status.json","pipeline-run.log","abort-audit.json"])
    check(original[gate_file.relative_to(root).as_posix()]==gate_bytes, "terminal marker changed during read")
    check(json.loads(original["controller/manifest.json"])==rows["manifest"], "controller manifest and journal disagree")
    manifest = rows["manifest"]
    template = public_config(config,manifest)
    if "pipeline-plan.json" in original:
        check(json.loads(original["pipeline-plan.json"])["config_sha256"]==digest(private_bytes), "pipeline private config binding differs")
    if aborted:
        marker = json.loads(gate_bytes)
        check(marker["protocol_config_sha256"]==digest(private_bytes), "abort private config binding differs")
    source_bytes = raw(source)
    source_manifest, source_files, bindings = check_source(source_bytes,manifest)
    for path in sorted((root/"controller/harnesses").glob("*.py")):
        data = raw(path)
        check(SHA.fullmatch(path.stem) and digest(data)==path.stem, "archived generator source hash differs")
        original[path.relative_to(root).as_posix()] = data
    rounds_paths = sorted((root/"controller/rounds").glob("*.json"))
    check({p.stem for p in rounds_paths} == {key.split("/",1)[1] for key in rows if key.startswith("round/")},
          "round artifact roster differs from the journal")
    for path in rounds_paths:
        data = raw(path)
        check(json.loads(data)==rows["round/"+path.stem], "round file differs from journal")
        original[path.relative_to(root).as_posix()] = data
    check(f"controller/harnesses/{manifest['seed_sha256']}.py" in original, "missing seed source")
    calls = verify_api(rows,manifest)
    check(all(f"controller/harnesses/{call['harness_sha256']}.py" in original for call in calls.values()), "missing call source")
    recv, index = receipt_index(private.parent/config["receipt_directory"],manifest)
    ref = set(receipt_ids(rows))
    check(ref <= set(index), "leased receipt named by journal is missing")
    if aborted:
        check_abort(rows,marker,index)
        status = "contract_abort"
        selections = {}
    else:
        terminal = check_settled(rows,original,index)
        status = "test_settled"
        selections = {}
        for arm in ("adaptive","fixed"):
            selected = terminal["frozen"][arm]
            trial = selected["trial"]
            sha = fingerprint(trial["program"])
            check(sha==trial["program_sha256"], "selected manager program hash differs")
            harness_sha = trial["harness_sha256"]
            generator = original[f"controller/harnesses/{harness_sha}.py"] if harness_sha else None
            api_key = "api/" + trial["id"] if harness_sha else None
            check(not api_key or calls[api_key]["harness_sha256"]==harness_sha, "selected generator is not the actual source for the selected trial")
            selections[arm] = {"program_sha256":sha,"trial_id":trial["id"],"harness_sha256":harness_sha,"provider_call_id":api_key,
                               "validation_score":selected["score"],"bootstrap":harness_sha is None,
                               "generator_filename":f"{arm}.generator.py" if generator is not None else None}
    hand_ids = {} if hand is None else json.loads(raw(hand))["leased_receipts"]
    check(all(JOB.fullmatch(k) for k in hand_ids) and not set(hand_ids)&set(index), "invalid or overlapping hand/formal job sets")
    check(raw(root/"controller/journal.sqlite")==original["controller/journal.sqlite"], "journal changed during export")
    check(raw(gate_file)==gate_bytes, "terminal marker changed during export")

    dest.mkdir(exist_ok=False,parents=True)
    for name,value in original.items():
        path = dest / name
        path.parent.mkdir(exist_ok=True,parents=True)
        path.write_bytes(value)
    record_entries = {f"{key}.json":value for key,value in row_bytes.items() if key.startswith(("api/","plan/"))}
    write_zip(dest/"original-api-and-plans.zip",record_entries)
    write_zip(dest/"leased-receipts.zip",recv)
    (dest/"frozen-source.zip").write_bytes(source_bytes)
    (dest/"frozen-source-manifest.json").write_bytes(source_manifest)
    (dest/"adapter_contract.json").write_bytes(nice(manifest["contract"]))
    (dest/"schema3.template.json").write_bytes(nice(template))
    if selections:
        chosen = dest / "selected"
        chosen.mkdir()
        for arm,meta in selections.items():
            trial = terminal["frozen"][arm]["trial"]
            (chosen/f"{arm}.program.json").write_bytes(nice(trial["program"]))
            if meta["harness_sha256"] is not None:
                (chosen/meta["generator_filename"]).write_bytes(original[f"controller/harnesses/{meta['harness_sha256']}.py"])
        active = terminal["frozen"]["active_harness_sha256"]
        (chosen/"active.generator.py").write_bytes(original[f"controller/harnesses/{active}.py"])
        (chosen/"selection.json").write_bytes(nice({"active_generator_sha256":active,"finalists":selections}))
    (dest/"server_job_index.json").write_bytes(nice({"schema":1,"hand_job_ids":sorted(hand_ids),"formal_jobs":index,
        "directly_referenced_formal_job_ids":sorted(ref),"extra_local_formal_job_ids":sorted(set(index)-ref),
        "additional_server_failed_jobs":"include any owned job absent from local receipts after server-side reconciliation",
        "server_evidence_allowlist":LOGICAL_EVIDENCE}))
    audit = {"schema":1,"status":status,"terminal_marker_sha256":seal,"private_schema3_bytes_sha256":digest(private_bytes),
        "contract_canonical_sha256":fingerprint(manifest["contract"]),"verified_frozen_source_files":source_files,
        "verified_controller_source_bindings":bindings,"journal_integrity":"ok","journal_row_order":list(rows),
        "receipt_count":len(index),"receipt_counts_by_split":dict(Counter(x.get("split","unknown") for x in index.values())),
        "provider_calls":calls,"selected":selections}
    if selections:
        audit["heldout_effect_status"] = terminal["heldout"]["status"]
        audit["heldout_comparisons"] = terminal["report"]["test_comparisons"]
    (dest/"export_audit.json").write_bytes(nice(audit))
    manifest = {"schema":1,"status":status,"private_schema3_bytes_sha256":digest(private_bytes),
        "files_sha256":{p.relative_to(dest).as_posix():digest(p.read_bytes()) for p in sorted(dest.rglob("*")) if p.is_file()},
        "receipt_member_sha256":{k:digest(v) for k,v in recv.items()},
        "api_and_plan_member_sha256":{k:digest(v) for k,v in record_entries.items()}}
    (dest/"artifact_manifest.json").write_bytes(nice(manifest))
    print(json.dumps(audit, ensure_ascii=False, sort_keys=True))

def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-root",type=Path,required=True)
    cli.add_argument("--private-config",type=Path,required=True)
    cli.add_argument("--source-archive",type=Path,required=True)
    cli.add_argument("--hand-audit",type=Path)
    cli.add_argument("--output",type=Path,required=True)
    cli.add_argument("--terminal-sha256",required=True)
    cli.add_argument("--aborted",action="store_true",help="require an explicit original abort marker instead of a settled report")
    args = cli.parse_args()
    export(args.run_root,args.private_config,args.source_archive,args.hand_audit,args.output,args.terminal_sha256,args.aborted)

if __name__ == "__main__":
    main()
