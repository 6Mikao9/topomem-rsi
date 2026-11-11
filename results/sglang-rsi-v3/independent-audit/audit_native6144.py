"""Check the preregistered supplementary native-6144 comparison after it settles."""

import argparse
from datetime import datetime
import json
from pathlib import Path
from statistics import median

from audit_protocol import digest, file_sha, fingerprint, read_json


NATIVE = "native6144_all_host"
ADAPTIVE = "frozen_adaptive"
PREREG_SHA = "47fb544e2c326d388d221101d652fb59073af1899ff69c949ffdf8bc07e7442d"
MATCH = ("effective_schedule_sha256", "dispatch_sha256", "output_ids_sha256", "runtime_sha256",
         "global_ticks_sha256", "input_tokenization_sha256")


def verify(root, formal):
    prereg_path = root / "preregistration.json"
    prereg = read_json(prereg_path)
    assert file_sha(prereg_path) == PREREG_SHA
    assert prereg["driver_sha256"] == "9a4eef769bc864ff45f123c4cbf7cfcb0a81881ca222abbf26ff412802247994"
    result = read_json(root / "result.json")
    assert result["preregistration_sha256"] == PREREG_SHA
    status = result["status"]
    report = {"schema": 1, "kind": "independent_sglang_native6144_supplement_audit",
              "scope": "fresh timed blocks on the same two held-out cases as formal RSI; no new generalization split",
              "preregistration_sha256": PREREG_SHA, "result_sha256": file_sha(root / "result.json"),
              "status": status, "timed_jobs_completed": result["timed_jobs_completed"],
              "performance_gate": result["performance_gate"]}
    if status.startswith("skipped_"):
        assert result["timed_jobs_completed"] == 0 and result["performance_gate"] == "not_evaluated"
        assert not (root / "jobs.jsonl").exists()
        return {**report, "reason": result["reason"]}

    binding = read_json(root / "manifest.json")["bound"]
    assert binding["preregistration_sha256"] == PREREG_SHA
    assert binding["formal_report_sha256"] == file_sha(formal / "controller/report.json")
    assert binding["formal_frozen_sha256"] == file_sha(formal / "controller/frozen.json")
    frozen = read_json(formal / "controller/frozen.json")
    config = read_json(formal / "schema3.json")
    assert binding["formal_private_config_sha256"] == file_sha(formal / "schema3.json")
    assert binding["adapter_config_field_diff"].keys() == {"receipt_directory"}
    assert fingerprint(frozen["adaptive"]["trial"]["program"]) == binding["adaptive_program_sha256"]
    program = read_json(root / "frozen_adaptive.program.json")
    assert program == frozen["adaptive"]["trial"]["program"]
    case_specs = prereg["cases"]
    assert case_specs == config["cases"]["test"]
    cases = {case["id"]: case for case in case_specs}
    assert len(cases) == 2
    if "started_utc" in result:
        latest = datetime.fromisoformat(prereg["cutoffs"]["latest_start_utc"].replace("Z", "+00:00"))
        assert datetime.fromisoformat(result["started_utc"]) <= latest
    else:
        assert status == "execution_incomplete" and result["timed_jobs_completed"] == 0
    log = root / "jobs.jsonl"
    events = [json.loads(line) for line in log.read_bytes().splitlines()] if log.exists() else []
    proof_by_case, timing, hashes = {}, {}, {}
    old_jobs = {path.stem for path in (formal / "receipts").glob("sg-*.json")}
    client = {k: config["executor"][k] for k in ("source", "gpu_indices", "gpu_uuids")}
    client.update(environment=config["executor"].get("environment", {}), source_manifest_sha256=config["source_manifest_sha256"])

    def receipt(job):
        path = root / "receipts" / f"{job}.json"
        row = read_json(path)
        summary = row["result"]
        assert row["status"] == "completed" and row["job_id"] == job and row["returncode"] == 0
        assert row["failure"] is None and row["environment_conflict"] is False
        assert row["process_group_remaining"] is row["process_group_cleanup_required"] is False
        assert row["remaining_compute_pids"] == ""
        assert all(row[key] == client[key] for key in ("source_manifest_sha256", "gpu_indices", "gpu_uuids"))
        assert fingerprint(summary) == row["result_canonical_sha256"]
        assert digest((json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()) == row["result_sha256"]
        hashes[job] = file_sha(path)
        return row, summary

    positions = []
    for event in events:
        phase = event["phase"]
        if phase == "qualification":
            case = cases[event["case"]]
            proof_by_case[case["id"]] = {}
            signatures = []
            for arm in (NATIVE, ADAPTIVE):
                bound = event["proofs"][arm]
                _, summary = receipt(bound["job_id"])
                assert summary["phase"] == "parity" and summary["case"] == case and summary["feasible"]
                assert summary["raw_result_sha256"] == bound["raw_result_sha256"]
                assert summary["program_sha256"] == (None if arm == NATIVE else binding["adaptive_program_sha256"])
                metrics = summary["metrics"]
                signatures.append(tuple(metrics[field] for field in (*MATCH, "packed_kv_sha256_by_rank")))
                proof_by_case[case["id"]][arm] = summary
                if arm == NATIVE:
                    start = summary["memory_compilation"]["startup"]
                    assert start["hot_slots"] == start["native_reference_override"] == 6144
                    assert summary["cross_capacity_native_parity"]["matched"] is True
                    assert all(row["layer_tiers"] == ["host"] * 6 for row in metrics["ranks"])
                    _, repeat = receipt(event["native_same_capacity_repeat_job_id"])
                    assert repeat["case"] == case and repeat["feasible"] and repeat["phase"] == "parity"
                    assert repeat["cross_capacity_native_parity"]["matched"] is True
                    assert tuple(repeat["metrics"][field] for field in (*MATCH, "packed_kv_sha256_by_rank")) == signatures[-1]
            assert signatures[0] == signatures[1]
        else:
            assert phase == "performance"
            ident = event["identity"]
            arm, case_id, block = (ident[key] for key in ("label", "case_id", "block_id"))
            case = cases[case_id]
            positions.append((block, case_id, arm))
            job = event["job_id"]
            assert job not in old_jobs
            row, summary = receipt(job)
            assert row["timeout"] is False and summary["phase"] == "performance" and summary["case"] == case
            assert summary["raw_result_sha256"] == event["raw_result_sha256"]
            payload = {"schema": 1, "case": case, "arm": "native" if arm == NATIVE else "generated",
                       "phase": "performance", "program": None if arm == NATIVE else program,
                       "qualification": event["qualification"], "deadline": 300,
                       "launch_sha256": config["launch_sha256"], "source_manifest_sha256": config["source_manifest_sha256"],
                       "expected_inputs": config["token_inputs"][case_id], "hot_profile": config["hot_profile"],
                       "native_hot": 6144 if arm == NATIVE else None, "cross_cap_reference": None}
            request = {"operation": "run", "job_id": job, "module": "topomem_sglang_rsi.sglang_eval", "payload": payload,
                       "timeout": 780, **client}
            assert row["request_sha256"] == fingerprint(request)
            if not summary["feasible"]:
                assert status == "execution_incomplete"
                break
            proof = proof_by_case[case_id][arm]
            assert event["qualification"]["raw_result_sha256"] == proof["raw_result_sha256"]
            assert tuple(summary["metrics"][field] for field in MATCH) == tuple(proof["metrics"][field] for field in MATCH)
            assert not summary["metrics"]["packed_kv_sha256_by_rank"]
            assert all(rank.get("resident_producer_oracle") is None for rank in summary["metrics"]["ranks"])
            assert summary["score"] == event["window_ms"] == summary["metrics"]["timing"]["window_seconds"] * 1000
            if arm == NATIVE:
                start = summary["memory_compilation"]["startup"]
                assert start["hot_slots"] == start["native_reference_override"] == 6144 and start["backing_mode"] == "tiered"
                assert all(rank["layer_tiers"] == ["host"] * 6 and rank["pinned_host_bytes"] > 0
                           and rank["local_hbm_bytes"] == rank["peer_hbm_bytes"] == 0 for rank in summary["metrics"]["ranks"])
            timing[block, case_id, arm] = (summary["score"], job)
    expected = [(block["block_id"], case["id"], arm) for block in prereg["blocks"]
                for case in case_specs for arm in block["order"]]
    assert positions == expected[:len(positions)]
    actual_startup = {}
    for case, arms in proof_by_case.items():
        values_by_arm = {arm: {key: summary["memory_compilation"]["startup"][key]
                              for key in ("hot_slots", "backing_mode")} for arm, summary in arms.items()}
        values_by_arm["same_hot_capacity"] = values_by_arm[NATIVE]["hot_slots"] == values_by_arm[ADAPTIVE]["hot_slots"]
        actual_startup[case] = values_by_arm
    assert result.get("startup_by_case", {}) == actual_startup
    values = []
    for index, block in enumerate(result.get("complete_blocks", ())):
        assert {key: block[key] for key in ("block_id", "order")} == prereg["blocks"][index]
        name = block["block_id"]
        totals = {}
        for arm in (NATIVE, ADAPTIVE):
            totals[arm] = sum(timing[name, case, arm][0] for case in cases)
            for case in cases:
                value, job = timing[name, case, arm]
                assert block["window_ms_by_arm_case"][arm][case] == value
                assert block["job_id_by_arm_case"][arm][case] == job
        assert totals == block["window_ms_sum_by_arm"]
        delta = totals[NATIVE] - totals[ADAPTIVE]
        assert delta == block["native_minus_adaptive_ms"]
        values.append(delta)
    cross_capacity_controls = {}
    if status == "completed":
        assert len(positions) == len(timing) == 32 and len(values) == 8
        extra = {path.stem for path in (root / "receipts").glob("sg-*.json")} - hashes.keys()
        assert len(extra) == 4
        for job in sorted(extra):
            _, summary = receipt(job)
            case = summary["case"]["id"]
            native = proof_by_case[case][NATIVE]
            assert summary["phase"] == "parity" and summary["arm"] == "native" and summary["feasible"]
            assert summary["program_sha256"] is None
            startup = summary["memory_compilation"]["startup"]
            assert startup["hot_slots"] == 4096 and startup["native_reference_override"] is None
            assert startup["backing_mode"] == "tiered"
            assert all(row["layer_tiers"] == ["host"] * 6 for row in summary["metrics"]["ranks"])
            fields = (*MATCH, "packed_kv_sha256_by_rank")
            assert all(summary["metrics"][field] == native["metrics"][field] for field in fields)
            cross_capacity_controls.setdefault(case, []).append(
                {"job_id": job, "raw_result_sha256": summary["raw_result_sha256"]})
        for case, controls in cross_capacity_controls.items():
            assert len(controls) == 2
            assert proof_by_case[case][NATIVE]["cross_capacity_native_parity"]["reference"] in controls
        assert len(cross_capacity_controls) == 2 and len(hashes) == 42
        assert median(values) == result["median_ms"] and sum(value > 0 for value in values) == result["positive_blocks"]
        passed = all(value > 0 for value in values) and median(values) > prereg["decision"]["median_strictly_greater_than_ms"]
        assert result["performance_gate"] == ("passed" if passed else "failed")
    else:
        assert result["performance_gate"] == "not_evaluated"
    return {**report, "receipt_sha256_by_job": hashes, "complete_blocks": len(values), "observed_paired_gains_ms": values,
            "median_for_completed_study_ms": median(values) if status == "completed" else None,
            "actual_startup_by_case": actual_startup,
            "qualification_matches": len(proof_by_case), "fresh_timing_events": len(positions),
            "native4096_cross_capacity_control_receipts": cross_capacity_controls,
            "limits": "Local leased summaries checked; cutoff is bound by preregistered driver. No independent GPU rerun."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("supplement", "formal", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = verify(args.supplement, args.formal)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
    print(file_sha(args.output))


if __name__ == "__main__":
    main()
