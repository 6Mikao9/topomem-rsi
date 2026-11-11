"""Run one frozen SGLang parity or performance phase inside a leased GPU job."""

import argparse
from hashlib import sha256
import json
import math
import os
from pathlib import Path
import subprocess
import sys

from integrations.topology_memory.sglang_observe import save
from integrations.topology_memory.sglang_resident import RESIDENT_TIER
from rsi_memory.execution_worker import digest
from rsi_memory.memory_program import MemoryProgram
from rsi_memory.topology import fingerprint

from .contract import NCCL_SHA256, SCOPE, checked_case, checked_id, checked_inputs, checked_sha, load_case, load_external, read_json
from .hot_config import HOT_SLOTS, MEASURED_FIELDS, PARITY_KEYS, check_live_metadata, compile_memory


def _qualification(jobs, spec, output, *, case, arm, program_sha256, source_sha256, compilation, same_backing=False):
    if type(spec) is not dict or spec.keys() != {"job_id", "raw_result_sha256"}:
        raise ValueError("require an immutable SGLang qualification receipt")
    job = jobs / checked_id(spec["job_id"])
    serving = job / "serving"
    proof, receipt_path, summary_path = serving / "result.json", job / "receipt.json", job / "result.json"
    if (any(path.is_symlink() for path in (job, serving, proof, receipt_path, summary_path))
            or not proof.resolve(strict=True).is_relative_to(jobs)):
        raise ValueError("qualification is outside the leased serving jobs")
    receipt = read_json(receipt_path)
    summary_raw = summary_path.read_bytes()
    summary = json.loads(summary_raw)
    if (receipt.get("schema") != 2 or receipt.get("status") != "completed" or receipt.get("job_id") != spec["job_id"]
            or receipt.get("source_manifest_sha256") != source_sha256 or receipt.get("returncode") != 0
            or receipt.get("process_group_remaining") is not False or receipt.get("process_group_cleanup_required") is not False
            or receipt.get("remaining_compute_pids") != "" or receipt.get("result") != summary
            or receipt.get("result_canonical_sha256") != digest(summary) or receipt.get("result_sha256") != sha256(summary_raw).hexdigest()
            or summary.get("scope") != SCOPE or summary.get("phase") != "parity" or summary.get("case") != case
            or summary.get("feasible") is not True or summary.get("arm") != arm
            or summary.get("program_sha256") != program_sha256
            or any(summary.get("memory_compilation", {}).get(key) != compilation[key] for key in
                   ("metadata_profile_sha256", "derived_target_sha256", "derived_topology_sha256", "derived_scenario_sha256",
                    "derived_schedule_sha256", "ingress_contract_sha256"))
            or summary.get("raw_result_sha256") != spec["raw_result_sha256"]):
        raise ValueError("qualification lacks a completed leased SGLang receipt for the bound case and arm")
    if same_backing and summary["memory_compilation"]["startup"] != compilation["startup"]:
        raise ValueError("qualification uses a different frozen backing mode or allocation proof")
    checked_sha(summary.get("metrics", {}).get("global_ticks_sha256"))
    raw = proof.read_bytes()
    if sha256(raw).hexdigest() != checked_sha(spec["raw_result_sha256"]):
        raise ValueError("qualification result digest changed")
    with output.open("xb") as stream:
        stream.write(raw)
    return output, summary


def _feedback(directory, result, workers):
    counts = [[0] * len(worker.layer_tiers) for worker in workers]
    raw_logs = []
    for worker in workers:
        rank = worker.rank
        raw = (directory / f"kv-rank-{rank}.jsonl").read_bytes()
        raw_logs.append(sha256(raw).hexdigest())
        for line in raw.splitlines():
            event = json.loads(line)
            layer = event["layer"]
            if (event["event"] != "kv_selected_exact" or event["rank"] != rank
                    or type(layer) is not int or not 0 <= layer < len(counts[rank])):
                raise ValueError("invalid selected KV diagnostic event")
            counts[rank][layer] += sum(row["miss_rows"] for row in event["rows"])
        rank_receipt = result["ranks"][rank]
        recorded = rank_receipt["dispatch"]["selected_miss_payload_bytes_by_tier"]
        tiers = tuple(rank_receipt["backing"]["layer_tiers"])
        if result.get("backing_mode") == RESIDENT_TIER:
            if tiers != (RESIDENT_TIER,) * len(counts[rank]) or any(counts[rank]):
                raise ValueError("resident packed KV diagnostics recorded cold misses or tiers")
        elif tiers != worker.layer_tiers:
            raise ValueError("packed KV diagnostics changed compiled backing tiers")
        reconstructed = {tier: sum(value for value, layer_tier in zip(counts[rank], tiers, strict=True)
                                   if layer_tier == tier) * worker.row_bytes for tier in tiers}
        if recorded != reconstructed:
            raise ValueError("diagnostic per-layer rows disagree with the real tier byte totals")
    return {"cold_miss_rows_by_rank_layer": counts, "kv_log_sha256_by_rank": raw_logs}


def _result_metadata(result):
    ranks = []
    for row in result["ranks"]:
        dispatch, backing = row["dispatch"], row["backing"]
        ranks.append({"rank": backing["rank"], "layer_tiers": backing["layer_tiers"],
                      "hot_slots": backing["hot_slots"], **{name: backing[name] for name in MEASURED_FIELDS},
                      "worker_limits": row["schedule"]["worker_limits"],
                      "pinned_host_bytes": backing["pinned_host_bytes"], "local_hbm_bytes": backing["local_hbm_bytes"],
                      "peer_hbm_bytes": backing["peer_hbm_bytes"], "relay_workspace_bytes": backing["relay_workspace_bytes"],
                      "selected_miss_payload_bytes_by_tier": dispatch["selected_miss_payload_bytes_by_tier"],
                      "backup_payload_bytes_by_tier": dispatch["backup_payload_bytes_by_tier"],
                      "relay": dispatch["relay"], "torch_cuda_memory": dispatch["torch_cuda_memory"]})
        if backing.get("backing_mode") == RESIDENT_TIER:
            ranks[-1].update({key: backing[key] for key in ("resident_proof_sha256", "resident_sentinel_logical_bytes",
                         "resident_sentinel_allocator_charge_bytes", "resident_sentinel_allocated_delta_bytes",
                         "resident_sentinel_reserved_delta_bytes")})
            ranks[-1]["resident_producer_oracle"] = dispatch.get("resident_producer_oracle")
    return {"ranks": ranks, "effective_schedule_sha256": result["effective_schedule_sha256"],
            "dispatch_sha256": result["dispatch_sha256"], "output_ids_sha256": result["output_ids_sha256"],
            "packed_kv_sha256_by_rank": result["packed_kv_sha256_by_rank"], "runtime_sha256": fingerprint(result["runtime"]),
            "global_ticks_sha256": result["global_ticks_sha256"], "global_ticks": result["global_ticks"],
            "ingress": result["ingress"], "global_tick_receipt_sha256_by_rank": result["global_tick_receipt_sha256_by_rank"],
            "backing_mode": result.get("backing_mode", "tiered"), "resident_binding_sha256": result.get("resident_binding_sha256")}


def run(payload, output, *, model, launch_sha256, source_sha256, execute=subprocess.run):
    """Validate trusted inputs, retain raw runner receipts, and summarize one observation."""
    required = {"schema", "case", "program", "arm", "phase", "qualification", "deadline", "launch_sha256", "source_manifest_sha256",
                "expected_inputs", "hot_profile", "native_hot", "cross_cap_reference"}
    if type(payload) is not dict or payload.keys() != required or type(payload["schema"]) is not int or payload["schema"] != 1:
        raise ValueError("invalid isolated SGLang evaluator request")
    output = Path(output).resolve()
    jobs = output.parent.parent
    if jobs.name != "jobs" or output.name != "result.json" or output.is_file():
        raise ValueError("serving evaluator must write a new leased job result")
    arm, phase = payload["arm"], payload["phase"]
    native_hot, cross_cap = payload["native_hot"], payload["cross_cap_reference"]
    if (arm not in ("native", "manual", "generated") or phase not in ("parity", "performance")
            or (arm == "native") != (payload["program"] is None)
            or ((phase == "performance" or arm != "native") != (payload["qualification"] is not None))
            or type(payload["deadline"]) is not int or not 1 <= payload["deadline"] <= 1800
            or (native_hot is not None and (arm != "native" or type(native_hot) is not int or native_hot not in HOT_SLOTS or native_hot == 4096))
            or ((arm == "native" and phase == "parity" and native_hot is not None) != (cross_cap is not None))):
        raise ValueError("invalid SGLang arm, phase, or qualification requirement")
    if checked_sha(launch_sha256) != checked_sha(payload["launch_sha256"]):
        raise ValueError("trusted launch configuration digest differs")
    if checked_sha(source_sha256) != checked_sha(payload["source_manifest_sha256"]):
        raise ValueError("trusted serving source manifest digest differs")
    spec = checked_case(payload["case"])
    expected_inputs = checked_inputs(payload["expected_inputs"])
    root = jobs.parent / "cases"
    case = root / spec["id"]
    if root.is_symlink() or case.is_symlink() or not case.resolve(strict=True).is_relative_to(root):
        raise ValueError("case must be staged in the trusted experiment directory")
    target, topology, scenario, evidence = load_case(case, spec, fixture=True)
    fixed_fixture = read_json(case / "fixture.json")
    if {row["rid"] for group in ("warmup", "requests") for row in fixed_fixture[group]} != expected_inputs.keys():
        raise ValueError("precommitted tokenization has a different fixed request set")
    if arm == "native":
        program = None
    else:
        program = MemoryProgram.from_dict(payload["program"])
    profile = payload["hot_profile"]
    external, _ = load_external(case, spec, expected_inputs=expected_inputs, require_gate=True)
    compiled = compile_memory(target, topology, scenario, external, profile, program, native_hot=native_hot)
    plan, workers, audit = compiled.plan, compiled.workers, compiled.audit
    startup = audit["startup"]
    mode = startup.get("backing_mode", "tiered")
    if mode not in ("tiered", RESIDENT_TIER) or mode == RESIDENT_TIER and (program is None or compiled.cold_program is not None):
        raise ValueError("invalid trusted backing compilation")
    resident_binding = None
    compiled_dir = output.parent / "compiled_case"
    compiled_dir.mkdir(exist_ok=False)
    for name, value in (("target", compiled.target), ("topology", compiled.topology.to_dict()), ("scenario", compiled.scenario.to_dict()),
                        ("external_schedule", compiled.external), ("cold_plan", plan), ("memory_compilation", audit)):
        save(compiled_dir / f"{name}.json", value)
    directory = output.parent / "serving"
    command = [sys.executable, "-B", "-m", "integrations.topology_memory.sglang_serving",
               "--model", str(model), "--target", str(compiled_dir / "target.json"), "--topology", str(compiled_dir / "topology.json"),
               "--scenario", str(compiled_dir / "scenario.json"), "--fixture", str(case / "fixture.json"), "--output", str(directory),
               "--external-schedule", str(compiled_dir / "external_schedule.json"),
               "--phase", phase, "--arm", arm, "--deadline", str(payload["deadline"])]
    for item in evidence:
        command.extend(("--link-evidence", str(item)))
    if program is not None:
        save(compiled_dir / "full_program.json", program.to_dict())
        if mode == RESIDENT_TIER:
            charge = startup["resident_sentinel_allocator_charge_bytes_by_node"]
            resident_binding = {"proof": startup["resident"],
                                "allocator_charge_bytes_by_rank": [charge[node] for _, node in compiled.scenario.rank_placement]}
            binding_path = compiled_dir / "resident_proof.json"
            save(binding_path, resident_binding)
            command.extend(("--resident-proof", str(binding_path)))
        else:
            program_path = compiled_dir / "cold_program.json"
            save(program_path, compiled.cold_program.to_dict())
            command.extend(("--program", str(program_path)))
    if payload["qualification"] is not None:
        proof, _ = _qualification(jobs, payload["qualification"], output.parent / "qualification.json", case=spec,
                                  arm=arm if phase == "performance" else "native", compilation=audit,
                                  program_sha256=audit["full_program_sha256"] if phase == "performance" else None,
                                  source_sha256=source_sha256, same_backing=phase == "performance")
        command.extend(("--qualification", str(proof)))
    reference = None
    if cross_cap is not None:
        default = compile_memory(target, topology, scenario, external, profile, None)
        _, reference = _qualification(jobs, cross_cap, output.parent / "cross_cap_reference.json", case=spec, arm="native",
                                      program_sha256=None, source_sha256=source_sha256, compilation=default.audit)
    completed = execute(command, check=False)
    raw = (directory / "result.json").read_bytes()
    result = json.loads(raw)
    result_sha = sha256(raw).hexdigest()
    summary = {"schema": 1, "scope": SCOPE, "case": spec, "phase": phase, "arm": arm, "score": None,
               "program_sha256": audit["full_program_sha256"], "cold_program_sha256": plan["program_sha256"],
               "compiled_plan_sha256": fingerprint(plan), "memory_compilation": audit,
               "raw_result_sha256": result_sha, "raw_result_relative": "serving/result.json"}
    if completed.returncode or result["status"] != "completed":
        message = result.get("error", "")
        if (arm != "native" and phase == "parity" and result["status"] == "failed" and
                (message in {"live output, selected KV, or actual dispatch differs from qualification",
                             "the bound staged route did not carry real cold KV",
                             "the staged route did not supply verified selected KV misses to attention",
                             "direct peer backing did not supply selected KV misses to attention",
                             "resident parity lacks complete independent packed producer evidence",
                             "resident backing did not meet its exact no-cold allocation and transfer contract"}
                 or any((directory / f"kv-rank-{worker.rank}.jsonl").is_file() and
                        b'"event":"kv_mismatch"' in (directory / f"kv-rank-{worker.rank}.jsonl").read_bytes()
                        for worker in workers))):
            return {**summary, "feasible": False, "failure": "candidate_failed_native_packed_kv_output_or_dispatch_gate"}
        raise RuntimeError("SGLang process did not complete its bound phase; inspect the raw job receipt")
    if (result["target_sha256"] != audit["derived_target_sha256"] or result["topology_sha256"] != audit["derived_topology_sha256"]
            or result["scenario_sha256"] != audit["derived_scenario_sha256"] or result["schedule_inputs_sha256"] != audit["derived_schedule_sha256"]
            or result["program_sha256"] != plan["program_sha256"] or result["compiled_plan_sha256"] != fingerprint(plan)
            or result["runtime"]["nccl_override_sha256"] != NCCL_SHA256 or result["attention_dp_size"] != 2
            or result["ingress_contract_sha256"] != audit["ingress_contract_sha256"]
            or result["ingress"]["contract_sha256"] != audit["ingress_contract_sha256"]
            or result["global_ticks"]["global_ticks_sha256"] != result["global_ticks_sha256"]
            or result.get("backing_mode", "tiered") != mode
            or result.get("resident_binding_sha256") != (fingerprint(resident_binding) if resident_binding is not None else None)
            or result["arm"] != arm or result["phase"] != phase):
        raise ValueError("SGLang raw result differs from the frozen evaluator request")
    checked_sha(result["global_ticks_sha256"])
    observed_inputs = {row["rid"]: {key: row[key] for key in ("input_tokens", "input_ids_sha256")}
                       for group in ("warmup", "requests") for row in result[group]}
    if observed_inputs != expected_inputs or result["input_tokenization_sha256"] != fingerprint(expected_inputs):
        raise ValueError("live serving used different tokenized input IDs from the frozen replay")
    check_live_metadata(result, profile, audit["startup"]["hot_slots"])
    summary.update(feasible=True, metrics=_result_metadata(result))
    summary["metrics"]["input_tokenization_sha256"] = fingerprint(observed_inputs)
    if reference is not None:
        if (reference["memory_compilation"]["external_scheduler_sha256"] != audit["external_scheduler_sha256"]
                or any(reference["metrics"][key] != summary["metrics"][key] for key in (*PARITY_KEYS, "input_tokenization_sha256"))):
            raise RuntimeError("cross-capacity native parity changed scheduling, dispatch, inputs, output, or packed KV")
        summary["cross_capacity_native_parity"] = {"reference": cross_cap, "matched": True}
    if phase == "parity":
        summary["feedback"] = _feedback(directory, result, workers)
    else:
        seconds = result["metrics"]["window_seconds"]
        if type(seconds) not in (int, float) or not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("invalid live SGLang replay window")
        summary["score"] = seconds * 1000
        summary["metrics"]["timing"] = result["metrics"]
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    source_sha256 = sha256((Path(__file__).resolve().parents[1] / "SOURCE_MANIFEST.json").read_bytes()).hexdigest()
    summary = run(read_json(args.input), args.output, model=os.environ["TOPOMEM_SGLANG_MODEL_DIR"],
                  launch_sha256=os.environ["TOPOMEM_SGLANG_LAUNCH_SHA256"], source_sha256=source_sha256)
    save(args.output, summary)


if __name__ == "__main__":
    main()
