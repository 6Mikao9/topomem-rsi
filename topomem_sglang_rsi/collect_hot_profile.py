"""Bind startup metadata to independent completed stock native parity receipts."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from integrations.topology_memory import sglang_backing as backing
from integrations.topology_memory import sglang_ingress_gate as ingress
from integrations.topology_memory.sglang_observe import save
from rsi_memory.memory_program import MemoryScenario
from rsi_memory.topology import HardwareTopology, fingerprint

from .contract import case_descriptor, checked_sha, compile_layout, load_case, load_external, read_json
from .hot_config import HOT_SLOTS, PROFILE_KIND, normalized_schedule, observed_rank_geometry, profile_runtime_identity, validate_profile


def collect(base, probes):
    base = Path(base)
    original = case_descriptor(base)
    target, topology, scenario, _ = load_case(base, original, fixture=True)
    external, tokens = load_external(base, original, require_gate=True)
    if type(probes) is not dict or set(probes) != set(HOT_SLOTS):
        raise ValueError("require a completed native probe at each of 4096, 6144 and 8192")
    profile = {"schema": 1, "kind": PROFILE_KIND, "status": "observed_native_parity",
               "base_target_sha256": original["target_sha256"], "base_topology_sha256": original["topology_sha256"],
               "calibration_workload_sha256": original["workload_sha256"], "runtime_identity_sha256": None,
               "calibration_adapter_sources_sha256": None, "observations": {}}
    baseline = None
    for slots in HOT_SLOTS:
        case, result_path = map(Path, probes[slots])
        spec = case_descriptor(case)
        observed_target = read_json(case / "target.json")
        observed_topology = HardwareTopology.from_dict(read_json(case / "topology.json"))
        observed_scenario = MemoryScenario.from_dict(read_json(case / "scenario.json"))
        observed_external, _ = load_external(case, spec, expected_inputs=tokens, require_gate=True)
        expected_target = {**target, "pool": {**target["pool"], "hot_slots": slots}}
        expected_scenario = {**scenario.to_dict(), "target_sha256": fingerprint(expected_target),
                             "schedule_sha256": fingerprint(observed_external)}
        if (observed_target != expected_target or observed_topology.fingerprint() != topology.fingerprint()
                or observed_scenario.to_dict() != expected_scenario or observed_scenario.fingerprint() != spec["scenario_sha256"]
                or fingerprint(observed_target) != spec["target_sha256"] or fingerprint(observed_external) != spec["schedule_sha256"]
                or spec["workload_sha256"] != original["workload_sha256"] or spec["topology_sha256"] != original["topology_sha256"]
                or normalized_schedule(observed_external) != normalized_schedule(external)
                or json.loads(observed_external["engine"]["hisparse_config"])["device_buffer_size"] != slots
                or (case / "fixture.json").read_bytes() != (base / "fixture.json").read_bytes()):
            raise ValueError("native hot probe changed a frozen input other than the memory knob")
        plan, workers = compile_layout(observed_target, observed_topology, observed_scenario, None)
        raw = result_path.read_bytes()
        result = json.loads(raw)
        if ((result.get("status"), result.get("arm"), result.get("phase"), result.get("attention_dp_size")) != ("completed", "native", "parity", 2)
                or any(result[key] != spec[key] for key in ("target_sha256", "topology_sha256", "scenario_sha256"))
                or result["schedule_inputs_sha256"] != spec["schedule_sha256"] or result["compiled_plan_sha256"] != fingerprint(plan)
                 or result["program_sha256"] != plan["program_sha256"]
                 or result.get("ingress_contract_sha256") != fingerprint(external[ingress.EXTERNAL_FIELD])
                 or result.get("input_tokenization_sha256") != fingerprint(tokens)
                 or result.get("global_ticks", {}).get("global_ticks_sha256") != result.get("global_ticks_sha256")):
            raise ValueError("profile inputs must be completed raw native parity receipts from the same calibration fixture")
        checked_sha(result.get("global_ticks_sha256"))
        max_live = max(row["input_tokens"] + len(row["output_ids"]) for group in ("warmup", "requests") for row in result[group])
        for rank, worker in enumerate(workers):
            record = result["ranks"][rank]
            observed = record["backing"]
            if (observed["hot_slots"] != slots or observed["layer_tiers"] != [backing.HOST] * len(worker.layer_tiers)
                    or observed["pinned_host_bytes"] != worker.pinned_host_bytes or observed["local_hbm_bytes"] or observed["peer_hbm_bytes"]
                    or record["parallel"]["attention_dp_size"] != 2 or record["parallel"]["attention_dp_rank"] != rank):
                raise ValueError("hot probe did not exercise the expected stock rank and native backing")
            if max_live <= slots and record["dispatch"]["selected_miss_payload_bytes_by_tier"] != {backing.HOST: 0}:
                raise ValueError("fully resident native hot probe reported unexpected cold reads")
        identity = profile_runtime_identity(result["runtime"])
        keys = ("effective_schedule_sha256", "dispatch_sha256", "output_ids_sha256", "packed_kv_sha256_by_rank",
                "global_ticks_sha256", "ingress_contract_sha256", "input_tokenization_sha256")
        comparison = [result[key] for key in keys]
        if baseline is None:
            baseline = comparison
            profile["runtime_identity_sha256"] = identity
            profile["calibration_adapter_sources_sha256"] = result["runtime"]["adapter_source_sha256"]
        if (comparison != baseline or identity != profile["runtime_identity_sha256"]
                or result["runtime"]["adapter_source_sha256"] != profile["calibration_adapter_sources_sha256"]):
            raise ValueError("native hot probes disagree on fixed scheduler, dispatch, selected packed KV, outputs or environment")
        profile["observations"][str(slots)] = {"result_sha256": sha256(raw).hexdigest(),
                                                "case_manifest_sha256": spec["manifest_sha256"], "ranks": observed_rank_geometry(result)}
    return validate_profile(profile, target, topology, external)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-case", type=Path, required=True)
    parser.add_argument("--probe", nargs=3, metavar=("HOT_SLOTS", "CASE", "RESULT_JSON"), action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    probes = {int(slots): (case, result) for slots, case, result in args.probe}
    if len(probes) != len(args.probe):
        parser.error("each hot capacity must occur once")
    save(args.output, collect(args.base_case, probes))


if __name__ == "__main__":
    main()
