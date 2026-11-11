"""Select each rank's one local HBM slab from a separate native DP2 calibration receipt."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from rsi_memory.memory_program import MemoryScenario
from rsi_memory.topology import HardwareTopology, fingerprint

from . import generate_dp2_case as dp2, sglang_backing as backing
from .sglang_observe import save


def _read(directory, name):
    return json.loads((directory / f"{name}.json").read_bytes())


def _calibration_counts(rank, raw, result):
    runtime = result["ranks"][rank]
    if (runtime["backing"]["layer_tiers"] != [backing.HOST] * dp2.POOL["layers"]
            or runtime["parallel"]["attention_dp_rank"] != rank):
        raise ValueError("calibration must use the native all-host worker on its bound rank")
    per_layer = [0] * dp2.POOL["layers"]
    decode_rows = [0] * len(per_layer)
    canonical_events = []
    for line in raw.splitlines():
        event = json.loads(line)
        layer = event["layer"]
        if (event["event"] != "kv_selected_exact" or event["rank"] != rank or type(layer) is not int
                or not 0 <= layer < len(per_layer) or not event["rows"]):
            raise ValueError("calibration includes an unverified rank or layer")
        rows = []
        for row in event["rows"]:
            miss = row["miss_rows"]
            history = row["history_bytes"]
            if (type(miss) is not int or type(history) is not int or history % dp2.ROW_BYTES
                    or not 0 <= miss <= history // dp2.ROW_BYTES <= dp2.TOP_K):
                raise ValueError("calibration has invalid packed history or miss rows")
            per_layer[layer] += miss
            rows.append({key: value for key, value in row.items() if key != "miss_rows"})
        decode_rows[layer] += len(rows)
        canonical_events.append({**event, "rows": rows})
    expected_rows = sum(len(request["output_ids"]) - 1 for group in ("warmup", "requests") for request in result[group]
                        if request["routed_dp_rank"] == rank)
    if (not expected_rows or decode_rows != [expected_rows] * len(per_layer)
            or fingerprint(canonical_events) != result["packed_kv_sha256_by_rank"][rank]
            or runtime["dispatch"]["selected_miss_payload_bytes_by_tier"] != {backing.HOST: sum(per_layer) * dp2.ROW_BYTES}):
        raise ValueError("calibration rows or aggregate misses disagree with completed parity")
    if not any(per_layer):
        raise ValueError("calibration observed no cold misses on this rank")
    return per_layer


def _manual_program(layers):
    local = {"op": "feature", "name": "target_is_local"}
    host = {"op": "feature", "name": "target_is_host"}
    rank = {"op": "feature", "name": "rank"}
    layer = {"op": "feature", "name": "layer"}

    def const(value):
        return {"op": "const", "value": value}

    def expr(op, *args):
        return {"op": op, "args": list(args)}

    target_layer = expr("if_lt", rank, const(1), const(layers[0]), const(layers[1]))
    layer_delta = expr("sub", layer, target_layer)
    # The exact selected layer prefers local HBM. All other layers prefer pinned host.
    selected = expr("if_lt", expr("mul", layer_delta, layer_delta), const(1), const(1), const(0))
    host_score = expr("sub", const(2), host)
    local_bonus = expr("mul", const(2), expr("mul", selected, local))
    return {"schema": 1, "kind": "kv_memory_program", "rules": [{"event": "allocate", "action": "place",
            "when": const(1), "score": expr("sub", host_score, local_bonus)}]}


def tune(output, *, calibration_case, calibration_result, evaluation_case):
    calibration_case, calibration_result, evaluation_case = map(Path, (calibration_case, calibration_result, evaluation_case))
    training, evaluation = (_read(path, "manifest") for path in (calibration_case, evaluation_case))
    if (training["workload_sha256"] == evaluation["workload_sha256"]
            or training["scenario_sha256"] == evaluation["scenario_sha256"]):
        raise ValueError("calibration and evaluation must use distinct frozen workloads")
    raw_result = (calibration_result / "result.json").read_bytes()
    result = json.loads(raw_result)
    if ((result["status"], result["arm"], result["phase"], result["attention_dp_size"]) != ("completed", "native", "parity", 2)
            or any(result[key] != training[key] for key in ("target_sha256", "topology_sha256", "scenario_sha256"))
            or result["schedule_inputs_sha256"] != training["schedule_sha256"]
            or result["compiled_plan_sha256"] != training["compiled_preview_sha256"]["native"]
            or any(training[key] != evaluation[key] for key in ("target_sha256", "topology_sha256"))):
        raise ValueError("require completed native parity on the declared calibration case and a common fixed target")
    schedule = _read(calibration_case, "external_schedule")
    if fingerprint(schedule) != training["schedule_sha256"]:
        raise ValueError("calibration schedule differs from its case manifest")
    for group in ("warmup", "requests"):
        completed = [{"rid": row["rid"], "arrival_ms": row["arrival_ms"], "routed_dp_rank": row["routed_dp_rank"],
                      "max_new_tokens": len(row["output_ids"])} for row in result[group]]
        if completed != schedule[group]:
            raise ValueError("calibration outputs differ from the declared fixed requests")
    counts, log_hashes = [], []
    for rank in range(2):
        raw = (calibration_result / f"kv-rank-{rank}.jsonl").read_bytes()
        counts.append(_calibration_counts(rank, raw, result))
        log_hashes.append(sha256(raw).hexdigest())
    hottest = [max(range(len(rows)), key=rows.__getitem__) for rows in counts]
    program = _manual_program(hottest)
    topology = HardwareTopology.from_dict(_read(evaluation_case, "topology"))
    scenario = MemoryScenario.from_dict(_read(evaluation_case, "scenario"))
    target = _read(evaluation_case, "target")
    if (topology.fingerprint() != evaluation["topology_sha256"] or scenario.fingerprint() != evaluation["scenario_sha256"]
            or scenario.workload_sha256 != evaluation["workload_sha256"] or scenario.schedule_sha256 != evaluation["schedule_sha256"]
            or fingerprint(target) != scenario.target_sha256 or scenario.target_sha256 != evaluation["target_sha256"]
            or target["pool"] != dp2.POOL or target["attention_dp_size"] != 2
            or target["allowed_nodes"] != dp2.ALLOWED or target["native_host"] != dp2.HOST):
        raise ValueError("evaluation target differs from the fixed DP2 case contract")
    preview = dp2._preview(topology, scenario, target["gpu_bindings"], program, backing.LOCAL_HBM)
    selected = [(action["rank"], action["layer"]) for action in preview["actions"] if action["node"] != dp2.HOST]
    if selected != list(enumerate(hottest)):
        raise ValueError("the compiled hand-written control did not select the tuned layers")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    save(output / "manual_tuned_program.json", program)
    save(output / "manual_tuned_plan_preview.json", preview)
    report = {"schema": 1, "status": "manual_tuned_unverified", "scope": "native calibration warmup and replay cold miss rows",
              "objective_proxy": "maximize observed native selected miss rows for one local slab per rank; no serving timing",
              "calibration_workload_sha256": training["workload_sha256"], "calibration_result_sha256": sha256(raw_result).hexdigest(),
              "calibration_kv_log_sha256_by_rank": log_hashes, "miss_rows_by_rank_and_layer": counts,
              "chosen_layers_by_rank": hottest, "evaluation_workload_sha256": evaluation["workload_sha256"],
              "evaluation_scenario_sha256": evaluation["scenario_sha256"], "program_sha256": preview["program_sha256"],
              "compiled_plan_sha256": fingerprint(preview)}
    save(output / "manual_tuned_scope.json", report)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "calibration-case", "calibration-result", "evaluation-case"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(tune(args.output, calibration_case=args.calibration_case, calibration_result=args.calibration_result,
                          evaluation_case=args.evaluation_case), sort_keys=True))


if __name__ == "__main__":
    main()
