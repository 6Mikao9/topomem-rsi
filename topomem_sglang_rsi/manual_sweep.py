"""Freeze a bounded, training-only hand-policy sweep for the qualified SGLang DP2 target."""

import argparse
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median

from integrations.topology_memory.sglang_backing import HOST, LOCAL_HBM
from integrations.topology_memory.sglang_observe import save
from rsi_memory.execution_client import ExecutionError
from rsi_memory.memory_program import MemoryProgram
from rsi_memory.topology import fingerprint

from .contract import checked_id, read_json
from .domain import SGLangDomain
from .hot_config import resident_control_program, stock_control_program


REFERENCE = "stock-6144-host"
DEFAULT = "stock-4096-host"
RESIDENT = "resident-6144"
LAYER_COUNT = 6


def hand_programs():
    stock = stock_control_program()
    default = stock.to_dict()
    default["rules"].pop(0)
    programs = {DEFAULT: (None, MemoryProgram.from_dict(default)), REFERENCE: (None, stock)}
    local = {"op": "feature", "name": "target_is_local"}
    host = {"op": "feature", "name": "target_is_host"}
    layer = {"op": "feature", "name": "layer"}

    def const(value):
        return {"op": "const", "value": value}

    def expr(op, *args):
        return {"op": op, "args": list(args)}

    for index in range(LAYER_COUNT):
        delta = expr("sub", layer, const(index))
        local_score = expr("if_lt", expr("mul", delta, delta), const(1), const(-1), const(1))
        raw = stock.to_dict()
        raw["rules"][1] = {"event": "allocate", "action": "place", "when": expr("add", local, host),
                           "score": expr("mul", local, local_score)}
        programs[f"local-layer-{index}"] = (index, MemoryProgram.from_dict(raw))
    programs[RESIDENT] = (None, resident_control_program())
    return programs


def prepare(adapter, sweep_id, blocks, minimum_gain_ms):
    checked_id(sweep_id)
    if type(blocks) is not int or not 3 <= blocks <= 8 or not math.isfinite(minimum_gain_ms) or minimum_gain_ms <= 0:
        raise ValueError("require 3–8 predeclared blocks and a finite positive practical margin")
    programs = hand_programs()
    cases = adapter.cases["train"]
    policies = {}
    for label, (layer, program) in programs.items():
        resident = label == RESIDENT
        row = {"program": program.to_dict(), "program_sha256": program.fingerprint(), "local_layer_on_each_rank": layer,
               "backing_mode": "resident" if resident else "tiered"}
        policies[label] = row
        try:
            compiled_by_case = {case["id"]: adapter._compile(program, case=case) for case in cases}
        except (ValueError, MemoryError) as error:
            if not resident and (layer is None or isinstance(error, ValueError)):
                raise
            row.update(status="capacity_rejected", reason=str(error))
            continue
        tiers = ["resident"] * LAYER_COUNT if resident else [LOCAL_HBM if layer == index else HOST for index in range(LAYER_COUNT)]
        slots = 4096 if label == DEFAULT else 6144
        for compiled in compiled_by_case.values():
            if (len(compiled.workers) != 2 or compiled.audit["startup"]["hot_slots"] != slots
                    or compiled.audit["startup"]["backing_mode"] != row["backing_mode"]
                    or any(len(worker.layer_tiers) != LAYER_COUNT for worker in compiled.workers)):
                raise ValueError("hand policies require the measured two-rank, six-layer target")
        if not resident and any(list(worker.layer_tiers) != tiers for compiled in compiled_by_case.values() for worker in compiled.workers):
            if layer is None:
                raise ValueError("stock hand policy did not lower to the expected all-host configuration")
            row.update(status="capacity_rejected", reason="the requested pair of local slabs did not fit")
            continue
        row.update(status="preflight_passed", compilation=compiled_by_case[cases[0]["id"]].audit,
                   compilation_by_case={name: compiled.audit for name, compiled in compiled_by_case.items()},
                   arm="native" if layer is None and not resident else "manual",
                   effective_sha256=adapter.effective_program_fingerprint(program),
                   layer_tiers_by_rank=[tiers] * 2)
    labels = [label for label, row in policies.items() if row["status"] == "preflight_passed"]
    if len({policies[label]["effective_sha256"] for label in labels}) != len(labels):
        raise ValueError("distinct hand policies must compile to distinct physical configurations")
    count = len(labels)
    offset = int(fingerprint(sweep_id)[:8], 16) % count
    base = [0] + [(index + 1) // 2 if index % 2 else count - index // 2 for index in range(1, count)]
    order = [[labels[(offset + shift + index) % count] for index in base] for shift in range(count)]
    if count % 2:
        order.extend([list(reversed(row)) for row in order])
    return {"schema": 1, "scope": "train-only independent hand-policy sweep", "sweep_id": sweep_id,
            "source_manifest_sha256": adapter.source, "launch_sha256": adapter.launch,
            "tuner_source_sha256": sha256(Path(__file__).read_bytes()).hexdigest(),
            "adapter_sources_sha256": adapter.contract["adapter_sources_sha256"],
            "metadata_profile_sha256": fingerprint(adapter.profile), "cases": cases,
            "training_tokens_sha256": fingerprint({case["id"]: adapter.token_inputs[case["id"]] for case in cases}),
            "reference": REFERENCE, "minimum_paired_median_gain_ms": minimum_gain_ms, "policies": policies,
            "declared_policy_order": list(policies),
            "order": [order[index % len(order)] for index in range(blocks)], "complete_order_cycle": len(order),
            "max_gpu_jobs": len(cases) * (4 + count - 2 + blocks * count)}


def execute(adapter, manifest, output):
    cases = manifest["cases"]
    if cases != adapter.cases["train"]:
        raise ValueError("hand sweep cannot open validation or test workloads")
    proofs, programs, failures = {}, {}, {}
    for case in cases:
        for label, slots in ((DEFAULT, 4096), (REFERENCE, 6144)):
            native_id, native = adapter._parity(case, "native", None, slots=slots)
            proofs[label, case["id"]] = adapter._proof(native_id, native)
            _event(output, {"case": case["id"], "phase": "native_aa", "label": label, "hot_slots": slots, "job_id": native_id,
                            "repeat_job_id": adapter.native_repeats[case["id"], slots], "raw_result_sha256": native["raw_result_sha256"]})
    for label, row in manifest["policies"].items():
        if row["status"] != "preflight_passed":
            continue
        if row["arm"] == "native":
            programs[label] = None
            continue
        program = programs[label] = MemoryProgram.from_dict(row["program"])
        for case in cases:
            job_id, parity = adapter._parity(case, "manual", program)
            _event(output, {"case": case["id"], "label": label, "phase": "parity", "job_id": job_id,
                            "feasible": parity["feasible"], "raw_result_sha256": parity["raw_result_sha256"],
                            "failure": parity.get("failure")})
            if not parity["feasible"]:
                failures[label] = {"job_id": job_id, "failure": parity["failure"]}
                break
            proofs[label, case["id"]] = adapter._proof(job_id, parity)
    samples = {label: [] for label in programs if label not in failures}
    identity = {"kind": "hand-sweep-v1", "sweep_sha256": fingerprint(manifest)}
    for block, order in enumerate(manifest["order"], 1):
        for label in order:
            if label in failures:
                continue
            scores = []
            arm = manifest["policies"][label]["arm"]
            for case in cases:
                job_id, result = adapter._job(case, arm, "performance", programs[label],
                                              {**identity, "block": block, "label": label}, proofs[label, case["id"]],
                                              native_hot=6144 if label == REFERENCE else None)
                score = result["score"]
                if not result["feasible"] or type(score) not in (int, float) or not math.isfinite(score) or score <= 0:
                    raise ExecutionError("qualified timed hand policy lacks a measured replay window " + job_id)
                scores.append(score)
                _event(output, {"case": case["id"], "label": label, "phase": "performance", "block": block,
                                "job_id": job_id, "window_ms": score, "raw_result_sha256": result["raw_result_sha256"]})
            samples[label].append(sum(scores))
    gains = {label: [baseline - current for baseline, current in zip(samples[REFERENCE], values, strict=True)]
             for label, values in samples.items()}
    eligible = [label for label, values in gains.items() if min(values) > 0 and median(values) >= manifest["minimum_paired_median_gain_ms"]]
    selected = min(eligible, key=lambda label: (-median(gains[label]), label)) if eligible else REFERENCE
    return {"schema": 1, "status": "training_selection_only", "sweep_sha256": identity["sweep_sha256"],
            "parity_failures": failures, "selected": selected, "selected_program_sha256": manifest["policies"][selected]["program_sha256"],
            "sum_window_ms_by_block": samples, "paired_gain_over_stock_6144_ms_by_block": gains,
            "eligible_to_displace_stock_6144": eligible}


def _event(output, row):
    with (output / "jobs.jsonl").open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, sort_keys=True, allow_nan=False) + "\n")


def sweep(config, output, *, sweep_id, blocks, minimum_gain_ms, expected_sha256=None):
    config, output = Path(config), Path(output)
    adapter = SGLangDomain(read_json(config), config.parent)
    try:
        manifest = prepare(adapter, sweep_id, blocks, minimum_gain_ms)
        manifest_sha = fingerprint(manifest)
        if expected_sha256 is not None and expected_sha256 != manifest_sha:
            raise ValueError("hand sweep differs from the reviewed preflight manifest")
        output.mkdir(parents=True, exist_ok=False)
        save(output / "manifest.json", manifest)
        if expected_sha256 is None:
            return {"status": "preflight_only", "max_gpu_jobs": manifest["max_gpu_jobs"], "sweep_sha256": manifest_sha}
        try:
            result = execute(adapter, manifest, output)
        except Exception as error:
            save(output / "result.json", {"schema": 1, "status": "failed", "sweep_sha256": manifest_sha, "error": str(error)})
            raise
        save(output / "selected_program.json", manifest["policies"][result["selected"]]["program"])
        save(output / "result.json", result)
        return result
    finally:
        adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sweep-id", required=True)
    parser.add_argument("--blocks", type=int, default=3)
    parser.add_argument("--minimum-gain-ms", type=float, required=True)
    parser.add_argument("--execute", metavar="PREFLIGHT_SHA256", help="launch the training-only jobs bound to the reviewed preflight digest")
    args = parser.parse_args()
    result = sweep(args.adapter_config, args.output, sweep_id=args.sweep_id, blocks=args.blocks,
                   minimum_gain_ms=args.minimum_gain_ms, expected_sha256=args.execute)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
