"""Trusted process entry point for real CUDA block-attention evaluations."""

from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
from statistics import median
import time

from rsi_memory.memory_program import MemoryProgram
from rsi_memory.scenarios import ScenarioWorkload
from rsi_memory.topology import HardwareTopology, fingerprint
from .torch_runtime import InfeasiblePlan, run_trial
from .torch_workload import compile_workload


def execution_signature(row):
    return fingerprint({"actions": row["actions"], "semantics": row["semantics"],
                        "selections": [[a["step"], a["session"], a["layer"], a["selected"]]
                                       for a in row["accesses"]],
                        "link_bytes": row["link_bytes"], "program": row["program_sha256"],
                        "topology": row["topology_sha256"], "scenario": row["scenario_sha256"],
                        "reserved": row["managed_reserved_bytes_by_node"],
                        "live_references_at_end": row["live_references_at_end"],
                        "residual": row["residual_before_teardown_bytes_by_node"]})


def evaluate(payload):
    required = {"schema", "arm", "program", "cases", "warmup", "samples", "expected_gpu_uuids"}
    if type(payload) is not dict or payload.keys() != required or payload["schema"] != 1:
        raise ValueError("invalid trusted CUDA evaluation request")
    if (type(payload["warmup"]) is not int or not 1 <= payload["warmup"] <= 10
            or type(payload["samples"]) is not int or not 1 <= payload["samples"] <= 100
            or type(payload["cases"]) is not list or not 1 <= len(payload["cases"]) <= 32):
        raise ValueError("invalid fixed evaluation budget")
    affinities = [case["target"].get("host_cpu_affinity") for case in payload["cases"]]
    if any(value != affinities[0] for value in affinities):
        raise ValueError("a process cannot change its host NUMA input between cases")
    if affinities[0] is not None:
        if not set(affinities[0]) <= os.sched_getaffinity(0):
            raise ValueError("fixed host affinity is unavailable")
        os.sched_setaffinity(0, set(affinities[0]))
    import torch

    hardware = [{"ordinal": i, "name": torch.cuda.get_device_name(i),
                 "uuid": str(torch.cuda.get_device_properties(i).uuid),
                 "capability": list(torch.cuda.get_device_capability(i))}
                for i in range(torch.cuda.device_count())]
    normalize = lambda x: str(x).lower().removeprefix("gpu-")
    if [normalize(row["uuid"]) for row in hardware] != [normalize(x) for x in payload["expected_gpu_uuids"]]:
        raise ValueError("CUDA visible ordinal to physical UUID mapping differs from the fixed target")
    if any(case["target"].get("torch_version", str(torch.__version__)) != str(torch.__version__)
           for case in payload["cases"]):
        raise ValueError("PyTorch version differs from the calibrated fixed target")
    program = MemoryProgram.from_dict(payload["program"]) if payload["program"] is not None else None
    reports = []
    started = time.monotonic()
    try:
        for case in payload["cases"]:
            if case.keys() != {"name", "workload", "compile", "topology", "target", "compute_node", "device_map"}:
                raise ValueError("invalid immutable CUDA case")
            topology = HardwareTopology.from_dict(case["topology"])
            packed = compile_workload(ScenarioWorkload.from_dict(case["workload"]), **case["compile"])
            scenario = packed.bind_scenario(fingerprint(case["target"]), case["compute_node"])
            arguments = {"arm": payload["arm"], "device_map": case["device_map"]}
            diagnostic = run_trial(topology, scenario, packed, program, verify=True, collect_work=True, **arguments)
            for _ in range(payload["warmup"]):
                run_trial(topology, scenario, packed, program, verify=False, **arguments)
            gc.collect()
            timings = []
            expected_signature = execution_signature(diagnostic)
            for _ in range(payload["samples"]):
                row = run_trial(topology, scenario, packed, program, verify=False, **arguments)
                if "work_counts" in row:
                    raise RuntimeError("timed CUDA trial emitted diagnostic work counts")
                signature = execution_signature(row)
                if (row["schedule_sha256"] != diagnostic["schedule_sha256"]
                        or row["workload_sha256"] != diagnostic["workload_sha256"]
                        or signature != expected_signature):
                    raise RuntimeError("timing workload differs from its verified workload")
                timings.append({**{key: value for key, value in row.items()
                                   if key not in {"actions", "semantics", "page_activity", "accesses"}},
                                "execution_signature": signature, "action_count": len(row["actions"]),
                                "attention_windows_ns": [access["access_ns"] for access in row["accesses"]]})
            reports.append({"name": case["name"], "case_sha256": fingerprint(case),
                            "median_elapsed_ns": median(row["elapsed_ns"] for row in timings),
                            "execution_signature": execution_signature(diagnostic),
                            "diagnostic": diagnostic, "timings": timings})
    except (InfeasiblePlan, AssertionError) as exc:
        return {"schema": 1, "feasible": False, "score": None, "error_type": type(exc).__name__,
                "error": str(exc), "scope": "cuda_text_derived_fixed_block_sparse",
                "hardware": hardware, "cases": reports, "elapsed_seconds": time.monotonic()-started}
    return {"schema": 1, "feasible": True, "score": sum(row["median_elapsed_ns"] for row in reports) / 1e6,
            "score_unit": "sum_case_median_ms", "scope": "cuda_text_derived_fixed_block_sparse",
            "hardware": hardware, "cases": reports, "elapsed_seconds": time.monotonic()-started}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = evaluate(json.loads(args.input.read_text()))
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
