"""Audit the 32 public v10 single-GPU engineering qualification receipts."""

import argparse
from collections import Counter
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median


ARMS = ("native", "lru-equivalent", "manual-direct", "writeback")
SCOPE = "cuda_text_derived_fixed_block_sparse"


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def execution_signature(row):
    # JSON serializes fork ownership keys as strings. Restore the runtime's integer-key sort order.
    semantics = [{**event, "owners_after": {int(key): count for key, count in event["owners_after"].items()}}
                 if "owners_after" in event else event for event in row["semantics"]]
    return digest({"actions": row["actions"], "semantics": semantics,
                   "selections": [[a["step"], a["session"], a["layer"], a["selected"]] for a in row["accesses"]],
                   "link_bytes": row["link_bytes"], "program": row["program_sha256"], "topology": row["topology_sha256"],
                   "scenario": row["scenario_sha256"], "reserved": row["managed_reserved_bytes_by_node"],
                   "live_references_at_end": row["live_references_at_end"], "residual": row["residual_before_teardown_bytes_by_node"]})


def load(path, arm):
    raw = path.read_bytes()
    receipt = json.loads(raw)
    result = receipt["result"]
    if (receipt["status"] != "completed" or receipt["returncode"] != 0 or receipt["timeout"]
            or receipt["failure"] is not None or receipt["environment_conflict"] or receipt["process_group_cleanup_required"]
            or receipt["process_group_remaining"] or receipt["remaining_compute_pids"] != ""
            or receipt["gpu_indices"] != [3] or len(receipt["gpu_uuids"]) != 1
            or len(result["hardware"]) != 1 or not result["feasible"] or result["scope"] != SCOPE
            or digest(result) != receipt["result_canonical_sha256"]):
        raise ValueError(f"invalid engineering receipt {path.name}")
    identity = receipt["gpu_uuids"][0].lower().removeprefix("gpu-")
    hardware = result["hardware"][0]
    if hardware["uuid"].lower().removeprefix("gpu-") != identity or hardware["ordinal"] != 0:
        raise ValueError("runtime GPU differs from the leased qualification device")
    cases = {}
    expected_arm = {"native": "native_lru", "manual-direct": "manual_topology"}.get(arm, "generated")
    for case in result["cases"]:
        diag = case["diagnostic"]
        selected = [{key: access[key] for key in ("step", "layer", "selected", "selected_sha256", "output_sha256")}
                    for access in diag["accesses"]]
        sig = execution_signature(diag)
        timings = case["timings"]
        if (diag["status"] != "verified" or diag["arm"] != expected_arm or diag["live_references_at_end"] != 0
                or not selected or digest(selected) != diag["selected_and_output_sha256"] or sig != case["execution_signature"]
                or "work_counts" not in diag or len(timings) != 5 or case["name"] in cases
                or median(row["elapsed_ns"] for row in timings) != case["median_elapsed_ns"]
                or any(row["status"] != "timed" or "work_counts" in row or row["execution_signature"] != sig
                       or row["link_bytes"] != diag["link_bytes"] or row["total_fetch_misses"] != diag["total_fetch_misses"]
                       for row in timings)):
            raise ValueError(f"invalid selected data, work signature or timing in {path.name}")
        cases[case["name"]] = {
            "case_sha256": case["case_sha256"], "workload_sha256": diag["workload_sha256"],
            "schedule_sha256": diag["schedule_sha256"], "scenario_sha256": diag["scenario_sha256"],
            "topology_sha256": diag["topology_sha256"], "selected_output_sha256": digest(selected),
            "execution_signature": sig, "program_sha256": diag["program_sha256"],
            "attention_calls": len(selected), "median_ms": case["median_elapsed_ns"]/1e6,
            "timings_ms": [row["elapsed_ns"]/1e6 for row in timings], "link_bytes": diag["link_bytes"],
            "total_fetch_misses": diag["total_fetch_misses"],
            "managed_reserved_bytes_by_node": diag["managed_reserved_bytes_by_node"],
            "managed_peak_bytes_by_node": diag["managed_peak_bytes_by_node"],
            "cuda_peak_tensor_bytes_delta_by_node": diag["cuda_peak_allocated_bytes_delta_by_node"],
            "work_counts": diag["work_counts"],
        }
    if len(cases) != 4 or not math.isclose(result["score"], sum(case["median_ms"] for case in cases.values()), rel_tol=1e-12):
        raise ValueError("qualification aggregate disagrees with fixed cases")
    output = {"receipt_file": path.name, "receipt_sha256": sha256(raw).hexdigest(), "request_sha256": receipt["request_sha256"],
              "source_manifest_sha256": receipt["source_manifest_sha256"], "raw_result_sha256": receipt["result_sha256"],
              "canonical_result_sha256": receipt["result_canonical_sha256"], "utc_start": receipt["utc_start"],
              "utc_end": receipt["utc_end"], "gpu_uuid_sha256": digest(identity), "gpu_index": 3,
              "hardware": {key: value for key, value in hardware.items() if key != "uuid"},
              "completed_and_compute_processes_empty_afterward": True, "aggregate_ms": result["score"], "cases": cases}
    return output, (receipt["pid"], receipt["proc_starttime"])


def build(root, version="v10"):
    blocks, processes = [], set()
    for block in range(8):
        receipts = {}
        for arm in ARMS:
            path = root / f"qualification-{version}-paired-gpu3-b{block}-h64-{arm}.json"
            row, process = load(path, arm)
            if process in processes or not process[1]:
                raise ValueError("missing or repeated process identity")
            processes.add(process)
            receipts[arm] = row
        order = sorted(ARMS, key=lambda arm: receipts[arm]["utc_start"])
        if any(receipts[a]["utc_end"] > receipts[b]["utc_start"] for a, b in zip(order, order[1:])):
            raise ValueError("qualification processes overlapped")
        blocks.append({"id": f"b{block}", "order": order, "arms": receipts})
    runs = [record for block in blocks for record in block["arms"].values()]
    source_hashes = {row["source_manifest_sha256"] for row in runs}
    if len(source_hashes) != 1 or len({row["gpu_uuid_sha256"] for row in runs}) != 1:
        raise ValueError("qualification mixed source or devices")
    for position in range(4):
        if Counter(block["order"][position] for block in blocks) != dict.fromkeys(ARMS, 2):
            raise ValueError("incomplete paired position balance")
    pairs = Counter(pair for block in blocks for pair in zip(block["order"], block["order"][1:]))
    if pairs != {(a, b): 2 for a in ARMS for b in ARMS if a != b}:
        raise ValueError("incomplete paired carryover balance")
    case_names = list(runs[0]["cases"])
    for name in case_names:
        identity = ("case_sha256", "workload_sha256", "schedule_sha256", "scenario_sha256", "topology_sha256",
                    "selected_output_sha256", "attention_calls", "managed_reserved_bytes_by_node")
        if any(len({digest(row["cases"][name][key]) for row in runs}) != 1 for key in identity):
            raise ValueError("qualification changed fixed case, data, or budget")
        for arm in ARMS:
            fields = ("execution_signature", "program_sha256", "link_bytes", "total_fetch_misses", "work_counts")
            if any(len({digest(block["arms"][arm]["cases"][name][key]) for block in blocks}) != 1 for key in fields):
                raise ValueError("within-arm qualification actions or counts drifted")
        for block in blocks:
            native, lru = (block["arms"][arm]["cases"][name] for arm in ("native", "lru-equivalent"))
            if any(native[key] != lru[key] for key in ("link_bytes", "total_fetch_misses")):
                raise ValueError("equivalent LRU did not preserve native traffic and misses")
    scopes = {name: describe(blocks, lambda row: row["cases"][name]["median_ms"]) for name in case_names}
    scopes["aggregate"] = describe(blocks, lambda row: row["aggregate_ms"])
    return {"schema": 1, "kind": "single_gpu_engineering_training_qualification", "scope": SCOPE,
            "heldout": False, "model_generated_outcome": False, "gpu_count": 1, "paired_blocks": len(blocks),
            "receipt_count": len(runs), "fresh_process_count": len(processes), "timed_samples_per_case_per_arm": 5,
            "position_and_carryover_balanced": True, "exact_case_selected_bytes_and_outputs_across_arms": True,
            "native_lru_link_bytes_and_misses_exact": True, "source_manifest_sha256": next(iter(source_hashes)),
            "environment_scope": "worker identity and pre/post device checks; no in-window occupancy or clock proof",
            "scopes": scopes, "blocks": blocks}


def describe(blocks, score):
    medians = {arm: median(score(block["arms"][arm]) for block in blocks) for arm in ARMS}
    comparisons = {}
    for arm in ARMS[1:]:
        gains = [score(block["arms"]["native"]) - score(block["arms"][arm]) for block in blocks]
        comparisons[arm] = {"native_minus_arm_ms_by_block": gains, "median_paired_gain_ms": median(gains),
                            "min_paired_gain_ms": min(gains), "max_paired_gain_ms": max(gains),
                            "positive_blocks": sum(gain > 0 for gain in gains)}
    return {"median_ms": medians, "descriptive_comparisons": comparisons}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = build(args.receipts)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
