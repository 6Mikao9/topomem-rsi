"""Read the explicitly named public qualification receipts and write their offline summary."""

import argparse
import csv
from hashlib import sha256
import json
import math
from pathlib import Path
from statistics import median


_POLICIES = ("lru-equivalent", "manual-direct", "manual-ir")
_SCOPE = "cuda_text_derived_fixed_block_sparse"


def digest(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def gpu_uuid(value):
    return value.lower().removeprefix("gpu-")


def environment_snapshot(raw, labels, uuids):
    rows = list(csv.reader(raw.splitlines(), skipinitialspace=True))
    if [gpu_uuid(row[0]) for row in rows] != uuids or any(len(row) != len(labels)+1 for row in rows):
        raise ValueError("GPU observation does not bind to the receipt's selected devices")
    return [dict(zip(labels, row[1:])) for row in rows]


def read_receipt(path):
    raw = path.read_bytes()
    receipt = json.loads(raw)
    result = receipt["result"]
    if (receipt["status"] != "completed" or receipt["returncode"] != 0 or receipt["timeout"]
            or receipt["failure"] is not None or receipt["environment_conflict"]
            or receipt["process_group_cleanup_required"] or receipt["process_group_remaining"]
            or receipt["remaining_compute_pids"] != "" or not result["feasible"] or result["scope"] != _SCOPE
            or digest(result) != receipt["result_canonical_sha256"]):
        raise ValueError(f"qualification receipt failed integrity checks: {path.name}")
    uuids = [gpu_uuid(value) for value in receipt["gpu_uuids"]]
    if [gpu_uuid(row["uuid"]) for row in result["hardware"]] != uuids:
        raise ValueError("runtime GPU identities disagree with the selected devices")
    evidence = {}
    for case in result["cases"]:
        diagnostic = case["diagnostic"]
        if diagnostic["status"] != "verified" or diagnostic["scope"] != _SCOPE or diagnostic["live_references_at_end"] != 0:
            raise ValueError("qualification does not verify the fixed case")
        if not diagnostic["selected_and_output_sha256"] or case["name"] in evidence:
            raise ValueError("qualification has a duplicate case or missing output evidence")
        times = [row["elapsed_ns"] for row in case["timings"]]
        if case["median_elapsed_ns"] != median(times):
            raise ValueError("qualification score does not match raw samples")
        if any(row.get("execution_signature", case["execution_signature"]) != case["execution_signature"]
               or row["managed_reserved_bytes_by_node"] != diagnostic["managed_reserved_bytes_by_node"]
               or row["live_references_at_end"] != 0 or row["link_bytes"] != diagnostic["link_bytes"]
               or row["total_fetch_misses"] != diagnostic["total_fetch_misses"] for row in case["timings"]):
            raise ValueError("timed qualification departed from its verified case")
        evidence[case["name"]] = {
            "case_sha256": case["case_sha256"], "arm": diagnostic["arm"], "fixture_kind": diagnostic["fixture_kind"],
            "program_sha256": diagnostic["program_sha256"], "topology_sha256": diagnostic["topology_sha256"],
            "scenario_sha256": diagnostic["scenario_sha256"], "workload_sha256": diagnostic["workload_sha256"],
            "schedule_sha256": diagnostic["schedule_sha256"], "execution_signature": case["execution_signature"],
            "selected_and_output_sha256": diagnostic["selected_and_output_sha256"],
            "ordered_attention_calls": len(diagnostic["accesses"]), "link_bytes": diagnostic["link_bytes"],
            "total_fetch_misses": diagnostic["total_fetch_misses"], "median_ms": case["median_elapsed_ns"]/1e6,
            "timing_samples_ms": [value/1e6 for value in times],
            "transfer_samples_ms": [row["transfer_wall_ns"]/1e6 for row in case["timings"]],
            "managed_peak_bytes_by_node": diagnostic["managed_peak_bytes_by_node"],
            "managed_reserved_bytes_by_node": diagnostic["managed_reserved_bytes_by_node"],
            "cuda_peak_tensor_bytes_delta_by_node": diagnostic["cuda_peak_allocated_bytes_delta_by_node"],
            "residual_before_teardown_bytes_by_node": diagnostic["residual_before_teardown_bytes_by_node"],
            "every_timing_records_execution_signature": all("execution_signature" in row for row in case["timings"]),
        }
    if not math.isclose(result["score"], sum(row["median_ms"] for row in evidence.values()), rel_tol=1e-12):
        raise ValueError("aggregate result differs from the sum of case medians")
    return {"receipt_file": path.name, "receipt_sha256": sha256(raw).hexdigest(),
            "request_sha256": receipt["request_sha256"], "source_manifest_sha256": receipt["source_manifest_sha256"],
            "recorded_raw_result_sha256": receipt["result_sha256"], "verified_canonical_result_sha256": digest(result),
            "utc_start": receipt["utc_start"], "utc_end": receipt["utc_end"],
            "environment": {"gpu_indices": receipt["gpu_indices"], "ordered_uuid_fingerprint": digest(uuids),
                            "hardware": [{k: v for k, v in row.items() if k != "uuid"} for row in result["hardware"]],
                            "before": environment_snapshot(receipt["before"], ("used_mib", "util_pct", "temperature_c", "power_w"), uuids),
                            "after": environment_snapshot(receipt["after"], ("used_mib", "util_pct"), uuids),
                            "completed_without_timeout_or_cleanup_residue": True, "remaining_compute_process_count": 0},
            "aggregate_ms": result["score"], "cases": evidence}


def traffic_same(a, b):
    return all(a[key] == b[key] for key in ("link_bytes", "total_fetch_misses"))


def build(v5_dir, v6_dir):
    v5 = {arm: read_receipt(v5_dir / f"qualification-v5-{arm}.json") for arm in ("native", *_POLICIES)}
    v6 = {arm: read_receipt(v6_dir / f"qualification-v6-{arm}.json") for arm in _POLICIES}
    aa = {f"{letter}{index}": read_receipt(v6_dir / f"qualification-v6-native-{letter}{index}.json")
          for letter in "ab" for index in range(1, 4)}
    order = [f"{letter}{index}" for index in range(1, 4) for letter in "ab"]
    if any(aa[before]["utc_end"] > aa[after]["utc_start"] for before, after in zip(order, order[1:])):
        raise ValueError("native repeat receipts do not form the declared independent A/B order")
    all_receipts = [*v5.values(), *v6.values(), *aa.values()]
    case_names = list(v5["native"]["cases"])
    identity_fields = ("case_sha256", "topology_sha256", "scenario_sha256", "workload_sha256", "schedule_sha256",
                       "selected_and_output_sha256", "managed_reserved_bytes_by_node", "ordered_attention_calls")
    if (any(list(row["cases"]) != case_names for row in all_receipts)
            or len({row["environment"]["ordered_uuid_fingerprint"] for row in all_receipts}) != 1
            or len({row["source_manifest_sha256"] for row in v5.values()}) != 1
            or len({row["source_manifest_sha256"] for row in (*v6.values(), *aa.values())}) != 1):
        raise ValueError("receipt set mixed fixed case, device, or version bindings")
    per_case = {}
    for name in case_names:
        a0 = aa["a1"]["cases"][name]
        if any(any(row["cases"][name][field] != a0[field] for field in identity_fields) for row in all_receipts):
            raise ValueError("qualification arms disagree on fixed inputs, selected bytes, or output")
        after = {"native": aa["a1"], **v6}
        if (not all(traffic_same(a0, row["cases"][name]) and row["cases"][name]["execution_signature"] == a0["execution_signature"]
                    for row in aa.values())
                or not all(traffic_same(old["cases"][name], after[arm]["cases"][name]) for arm, old in v5.items())
                or not all(old["cases"][name]["execution_signature"] == after[arm]["cases"][name]["execution_signature"]
                           and old["cases"][name]["program_sha256"] == after[arm]["cases"][name]["program_sha256"]
                           for arm, old in v5.items())
                or not traffic_same(a0, v6["lru-equivalent"]["cases"][name])
                or not traffic_same(v5["native"]["cases"][name], v5["lru-equivalent"]["cases"][name])):
            raise ValueError("a claimed same-action or cross-version transport path changed")
        per_case[name] = describe(v5, v6, aa, lambda row: row["cases"][name]["median_ms"])
    aggregate = describe(v5, v6, aa, lambda row: row["aggregate_ms"])
    return {"schema": 1, "kind": "public_engineering_calibration", "scope": _SCOPE,
            "comparison_kind": "descriptive_cross_snapshot_engineering_qualification_not_rsi_or_heldout",
            "paired_native_repeat_count": 3, "paired_native_execution_order": order, "case_and_output_bindings_exact": True,
            "native_and_lru_equivalent_wire_and_miss_counts_exact_in_each_snapshot": True,
            "each_arm_wire_and_miss_counts_exact_across_snapshots": True,
            "each_arm_execution_signature_and_program_exact_across_snapshots": True,
            "environment_scope": "selected GPU identity and clean worker pre/post checks; no in-window contamination or fixed clock proof",
            "raw_receipts": {"v5": v5, "v6_non_native": v6, "v6_native": aa}, "per_case": per_case, "aggregate": aggregate}


def describe(v5, v6, aa, score):
    pairs = [{"pair": i, "a_ms": score(aa[f"a{i}"]), "b_ms": score(aa[f"b{i}"]),
              "b_minus_a_ms": score(aa[f"b{i}"]) - score(aa[f"a{i}"])} for i in range(1, 4)]
    differences = [pair["b_minus_a_ms"] for pair in pairs]
    native = median(score(row) for row in aa.values())
    after = {"native": native, **{arm: score(row) for arm, row in v6.items()}}
    before = {arm: score(row) for arm, row in v5.items()}
    best = min(("native", "manual-direct", "manual-ir"), key=after.__getitem__)
    return {"v6_native_a_a": {"pairs": pairs, "median_signed_difference_ms": median(differences),
                              "median_absolute_difference_ms": median(map(abs, differences)),
                              "max_absolute_difference_ms": max(map(abs, differences)), "median_native_ms": native},
            "v5_ms": before, "v6_ms": after,
            "v5_minus_v6_ms": {arm: before[arm] - after[arm] for arm in before},
            "v6_ir_lru_to_native_latency_ratio": after["lru-equivalent"]/native,
            "v5_ir_lru_to_native_latency_ratio": before["lru-equivalent"]/before["native"],
            "v6_fastest_observed_non_generated_baseline": {"arm": best, "ms": after[best]}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v5", type=Path, required=True)
    parser.add_argument("--v6", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    summary = build(args.v5, args.v6)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(summary, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
