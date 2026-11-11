"""Audit the 32 v11 single-device receipts against the frozen v10 design."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from build_v10_gpu3_calibration import ARMS, build


V10_AUDIT = "58af16d2f1e18f21d472ec815f2ebaf8655a101b7ac108d02c7819300b4c3b7b"
V11_PLAN = "8da4c6d303f22133e5084f47dd39a0cc970d26f3faaa4b8375a1f93c63764390"
V11_SOURCE = "944183b276b231c2dd306dbfc9ddbbf9823d94875e21a94b76008ef2b91ba0a8"


def read(path, expected):
    raw = path.read_bytes()
    if sha256(raw).hexdigest() != expected:
        raise ValueError(f"frozen digest differs {path.name}")
    return json.loads(raw)


def audit(receipts, preregistration, previous):
    plan = read(preregistration, V11_PLAN)
    v10 = read(previous, V10_AUDIT)
    v11 = build(receipts, "v11")
    if v11["source_manifest_sha256"] != V11_SOURCE or V11_SOURCE != plan["source_manifest_sha256"]:
        raise ValueError("engineering receipts differ from the preregistered single-device source")
    schedule = [{"block": index, "arm": label} for index, block in enumerate(v11["blocks"]) for label in block["order"]]
    if schedule != plan["schedule"]:
        raise ValueError("completed sequence differs from the frozen Williams schedule")
    equal_fields = ("case_sha256", "workload_sha256", "schedule_sha256", "scenario_sha256", "topology_sha256",
                    "selected_output_sha256", "execution_signature", "program_sha256", "attention_calls", "link_bytes",
                    "total_fetch_misses", "managed_reserved_bytes_by_node", "managed_peak_bytes_by_node")
    for current, past in zip(v11["blocks"], v10["blocks"]):
        if current["order"] != past["order"]:
            raise ValueError("v10 and v11 block orders differ")
        for arm in ARMS:
            fresh = current["arms"][arm]
            prior = past["arms"][arm]
            if fresh["gpu_uuid_sha256"] != prior["gpu_uuid_sha256"]:
                raise ValueError("v10 and v11 physical GPU identities differ")
            for case, row in fresh["cases"].items():
                if (row["case_sha256"] != plan["case_sha256"][case] or
                        any(row[key] != prior["cases"][case][key] for key in equal_fields)):
                    raise ValueError(f"v11 changed the v10 fixed input or execution {arm}/{case}")
    compare = {}
    for name in v11["scopes"]:
        def time(block, arm):
            row = block["arms"][arm]
            return row["aggregate_ms"] if name == "aggregate" else row["cases"][name]["median_ms"]
        gaps = {arm: [] for arm in ARMS}
        for current, past in zip(v11["blocks"], v10["blocks"]):
            for arm in ARMS:
                gaps[arm].append(time(past, arm)-time(current, arm))
        compare[name] = {arm: {"v10_minus_v11_ms_by_same_design_block": gaps[arm],
                              "descriptive_median_difference_ms": median(gaps[arm])} for arm in ARMS}
    v11.update(preregistration_sha256=V11_PLAN, v10_audit_sha256=V10_AUDIT,
               matches_v10_case_program_selected_output_actions_traffic_misses_and_managed_peaks=True,
               version_comparison={"scope": "separate chronological engineering runs; descriptive only",
                                   "scopes": compare})
    return v11


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("receipts", "preregistration", "v10-audit", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.receipts, args.preregistration, args.v10_audit)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(json.dumps(result, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
