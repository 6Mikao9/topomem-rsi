"""Freeze and run the v11 single-GPU public engineering calibration."""

import argparse
from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import sys


ARMS = (("native", "native_lru"), ("lru-equivalent", "generated"),
        ("manual-direct", "manual_topology"), ("writeback", "generated"))
BASE_ORDER = (0, 1, 3, 2)
V10_PREREG = "556ff90e8aa7bad3ab6c26acfe145aeb37a0a5d0d0eeaaeffb12a03b45919451"
V10_AUDIT = "58af16d2f1e18f21d472ec815f2ebaf8655a101b7ac108d02c7819300b4c3b7b"
BASE_PAYLOAD = "de46dc63e16c038773e80c861700965e41b05fbabb0850474ded0efedc8916f0"
PROFILE = "dc575e2fa50211554a9f91d7ce75260b1cf56223fc4bec3daa8500721fde46e4"
SOURCE = "944183b276b231c2dd306dbfc9ddbbf9823d94875e21a94b76008ef2b91ba0a8"


def read(path, expected=None):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"frozen input digest differs for {path.name}")
    return json.loads(raw), digest


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def schedule():
    return [{"block": block, "arm": ARMS[(index+block) % len(ARMS)][0]}
            for block in range(8) for index in BASE_ORDER]


def prepare(args):
    prereg, _ = read(args.v10_preregistration, V10_PREREG)
    if (prereg["arms"] != [name for name, _ in ARMS] or prereg["base_order"] != list(BASE_ORDER)
            or (prereg["blocks"], prereg["warmup"], prereg["samples"]) != (8, 3, 5)):
        raise ValueError("v10 preregistration differs from the replicated qualification design")
    _, source_sha = read(args.source_manifest, SOURCE)
    audit, _ = read(args.v10_audit, V10_AUDIT)
    payload, _ = read(args.base_payload, BASE_PAYLOAD)
    profile, profile_sha = read(args.profile, PROFILE)
    if profile["host_cpu_affinity"] != [64] or len(profile["devices"]) != 1:
        raise ValueError("qualification requires the previously measured GPU3 and CPU64 route")
    device = profile["devices"][0]
    uuid = "GPU-" + device["uuid"].lower().removeprefix("gpu-")
    fits = {row["direction"]: row for row in profile["affine_link_fits"]}
    payload.update(expected_gpu_uuids=[uuid], samples=5, warmup=3)
    for case in payload["cases"]:
        case["compile"]["head_dim"] = 64
        graph = case["topology"]
        graph["nodes"] = [node for node in graph["nodes"] if node["id"] != "gpu1"]
        graph["links"] = [link for link in graph["links"] if "gpu1" not in (link["src"], link["dst"])]
        for node in graph["nodes"]:
            if node["kind"] == "gpu":
                node["physical_bytes"] = device["total_memory"]
            node["evidence_sha256"] = profile_sha
        for link in graph["links"]:
            observed = fits["h2d" if link["dst"] == "gpu0" else "d2h"]
            link.update(bandwidth_bytes_per_s=observed["bandwidth_bytes_per_s"], latency_ns=observed["latency_ns"],
                        evidence_sha256=profile_sha)
        case.update(device_map={"gpu0": 0}, target={"scope": "public-training-geometry-qualification-only",
                    "gpu_uuid": uuid, "host_cpu_affinity": [64], "torch_version": profile["torch_version"],
                    "profile_sha256": profile_sha, "head_dim": 64})
    reference = audit["blocks"][0]["arms"]
    expected_cases = {name: row["case_sha256"] for name, row in reference["native"]["cases"].items()}
    if {case["name"]: fingerprint(case) for case in payload["cases"]} != expected_cases:
        raise ValueError("rebuilt case content differs from actual v10 qualification receipts")
    paths = {"lru-equivalent": args.lru_policy, "writeback": args.writeback_policy}
    arms = {}
    for name, arm in ARMS:
        program, raw_sha = read(paths[name]) if name in paths else (None, None)
        program_sha = None if program is None else fingerprint(program)
        if any(row["program_sha256"] != program_sha for row in reference[name]["cases"].values()):
            raise ValueError(f"policy differs from the v10 qualification for {name}")
        arms[name] = {"arm": arm, "program": program, "program_file_sha256": raw_sha}
    plan = {"schema": 1, "kind": "v11_single_gpu3_public_engineering_calibration", "source_manifest_sha256": source_sha,
            "script_sha256": sha256(Path(__file__).read_bytes()).hexdigest(), "v10_preregistration_sha256": V10_PREREG,
            "v10_audit_sha256": V10_AUDIT, "base_payload_sha256": BASE_PAYLOAD, "calibration_sha256": PROFILE,
            "gpu_indices": [3], "host_cpu_affinity": [64], "job_timeout_seconds": 600, "case_sha256": expected_cases,
            "public_training_only": True, "heldout": False, "payload": payload, "arms": arms, "schedule": schedule()}
    raw = (json.dumps(plan, sort_keys=True, indent=2, allow_nan=False)+"\n").encode()
    with args.output.open("xb") as stream:
        stream.write(raw)
    print(json.dumps({"plan_sha256": sha256(raw).hexdigest(), "source_manifest_sha256": source_sha,
                      "script_sha256": plan["script_sha256"], "jobs": len(plan["schedule"])}))


def execute(args):
    plan, _ = read(args.plan, args.expected_plan_sha256)
    _, source_sha = read(args.source_manifest, SOURCE)
    if (plan["kind"] != "v11_single_gpu3_public_engineering_calibration" or plan["source_manifest_sha256"] != source_sha
            or plan["script_sha256"] != sha256(Path(__file__).read_bytes()).hexdigest() or plan["schedule"] != schedule()):
        raise ValueError("execution differs from the frozen engineering plan")
    config, _ = read(args.executor_config)
    if (type(config) is not dict or config.keys() != {"host", "worker", "root", "python", "source", "gpu_indices",
                                                    "gpu_uuids", "environment"}
            or config["gpu_indices"] != plan["gpu_indices"] or config["gpu_uuids"] != plan["payload"]["expected_gpu_uuids"]
            or config["environment"] != {"OMP_NUM_THREADS": "2"}):
        raise ValueError("operator executor differs from the frozen GPU3 qualification")
    sys.path.insert(0, str(args.controller_root.resolve(strict=True)))
    from rsi_memory.execution_client import PersistentExecutor

    args.audit_directory.mkdir(exist_ok=False, parents=True)
    with PersistentExecutor(**config, source_manifest_sha256=source_sha, audit_directory=args.audit_directory) as executor:
        for row in plan["schedule"]:
            label, block = row["arm"], row["block"]
            arm = plan["arms"][label]
            payload = deepcopy(plan["payload"])
            payload.update(arm=arm["arm"], program=arm["program"])
            job = f"qualification-v11-paired-gpu3-b{block}-h64-{label}"
            receipt = executor.run(job, "integrations.topology_memory.torch_eval", payload, timeout=plan["job_timeout_seconds"])
            result = receipt.get("result", {})
            print(json.dumps({"job": job, "status": receipt["status"], "feasible": result.get("feasible"),
                              "score": result.get("score")}), flush=True)
            actual = {case["name"]: case["case_sha256"] for case in result.get("cases", [])}
            if receipt["status"] != "completed" or result.get("feasible") is not True or actual != plan["case_sha256"]:
                raise RuntimeError(f"inspect immutable failed qualification receipt {job}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    phases = parser.add_subparsers(dest="phase", required=True)
    prep, run = phases.add_parser("prepare"), phases.add_parser("run")
    for flag in ("v10-preregistration", "v10-audit", "base-payload", "profile", "source-manifest", "lru-policy",
                 "writeback-policy", "output"):
        prep.add_argument(f"--{flag}", type=Path, required=True)
    for flag in ("plan", "source-manifest", "executor-config", "controller-root", "audit-directory"):
        run.add_argument(f"--{flag}", type=Path, required=True)
    run.add_argument("--expected-plan-sha256", required=True)
    args = parser.parse_args()
    (prepare if args.phase == "prepare" else execute)(args)


if __name__ == "__main__":
    main()
