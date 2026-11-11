"""Parameterized derivative of the archived v10 engineering launch recipe."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
import sys


V10_PLAN = "556ff90e8aa7bad3ab6c26acfe145aeb37a0a5d0d0eeaaeffb12a03b45919451"
V11_PLAN = "8da4c6d303f22133e5084f47dd39a0cc970d26f3faaa4b8375a1f93c63764390"
ORIGINAL_RECIPE = "ca4aeafddb5bd52cc66df605bc0db1f4c647d3f505b272c57748dcc74311e3dc"
SOURCE = "68fe058da35c3b0517b6a2a769077693cea0774277ebdb8cd420958f4c774367"
MODULE = "integrations.topology_memory.torch_eval"


def read(path, expected=None):
    raw = path.read_bytes()
    if expected is not None and sha256(raw).hexdigest() != expected:
        raise ValueError("frozen input digest differs for " + path.name)
    return json.loads(raw)


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def verify_plan(v10_path, v11_path, source_path):
    v10 = read(v10_path, V10_PLAN)
    v11 = read(v11_path, V11_PLAN)
    read(source_path, SOURCE)
    if (v10["source_manifest_sha256"] != SOURCE or v10["script_sha256"] != ORIGINAL_RECIPE
            or v11["v10_preregistration_sha256"] != V10_PLAN or v11["public_training_only"] is not True
            or v11["heldout"] is not False or (v10["blocks"], v10["warmup"], v10["samples"]) != (8, 3, 5)):
        raise ValueError("public paired engineering preregistration differs")
    order = [{"block": block, "arm": v10["arms"][(index + block) % 4]}
             for block in range(v10["blocks"]) for index in v10["base_order"]]
    if order != v11["schedule"] or len(order) != 32:
        raise ValueError("the public v11 replay payload does not preserve v10 order")
    payload = v11["payload"]
    if ({case["name"]: fingerprint(case) for case in payload["cases"]} != v11["case_sha256"]
            or (payload["warmup"], payload["samples"]) != (3, 5)
            or any(case["compile"]["head_dim"] != 64 for case in payload["cases"])):
        raise ValueError("the exact public v10 case content is unavailable")
    return v11, order


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("v10-preregistration", "v11-preregistration", "source-manifest"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--execute", action="store_true")
    for key in ("controller-root", "executor-config", "audit-directory"):
        parser.add_argument(f"--{key}", type=Path)
    args = parser.parse_args()
    plan, order = verify_plan(args.v10_preregistration, args.v11_preregistration, args.source_manifest)
    if not args.execute:
        print(json.dumps({"status": "verified_reconstruction", "original_recipe_sha256": ORIGINAL_RECIPE,
                          "v10_source_manifest_sha256": SOURCE, "jobs": len(order)}))
        return
    if not (args.controller_root and args.executor_config and args.audit_directory):
        parser.error("execution requires a local controller, private executor config and new audit directory")
    config = read(args.executor_config)
    if (set(config) != {"host", "worker", "root", "python", "source", "gpu_indices", "gpu_uuids", "environment"}
            or config["gpu_indices"] != plan["gpu_indices"] or config["gpu_uuids"] != plan["payload"]["expected_gpu_uuids"]
            or config["environment"] != {"OMP_NUM_THREADS": "2"}):
        raise ValueError("executor does not match the frozen GPU3 engineering contract")
    sys.path.insert(0, str(args.controller_root.resolve(strict=True)))
    from rsi_memory.execution_client import PersistentExecutor

    args.audit_directory.mkdir(parents=True, exist_ok=False)
    with PersistentExecutor(**config, source_manifest_sha256=SOURCE, audit_directory=args.audit_directory) as executor:
        for row in order:
            label, block = row["arm"], row["block"]
            arm = plan["arms"][label]
            payload = {**plan["payload"], "arm": arm["arm"], "program": arm["program"]}
            job = f"qualification-v10-paired-gpu3-b{block}-h64-{label}"
            receipt = executor.run(job, MODULE, payload, timeout=plan["job_timeout_seconds"])
            print(json.dumps({"job_id": job, "status": receipt["status"], "score": receipt.get("result", {}).get("score")}), flush=True)
            if receipt["status"] != "completed":
                raise RuntimeError("engineering replay worker did not complete: " + job)


if __name__ == "__main__":
    main()
