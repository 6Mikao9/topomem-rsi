"""Pack the SGLang DP2 evaluator for its qualified CUDA 13 and isolated NCCL 2.32.3 environment."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


MEMBERS = (
    "sglang_executor_worker.py", "sglang_launch.py", "sglang_nccl_manifest.json", "integrations/__init__.py",
    *(f"rsi_memory/{name}.py" for name in ("__init__", "execution_worker", "memory_program", "scalar_program", "topology")),
    *(f"integrations/topology_memory/{name}.py" for name in ("__init__", "sglang_backing", "sglang_dependency", "sglang_observe",
                                                         "sglang_relay", "sglang_serving", "sglang_smoke", "sglang_sources",
                                                         "sglang_ingress_gate", "sglang_global_ticks", "sglang_resident", "sglang_resident_oracle")),
    *(f"topomem_sglang_rsi/{name}.py" for name in ("__init__", "contract", "collect_hot_profile", "hot_config", "sglang_eval")),
)


def pack(output, launch):
    root = Path(__file__).resolve().parent
    files = {name: (root / name).read_bytes() for name in MEMBERS}
    launch_raw = Path(launch).read_bytes()
    launch_config = json.loads(launch_raw)
    fields = {"schema", "python", "model", "nccl_isolate", "cuda_home", "cache_root", "gpu_indices", "gpu_uuids"}
    if (type(launch_config) is not dict or launch_config.keys() != fields or launch_config["schema"] != 2
            or launch_config["gpu_indices"] != [0, 1] or any(not launch_config[key] for key in ("nccl_isolate", "cuda_home", "cache_root"))):
        raise ValueError("require the qualified schema-2 DP2 CUDA and isolated NCCL launcher config")
    files["sglang_launch.json"] = launch_raw
    manifest = {"files": {name: sha256(raw).hexdigest() for name, raw in files.items()}}
    raw = (json.dumps(manifest, sort_keys=True, indent=2) + "\n").encode()
    files["SOURCE_MANIFEST.json"] = raw
    output = Path(output)
    sidecar = output.with_suffix(".source-manifest.json")
    if sidecar.exists():
        raise FileExistsError(sidecar)
    with ZipFile(output, "x") as archive:
        for name, payload in sorted(files.items()):
            member = ZipInfo(name)
            member.create_system = 3
            member.external_attr = (0o100755 if name == "sglang_launch.py" else 0o100600 if name == "sglang_launch.json" else 0o100644) << 16
            archive.writestr(member, payload, compress_type=ZIP_DEFLATED)
    with sidecar.open("xb") as stream:
        stream.write(raw)
    return {"zip_sha256": sha256(output.read_bytes()).hexdigest(), "source_manifest_sha256": sha256(raw).hexdigest(),
            "launch_sha256": manifest["files"]["sglang_launch.json"]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--launch", type=Path, required=True, help="private schema-2 launch config for the dedicated CUDA/NCCL environment")
    args = parser.parse_args()
    print(json.dumps(pack(args.output, args.launch), sort_keys=True))


if __name__ == "__main__":
    main()
