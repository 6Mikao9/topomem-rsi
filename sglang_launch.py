#!/usr/bin/python3
"""Fixed launcher for the already qualified numeric-CVD NCCL2.32.3 environment."""

from hashlib import sha256
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys


MODULE = "topomem_sglang_rsi.sglang_eval"
NCCL_SHA256 = "651cf05af28976ca8d996f4e2750e69db1fa427667ee3405b9e576e9d2960eaf"
NCCL_WHEEL_SHA256 = "1459723080ac889d73a26edfa3e04383a7928ab31ac8f0ec43b3ea9548b04ff3"
LIBRARY = "nvidia/nccl/lib/libnccl.so.2"
CUDA_COMPAT_LIB = "/usr/local/cuda-12.8/lib64"


def verify_nccl(isolate, manifest):
    if (type(manifest) is not dict or manifest.keys() != {"schema", "wheel_sha256", "files"}
            or manifest["schema"] != 1 or manifest["wheel_sha256"] != NCCL_WHEEL_SHA256
            or manifest["files"].get(LIBRARY) != NCCL_SHA256 or isolate.is_symlink()):
        raise ValueError("invalid trusted NCCL wheel manifest or isolated root")
    expected = manifest["files"]
    required = {LIBRARY} | {name for name in expected if name.startswith("nvidia/nccl/include/")}
    directories = {parent.as_posix() for name in expected for parent in PurePosixPath(name).parents}
    seen = set()
    for path in isolate.rglob("*"):
        relative = path.relative_to(isolate).as_posix()
        if path.is_symlink():
            raise ValueError("isolated NCCL tree cannot contain symlinks")
        if path.is_dir() and relative in directories:
            continue
        if relative == "provenance.json" and path.is_file() and path.stat().st_size <= 65536:
            if type(json.loads(path.read_bytes())) is dict:
                continue
        if relative not in expected or not path.is_file():
            raise ValueError("unexpected file or directory in the isolated NCCL tree")
        digest = sha256()
        with path.open("rb") as stream:
            for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
                digest.update(chunk)
        if digest.hexdigest() != expected[relative]:
            raise ValueError("isolated NCCL bytes differ from the qualified wheel")
        seen.add(relative)
    if not required <= seen:
        raise ValueError("isolated NCCL tree is missing its library or headers")
    return isolate / LIBRARY


def prepare(config_raw, argv, env, nccl_manifest):
    config = json.loads(config_raw)
    if (type(config) is not dict or config.keys() != {"schema", "python", "model", "nccl_isolate", "cuda_home", "cache_root",
                                                     "gpu_indices", "gpu_uuids"}
            or type(config["schema"]) is not int or config["schema"] != 2 or config["gpu_indices"] != [0, 1]
            or any(type(index) is not int for index in config["gpu_indices"])):
        raise ValueError("invalid fixed SGLang launch contract")
    uuids = config["gpu_uuids"]
    if (type(uuids) is not list or len(uuids) != 2 or any(type(uuid) is not str or not re.fullmatch(
            r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", uuid) for uuid in uuids) or len(set(uuids)) != 2
            or env.get("CUDA_VISIBLE_DEVICES") != ",".join(uuids)):
        raise ValueError("supervisor's verified GPU UUID order differs from the launch contract")
    if len(argv) != 7 or argv[:3] != ["-B", "-m", MODULE] or (argv[3], argv[5]) != ("--input", "--output"):
        raise ValueError("launcher accepts only the bound evaluator invocation")
    paths = [Path(config[key]) for key in ("python", "model", "nccl_isolate", "cuda_home", "cache_root")]
    source = Path(env.get("PYTHONPATH", ""))
    if any(not path.is_absolute() or os.pathsep in str(path) for path in [*paths, source]):
        raise ValueError("launch paths must be fixed absolute single paths")
    python, model, isolate, cuda, cache = paths
    if not python.is_file() or not model.is_dir() or not source.is_dir():
        raise ValueError("launch Python, model, or source is missing")
    nccl_root = isolate / "nvidia" / "nccl"
    nccl = verify_nccl(isolate, nccl_manifest)
    if not (cuda / "bin").is_dir() or not (cuda / "lib").is_dir() or not cache.is_dir():
        raise ValueError("qualified CUDA toolkit or SGLang cache root is missing")
    libraries = [str(cuda / "lib"), CUDA_COMPAT_LIB]
    if env.get("LD_LIBRARY_PATH"):
        libraries.append(env["LD_LIBRARY_PATH"])
    launch_env = {**env, "CUDA_VISIBLE_DEVICES": "0,1", "PYTHONPATH": f"{source}{os.pathsep}{isolate}",
                  "LD_PRELOAD": str(nccl), "SGLANG_NCCL_SO_PATH": str(nccl), "EP_NCCL_ROOT_DIR": str(nccl_root),
                  "CUDA_HOME": str(cuda), "CUDA_PATH": str(cuda), "PATH": f"{cuda / 'bin'}{os.pathsep}{env.get('PATH', os.defpath)}",
                  "LD_LIBRARY_PATH": os.pathsep.join(libraries), "SGLANG_CACHE_DIR": str(cache), "SGLANG_JIT_CACHE_DIR": str(cache / "jit"),
                  "SGLANG_DISABLE_HISPARSE_PREFETCH": "1", "SGLANG_DSA_FUSE_TOPK": "0", "OMP_NUM_THREADS": "2",
                  "TOPOMEM_SGLANG_MODEL_DIR": str(model), "TOPOMEM_SGLANG_LAUNCH_SHA256": sha256(config_raw).hexdigest()}
    return str(python), launch_env


def main():
    launcher = Path(__file__).resolve()
    config = launcher.with_name("sglang_launch.json")
    if config.is_symlink() or config.stat().st_mode & 0o022:
        raise ValueError("launch configuration must be a trusted regular file")
    args = sys.argv[1:]
    wheel = json.loads(launcher.with_name("sglang_nccl_manifest.json").read_bytes())
    python, environment = prepare(config.read_bytes(), args, os.environ, wheel)
    os.execve(python, [python, *args], environment)


if __name__ == "__main__":
    main()
