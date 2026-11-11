"""Trusted Linux process supervisor for a single persistent JSON/SSH channel."""

from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import tempfile
import time


PROTOCOL = "topology-memory-executor-v2"
MODULES = {"integrations.topology_memory.torch_eval"}
ENVIRONMENT = {"CUDA_HOME", "CUDA_PATH", "LD_LIBRARY_PATH", "OMP_NUM_THREADS",
               "SGLANG_CACHE_DIR", "SGLANG_JIT_CACHE_DIR", "SGLANG_NCCL_SO_PATH",
               "CUDA_DEVICE_ORDER", "NCCL_DEBUG", "NCCL_DEBUG_FILE"}
INHERITED_ENVIRONMENT = {"PATH", "HOME", "USER", "LOGNAME", "SHELL", "LANG", "LC_ALL", "TZ", "TMPDIR",
                         "HF_HOME", "HF_HUB_CACHE", "HF_HUB_OFFLINE", "TRANSFORMERS_CACHE", "XDG_CACHE_HOME"}
SHA256 = re.compile(r"[0-9a-f]{64}")
GPU_UUID = re.compile(r"GPU-[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}", re.IGNORECASE)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _sync_directory(path):
    # The supervisor runs on Linux. Windows test runs sync file contents only.
    if os.name == "posix":
        fd = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(fd)
        finally:
            os.close(fd)


def atomic(path, value):
    pending = path.with_suffix(".pending")
    with pending.open("w", encoding="utf-8") as stream:
        stream.write(json.dumps(value, indent=2, allow_nan=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(pending, path)
    _sync_directory(path.parent)


def validate_job_id(name):
    if type(name) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}", name):
        raise ValueError("invalid trusted job ID")
    return name


def job_directory(root, name):
    jobs = root / "jobs"
    path = jobs / validate_job_id(name)
    if jobs.is_symlink() or path.is_symlink():
        raise ValueError("job path cannot be a symlink")
    return path


def verify_source(root, name, expected, module):
    if type(name) is not str or not name or "\\" in name or type(expected) is not str or not SHA256.fullmatch(expected):
        raise ValueError("invalid source location or expected manifest hash")
    relative = PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or relative.as_posix() != name:
        raise ValueError("source must use a normalized path within the experiment")
    source = (root / name).resolve(strict=True)
    if not source.is_relative_to(root) or source == root:
        raise ValueError("source path escapes experiment directory")
    manifest_path = source / "SOURCE_MANIFEST.json"
    if manifest_path.is_symlink():
        raise ValueError("source manifest cannot be a symlink")
    raw = manifest_path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != expected:
        raise ValueError("source manifest differs from the expected SHA-256")
    manifest = json.loads(raw)
    hashes = manifest.get("files", manifest) if type(manifest) is dict else None
    evaluator = module.replace(".", "/") + ".py"
    if type(hashes) is not dict or evaluator not in hashes:
        raise ValueError("source manifest must pin the trusted evaluator")
    for name, sha in hashes.items():
        relative = PurePosixPath(name)
        if ("\\" in name or relative.as_posix() != name or relative.is_absolute() or ".." in relative.parts
                or type(sha) is not str or not SHA256.fullmatch(sha)):
            raise ValueError("invalid source manifest path or digest")
        path = source / name
        if not path.is_file() or not path.resolve().is_relative_to(source):
            raise ValueError("source manifest entry escapes or is not a file")
        if hashlib.sha256(path.read_bytes()).hexdigest() != sha:
            raise ValueError("immutable evaluation source manifest mismatch")
    python_files = set()
    for path in source.rglob("*"):
        if path.is_symlink():
            raise ValueError("immutable source cannot contain symlinks")
        if path.suffix.lower() in {".py", ".pyc", ".pyo", ".pyd", ".so"} and path.is_file():
            python_files.add(path.relative_to(source).as_posix())
    if python_files - hashes.keys():
        raise ValueError("source manifest does not pin all importable files")
    return source


@contextmanager
def gpu_leases(uuids):
    import fcntl

    with ExitStack() as stack:
        for uuid in sorted(uuids):
            path = Path(tempfile.gettempdir()) / f"topomem-rsi-gpu-{uuid.lower()}.lock"
            fd = os.open(path, os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW | os.O_CLOEXEC, 0o600)
            stack.callback(os.close, fd)
            try:
                fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise ValueError("a trusted evaluator already holds this GPU lease") from None
        yield


def query(indices, fields, kind="gpu"):
    return subprocess.check_output(
        ["nvidia-smi", "-i", ",".join(map(str, indices)), "--query-" + kind + "=" + fields,
         "--format=csv,noheader,nounits"], text=True, timeout=15).strip()


def execute(root, python, message):
    directory = job_directory(root, message["job_id"])
    request = {k: v for k, v in message.items() if k != "sequence"}
    request_sha = digest(request)
    if directory.exists():
        previous = status(root, message["job_id"])
        if previous.get("request_sha256") is None:
            raise ValueError("job reservation has no valid request record; execution state is unknown")
        if previous["request_sha256"] != request_sha:
            raise ValueError("job ID already belongs to a different immutable request")
        if previous["status"] != "started_unknown":
            return {**previous, "replayed_receipt": True}
        raise ValueError("job reservation has no valid completion receipt; execution state is unknown")
    if message["module"] not in MODULES:
        raise ValueError("module is not a trusted evaluator")
    source = verify_source(root, message["source"], message["source_manifest_sha256"], message["module"])
    indices, uuids = message["gpu_indices"], message["gpu_uuids"]
    if (type(indices) is not list or not indices or any(type(n) is not int or n < 0 for n in indices)
            or len(set(indices)) != len(indices) or type(uuids) is not list or len(indices) != len(uuids)
            or any(type(uuid) is not str or not GPU_UUID.fullmatch(uuid) for uuid in uuids)
            or len({uuid.lower() for uuid in uuids}) != len(uuids)):
        raise ValueError("invalid fixed physical device mapping")
    environment = message.get("environment", {})
    if (type(environment) is not dict or environment.keys() - ENVIRONMENT
            or any(type(value) is not str or "\0" in value for value in environment.values())):
        raise ValueError("unexpected external environment override")
    timeout = message.get("timeout", 900)
    if type(timeout) is not int or not 1 <= timeout <= 3600:
        raise ValueError("invalid evaluation timeout")
    with gpu_leases(uuids):
        if query(indices, "uuid").splitlines() != uuids:
            raise ValueError("physical GPU identities changed")
        before = query(indices, "uuid,memory.used,utilization.gpu,temperature.gpu,power.draw")
        if query(indices, "pid", "compute-apps"):
            raise ValueError("selected GPU has an existing compute process")
        return _launch(directory, source, python, message, request, request_sha, before, timeout, environment)


def _group_exists(pgid):
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    return True


def _stop_group(pgid, signum):
    try:
        os.killpg(pgid, signum)
    except ProcessLookupError:
        pass


def _clean_group(process, *, wait_seconds=5):
    """Only signal the isolated process group created for this evaluator."""
    pgid = process.pid
    if not _group_exists(pgid):
        return False, False
    _stop_group(pgid, signal.SIGTERM)
    deadline = time.monotonic() + wait_seconds
    while _group_exists(pgid) and time.monotonic() < deadline:
        process.poll()
        time.sleep(0.05)
    if _group_exists(pgid):
        _stop_group(pgid, signal.SIGKILL)
    try:
        process.wait(timeout=wait_seconds)
    except subprocess.TimeoutExpired:
        pass
    deadline = time.monotonic() + wait_seconds
    while _group_exists(pgid) and time.monotonic() < deadline:
        time.sleep(0.05)
    return True, _group_exists(pgid)


def _proc_starttime(pid):
    try:
        stat = (Path("/proc") / str(pid) / "stat").read_text()
        return stat.rsplit(")", 1)[1].split()[19]
    except (OSError, IndexError):
        return None


def _launch(directory, source, python, message, request, request_sha, before, timeout, environment):
    directory.mkdir(parents=True)
    atomic(directory / "request.json", {"sha256": request_sha, "request": request})
    atomic(directory / "input.json", message["payload"])
    # Persist both newly created directory entries before the child can run.
    _sync_directory(directory.parent)
    _sync_directory(directory.parent.parent)
    env = {key: value for key, value in os.environ.items() if key in INHERITED_ENVIRONMENT}
    env.update(environment)
    indices, uuids = message["gpu_indices"], message["gpu_uuids"]
    env.update(CUDA_VISIBLE_DEVICES=",".join(uuids), PYTHONPATH=str(source),
               PYTHONDONTWRITEBYTECODE="1", PYTHONNOUSERSITE="1")
    argv = [python, "-B", "-m", message["module"], "--input", str(directory / "input.json"),
            "--output", str(directory / "result.json")]
    receipt = {"schema": 2, "job_id": message["job_id"], "request_sha256": request_sha,
               "source_manifest_sha256": message["source_manifest_sha256"], "gpu_indices": indices,
               "gpu_uuids": uuids, "utc_start": datetime.now(timezone.utc).isoformat(), "before": before,
               "argv": argv, "returncode": None, "timeout": False, "failure": None, "environment_conflict": False}
    started = time.monotonic()
    with (directory / "process.log").open("x") as log:
        process = None
        try:
            process = subprocess.Popen(argv, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT,
                                       start_new_session=True)
            receipt.update(pid=process.pid, proc_starttime=_proc_starttime(process.pid))
            atomic(directory / "started.json", receipt)
            receipt["returncode"] = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            receipt["timeout"] = True
        except (OSError, ValueError) as exc:
            receipt["failure"] = "process_supervision_" + type(exc).__name__
        finally:
            if process is not None:
                try:
                    residue, remains = _clean_group(process)
                    receipt.update(process_group_cleanup_required=residue, process_group_remaining=remains)
                    receipt["returncode"] = process.poll()
                except OSError:
                    receipt["failure"] = "could_not_verify_process_group_cleanup"
    receipt.update(elapsed_seconds=time.monotonic() - started, utc_end=datetime.now(timezone.utc).isoformat(),
                   after=None, remaining_compute_pids=None)
    try:
        receipt["after"] = query(indices, "uuid,memory.used,utilization.gpu")
        receipt["remaining_compute_pids"] = query(indices, "pid", "compute-apps")
        receipt["environment_conflict"] = bool(receipt["remaining_compute_pids"])
    except (OSError, subprocess.SubprocessError) as exc:
        receipt["after_query_failure"] = type(exc).__name__
    result_path = directory / "result.json"
    if result_path.is_file():
        try:
            raw = result_path.read_bytes()
            result = json.loads(raw)
            if type(result) is not dict:
                raise ValueError("evaluator result must be a JSON object")
            receipt["result_canonical_sha256"] = digest(result)
            receipt["result_sha256"] = hashlib.sha256(raw).hexdigest()
            receipt["result"] = result
        except (OSError, ValueError) as exc:
            receipt["result_failure"] = type(exc).__name__
    else:
        receipt["result_failure"] = "missing_result"
    receipt["log_sha256"] = hashlib.sha256((directory / "process.log").read_bytes()).hexdigest()
    if receipt["environment_conflict"]:
        receipt["status"] = "environment_conflict"
    elif receipt["timeout"]:
        receipt["status"] = "timed_out"
    elif (receipt["returncode"] != 0 or receipt["failure"] or receipt.get("after_query_failure")
          or receipt.get("result_failure") or receipt.get("process_group_cleanup_required")
          or receipt.get("process_group_remaining")):
        receipt["status"] = "failed"
    else:
        receipt["status"] = "completed"
    if receipt["status"] != "completed":
        receipt["log_tail"] = (directory / "process.log").read_text(encoding="utf-8", errors="replace")[-8000:]
    atomic(directory / "receipt.json", receipt)
    return receipt


def _read_record(path):
    try:
        record = json.loads(path.read_bytes())
        return record if type(record) is dict else {}
    except (OSError, ValueError):
        return {}


def status(root, job_id):
    directory = job_directory(root, job_id)
    if not directory.exists():
        return {"job_id": job_id, "status": "not_found"}
    result = {"job_id": job_id, "status": "started_unknown"}
    record = _read_record(directory / "request.json")
    request = record.get("request")
    try:
        if type(request) is not dict or request.get("job_id") != job_id or record.get("sha256") != digest(request):
            return result
    except ValueError:
        return result
    result["request_sha256"] = record["sha256"]
    receipt = _read_record(directory / "receipt.json")
    if (receipt.get("job_id") == job_id and receipt.get("request_sha256") == record["sha256"]
            and receipt.get("status") in ("completed", "failed", "environment_conflict", "timed_out")):
        return receipt
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--python", required=True)
    args = parser.parse_args()
    root = args.root.resolve(strict=True)
    print(json.dumps({"protocol": PROTOCOL}), flush=True)
    for line in sys.stdin:
        sequence = None
        message = None
        try:
            message = json.loads(line)
            sequence = message["sequence"]
            if message["operation"] == "run":
                result = execute(root, args.python, message)
            elif message["operation"] == "status":
                result = status(root, message["job_id"])
            else:
                raise ValueError("unknown trusted operation")
            response = {"sequence": sequence, "ok": True, "result": result}
        except Exception as exc:
            response = {"sequence": sequence, "ok": False, "error": type(exc).__name__ + ": " + str(exc)}
            if type(message) is dict and message.get("operation") == "run":
                try:
                    response["effect_unknown"] = status(root, message.get("job_id"))["status"] == "started_unknown"
                except (ValueError, OSError):
                    response["effect_unknown"] = True
        print(json.dumps(response, allow_nan=False), flush=True)


if __name__ == "__main__":
    main()
