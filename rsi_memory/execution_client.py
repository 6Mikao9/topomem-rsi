"""One persistent SSH process for trusted evaluations; no reconnect/retry loop."""

from __future__ import annotations

import json
from pathlib import Path
import queue
import shlex
import subprocess
import threading
import time

from .execution_worker import PROTOCOL, SHA256, atomic, digest, validate_job_id


class ExecutionError(RuntimeError):
    pass


class PersistentExecutor:
    """Serialize trusted jobs over an existing OpenSSH identity/configuration.

    The model never receives this object's host, credentials, paths, or protocol.
    Lost responses are unknown effects, not a reason to submit a second job.
    A later manually created executor can query a specific immutable job ID.
    """

    def __init__(self, *, host, worker, root, python, audit_directory,
                 source, source_manifest_sha256, gpu_indices, gpu_uuids, environment=None):
        if type(source_manifest_sha256) is not str or not SHA256.fullmatch(source_manifest_sha256):
            raise ValueError("source_manifest_sha256 must be the expected manifest byte digest")
        self.directory = Path(audit_directory)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.config = {"source": source, "source_manifest_sha256": source_manifest_sha256, "gpu_indices": list(gpu_indices),
                       "gpu_uuids": list(gpu_uuids), "environment": environment or {}}
        remote = shlex.join(["python3", "-u", worker, "--root", root, "--python", python])
        self._stderr = (self.directory / "connection.log").open("ab")
        self._responses = queue.Queue()
        self._lock = threading.RLock()
        self._serial = 0
        self._unusable = False
        try:
            self.process = subprocess.Popen(
                ["ssh", "-T", "-o", "BatchMode=yes", "-o", "ConnectTimeout=15", "-o", "ServerAliveInterval=30",
                 "-o", "ServerAliveCountMax=3", "--", host, remote],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=self._stderr,
                text=True, encoding="utf-8", bufsize=1,
            )
            threading.Thread(target=self._reader, daemon=True).start()
            greeting = self._receive(30)
            if greeting.get("protocol") != PROTOCOL:
                raise ExecutionError("remote evaluator protocol mismatch")
        except BaseException:
            self.close()
            raise

    def _reader(self):
        try:
            for line in self.process.stdout:
                try:
                    value = json.loads(line)
                except (ValueError, TypeError):
                    value = {"transport_error": "unexpected remote stdout"}
                self._responses.put(value)
        finally:
            self._responses.put({"transport_error": "SSH channel ended"})

    def _receive(self, timeout):
        try:
            reply = self._responses.get(timeout=timeout)
        except queue.Empty:
            raise ExecutionError("remote response timeout; effect state is unknown; do not resubmit") from None
        if type(reply) is not dict:
            raise ExecutionError("invalid remote response; effect state is unknown")
        if "transport_error" in reply:
            raise ExecutionError(reply["transport_error"])
        return reply

    def request(self, operation, **fields):
        with self._lock:
            if self._unusable:
                raise ExecutionError("execution channel is unusable; use a new channel for explicit status lookup")
            self._serial += 1
            sequence = self._serial
            message = {"sequence": sequence, "operation": operation, **fields}
            raw = json.dumps(message, allow_nan=False)
            try:
                self.process.stdin.write(raw + "\n")
                self.process.stdin.flush()
                timeout = fields.get("timeout", 900) + 120 if operation == "run" else 30
                reply = self._receive(timeout)
                if reply.get("sequence") != sequence or type(reply.get("ok")) is not bool:
                    raise ExecutionError("response sequence or shape mismatch; effect state is unknown")
                if reply["ok"] and type(reply.get("result")) is not dict:
                    raise ExecutionError("missing remote result object; effect state is unknown")
            except (ExecutionError, OSError, ValueError):
                self._unusable = True
                raise
            if not reply["ok"]:
                if reply.get("effect_unknown"):
                    self._unusable = True
                raise ExecutionError(reply.get("error", "remote execution rejected"))
            return reply["result"]

    def _check_receipt(self, receipt, job_id, expected_request=None):
        if receipt.get("job_id") != job_id:
            raise ExecutionError("remote receipt job identity mismatch")
        if receipt.get("status") in {"not_found", "started_unknown"} and expected_request is None:
            return
        if (receipt.get("schema") != 2 or receipt.get("source_manifest_sha256") != self.config["source_manifest_sha256"]
                or receipt.get("gpu_indices") != self.config["gpu_indices"] or receipt.get("gpu_uuids") != self.config["gpu_uuids"]
                or (expected_request is not None and receipt.get("request_sha256") != expected_request)
                or receipt.get("status") not in {"completed", "failed", "timed_out", "environment_conflict"}):
            raise ExecutionError("remote receipt source, request, device, or status mismatch")
        if "result" in receipt:
            if (type(receipt["result"]) is not dict or receipt.get("result_canonical_sha256") != digest(receipt["result"])
                    or type(receipt.get("result_sha256")) is not str or not SHA256.fullmatch(receipt["result_sha256"])):
                raise ExecutionError("remote receipt result hash mismatch")
        if receipt["status"] == "completed" and (receipt.get("returncode") != 0 or "result" not in receipt
                or receipt.get("remaining_compute_pids") != "" or receipt.get("process_group_remaining") is not False):
            raise ExecutionError("remote completion receipt lacks result or verified process cleanup")

    def run(self, job_id, module, payload, *, timeout=900):
        local = self.directory / (validate_job_id(job_id) + ".json")
        if type(timeout) is not int or not 1 <= timeout <= 3600:
            raise ValueError("invalid trusted execution timeout")
        with self._lock:
            if local.exists():
                raise ExecutionError("local receipt already exists; use explicit status lookup")
            started = time.monotonic()
            fields = {"job_id": job_id, "module": module, "payload": payload, "timeout": timeout, **self.config}
            expected = digest({"operation": "run", **fields})
            receipt = self.request("run", **fields)
            try:
                self._check_receipt(receipt, job_id, expected)
            except (ExecutionError, ValueError):
                self._unusable = True
                raise
            receipt["client_elapsed_seconds"] = time.monotonic() - started
            atomic(local, receipt)
            return receipt

    def status(self, job_id):
        receipt = self.request("status", job_id=validate_job_id(job_id))
        self._check_receipt(receipt, job_id)
        return receipt

    def close(self):
        self._unusable = True
        process = getattr(self, "process", None)
        if process is not None and process.poll() is None:
            try:
                process.stdin.close()
                process.wait(timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
        stream = getattr(self, "_stderr", None)
        if stream is not None:
            stream.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()
