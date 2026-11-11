from contextlib import nullcontext
import hashlib
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

from rsi_memory.execution_client import ExecutionError, PersistentExecutor
from rsi_memory import execution_worker as worker


MODULE = "integrations.topology_memory.torch_eval"
GPU_UUID = "GPU-00000000-0000-0000-0000-000000000001"


class ExecutionProtocolTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.source = self.root / "snapshot"
        evaluator = self.source / (MODULE.replace(".", "/") + ".py")
        evaluator.parent.mkdir(parents=True)
        evaluator.write_bytes(b"# CPU test fixture\n")
        self.manifest = self.source / "SOURCE_MANIFEST.json"
        self.manifest.write_text(json.dumps({"files": {evaluator.relative_to(self.source).as_posix():
            hashlib.sha256(evaluator.read_bytes()).hexdigest()}}), encoding="utf-8")
        self.config = {"source": "snapshot", "source_manifest_sha256": hashlib.sha256(self.manifest.read_bytes()).hexdigest(),
                       "gpu_indices": [0], "gpu_uuids": [GPU_UUID], "environment": {}}
        self.message = {"operation": "run", "job_id": "case-1", "module": MODULE, "payload": {"only": "fixed input"},
                        "timeout": 20, **self.config}
        self.audit = self.root / "local-audit"

    def _query(self, indices, fields, kind="gpu"):
        self.assertEqual(indices, [0])
        return "" if kind == "compute-apps" else GPU_UUID

    def _fake_child(self, argv, **kwargs):
        self.assertTrue(kwargs["start_new_session"])
        self.assertNotIn("OPENCODE_GO_API_KEY", kwargs["env"])
        self.assertEqual(kwargs["env"]["CUDA_VISIBLE_DEVICES"], GPU_UUID)
        Path(argv[-1]).write_bytes(b'{ "score": 12 }\n')
        return Mock(pid=222, wait=Mock(return_value=0), poll=Mock(return_value=0))

    def _run_worker(self, query=None):
        with (patch.object(worker, "gpu_leases", return_value=nullcontext()),
              patch.object(worker, "query", side_effect=query or self._query),
              patch.object(worker.subprocess, "Popen", side_effect=self._fake_child),
              patch.object(worker, "_group_exists", return_value=False),
              patch.dict(os.environ, {"OPENCODE_GO_API_KEY": "test-placeholder"})):
            return worker.execute(self.root, "fake-python", self.message)

    def test_worker_pins_full_source_before_effect(self):
        with patch.object(worker, "query") as query:
            self.manifest.write_bytes(self.manifest.read_bytes() + b"\n")
            with self.assertRaisesRegex(ValueError, "expected SHA-256"):
                worker.execute(self.root, "fake-python", self.message)
            self.message["source_manifest_sha256"] = hashlib.sha256(self.manifest.read_bytes()).hexdigest()
            (self.source / "extra.py").write_bytes(b"# unpinned dependency\n")
            with self.assertRaisesRegex(ValueError, "all importable files"):
                worker.execute(self.root, "fake-python", self.message)
            query.assert_not_called()
        self.assertFalse((self.root / "jobs").exists())

    def test_gpu_busy_refuses_before_process_or_job_reservation(self):
        def query(indices, fields, kind="gpu"):
            return "12345" if kind == "compute-apps" else self._query(indices, fields, kind)

        with (patch.object(worker, "gpu_leases", return_value=nullcontext()),
              patch.object(worker, "query", side_effect=query),
              patch.object(worker.subprocess, "Popen") as start):
            with self.assertRaisesRegex(ValueError, "existing compute process"):
                worker.execute(self.root, "fake-python", self.message)
            start.assert_not_called()
        self.assertEqual(worker.status(self.root, "case-1")["status"], "not_found")

    def test_worker_receipt_keeps_raw_and_canonical_hash_and_replays_once(self):
        receipt = self._run_worker()
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(receipt["result_canonical_sha256"], worker.digest({"score": 12}))
        self.assertEqual(receipt["result_sha256"], hashlib.sha256(b'{ "score": 12 }\n').hexdigest())
        self.assertEqual(worker.status(self.root, "case-1"), receipt)
        with patch.object(worker.subprocess, "Popen") as start:
            replay = worker.execute(self.root, "fake-python", {"sequence": 52, **self.message})
            self.assertTrue(replay["replayed_receipt"])
            with self.assertRaisesRegex(ValueError, "different immutable request"):
                worker.execute(self.root, "fake-python", {**self.message, "payload": {}})
            start.assert_not_called()

    def test_worker_reports_new_compute_pid_without_signaling_it(self):
        pid_queries = iter(("", "98765"))

        def query(indices, fields, kind="gpu"):
            return next(pid_queries) if kind == "compute-apps" else self._query(indices, fields, kind)

        with patch.object(worker, "_stop_group") as signal:
            receipt = self._run_worker(query)
            signal.assert_not_called()
        self.assertEqual(receipt["status"], "environment_conflict")
        self.assertEqual(receipt["remaining_compute_pids"], "98765")

    def test_reserved_job_without_receipt_is_not_submitted_again(self):
        directory = worker.job_directory(self.root, "case-1")
        directory.mkdir(parents=True)
        worker.atomic(directory / "request.json", {"sha256": worker.digest(self.message), "request": self.message})
        self.assertEqual(worker.status(self.root, "case-1")["status"], "started_unknown")
        with patch.object(worker.subprocess, "Popen") as start:
            with self.assertRaisesRegex(ValueError, "execution state is unknown"):
                worker.execute(self.root, "fake-python", self.message)
            start.assert_not_called()

    def _client(self, replies):
        transcript = [{"protocol": worker.PROTOCOL}, *replies]
        channel = Mock(stdin=io.StringIO(), stdout=io.StringIO("".join(json.dumps(row) + "\n" for row in transcript)),
                       poll=Mock(return_value=0))
        with patch("rsi_memory.execution_client.subprocess.Popen", return_value=channel):
            client = PersistentExecutor(host="fake-host", worker="/fake/worker.py", root="/fake/root", python="/fake/venv/bin/python",
                                        audit_directory=self.audit, **self.config)
        self.addCleanup(client.close)
        return client, channel

    def test_client_validates_receipt_and_checks_local_replay_before_write(self):
        reply = self._run_worker()
        client, channel = self._client([{"sequence": 1, "ok": True, "result": reply}])
        receipt = client.run("case-1", MODULE, self.message["payload"], timeout=20)
        self.assertEqual(receipt["result"], {"score": 12})
        self.assertTrue((self.audit / "case-1.json").is_file())
        outbound = channel.stdin.getvalue()
        with self.assertRaisesRegex(ExecutionError, "local receipt already exists"):
            client.run("case-1", MODULE, self.message["payload"], timeout=20)
        self.assertEqual(channel.stdin.getvalue(), outbound)

    def test_client_does_not_dispatch_after_unknown_response(self):
        client, channel = self._client([{"sequence": 999, "ok": True, "result": {}}])
        with self.assertRaisesRegex(ExecutionError, "state is unknown"):
            client.run("case-1", MODULE, {}, timeout=20)
        sent = channel.stdin.getvalue()
        with self.assertRaisesRegex(ExecutionError, "channel is unusable"):
            client.run("case-2", MODULE, {}, timeout=20)
        self.assertEqual(channel.stdin.getvalue(), sent)

    def test_client_rejects_tampered_receipt_result(self):
        reply = self._run_worker()
        reply["result"]["score"] = 0
        client, _ = self._client([{"sequence": 1, "ok": True, "result": reply}])
        with self.assertRaisesRegex(ExecutionError, "result hash mismatch"):
            client.run("case-1", MODULE, self.message["payload"], timeout=20)
        self.assertFalse((self.audit / "case-1.json").exists())


if __name__ == "__main__":
    unittest.main()
