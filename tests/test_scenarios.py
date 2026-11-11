from hashlib import sha256
from pathlib import Path
import unittest

from rsi_memory.scenarios import ScenarioWorkload, content_fingerprint


WORKLOADS = Path(__file__).resolve().parents[1] / "examples" / "workloads"


def load(name):
    return ScenarioWorkload.read(WORKLOADS / f"{name}.json")


class ScenarioTests(unittest.TestCase):
    def test_agent_two_tool_returns_retain_actual_prefix_rows(self):
        workload = load("agent-water-inspection")
        report = workload.reference_report()
        self.assertEqual(report["execution_scope"], "cpu_utf8_page_reference")
        self.assertEqual(report["distinct_tool_calls"], 2)
        self.assertEqual(set(report["witnesses"]), set(workload.required_capabilities))
        first, second, third = report["inferences"]
        self.assertEqual(second["retained_rows"], first["context_rows"])
        self.assertEqual(third["retained_rows"], second["context_rows"])
        self.assertEqual(report["live_pages_at_end"], 0)

        bad = workload.to_dict()
        next(s for s in bad["segments"] if s["id"] == "code")["call_id"] = "wrong-call"
        with self.assertRaisesRegex(ValueError, "content hash"):
            ScenarioWorkload.from_dict(bad)
        bad["source"]["content_sha256"] = content_fingerprint(bad)
        with self.assertRaisesRegex(ValueError, "does not match"):
            ScenarioWorkload.from_dict(bad)

    def test_chat_content_grows_and_schedule_hash_binds_timestamps(self):
        workload = load("chatbot-trip-planning")
        first, second, third = workload.reference_report()["inferences"]
        self.assertLess(first["context_rows"], second["context_rows"])
        self.assertLess(second["context_rows"], third["context_rows"])
        data = workload.to_dict()
        data["events"][-1]["at_ns"] += 1
        shifted = ScenarioWorkload.from_dict(data)
        self.assertNotEqual(shifted.schedule_sha256, workload.schedule_sha256)
        self.assertNotEqual(shifted.workload_sha256, workload.workload_sha256)

    def test_document_qa_keeps_gold_separate_and_checks_ordered_payload(self):
        workload = load("long-document-estuary-qa")
        manifest = workload.to_dict()
        event = next(e for e in manifest["events"] if e["op"] == "infer")
        payload = b"".join(segment["text"].encode("utf-8") for segment in manifest["segments"])
        self.assertGreater(len(payload), manifest["task"]["minimum_document_bytes"])
        self.assertTrue(manifest["task"]["expected_answers"])
        selected = bytes(payload[row] for row in event["select_rows"])
        self.assertEqual(workload.reference_report()["inferences"][0]["selected_sha256"], sha256(selected).hexdigest())
        event["select_rows"] = [len(payload)]
        with self.assertRaisesRegex(ValueError, "exceeds the current context"):
            ScenarioWorkload.from_dict(manifest)

    def test_sibling_rollouts_require_actual_shared_tail_and_divergence(self):
        workload = load("rollout-bridge-options")
        parent, north, south = workload.reference_report()["inferences"]
        self.assertEqual(north["cow"]["prefix_sha256"], parent["context_sha256"])
        self.assertEqual(south["cow"]["prefix_sha256"], parent["context_sha256"])
        self.assertGreater(north["cow"]["shared_pages_at_fork"], 0)
        self.assertNotEqual(north["cow"]["suffix_sha256"], south["cow"]["suffix_sha256"])
        self.assertNotEqual(north["selected_sha256"], south["selected_sha256"])

        data = workload.to_dict()
        data["page_rows"] = parent["context_rows"]  # Appending now allocates a new page, with no COW.
        with self.assertRaisesRegex(ValueError, "shared_prefix_cow"):
            ScenarioWorkload.from_dict(data)
        data = workload.to_dict()
        data["segments"][-1]["text"] = data["segments"][-2]["text"]
        data["source"]["content_sha256"] = content_fingerprint(data)
        with self.assertRaisesRegex(ValueError, "shared_prefix_cow"):
            ScenarioWorkload.from_dict(data)
        data = workload.to_dict()
        data["events"].pop()
        with self.assertRaisesRegex(ValueError, "close every page owner"):
            ScenarioWorkload.from_dict(data)

    def test_capability_gate_rejects_independent_request_adapter(self):
        workload = load("agent-water-inspection")
        with self.assertRaisesRegex(ValueError, "retained_prefix"):
            workload.require_backend_capabilities({"tool_roundtrip"})
        workload.require_backend_capabilities(workload.required_capabilities)
