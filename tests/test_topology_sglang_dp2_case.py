from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from integrations.topology_memory import generate_dp2_case as dp2, tune_sglang_manual as tuner
from rsi_memory.topology import fingerprint


def copy_profile():
    return {"schema": 1, "scope": dp2.PROFILE_SCOPE, "utc": "test", "torch_version": "test",
            "host_cpu_affinity": [0], "fit_limitation": "test", "samples": 2,
            "peer_capability": [[True, False], [False, True]],
            "devices": [{"ordinal": index, "name": "test", "uuid": f"00000000-0000-0000-0000-{index + 1:012d}",
                         "total_memory": 1 << 30, "capability": [12, 0]} for index in range(2)],
            "records": [{"gpu_ordinal": index, "direction": direction, "bytes": size,
                         "samples_ns": [size + 99, size + 101], "median_ns": size + 100, "exact_bytes": True}
                        for index in range(2) for direction in ("h2d", "d2h") for size in (4096, 65536, 4194304)],
            "affine_link_fits": [{"gpu_ordinal": index, "direction": direction,
                                  "bandwidth_bytes_per_s": 10**9, "latency_ns": 100}
                                 for index in range(2) for direction in ("h2d", "d2h")]}


class DP2CaseTests(unittest.TestCase):
    def test_profiles_require_unmodified_byte_oracles_and_fits(self):
        profile = copy_profile()
        graph, bindings = dp2.measured_topology(json.dumps(profile).encode())
        self.assertEqual({(row["src"], row["dst"], row["transport"]) for row in graph["links"]},
                         {("host", node, "pinned_h2d") for node in dp2.COMPUTE}
                         | {(node, "host", "pinned_d2h") for node in dp2.COMPUTE})
        self.assertEqual([bindings[node]["ordinal"] for node in dp2.COMPUTE], [0, 1])
        broken = deepcopy(profile)
        broken["records"][0]["exact_bytes"] = False
        with self.assertRaisesRegex(ValueError, "verified raw directed samples"):
            dp2.measured_topology(json.dumps(broken).encode())
        broken = deepcopy(profile)
        broken["affine_link_fits"][0]["latency_ns"] += 1
        with self.assertRaisesRegex(ValueError, "fit differs"):
            dp2.measured_topology(json.dumps(broken).encode())

    def test_case_binds_explicit_routes_and_labels_witness_as_manual(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile = root / "profile.json"
            raw = json.dumps(copy_profile()).encode()
            profile.write_bytes(raw)
            fixture_root = Path(__file__).resolve().parents[1] / "examples" / "topology_memory"
            case = root / "case"
            result = dp2.build_dp2_case(case, profile=profile, briefing=fixture_root / "radix_handover_authored_v1.json",
                                        supplement=fixture_root / "archive_supplement_authored_v1.md",
                                        config_sha256="a" * 64, weights_sha256="b" * 64)
            self.assertEqual(next((case / "link_evidence").iterdir()).read_bytes(), raw)
            self.assertEqual(result["status"], "bound_unverified")
            self.assertIsNone(result["case_scope"]["generated"])
            self.assertIsNone(result["case_scope"]["manual_tuned"])
            self.assertIsNone(result["case_scope"]["token_lengths"])
            witness = json.loads((case / "relay_witness_plan_preview.json").read_bytes())
            peer = [row for row in witness["actions"] if row["node"] != dp2.HOST]
            self.assertEqual({(row["rank"], row["node"]) for row in peer}, {(0, "gpu-b"), (1, "gpu-a")})
            self.assertTrue(all(len(row["read_links"]) == len(row["write_links"]) == 2 for row in peer))
            schedule = json.loads((case / "external_schedule.json").read_bytes())
            self.assertEqual(schedule["engine"]["dp_size"], 2)
            for group in ("warmup", "requests"):
                self.assertEqual([(row["routed_dp_rank"], row["max_new_tokens"]) for row in schedule[group]], [(0, 64), (1, 64)])

    def test_manual_tuner_uses_native_training_rows_and_binds_evaluation(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            profile, receipt, train, evaluation = (root / name for name in ("profile.json", "receipt", "train", "evaluation"))
            profile.write_text(json.dumps(copy_profile()), encoding="utf-8")
            source = Path(__file__).resolve().parents[1] / "examples" / "topology_memory"
            supplement = root / "different-supplement.md"
            supplement.write_text("Independent prompt content for the tuner binding test.", encoding="utf-8")
            args = {"profile": profile, "briefing": source / "radix_handover_authored_v1.json",
                    "config_sha256": "a" * 64, "weights_sha256": "b" * 64}
            training = dp2.build_dp2_case(train, supplement=source / "archive_supplement_authored_v1.md", **args)
            dp2.build_dp2_case(evaluation, supplement=supplement, **args)
            # The tuner consumes only metadata for evaluation. No evaluation prompt is needed.
            (evaluation / "fixture.json").unlink()
            receipt.mkdir()
            result = {**{key: training[key] for key in ("target_sha256", "topology_sha256", "scenario_sha256")},
                      "schedule_inputs_sha256": training["schedule_sha256"], "compiled_plan_sha256": training["compiled_preview_sha256"]["native"],
                      "status": "completed", "arm": "native", "phase": "parity", "attention_dp_size": 2,
                      "ranks": [], "packed_kv_sha256_by_rank": []}
            schedule = json.loads((train / "external_schedule.json").read_bytes())
            for group in ("warmup", "requests"):
                result[group] = [{**row, "output_ids": [1] * row["max_new_tokens"]} for row in schedule[group]]
            for rank, chosen in enumerate((3, 5)):
                events = [{"event": "kv_selected_exact", "rank": rank, "layer": layer,
                           "rows": [{"history_bytes": 16 * dp2.ROW_BYTES}]} for _ in range(2 * (dp2.NEW_TOKENS - 1))
                          for layer in range(dp2.POOL["layers"])]
                result["packed_kv_sha256_by_rank"].append(fingerprint(events))
                result["ranks"].append({"backing": {"layer_tiers": ["host"] * dp2.POOL["layers"]}, "parallel": {"attention_dp_rank": rank},
                    "dispatch": {"selected_miss_payload_bytes_by_tier": {"host": 2 * (dp2.NEW_TOKENS - 1) * 7 * dp2.ROW_BYTES}}})
                for event in events:
                    event["rows"][0]["miss_rows"] = 7 if event["layer"] == chosen else 0
                (receipt / f"kv-rank-{rank}.jsonl").write_text("\n".join(json.dumps(event) for event in events), encoding="utf-8")
            (receipt / "result.json").write_text(json.dumps(result), encoding="utf-8")
            output = root / "tuned"
            report = tuner.tune(output, calibration_case=train, calibration_result=receipt, evaluation_case=evaluation)
            self.assertEqual(report["chosen_layers_by_rank"], [3, 5])
            self.assertEqual(report["status"], "manual_tuned_unverified")
            with self.assertRaisesRegex(ValueError, "distinct frozen workloads"):
                tuner.tune(root / "same", calibration_case=train, calibration_result=receipt, evaluation_case=train)
            bad = receipt / "kv-rank-0.jsonl"
            bad.write_text(bad.read_text(encoding="utf-8").replace('"miss_rows": 7', '"miss_rows": 8'), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "aggregate misses"):
                tuner.tune(root / "bad", calibration_case=train, calibration_result=receipt, evaluation_case=evaluation)


if __name__ == "__main__":
    unittest.main()
