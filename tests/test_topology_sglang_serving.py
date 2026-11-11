import asyncio
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from integrations.topology_memory import generate_serving_case as cases
from integrations.topology_memory import sglang_backing as backing, sglang_observe as observe, sglang_serving as serving
from rsi_memory.topology import fingerprint


def _replay_fixture():
    return {"schema": 1, "name": "cpu-binding-fixture", "warmup": [
        {"rid": "warm", "arrival_ms": 0, "input_ids": [1, 2], "max_new_tokens": 3}], "requests": [
        {"rid": "measured", "arrival_ms": 0, "prompt": "known passage", "max_new_tokens": 3, "session": "s1"}]}


class ServingBindingTests(unittest.TestCase):
    def test_worker_receipt_is_complete_on_publish_and_never_replaced(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            path = root / "dispatch-rank-0.json"
            pending = root / "dispatch-rank-0.json.pending"
            reference = root / "reference.json"
            value = {"finished": ["warm", "measured"], "batches": [{"mode": "DECODE", "request_ids": ["measured"]}]}
            observe.save(reference, value)
            expected = reference.read_bytes()
            original_link = os.link

            def link(source, destination):
                self.assertEqual((source, destination), (pending, path))
                self.assertFalse(path.exists())
                self.assertEqual(source.read_bytes(), expected)
                original_link(source, destination)

            with patch.object(observe.os, "link", side_effect=link):
                observe.publish(path, value)
            self.assertEqual(path.read_bytes(), expected)
            self.assertFalse(pending.exists())
            with self.assertRaises(FileExistsError):
                observe.publish(path, {"wrong": 1})
            self.assertEqual(path.read_bytes(), expected)
            failed = root / "bad.json"
            with self.assertRaises(ValueError):
                observe.publish(failed, {"invalid": float("nan")})
            self.assertFalse(failed.exists())
            self.assertFalse((root / "bad.json.pending").exists())

    def test_native_and_generated_share_bound_caps_but_take_actual_different_routes(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            model = root / "model"
            model.mkdir()
            (model / "config.json").write_bytes(b"{}")
            (model / "model.safetensors").write_bytes(b"test weights")
            evidence = root / "measured.json"
            evidence.write_bytes(b"test measurement")
            evidence_sha = serving._hash_file(evidence)
            rows, row_bytes = 8192, 656
            slab = rows * row_bytes
            relay = backing.workspace_bytes(2048, row_bytes, 1)
            topology = {"schema": 1, "nodes": [
                {"id": "gpu-a", "kind": "gpu", "capacity_bytes": relay["compute"]},
                {"id": "gpu-b", "kind": "gpu", "capacity_bytes": slab + relay["peer"]},
                {"id": "host", "kind": "host", "capacity_bytes": 2 * slab + relay["host"]}],
                "links": [{"id": edge, "src": src, "dst": dst, "transport": transport,
                           "bandwidth_bytes_per_s": 1, "latency_ns": 1, "bottlenecks": [], "evidence_sha256": evidence_sha}
                          for edge, src, dst, transport in (("h2d-a", "host", "gpu-a", "pinned_h2d"),
                          ("h2d-b", "host", "gpu-b", "pinned_h2d"), ("d2h-a", "gpu-a", "host", "pinned_d2h"),
                          ("d2h-b", "gpu-b", "host", "pinned_d2h"))], "bottlenecks": []}
            target = {"schema": 1, "gpu_bindings": {node: {"ordinal": i, "uuid": f"00000000-0000-0000-0000-{i:012d}",
                                                            "name": "CPU test", "capability": [12, 0]}
                                                     for i, node in enumerate(("gpu-a", "gpu-b"))},
                      "model_config_sha256": serving._hash_file(model / "config.json"),
                      "model_weights_sha256": serving._hash_file(model / "model.safetensors"),
                      "pool": {"rows": rows, "layers": 2, "page_size": 64, "hot_slots": 4096},
                      "native_host": "host", "allowed_nodes": {"gpu-a": ["host", "gpu-b"]}}
            fixture = _replay_fixture()
            scenario = {"schema": 1, "rank_placement": [{"rank": 0, "node": "gpu-a"}], "target_sha256": fingerprint(target),
                        "workload_sha256": fingerprint(fixture),
                        "schedule_sha256": fingerprint(serving.schedule_inputs(fixture, 1, "torch-unfused")),
                        "attention": {"family": "dsa", "selector": "torch-unfused", "selection_unit": "token",
                                      "top_k": 2048, "kv_row_bytes": 656}}
            program = {"schema": 1, "kind": "kv_memory_program", "rules": [
                {"event": "allocate", "action": "place", "when": {"op": "const", "value": 1},
                 "score": {"op": "feature", "name": "target_is_host"}}]}
            paths = {}
            for name, value in (("fixture", fixture), ("topology", topology), ("target", target),
                                ("scenario", scenario), ("program", program)):
                paths[name] = root / f"{name}.json"
                observe.save(paths[name], value)
            load_tokenizer = lambda _model: SimpleNamespace(encode=lambda _text, add_special_tokens: [1, 2, 3])
            args = SimpleNamespace(**paths, model=model, link_evidence=[evidence], arm="native", require_remote=False,
                                   external_schedule=None)
            native = serving._bind(args, load_tokenizer=load_tokenizer)
            args.arm, args.require_remote = "generated", True
            generated = serving._bind(args, load_tokenizer=load_tokenizer)
            self.assertEqual(native[:5], generated[:5])
            self.assertEqual(native[6][0].layer_tiers, (backing.HOST, backing.HOST))
            self.assertEqual(generated[6][0].layer_tiers, (backing.PEER_RELAY, backing.HOST))
            self.assertEqual(generated[5]["actions"][0]["read_links"], ["d2h-b", "h2d-a"])
            args.link_evidence = []
            with self.assertRaisesRegex(ValueError, "link measurement"):
                serving._bind(args, load_tokenizer=load_tokenizer)

            case = root / "frozen-case"
            fields = {key: value for key, value in target.items() if key != "schema"}
            case_args = {**fields, "topology": topology, "fixture": fixture, "compute_nodes": ["gpu-a"],
                         "selector": "torch-unfused", "top_k": 2048, "link_evidence": [evidence], "manual": program}
            manifest = cases.build_case(case, **case_args)
            self.assertEqual(manifest["status"], "bound_unverified")
            self.assertEqual(manifest["workload_sha256"], fingerprint(fixture))
            self.assertEqual(manifest["compiled_preview_sha256"]["manual"], fingerprint(generated[5]))
            copies = [case / path for path in manifest["link_evidence"].values()]
            self.assertEqual(copies[0].read_bytes(), evidence.read_bytes())
            restored_args = SimpleNamespace(**{name: case / f"{name}.json" for name in ("target", "scenario", "topology", "fixture")},
                                            model=model, arm="manual", require_remote=True,
                                            program=case / "manual_program.json", link_evidence=copies, external_schedule=None)
            self.assertEqual(serving._bind(restored_args, load_tokenizer=load_tokenizer)[5:], generated[5:])
            with self.assertRaises(FileExistsError):
                cases.build_case(case, **case_args)

    def test_fixed_request_contract_rejects_implicit_or_out_of_order_schedules(self):
        for changed in ({"input_ids": [1]}, {"arrival_ms": -1}, {"rid": "warm"}, {"max_new_tokens": True}, {"cow": True}):
            with self.subTest(changed=changed):
                fixture = _replay_fixture()
                fixture["requests"][0].update(changed)
                with self.assertRaises(ValueError):
                    serving.validate_fixture(fixture)

    def test_stream_output_and_timing_are_measured_from_native_cumulative_events(self):
        class Engine:
            async def async_generate(self, input_ids, rid, sampling_params, stream):
                self.arguments = (input_ids, rid, sampling_params, stream)

                async def chunks():
                    for count in range(1, 4):
                        await asyncio.sleep(.001)
                        yield {"output_ids": [19] * count, "text": "t" * count, "meta_info": {
                            "id": rid, "completion_tokens": count, "finish_reason": {"type": "length"} if count == 3 else None}}

                return chunks()

        engine = Engine()
        tokenizer = SimpleNamespace(encode=lambda _text, add_special_tokens: [1, 2, 3])
        fixture = _replay_fixture()
        (row,) = rows = asyncio.run(serving._replay(engine, fixture["requests"], serving._tokenize(fixture, tokenizer), 2))
        self.assertEqual(row["session"], "s1")
        self.assertEqual(row["output_ids"], [19] * 3)
        self.assertEqual(len(row["chunks"]), 3)
        self.assertEqual(engine.arguments[:2], ([1, 2, 3], "measured"))
        self.assertEqual(serving._metrics(rows)["tpot_ms"]["samples"], 1)

    def test_qualification_binds_dispatch_outputs_and_selected_packed_bytes(self):
        record = {"effective_schedule_sha256": "effective", "dispatch_sha256": "dispatch", "output_ids_sha256": "outputs", "global_ticks_sha256": None,
                  "packed_kv_sha256_by_rank": ["rank0", "rank1"]}
        serving._compare(record, record, parity=True)
        with self.assertRaisesRegex(ValueError, "actual dispatch"):
            serving._compare(record, {**record, "dispatch_sha256": "drift"}, parity=False)
        with self.assertRaisesRegex(ValueError, "selected KV"):
            serving._compare(record, {**record, "packed_kv_sha256_by_rank": ["rank0", "corrupt"]}, parity=True)

    def test_dp2_explicit_routing_enters_the_frozen_schedule(self):
        fixture = _replay_fixture()
        for group in ("warmup", "requests"):
            row = fixture[group][0]
            row["routed_dp_rank"] = 0
            fixture[group].append({**row, "rid": f"{row['rid']}-rank1", "routed_dp_rank": 1})
        serving.validate_fixture(fixture)
        settings = serving.schedule_inputs(fixture, 2, "torch-unfused", attention_dp_size=2)
        self.assertEqual((settings["engine"]["tp_size"], settings["engine"]["dp_size"], settings["engine"]["enable_dp_attention"]), (2, 2, True))
        self.assertEqual([row["routed_dp_rank"] for row in settings["requests"]], [0, 1])
        with self.assertRaisesRegex(ValueError, "two fixed"):
            serving.schedule_inputs(fixture, 1, "torch-unfused", attention_dp_size=2)
        fixture["requests"][1].pop("routed_dp_rank")
        with self.assertRaisesRegex(ValueError, "both ranks"):
            serving.schedule_inputs(fixture, 2, "torch-unfused", attention_dp_size=2)

    def test_dp2_proof_requires_only_assigned_requests_and_each_local_decode(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"rid": f"req-{rank}", "routed_dp_rank": rank, "output_ids": [1] * (2 + rank)} for rank in range(2)]
            plans = [backing.RankBacking(rank, 8192, 656, 64, 2048, 4096, (backing.HOST,), 0, 8192 * 656,
                                        world_size=2, compute_device=rank, attention_tp_size=1) for rank in range(2)]
            for rank in range(2):
                rids = [f"req-{rank}"]
                observe.save(root / f"dispatch-rank-{rank}.json", {"rank": rank, "finished": rids,
                                                                    "batches": [{"request_ids": rids}]})
                observe.save(root / f"schedule-rank-{rank}.json", {"effective": {}, "parallel": {
                    "attention_dp_size": 2, "attention_tp_size": 1, "attention_dp_rank": rank}})
                observe.append(root / f"backup-rank-{rank}.jsonl", {"event": "backup_dependency_installed"})
                if rank:
                    observe.append(root / f"backup-rank-{rank}.jsonl", {"event": "backup_dependency_exercised"})
                observe.append(root / f"backing-rank-{rank}.jsonl", {"event": "hisparse_cold_backing_bound", "cuda_device": rank,
                                                                      "layer_tiers": [backing.HOST], "attention_tp_size": 1})
                for _ in range(rank + 1):
                    observe.append(root / f"kv-rank-{rank}.jsonl", {"event": "kv_selected_exact", "rank": rank,
                                                                     "layer": 0, "rows": [{"miss_rows": 0}]})
            observed, digests = serving._verify_run(root, rows, plans, diagnostic=True, attention_dp_size=2)
            self.assertEqual((len(observed), len(digests)), (2, 2))
            with self.assertRaisesRegex(ValueError, "fixed request set"):
                serving._verify_run(root, rows, plans, diagnostic=True, attention_dp_size=1)

    def test_staged_padding_bytes_do_not_count_as_attention_consumption(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            rows = [{"rid": "req", "output_ids": [1, 2]}]
            workspace = backing.workspace_bytes(2048, 656)
            plan = backing.RankBacking(0, 8192, 656, 64, 2048, 4096, (backing.PEER_RELAY,), workspace["compute"],
                                       workspace["host"], world_size=1, compute_device=0, peer_device=1,
                                       peer_hbm_cap_bytes=8192 * 656 + workspace["peer"])
            observe.save(root / "dispatch-rank-0.json", {"rank": 0, "finished": ["req"], "batches": [{"request_ids": ["req"]}],
                "relay": {"peer_to_host_read_payload_bytes": 2048 * 656, "host_to_peer_write_payload_bytes": 656,
                          "host_to_compute_read_payload_bytes": 0}, "selected_miss_payload_bytes_by_tier": {backing.PEER_RELAY: 0}})
            observe.save(root / "schedule-rank-0.json", {"effective": {}})
            observe.append(root / "backup-rank-0.jsonl", {"event": "backup_dependency_installed"})
            observe.append(root / "backing-rank-0.jsonl", {"event": "hisparse_cold_backing_bound", "cuda_device": 0,
                           "layer_tiers": [backing.PEER_RELAY], "attention_tp_size": 1})
            with self.assertRaisesRegex(ValueError, "did not supply verified selected KV misses"):
                serving._verify_run(root, rows, [plan], diagnostic=True, attention_dp_size=1)


if __name__ == "__main__":
    unittest.main()
