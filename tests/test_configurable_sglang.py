from copy import deepcopy
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

from integrations.topology_memory import generate_dp2_case as dp2, sglang_global_ticks as global_ticks, sglang_ingress_gate as ingress
from integrations.topology_memory.torch_runtime import SUPPORTED_ACTIONS as CUDA_ACTIONS, SUPPORTED_FEATURES as CUDA_FEATURES
from integrations.topology_memory.generate_serving_case import build_case
from rsi_memory.execution_worker import digest
from rsi_memory.memory_program import MemoryProgram, MemoryScenario
from rsi_memory.topology import fingerprint

from topomem_sglang_rsi.collect_hot_profile import collect
from topomem_sglang_rsi.contract import NCCL_SHA256, SCOPE, case_descriptor, compile_layout, load_case, read_json
from topomem_sglang_rsi.domain import SGLangDomain
from topomem_sglang_rsi.hot_config import (HOT_SLOTS, PARITY_KEYS, check_live_metadata, compile_memory, normalized_schedule,
                                          profile_adapter_sources, profile_runtime_identity, schedule_at_capacity, stock_control_program)
from topomem_sglang_rsi.sglang_eval import run


UUIDS = ["GPU-00000000-0000-0000-0000-000000000001", "GPU-00000000-0000-0000-0000-000000000002"]
RUNTIME = {"nccl_override_sha256": NCCL_SHA256, "adapter_source_sha256": profile_adapter_sources()}


def write(path, value):
    path.write_text(json.dumps(value), encoding="utf-8")


def fixture_case(root, case_id):
    evidence = root / "evidence.data"
    if not evidence.exists():
        evidence.write_bytes(b"CPU mock profile")
    measured = sha256(evidence.read_bytes()).hexdigest()
    gpu = dp2.COMPUTE
    topology = {"schema": 1, "nodes": [{"id": dp2.HOST, "kind": "host", "capacity_bytes": 266612736}]
                + [{"id": node, "kind": "gpu", "capacity_bytes": 25715808} for node in gpu], "bottlenecks": [],
                "links": [{"id": f"{node}-{direction}", "src": source, "dst": dest, "transport": f"pinned_{direction}",
                           "bandwidth_bytes_per_s": 10**9, "latency_ns": 100, "bottlenecks": [], "evidence_sha256": measured}
                          for node in gpu for direction, source, dest in (("h2d", dp2.HOST, node), ("d2h", node, dp2.HOST))]}
    fixture = {"schema": 1, "name": case_id}
    inputs = {}
    for group in ("warmup", "requests"):
        fixture[group] = []
        for rank in range(2):
            rid = f"{case_id}-{group}-{rank}"
            fixture[group].append({"rid": rid, "prompt": "mock independent request", "arrival_ms": 0, "max_new_tokens": 64,
                                   "routed_dp_rank": rank})
            inputs[rid] = {"input_tokens": 5000, "input_ids_sha256": fingerprint(rid)}
    case = root / "cases" / case_id
    build_case(case, topology=topology, gpu_bindings={node: {"ordinal": i, "uuid": UUIDS[i], "name": "mock", "capability": [12, 0]}
                  for i, node in enumerate(gpu)}, fixture=fixture, model_config_sha256="a" * 64, model_weights_sha256="b" * 64,
               pool=dp2.POOL, native_host=dp2.HOST, compute_nodes=gpu, allowed_nodes=dp2.ALLOWED, selector="torch-unfused",
               top_k=dp2.TOP_K, link_evidence=[evidence], attention_dp_size=2)
    spec = case_descriptor(case)
    target, graph, scenario, _ = load_case(case, spec, fixture=True)
    gated = ingress.bind_schedule(read_json(case / "external_schedule.json"), inputs)
    scenario = MemoryScenario.from_dict({**scenario.to_dict(), "schedule_sha256": fingerprint(gated)})
    write(case / "external_schedule.json", gated)
    write(case / "scenario.json", scenario.to_dict())
    write(case / "manifest.json", {**read_json(case / "manifest.json"), "scenario_sha256": scenario.fingerprint(),
                                  "schedule_sha256": scenario.schedule_sha256})
    return case_descriptor(case), inputs, (target, graph, scenario, gated)


def fake_observed_profile(spec):
    # Synthetic records solely exercise validators. No GPU observation is claimed by these tests.
    return {"schema": 1, "kind": "sglang_native_hot_metadata", "status": "observed_native_parity",
            "base_target_sha256": spec["target_sha256"], "base_topology_sha256": spec["topology_sha256"],
            "calibration_workload_sha256": spec["workload_sha256"], "runtime_identity_sha256": profile_runtime_identity(RUNTIME),
            "calibration_adapter_sources_sha256": RUNTIME["adapter_source_sha256"],
            "observations": {str(slots): {"result_sha256": fingerprint(slots), "case_manifest_sha256": fingerprint([slots]),
                "ranks": [{"device_physical_rows": 16448, "device_logical_row_capacity": 16384,
                           "packed_device_payload_bytes": 64739328, "request_metadata_slots_including_pad": 2,
                           "hot_metadata_bytes": slots * 142 + 7168} for _ in range(2)]} for slots in HOT_SLOTS}}


def configure_choice(slots, placement):
    raw = stock_control_program().to_dict()
    when = raw["rules"][0]["when"]
    when["args"][1]["value"] = slots
    when["args"][3]["args"][0]["value"] = slots
    raw["rules"][1:] = placement["rules"]
    return MemoryProgram.from_dict(raw)


class ConfigurableMemoryTests(unittest.TestCase):
    def test_profile_collection_checks_exact_native_sources_and_resident_read_gate(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec, inputs, (target, graph, scenario, external) = fixture_case(root, "public")
            base = root / "cases" / "public"
            synthetic = fake_observed_profile(spec)
            probes = {}
            for slots in HOT_SLOTS:
                cap_target = {**target, "pool": {**target["pool"], "hot_slots": slots}}
                cap_schedule = schedule_at_capacity(external, slots)
                cap_scenario = MemoryScenario.from_dict({**scenario.to_dict(), "target_sha256": fingerprint(cap_target),
                                                         "schedule_sha256": fingerprint(cap_schedule)})
                plan, workers = compile_layout(cap_target, graph, cap_scenario, None)
                case = base if slots == 4096 else root / f"control-{slots}"
                if slots != 4096:
                    case.mkdir()
                    for name in ("fixture.json", "topology.json"):
                        (case / name).write_bytes((base / name).read_bytes())
                    write(case / "target.json", cap_target)
                    write(case / "scenario.json", cap_scenario.to_dict())
                    write(case / "external_schedule.json", cap_schedule)
                    write(case / "manifest.json", {**read_json(base / "manifest.json"), "target_sha256": cap_scenario.target_sha256,
                          "schedule_sha256": cap_scenario.schedule_sha256, "scenario_sha256": cap_scenario.fingerprint()})
                raw = {"status": "completed", "arm": "native", "phase": "parity", "attention_dp_size": 2, "runtime": RUNTIME,
                       "target_sha256": cap_scenario.target_sha256, "schedule_inputs_sha256": cap_scenario.schedule_sha256,
                       "scenario_sha256": cap_scenario.fingerprint(), "topology_sha256": graph.fingerprint(),
                       "compiled_plan_sha256": fingerprint(plan), "program_sha256": plan["program_sha256"], "ranks": [],
                       **{key: fingerprint(key) for key in PARITY_KEYS}}
                raw.update(ingress_contract_sha256=fingerprint(external[ingress.EXTERNAL_FIELD]),
                           input_tokenization_sha256=fingerprint(inputs), global_ticks={"global_ticks_sha256": raw["global_ticks_sha256"]})
                for group in ("warmup", "requests"):
                    raw[group] = [{"rid": rid, **row, "output_ids": [1] * 64} for rid, row in inputs.items() if f"-{group}-" in rid]
                for worker, measured in zip(workers, synthetic["observations"][str(slots)]["ranks"], strict=True):
                    raw["ranks"].append({"parallel": {"attention_dp_size": 2, "attention_dp_rank": worker.rank},
                                         "dispatch": {"selected_miss_payload_bytes_by_tier": {"host": 0}},
                                         "backing": {**measured, "rank": worker.rank, "cuda_device": worker.rank, "hot_slots": slots,
                                             "layer_tiers": list(worker.layer_tiers), "pinned_host_bytes": worker.pinned_host_bytes,
                                             "local_hbm_bytes": 0, "peer_hbm_bytes": 0}})
                path = root / f"native-{slots}.json"
                write(path, raw)
                probes[slots] = (case, path)
            profile = collect(base, probes)
            self.assertEqual(profile["observations"]["8192"]["result_sha256"], sha256(probes[8192][1].read_bytes()).hexdigest())
            wrong = read_json(probes[8192][1])
            wrong["ranks"][0]["dispatch"]["selected_miss_payload_bytes_by_tier"]["host"] = 656
            write(probes[8192][1], wrong)
            with self.assertRaisesRegex(ValueError, "fully resident"):
                collect(base, probes)

    def test_measured_budget_and_lowering_separate_hot_from_cold(self):
        with tempfile.TemporaryDirectory() as temp:
            spec, _, geometry = fixture_case(Path(temp), "train")
            profile = fake_observed_profile(spec)
            stock = stock_control_program()
            compiled = compile_memory(*geometry, profile, stock)
            self.assertEqual(compiled.audit["startup"]["hot_slots"], 6144)
            self.assertEqual(compiled.audit["startup"]["rule_index"], 0)
            self.assertEqual(compiled.audit["startup"]["metadata_bytes_reserved_by_node"], {node: 290816 for node in dp2.COMPUTE})
            self.assertEqual(compiled.audit["full_program_sha256"], stock.fingerprint())
            self.assertEqual(compiled.audit["cold_program_sha256"], compiled.cold_program.fingerprint())
            self.assertEqual(compiled.audit["placement_rule_indices_in_full_program"], [1])
            self.assertTrue(compiled.audit["equivalent_to_strong_stock_control"])
            self.assertTrue(all(set(worker.layer_tiers) == {"host"} for worker in compiled.workers))
            self.assertEqual(compiled.topology.node(dp2.COMPUTE[0]).capacity_bytes, 25715808 - 290816)
            self.assertEqual(normalized_schedule(compiled.external), normalized_schedule(geometry[-1]))
            _, peer = dp2._controls()
            ordinary = compile_memory(*geometry, profile, MemoryProgram.from_dict(peer))
            self.assertEqual(ordinary.audit["startup"]["hot_slots"], 4096)
            self.assertTrue(any(worker.peer_hbm_bytes for worker in ordinary.workers))
            with self.assertRaisesRegex(ValueError, "budget|cap"):
                compile_memory(*geometry, profile, configure_choice(6144, peer))
            with self.assertRaisesRegex(ValueError, "blocked until"):
                compile_memory(*geometry, None, stock)
            unmeasured = deepcopy(profile)
            unmeasured["observations"].pop("8192")
            with self.assertRaisesRegex(ValueError, "all three"):
                compile_memory(*geometry, unmeasured, stock)

    def test_configuration_and_cold_features_are_separate(self):
        self.assertNotIn(("configure", "hot_capacity"), CUDA_ACTIONS)
        self.assertNotIn("candidate_hot_slots", CUDA_FEATURES)
        with tempfile.TemporaryDirectory() as temp:
            spec, _, geometry = fixture_case(Path(temp), "train")
            profile = fake_observed_profile(spec)
            for index, feature in ((0, "layer"), (1, "candidate_hot_slots")):
                raw = stock_control_program().to_dict()
                raw["rules"][index]["score"] = {"op": "feature", "name": feature}
                with self.assertRaisesRegex(ValueError, "cannot observe"):
                    compile_memory(*geometry, profile, MemoryProgram.from_dict(raw))
            raw = stock_control_program().to_dict()
            raw["rules"][0]["when"] = {"op": "const", "value": 0}
            with self.assertRaisesRegex(MemoryError, "selected no measured"):
                compile_memory(*geometry, profile, MemoryProgram.from_dict(raw))

    def test_domain_routes_native_aa_candidate_parity_then_scoring(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            cases = {name: fixture_case(root, name) for name in ("train", "validation", "test")}
            (root / "cases" / "test" / "fixture.json").unlink()
            profile = fake_observed_profile(cases["train"][0])
            calls, receipts = [], {}

            class MockExecutor:
                def __init__(self, source, source_manifest_sha256, gpu_indices, gpu_uuids, **unused):
                    self.config = {"source": source, "source_manifest_sha256": source_manifest_sha256,
                                   "gpu_indices": gpu_indices, "gpu_uuids": gpu_uuids, "environment": {}}

                def status(self, job_id):
                    return receipts.get(job_id, {"status": "not_found"})

                def run(self, job_id, module, payload, *, timeout):
                    calls.append((job_id, payload))
                    program = None if payload["program"] is None else MemoryProgram.from_dict(payload["program"])
                    geometry = cases[payload["case"]["id"]][2]
                    audit = compile_memory(*geometry, profile, program, native_hot=payload["native_hot"]).audit
                    result = {"scope": SCOPE, "case": payload["case"], "phase": payload["phase"], "arm": payload["arm"],
                              "feasible": True, "score": 10 if payload["phase"] == "performance" else None,
                              "program_sha256": audit["full_program_sha256"], "memory_compilation": audit,
                              "raw_result_sha256": fingerprint(payload), "feedback": {"cold_miss_rows_by_rank_layer": []},
                              "metrics": {"ranks": [], "timing": {}, **{key: "9" * 64 for key in PARITY_KEYS},
                                          "input_tokenization_sha256": fingerprint(payload["expected_inputs"]),
                                          "global_ticks_sha256": fingerprint(payload["case"]["id"])}}
                    if payload["cross_cap_reference"] is not None:
                        result["cross_capacity_native_parity"] = {"matched": True}
                    request = digest({"operation": "run", "job_id": job_id, "module": module, "payload": payload,
                                      "timeout": timeout, **self.config})
                    receipts[job_id] = {"status": "completed", "request_sha256": request, "result": result}
                    return receipts[job_id]

                def close(self):
                    pass

            config = {"schema": 2, "experiment_id": "mock-configurable", "source_manifest_sha256": "f" * 64,
                      "launch_sha256": "e" * 64, "receipt_directory": "receipts", "local_cases": "cases",
                      "cases": {name: [row[0]] for name, row in cases.items()}, "token_inputs": {name: row[1] for name, row in cases.items()},
                      "paired_block_ids": {"validation": ["v1"], "test": ["t1"]}, "deadline": 10, "job_timeout": 30,
                      "executor": {"host": "mock", "worker": "mock", "root": "mock", "python": "mock", "source": "mock",
                                   "gpu_indices": [0, 1], "gpu_uuids": UUIDS},
                      "initial_program": stock_control_program().to_dict(), "hot_profile": profile}
            adapter = SGLangDomain(config, root, executor_factory=MockExecutor)
            domain = adapter.domain()
            equivalent = stock_control_program().to_dict()
            equivalent["rules"][1]["score"] = {"op": "const", "value": 777}
            effective = domain.effective_program_fingerprint
            self.assertEqual(effective(MemoryProgram.from_dict(equivalent)), effective(domain.manual_program))
            self.assertNotEqual(effective(configure_choice(8192, {"rules": equivalent["rules"][1:]})), effective(domain.manual_program))
            self.assertNotIn(cases["test"][0]["workload_sha256"], json.dumps(domain.training_context))
            self.assertEqual(domain.evaluate("train", domain.initial_program, "a")["score"], 10)
            self.assertEqual([(job["arm"], job["phase"], job["native_hot"]) for _, job in calls],
                             [("native", "parity", None)] * 2 + [("native", "parity", 6144)] * 2
                             + [("generated", "parity", None), ("generated", "performance", None)])
            self.assertEqual(calls[2][1]["cross_cap_reference"]["job_id"], calls[0][0])
            self.assertEqual(calls[3][1]["cross_cap_reference"]["job_id"], calls[0][0])
            self.assertEqual(calls[4][1]["qualification"]["job_id"], calls[2][0])
            self.assertEqual(calls[5][1]["qualification"]["job_id"], calls[4][0])
            paired = domain.evaluate_pair("validation", {"native": None, "manual": domain.manual_program,
                                                         "candidate": domain.initial_program}, "b")
            self.assertTrue(all(row["feasible"] for row in paired.values()))
            self.assertTrue(paired["candidate"]["metrics"]["equivalent_to_strong_stock_control"])
            self.assertNotEqual(adapter.contract["ingress_contract_sha256_by_case"]["train"],
                                adapter.contract["ingress_contract_sha256_by_case"]["validation"])
            adapter.close()

    def test_evaluator_checks_live_geometry_and_cross_capacity_parity(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            spec, inputs, geometry = fixture_case(root, "train")
            profile = fake_observed_profile(spec)
            launch, source = "a" * 64, "b" * 64
            base = {"schema": 1, "case": spec, "program": None, "arm": "native", "phase": "parity", "qualification": None,
                    "deadline": 10, "launch_sha256": launch, "source_manifest_sha256": source, "expected_inputs": inputs,
                    "hot_profile": profile, "native_hot": None, "cross_cap_reference": None}

            def execute(command, *, check):
                directory = Path(command[command.index("--output") + 1])
                directory.mkdir()
                cap = read_json(Path(command[command.index("--target") + 1]))["pool"]["hot_slots"]
                compiled = compile_memory(*geometry, profile, None, native_hot=None if cap == 4096 else cap)
                audit = compiled.audit
                receipt = {"status": "completed", "arm": "native", "phase": "parity", "attention_dp_size": 2, "runtime": RUNTIME,
                           "target_sha256": audit["derived_target_sha256"], "topology_sha256": audit["derived_topology_sha256"],
                           "scenario_sha256": audit["derived_scenario_sha256"], "schedule_inputs_sha256": audit["derived_schedule_sha256"],
                           "program_sha256": audit["cold_program_sha256"], "compiled_plan_sha256": audit["cold_plan_sha256"],
                           "effective_schedule_sha256": "1" * 64, "dispatch_sha256": "2" * 64,
                           "output_ids_sha256": "3" * 64, "packed_kv_sha256_by_rank": ["4" * 64, "5" * 64], "ranks": []}
                self.assertEqual(read_json(Path(command[command.index("--external-schedule")+1])), compiled.external)
                receipt.update(ingress_contract_sha256=audit["ingress_contract_sha256"],
                               ingress={"contract_sha256": audit["ingress_contract_sha256"]},
                               global_ticks_sha256="a"*64, global_ticks={"global_ticks_sha256": "a"*64},
                               global_tick_receipt_sha256_by_rank=["d"*64, "e"*64], input_tokenization_sha256=fingerprint(inputs))
                for group in ("warmup", "requests"):
                    receipt[group] = [{"rid": rid, **row} for rid, row in inputs.items() if f"-{group}-" in rid]
                for worker, measured in zip(compiled.workers, profile["observations"][str(cap)]["ranks"], strict=True):
                    (directory / f"kv-rank-{worker.rank}.jsonl").write_text(json.dumps({"event": "kv_selected_exact", "rank": worker.rank,
                         "layer": 0, "rows": [{"miss_rows": 0}]}) + "\n", encoding="utf-8")
                    receipt["ranks"].append({"backing": {"rank": worker.rank, "cuda_device": worker.rank, "hot_slots": cap,
                        "layer_tiers": list(worker.layer_tiers), "pinned_host_bytes": worker.pinned_host_bytes, "local_hbm_bytes": 0,
                        "peer_hbm_bytes": 0, "relay_workspace_bytes": {}, **measured},
                        "schedule": {"worker_limits": {"max_running_requests": 1, "max_total_num_tokens": 16384}},
                        "dispatch": {"selected_miss_payload_bytes_by_tier": {"host": 0}, "backup_payload_bytes_by_tier": {"host": 1024},
                                     "relay": None, "torch_cuda_memory": {}}})
                check_live_metadata(receipt, profile, cap)
                if directory.parent.name == "different-output":
                    receipt["output_ids_sha256"] = "6" * 64
                if directory.parent.name == "different-global-ticks":
                    receipt.update(global_ticks_sha256="b"*64, global_ticks={"global_ticks_sha256": "b"*64})
                write(directory / "result.json", receipt)
                return SimpleNamespace(returncode=0)

            job = root / "jobs" / "base"
            job.mkdir(parents=True)
            first = run(base, job / "result.json", model="mock", launch_sha256=launch, source_sha256=source, execute=execute)
            write(job / "result.json", first)
            write(job / "receipt.json", {"schema": 2, "status": "completed", "job_id": job.name, "source_manifest_sha256": source,
                  "returncode": 0, "process_group_remaining": False, "process_group_cleanup_required": False, "remaining_compute_pids": "",
                  "result": first, "result_canonical_sha256": digest(first), "result_sha256": sha256((job / "result.json").read_bytes()).hexdigest()})
            tuned = {**base, "native_hot": 6144, "cross_cap_reference": {"job_id": job.name, "raw_result_sha256": first["raw_result_sha256"]}}
            other = root / "jobs" / "tuned"
            other.mkdir()
            second = run(tuned, other / "result.json", model="mock", launch_sha256=launch, source_sha256=source, execute=execute)
            self.assertTrue(second["cross_capacity_native_parity"]["matched"])
            bad = root / "jobs" / "different-output"
            bad.mkdir()
            with self.assertRaisesRegex(RuntimeError, "cross-capacity native parity"):
                run(tuned, bad / "result.json", model="mock", launch_sha256=launch, source_sha256=source, execute=execute)
            bad_ticks = root / "jobs" / "different-global-ticks"
            bad_ticks.mkdir()
            with self.assertRaisesRegex(RuntimeError, "cross-capacity native parity"):
                run(tuned, bad_ticks / "result.json", model="mock", launch_sha256=launch, source_sha256=source, execute=execute)


if __name__ == "__main__":
    unittest.main()
