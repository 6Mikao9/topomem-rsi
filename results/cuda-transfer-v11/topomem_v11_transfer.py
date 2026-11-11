"""Run the frozen four-scenario generator transfer study in four independent phases."""

import argparse
from collections import Counter
from contextlib import contextmanager, ExitStack
from hashlib import sha256
import json
from math import isfinite
import os
from pathlib import Path
import sqlite3
import sys
import time


RUNTIME = "877873960cd4da3c332891495fdc86d30fa59725b879d0fcffccb81fdef0c767"
SOURCES = {"seed": "42699e4d5148ee3dc3f41f7ffeab8a3020e8fb75a5a433940d7ae88e4c08ca38",
           "admitted": "7d9d37e11e08ac084b3b2b4eb3579178652fc5258dc53c2f502b0f995621c201"}
MARGINS = {"agent_tools": 0.5, "chatbot_growth": 1.5, "long_document_qa": 3.5, "rollout_cow": 1.0}
ANCHORS = ("native", "initial", "manual-ir")
REFERENCES = ("seed", "native", "manual-direct")
GENERATION = {"round_input": 2, "slots_per_source": 2, "model": "deepseek-v4.1-flash",
              "max_output_tokens": 65536, "reasoning_effort": "high", "request_timeout_seconds": 240,
              "provider_calls": 16, "maximum_output_tokens": 1048576}
BUDGET = {"baseline_jobs": 12, "max_candidate_jobs": 16, "max_test_jobs": 192,
          "blocks": 12, "family_alpha": 0.05, "family_comparisons": 12,
          "bootstrap_draws": 10000, "bootstrap_seed": 20261111}


def generation_order():
    return [[kind, slot, source] for index, kind in enumerate(MARGINS) for slot in range(2)
            for source in (tuple(SOURCES) if (index+slot) % 2 == 0 else tuple(reversed(SOURCES)))]


def raw_hash(path):
    return sha256(path.read_bytes()).hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def fingerprint(value):
    return sha256(canonical(value)).hexdigest()


def read(path, expected=None):
    raw = path.read_bytes()
    if expected is not None and sha256(raw).hexdigest() != expected:
        raise ValueError(f"frozen artifact digest mismatch {path.name}")
    return json.loads(raw)


def write(path, value):
    raw = canonical(value) + b"\n"
    with path.open("xb") as stream:
        stream.write(raw)
    return sha256(raw).hexdigest()


def runtime(args):
    manifest = read(args.source_manifest, RUNTIME)
    root = args.source_root.resolve(strict=True)
    for member, digest in manifest["files"].items():
        path = (root / member).resolve(strict=True)
        if not path.is_relative_to(root) or raw_hash(path) != digest:
            raise ValueError(f"untrusted local evaluator source {member}")
    sys.path.insert(0, str(root))


def orders(labels):
    labels = sorted(labels)
    count = len(labels)
    base = [0] + [(position+1)//2 if position % 2 else count-position//2 for position in range(1, count)]
    rows = [[labels[(position+shift) % count] for position in base] for shift in range(count)]
    if count % 2:
        rows += [row[::-1] for row in rows.copy()]
    return [rows[index % len(rows)] for index in range(BUDGET["blocks"])]


def prepare(args):
    from rsi_memory.harness import validate_harness
    from rsi_memory.memory_program import MemoryProgram
    from integrations.topology_memory.torch_runtime import SUPPORTED_ACTIONS, SUPPORTED_FEATURES
    from rsi_memory.topology import fingerprint as worker_fingerprint

    sealed = {}
    executor_profiles = {}
    sources = {}
    for label, digest in SOURCES.items():
        path = args.harness_directory / f"{digest}.py"
        raw = path.read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError(f"untrusted generator {label}")
        validate_harness(raw.decode())
        sources[label] = raw
    for kind in MARGINS:
        path = args.config_directory / f"{kind}.json"
        config = read(path)
        train, heldout = (config["cases"][split] for split in ("train", "test"))
        if any(len(cases) != 2 or any(case["workload"]["kind"] != kind for case in cases) for cases in (train, heldout)):
            raise ValueError(f"expected two existing cases of scenario {kind} in each split")
        if {case["workload"]["source"]["content_sha256"] for case in train} & {
                case["workload"]["source"]["content_sha256"] for case in heldout}:
            raise ValueError(f"training and held-out content overlap for {kind}")
        for case in train + heldout:
            if any(case[key] != train[0][key] for key in ("topology", "target", "device_map", "compute_node")):
                raise ValueError(f"physical target differs within scenario {kind}")
        profile = {key: config["executor"][key] for key in ("gpu_indices", "gpu_uuids", "environment")}
        if (len(profile["gpu_indices"]) < 2 or len(set(profile["gpu_indices"])) != len(profile["gpu_indices"])
                or len(profile["gpu_uuids"]) != len(profile["gpu_indices"])
                or len(set(profile["gpu_uuids"])) != len(profile["gpu_indices"])):
            raise ValueError("transfer requires a measured multi-device topology in every scenario")
        executor_profiles[kind] = profile
        anchors = {"native": None}
        for label, key in (("initial", "initial_program"), ("manual-ir", "manual_program")):
            program = MemoryProgram.from_dict(config[key])
            program.validate_actions(SUPPORTED_ACTIONS)
            program.validate_features(SUPPORTED_FEATURES)
            if len(program.rules) < 6:
                raise ValueError("shared anchors must support the admitted source's rule index")
            anchors[label] = program.to_dict()
        blocks = config["paired_block_ids"]["test"]
        if len(blocks) != BUDGET["blocks"] or len(set(blocks)) != len(blocks):
            raise ValueError(f"expected 12 unique existing held-out blocks for {kind}")
        if (config["warmup"], config["samples"]) != (2, 3):
            raise ValueError("unexpected frozen trial sampling budget")
        sealed[kind] = {"config_sha256": raw_hash(path), "train": {"cases": train, "anchors": anchors},
                        "test": {"cases": heldout}, "block_ids": blocks,
                        "warmup": config["warmup"], "samples": config["samples"], "timeout": config["timeout"],
                        "train_case_sha256": [worker_fingerprint(case) for case in train],
                        "test_case_sha256": [worker_fingerprint(case) for case in heldout]}
    args.study.mkdir(exist_ok=False, parents=True)
    for label, raw in sources.items():
        with (args.study / f"{label}.py").open("xb") as stream:
            stream.write(raw)
    for kind, row in sealed.items():
        for phase in ("train", "test"):
            row[phase + "_sha256"] = write(args.study / f"{kind}-{phase}.json", row.pop(phase))
    manifest = {"schema": 1, "kind": "frozen_v7_generators_transfer_on_v11", "runtime_sha256": RUNTIME,
                "driver_sha256": raw_hash(Path(__file__)), "generators": SOURCES, "generation": GENERATION,
                "budget": BUDGET, "margins_ms": MARGINS, "operator_profiles": executor_profiles, "scenarios": sealed,
                "generation_schedule": generation_order(),
                "test_schedule_by_eligible_source_set": {key: orders(["native", "manual-direct", *labels])
                    for key, labels in (("both", SOURCES), ("seed", ("seed",)), ("admitted", ("admitted",)), ("neither", ()))},
                "scope": {"agent_tools": "same-domain-engine-transfer", "other": "cross-scenario-transfer",
                          "runtime": "fixed synthetic CUDA block-sparse attention", "new_rsi_round": False,
                          "test_split": "existing cases, no validation or post-selection changes"}}
    print(json.dumps({"preregistration_sha256": write(args.study / "preregistration.json", manifest)}))


class Journal:
    def __init__(self, path, plan_sha256):
        self.db = sqlite3.connect(path, timeout=0)
        self.lease = sqlite3.connect(path.with_name("lease.sqlite"), timeout=0)
        try:
            self.lease.execute("BEGIN EXCLUSIVE")
            self.db.execute("CREATE TABLE IF NOT EXISTS records (key TEXT PRIMARY KEY, data TEXT NOT NULL)")
            identity = {"preregistration_sha256": plan_sha256}
            if self.get("identity") is None:
                self.save("identity", identity)
            elif self.get("identity") != identity:
                raise ValueError("journal belongs to a different preregistration")
        except BaseException:
            self.close()
            raise

    def close(self):
        self.db.close()
        self.lease.close()

    def get(self, key):
        row = self.db.execute("SELECT data FROM records WHERE key = ?", (key,)).fetchone()
        return json.loads(row[0]) if row is not None else None

    def save(self, key, data):
        with self.db:
            self.db.execute("INSERT OR REPLACE INTO records VALUES (?, ?)", (key, canonical(data)))
        return data

    def effect(self, key, details, operation):
        previous = self.get(key)
        if previous is not None:
            if previous["input"] != details:
                raise ValueError(f"resumed effect differs {key}")
            if previous["status"] == "pending":
                return self.save(key, {**previous, "status": "interrupted_unknown"})
            return previous
        row = {"input": details, "status": "pending"}
        self.save(key, row)
        started = time.monotonic()
        try:
            result = operation()
            canonical(result)
        except Exception as exc:
            row.update(status="failed", failure=type(exc).__name__)
        else:
            row.update(status="completed", result=result)
        row["elapsed_seconds"] = time.monotonic() - started
        return self.save(key, row)


def seal(study, name, content):
    path = study / f"{name}.json"
    if path.exists():
        if read(path) != content:
            raise ValueError(f"existing phase seal differs {name}")
        return raw_hash(path)
    return write(path, content)


def finish(study, journal, name, data):
    digest = seal(study, name, data)
    previous = journal.get("phase/" + name)
    if previous is not None and previous != digest:
        raise ValueError(f"journal phase seal differs {name}")
    journal.save("phase/" + name, digest)
    return digest


def require(study, journal, name):
    digest = journal.get("phase/" + name)
    if digest is None:
        raise ValueError(f"required phase has not settled {name}")
    return read(study / f"{name}.json", digest)


@contextmanager
def executors(args, plan):
    from rsi_memory.execution_client import PersistentExecutor

    configs = read(args.executor_config)
    if configs.keys() != plan["operator_profiles"].keys():
        raise ValueError("one private executor configuration is required per scenario")
    for kind, config in configs.items():
        if (config.keys() != {"host", "worker", "root", "python", "source", "gpu_indices", "gpu_uuids", "environment"}
                or any(config[key] != val for key, val in plan["operator_profiles"][kind].items())):
            raise ValueError("private executor differs from the preregistered multi-GPU target")
    clients = {}
    with ExitStack() as stack:
        def worker(kind):
            config = configs[kind]
            key = fingerprint(config)
            if key not in clients:
                client = PersistentExecutor(**config, source_manifest_sha256=RUNTIME, audit_directory=args.study / "receipts")
                clients[key] = stack.enter_context(client)
            return clients[key]
        yield worker


def case_evidence(report, case, expected_hash, samples, program_sha):
    from rsi_memory.topology import HardwareTopology

    diagnostic = report["diagnostic"]
    expected_cap = {node["id"]: node["capacity_bytes"] for node in case["topology"]["nodes"]}
    reserved = diagnostic["managed_reserved_bytes_by_node"]
    peak = diagnostic["managed_peak_bytes_by_node"]
    if (report["name"] != case["name"] or report["case_sha256"] != expected_hash
            or diagnostic["status"] != "verified" or diagnostic["program_sha256"] != program_sha
            or diagnostic["topology_sha256"] != HardwareTopology.from_dict(case["topology"]).fingerprint()
            or diagnostic["live_references_at_end"] != 0 or not diagnostic["selected_and_output_sha256"]
            or reserved.keys() != expected_cap.keys() or peak.keys() != expected_cap.keys()
            or any(not 0 <= peak[key] <= reserved[key] <= expected_cap[key] for key in expected_cap)
            or len(report["timings"]) != samples):
        raise ValueError("GPU case failed its frozen semantic, ownership or managed capacity checks")
    for row in report["timings"]:
        if (row["execution_signature"] != report["execution_signature"] or row.get("status") != "timed"
                or row.get("selected_and_output_sha256") is not None or "work_counts" in row):
            raise ValueError("verified execution signature differs from uninstrumented timing")
    return {"name": report["name"], "case_sha256": report["case_sha256"],
            "attention_sha256": diagnostic["selected_and_output_sha256"],
            "execution_signature": report["execution_signature"],
            "physical_trace_sha256": fingerprint({"actions": [{key: value for key, value in row.items() if key != "rule"}
                                                              for row in diagnostic["actions"]],
                                                  "semantics": diagnostic["semantics"]}),
            "score_ms": report["median_elapsed_ns"] / 1e6, "peak_bytes_by_node": peak,
            "wire_bytes": diagnostic["link_bytes"], "fetch_misses": diagnostic["total_fetch_misses"]}


def gpu_trial(journal, worker, plan, kind, key, cases, arm, program, *, split):
    from rsi_memory.cuda_domain import _feedback
    from rsi_memory.topology import fingerprint as worker_fingerprint

    scenario = plan["scenarios"][kind]
    hashes = scenario[split + "_case_sha256"]
    if hashes != [worker_fingerprint(case) for case in cases]:
        raise ValueError("case payload changed after preregistration")
    program_sha = None if program is None else worker_fingerprint(program)
    payload = {"schema": 1, "arm": arm, "program": program, "cases": cases,
               "warmup": scenario["warmup"], "samples": scenario["samples"],
               "expected_gpu_uuids": plan["operator_profiles"][kind]["gpu_uuids"]}
    details = {"case_sha256": hashes, "arm": arm, "program_sha256": program_sha,
               "payload_sha256": worker_fingerprint(payload), "timeout": scenario["timeout"]}
    job = "transfer-" + fingerprint([plan["driver_sha256"], kind, key, details])[:48]
    details["receipt_id"] = job

    def submit():
        receipt = worker.run(job, "integrations.topology_memory.torch_eval", payload, timeout=scenario["timeout"])
        evidence = {"receipt_id": job, "result_sha256": receipt.get("result_sha256"),
                    "receipt_status": receipt["status"], "feasible": False, "score": None, "cases": []}
        if receipt["status"] != "completed":
            return evidence
        result = receipt["result"]
        if result.get("scope") != "cuda_text_derived_fixed_block_sparse" or type(result.get("feasible")) is not bool:
            raise ValueError("GPU result has the wrong scope or feasibility status")
        if result["feasible"]:
            if len(result["cases"]) != len(cases) or not isfinite(result["score"]) or result["score"] <= 0:
                raise ValueError("GPU result lacks a verified case or positive score")
            evidence.update(feasible=True, score=result["score"], cases=[case_evidence(report, case, digest,
                            scenario["samples"], program_sha) for report, case, digest in zip(result["cases"], cases, hashes)])
        else:
            evidence["failure_type"] = result["error_type"]
        if split == "train":
            evidence["training_feedback"] = _feedback(result)
        return evidence

    return journal.effect(f"gpu/{kind}/{key}", details, submit)


def outcome(effect):
    result = effect.get("result")
    if result is None:
        return {"feasible": False, "score": None, "failure_type": effect.get("failure", effect["status"]),
                "receipt_id": effect["input"].get("receipt_id"), "receipt_status": effect["status"]}
    return result


def signature(row):
    return [(case["case_sha256"], case["attention_sha256"]) for case in row["cases"]]


def baseline(args, plan, journal):
    evidence = {}
    with executors(args, plan) as worker:
        for kind, scenario in plan["scenarios"].items():
            train = read(args.study / f"{kind}-train.json", scenario["train_sha256"])
            results = {label: outcome(gpu_trial(journal, worker(kind), plan, kind, f"anchor/{label}", train["cases"],
                       "native_lru" if label == "native" else "generated", train["anchors"][label], split="train"))
                       for label in ANCHORS}
            ref = results["native"]
            for label in ANCHORS:
                row = results[label]
                if label != "native" and row["feasible"] and ref["feasible"] and signature(row) != signature(ref):
                    row = {**row, "feasible": False, "score": None, "failure_type": "selected_bytes_or_attention_mismatch"}
                results[label] = row
            evidence[kind] = results
    digest = finish(args.study, journal, "baseline", evidence)
    print(json.dumps({"baseline_sha256": digest, "feasible": {kind: {label: row["feasible"] for label, row in rows.items()}
                                                            for kind, rows in evidence.items()}}))


def training_context(train, controls):
    from rsi_memory.cuda_domain import RULE_SEMANTICS
    from rsi_memory.memory_program import FEATURE_UNITS
    from rsi_memory.topology import fingerprint as worker_fingerprint
    from integrations.topology_memory.torch_runtime import SUPPORTED_FEATURES

    evidence = {label: {"status": "feasible" if controls[label]["feasible"] else "unavailable",
                        "score": controls[label]["score"],
                        "feedback": (controls[label].get("training_feedback", {}) if controls[label]["feasible"]
                                     else {"failure_type": controls[label].get("failure_type", "unavailable")})}
                for label in ANCHORS}
    eligible = [label for label in ANCHORS[1:] if controls[label]["feasible"]]
    eligible.sort(key=lambda label: (controls[label]["score"], worker_fingerprint(train["anchors"][label])))
    candidates = {worker_fingerprint(train["anchors"][label]): {"program": train["anchors"][label],
                                                               "score": controls[label]["score"]} for label in eligible}
    contexts = [{"name": case["name"], "topology": case["topology"], "geometry": case["compile"],
                 "workload_kind": case["workload"]["kind"], "compute_node": case["compute_node"],
                 "external_event_counts": dict(Counter(row["op"] for row in case["workload"]["events"]))}
                for case in train["cases"]]
    domain = {"cases": contexts, "feature_units": {name: FEATURE_UNITS[name] for name in sorted(SUPPORTED_FEATURES)},
              "runtime_semantics": RULE_SEMANTICS, "score_unit": "sum_case_median_ms", "lower_is_better": True}
    return {"domain": domain, "parent_ids": list(candidates), "candidates": candidates,
            "evidence_ids": list(evidence), "evidence": evidence}


def program_request(source, slot, context):
    from rsi_memory.memory_program import MAX_RULES
    from rsi_memory.topology_rsi import ENTRY_CONTRACT, PROGRAM_SYSTEM, _entry
    from integrations.topology_memory.torch_runtime import SUPPORTED_ACTIONS

    context = {**context, "round": GENERATION["round_input"], "slot": slot,
               "harness_source": source, "trusted_entry_contract": ENTRY_CONTRACT}
    candidates = context["candidates"]
    decision = _entry(source, "build_prompt", context, candidates)
    events_and_actions = {}
    for event, action in sorted(SUPPORTED_ACTIONS):
        events_and_actions.setdefault(event, []).append(action)
    rule = {"event": "event", "action": "action", "when": "AST", "score": "AST"}
    contract = {"schema": 1, "kind": "kv_memory_program", "max_rules": MAX_RULES, "events_and_actions": events_and_actions,
                "required_rule": {"event": "allocate", "action": "place"},
                "execution_labels": "Action counts and receipt labels are read-only observations."}
    envelope = {"round": context["round"], "slot": slot, "trusted_domain": context["domain"],
                "trusted_program_contract": contract, "generator_decision": decision,
                "selected_training_evidence": {key: context["evidence"][key] for key in decision["evidence"]}}
    operator = decision["operator"]
    if operator == "construct":
        envelope["response_shape"] = {"schema": 1, "kind": "kv_memory_program", "rules": [rule]}
    else:
        envelope["selected_parents"] = {key: candidates[key] for key in decision["parents"]}
        envelope["response_shape"] = {"rule": rule}
        envelope["trusted_edit_instruction"] = (
            "Return the replacement for the chosen first-parent rule." if operator == "mutate" else
            "Return a replacement inspired by the second parent. The runtime also appends the second parent's first rule.")
    system = PROGRAM_SYSTEM + "\nArchived guidance may mention prior-run labels. Only IDs in the supplied evidence are current."
    payload = {"model": GENERATION["model"], "messages": [{"role": "system", "content": system},
               {"role": "user", "content": canonical(envelope).decode()}], "max_tokens": GENERATION["max_output_tokens"],
               "reasoning_effort": GENERATION["reasoning_effort"], "thinking": {"type": "enabled"}, "stream": True,
               "stream_options": {"include_usage": True}}
    return decision, payload


def program_response(raw, decision, context):
    from rsi_memory.memory_program import MemoryProgram
    from integrations.topology_memory.torch_runtime import SUPPORTED_ACTIONS, SUPPORTED_FEATURES

    if type(raw) is not str or len(raw) > 100000:
        raise ValueError("provider did not return bounded JSON text")
    program = json.loads(raw)
    parents = decision["parents"]
    if parents:
        if type(program) is not dict or program.keys() != {"rule"}:
            raise ValueError("invalid mutate/crossover response shape")
        rule = program["rule"]
        program = json.loads(canonical(context["candidates"][parents[0]]["program"]))
        program["rules"][decision["rule"]] = rule
        if len(parents) == 2:
            program["rules"].append(context["candidates"][parents[1]]["program"]["rules"][0])
    program = MemoryProgram.from_dict(program)
    program.validate_actions(SUPPORTED_ACTIONS)
    program.validate_features(SUPPORTED_FEATURES)
    return program.to_dict(), program.fingerprint()


def generate(args, plan, journal, requester=None):
    from rsi_memory.model_client import load_api_key, make_requester, token_usage

    anchors = require(args.study, journal, "baseline")
    contexts = {kind: training_context(read(args.study / f"{kind}-train.json", scenario["train_sha256"]), anchors[kind])
                for kind, scenario in plan["scenarios"].items()}
    context_sha = finish(args.study, journal, "contexts", contexts)
    sources = {}
    for label, digest in SOURCES.items():
        raw = (args.study / f"{label}.py").read_bytes()
        if sha256(raw).hexdigest() != digest:
            raise ValueError("frozen generator bytes changed")
        sources[label] = raw.decode()
    key = (load_api_key(args.key_file) if requester is None and any(rows["native"]["feasible"] for rows in anchors.values())
           else None)

    def call(payload):
        if requester is None:
            response = make_requester(key, timeout=GENERATION["request_timeout_seconds"])(payload)
        else:
            response = requester(payload)
        choice = (response.get("choices") or [{}])[0]
        return {"content": choice.get("message", {}).get("content"), "finish_reason": choice.get("finish_reason"),
                "usage": token_usage(response)}

    generated = {}
    for kind, slot, label in plan["generation_schedule"]:
        slot_id = f"{kind}/{label}/{slot}"
        row = {"source": label, "source_sha256": SOURCES[label], "scenario": kind, "slot": slot,
               "context_sha256": fingerprint(contexts[kind]), "decision": None, "program": None, "program_sha256": None}
        if not anchors[kind]["native"]["feasible"]:
            generated[slot_id] = {**row, "status": "native_training_control_unavailable"}
            continue
        try:
            decision, payload = program_request(sources[label], slot, contexts[kind])
        except (ValueError, TypeError, TimeoutError, KeyError) as exc:
            generated[slot_id] = {**row, "status": "invalid_harness_decision", "failure": type(exc).__name__}
            continue
        row["decision"] = decision
        effect = journal.effect("api/" + slot_id, {"source_sha256": row["source_sha256"], "context_sha256": row["context_sha256"],
                                                    "payload": payload}, lambda: call(payload))
        if effect["status"] != "completed":
            generated[slot_id] = {**row, "status": "api_" + effect["status"], "failure": effect.get("failure")}
            continue
        result = effect["result"]
        row.update(usage=result["usage"], finish_reason=result["finish_reason"], content_sha256=fingerprint(result["content"]))
        try:
            program, digest = program_response(result["content"], decision, contexts[kind])
        except (ValueError, TypeError, KeyError, RecursionError) as exc:
            generated[slot_id] = {**row, "status": "invalid_program", "failure": type(exc).__name__}
        else:
            generated[slot_id] = {**row, "status": "program_ready", "program": program, "program_sha256": digest}
        print(json.dumps({"slot": slot_id, "status": generated[slot_id]["status"]}), flush=True)
    digest = finish(args.study, journal, "generated", {"context_sha256": context_sha, "slots": generated})
    print(json.dumps({"generated_sha256": digest, "statuses": dict(Counter(row["status"] for row in generated.values()))}))


def select(args, plan, journal):
    anchors = require(args.study, journal, "baseline")
    generated = require(args.study, journal, "generated")
    results, winners = {}, {}
    with executors(args, plan) as worker:
        trains = {kind: read(args.study / f"{kind}-train.json", scenario["train_sha256"])
                  for kind, scenario in plan["scenarios"].items()}
        for kind, slot, label in plan["generation_schedule"]:
            slot_id = f"{kind}/{label}/{slot}"
            row = generated["slots"][slot_id]
            if row["status"] != "program_ready":
                results[slot_id] = {"slot_id": slot_id, "status": row["status"], "feasible": False, "score": None}
                continue
            trial = gpu_trial(journal, worker(kind), plan, kind, f"candidate/{label}/{slot}", trains[kind]["cases"],
                              "generated", row["program"], split="train")
            result = outcome(trial)
            if result["feasible"] and signature(result) != signature(anchors[kind]["native"]):
                result = {**result, "feasible": False, "score": None, "failure_type": "selected_bytes_or_attention_mismatch"}
            status = "feasible" if result["feasible"] else "train_rejected"
            results[slot_id] = {"slot_id": slot_id, "status": status, "program_sha256": row["program_sha256"], **result}
    for kind in plan["scenarios"]:
        winners[kind] = {}
        for label in SOURCES:
            rows = [results[f"{kind}/{label}/{slot}"] for slot in range(GENERATION["slots_per_source"])]
            feasible = [row for row in rows if row["feasible"]]
            winner = min(feasible, key=lambda row: (row["score"], row["slot_id"])) if feasible else None
            winners[kind][label] = None if winner is None else {
                "slot_id": winner["slot_id"], "program_sha256": winner["program_sha256"], "train_score_ms": winner["score"],
                "program": generated["slots"][winner["slot_id"]]["program"], "training_receipt_id": winner["receipt_id"]}
    digest = finish(args.study, journal, "selected", {"generated_sha256": journal.get("phase/generated"),
                   "rule": "lowest feasible train score within each source and scenario, tie by slot ID",
                   "training": results, "winners": winners})
    print(json.dumps({"selection_sha256": digest, "eligible": {kind: {label: row is not None for label, row in rows.items()}
                                                              for kind, rows in winners.items()}}))


def summarize(kind, cases, scenario, paired, selected):
    from rsi_memory.statistics import summarize_paired
    from rsi_memory.topology import fingerprint as worker_fingerprint

    case_names = [case["name"] for case in cases]
    scopes = [*case_names, "aggregate"]
    threshold = MARGINS[kind]
    # The helper divides family alpha across all its scopes. Calibrate that
    # divisor so the reported aggregate intervals use 0.05/12 per contrast.
    alpha = BUDGET["family_alpha"] * len(scopes) / len(MARGINS)
    stats_plan = {"schema": 1, "kind": "paired_latency_preregistration", "block_ids": scenario["block_ids"],
                  "case_names": case_names, "candidate": "admitted", "comparators": list(REFERENCES),
                  "threshold_ms": {name: threshold if name == "aggregate" else 0 for name in scopes},
                  "bootstrap": {"seed": BUDGET["bootstrap_seed"], "draws": BUDGET["bootstrap_draws"], "family_alpha": alpha}}
    evaluated = summarize_paired(stats_plan, paired, expected_plan_sha256=worker_fingerprint(stats_plan))
    contrasts = evaluated["comparators"]
    same_program = (selected["seed"] is not None and selected["admitted"] is not None
                    and selected["seed"]["program_sha256"] == selected["admitted"]["program_sha256"])
    for label, contrast in contrasts.items():
        row = contrast["scopes"]["aggregate"]
        exact_same_arm = label == "seed" and same_program
        contrast["identical_program_to_candidate"] = exact_same_arm
        contrast["strict_all_blocks_above_margin"] = (contrast["complete"] and not exact_same_arm
            and all(sample["gain_ms"] > threshold for sample in row["blocks"]))
        contrast["v7_style_all_positive_and_median_above_margin"] = (contrast["complete"] and not exact_same_arm
            and row["all_blocks_positive"] is True and row["median_above_minimum"] is True)
        for name in case_names:
            contrast["scopes"][name].pop("paired_bootstrap_interval_ms")
            contrast["scopes"][name].pop("interval_above_minimum")
    return {"statistics_plan": stats_plan, "statistics_plan_sha256": worker_fingerprint(stats_plan),
            "global_aggregate_interval_method": "whole-block paired percentile bootstrap, Bonferroni 0.05/12, approximate",
            "aggregate_interval_level": 1 - BUDGET["family_alpha"] / BUDGET["family_comparisons"], "comparisons": contrasts}


def heldout(args, plan, journal):
    selected = require(args.study, journal, "selected")
    if args.expected_selection_sha256 != journal.get("phase/selected"):
        raise ValueError("held-out stage requires the exact prior selection digest")
    results = {}
    with executors(args, plan) as worker:
        for kind, scenario in plan["scenarios"].items():
            cases = read(args.study / f"{kind}-test.json", scenario["test_sha256"])["cases"]
            winners = selected["winners"][kind]
            active = [label for label in SOURCES if winners[label] is not None]
            key = "both" if len(active) == 2 else (active[0] if active else "neither")
            schedule = plan["test_schedule_by_eligible_source_set"][key]
            paired = {label: {"feasible": True, "samples": []} for label in schedule[0]}
            blocks = []
            for block, order in zip(scenario["block_ids"], schedule):
                rows = {}
                for label in order:
                    arm = {"native": "native_lru", "manual-direct": "manual_topology"}.get(label, "generated")
                    program = winners[label]["program"] if label in winners else None
                    trial = gpu_trial(journal, worker(kind), plan, kind, f"test/{block}/{label}", cases, arm, program, split="test")
                    rows[label] = outcome(trial)
                ref = rows["native"]
                for label, row in rows.items():
                    if label != "native" and row["feasible"] and (not ref["feasible"] or signature(row) != signature(ref)):
                        reason = "selected_bytes_or_attention_mismatch" if ref["feasible"] else "native_control_unavailable"
                        row = rows[label] = {**row, "feasible": False, "score": None, "failure_type": reason}
                    paired[label]["feasible"] &= row["feasible"]
                    if row["feasible"]:
                        paired[label]["samples"].append({"block_id": block, "score": row["score"],
                            "case_scores_ms": {case["name"]: case["score_ms"] for case in row["cases"]},
                            "receipt_id": row["receipt_id"]})
                blocks.append({"block_id": block, "order": order, "arms": rows})
                print(json.dumps({"scenario": kind, "block": block,
                                  "feasible": {label: row["feasible"] for label, row in rows.items()}}), flush=True)
            results[kind] = {"source_eligibility": key, "blocks": blocks, "paired": paired,
                             "analysis": summarize(kind, cases, scenario, paired, winners)}
    digest = finish(args.study, journal, "heldout", {"selection_sha256": args.expected_selection_sha256,
                   "scenarios": results, "primary_comparisons": BUDGET["family_comparisons"],
                   "family_alpha": BUDGET["family_alpha"], "strict_gate": "all 12 complete paired gains exceed scenario margin"})
    print(json.dumps({"heldout_sha256": digest, "strict_pass": {kind: {label: row["strict_all_blocks_above_margin"]
              for label, row in results[kind]["analysis"]["comparisons"].items()} for kind in results}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subs = parser.add_subparsers(dest="phase", required=True)
    for phase in ("prepare", "baseline", "generate", "select", "heldout"):
        sub = subs.add_parser(phase)
        for option in ("study", "source-root", "source-manifest"):
            sub.add_argument("--" + option, type=Path, required=True)
        if phase == "prepare":
            for option in ("config-directory", "harness-directory"):
                sub.add_argument("--" + option, type=Path, required=True)
        else:
            sub.add_argument("--expected-plan-sha256", required=True)
        if phase in {"baseline", "select", "heldout"}:
            sub.add_argument("--executor-config", type=Path, required=True)
        if phase == "generate":
            sub.add_argument("--key-file", type=Path)
        if phase == "heldout":
            sub.add_argument("--expected-selection-sha256", required=True)
    args = parser.parse_args()
    runtime(args)
    if args.phase == "prepare":
        prepare(args)
        return
    plan = read(args.study / "preregistration.json", args.expected_plan_sha256)
    if plan["driver_sha256"] != raw_hash(Path(__file__)) or plan["runtime_sha256"] != RUNTIME:
        raise ValueError("driver source differs from preregistration")
    journal = Journal(args.study / "journal.sqlite", args.expected_plan_sha256)
    try:
        {"baseline": baseline, "generate": generate, "select": select, "heldout": heldout}[args.phase](args, plan, journal)
    finally:
        journal.close()


if __name__ == "__main__":
    main()
