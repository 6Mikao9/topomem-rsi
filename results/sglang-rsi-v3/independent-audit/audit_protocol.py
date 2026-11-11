"""Audit the frozen SGLang generation protocol without opening held-out data."""

import argparse
import ast
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sqlite3


WORKER_SHA256 = "b73995aea418432831366a7c2de0af038fbe8eae2f2f85ac164d2e88e4c81909"
SEED_SHA256 = "42699e4d5148ee3dc3f41f7ffeab8a3020e8fb75a5a433940d7ae88e4c08ca38"
EXPECTED_ACTIONS = [["allocate", "place"], ["configure", "backing_mode"], ["configure", "hot_capacity"]]
PLAN_FIELDS = ("observed_problem", "mechanism_change", "expected_effect", "falsification")


def digest(raw):
    return sha256(raw).hexdigest()


def fingerprint(value):
    return digest(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode())


def file_sha(path):
    return digest(path.read_bytes())


def read_json(path):
    return json.loads(path.read_bytes())


def snapshot(run, *, terminal=False):
    where = """key = 'manifest' OR key LIKE 'api/%' OR key LIKE 'plan/%' OR key LIKE 'round/%'
               OR key LIKE 'pair/r%/validation' OR key LIKE 'eval/r%/%'
               OR key IN ('eval/native/train', 'eval/native/validation', 'eval/bootstrap/initial-incumbent')"""
    if terminal:
        where += " OR key IN ('frozen', 'pair/frozen/test', 'test_comparisons')"
    path = run / "controller/journal.sqlite"
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1) as db:
        db.execute("PRAGMA query_only=ON")
        observed = list(db.execute(f"SELECT rowid, key, data FROM records WHERE {where} ORDER BY rowid"))
    return {key: json.loads(raw) for _, key, raw in observed}, {key: number for number, key, _ in observed}


def archived_source(run, source_sha):
    path = run / "controller/harnesses" / f"{source_sha}.py"
    raw = path.read_bytes()
    assert digest(raw) == source_sha
    return raw.decode("utf-8")


def audit_manifest(run, source, records):
    manifest = records["manifest"]
    cfg = read_json(run / "schema3.json")
    contract = manifest["contract"]
    budget = cfg["controller_preregistration"]
    assert manifest == read_json(run / "controller/manifest.json")
    assert manifest["schema"] == 2 and cfg["schema"] == 3
    assert manifest["seed_sha256"] == SEED_SHA256
    assert digest(archived_source(run, SEED_SHA256).encode()) == SEED_SHA256
    assert {name: file_sha(source / "rsi_memory" / name) for name in manifest["source_sha256"]} == manifest["source_sha256"]
    assert {name: file_sha(source / name) for name in contract["adapter_sources_sha256"]} == contract["adapter_sources_sha256"]
    assert contract["source_manifest_sha256"] == cfg["source_manifest_sha256"] == WORKER_SHA256
    assert contract["launch_sha256"] == cfg["launch_sha256"]
    assert contract["controller_preregistration"] == budget
    assert contract["manual_selection"]["selected"] == "resident-6144" and contract["manual_selection"]["receipt_count"] == 38
    assert manifest["manual_program"] == cfg["manual_program"] and manifest["initial_program"] == cfg["initial_program"]
    assert manifest["supported_actions"] == EXPECTED_ACTIONS
    assert manifest["task_output_protocol"] == "shared_measured_train_incumbent_v1"
    assert manifest["effective_program_fingerprint"] == "adapter_training_only_v1"
    assert manifest["validation_arms"] == ["native", "manual", "parent", "child", "fixed", "incumbent"]
    assert manifest["candidate_slots_per_arm_per_round"] == 4 and manifest["provider_calls_per_arm_per_round_cap"] == 5
    assert budget["min_improvement_ms"] == max(5, 2 * (max(budget["native_default_training_scores_ms"]) -
                                                      min(budget["native_default_training_scores_ms"])))
    assert manifest["config"] == {"rounds": 2, "slots": 2, "max_output_tokens": 65536,
                                  "min_improvement": budget["min_improvement_ms"], "min_paired_blocks": 6,
                                  "model": "deepseek-v4.1-flash", "wall_seconds": 14400}
    assert contract["paired_block_ids"] == cfg["paired_block_ids"]
    assert len(contract["paired_block_ids"]["validation"]) == 6 and len(contract["paired_block_ids"]["test"]) == 12
    protocol = contract.get("generator_program_protocol")
    if protocol is not None:
        assert protocol["kind"] == "backend_program_contract_v3"
        assert {name: file_sha(source / name) for name in protocol["source_sha256"]} == protocol["source_sha256"]
        grammar = protocol["trusted_program_contract"]
        assert "required_rule" not in grammar
        assert grammar["required_any_rule"] == [{"event": "allocate", "action": "place"},
                                                {"event": "configure", "action": "backing_mode"}]
        families = {row["name"]: row for row in grammar["program_families"]}
        assert set(families) == {"tiered", "resident"}
        assert families["tiered"]["required_rules"] == [{"event": "allocate", "action": "place"}]
        assert families["resident"]["forbidden_rules"] == [{"event": "allocate", "action": "place"}]
        assert "6144" in families["resident"]["selection_condition"]
    return manifest, protocol


def entries(source):
    return {node.name: ast.dump(node, include_attributes=False) for node in ast.parse(source).body}


def audit_calls(run, records, ordinals, manifest, protocol):
    calls = {}
    usage = {arm: {"calls": 0, "completed": 0, "calls_with_usage": 0, "known_tokens": Counter(),
                   "states": Counter()} for arm in ("adaptive", "fixed")}
    per_round = Counter()
    envelopes = {}
    for key, row in records.items():
        if not key.startswith("api/"):
            continue
        _, prefix, label, *_ = key.split("/")
        arm = "fixed" if label.startswith("fixed") else "adaptive"
        assert prefix in ("r0000", "r0001")
        per_round[prefix, arm] += 1
        assert per_round[prefix, arm] <= 5
        group = usage[arm]
        group["calls"] += 1
        group["completed"] += row["status"] == "completed"
        group["states"][row["status"]] += 1
        payload = row["input"]["payload"]
        assert payload["model"] == "deepseek-v4.1-flash" and payload["max_tokens"] == 65536
        assert payload["reasoning_effort"] == "high" and payload["thinking"] == {"type": "enabled"}
        assert payload["stream"] is True and payload["stream_options"] == {"include_usage": True}
        assert [message["role"] for message in payload["messages"]] == ["system", "user"]
        env = json.loads(payload["messages"][1]["content"])
        assert env["round"] == int(prefix[1:])
        assert fingerprint(env["trusted_domain"]) == manifest["training_context_sha256"]
        assert set(env["selected_training_evidence"]) == set(env["generator_decision"]["evidence"])
        if prefix == "r0000":
            assert set(env["selected_training_evidence"]) <= {"native", "bootstrap/initial-incumbent"}
        if protocol:
            assert env["trusted_program_contract"] == protocol["trusted_program_contract"]
        else:
            assert env["trusted_program_contract"]["required_rule"] == {"event": "allocate", "action": "place"}
        source_sha = row["input"]["harness_sha256"]
        source = archived_source(run, source_sha)
        if arm == "fixed":
            assert source_sha == SEED_SHA256
        if label.endswith("-meta"):
            assert source == env["harness_source"]
            assert env["trusted_parent_behavior"]["candidate_slot_count"] == 2
        if label.startswith(("parent", "child", "fixed-a", "fixed-b")):
            assert env["slot"] == int(key.rsplit("/", 1)[-1]) + (2 if label == "fixed-b" else 0)
            selected = env.get("selected_parents", {})
            assert set(selected) == set(env["generator_decision"]["parents"])
            assert all(fingerprint(item["program"]) == sha for sha, item in selected.items())
        result = row.get("result", {})
        tokens = result.get("usage", {})
        group["calls_with_usage"] += bool(tokens)
        for name, value in tokens.items():
            if isinstance(value, dict):
                for field, count in value.items():
                    group["known_tokens"][f"{name}.{field}"] += count
            else:
                group["known_tokens"][name] += value
        calls[key] = {"state": row["status"], "harness_sha256": source_sha, "reported_usage": tokens,
                      "finish_reason": result.get("finish_reason"),
                      "generator_decision": env["generator_decision"]}
        envelopes[key] = env

    plans = {}
    for key, row in records.items():
        if not key.startswith("plan/"):
            continue
        _, prefix, arm = key.split("/")
        api_id = f"api/{prefix}/{arm}-meta"
        assert ordinals[api_id] < ordinals[key]
        first_task = min((ordinals[other] for other in calls if other.startswith(f"api/{prefix}/")
                          and not other.endswith("-meta")), default=float("inf"))
        assert ordinals[key] < first_task
        plan = row.get("plan")
        source_sha = row.get("proposed_child_sha256")
        assert row["parent_sha256"] == calls[api_id]["harness_sha256"]
        if plan is not None:
            assert row["status"] == "recorded_before_evaluation"
            response = json.loads(records[api_id]["result"]["content"])
            assert set(response) == {"source", "plan"}
            assert response["plan"] == {field: plan[field] for field in PLAN_FIELDS}
            assert plan["training_evidence_ids"] == row["decision"]["evidence"]
            assert row["decision"] == envelopes[api_id]["generator_decision"]
            assert plan["kind"] == "pre_evaluation_hypothesis"
        if source_sha is not None:
            response_source = response["source"]
            assert digest(response_source.encode()) == source_sha
            if arm == "adaptive":
                assert response_source == archived_source(run, source_sha)
            parent = entries(archived_source(run, row["parent_sha256"]))
            candidate = entries(response_source)
            changed = {name: parent[name] != candidate[name] for name in parent}
        else:
            changed = None
        plans[key] = {**row, "response_and_plan_jointly_proposed": True, "archived_before_task_calls": True,
                      "function_ast_changes": changed}

    return calls, envelopes, usage, {f"{key[0]}/{key[1]}": count for key, count in per_round.items()}, plans


def audit_active(run, source):
    records, ordinals = snapshot(run)
    manifest, protocol = audit_manifest(run, source, records)
    calls, envelopes, usage, counts, plans = audit_calls(run, records, ordinals, manifest, protocol)
    observed = {}
    for key, row in records.items():
        if key.startswith("eval/r"):
            result = row.get("result", {})
            program = row["input"]["program"]
            observed[key] = {"state": row["status"], "feasible": result.get("feasible"), "score": result.get("score"),
                             "failure": result.get("feedback", {}).get("failure"), "program_sha256": fingerprint(program),
                             "program_actions": [[item["event"], item["action"]] for item in program["rules"]]}
    old_conflict = observed.get("eval/r0000/parent/00")
    conflict_matches = (protocol is None and old_conflict is not None and
                        old_conflict["failure"] == "resident backing cannot include cold placement rules")
    return {"schema": 1, "kind": "sglang_controller_pre_heldout_protocol_audit",
            "scope": "read-only journal snapshot; no frozen/test records, case text, API or GPU accessed",
            "manifest_sha256": fingerprint(manifest), "source_manifest_sha256": WORKER_SHA256,
            "seed_sha256": SEED_SHA256, "new_generator_protocol": protocol, "config": manifest["config"],
            "configured_total_provider_call_cap": 20, "configured_total_output_token_cap": 20 * 65536,
            "counts_by_round_and_arm": counts, "provider_usage": usage, "calls": calls, "plans": plans,
            "observed_training_evaluations": observed, "actual_legacy_grammar_conflict_rejection": conflict_matches,
            "round_records_available": sorted(key for key in records if key.startswith("round/")),
            "validation_effect_states": {key: row["status"] for key, row in records.items() if key.startswith("pair/")}}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    result = audit_active(args.run, args.source)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(file_sha(args.output))


if __name__ == "__main__":
    main()
