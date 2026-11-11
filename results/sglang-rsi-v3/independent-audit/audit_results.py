"""Audit settled SGLang RSI rounds and bound serving comparisons from local records."""

import argparse
from collections import defaultdict
from copy import deepcopy
import json
from pathlib import Path
from statistics import median
import sys

from audit_protocol import SEED_SHA256, archived_source, audit_calls, audit_manifest, digest, entries, file_sha, fingerprint, read_json, snapshot


PARITY = ("effective_schedule_sha256", "dispatch_sha256", "output_ids_sha256", "packed_kv_sha256_by_rank",
          "global_ticks_sha256", "input_tokenization_sha256", "runtime_sha256")


def score(row):
    result = row.get("result", {})
    return result["score"] if result.get("feasible") else None


def best(items, key):
    feasible = [row for row in items if row is not None and row["score"] is not None]
    return min(feasible, key=key) if feasible else None


def training_archive(previous, arm, native, bootstrap):
    candidates = {bootstrap["program_sha256"]: bootstrap}
    evidence = {"native": {"score": score(native), "status": native["status"], "feedback": native["result"]["feedback"]},
                bootstrap["id"]: {field: bootstrap[field] for field in ("status", "score", "feedback")}}
    for row in previous:
        for trial in row["fixed"] if arm == "fixed" else row["parent"] + row["child"]:
            evidence[trial["id"]] = {field: trial[field] for field in ("status", "score", "feedback")}
            if trial["program"] is not None and trial["score"] is not None:
                candidates[trial["program_sha256"]] = trial
    ordered = sorted(candidates, key=lambda sha: (candidates[sha]["score"] is None, candidates[sha]["score"], sha))[:10]
    recent = [*list(reversed(list(evidence)[1:]))[:9], "native"]
    ranked = {sha: {field: candidates[sha][field] for field in ("program", "score")} for sha in ordered}
    incumbent = candidates[ordered[0]] if ordered and candidates[ordered[0]]["score"] is not None else None
    return ranked, {eid: evidence[eid] for eid in recent}, incumbent


def comparison(effect, candidate, reference, threshold):
    arms = effect.get("result", {})
    left, right = arms.get(candidate), arms.get(reference)
    if not left or not right or not left["feasible"] or not right["feasible"]:
        return {"passed": False, "reason": "missing_feasible_pair", "deltas": []}
    pairs = list(zip(left["samples"], right["samples"], strict=True))
    assert all(a["block_id"] == b["block_id"] for a, b in pairs)
    values = [b["score"] - a["score"] for a, b in pairs]
    middle = median(values)
    passed = all(value > 0 for value in values) and middle > threshold
    return {"passed": passed, "reason": "stable_gain" if passed else "insufficient_stable_gain",
            "median_improvement": middle, "deltas": [{"block_id": a["block_id"], "reference_minus_candidate": value}
                                                     for (a, _), value in zip(pairs, values, strict=True)]}


def effect_by_case(adapter, split, program, program_type):
    value = None if program is None else program_type.from_dict(program)
    effects = {}
    for case in adapter.cases[split]:
        try:
            comp = adapter._compile(value, case=case)
        except (ValueError, MemoryError):
            effects[case["id"]] = None
            continue
        start = comp.audit["startup"]
        actions = comp.plan["actions"] if start["backing_mode"] == "tiered" else []
        locations = sorted((action["rank"], action["layer"], action["node"],
                            comp.workers[action["rank"]].layer_tiers[action["layer"]],
                            action["read_links"], action["write_links"]) for action in actions)
        effects[case["id"]] = {"hot_slots": start["hot_slots"], "backing_mode": start["backing_mode"], "locations": locations}
    return effects


def audit_receipts(run, effect, adapter, program_type):
    if effect["status"] != "completed":
        return {"status": effect["status"], "failure": effect.get("failure")}
    result = effect["result"]
    block_ids = effect["input"]["block_ids"]
    by_case = {case["id"]: case for case in adapter.cases[effect["input"]["split"]]}
    cases = list(by_case)
    observed, shared = {}, defaultdict(dict)
    for label, arm in result.items():
        seen = {}
        timing = []
        program = effect["input"]["programs"][label]
        candidate = None if program is None else program_type.from_dict(program)
        comp_by_case = {}
        for job in arm["metrics"].get("receipts", ()):
            if job in seen:
                continue
            path = run / "receipts" / f"{job}.json"
            receipt = read_json(path)
            summary = receipt["result"]
            assert receipt["job_id"] == job and receipt["status"] == "completed" and receipt["returncode"] == 0
            assert receipt["failure"] is None and receipt["environment_conflict"] is False
            assert receipt["process_group_remaining"] is receipt["process_group_cleanup_required"] is False
            assert receipt["remaining_compute_pids"] == "" and receipt["source_manifest_sha256"] == adapter.source
            assert fingerprint(summary) == receipt["result_canonical_sha256"]
            raw = (json.dumps(summary, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
            assert digest(raw) == receipt["result_sha256"]
            assert summary["case"]["id"] in cases
            seen[job] = file_sha(path)
            if not summary["feasible"]:
                assert summary["phase"] == "parity"
                continue
            metrics = summary["metrics"]
            assert metrics["input_tokenization_sha256"] == fingerprint(adapter.token_inputs[summary["case"]["id"]])
            case_id = summary["case"]["id"]
            for field in PARITY:
                if summary["phase"] != "parity" and field == "packed_kv_sha256_by_rank":
                    continue
                value = metrics[field]
                assert shared[case_id].setdefault(field, value) == value
            if summary["phase"] == "performance":
                assert summary["score"] == metrics["timing"]["window_seconds"] * 1000
                assert not metrics["packed_kv_sha256_by_rank"]
                assert all(row.get("resident_producer_oracle") is None for row in metrics["ranks"])
                if case_id not in comp_by_case:
                    comp_by_case[case_id] = adapter._compile(candidate, case=by_case[case_id])
                compiled = comp_by_case[case_id]
                assert compiled.audit == summary["memory_compilation"]
                start = compiled.audit["startup"]
                assert metrics["backing_mode"] == start["backing_mode"]
                for rank, row in enumerate(metrics["ranks"]):
                    if start["backing_mode"] == "resident":
                        assert row["layer_tiers"] == ["resident"] * 6 and row["pinned_host_bytes"] == 0
                    else:
                        assert tuple(row["layer_tiers"]) == compiled.workers[rank].layer_tiers
                    assert row["hot_slots"] == start["hot_slots"]
                assert summary["program_sha256"] == (fingerprint(program) if program is not None else None)
                timing.append((case_id, summary["score"]))
        if arm["feasible"]:
            samples = arm["samples"]
            assert [row["block_id"] for row in samples] == block_ids
            assert [(case, row["case_scores_ms"][case]) for row in samples for case in cases] == timing
            assert all(row["score"] == sum(row["case_scores_ms"].values()) for row in samples)
            assert arm["score"] == median(row["score"] for row in samples)
        observed[label] = {"feasible": arm["feasible"], "median_window_ms": arm["score"], "timing_receipts": len(timing),
                           "receipt_sha256": seen, "samples": arm["samples"]}
    return {"status": "completed", "arms": observed, "per_case_matching_receipt_signatures": dict(shared)}


def audit_rounds(run, records, calls, envelopes, plans, manifest, adapter, controller, program_type):
    threshold = manifest["config"]["min_improvement"]
    rounds = [records[key] for key in sorted(records) if key.startswith("round/")]
    initial = records["eval/bootstrap/initial-incumbent"]
    bootstrap = {"id": "bootstrap/initial-incumbent", "origin": "initial_program_bootstrap", "harness_sha256": None,
                 "decision": None, "program": manifest["initial_program"], "program_sha256": fingerprint(manifest["initial_program"]),
                 "score": score(initial), "status": "feasible" if score(initial) is not None else "infeasible",
                 "feedback": initial["result"]["feedback"]}
    assert fingerprint(adapter.training_context) == manifest["training_context_sha256"]
    audit = []
    active = SEED_SHA256
    for index, row in enumerate(rounds):
        prefix = f"r{index:04d}"
        assert row["round"] == index and row["active_before"] == active
        assert row == read_json(run / f"controller/rounds/{prefix}.json")
        adaptive, evidence, incumbent = training_archive(rounds[:index], "adaptive", records["eval/native/train"], bootstrap)
        fixed, fixed_evidence, fixed_incumbent = training_archive(rounds[:index], "fixed", records["eval/native/train"], bootstrap)
        assert row["matched_context_sha256"] == fingerprint({"parents": adaptive, "evidence": evidence})
        assert len(row["parent"]) == len(row["child"]) == 2 and len(row["fixed"]) == 4
        adaptive_plan, fixed_plan = (plans[row[f"{arm}_meta_plan_id"]] for arm in ("adaptive", "fixed"))
        assert adaptive_plan["proposed_child_sha256"] == row["proposed_child_sha256"]
        assert fixed_plan["proposed_child_sha256"] == row["fixed_meta_discarded_sha256"]
        parent_source = archived_source(run, active)
        child_sha = row["proposed_child_sha256"]
        child_source = archived_source(run, child_sha) if child_sha else None
        source_change = row["controlled_behavior"]
        fallback = child_source is None or source_change is None

        def context(slot, source, candidates, feedback):
            return {"round": index, "slot": slot, "harness_source": source,
                    "domain": adapter.training_context, "parent_ids": list(candidates), "candidates": candidates,
                    "evidence_ids": list(feedback), "evidence": feedback, "trusted_entry_contract": controller.ENTRY_CONTRACT}

        actual_meta = {}
        for arm, sha, pool, feedback in (("adaptive", active, adaptive, evidence), ("fixed", SEED_SHA256, fixed, fixed_evidence)):
            api_key = f"api/{prefix}/{arm}-meta"
            meta_source = archived_source(run, sha)
            if api_key not in envelopes:
                continue
            meta = controller._entry(meta_source, "build_meta_prompt", context(0, meta_source, pool, feedback), pool)
            assert meta == envelopes[api_key]["generator_decision"]
            assert meta == row[f"{arm}_meta_decision"]
            assert calls[api_key]["harness_sha256"] == sha
            actual_meta[arm] = {"api_id": api_key, "state": calls[api_key]["state"],
                                "harness_sha256": sha, "actual_decision": meta}
            if arm == "adaptive" and index and rounds[index - 1]["decision"] == "admitted":
                prior_sha = rounds[index - 1]["active_before"]
                prior_source = archived_source(run, prior_sha)
                alternate = controller._entry(prior_source, "build_meta_prompt", context(0, prior_source, pool, feedback), pool)
                actual_meta[arm]["prior_source_same_round_shadow"] = {
                    "harness_sha256": prior_sha, "decision": alternate,
                    "changed_structural_fields": sorted(k for k in meta if k != "guidance" and meta[k] != alternate[k]),
                    "guidance_changed": meta["guidance"] != alternate["guidance"],
                    "evidence_id_set_changed": set(meta["evidence"]) != set(alternate["evidence"]),
                }

        if source_change is not None:
            assert child_source is not None
            parent_nodes, child_nodes = entries(parent_source), entries(child_source)
            for entry, info in source_change.items():
                assert info["ast_changed"] == (parent_nodes[entry] != child_nodes[entry])
                for item in info["decisions"]:
                    old = controller._entry(parent_source, entry, context(item["slot"], parent_source, adaptive, evidence), adaptive)
                    new = controller._entry(child_source, entry, context(item["slot"], child_source, adaptive, evidence), adaptive)
                    assert item["before"] == old and item["after"] == new
                    assert item["changed_fields"] == sorted(k for k in old if k != "guidance" and old[k] != new[k])

        candidates_audit, same_effect_groups = {}, defaultdict(list)
        for label, trials in (("parent", row["parent"]), ("child", row["child"]), ("fixed", row["fixed"])):
            pool, feedback = (fixed, fixed_evidence) if label == "fixed" else (adaptive, evidence)
            sha = SEED_SHA256 if label == "fixed" else child_sha if label == "child" and not fallback else active
            source = archived_source(run, sha)
            for trial in trials:
                assert trial["harness_sha256"] == sha
                call_id = "api/" + trial["id"]
                if call_id in calls:
                    envelope = envelopes[call_id]
                    expected = controller._entry(source, "build_prompt", context(envelope["slot"], source, pool, feedback), pool)
                    assert trial["decision"] == expected == envelope["generator_decision"]
                    assert envelope["selected_training_evidence"] == {key: feedback[key] for key in expected["evidence"]}
                    assert envelope.get("selected_parents", {}) == {key: pool[key] for key in expected["parents"]}
                checked_candidate = {"arm": label, "harness_sha256": trial["harness_sha256"],
                                     "raw_program_sha256": trial["program_sha256"], "training_score_ms": trial["score"],
                                     "status": trial["status"], "effective_training_program_sha256": None}
                candidates_audit[trial["id"]] = checked_candidate
                if trial["program"] is not None:
                    try:
                        physical = adapter.effective_program_fingerprint(program_type.from_dict(trial["program"]))
                    except (MemoryError, ValueError):
                        assert trial["score"] is None
                    else:
                        checked_candidate["effective_training_program_sha256"] = physical
                        same_effect_groups[physical].append(trial["id"])
                    reply = json.loads(records[call_id]["result"]["content"])
                    decision = trial["decision"]
                    if decision["operator"] != "construct":
                        assembled = deepcopy(pool[decision["parents"][0]]["program"])
                        assembled["rules"][decision["rule"]] = reply["rule"]
                        if decision["operator"] == "crossover":
                            assembled["rules"].append(pool[decision["parents"][1]]["program"]["rules"][0])
                        reply = assembled
                    assert program_type.from_dict(reply).to_dict() == trial["program"]
                    eval_row = records["eval/" + trial["id"]]
                    assert eval_row["input"] == {"split": "train", "program": trial["program"]}
                    assert trial["program_sha256"] == fingerprint(trial["program"])
                    assert trial["score"] == score(eval_row)

        task = row["task_output_selection"]
        assert task["shared_incumbent"] == incumbent and task["fixed_incumbent"] == fixed_incumbent
        subsets = {"parent": row["parent"], "child": row["child"], "fixed": row["fixed"],
                   "fixed_a": row["fixed"][:2], "fixed_b": row["fixed"][2:]}
        fresh = {name: best(trials, lambda item: (item["score"], item["id"])) for name, trials in subsets.items()}
        assert task["fresh"] == fresh
        actual = {item["program_sha256"]: adapter.effective_program_fingerprint(program_type.from_dict(item["program"]))
                  for item in (*fresh.values(), incumbent, fixed_incumbent) if item is not None}
        assert actual == task["effective_program_sha256_by_raw_program"]
        for label in ("parent", "child", "fixed"):
            pick = fresh[label]
            retained = fixed_incumbent if label == "fixed" else incumbent
            if retained is None:
                origin = "fresh" if pick else "unavailable"
            elif pick is None or pick["score"] >= retained["score"] or actual[pick["program_sha256"]] == actual[retained["program_sha256"]]:
                pick, origin = retained, "incumbent"
            else:
                origin = "fresh"
            assert task["selected"][label] == {"trial": pick, "origin": origin}

        pair = records[f"pair/{prefix}/validation"]
        saved = row["paired_validation"]
        assert saved["status"] == pair["status"]
        for name, result in saved["comparisons"].items():
            left, right = name.split("_vs_")
            assert result == comparison(pair, left, right, threshold)
        reason = adaptive_plan["failure"]
        if reason is None and child_source is not None and source_change is None:
            assert row["reason"].startswith("invalid_child_entry_")
            reason = row["reason"]
        if reason is None and not all(info["ast_changed"] and any(item["changed_fields"] for item in info["decisions"])
                                      for info in source_change.values()):
            reason = "both_entries_need_structural_behavior_change"
        if reason is None and (any(row[f"{arm}_validation"] is None or row[f"{arm}_validation"]["score"] is None for arm in ("parent", "child"))
                               or score(records["eval/native/validation"]) is None):
            reason = "missing_feasible_matched_validation"
        if reason is None and incumbent is None:
            reason = "missing_measured_incumbent"
        selected = task["selected"]
        if reason is None and selected["child"]["origin"] != "fresh":
            reason = "child_retained_incumbent"
        if reason is None and actual[selected["parent"]["trial"]["program_sha256"]] == actual[selected["child"]["trial"]["program_sha256"]]:
            reason = "identical_effective_parent_and_child_outputs"
        if reason is None and not saved["comparisons"]["child_vs_parent"]["passed"]:
            reason = "insufficient_stable_paired_improvement"
        if reason is None and not saved["comparisons"]["child_vs_incumbent"]["passed"]:
            reason = "insufficient_stable_paired_incumbent_improvement"
        assert row["decision"] == ("admitted" if reason is None else "rejected")
        assert row["reason"] == ("matched_validation_improvement" if reason is None else reason)
        active = child_sha if reason is None else active
        assert row["active_after"] == active
        reuse = row.get("previous_admission_reuse")
        if index and rounds[index - 1]["decision"] == "admitted":
            assert reuse["source_sha256"] == row["active_before"]
            for task_id in reuse["task_api_ids"]:
                assert calls[task_id]["harness_sha256"] == reuse["source_sha256"]
                assert calls[task_id]["state"] == "completed"
            if reuse["meta_api_completed"]:
                assert calls[reuse["meta_api_id"]]["harness_sha256"] == reuse["source_sha256"]
                assert calls[reuse["meta_api_id"]]["state"] == "completed"
            expected_reuse = [trial["id"] for trial in row["parent"]
                              if "api/" + trial["id"] in reuse["task_api_ids"] and trial["program"] is not None]
            assert reuse["task_program_trial_ids"] == expected_reuse
        else:
            assert reuse is None
        audit.append({"round": index, "active_before": row["active_before"], "active_after": active,
                      "proposed_child_sha256": child_sha, "fixed_discarded_child_sha256": row["fixed_meta_discarded_sha256"],
                      "decision": row["decision"], "reason": row["reason"], "source_diff": row["source_diff"],
                      "controlled_behavior": source_change, "effective_program_fingerprints": actual,
                      "all_candidates": candidates_audit, "candidate_ids_by_effect": dict(same_effect_groups),
                      "selected_output_origins": {label: pick["origin"] for label, pick in selected.items()},
                      "fresh_best_ids": {label: item["id"] if item else None for label, item in fresh.items()},
                      "selected_output_ids": {label: item["trial"]["id"] if item["trial"] else None
                                              for label, item in selected.items()},
                      "shared_incumbent_id": incumbent["id"] if incumbent else None,
                      "actual_meta_requests": actual_meta, "previous_admission_reuse": reuse,
                      "validation_comparisons": saved["comparisons"],
                      "validation_receipts": audit_receipts(run, pair, adapter, program_type)})
    return rounds, audit


def audit(run, source, *, terminal):
    sys.path.insert(0, str(source.resolve(strict=True)))
    from rsi_memory import topology_rsi as controller
    from rsi_memory.memory_program import MemoryProgram
    from topomem_sglang_rsi.domain import SGLangDomain

    records, ordinals = snapshot(run, terminal=terminal)
    manifest, protocol = audit_manifest(run, source, records)
    calls, envelopes, usage, counts, plans = audit_calls(run, records, ordinals, manifest, protocol)
    path = run / "schema3.json"

    def no_exec(**_):
        raise RuntimeError("offline controller audit cannot start the GPU executor")

    config = read_json(path)
    if terminal:
        # The terminal audit opens case manifests and commitments, with no fixture text.
        adapter = SGLangDomain(config, path.parent, executor_factory=no_exec)
    else:
        # Active validation reads only training and validation metadata.
        from topomem_sglang_rsi.contract import load_case, load_external

        adapter = object.__new__(SGLangDomain)
        adapter.executor = None
        adapter.source = config["source_manifest_sha256"]
        adapter.profile = config["hot_profile"]
        adapter.cases = {split: config["cases"][split] for split in ("train", "validation")}
        adapter.token_inputs = {case["id"]: config["token_inputs"][case["id"]]
                                for cases in adapter.cases.values() for case in cases}
        adapter.geometry_by_case = {}
        local = path.parent / config["local_cases"]
        for cases in adapter.cases.values():
            for case in cases:
                directory = local / case["id"]
                target, topology, scenario, _ = load_case(directory, case)
                external, _ = load_external(directory, case, expected_inputs=adapter.token_inputs[case["id"]], require_gate=True)
                adapter.geometry_by_case[case["id"]] = target, topology, scenario, external
        adapter.preflight = adapter.geometry_by_case[adapter.cases["train"][0]["id"]]
        adapter.training_context = next(iter(envelopes.values()))["trusted_domain"]
    try:
        rounds, rounds_audit = audit_rounds(run, records, calls, envelopes, plans, manifest, adapter, controller, MemoryProgram)
        result = {"schema": 1, "kind": "sglang_strict_rsi_independent_audit", "manifest_sha256": fingerprint(manifest),
                  "provider_usage": usage, "counts_by_round_and_arm": counts, "meta_plans": plans, "rounds": rounds_audit,
                  "frozen": None, "heldout": None,
                  "limits": "Local proof and supervisor summaries only. No model calls, original packed payload replay, GPU or benchmark fixture contents."}
        if not terminal or "frozen" not in records:
            return result

        frozen = records["frozen"]
        assert len(rounds) == frozen["rounds"] == 2 and frozen["partial"] is False
        assert frozen == read_json(run / "controller/frozen.json")
        assert frozen["rounds_sha256"] == fingerprint(rounds) and frozen["manifest_sha256"] == fingerprint(manifest)
        admitted = {SEED_SHA256, *(row["active_after"] for row in rounds if row["decision"] == "admitted")}
        assert frozen["admitted_harness_sha256s"] == sorted(admitted)
        reused = {entry["source_sha256"] for row in rounds if (entry := row.get("previous_admission_reuse"))
                  and entry["meta_api_completed"] and entry["task_program_trial_ids"]}
        assert frozen["reused_harness_sha256s"] == sorted(reused)
        choices = [row[name] for row in rounds for name in ("parent_validation", "child_validation") if row[name]]
        fixed = [entry for row in rounds for entry in row["fixed_validation"]]
        sort = lambda entry: (entry["score"], entry["trial"]["id"])
        assert frozen["adaptive"] == best([row for row in choices if row["trial"]["harness_sha256"] in admitted], sort)
        assert frozen["adaptive_archive_best"] == best(choices, sort)
        assert frozen["fixed"] == best(fixed, sort)
        result["frozen"] = {key: frozen[key] for key in ("rounds", "active_harness_sha256", "admitted_harness_sha256s",
                                                        "reused_harness_sha256s", "manifest_sha256", "rounds_sha256")}
        result["frozen"]["finalists"] = {name: {"id": frozen[name]["trial"]["id"],
                                                "harness_sha256": frozen[name]["trial"]["harness_sha256"],
                                                "program_sha256": frozen[name]["trial"]["program_sha256"],
                                                "validation_score": frozen[name]["score"]}
                                         for name in ("adaptive", "adaptive_archive_best", "fixed") if frozen[name]}
        if "pair/frozen/test" not in records:
            return result
        test = records["pair/frozen/test"]
        assert ordinals["frozen"] < ordinals["pair/frozen/test"]
        for name in ("adaptive", "fixed"):
            assert test["input"]["programs"][name] == frozen[name]["trial"]["program"]
        assert test["input"]["programs"]["native"] is None and test["input"]["programs"]["manual"] == manifest["manual_program"]
        effects = {name: effect_by_case(adapter, "test", program, MemoryProgram)
                   for name, program in test["input"]["programs"].items()}
        report = audit_receipts(run, test, adapter, MemoryProgram)
        report["effective_case_layouts"] = effects
        report["same_physical_layout_as_adaptive"] = {name: effects[name] == effects["adaptive"] and all(
            item is not None for item in effects[name].values()) for name in ("native", "manual", "fixed")}
        if "test_comparisons" in records:
            saved = records["test_comparisons"]
            observed = {f"adaptive_vs_{name}": comparison(test, "adaptive", name, manifest["config"]["min_improvement"])
                        for name in ("native", "manual", "fixed")}
            assert saved == observed
            report["recorded_comparisons"] = observed
            report["distinct_layout_gate_and_timing_passed"] = {
                name: observed[f"adaptive_vs_{name}"]["passed"] and not report["same_physical_layout_as_adaptive"][name]
                for name in ("native", "manual", "fixed")}
        result["heldout"] = report
        return result
    finally:
        adapter.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    parser.add_argument("--terminal", action="store_true", help="also inspect sealed result metadata after operator release")
    args = parser.parse_args()
    result = audit(args.run, args.source, terminal=args.terminal)
    with args.output.open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(result, stream, sort_keys=True, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    print(file_sha(args.output))


if __name__ == "__main__":
    main()
