"""Independently replay completed v4 training selection and source provenance."""

import argparse
import ast
from collections import Counter
import difflib
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys
from statistics import median


def sha(raw):
    return sha256(raw).hexdigest()


def read(path):
    return json.loads(path.read_bytes())


def ledger(path):
    with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=1) as db:
        db.execute("PRAGMA query_only=ON")
        rounds = {key: json.loads(data) for key, data in db.execute(
            "SELECT key,data FROM records WHERE key LIKE 'round/r%' ORDER BY key")}
        calls = {key: (status, source) for key, status, source in db.execute(
            "SELECT key,json_extract(data,'$.status'),json_extract(data,'$.input.harness_sha256') "
            "FROM records WHERE key LIKE 'api/r%'")}
        measured = {key: json.loads(data) for key, data in db.execute(
            "SELECT key,data FROM records WHERE key LIKE 'eval/r%/training-matrix/%'")}
        manifest = json.loads(db.execute("SELECT data FROM records WHERE key='manifest'").fetchone()[0])
        native = json.loads(db.execute("SELECT data FROM records WHERE key='eval/native/train'").fetchone()[0])
        validation = {key: json.loads(data) for key, data in db.execute(
            "SELECT key,data FROM records WHERE key LIKE 'pair/r%/validation' AND json_extract(data,'$.status')='completed'")}
        completed_prefixes = {key.removeprefix("round/") for key in rounds}
        payload_contracts = {key: {
            "system_sha256": sha(system.encode("utf-8")),
            "legacy_selection_clause_absent": "Both sides may retain" not in system,
            "meta_system_matches_v41": "separate paired validation reference" in system,
            "trusted_selection_contract": json.loads(contract) if contract is not None else None,
            "selected_training_evidence_ids": sorted(json.loads(evidence)),
        } for key, system, contract, evidence in db.execute(
            "SELECT key, json_extract(data,'$.input.payload.messages[0].content'), "
            "json_extract(json_extract(data,'$.input.payload.messages[1].content'),'$.trusted_selection_contract'), "
            "json_extract(json_extract(data,'$.input.payload.messages[1].content'),'$.selected_training_evidence') "
            "FROM records WHERE key LIKE 'api/r%'")
            if key.removeprefix("api/").split("/")[0] in completed_prefixes}
        contexts = {key: json.loads(data) for key, data in db.execute(
            "SELECT key,data FROM records WHERE key LIKE 'selection_context/r%'")
            if key.removeprefix("selection_context/").split("/")[0] in completed_prefixes}
    return manifest, rounds, calls, measured, native, payload_contracts, validation, contexts


def execution_receipt(receipt_dir, result, config, expected_cases, fingerprint):
    job, = result["metrics"]["receipts"]
    path = receipt_dir / (job + ".json")
    receipt = read(path)
    assert receipt["job_id"] == job and receipt["status"] == "completed" and receipt["returncode"] == 0
    assert receipt["timeout"] is False and receipt["remaining_compute_pids"] == ""
    assert receipt["process_group_remaining"] is False and receipt["process_group_cleanup_required"] is False
    assert receipt["source_manifest_sha256"] == config["source_manifest_sha256"]
    result_value = receipt["result"]
    assert result_value["score"] == result["score"] and result_value["feasible"] is True
    assert fingerprint(result_value) == receipt["result_canonical_sha256"]
    assert [case["case_sha256"] for case in result_value["cases"]] == expected_cases
    signatures = {}
    for case in result_value["cases"]:
        diagnostic = case["diagnostic"]
        assert diagnostic["status"] == "verified"
        assert diagnostic["live_references_at_end"] == 0
        signatures[case["case_sha256"]] = diagnostic["selected_and_output_sha256"]
    return job, sha(path.read_bytes()), signatures


def harness(run, digest):
    data = (run / "controller/harnesses" / (digest + ".py")).read_bytes()
    assert sha(data) == digest
    return data.decode("utf-8")


def changed_decision_fields(before, after):
    return sorted(key for key in before if key != "guidance" and (
        set(before[key]) != set(after[key]) if key == "evidence" else before[key] != after[key]))


def independently_select(trials, timing_rows, ids, lane, allowed, incumbent, margin):
    members, excluded = {}, []
    for trial_id in ids:
        row = trials[trial_id]
        if row["lane"] != lane or allowed is not None and row["harness_sha256"] not in allowed:
            continue
        effect = row["training_effect_sha256"]
        if effect is None or timing_rows[effect]["status"] == "correctness_failed":
            excluded.append(trial_id)
        else:
            assert timing_rows[effect]["status"] == "complete"
            members.setdefault(effect, []).append(trial_id)
    times = {effect: [item["window_ms"] for item in timing_rows[effect]["evaluations"]] for effect in members}
    dominated = set()
    for slower, slow_samples in times.items():
        for faster, fast_samples in times.items():
            if slower == faster:
                continue
            gains = [b - a for a, b in zip(fast_samples, slow_samples, strict=True)]
            if all(gain > 0 for gain in gains) and median(gains) > margin:
                dominated.add(slower)
    frontier = sorted(set(members) - dominated)
    if not frontier:
        assert not members
        return {"status": "no_eligible_proposal", "proposal_id": None, "effect_sha256": None,
                "frontier_effects": [], "excluded_infeasible_proposals": sorted(excluded), "source_sha256": None}
    frontier_ids = {trial_id for effect in frontier for trial_id in members[effect]}
    trial_id = incumbent if incumbent in frontier_ids else min(members[frontier[0]])
    row = trials[trial_id]
    return {"status": "selected", "proposal_id": trial_id, "effect_sha256": row["training_effect_sha256"],
            "frontier_effects": frontier, "excluded_infeasible_proposals": sorted(excluded),
            "source_sha256": row["harness_sha256"]}


def verify_validation(row, pair, expected_blocks, expected_cases, receipt_dir, config, fingerprint):
    assert pair["status"] == row["paired_validation"]["status"] == "completed"
    assert pair["input"]["split"] == "validation"
    assert pair["input"]["block_ids"] == expected_blocks
    for name in ("parent", "child", "fixed", "incumbent"):
        selected = row["fixed_validation"][0] if name == "fixed" else row[name + "_validation"]
        if selected is None:
            assert name not in pair["input"]["programs"]
        else:
            assert pair["input"]["programs"][name] == selected["trial"]["program"]
    receipts, signatures = {}, {}
    arms = pair["result"]
    for arm, record in arms.items():
        if not record["feasible"]:
            continue
        samples = record["samples"]
        assert [sample["block_id"] for sample in samples] == expected_blocks
        assert median(sample["score"] for sample in samples) == record["score"]
        for sample, job in zip(samples, record["metrics"]["receipts"], strict=True):
            result = {"metrics": {"receipts": [job]}, "score": sample["score"]}
            _, digest, case_signatures = execution_receipt(receipt_dir, result, config, expected_cases, fingerprint)
            assert job not in receipts
            receipts[job] = digest
            for case, selected in case_signatures.items():
                assert signatures.setdefault(case, selected) == selected
    for key, comparison in row["paired_validation"]["comparisons"].items():
        candidate, reference = key.split("_vs_", 1)
        a, b = arms.get(candidate), arms.get(reference)
        if not a or not b or not a["feasible"] or not b["feasible"]:
            assert comparison == {"passed": False, "reason": "missing_feasible_pair", "deltas": []}
            continue
        gains = [right["score"] - left["score"] for left, right in zip(a["samples"], b["samples"], strict=True)]
        assert comparison["deltas"] == [
            {"block_id": block, "reference_minus_candidate": gain} for block, gain in zip(expected_blocks, gains, strict=True)]
        assert comparison["median_improvement"] == median(gains)
        assert comparison["passed"] == (all(gain > 0 for gain in gains) and median(gains) >
                                        config["controller_preregistration"]["min_improvement_ms"])
    return receipts, signatures


def verify(run, source, config_path):
    sys.path.insert(0, str(source.resolve(strict=True)))
    from rsi_memory.selection_protocol import Measurement, Proposal, Sample, TrainingArchive, TrainingPlan
    from rsi_memory.topology import fingerprint

    manifest, rounds, calls, measured, native, payload_contracts, validation, contexts = ledger(run / "controller/journal.sqlite")
    assert rounds
    config = read(config_path)
    assert manifest["contract"]["controller_preregistration"] == config["controller_preregistration"]
    assert manifest["contract"]["training_effect_identity"].startswith("exact raw program SHA-256")
    for key in ("generator_program_protocol", "selection_protocol"):
        bound = manifest["contract"][key]["source_sha256"]
        assert all(sha((source / name).read_bytes()) == digest for name, digest in bound.items())
    receipt_dir = (config_path.parent / config["output_directory"]).resolve() / "receipts"
    seed = manifest["seed_sha256"]
    admitted, previous, proposals, raw_trials, output = {seed}, None, [], {}, []
    expected_cases = manifest["contract"]["split_case_sha256"]["train"]
    assert native["status"] == "completed"
    native_job, native_sha, native_signatures = execution_receipt(
        receipt_dir, native["result"], config, expected_cases, fingerprint)
    protocol = manifest["contract"]["selection_protocol"]
    if protocol.get("version") == "4.1":
        for key, proof in payload_contracts.items():
            assert proof["trusted_selection_contract"] == protocol["trusted_selection_contract"], key
            assert proof["legacy_selection_clause_absent"], key
            if key.endswith("-meta"):
                assert proof["meta_system_matches_v41"], key
    all_receipt_ids = set()
    active = seed
    for key, row in rounds.items():
        assert row["active_before"] == active
        assert row["decision"] in ("admitted", "rejected")
        prefix = key.removeprefix("round/")
        round_context = contexts["selection_context/" + prefix]
        for lane, arms in (("adaptive", ("parent", "child")), ("fixed", ("fixed-a", "fixed-b"))):
            for arm in arms:
                first = contexts[f"selection_context/{prefix}/{arm}/00"]
                assert first["archive_sha256"] == round_context[lane + "_sha256"]
                assert first["parent_proposal_ids"] == round_context[lane + "_parent_ids"]
        for api_id, proof in payload_contracts.items():
            if not api_id.startswith(f"api/{prefix}/"):
                continue
            branch = api_id.split("/")[2]
            fixed = branch.startswith("fixed")
            for evidence_id in proof["selected_training_evidence_ids"]:
                if evidence_id in ("native", "bootstrap/initial-incumbent"):
                    continue
                assert ("/fixed-" in evidence_id) == fixed
                if evidence_id.startswith(prefix + "/"):
                    assert not branch.endswith("-meta") and api_id.startswith("api/" + evidence_id.rsplit("/", 1)[0])
                    assert int(evidence_id.rsplit("/", 1)[1]) < int(api_id.rsplit("/", 1)[1])
        parent_source = harness(run, active)
        child_sha = row["proposed_child_sha256"]
        child_source = harness(run, child_sha) if child_sha else None
        assert row["source_diff"] == (list(difflib.unified_diff(
            parent_source.splitlines(), child_source.splitlines(), n=2)) if child_source else [])
        matrix = row["v4_training"]
        plan = matrix["plan"]
        stored = row["v4_selection"]
        fresh_ids = {}
        groups = Counter()
        for arm, lane in (("parent", "adaptive"), ("child", "adaptive"), ("fixed", "fixed")):
            fresh_ids[arm] = []
            for trial in row[arm]:
                trial_id = trial["id"]
                if trial["program"] is None:
                    continue
                call = calls["api/" + trial_id]
                assert call == ("completed", trial["harness_sha256"])
                harness(run, call[1])
                raw = fingerprint(trial["program"])
                assert trial["program_sha256"] == trial["training_effect_sha256"] == raw
                proposal = Proposal(trial_id, lane, call[1], raw, raw, tuple(trial["parent_proposal_ids"]))
                proposals.append(proposal)
                raw_trials[trial_id] = trial
                groups[raw] += 1
                fresh_ids[arm].append(trial_id)
        cohort = matrix["cohort_proposals"]
        expected = {trial_id for ids in fresh_ids.values() for trial_id in ids}
        if previous:
            expected.update(choice["proposal_id"] for name, choice in previous.items()
                            if name in ("manager_best", "admitted_output", "fixed") and choice["proposal_id"])
        assert set(cohort) == expected
        assert all(cohort[name] == raw_trials[name] for name in cohort)
        scope = plan["scope_sha256"]
        assert scope == fingerprint({"contract": manifest["contract"],
                                     "plan": {name: value for name, value in plan.items() if name != "scope_sha256"}})
        schedule = {effect for order in plan["order"] for effect in order}
        assert schedule == set(plan["representatives"]) == set(matrix["measurements"])
        assert all(len(order) == len(set(order)) == len(schedule) for order in plan["order"])
        samples, signatures, receipt_hashes = [], {}, {}
        for effect, result in matrix["measurements"].items():
            times = []
            representative = plan["representatives"][effect]
            representative_id = representative["trial_id"]
            program = manifest["initial_program"] if representative_id.startswith("bootstrap/") else raw_trials[representative_id]["program"]
            assert fingerprint(program) == effect == representative["program_sha256"]
            assert fingerprint({"scope_sha256": scope, "effect_sha256": effect, "evaluations": result["evaluations"]}) == result[
                "qualification_record_sha256"]
            for item in result["evaluations"]:
                eval_key = item["evaluation_key"]
                evaluation = measured[eval_key]
                assert evaluation["input"] == {"split": "train", "program": program}
                assert fingerprint({"evaluation_key": eval_key.removeprefix("eval/"), "record": evaluation}) == item["evaluation_sha256"]
                assert evaluation["status"] == item["status"]
                if "window_ms" not in item:
                    continue
                value = evaluation["result"]
                assert value["feasible"] and value["score"] == item["window_ms"]
                receipts = value["metrics"]["receipts"]
                assert len(receipts) == 1 and item["reported_receipts"] == receipts
                assert fingerprint(receipts) == item["receipt_bundle_sha256"]
                job = receipts[0]
                assert job not in all_receipt_ids
                all_receipt_ids.add(job)
                actual_job, receipt_sha, case_signatures = execution_receipt(
                    receipt_dir, value, config, expected_cases, fingerprint)
                assert actual_job == job and case_signatures == native_signatures
                signatures.update(case_signatures)
                receipt_hashes[job] = receipt_sha
                times.append(Sample(item["block_id"], item["window_ms"], item["receipt_bundle_sha256"]))
            status = result["status"]
            if status == "complete":
                assert median(sample.window_ms for sample in times) == result["median_ms"]
            samples.append(Measurement(scope, effect, result["qualification_record_sha256"], status,
                                       () if status == "correctness_failed" else tuple(times)))

        archive = TrainingArchive(TrainingPlan(scope, tuple(plan["block_ids"]), plan["noise_margin_ms"]),
                                  proposals, samples, trusted_anchor_ids=("bootstrap/initial-incumbent",))
        if row["decision"] == "admitted":
            active = row["proposed_child_sha256"]
            admitted.add(active)
        assert row["active_after"] == active
        choices = {
            "parent": archive.choose("adaptive", proposal_ids=tuple(fresh_ids["parent"])),
            "child": archive.choose("adaptive", proposal_ids=tuple(fresh_ids["child"])),
            "manager_best": archive.choose("adaptive", proposal_ids=tuple(cohort),
                                           incumbent_id=previous["manager_best"]["proposal_id"] if previous else None),
            "admitted_output": archive.choose("adaptive", proposal_ids=tuple(cohort), admitted_sources=admitted,
                                              incumbent_id=previous["admitted_output"]["proposal_id"] if previous else None),
            "fixed": archive.choose("fixed", proposal_ids=tuple(cohort), admitted_sources=(seed,),
                                     incumbent_id=previous["fixed"]["proposal_id"] if previous else None),
        }
        for name, choice in choices.items():
            actual = stored[name]
            ids = fresh_ids[name] if name in ("parent", "child") else tuple(cohort)
            lane = "fixed" if name == "fixed" else "adaptive"
            allowed = {"admitted_output": admitted, "fixed": {seed}}.get(name)
            incumbent_id = previous[name]["proposal_id"] if previous and name in ("manager_best", "admitted_output", "fixed") else None
            assert actual == independently_select(
                raw_trials, matrix["measurements"], ids, lane, allowed, incumbent_id, plan["noise_margin_ms"]), name
            assert actual["status"] == choice.status and actual["frontier_effects"] == list(choice.frontier_effects)
            assert actual["proposal_id"] == (choice.proposal.proposal_id if choice.proposal else None)
            if choice.proposal:
                assert actual["source_sha256"] == choice.proposal.source_sha256
        structural = {}
        for entry, details in (row.get("controlled_behavior") or {}).items():
            parent_ast, child_ast = [{node.name: ast.dump(node) for node in ast.parse(item).body}
                                     for item in (parent_source, child_source)]
            ast_changed = parent_ast[entry] != child_ast[entry]
            assert details["ast_changed"] == ast_changed
            structural[entry] = {"ast_changed": ast_changed, "changed_fields_by_slot": {
                str(slot["slot"]): changed_decision_fields(slot["before"], slot["after"])
                for slot in details["decisions"]}}
        reuse = row.get("previous_admission_reuse")
        if reuse:
            assert reuse["source_sha256"] == row["active_before"]
            for api_id in reuse["task_api_ids"] + ([reuse["meta_api_id"]] if reuse["meta_api_completed"] else []):
                assert calls[api_id] == ("completed", reuse["source_sha256"])
            fields = changed_decision_fields(reuse["previous_source_same_input"], reuse["actual_meta_decision"]) if (
                reuse["previous_source_same_input"] is not None and reuse["actual_meta_decision"] is not None) else []
            assert reuse["changed_structural_fields"] == fields
            assert reuse["recursive_behavior_observed"] == bool(reuse["task_api_ids"] and reuse["meta_api_completed"] and fields)
        validation_record = validation.get("pair/" + key.removeprefix("round/") + "/validation")
        validation_receipts, validation_signatures = {}, {}
        if validation_record:
            validation_receipts, validation_signatures = verify_validation(
                row, validation_record, manifest["contract"]["paired_block_ids"]["validation"],
                manifest["contract"]["split_case_sha256"]["validation"], receipt_dir, config, fingerprint)
        if row["decision"] == "admitted":
            assert row["active_before"] != child_sha == row["active_after"]
            assert set(structural) == {"build_prompt", "build_meta_prompt"}
            assert all(details["ast_changed"] and any(details["changed_fields_by_slot"].values())
                       for details in structural.values())
            parent_effect = stored["parent"]["effect_sha256"]
            child_effect = stored["child"]["effect_sha256"]
            incumbent_effect = fingerprint(row["incumbent_validation"]["trial"]["program"])
            assert child_effect not in (parent_effect, incumbent_effect)
            assert row["child_validation"]["trial"]["harness_sha256"] == child_sha
            for ref in ("parent", "incumbent"):
                assert row["paired_validation"]["comparisons"]["child_vs_" + ref]["passed"]
        output.append({"round": row["round"], "decision": row["decision"], "reason": row["reason"], "active_after": active,
                       "training_scope_sha256": scope, "fresh_program_counts_by_exact_identity": dict(groups),
                       "training_receipt_sha256_by_job": receipt_hashes,
                       "verified_train_case_signatures": signatures, "selected": stored,
                       "validation_receipt_sha256_by_job": validation_receipts,
                       "verified_validation_case_signatures": validation_signatures, "structural_behavior": structural,
                       "validation_comparisons": row["paired_validation"]["comparisons"],
                       "previous_admission_reuse": row.get("previous_admission_reuse")})
        previous = stored
    return {"schema": 1, "kind": "cuda_v4_independent_completed_training_rounds",
            "scope": "Completed discovery rounds only. API projection covers status, source hashes, system digest and selection contract, without model responses or held-out rows.",
            "source_manifest_sha256": sha((source / "SOURCE_MANIFEST.json").read_bytes()),
            "native_train": {"job_id": native_job, "receipt_sha256": native_sha, "case_signatures": native_signatures},
            "provider_selection_contract_proofs": payload_contracts,
            "rounds": output, "training_receipt_count": len(all_receipt_ids), "admitted_sources": sorted(admitted)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source", "config", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.run, args.source, args.config)
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print(sha(args.output.read_bytes()))


if __name__ == "__main__":
    main()
