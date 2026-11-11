"""Render an offline evidence viewer from topology RSI runs and GPU receipts."""

from __future__ import annotations

import argparse
from collections import Counter
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from statistics import median
from zipfile import BadZipFile, ZipFile

from .topology_report_ablation import read_ablation
from .topology_report_assets import PAGE
from .topology_report_sglang import compare_heldout_configurations
from .topology_report_transfer import read_transfer


SCENARIOS = {
    "agent_tools": "Agent · 工具往返",
    "long_document_qa": "LongBench 风格 · 长文问答",
    "chatbot_growth": "Chatbot · 多轮增长",
    "rollout_cow": "Rollout · 分支与写时复制",
}
SHA = re.compile(r"[0-9a-f]{64}")
SGLANG_SCOPE = "real_model_sglang_hisparse_configurable_memory"
CUDA_SCOPE = "real_cuda_fixed_block_sparse"
RESIDENT_ENGINEERING = "human_engineering_resident_backing_probe"


def _read(path):
    value = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(value, dict):
        raise ValueError(f"expected a JSON object in {path.name}")
    return value


def _fingerprint(value):
    data = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)
    return hashlib.sha256(data.encode()).hexdigest()


def _take(value, fields):
    return {key: value[key] for key in fields.split() if key in value} if isinstance(value, dict) else {}


def _number(value):
    return value if type(value) in (int, float) and math.isfinite(value) else None


def _metrics(value, depth=0):
    """Project recorded counters and identifiers without embedding prompts or raw evaluator text."""
    if not isinstance(value, dict) or depth > 3:
        return {}
    projected = {}
    for key, item in value.items():
        if type(item) is bool or _number(item) is not None:
            projected[key] = item
        elif isinstance(item, dict):
            nested = _metrics(item, depth + 1)
            if nested:
                projected[key] = nested
        elif key == "receipts" and isinstance(item, list):
            projected[key] = [value for value in item if isinstance(value, str) and len(value) <= 180]
        elif key == "case_metrics" and isinstance(item, list):
            projected[key] = [_metrics(value, depth + 1) for value in item]
        elif key == "execution_order" and isinstance(item, list):
            projected[key] = [{"block_id": row["block_id"], "order": row["order"]} for row in item
                              if isinstance(row, dict) and isinstance(row.get("block_id"), str)
                              and isinstance(row.get("order"), list)
                              and all(isinstance(label, str) for label in row["order"])]
        elif isinstance(item, str) and len(item) <= 180 and (
            key in {"scope", "status", "score_unit", "receipt", "receipt_id", "job_id", "arm", "fixture_kind",
                    "case_name", "error_type", "failure_category", "block_id", "backend", "evidence", "name", "kind",
                    "experiment_id", "backing"} or key.endswith("_sha256")
        ):
            projected[key] = item
    return projected


def _call(key, value):
    return {"id": key, **_take(value, "status seconds failure"),
            **_take(value.get("input", {}), "harness_sha256"),
            **_take(value.get("input", {}).get("payload", {}), "model max_tokens reasoning_effort"),
            "usage": _metrics(value.get("result", {}).get("usage", {})),
            **_take(value.get("result", {}), "finish_reason")}


def _evaluation(key, value):
    return {"id": key, **_take(value, "status seconds failure"), **_take(value.get("input", {}), "split"),
            **_take(value.get("result", {}), "score feasible"),
            "metrics": _metrics(value.get("result", {}).get("metrics", {}))}


def _pair(key, value):
    return {"id": key, **_take(value, "status seconds failure"),
            **_take(value.get("input", {}), "split block_ids"),
            "arms": {label: {**_take(arm, "feasible score"), "metrics": _metrics(arm.get("metrics")),
                             "samples": [_metrics(sample) for sample in arm.get("samples", [])]}
                     for label, arm in value.get("result", {}).items()}}


def _decision(value):
    return _take(value, "operator parents evidence rule mode") if isinstance(value, dict) else None


def _plan(value):
    return (_take(value, "observed_problem mechanism_change expected_effect falsification training_evidence_ids kind")
            if isinstance(value, dict) else None)


def _recorded_plan(key, value):
    return {"id": key, **_take(value, "arm parent_sha256 proposed_child_sha256 status failure"),
            "decision": _decision(value.get("decision")), "plan": _plan(value.get("plan"))}


def _trial(value):
    if not isinstance(value, dict):
        return None
    program = value.get("program")
    rejection = {key: text[:400] for key, text in _take(value.get("feedback"), "failure_category failure_type failure").items()
                 if isinstance(text, str)}
    return {**_take(value, "id status score failure origin program program_sha256 harness_sha256"),
            "decision": _decision(value.get("decision")), "feedback": _metrics(value.get("feedback")),
            "rejection": rejection,
            "program_hash_matches": _fingerprint(program) == value.get("program_sha256") if program is not None else None}


def _finalist(value):
    if not isinstance(value, dict):
        return None
    return {**_take(value, "score status"), "trial": _trial(value.get("trial"))}


def _task_outputs(value):
    if not isinstance(value, dict):
        raise ValueError("schema 2 round lacks its task output selection")
    effective = value.get("effective_program_sha256_by_raw_program")
    selected, fresh = value.get("selected"), value.get("fresh")
    if (not isinstance(effective, dict) or any(not SHA.fullmatch(k) or not isinstance(v, str) or not SHA.fullmatch(v)
                                               for k, v in effective.items())
            or not isinstance(selected, dict) or set(selected) != {"parent", "child", "fixed"}
            or not isinstance(fresh, dict) or set(fresh) != {"parent", "child", "fixed", "fixed_a", "fixed_b"}):
        raise ValueError("malformed schema 2 task output selection")
    for label, choice in selected.items():
        if (not isinstance(choice, dict) or choice.get("origin") not in {"fresh", "incumbent", "unavailable"}
                or (choice.get("trial") is None) != (choice["origin"] == "unavailable")):
            raise ValueError(f"malformed schema 2 selected output for {label}")
        if choice["origin"] == "incumbent" and choice["trial"] != value.get("fixed_incumbent" if label == "fixed" else "shared_incumbent"):
            raise ValueError(f"schema 2 selected output differs from its incumbent for {label}")
        if choice["origin"] == "fresh" and choice["trial"] != fresh[label]:
            raise ValueError(f"schema 2 selected output differs from its fresh trial for {label}")
    if value.get("parent_incumbent_aa") is not (selected["parent"]["origin"] == "incumbent"):
        raise ValueError("schema 2 A/A flag differs from the selected parent origin")
    def entry(trial):
        result = _trial(trial)
        if result is not None:
            digest = effective.get(result.get("program_sha256"))
            if digest is None:
                raise ValueError("measured task output lacks its effective fingerprint")
            result["effective_program_sha256"] = digest
        return result
    return {"shared_incumbent": entry(value.get("shared_incumbent")), "fixed_incumbent": entry(value.get("fixed_incumbent")),
            "selected": {label: {"origin": choice["origin"], "trial": entry(choice.get("trial"))}
                         for label, choice in selected.items()},
            "fresh": {label: entry(trial) for label, trial in fresh.items()},
            "parent_incumbent_aa": value.get("parent_incumbent_aa"),
            "effective_program_sha256_by_raw_program": effective,
            "fresh_paired_comparison": value.get("fresh_paired_comparison")}


def _round(value, schema):
    result = _take(value, "round active_before active_after proposed_child_sha256 decision reason source_diff "
                          "matched_context_sha256 child_fallback_to_parent previous_admission_reuse "
                          "paired_validation fixed_meta_failure fixed_meta_discarded_sha256 adaptive_meta_plan_id fixed_meta_plan_id")
    for name in ("parent", "child", "fixed"):
        result[name] = [_trial(trial) for trial in value.get(name, [])]
    for name in ("parent_validation", "child_validation"):
        result[name] = _finalist(value.get(name))
    result["fixed_validation"] = [_finalist(item) for item in value.get("fixed_validation", [])]
    result["adaptive_meta_decision"] = _decision(value.get("adaptive_meta_decision"))
    result["fixed_meta_decision"] = _decision(value.get("fixed_meta_decision"))
    result["adaptive_meta_plan"] = _plan(value.get("adaptive_meta_plan"))
    result["fixed_meta_plan"] = _plan(value.get("fixed_meta_plan"))
    result["controlled_behavior"] = {
        name: {"ast_changed": data.get("ast_changed"), "decisions": [
            {**_take(row, "slot changed_fields"), "before": _decision(row.get("before")),
             "after": _decision(row.get("after"))} for row in data.get("decisions", [])]}
        for name, data in (value.get("controlled_behavior") or {}).items()}
    if schema == 2:
        result["task_output_selection"] = _task_outputs(value.get("task_output_selection"))
        result["incumbent_validation"] = _finalist(value.get("incumbent_validation"))
    return result


def _records(directory):
    journal = directory / "journal.sqlite"
    if journal.is_file():
        with closing(sqlite3.connect(journal.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)) as connection:
            connection.execute("PRAGMA query_only=ON")
            rows = connection.execute("SELECT key, data FROM records ORDER BY key").fetchall()
        return {key: json.loads(data) for key, data in rows}, "journal.sqlite"
    report = directory / "report.json"
    exported = _read(report) if report.is_file() else {}
    if not exported:
        manifest = directory / "manifest.json"
        if not manifest.is_file():
            raise ValueError(f"no topology RSI journal, report or manifest in {directory.name}")
        exported = {"manifest": _read(manifest)}
        frozen = directory / "frozen.json"
        if frozen.is_file():
            exported["frozen"] = _read(frozen)
        exported["rounds"] = [_read(path) for path in sorted((directory / "rounds").glob("r*.json"))]
    return {"manifest": exported.get("manifest"), "frozen": exported.get("frozen"),
            "test_comparisons": exported.get("test_comparisons"), "status": exported.get("status"),
            **{f"round/r{row['round']:04d}": row for row in exported.get("rounds", [])},
            **exported.get("calls", {}), **exported.get("evaluations", {}),
            **exported.get("paired_evaluations", {}), **exported.get("meta_plans", {})}, "report.json" if report.is_file() else "JSON checkpoints"


def _source(directory, sha):
    if not isinstance(sha, str) or not SHA.fullmatch(sha):
        return None
    path = directory / "harnesses" / f"{sha}.py"
    if not path.is_file() or not path.resolve().is_relative_to(directory.resolve()):
        return {"status": "missing"}
    raw = path.read_bytes()
    if hashlib.sha256(raw).hexdigest() != sha:
        return {"status": "hash_mismatch"}
    return {"status": "verified", "source": raw.decode("utf-8")}


def _sglang_contract(contract):
    if not isinstance(contract, dict) or contract.get("kind") != SGLANG_SCOPE:
        return None
    selection = contract.get("manual_selection")
    return {"native_comparators": {key: _take(row, "hot_slots backing backing_mode layer_tiers_by_rank")
                                   for key, row in contract.get("native_comparators", {}).items()},
            "manual_selection": _take(selection, "scope selected selected_program_sha256 selected_effective_sha256 "
                                                 "hot_slots layer_tiers_by_rank backing_mode sweep_sha256 artifact_sha256 "
                                                 "receipts_sha256 receipt_count") or None,
            "preregistration": _take(contract.get("controller_preregistration"), "rounds slots validation_blocks test_blocks "
                                      "minimum_paired_blocks min_improvement_ms min_improvement_formula"),
            "manual_program_sha256": contract.get("manual_program_sha256")}


def _heldout_outcome(records, manifest, frozen):
    pair = records.get("pair/frozen/test") or {}
    comparisons = records.get("test_comparisons")
    if not pair and frozen and frozen["hashes_match"] and all(
        records.get(key, {}).get("status") == "completed"
        for key in ("eval/native/test", "eval/frozen/adaptive", "eval/frozen/fixed")
    ):
        return {"status": "unpaired"}
    if not pair or pair.get("status") != "completed" or not comparisons:
        return {"status": "pending", "pair_status": pair.get("status")}
    if not frozen or not frozen["hashes_match"]:
        return {"status": "unverified"}
    inputs, arms = pair.get("input", {}), pair.get("result", {})
    blocks = inputs.get("block_ids")
    contract = manifest.get("contract") or {}
    declared = contract.get("paired_block_ids", {}).get("test")
    programs = inputs.get("programs", {})
    labels = ["native", "fixed", *(["manual"] if "manual" in programs else [])]
    keys = {f"adaptive_vs_{label}" for label in labels}
    if (inputs.get("split") != "test" or not isinstance(blocks, list) or not blocks
            or any(not isinstance(block, str) for block in blocks) or len(blocks) != len(set(blocks))
            or declared != blocks or set(programs) != set(arms) or set(arms) != {"adaptive", *labels}
            or set(comparisons) != keys or programs["native"] is not None
            or any(not frozen[label] or not frozen[label]["trial"] or not frozen[label]["trial"]["program_hash_matches"]
                   or frozen[label]["trial"]["program"] != programs[label] for label in ("adaptive", "fixed"))):
        return {"status": "unverified"}
    if "manual" in programs and _fingerprint(programs["manual"]) != contract.get("manual_program_sha256"):
        return {"status": "unverified"}
    values = {}
    margin = _number(manifest["config"].get("min_improvement"))
    for label in labels:
        claim = comparisons[f"adaptive_vs_{label}"]
        candidate, reference = arms["adaptive"], arms[label]
        if not candidate.get("feasible") or not reference.get("feasible"):
            if claim.get("passed") is not False or claim.get("deltas") != []:
                return {"status": "unverified"}
            values[label] = {"status": "infeasible"}
            continue
        matched = [arm.get("samples", []) for arm in (candidate, reference)]
        if any([row.get("block_id") for row in samples] != blocks
               or any(_number(row.get("score")) is None for row in samples) for samples in matched):
            return {"status": "unverified"}
        deltas = [right["score"] - left["score"] for left, right in zip(*matched, strict=True)]
        gain = median(deltas)
        wins = sum(delta > 0 for delta in deltas)
        expected = [{"block_id": block, "reference_minus_candidate": delta} for block, delta in zip(blocks, deltas, strict=True)]
        passed = claim.get("passed")
        if (claim.get("deltas") != expected or claim.get("median_improvement") != gain or type(passed) is not bool
                or margin is not None and passed != (wins == len(blocks) and gain > margin)):
            return {"status": "unverified"}
        values[label] = {"status": "passed" if passed else "rejected", "wins": wins, "median_gain": gain}
    scope = contract.get("kind")
    return {"status": "settled", "scope": scope, "unit": "ms" if scope in (SGLANG_SCOPE, CUDA_SCOPE) else None,
            "blocks": len(blocks), "margin": margin, "comparisons": values}


def _formal_anchor(directory, manifest, frozen, frozen_view):
    contract = manifest.get("contract", {})
    if (contract.get("kind") != SGLANG_SCOPE or not frozen_view or not frozen_view["hashes_match"]
            or frozen.get("partial") is not False or frozen.get("rounds") != manifest["config"].get("rounds")
            or contract.get("generator_program_protocol", {}).get("kind") != "backend_program_contract_v3"):
        return None
    frozen_path, report_path = directory / "frozen.json", directory / "report.json"
    pipeline_path, archive_path = directory.parent / "pipeline-status.json", directory.parent / "frozen-source.zip"
    if any(not path.is_file() or path.is_symlink() for path in (frozen_path, report_path, pipeline_path, archive_path)):
        return None
    try:
        frozen_raw, report_raw, pipeline_raw = (path.read_bytes() for path in (frozen_path, report_path, pipeline_path))
        report, pipeline = json.loads(report_raw), json.loads(pipeline_raw)
        terminal = report.get("paired_evaluations", {}).get("pair/frozen/test", {})
        trial = frozen["adaptive"]["trial"]
        if (json.loads(frozen_raw) != frozen or report.get("status") != "test_settled" or report.get("frozen") != frozen
                or report.get("manifest") != manifest or terminal.get("status") not in ("completed", "failed", "interrupted_unknown")
                or terminal.get("input", {}).get("split") != "test" or terminal["input"].get("programs", {}).get("adaptive") != trial["program"]
                or trial["program_sha256"] != _fingerprint(trial["program"]) or trial["harness_sha256"] not in frozen["admitted_harness_sha256s"]
                or [row.get("phase") for row in pipeline] != ["run", "freeze", "test"] or any(row.get("returncode") != 0 for row in pipeline)):
            return None
        with ZipFile(archive_path) as archive:
            local_manifest = archive.read("LOCAL_SOURCE_MANIFEST.json")
    except (BadZipFile, OSError, KeyError, TypeError, AttributeError, ValueError):
        return None
    return {"formal_frozen_sha256": hashlib.sha256(frozen_raw).hexdigest(),
            "formal_report_sha256": hashlib.sha256(report_raw).hexdigest(), "formal_manifest_canonical_sha256": _fingerprint(manifest),
            "formal_pipeline_status_sha256": hashlib.sha256(pipeline_raw).hexdigest(),
            "controller_source_archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            "controller_source_manifest_sha256": hashlib.sha256(local_manifest).hexdigest(),
            "adaptive_program_sha256": trial["program_sha256"], "adaptive_trial_id": trial["id"],
            "adaptive_harness_sha256": trial.get("harness_sha256"), "cases": contract.get("cases", {}).get("test"),
            "worker_source_manifest_sha256": contract.get("source_manifest_sha256"),
            "metadata_profile_sha256": contract.get("metadata_profile_sha256"), "launch_sha256": contract.get("launch_sha256"),
            "tokenization_sha256": contract.get("tokenization_sha256"), "margin_ms": manifest["config"].get("min_improvement"),
            "margin_hand_sweep_canonical_sha256": contract.get("manual_selection", {}).get("sweep_sha256")}


def read_run(directory, label=None, *, receipt_index=None, include_ablation_anchor=False):
    records, origin = _records(directory)
    manifest = records.get("manifest")
    if not isinstance(manifest, dict) or not isinstance(manifest.get("config"), dict):
        raise ValueError(f"missing topology RSI manifest in {directory.name}")
    schema = manifest.get("schema", 1)
    if type(schema) is not int or schema not in (1, 2):
        raise ValueError(f"unsupported RSI manifest schema in {directory.name}")
    if schema == 2 and manifest.get("task_output_protocol") != "shared_measured_train_incumbent_v1":
        raise ValueError(f"unknown schema 2 output protocol in {directory.name}")
    rows = [value for key, value in records.items() if key.startswith("round/")]
    rows.sort(key=lambda value: value["round"])
    calls = [_call(key, value) for key, value in records.items() if key.startswith("api/")]
    evaluations = [_evaluation(key, value) for key, value in records.items() if key.startswith("eval/")]
    pairs = [_pair(key, value) for key, value in records.items() if key.startswith("pair/")]
    plans = [_recorded_plan(key, value) for key, value in records.items() if key.startswith("plan/")]
    frozen = records.get("frozen")
    frozen_view = None
    if isinstance(frozen, dict):
        frozen_view = {**_take(frozen, "rounds partial active_harness_sha256 native_validation_score "
                                      "manifest_sha256 rounds_sha256 admitted_harness_sha256s reused_harness_sha256s"),
                       "hashes_match": (frozen.get("manifest_sha256") == _fingerprint(manifest)
                                        and frozen.get("rounds_sha256") == _fingerprint(rows)),
                       "adaptive": _finalist(frozen.get("adaptive")), "fixed": _finalist(frozen.get("fixed")),
                       "adaptive_archive_best": _finalist(frozen.get("adaptive_archive_best"))}
    hashes = {manifest.get("seed_sha256")}
    for row in rows:
        hashes.update(row.get(key) for key in ("active_before", "active_after", "proposed_child_sha256"))
    contract = manifest.get("contract", {})
    outcome = _heldout_outcome(records, manifest, frozen_view)
    if outcome.get("scope") == SGLANG_SCOPE and outcome["status"] == "settled":
        pair = records["pair/frozen/test"]
        programs = {arm: None if program is None else _fingerprint(program)
                    for arm, program in pair["input"]["programs"].items()}
        outcome["recorded_configurations"] = compare_heldout_configurations(pair, contract, programs, receipt_index or {})
    return {"label": label or directory.name, "origin": origin, "recorded_status": records.get("status"), "schema": schema,
            "manifest_sha256": _fingerprint(manifest), "contract": _metrics(contract), "sglang": _sglang_contract(contract),
            "manifest": _take(manifest, "config paired_gate supported_actions started_unix seed_sha256 source_sha256 "
                                        "candidate_slots_per_arm_per_round provider_calls_per_arm_per_round_cap "
                                        "task_output_protocol effective_program_fingerprint validation_arms"),
            "rounds": [_round(row, schema) for row in rows], "frozen": frozen_view,
            "heldout_outcome": outcome,
            **({"ablation_anchor": _formal_anchor(directory, manifest, frozen, frozen_view)} if include_ablation_anchor else {}),
            "test_comparisons": records.get("test_comparisons"), "calls": calls,
            "evaluations": evaluations, "pairs": pairs, "meta_plans": plans,
            "sources": {sha: _source(directory, sha) for sha in sorted(sha for sha in hashes if isinstance(sha, str))}}


def _case(value, max_actions):
    diagnostic = value.get("diagnostic", {})
    actions = diagnostic.get("actions", [])
    return {**_take(value, "name case_sha256 median_elapsed_ns"),
            "diagnostic": {**_take(diagnostic, "status scope arm fixture_kind schedule_execution elapsed_ns transfer_wall_ns "
                                              "managed_peak_bytes_by_node managed_reserved_bytes_by_node "
                                              "cuda_peak_allocated_bytes_delta_by_node link_bytes shared_bottleneck_bytes "
                                              "live_references_at_end total_fetch_misses workload_sha256 schedule_sha256 "
                                              "program_sha256 topology_sha256 scenario_sha256 selected_and_output_sha256"),
                           "action_count": len(actions),
                           "action_types": dict(Counter(row.get("action", "unknown") for row in actions)),
                           "actions": actions[:max_actions], "semantics": diagnostic.get("semantics", [])},
            "timings_elapsed_ns": [row.get("elapsed_ns") for row in value.get("timings", [])]}


def _serving(value):
    compilation = value.get("memory_compilation", {})
    startup = _take(compilation.get("startup"), "hot_slots rule_index native_reference_override eligible_hot_slots "
                                                "metadata_bytes_reserved_by_node backing_mode backing_rule_index "
                                                "case_max_request_tokens resident_sentinel_logical_bytes_by_node "
                                                "resident_sentinel_allocator_charge_bytes_by_node")
    proof = compilation.get("startup", {}).get("resident")
    if isinstance(proof, dict):
        startup["resident"] = {**_take(proof, "schema engine_sha256 device_rows reserved_rows_by_rank"),
                               "proof_sha256": _fingerprint(proof),
                               "requests": [_take(row, "phase rid rank prompt_length prompt_sha256 max_new_tokens")
                                            for row in proof.get("requests", [])]}
    metrics = value.get("metrics", {})
    ranks = [_take(row, "rank layer_tiers hot_slots packed_device_payload_bytes device_physical_rows "
                        "device_logical_row_capacity request_metadata_slots_including_pad hot_metadata_bytes worker_limits "
                        "pinned_host_bytes local_hbm_bytes peer_hbm_bytes relay_workspace_bytes "
                        "selected_miss_payload_bytes_by_tier backup_payload_bytes_by_tier relay torch_cuda_memory "
                        "resident_proof_sha256 resident_sentinel_logical_bytes resident_sentinel_allocated_delta_bytes "
                        "resident_sentinel_reserved_delta_bytes resident_sentinel_allocator_charge_bytes resident_producer_oracle")
             for row in metrics.get("ranks", [])]
    return {**_take(value, "arm phase program_sha256 cold_program_sha256 compiled_plan_sha256 failure raw_result_sha256"),
            "case": _take(value.get("case"), "id manifest_sha256 target_sha256 topology_sha256 scenario_sha256 "
                                             "workload_sha256 schedule_sha256"),
            "memory_compilation": {**_take(compilation, "full_program_sha256 cold_program_sha256 placement_rule_indices_in_full_program "
                                                      "metadata_profile_sha256 base_target_sha256 base_topology_sha256 "
                                                      "external_scheduler_sha256 ingress_contract_sha256 derived_target_sha256 derived_schedule_sha256 "
                                                      "derived_topology_sha256 derived_scenario_sha256 cold_plan_sha256 "
                                                      "total_incremental_bytes_by_node equivalent_to_strong_stock_control cold_plan_role"),
                                   "startup": startup},
            "metrics": {**_take(metrics, "effective_schedule_sha256 dispatch_sha256 output_ids_sha256 packed_kv_sha256_by_rank "
                                         "runtime_sha256 input_tokenization_sha256 timing global_ticks_sha256 "
                                         "backing_mode resident_binding_sha256"), "ranks": ranks},
            "feedback": _take(value.get("feedback"), "cold_miss_rows_by_rank_layer kv_log_sha256_by_rank"),
            "cross_capacity_native_parity": _take(value.get("cross_capacity_native_parity"), "reference matched")}


def read_receipt(path, max_actions):
    value = _read(path)
    if value.get("schema") != 2 or not isinstance(value.get("job_id"), str):
        raise ValueError(f"expected a GPU execution v2 receipt in {path.name}")
    result = value.get("result")
    digest_matches = isinstance(result, dict) and value.get("result_canonical_sha256") == _fingerprint(result)
    serving = _serving(result) if digest_matches and result.get("scope") == SGLANG_SCOPE else None
    return {"file": path.name, **_take(value, "job_id status request_sha256 source_manifest_sha256 gpu_indices gpu_uuids "
                                             "utc_start utc_end elapsed_seconds returncode failure timeout environment_conflict "
                                             "result_sha256 result_canonical_sha256 log_sha256 remaining_compute_pids "
                                             "after_query_failure result_failure replayed_receipt process_group_remaining "
                                             "process_group_cleanup_required"),
            "canonical_result_matches": digest_matches if result is not None else None,
            "result": ({**_take(result, "scope score score_unit feasible hardware error_type"),
                        "cases": [_case(case, max_actions) for case in result.get("cases", [])],
                        **({"serving": serving} if serving is not None else {})}
                       if digest_matches else None)}


def _scenario(path, kind=None):
    value = _read(path)
    recorded_kind = value.get("kind")
    if recorded_kind in SCENARIOS and kind not in (None, recorded_kind):
        raise ValueError(f"scenario kind does not match {path.name}")
    if kind in SCENARIOS and {"rank_placement", "attention", "target_sha256", "workload_sha256", "schedule_sha256"} <= value.keys():
        return {"kind": kind, "file": path.name, "binding_sha256": _fingerprint(value),
                "binding": _take(value, "rank_placement attention target_sha256 workload_sha256 schedule_sha256")}
    if recorded_kind not in SCENARIOS:
        raise ValueError(f"unknown scenario kind in {path.name}")
    return {**_take(value, "kind name page_rows"), "file": path.name, "fixture_sha256": _fingerprint(value),
            "segments": len(value.get("segments", [])),
            "events": dict(Counter(row.get("kind", row.get("op", "unknown")) for row in value.get("events", [])))}


def _manual_sweep(directory):
    manifest = _read(directory / "manifest.json")
    policies, order = manifest.get("policies"), manifest.get("declared_policy_order")
    margin = _number(manifest.get("minimum_paired_median_gain_ms"))
    reference = manifest.get("reference")
    if (manifest.get("schema") != 1 or manifest.get("scope") != "train-only independent hand-policy sweep"
            or not isinstance(policies, dict) or not isinstance(order, list) or len(order) != len(policies)
            or set(order) != policies.keys() or reference not in policies or margin is None or margin <= 0):
        raise ValueError("invalid training-only SGLang hand sweep")
    sweep_sha = _fingerprint(manifest)
    result_path = directory / "result.json"
    outcome = _read(result_path) if result_path.is_file() else {}
    status = outcome.get("status", "preflight_only")
    if outcome and (outcome.get("sweep_sha256") != sweep_sha or status not in ("failed", "training_selection_only")):
        raise ValueError("hand sweep result differs from its preflight manifest")
    selected = None
    samples, gains, failures, eligible = {}, {}, {}, []
    if status == "training_selection_only":
        samples = outcome["sum_window_ms_by_block"]
        failures = outcome["parity_failures"]
        passed_preflight = {label for label, row in policies.items() if row["status"] == "preflight_passed"}
        block_count = len(manifest["order"])
        if (not block_count or reference not in samples or set(samples) & failures.keys()
                or set(samples) | failures.keys() != passed_preflight
                or any(len(values) != block_count or any(_number(v) is None or v <= 0 for v in values)
                       for values in samples.values())):
            raise ValueError("hand sweep lacks its declared training-only observations")
        gains = {label: [baseline - score for baseline, score in zip(samples[reference], values, strict=True)]
                 for label, values in samples.items()}
        eligible = [label for label, values in gains.items() if min(values) > 0 and median(values) >= margin]
        selected = min(eligible, key=lambda label: (-median(gains[label]), label)) if eligible else reference
        if (gains != outcome["paired_gain_over_stock_6144_ms_by_block"]
                or set(eligible) != set(outcome["eligible_to_displace_stock_6144"])
                or selected != outcome["selected"] or outcome["selected_program_sha256"] != policies[selected]["program_sha256"]
                or _fingerprint(_read(directory / "selected_program.json")) != policies[selected]["program_sha256"]):
            raise ValueError("hand sweep winner differs from the recorded training comparison")
    artifact_sha = {name: hashlib.sha256(path.read_bytes()).hexdigest()
                    for name in ("manifest.json", "result.json", "selected_program.json", "jobs.jsonl")
                    if (path := directory / name).is_file()}
    return {"name": directory.name, "sweep_sha256": sweep_sha, "artifact_sha256": artifact_sha, "status": status,
            "reference": reference, "selected": selected, "source_manifest_sha256": manifest.get("source_manifest_sha256"),
            "minimum_gain_ms": margin, "block_count": len(manifest["order"]), "case_count": len(manifest["cases"]),
            "policies": [{"label": label, **_take(policies[label], "status backing_mode local_layer_on_each_rank "
                              "program_sha256 effective_sha256 layer_tiers_by_rank"),
                          "hot_slots": policies[label].get("compilation", {}).get("startup", {}).get("hot_slots"),
                          "window_ms_by_block": samples.get(label), "gain_ms_by_block": gains.get(label),
                          "eligible": label in eligible, "failure": _take(failures.get(label), "failure job_id") or None}
                         for label in order]}


def _resident_engineering(path, value):
    phases = value.get("phases", {})
    order = ["stock", "resident", "resident", "stock"]
    if (value.get("schema") != 1 or any(value.get(key) is not False for key in
               ("generated_program", "recursive_improvement", "heldout", "posthoc_stability_claim"))
            or phases.keys() != {"parity", "performance"}):
        raise ValueError(f"expected a public resident engineering audit in {path.name}")
    trials = {}
    members = value.get("members", {})
    for phase, rows in phases.items():
        if [row.get("mode") for row in rows] != order:
            raise ValueError("resident engineering audit differs from its paired order")
        trials[phase] = []
        for index, row in enumerate(rows, 1):
            mode = row["mode"]
            member = f"resident-{phase}-v3/{index:02d}-{mode}/result.json"
            if (row.get("trial") != index or row.get("raw_result_sha256") != members.get(member, {}).get("sha256")
                    or not SHA.fullmatch(str(row.get("raw_result_sha256")))
                    or [rank.get("rank") for rank in row["ranks"]] != [0, 1]
                    or any(rank.get("mode") != mode or (rank.get("oracle") is not None) != (phase == "parity")
                           for rank in row["ranks"])):
                raise ValueError("resident engineering audit lacks matching result inventory or rank evidence")
            trials[phase].append({**_take(row, "trial mode raw_result_sha256 window_ms"),
                                  "ranks": [_take(rank, "rank cold_payload_bytes cuda_pointer_sentinel_logical_bytes "
                                                        "hot_geometry oracle proof_sha256") for rank in row["ranks"]]})
    windows = [_number(row.get("window_ms")) for row in trials["performance"]]
    if any(window is None or window <= 0 for window in windows):
        raise ValueError("resident engineering audit lacks all timed windows")
    pairs = [{"stock_ms": windows[a], "resident_ms": windows[b], "gain_ms": windows[a] - windows[b]}
             for a, b in ((0, 1), (3, 2))]
    if [pair["gain_ms"] for pair in pairs] != value.get("paired_gain_ms"):
        raise ValueError("resident engineering audit paired differences changed")
    archive = path.with_name("resident-v3-evidence.zip")
    archive_matches = None
    if archive.is_file():
        archive_matches = hashlib.sha256(archive.read_bytes()).hexdigest() == value.get("evidence_archive_sha256")
        if not archive_matches:
            raise ValueError("resident engineering evidence archive digest changed")
    return {"file": path.name, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "kind": RESIDENT_ENGINEERING,
            **_take(value, "scope source_manifest_sha256 evidence_archive_sha256"), "archive_matches": archive_matches,
            "trials": trials, "pairs": pairs}


def _engineering(path):
    value = _read(path)
    if value.get("kind") == RESIDENT_ENGINEERING:
        return _resident_engineering(path, value)
    if (value.get("kind") != "single_gpu_engineering_training_qualification" or value.get("gpu_count") != 1
            or value.get("heldout") is not False or value.get("model_generated_outcome") is not False):
        raise ValueError(f"expected a public single-GPU engineering audit in {path.name}")
    return {"file": path.name, "file_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            **_take(value, "scope source_manifest_sha256 preregistration_sha256 receipt_count fresh_process_count paired_blocks "
                           "environment_scope native_lru_link_bytes_and_misses_exact "
                           "exact_case_selected_bytes_and_outputs_across_arms "
                           "matches_v10_case_program_selected_output_actions_traffic_misses_and_managed_peaks"),
            "scopes": {name: _take(row, "median_ms descriptive_comparisons") for name, row in value["scopes"].items()}}


def render(output, *, runs=(), receipts=(), topologies=(), scenarios=(), transfers=(), engineering=(), manual_sweeps=(), ablations=(), max_actions=200):
    if not 0 <= max_actions <= 10000:
        raise ValueError("max_actions must be between 0 and 10000")
    graphs = []
    for path in topologies:
        value = _read(path)
        if not isinstance(value.get("nodes"), list) or not isinstance(value.get("links"), list):
            raise ValueError(f"expected topology nodes and links in {path.name}")
        graphs.append({"name": path.name, "sha256": _fingerprint(value),
                       **_take(value, "nodes links bottlenecks")})
    observations = [read_receipt(path, max_actions) for path in receipts]
    receipt_index = {}
    for receipt in observations:
        key = receipt["job_id"]
        receipt_index[key] = receipt if key not in receipt_index else None
    ablations = tuple(ablations)
    run_views = [read_run(path, label, receipt_index=receipt_index, include_ablation_anchor=bool(ablations)) for label, path in runs]
    anchors = [view.pop("ablation_anchor") for view in run_views] if ablations else []
    data = {"runs": run_views,
            "receipts": observations, "topologies": graphs,
            "scenarios": [_scenario(*entry) if isinstance(entry, tuple) else _scenario(entry) for entry in scenarios],
            "transfers": [read_transfer(path) for path in transfers], "engineering": [_engineering(path) for path in engineering],
            "manual_sweeps": [_manual_sweep(path) for path in manual_sweeps],
            "ablations": [read_ablation(path, [anchor for anchor in anchors if anchor], read_receipt, _fingerprint) for path in ablations],
            "scenario_labels": SCENARIOS}
    embedded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8", newline="\n") as stream:
        stream.write(PAGE.replace("/* EMBEDDED_RUN_DATA */", embedded, 1))
    return data


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--run", action="append", default=[], metavar="[LABEL=]DIRECTORY")
    parser.add_argument("--receipt", type=Path, action="append", default=[])
    parser.add_argument("--receipts", type=Path, action="append", default=[], metavar="DIRECTORY")
    parser.add_argument("--topology", type=Path, action="append", default=[])
    parser.add_argument("--scenario", action="append", default=[], metavar="[KIND=]JSON")
    parser.add_argument("--transfer", type=Path, action="append", default=[], metavar="STUDY_DIRECTORY")
    parser.add_argument("--engineering", type=Path, action="append", default=[], metavar="AUDIT_JSON")
    parser.add_argument("--manual-sweep", type=Path, action="append", default=[], metavar="TRAIN_SWEEP_DIRECTORY")
    parser.add_argument("--ablation", type=Path, action="append", default=[], metavar="POST_FREEZE_STUDY_DIRECTORY")
    parser.add_argument("--max-actions", type=int, default=200)
    args = parser.parse_args(argv)
    runs = []
    for entry in args.run:
        label, separator, path = entry.partition("=")
        runs.append((label, Path(path)) if separator else (None, Path(entry)))
    receipts = dict.fromkeys([*args.receipt, *(path for directory in args.receipts
                            for path in sorted(directory.glob("*.json")))])
    scenarios = []
    for entry in args.scenario:
        kind, separator, path = entry.partition("=")
        scenarios.append((Path(path), kind) if separator else Path(entry))
    render(args.output, runs=runs, receipts=receipts, topologies=args.topology, scenarios=scenarios,
           transfers=args.transfer, engineering=args.engineering, manual_sweeps=args.manual_sweep, ablations=args.ablation,
           max_actions=args.max_actions)
    print(args.output.resolve())


if __name__ == "__main__":
    main()
