"""Project the frozen transfer study into the viewer without reading model requests or fixtures."""

from contextlib import closing
from hashlib import sha256
import json
import math
import sqlite3


_PHASES = ("baseline", "contexts", "generated", "selected", "heldout")
_SCENARIOS = ("agent_tools", "chatbot_growth", "long_document_qa", "rollout_cow")
_SOURCES = ("seed", "admitted")
_REFERENCES = ("seed", "native", "manual-direct")


def _take(value, *fields):
    return {key: value[key] for key in fields if key in value} if isinstance(value, dict) else {}


def _read(path, expected=None):
    raw = path.read_bytes()
    digest = sha256(raw).hexdigest()
    if expected is not None and digest != expected:
        raise ValueError(f"transfer seal does not match {path.name}")
    value = json.loads(raw)
    if not isinstance(value, dict):
        raise ValueError(f"expected a transfer object in {path.name}")
    return value, digest


def _seals(directory, plan_sha):
    path = directory / "journal.sqlite"
    if not path.is_file():
        return None
    keys = ("identity", *[f"phase/{name}" for name in _PHASES])
    placeholders = ",".join("?" for _ in keys)
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True, timeout=3)) as db:
        db.execute("PRAGMA query_only=ON")
        records = {key: json.loads(data) for key, data in
                   db.execute(f"SELECT key, data FROM records WHERE key IN ({placeholders})", keys)}
    if records.pop("identity", None) != {"preregistration_sha256": plan_sha}:
        raise ValueError("transfer journal belongs to a different preregistration")
    return {key.split("/", 1)[1]: value for key, value in records.items()}


def _program(trial):
    if trial is None:
        return None
    program = trial.get("program")
    payload = json.dumps(program, sort_keys=True, ensure_ascii=True, separators=(",", ":"), allow_nan=False).encode()
    return {**_take(trial, "slot_id", "program_sha256", "train_score_ms", "training_receipt_id"),
            "program_hash_matches": program is not None and sha256(payload).hexdigest() == trial.get("program_sha256")}


def _block(row):
    arms = {}
    for label, arm in row["arms"].items():
        arms[label] = {**_take(arm, "feasible", "score", "failure_type", "receipt_status", "receipt_id", "result_sha256"),
                       "cases": [_take(case, "name", "case_sha256", "attention_sha256", "physical_trace_sha256",
                                       "execution_signature", "score_ms", "peak_bytes_by_node", "wire_bytes", "fetch_misses")
                                 for case in arm.get("cases", [])]}
    return {"block_id": row["block_id"], "order": row["order"], "arms": arms}


def _comparison(value, margin, block_ids, blocks, label):
    aggregate = value["scopes"]["aggregate"]
    rows = aggregate["blocks"]
    paired = {row["block_id"]: (row["arms"]["admitted"]["score"], row["arms"][label]["score"])
              for row in blocks if all(row["arms"].get(arm, {}).get("feasible") for arm in ("admitted", label))}
    if ([row["block_id"] for row in rows] != list(paired) or aggregate["preregistered_minimum_gain_ms"] != margin
            or any((row["candidate_ms"], row["reference_ms"]) != paired[row["block_id"]]
                   or not math.isclose(row["reference_ms"] - row["candidate_ms"], row["gain_ms"], abs_tol=1e-12)
                   for row in rows)):
        raise ValueError("transfer comparison differs from actual paired arm scores")
    complete = len(paired) == len(block_ids) == 12
    if complete != value["complete"]:
        raise ValueError("transfer comparison feasibility differs from its blocks")
    strict = complete and not value.get("identical_program_to_candidate") and all(row["gain_ms"] > margin for row in rows)
    if strict != value["strict_all_blocks_above_margin"]:
        raise ValueError("transfer strict decision differs from its recorded block gains")
    return {**_take(value, "candidate_status", "comparator_status", "missing_block_ids", "complete",
                    "strict_all_blocks_above_margin", "v7_style_all_positive_and_median_above_margin",
                    "identical_program_to_candidate"),
            "aggregate": {**_take(aggregate, "median_gain_ms", "median_relative_gain_pct", "paired_bootstrap_interval_ms",
                                   "preregistered_minimum_gain_ms", "all_blocks_positive", "interval_above_minimum"),
                          "blocks": [_take(row, "block_id", "candidate_ms", "reference_ms", "gain_ms") for row in rows]}}


def _observed_blocks(blocks, planned, schedule, case_hashes):
    if [row["block_id"] for row in blocks] != planned or [row["order"] for row in blocks] != schedule:
        raise ValueError("transfer held-out block IDs or order differ from preregistration")
    for row in blocks:
        reference = row["arms"]["native"]
        if set(row["order"]) != set(row["arms"]):
            raise ValueError("transfer held-out block lacks a scheduled arm")
        for label, arm in row["arms"].items():
            if not arm.get("feasible"):
                continue
            if [case["case_sha256"] for case in arm["cases"]] != case_hashes:
                raise ValueError("a feasible transfer arm lacks the planned case identities")
            if label == "native":
                continue
            fields = ("case_sha256", "attention_sha256")
            if not reference.get("feasible") or ([tuple(case[key] for key in fields) for case in arm["cases"]]
                    != [tuple(case[key] for key in fields) for case in reference["cases"]]):
                raise ValueError("a feasible transfer arm lacks matching selected-KV/output hashes")


def read_transfer(directory):
    plan, plan_sha = _read(directory / "preregistration.json")
    if (plan.get("schema") != 1 or plan.get("kind") != "frozen_v7_generators_transfer_on_v11"
            or set(plan["scenarios"]) != set(_SCENARIOS) or set(plan["generators"]) != set(_SOURCES)
            or plan["budget"]["family_comparisons"] != 12):
        raise ValueError(f"unsupported generator transfer preregistration in {directory.name}")
    seals = _seals(directory, plan_sha)
    phases = {}
    # Contexts are intentionally not read. Their digest still links all generated slots to the same frozen inputs.
    for name in ("baseline", "generated", "selected", "heldout"):
        digest = (seals or {}).get(name)
        if digest is not None:
            phases[name], _ = _read(directory / f"{name}.json", digest)
    generated, selected, heldout = (phases.get(name, {}) for name in ("generated", "selected", "heldout"))
    if (selected and selected["generated_sha256"] != seals.get("generated")
            or heldout and (heldout["selection_sha256"] != seals.get("selected") or heldout["primary_comparisons"] != 12)
            or generated and generated["context_sha256"] != seals.get("contexts")):
        raise ValueError("transfer phase seal lineage does not match")
    if heldout and set(heldout["scenarios"]) != set(_SCENARIOS):
        raise ValueError("settled transfer result lacks a fixed scenario")
    scenario_views = {}
    for name in _SCENARIOS:
        design = plan["scenarios"][name]
        margin = plan["margins_ms"][name]
        block_ids = design["block_ids"]
        controls = phases.get("baseline", {}).get(name, {})
        slots = [slot for slot in generated.get("slots", {}).values() if slot.get("scenario") == name]
        winners = {label: _program(row) for label, row in selected.get("winners", {}).get(name, {}).items()}
        if any(row is not None and not row["program_hash_matches"] for row in winners.values()):
            raise ValueError("a sealed transfer winner differs from its program digest")
        outcome = heldout.get("scenarios", {}).get(name)
        row = {"margin_ms": margin, "block_ids": block_ids, "gpu_indices": plan["operator_profiles"][name]["gpu_indices"],
               "design": _take(design, "train_case_sha256", "test_case_sha256", "warmup", "samples"),
               "scope": plan["scope"]["agent_tools" if name == "agent_tools" else "other"],
               "controls": {label: _take(value, "feasible", "score", "receipt_id", "failure_type")
                            for label, value in controls.items()},
               "slots": [_take(slot, "source", "source_sha256", "slot", "context_sha256", "status", "program_sha256", "usage",
                               "finish_reason", "failure") for slot in slots],
               "winners": winners, "heldout": None}
        if outcome is not None:
            eligibility = outcome["source_eligibility"]
            eligible = [label for label in _SOURCES if winners.get(label) is not None]
            expected = "both" if len(eligible) == 2 else (eligible[0] if eligible else "neither")
            if eligibility != expected:
                raise ValueError("transfer evaluated an unfrozen winner")
            _observed_blocks(outcome["blocks"], block_ids, plan["test_schedule_by_eligible_source_set"][eligibility],
                             design["test_case_sha256"])
            analysis = outcome["analysis"]
            if set(analysis["comparisons"]) != set(_REFERENCES):
                raise ValueError("transfer report lacks a preregistered aggregate comparison")
            row["heldout"] = {"source_eligibility": eligibility, "blocks": [_block(item) for item in outcome["blocks"]],
                              "analysis": {**_take(analysis, "global_aggregate_interval_method", "aggregate_interval_level",
                                                    "statistics_plan_sha256"),
                                           "comparisons": {label: _comparison(value, margin, block_ids, outcome["blocks"], label)
                                                           for label, value in analysis["comparisons"].items()}}}
        scenario_views[name] = row
    return {"name": directory.name, "preregistration_sha256": plan_sha, "runtime_sha256": plan["runtime_sha256"],
            "driver_sha256": plan["driver_sha256"], "sources": plan["generators"], "budget": plan["budget"],
            "generation": plan["generation"], "scope": plan["scope"], "phase_seals": seals or {},
            "journal_verified": seals is not None, "scenarios": scenario_views}
