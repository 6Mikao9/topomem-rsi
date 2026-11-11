"""Read the independent post-freeze SGLang native-6144 comparison."""

from hashlib import sha256
import json
import math
import re
from statistics import median

from .topology_report_sglang import _observed, _sha


FORMAT = "sglang_post_freeze_native6144_v1"
ADAPTIVE, NATIVE = "frozen_adaptive", "native6144_all_host"
LABELS = (ADAPTIVE, NATIVE)
BLOCKS = [{"block_id": f"n6144-b{index:02d}", "order": list(LABELS if index % 2 else LABELS[::-1])}
          for index in range(1, 9)]
MATCH = ("effective_schedule_sha256", "dispatch_sha256", "output_ids_sha256", "runtime_sha256",
         "global_ticks_sha256", "input_tokenization_sha256")
JOB = re.compile(r"sg-[a-f0-9]{48}")


def _require(condition):
    if not condition:
        raise ValueError("ablation evidence did not match its sealed study")


def _json(path):
    raw = path.read_bytes()
    return json.loads(raw), sha256(raw).hexdigest()


def _match(left, right, *, parity=False):
    a, b = left["metrics"], right["metrics"]
    _require(all(_sha(a.get(key)) and a[key] == b.get(key) for key in MATCH))
    for key in ("external_scheduler_sha256", "ingress_contract_sha256"):
        value = left["memory_compilation"].get(key)
        _require(_sha(value) and value == right["memory_compilation"].get(key))
    if parity:
        packed = a.get("packed_kv_sha256_by_rank", [])
        _require(len(packed) == 2 and all(_sha(value) for value in packed) and packed == b.get("packed_kv_sha256_by_rank"))


def _receipt(directory, receipt_reader, index, job, case, spec, program, label, phase):
    _require(isinstance(job, str) and JOB.fullmatch(job))
    if job not in index:
        path = directory / "receipts" / f"{job}.json"
        _require(path.is_file() and not path.is_symlink())
        index[job] = receipt_reader(path, 0)
    value = index[job]
    native = label == NATIVE
    _require(value["job_id"] == job)
    configuration = _observed(value, case, spec["worker_source_manifest_sha256"], spec["metadata_profile_sha256"],
                              None if native else program, "native" if native else "generated", phase)
    _require(configuration is not None)
    result = value["result"]
    serving = result["serving"]
    if native:
        _require(configuration["hot_slots"] == serving["memory_compilation"]["startup"].get("native_reference_override") == 6144
                 and configuration["backing_mode"] == "tiered"
                 and all(rank["layer_tiers"] == ["host"] * 6 and rank["pinned_host_bytes"] > 0
                         and rank["local_hbm_bytes"] == rank["peer_hbm_bytes"] == 0 for rank in configuration["ranks"]))
        if phase == "parity":
            _require(serving["cross_capacity_native_parity"].get("matched") is True)
    if phase == "performance":
        score = result.get("score")
        seconds = serving["metrics"].get("timing", {}).get("window_seconds")
        _require(type(score) in (int, float) and math.isfinite(score) and score > 0
                 and type(seconds) in (int, float) and seconds * 1000 == score
                 and serving["metrics"].get("packed_kv_sha256_by_rank") == []
                 and all(rank.get("resident_producer_oracle") is None and rank.get("selected_miss_payload_bytes_by_tier") is None
                         for rank in serving["metrics"]["ranks"]))
    return result, serving, configuration


def _complete(directory, result, spec, prereg_sha, anchors, receipt_reader, fingerprint):
    manifest, manifest_sha = _json(directory / "manifest.json")
    bound = manifest["bound"]
    _require(manifest.get("schema") == spec.get("schema") == 1 and manifest.get("kind") == spec.get("kind") == FORMAT
             and bound.get("preregistration_sha256") == result.get("preregistration_sha256") == prereg_sha
             and manifest.get("planned_jobs") == spec.get("timed_job_count") == result.get("timed_jobs_completed") == 32
             and spec.get("blocks") == BLOCKS
             and spec.get("native") == {"arm": "native", "native_hot": 6144, "program": None, "backing": "all_host"}
             and spec.get("adaptive") == {"arm": "generated", "program_source": "frozen.adaptive.trial.program"}
             and set(bound.get("adapter_config_field_diff", {})) == {"receipt_directory"}
             and _sha(spec.get("driver_sha256")))
    driver = directory / "sglang_post_freeze_native6144.py"
    if driver.is_file():
        _require(sha256(driver.read_bytes()).hexdigest() == spec["driver_sha256"])
    keys = ("formal_report_sha256", "formal_frozen_sha256", "formal_manifest_canonical_sha256", "formal_pipeline_status_sha256",
            "adaptive_program_sha256", "adaptive_trial_id", "adaptive_harness_sha256")
    matches = [anchor for anchor in anchors if all(bound.get(key) == anchor.get(key) for key in keys)]
    _require(len(matches) == 1)
    anchor = matches[0]
    for key in ("cases", "worker_source_manifest_sha256", "metadata_profile_sha256", "launch_sha256", "tokenization_sha256",
                "margin_hand_sweep_canonical_sha256", "controller_source_archive_sha256", "controller_source_manifest_sha256"):
        _require(spec.get(key) == anchor.get(key) and (len(spec[key]) == 2 if key == "cases" else _sha(spec[key])))
    margin = anchor["margin_ms"]
    _require(type(margin) in (int, float) and margin > 0 and math.isfinite(margin)
             and spec.get("decision") == {"positive_in_every_block": True, "median_strictly_greater_than_ms": margin}
             and result.get("practical_margin_ms") == margin)
    program, _ = _json(directory / "frozen_adaptive.program.json")
    program_sha = fingerprint(program)
    _require(program_sha == bound.get("adaptive_program_sha256"))

    rows = [json.loads(line) for line in (directory / "jobs.jsonl").read_bytes().splitlines()]
    cases = spec["cases"]
    _require(len(rows) == 34 and len({case["id"] for case in cases}) == 2 and len(result["complete_blocks"]) == 8)
    index, qualified, proof_by_case, startup = {}, {}, {}, {}
    for case, row in zip(cases, rows[:2], strict=True):
        key = case["id"]
        _require(row.get("phase") == "qualification" and row.get("case") == key and set(row["proofs"]) == set(LABELS))
        proof_by_case[key] = row["proofs"]
        qualified[key], startup[key] = {}, {}
        for label in LABELS:
            proof = row["proofs"][label]
            _, serving, layout = _receipt(directory, receipt_reader, index, proof["job_id"], case, spec, program_sha, label, "parity")
            _require(proof.get("raw_result_sha256") == serving["raw_result_sha256"])
            qualified[key][label] = serving, layout
            startup[key][label] = {field: layout[field] for field in ("hot_slots", "backing_mode")}
        _, repeat, repeat_layout = _receipt(directory, receipt_reader, index, row["native_same_capacity_repeat_job_id"],
                                             case, spec, program_sha, NATIVE, "parity")
        _require(row["native_same_capacity_repeat_job_id"] != row["proofs"][NATIVE]["job_id"] and repeat_layout == qualified[key][NATIVE][1])
        _match(repeat, qualified[key][NATIVE][0], parity=True)
        _match(qualified[key][ADAPTIVE][0], qualified[key][NATIVE][0], parity=True)
        _require([r["gpu_uuid"] for r in qualified[key][ADAPTIVE][1]["ranks"]] == [r["gpu_uuid"] for r in repeat_layout["ranks"]])
        startup[key]["same_hot_capacity"] = startup[key][ADAPTIVE]["hot_slots"] == 6144
    _require(result.get("startup_by_case") == startup)

    timed_rows, timed_ids, blocks = iter(rows[2:]), set(), []
    for planned, recorded in zip(BLOCKS, result["complete_blocks"], strict=True):
        _require(all(recorded.get(key) == value for key, value in planned.items()))
        windows, ids = ({label: {} for label in LABELS} for _ in range(2))
        for case in cases:
            key = case["id"]
            observations = {}
            for label in planned["order"]:
                row = next(timed_rows)
                job = row["job_id"]
                _require(row.get("phase") == "performance" and job not in timed_ids and row.get("qualification") == proof_by_case[key][label]
                         and row.get("identity") == {"kind": FORMAT, "preregistration_sha256": prereg_sha,
                              "driver_sha256": spec["driver_sha256"], "formal_frozen_sha256": bound["formal_frozen_sha256"],
                              "block_id": planned["block_id"], "case_id": key, "label": label})
                value, serving, layout = _receipt(directory, receipt_reader, index, job, case, spec, program_sha, label, "performance")
                proof, expected = qualified[key][label]
                _require(layout == expected and serving["raw_result_sha256"] == row.get("raw_result_sha256") and value["score"] == row.get("window_ms"))
                _match(serving, proof)
                observations[label], windows[label][key], ids[label][key] = serving, value["score"], job
                timed_ids.add(job)
            _match(observations[ADAPTIVE], observations[NATIVE])
        totals = {label: sum(windows[label].values()) for label in LABELS}
        gain = totals[NATIVE] - totals[ADAPTIVE]
        _require(recorded.get("window_ms_by_arm_case") == windows and recorded.get("job_id_by_arm_case") == ids
                 and recorded.get("window_ms_sum_by_arm") == totals and recorded.get("native_minus_adaptive_ms") == gain)
        blocks.append({**planned, "adaptive_ms": totals[ADAPTIVE], "native_ms": totals[NATIVE], "gain_ms": gain})
    gains = [row["gain_ms"] for row in blocks]
    wins, middle = sum(gain > 0 for gain in gains), median(gains)
    passed = wins == len(BLOCKS) and middle > margin
    _require(result.get("positive_blocks") == wins and result.get("median_ms") == middle
             and result.get("performance_gate") == ("passed" if passed else "failed"))
    return {"status": "complete", "case_count": len(cases), "same_hot_capacity": all(row["same_hot_capacity"] for row in startup.values()),
            "adaptive_hot_slots": sorted({row[ADAPTIVE]["hot_slots"] for row in startup.values()}), "blocks": blocks,
            "wins": wins, "median_gain_ms": middle, "passed": passed, "margin_ms": margin,
            "provenance": {**{key: bound[key] for key in keys}, "preregistration_sha256": prereg_sha, "manifest_sha256": manifest_sha,
                           "jobs_sha256": sha256((directory / "jobs.jsonl").read_bytes()).hexdigest(), "startup_by_case": startup}}


def read_ablation(directory, anchors, receipt_reader, fingerprint):
    if not (directory / "result.json").is_file():
        return {"status": "pending", "completed_blocks": 0}
    try:
        result, _ = _json(directory / "result.json")
        spec, prereg_sha = _json(directory / "preregistration.json")
        _require(spec.get("schema") == 1 and spec.get("kind") == FORMAT and spec.get("blocks") == BLOCKS
                 and result.get("preregistration_sha256") == prereg_sha)
        state = result.get("status")
        if state == "completed":
            return _complete(directory, result, spec, prereg_sha, anchors, receipt_reader, fingerprint)
        reasons = {"skipped_budget": "超过预登记启动时限", "skipped_formal_pipeline_failed": "正式流程未成功完成",
                   "budget_incomplete": "达到提交时限", "execution_incomplete": "执行未完成"}
        _require(state in reasons and result.get("performance_gate") == "not_evaluated")
        records = result.get("complete_blocks", [])
        _require(isinstance(records, list) and len(records) < 8
                 and all(all(row.get(key) == value for key, value in expected.items()) for row, expected in zip(records, BLOCKS)))
        skipped = state.startswith("skipped")
        _require(not skipped or (not records and result.get("timed_jobs_completed") == 0))
        return {"status": "skipped" if skipped else "partial", "completed_blocks": len(records), "reason": reasons[state]}
    except (KeyError, TypeError, ValueError, AttributeError, OSError):
        return {"status": "unverified", "completed_blocks": None}
