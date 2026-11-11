"""Audit the settled GPU3 v4.1 test without launching a GPU worker."""

import argparse
from datetime import datetime
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
from statistics import median
import sys

from audit_training_rounds import execution_receipt, read


FINALISTS = ("manager_best", "admitted_output", "fixed")


def compare(candidate, reference, margin):
    after, before = candidate["samples"], reference["samples"]
    assert candidate["feasible"] and reference["feasible"]
    assert [s["block_id"] for s in after] == [s["block_id"] for s in before]
    gains = [b["score"] - a["score"] for a, b in zip(after, before, strict=True)]
    result = {"positive_blocks": sum(gain > 0 for gain in gains), "median_ms": median(gains),
              "all_block_deltas_ms": gains, "margin_ms": margin,
              "passes": all(gain > 0 for gain in gains) and median(gains) > margin}
    result["per_case"] = {
        case: {"all_block_deltas_ms": [
            b["case_scores_ms"][case] - a["case_scores_ms"][case] for a, b in zip(after, before, strict=True)]}
        for case in after[0]["case_scores_ms"]}
    for row in result["per_case"].values():
        row["median_ms"] = median(row["all_block_deltas_ms"])
        row["positive_blocks"] = sum(value > 0 for value in row["all_block_deltas_ms"])
    return result


def verify(run, source, config_path):
    phases = read(run / "pipeline-status.json")
    assert [(p["phase"], p["returncode"]) for p in phases] == [("run", 0), ("freeze", 0), ("test", 0)]
    sys.path.insert(0, str(source.resolve(strict=True)))
    from rsi_memory.topology import fingerprint

    path = (run / "controller/journal.sqlite").resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2) as db:
        db.execute("PRAGMA query_only=ON")

        def record(key):
            ordinal, raw = db.execute("SELECT rowid,data FROM records WHERE key=?", (key,)).fetchone()
            return ordinal, json.loads(raw)

        _, manifest = record("manifest")
        rows = [json.loads(raw) for raw, in db.execute("SELECT data FROM records WHERE key LIKE 'round/r%' ORDER BY key")]
        freeze_id, frozen = record("frozen")
        test_id, test = record("pair/frozen/test")
        _, saved_comparisons = record("test_comparisons")
        _, saved_relations = record("test_program_relations")
    assert freeze_id < test_id
    assert frozen == read(run / "controller/frozen.json")
    assert len(rows) == frozen["rounds"] == 2 and frozen["partial"] is False
    assert frozen["manifest_sha256"] == fingerprint(manifest)
    assert frozen["rounds_sha256"] == fingerprint(rows)
    last = rows[-1]
    assert frozen["training_matrix_sha256"] == fingerprint(last["v4_training"])
    assert frozen["active_harness_sha256"] == last["active_after"]
    admitted_rows = [row for row in rows if row["decision"] == "admitted"]
    admitted = {manifest["seed_sha256"], *(row["active_after"] for row in admitted_rows)}
    reused = {row["previous_admission_reuse"]["source_sha256"] for row in rows
              if row.get("previous_admission_reuse", {}).get("recursive_behavior_observed")}
    assert frozen["admitted_harness_sha256s"] == sorted(admitted)
    assert frozen["reused_harness_sha256s"] == sorted(reused)

    config = read(config_path)
    paired_programs = {"native": None, "manual": config["manual_program"]}
    choices = {}
    for label in FINALISTS:
        final = frozen[label]
        choice = last["v4_selection"][label]
        assert final["selection"] == choice
        assert final["trial"] == last["v4_training"]["cohort_proposals"][choice["proposal_id"]]
        assert final["training_score"] == last["v4_training"]["measurements"][choice["effect_sha256"]]["median_ms"]
        program = final["trial"]["program"]
        assert fingerprint(program) == choice["effect_sha256"] == final["trial"]["program_sha256"]
        paired_programs[label] = program
        choices[label] = {"proposal_id": choice["proposal_id"], "source_sha256": choice["source_sha256"],
                          "program_sha256": choice["effect_sha256"], "training_median_ms": final["training_score"],
                          "from_admitted_generator": choice["source_sha256"] in admitted}
    assert choices["admitted_output"]["from_admitted_generator"]
    assert choices["fixed"]["source_sha256"] == manifest["seed_sha256"]
    assert test["status"] == "completed"
    assert test["input"]["split"] == "test" and test["input"]["programs"] == paired_programs
    expected_blocks = manifest["contract"]["paired_block_ids"]["test"]
    assert test["input"]["block_ids"] == expected_blocks and len(expected_blocks) == 6
    expected_cases = manifest["contract"]["split_case_sha256"]["test"]
    assert len(expected_cases) == 4
    receipt_dir = (config_path.parent / config["output_directory"]).resolve() / "receipts"
    signatures, receipts, processes, proof_by_arm, wall_jobs = {}, {}, set(), {}, []
    assert test["result"].keys() == paired_programs.keys()
    orders = test["result"]["native"]["metrics"]["execution_order"]
    assert [row["block_id"] for row in orders] == expected_blocks
    for label, row in test["result"].items():
        assert row["feasible"] and row["metrics"]["execution_order"] == orders
        assert [sample["block_id"] for sample in row["samples"]] == expected_blocks
        assert median(sample["score"] for sample in row["samples"]) == row["score"]
        proof_by_arm[label] = []
        for sample, job in zip(row["samples"], row["metrics"]["receipts"], strict=True):
            assert sample["receipt_id"] == job
            result = {"metrics": {"receipts": [job]}, "score": sample["score"]}
            _, digest, case_signatures = execution_receipt(receipt_dir, result, config, expected_cases, fingerprint)
            for case, selected in case_signatures.items():
                assert signatures.setdefault(case, selected) == selected
            receipt = read(receipt_dir / (job + ".json"))
            assert receipt["gpu_indices"] == config["executor"]["gpu_indices"] == [3]
            assert receipt["gpu_uuids"] == config["executor"]["gpu_uuids"]
            assert job not in receipts
            receipts[job] = digest
            pid_start = (receipt["pid"], receipt["proc_starttime"])
            assert pid_start not in processes
            processes.add(pid_start)
            start, end = (datetime.fromisoformat(receipt[k]) for k in ("utc_start", "utc_end"))
            assert start < end
            wall_jobs.append((start, end, sample["block_id"], label))
            cases = receipt["result"]["cases"]
            case_times = {case["name"]: case["median_elapsed_ns"] / 1e6 for case in cases}
            assert case_times == sample["case_scores_ms"]
            program_sha = fingerprint(paired_programs[label]) if paired_programs[label] is not None else None
            for case in cases:
                assert case["diagnostic"]["program_sha256"] == program_sha
                assert len(case["timings"]) == config["samples"]
                assert median(timing["elapsed_ns"] for timing in case["timings"]) == case["median_elapsed_ns"]
                assert all(t["execution_signature"] == case["execution_signature"] for t in case["timings"])
            proof_by_arm[label].append({"block_id": sample["block_id"], "score_ms": sample["score"],
                                       "case_scores_ms": case_times, "receipt_id": job})
    ordered = sorted(wall_jobs)
    assert all(a[1] <= b[0] for a, b in zip(ordered, ordered[1:]))
    assert [(block, label) for _, _, block, label in ordered] == [
        (entry["block_id"], label) for entry in orders for label in entry["order"]]
    assert len(receipts) == 30
    margin = config["controller_preregistration"]["min_improvement_ms"]
    arms = test["result"]
    comparisons = {}
    for candidate in FINALISTS:
        for reference in ("native", "manual", "fixed"):
            if candidate == reference:
                continue
            name = f"{candidate}_vs_{reference}"
            result = compare(arms[candidate], arms[reference], margin)
            comparisons[name] = result
            if name in saved_comparisons:
                saved = saved_comparisons[name]
                assert saved["passed"] == result["passes"] and saved["median_improvement"] == result["median_ms"]
                assert [s["reference_minus_candidate"] for s in saved["deltas"]] == result["all_block_deltas_ms"]
                raw_alias = reference != "native" and fingerprint(paired_programs[candidate]) == fingerprint(paired_programs[reference])
                assert saved_relations[name] == {"same_raw_program": raw_alias, "same_training_effect": raw_alias}
    return {"schema": 1, "kind": "cuda_v41_gpu3_independent_frozen_test",
            "scope": "Single GPU3 engineering replay on four previously public workloads in each split. Receipt digests attest to selected bytes and attention outputs; no raw KV replay.",
            "frozen_sha256": sha256((run / "controller/frozen.json").read_bytes()).hexdigest(),
            "controller_manifest_sha256": sha256((source / "SOURCE_MANIFEST.json").read_bytes()).hexdigest(),
            "worker_manifest_sha256": config["source_manifest_sha256"], "frozen_finalists": choices,
            "admissions": len(admitted_rows), "observed_recursive_source_reuses": len(reused),
            "blocks": expected_blocks, "receipt_sha256_by_job": receipts, "selected_output_digests": signatures,
            "observations_by_arm": proof_by_arm, "pairwise_comparisons": comparisons,
            "declared_program_relations": saved_relations,
            "score_unit": "sum of four case median elapsed milliseconds; lower is better"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source", "config", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.run, args.source, args.config)
    with args.output.open("x", encoding="utf-8", newline="\n") as f:
        json.dump(result, f, sort_keys=True, indent=2, allow_nan=False)
        f.write("\n")
    print(sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
