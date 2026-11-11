"""Read-only audit of the v4.1 CUDA R0 incomplete training matrix."""

import argparse
from collections import Counter
from hashlib import sha256
import json
from pathlib import Path
import sqlite3
import sys

from audit_training_rounds import execution_receipt, harness, read


def verify(run, source, config_path):
    sys.path.insert(0, str(source.resolve(strict=True)))
    from rsi_memory.topology import fingerprint

    source_manifest = source / "SOURCE_MANIFEST.json"
    for name, digest in read(source_manifest)["files"].items():
        assert sha256((source / name).read_bytes()).hexdigest() == digest, name
    config = read(config_path)
    path = (run / "controller/journal.sqlite").resolve()
    with sqlite3.connect(path.as_uri() + "?mode=ro", uri=True, timeout=2) as db:
        db.execute("PRAGMA query_only=ON")

        def record(key):
            row = db.execute("SELECT data FROM records WHERE key=?", (key,)).fetchone()
            return json.loads(row[0]) if row else None

        manifest = record("manifest")
        protocol = manifest["contract"]["selection_protocol"]
        contract = protocol["trusted_selection_contract"]
        assert contract["version"] == "4.1"
        for name, digest in protocol["source_sha256"].items():
            assert sha256((source / name).read_bytes()).hexdigest() == digest
        api_proof = {}
        for key, status, source_sha, system, sent in db.execute(
            "SELECT key,json_extract(data,'$.status'),json_extract(data,'$.input.harness_sha256'), "
            "json_extract(data,'$.input.payload.messages[0].content'), "
            "json_extract(json_extract(data,'$.input.payload.messages[1].content'),'$.trusted_selection_contract') "
            "FROM records WHERE key LIKE 'api/r0000/%' ORDER BY key"
        ):
            assert status == "completed"
            assert json.loads(sent) == contract
            assert "Both sides may retain" not in system
            assert ("separate paired validation reference" in system) == key.endswith("-meta")
            harness(run, source_sha)
            api_proof[key] = {"status": status, "source_sha256": source_sha,
                              "system_sha256": sha256(system.encode()).hexdigest(), "contract_matches": True}
        assert len(api_proof) == 10 and sum(key.endswith("-meta") for key in api_proof) == 2
        assert record("round/r0000") is None
        assert record("pair/r0000/validation") is None
        assert record("frozen") is None
        matrix = record("training_matrix/r0000/result")
        plan = matrix["plan"]
        assert plan == record("training_matrix/r0000/plan")
        scope = plan["scope_sha256"]
        assert scope == fingerprint({"contract": manifest["contract"],
                                     "plan": {name: item for name, item in plan.items() if name != "scope_sha256"}})
        native = record("eval/native/train")
        assert native["status"] == "completed"
        receipt_dir = (config_path.parent / config["output_directory"]).resolve() / "receipts"
        expected_cases = manifest["contract"]["split_case_sha256"]["train"]
        native_job, native_sha, native_sig = execution_receipt(
            receipt_dir, native["result"], config, expected_cases, fingerprint)
        cohort = matrix["cohort_proposals"]
        for trial_id, trial in cohort.items():
            assert trial_id == trial["id"]
            assert trial["harness_sha256"] == api_proof["api/" + trial_id]["source_sha256"]
            assert trial["training_effect_sha256"] == trial["program_sha256"] == fingerprint(trial["program"])
        observations, verified_receipts, failures = [], {}, []
        for effect, result in matrix["measurements"].items():
            rep = plan["representatives"][effect]
            trial_id = rep["trial_id"]
            program = manifest["initial_program"] if trial_id == "bootstrap/initial-incumbent" else cohort[trial_id]["program"]
            assert fingerprint(program) == effect == rep["program_sha256"]
            assert fingerprint({"scope_sha256": scope, "effect_sha256": effect,
                                "evaluations": result["evaluations"]}) == result["qualification_record_sha256"]
            for item in result["evaluations"]:
                key = item["evaluation_key"]
                evaluation = record(key)
                assert evaluation["input"] == {"split": "train", "program": program}
                assert evaluation["status"] == item["status"]
                assert fingerprint({"evaluation_key": key.removeprefix("eval/"), "record": evaluation}) == item["evaluation_sha256"]
                if "window_ms" in item:
                    value = evaluation["result"]
                    assert value["feasible"] is True and value["score"] == item["window_ms"]
                    assert value["metrics"]["receipts"] == item["reported_receipts"]
                    assert fingerprint(item["reported_receipts"]) == item["receipt_bundle_sha256"]
                    job, digest, case_sig = execution_receipt(receipt_dir, value, config, expected_cases, fingerprint)
                    assert job not in verified_receipts and case_sig == native_sig
                    verified_receipts[job] = digest
                elif evaluation["status"] == "completed":
                    assert evaluation["result"]["feasible"] is False
                    failures.append({"trial": trial_id, "block_id": item["block_id"],
                                     "kind": "correctness_or_preflight_infeasible",
                                     "reported_receipts": item["reported_receipts"]})
                else:
                    failures.append({"trial": trial_id, "block_id": item["block_id"], "kind": evaluation["status"],
                                     "reported_error_type": evaluation.get("failure"), "elapsed_seconds": evaluation["seconds"]})
                observations.append((item["block_id"], effect))
        for index, block in enumerate(plan["block_ids"]):
            assert set(effect for observed, effect in observations if observed == block) <= set(plan["order"][index])
        assert len(verified_receipts) == 13
        assert Counter(item["kind"] for item in failures) == {"failed": 1, "correctness_or_preflight_infeasible": 2}
        assert len(observations) == 16 and len({effect for _, effect in observations}) == 9
        assert not any(block == plan["block_ids"][-1] for block, _ in observations)
        states = Counter(row["status"] for row in matrix["measurements"].values())
        assert states == {"incomplete": 7, "correctness_failed": 2}
    return {"schema": 1, "kind": "v41_cuda_incomplete_r0_audit", "status": "training_matrix_incomplete",
            "scope": "Four training cases, completed R0 call payload contract projections and matrix receipts. GPU receipts retain selected-and-output hashes, not raw KV payloads.",
            "controller_manifest_sha256": sha256(source_manifest.read_bytes()).hexdigest(),
            "worker_manifest_sha256": config["source_manifest_sha256"], "training_scope_sha256": scope,
            "provider_api_proof": api_proof, "exact_candidate_identities": len(plan["representatives"]),
            "planned_block_ids": plan["block_ids"], "observations_by_block": dict(Counter(block for block, _ in observations)),
            "declared_identity_status_counts": dict(states), "candidate_or_execution_failures": failures,
            "native_training_receipt": {"job_id": native_job, "receipt_sha256": native_sha, "case_signatures": native_sig},
            "timing_receipt_sha256_by_job": verified_receipts, "round_committed": False, "validation_submitted": False,
            "selection_frozen": False, "source_admission_observed": False}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("run", "source", "config", "output"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    result = verify(args.run, args.source, args.config)
    with args.output.open("x", encoding="utf-8", newline="\n") as out:
        json.dump(result, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")
    print(sha256(args.output.read_bytes()).hexdigest())


if __name__ == "__main__":
    main()
