"""Seal the original six authored cases and a predetermined four-case gated subset."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from integrations.topology_memory import sglang_ingress_gate as ingress
from integrations.topology_memory.sglang_observe import save
from rsi_memory.memory_program import MemoryProgram, MemoryScenario
from rsi_memory.topology import fingerprint

from .contract import case_descriptor, checked_inputs, compile_layout, load_case, load_external, read_json


ORIGINAL_CASE_SET = "9b6ab1e3b32cbc312d26d193018d50d0574c7b364cc41d886915d122d918ea9f"
ORIGINAL_TOKENIZATION = "ce584ca3053dfc5ea99e2f39670b3939ab172375fbceadc28b93d1546e150417"
INDICES = {"train": [0], "validation": [0], "test": [0, 1]}
FORMAL_KIND = "gated_sglang_first_train_first_validation_all_test"
FORMAL_IDS = {"train": ["archive-train-reproduction"], "validation": ["archive-validation-support"],
              "test": ["archive-test-delayed-review", "archive-test-new-observation"]}


def file_sha(path):
    return sha256(path.read_bytes()).hexdigest()


def formal_subset(path):
    """Read and revalidate the preregistered indices using only case descriptors."""
    path = Path(path)
    formal = read_json(path)
    parent_path = path.with_name("case_set_manifest.json")
    parent = read_json(parent_path)
    if (formal.get("kind") != FORMAL_KIND or formal.get("preregistered_zero_based_indices") != INDICES
            or formal.get("parent_case_set_manifest_sha256") != file_sha(parent_path)
            or parent.get("prior_case_set_manifest_sha256") != ORIGINAL_CASE_SET
            or formal.get("prior_case_set_manifest_sha256") != ORIGINAL_CASE_SET):
        raise ValueError("formal SGLang cases must bind the original list and prespecified indices")
    expected = {split: [parent["cases"][split][i] for i in indices] for split, indices in INDICES.items()}
    if formal["cases"] != expected or {split: [spec["id"] for spec in rows] for split, rows in expected.items()} != FORMAL_IDS:
        raise ValueError("formal SGLang subset differs from first train, first validation, and both test cases")
    return formal


def rebind(source, tokenizer_receipt, output):
    source, tokenizer_receipt, output = map(Path, (source, tokenizer_receipt, output))
    original = source / "case_set_manifest.json"
    if file_sha(original) != ORIGINAL_CASE_SET or file_sha(tokenizer_receipt) != ORIGINAL_TOKENIZATION:
        raise ValueError("original case list or real tokenizer commitment receipt differs from the frozen inputs")
    cases = read_json(original)
    tokens = read_json(tokenizer_receipt)
    if (tokens.get("schema") != 1 or tokens.get("status") != "tokenized" or tokens.get("case_set_manifest_sha256") != ORIGINAL_CASE_SET
            or cases["cases"].keys() != INDICES.keys() or any(len(rows) != 2 for rows in cases["cases"].values())
            or set(tokens["tokens_by_case"]) != {spec["id"] for rows in cases["cases"].values() for spec in rows}):
        raise ValueError("original tokenizer commitment does not cover the six authored cases")
    output.mkdir(parents=True, exist_ok=False)
    rebound, mapping = {}, {}
    for split, specs in cases["cases"].items():
        rebound[split] = []
        for spec in specs:
            case_id = spec["id"]
            old, new = source / case_id, output / case_id
            target, topology, scenario, evidence = load_case(old, spec, fixture=True)
            external, _ = load_external(old, spec)
            token_commitments = checked_inputs(tokens["tokens_by_case"][case_id])
            gated = ingress.bind_schedule(external, token_commitments)
            rebound_scenario = MemoryScenario.from_dict({**scenario.to_dict(), "schedule_sha256": fingerprint(gated)})
            manual = MemoryProgram.from_dict(read_json(old / "manual_program.json"))
            plans = {name: compile_layout(target, topology, rebound_scenario, program)[0]
                     for name, program in (("native", None), ("manual", manual))}
            new.mkdir()
            for file in (*(old / f"{name}.json" for name in ("fixture", "target", "topology", "manual_program")), *evidence):
                dst = new / file.relative_to(old)
                dst.parent.mkdir(exist_ok=True)
                with dst.open("xb") as stream:
                    stream.write(file.read_bytes())
            manifest = {**read_json(old / "manifest.json"), "schedule_sha256": rebound_scenario.schedule_sha256,
                        "scenario_sha256": rebound_scenario.fingerprint(), "compiled_preview_sha256": {key: fingerprint(plan) for key, plan in plans.items()},
                        "prior_case_manifest_sha256": spec["manifest_sha256"], "gpu_qualification": "pending"}
            for name, data in (("manifest", manifest), ("scenario", rebound_scenario.to_dict()), ("external_schedule", gated),
                               *((f"{key}_plan_preview", plan) for key, plan in plans.items())):
                save(new / f"{name}.json", data)
            desc = case_descriptor(new)
            load_case(new, desc, fixture=True)
            load_external(new, desc, expected_inputs=token_commitments, require_gate=True)
            rebound[split].append(desc)
            mapping[case_id] = {"split": split, "prior_manifest_sha256": spec["manifest_sha256"],
                "rebound_manifest_sha256": desc["manifest_sha256"], "prior_schedule_sha256": spec["schedule_sha256"],
                "rebound_schedule_sha256": desc["schedule_sha256"], "fixture_file_sha256": file_sha(new / "fixture.json"),
                "target_file_sha256": file_sha(new / "target.json"), "input_tokenization_sha256": fingerprint(token_commitments),
                "ingress_contract_sha256": fingerprint(gated[ingress.EXTERNAL_FIELD])}
    all_path = output / "case_set_manifest.json"
    save(all_path, {**cases, "status": "pending_gpu_qualification", "prior_case_set_manifest_sha256": ORIGINAL_CASE_SET,
                    "prior_tokenization_receipt_sha256": ORIGINAL_TOKENIZATION, "cases": rebound, "case_manifest_mapping": mapping})
    formal = {split: [rebound[split][i] for i in indices] for split, indices in INDICES.items()}
    formal_path = output / "formal_case_set_manifest.json"
    save(formal_path, {"schema": 1, "kind": FORMAL_KIND, "status": "pending_gpu_qualification", "cases": formal,
                       "preregistered_zero_based_indices": INDICES, "prior_case_set_manifest_sha256": ORIGINAL_CASE_SET,
                       "parent_case_set_manifest_sha256": file_sha(all_path)})
    formal_subset(formal_path)
    for prefix, manifest_path, splits in (("", all_path, rebound), ("formal_", formal_path, formal)):
        bound = {spec["id"]: tokens["tokens_by_case"][spec["id"]] for specs in splits.values() for spec in specs}
        save(output / f"{prefix}tokenization_receipt.json", {"schema": 1, "status": "tokenized",
            "gpu_qualification": "pending", "case_set_manifest_sha256": file_sha(manifest_path),
            "prior_tokenization_receipt_sha256": ORIGINAL_TOKENIZATION, "tokens_by_case": bound})
    return {"case_set_manifest_sha256": file_sha(all_path), "formal_case_set_manifest_sha256": file_sha(formal_path),
            "formal_tokenization_receipt_sha256": file_sha(output / "formal_tokenization_receipt.json"),
            "formal_cases": {split: [spec["id"] for spec in rows] for split, rows in formal.items()},
            "status": "pending_gpu_qualification"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("source", "tokenization-receipt", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(rebind(args.source, args.tokenization_receipt, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
