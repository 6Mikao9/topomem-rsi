"""Freeze separate authored SGLang request cases and verify their real token lengths."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from integrations.topology_memory import generate_dp2_case as dp2
from integrations.topology_memory.generate_serving_case import build_case
from integrations.topology_memory.sglang_observe import save
from rsi_memory.topology import fingerprint

from .contract import case_descriptor, checked_id, checked_inputs, load_case, read_json


SPLITS = ("train", "validation", "test")


def _fixture(context, case):
    case_id = checked_id(case["id"])
    if (case.keys() != {"id", "packet", "requests"} or type(case["packet"]) is not str or not case["packet"].strip()
            or type(case["requests"]) is not list or len(case["requests"]) != 2
            or any(type(question) is not str or not question.strip() for question in case["requests"])):
        raise ValueError("each authored serving case requires a distinct packet and two fixed requests")
    prompt = f"{context}\n\nIndependent agent work packet\n{case['packet'].strip()}"
    questions = {"warmup": ["List the two strongest pieces of completed evidence relevant to this work packet.",
                             "State the handoff boundary in this work packet. Distinguish evidence from a future action."],
                 "requests": case["requests"]}
    return {"schema": 1, "name": case_id, **{
        group: [{"rid": f"{case_id}-{group}-r{rank}", "arrival_ms": 0, "max_new_tokens": dp2.NEW_TOKENS,
                 "routed_dp_rank": rank, "prompt": f"{prompt}\n\nRequest\n{question}"}
                for rank, question in enumerate(rows)] for group, rows in questions.items()}}


def build(output, *, profile, briefing, appendix, packets, config_sha256, weights_sha256):
    inputs = {key: Path(path).read_bytes() for key, path in {"profile": profile, "briefing": briefing,
                                                           "appendix": appendix, "packets": packets}.items()}
    cases = json.loads(inputs["packets"])
    if (type(cases) is not dict or cases.keys() != {"schema", "shared", "splits"} or cases["schema"] != 1
            or type(cases["shared"]) is not str or not cases["shared"].strip()
            or type(cases["splits"]) is not dict or cases["splits"].keys() != set(SPLITS)
            or any(type(rows) is not list or len(rows) != 2 for rows in cases["splits"].values())):
        raise ValueError("expected two authored case packets per serving split")
    source = dp2.make_fixture(json.loads(inputs["briefing"]), inputs["appendix"].decode("utf-8"))
    base = source["warmup"][0]["prompt"].rsplit("\n\nRequest\n", 1)[0]
    context = f"{base}\n\n{cases['shared'].strip()}"
    fixtures = {split: [_fixture(context, row) for row in cases["splits"][split]] for split in SPLITS}
    if len({fixture["name"] for rows in fixtures.values() for fixture in rows}) != 6:
        raise ValueError("all serving case IDs must be unique")
    prompts = [row["prompt"] for rows in fixtures.values() for fixture in rows for group in ("warmup", "requests") for row in fixture[group]]
    if len(set(prompts)) != len(prompts):
        raise ValueError("authored serving case prompts must be independent replays")
    topology, bindings = dp2.measured_topology(inputs["profile"])
    manual, _ = dp2._controls()
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    splits = {}
    for split, rows in fixtures.items():
        descriptors = []
        for fixture in rows:
            case = output / fixture["name"]
            build_case(case, topology=topology, gpu_bindings=bindings, fixture=fixture,
                       model_config_sha256=config_sha256, model_weights_sha256=weights_sha256,
                       pool=dp2.POOL, native_host=dp2.HOST, compute_nodes=dp2.COMPUTE, allowed_nodes=dp2.ALLOWED,
                       selector="torch-unfused", top_k=dp2.TOP_K, link_evidence=[profile], manual=manual, attention_dp_size=2)
            descriptors.append(case_descriptor(case))
        splits[split] = descriptors
    report = {"schema": 1, "status": "unverified", "cases": splits, "source_sha256": {key: sha256(raw).hexdigest() for key, raw in inputs.items()},
              "scope": "independent real model requests in an authored archive agent work-packet family; shared public background; no tool execution or retained session KV"}
    save(output / "case_set_manifest.json", report)
    return report


def verify_tokens(cases, tokenizer, output):
    cases = Path(cases)
    report = read_json(cases / "case_set_manifest.json")
    tokens_by_case = {}
    for specs in report["cases"].values():
        for spec in specs:
            path = cases / spec["id"]
            load_case(path, spec, fixture=True)
            fixture = read_json(path / "fixture.json")
            observed = {}
            for group in ("warmup", "requests"):
                for row in fixture[group]:
                    ids = tokenizer.encode(row["prompt"], add_special_tokens=False)
                    if row["max_new_tokens"] != dp2.NEW_TOKENS:
                        raise ValueError("serving tokenization requires the fixed 64-token decode")
                    observed[row["rid"]] = {"input_tokens": len(ids), "input_ids_sha256": fingerprint(ids)}
            tokens_by_case[spec["id"]] = checked_inputs(observed)
    save(output, {"schema": 1, "status": "tokenized", "case_set_manifest_sha256": sha256((cases / "case_set_manifest.json").read_bytes()).hexdigest(),
                  "tokens_by_case": tokens_by_case})


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    make = commands.add_parser("build")
    for key in ("output", "profile", "briefing", "appendix", "packets"):
        make.add_argument(f"--{key}", type=Path, required=True)
    make.add_argument("--model-config-sha256", required=True)
    make.add_argument("--model-weights-sha256", required=True)
    verify = commands.add_parser("verify-tokens")
    for key in ("cases", "model", "output"):
        verify.add_argument(f"--{key}", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "build":
        result = build(args.output, profile=args.profile, briefing=args.briefing, appendix=args.appendix, packets=args.packets,
                       config_sha256=args.model_config_sha256, weights_sha256=args.model_weights_sha256)
        print(json.dumps(result, sort_keys=True))
    else:
        from transformers import AutoTokenizer

        verify_tokens(args.cases, AutoTokenizer.from_pretrained(args.model, local_files_only=True), args.output)


if __name__ == "__main__":
    main()
