"""Run the v4 controller on the existing CUDA block-sparse domain."""

import argparse
from dataclasses import fields
import json
from pathlib import Path

from .cuda_domain import CudaDomain
from .topology import fingerprint
from .topology_rsi import Domain, _no_api, load_api_key, make_requester
from .topology_rsi_contract import ContractDomain
from .topology_rsi_v4 import Config, TopologyRSIV4


_COUNTS = ("rounds", "slots", "training_blocks", "validation_blocks", "test_blocks", "max_output_tokens")
_TIMES = ("training_margin_ms", "min_improvement_ms", "wall_seconds")


def checked_preregistration(raw, block_ids):
    if type(raw) is not dict or set(raw) != {*_COUNTS, *_TIMES, "model"}:
        raise ValueError("v4 controller preregistration fields differ")
    if any(type(raw[name]) is not int or raw[name] <= 0 for name in _COUNTS):
        raise ValueError("preregistered counts must be positive integers")
    if any(type(raw[name]) not in (int, float) for name in _TIMES):
        raise ValueError("preregistered time and margin values must be numeric")
    if type(raw["model"]) is not str or not raw["model"].strip():
        raise ValueError("invalid preregistered model")
    if any(type(block_ids[split]) is not list or len(block_ids[split]) != raw[f"{split}_blocks"]
           for split in ("validation", "test")):
        raise ValueError("paired adapter blocks differ from the controller preregistration")
    config = Config(rounds=raw["rounds"], slots=raw["slots"], max_output_tokens=raw["max_output_tokens"],
                    min_improvement=raw["min_improvement_ms"], wall_seconds=raw["wall_seconds"], model=raw["model"],
                    min_paired_blocks=raw["validation_blocks"], training_blocks=raw["training_blocks"],
                    training_margin_ms=raw["training_margin_ms"])
    config.validate()
    return config


def exact_program_identity(program):
    return program.fingerprint()


def contract_domain(domain, preregistration):
    values = {field.name: getattr(domain, field.name) for field in fields(Domain)}
    values["contract"] = {**domain.contract, "controller_preregistration": preregistration,
                          "controller_preregistration_sha256": fingerprint(preregistration),
                          "training_effect_identity": "exact raw program SHA-256; dynamic physical equivalence is unproved"}
    values["effective_program_fingerprint"] = exact_program_identity
    return ContractDomain(**values)


def _read(path):
    return json.loads(path.read_bytes())


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "freeze", "test"))
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--preregistration", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--round-limit", type=int)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    path = args.adapter_config.resolve(strict=True)
    raw = _read(path)
    if type(raw) is not dict:
        parser.error("adapter configuration must be an object")
    embedded = raw.get("controller_preregistration")
    if args.preregistration is not None and embedded is not None:
        parser.error("provide the controller preregistration once")
    preregistration = _read(args.preregistration) if args.preregistration is not None else embedded
    config = checked_preregistration(preregistration, raw["paired_block_ids"])
    adapter = CudaDomain(raw, path.parent)
    try:
        request = make_requester(load_api_key(args.key_file)) if args.command == "run" else _no_api
        controller = TopologyRSIV4(args.output, config, contract_domain(adapter.domain(), preregistration),
                                   request, backend_source=Path(__file__))
        try:
            if args.command == "run":
                summary = {"status": controller.run(args.round_limit)["status"]}
            elif args.command == "freeze":
                summary = {"status": "frozen", "rounds": controller.freeze(allow_partial=args.allow_partial)["rounds"]}
            else:
                summary = {"status": controller.test()["status"]}
        finally:
            controller.close()
    finally:
        adapter.close()
    print(json.dumps(summary, sort_keys=True))
    return int(args.command == "test" and summary["status"] != "test_settled")


if __name__ == "__main__":
    raise SystemExit(main())
