"""Launch the frozen SGLang judge with a parser-consistent generation contract."""

import argparse
from dataclasses import fields
import json
from pathlib import Path

from rsi_memory import topology_rsi
from rsi_memory.topology_rsi_contract import ContractDomain, ContractTopologyRSI, ProgramFamily

from .contract import read_json
from .domain import SGLangDomain


FAMILIES = (
    ProgramFamily("tiered", (("allocate", "place"),), (),
                  "Use the ordinary cold backing path. If configure/backing_mode is present, it must select tiered."),
    ProgramFamily("resident", (("configure", "hot_capacity"), ("configure", "backing_mode")),
                  (("allocate", "place"),),
                  "configure/hot_capacity must select the qualified 6144-slot setting and configure/backing_mode "
                  "must select resident. The evaluator checks every committed prompt, decode bound and physical page budget. "
                  "No allocate/place rule is permitted anywhere in a resident program."),
)


def contract_domain(domain):
    args = {field.name: getattr(domain, field.name) for field in fields(topology_rsi.Domain)}
    return ContractDomain(**args, program_families=FAMILIES)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "freeze", "test"))
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--round-limit", type=int)
    args = parser.parse_args()
    path = args.adapter_config.resolve(strict=True)
    config = read_json(path)
    if type(config) is not dict or config.get("schema") != 3:
        parser.error("require a schema-3 adapter with a verified hand control")
    adapter = SGLangDomain(config, path.parent)
    try:
        budget = adapter.contract["controller_preregistration"]
        controller_config = topology_rsi.Config(
            rounds=budget["rounds"], slots=budget["slots"], max_output_tokens=budget["max_output_tokens"],
            min_improvement=budget["min_improvement_ms"], wall_seconds=budget["wall_seconds"],
            min_paired_blocks=budget["minimum_paired_blocks"],
        )
        request = (topology_rsi.make_requester(topology_rsi.load_api_key(args.key_file))
                   if args.command == "run" else topology_rsi._no_api)
        controller = ContractTopologyRSI(args.output, controller_config, contract_domain(adapter.domain()), request,
                                         backend_source=Path(__file__))
        try:
            if args.command == "run":
                summary = {"status": controller.run(args.round_limit)["status"]}
            elif args.command == "freeze":
                summary = {"status": "frozen", "rounds": controller.freeze()["rounds"]}
            else:
                summary = {"status": controller.test()["status"]}
        finally:
            controller.close()
    finally:
        adapter.close()
    print(json.dumps(summary, sort_keys=True))


if __name__ == "__main__":
    main()
