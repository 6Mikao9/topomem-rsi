"""Bind a backend-specific generation grammar without changing the frozen judge."""

from dataclasses import dataclass, replace
from hashlib import sha256
from pathlib import Path
import re

from .memory_program import MAX_RULES, MemoryProgram
from .topology_rsi import Domain, SEED_SOURCE, TopologyRSI


@dataclass(frozen=True)
class ProgramFamily:
    name: str
    required_rules: tuple[tuple[str, str], ...]
    forbidden_rules: tuple[tuple[str, str], ...]
    selection_condition: str


@dataclass(frozen=True)
class ContractDomain(Domain):
    program_families: tuple[ProgramFamily, ...] = ()


def _rule(pair):
    return {"event": pair[0], "action": pair[1]}


def _parser_accepts_single_rule(pair):
    # The existing parser is also shipped to the worker. Use its actual
    # minimum admissible shapes without changing its bytes or duplicating them.
    try:
        MemoryProgram.from_dict({"schema": 1, "kind": "kv_memory_program", "rules": [
            {**_rule(pair), "when": {"op": "const", "value": 1}, "score": {"op": "const", "value": 0}},
        ]})
    except ValueError:
        return False
    return True


def _checked_actions(value):
    if (type(value) is not tuple or any(type(pair) is not tuple or len(pair) != 2
            or any(type(item) is not str or not re.fullmatch("[a-z_]+", item) for item in pair) for pair in value)
            or len(set(value)) != len(value)):
        raise ValueError("invalid trusted program action pairs")
    return set(value)


def build_program_contract(domain):
    if not isinstance(domain, ContractDomain):
        raise ValueError("require an explicit trusted program-contract domain")
    enabled = _checked_actions(domain.supported_actions)
    required_any = tuple(pair for pair in sorted(enabled) if _parser_accepts_single_rule(pair))
    if not required_any:
        raise ValueError("backend exposes no program rule accepted by the frozen parser")
    events_and_actions = {}
    for event, action in sorted(enabled):
        events_and_actions.setdefault(event, []).append(action)
    contract = {"schema": 1, "kind": "kv_memory_program", "max_rules": MAX_RULES,
                "events_and_actions": events_and_actions,
                "execution_labels": "Action counts and receipt labels are read-only observations."}
    if len(required_any) == 1:
        contract["required_rule"] = _rule(required_any[0])
    else:
        contract["required_any_rule"] = [_rule(pair) for pair in required_any]
    families = domain.program_families
    if type(families) is not tuple:
        raise ValueError("trusted program families must be a tuple")
    if not families:
        return contract
    covered, names, rules = set(), set(), []
    for family in families:
        if (type(family) is not ProgramFamily or type(family.name) is not str
                or not re.fullmatch("[a-z][a-z_]{0,31}", family.name) or family.name in names
                or type(family.selection_condition) is not str
                or not family.selection_condition.strip() or len(family.selection_condition) > 1000):
            raise ValueError("invalid trusted program family")
        required = _checked_actions(family.required_rules)
        forbidden = _checked_actions(family.forbidden_rules)
        permitted_root = required & set(required_any)
        if not permitted_root or required & forbidden or not (required | forbidden) <= enabled:
            raise ValueError("program family contradicts the frozen parser or enabled actions")
        names.add(family.name)
        covered.update(permitted_root)
        rules.append({"name": family.name, "required_rules": [_rule(pair) for pair in family.required_rules],
                      "forbidden_rules": [_rule(pair) for pair in family.forbidden_rules],
                      "selection_condition": family.selection_condition})
    if covered != set(required_any):
        raise ValueError("trusted program families omit a required rule alternative")
    contract["program_families"] = rules
    contract["rule_requirement_scope"] = (
        "These rules describe the complete assembled program. A mutate response can replace only one existing rule. "
        "A construct response can define a complete program. An edit must still leave a valid family."
    )
    return contract


class ContractTopologyRSI(TopologyRSI):
    """Run the frozen loop with the manifested backend program grammar."""

    def __init__(self, output, config, domain, request, *, backend_source, seed_source=SEED_SOURCE):
        program_contract = build_program_contract(domain)
        root = Path(__file__).resolve().parents[1]
        sources = (Path(__file__).resolve(), Path(backend_source).resolve(strict=True))
        source_hashes = {path.relative_to(root).as_posix(): sha256(path.read_bytes()).hexdigest() for path in sources}
        if len(source_hashes) != 2 or "generator_program_protocol" in domain.contract:
            raise ValueError("require an independent and unbound trusted generator protocol")
        provenance = {"schema": 1, "kind": "backend_program_contract_v3",
                      "trusted_program_contract": program_contract, "source_sha256": source_hashes}
        bound_domain = replace(domain, contract={**domain.contract, "generator_program_protocol": provenance})
        super().__init__(output, config, bound_domain, request, seed_source)
        # The base constructor only validates and records the run identity.
        # All provider calls use the exact grammar bound in that identity.
        self.program_contract = program_contract
