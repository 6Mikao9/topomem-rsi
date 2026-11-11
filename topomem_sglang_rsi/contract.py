"""Share the immutable SGLang case and static placement contracts."""

from hashlib import sha256
import json
from pathlib import Path
import re

from integrations.topology_memory import sglang_backing as backing, sglang_serving as serving
from integrations.topology_memory import sglang_ingress_gate as ingress
from rsi_memory.memory_program import MemoryProgram, MemoryScenario, plan_backing
from rsi_memory.topology import HardwareTopology, fingerprint


EVALUATOR = "topomem_sglang_rsi.sglang_eval"
SCOPE = "real_model_sglang_hisparse_configurable_memory"
ACTIONS = (("allocate", "place"),)
FEATURES = frozenset({
    "rank", "rank_count", "layer", "bytes", "top_k", "kv_row_bytes", "target_index",
    "target_is_local", "target_is_gpu", "target_is_host", "target_capacity_bytes", "target_used_bytes",
    "target_free_bytes", "target_utilization", "route_hops", "route_latency_ns", "route_bandwidth_bytes_per_s",
    "route_transfer_ns", "shared_bottlenecks", "write_route_hops", "write_route_transfer_ns", "write_route_bandwidth_bytes_per_s",
})
CASE_HASHES = ("target_sha256", "topology_sha256", "scenario_sha256", "workload_sha256", "schedule_sha256")
NCCL_SHA256 = "651cf05af28976ca8d996f4e2750e69db1fa427667ee3405b9e576e9d2960eaf"


def checked_id(value):
    if type(value) is not str or not re.fullmatch(r"[a-z0-9][a-z0-9_-]{0,95}", value):
        raise ValueError("invalid serving experiment, case, or job ID")
    return value


def checked_sha(value):
    if type(value) is not str or not re.fullmatch(r"[a-f0-9]{64}", value):
        raise ValueError("expected a lowercase SHA-256")
    return value


def read_json(path):
    return json.loads(Path(path).read_bytes())


def checked_case(spec):
    if type(spec) is not dict or spec.keys() != {"id", "manifest_sha256", *CASE_HASHES}:
        raise ValueError("invalid immutable serving case descriptor")
    checked_id(spec["id"])
    for key in ("manifest_sha256", *CASE_HASHES):
        checked_sha(spec[key])
    return spec


def checked_inputs(inputs):
    if type(inputs) is not dict or len(inputs) != 4:
        raise ValueError("require four precommitted tokenized serving inputs")
    for rid, row in inputs.items():
        checked_id(rid)
        if (type(row) is not dict or row.keys() != {"input_tokens", "input_ids_sha256"}
                or type(row["input_tokens"]) is not int or not 4096 < row["input_tokens"] <= 8128):
            raise ValueError("tokenized serving request is outside the fixed long-context envelope")
        checked_sha(row["input_ids_sha256"])
    return inputs


def case_descriptor(path, case_id=None):
    path = Path(path)
    raw = (path / "manifest.json").read_bytes()
    manifest = json.loads(raw)
    return checked_case({"id": case_id if case_id is not None else path.name, "manifest_sha256": sha256(raw).hexdigest(),
                         **{key: manifest[key] for key in CASE_HASHES}})


def load_external(path, spec, *, expected_inputs=None, require_gate=False):
    external = read_json(Path(path) / "external_schedule.json")
    if fingerprint(external) != spec["schedule_sha256"]:
        raise ValueError("external schedule does not match the immutable case")
    gate = external.get(ingress.EXTERNAL_FIELD)
    if gate is None:
        if require_gate or expected_inputs is not None:
            raise ValueError("the serving evaluator requires a committed ingress rendezvous")
        return external, None
    groups = ingress.checked_contract(gate)
    tokens = {member.rid: {"input_tokens": member.input_tokens, "input_ids_sha256": member.input_ids_sha256}
              for group in groups for member in group}
    checked_inputs(tokens)
    if expected_inputs is not None and tokens != checked_inputs(expected_inputs):
        raise ValueError("external ingress group differs from the precommitted tokenizer inputs")
    return external, tokens


def load_case(path, spec, *, fixture=False):
    """Check all external inputs. Model projection can omit the fixture entirely."""
    checked_case(spec)
    path = Path(path)
    if case_descriptor(path, spec["id"]) != spec:
        raise ValueError("serving case manifest changed")
    manifest = read_json(path / "manifest.json")
    target = read_json(path / "target.json")
    topology = HardwareTopology.from_dict(read_json(path / "topology.json"))
    scenario = MemoryScenario.from_dict(read_json(path / "scenario.json"))
    if (fingerprint(target) != spec["target_sha256"] or topology.fingerprint() != spec["topology_sha256"]
            or scenario.fingerprint() != spec["scenario_sha256"] or scenario.target_sha256 != spec["target_sha256"]
            or scenario.workload_sha256 != spec["workload_sha256"] or scenario.schedule_sha256 != spec["schedule_sha256"]
            or target.get("attention_dp_size") != 2 or len(scenario.rank_placement) != 2):
        raise ValueError("serving target, topology, scenario, or fixed DP2 placement changed")
    scenario.validate_topology(topology)
    evidence = []
    for digest, relative in manifest["link_evidence"].items():
        expected = f"link_evidence/{checked_sha(digest)}.data"
        if relative != expected:
            raise ValueError("unexpected serving link evidence path")
        item = path / relative
        if item.is_symlink() or not item.resolve(strict=True).is_relative_to(path.resolve(strict=True)):
            raise ValueError("serving evidence must be an immutable case file")
        if sha256(item.read_bytes()).hexdigest() != digest:
            raise ValueError("serving link evidence bytes changed")
        evidence.append(item)
    if any(link.evidence_sha256 not in manifest["link_evidence"] for link in topology.links):
        raise ValueError("serving topology lacks required link evidence")
    if fixture:
        raw_fixture = serving.validate_fixture(read_json(path / "fixture.json"))
        external, tokens = load_external(path, spec)
        actual_schedule = serving.schedule_inputs(raw_fixture, 2, scenario.selector, attention_dp_size=2,
                                                  hot_slots=target["pool"]["hot_slots"])
        if tokens is not None:
            gate = external[ingress.EXTERNAL_FIELD]
            actual_schedule = ingress.bind_schedule(actual_schedule, tokens, gate_timeout_ms=gate["gate_timeout_ms"],
                                                    collective_timeout_ms=gate["collective_timeout_ms"])
        if (fingerprint(raw_fixture) != spec["workload_sha256"] or fingerprint(actual_schedule) != spec["schedule_sha256"]
                or external != actual_schedule):
            raise ValueError("fixed serving request replay or scheduler inputs changed")
    return target, topology, scenario, evidence


def compile_layout(target, topology, scenario, program):
    pool = target["pool"]
    if program is None:
        checked = MemoryProgram.from_dict(serving._PROGRAM)
        allowed = {node: [target["native_host"]] for _, node in scenario.rank_placement}
    else:
        program.validate_actions(ACTIONS)
        program.validate_features(FEATURES)
        checked, allowed = program, target["allowed_nodes"]
    slabs = [{"key": f"r{rank}-l{layer}", "rank": rank, "layer": layer, "bytes": pool["rows"] * scenario.kv_row_bytes,
              "allowed_nodes": allowed[node]} for rank, node in scenario.rank_placement for layer in range(pool["layers"])]
    plan = plan_backing(topology, scenario, checked, slabs, transports=backing.NATIVE_TRANSPORTS)
    workers = backing.bind_plans(topology, scenario, plan, gpu_ordinals={node: spec["ordinal"] for node, spec in target["gpu_bindings"].items()},
                                 attention_tp_size=1, **pool)
    return plan, workers
