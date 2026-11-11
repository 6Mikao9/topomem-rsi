"""Compile startup hot retention and cold placement against one measured budget."""

from copy import deepcopy
from dataclasses import dataclass
from hashlib import sha256
import json
from pathlib import Path

from integrations.topology_memory import sglang_ingress_gate as ingress, sglang_serving as serving
from integrations.topology_memory import sglang_resident
from rsi_memory.memory_program import BACKING_MODE_FEATURE_UNITS, HOT_CAPACITY_FEATURE_UNITS, MemoryProgram, MemoryScenario
from rsi_memory.topology import HardwareTopology, fingerprint

from .contract import ACTIONS as COLD_ACTIONS, FEATURES as COLD_FEATURES, checked_sha, compile_layout


HOT_SLOTS = (4096, 6144, 8192)
HOT_ACTION = ("configure", "hot_capacity")
BACKING_ACTION = ("configure", "backing_mode")
ACTIONS = (HOT_ACTION, BACKING_ACTION, *COLD_ACTIONS)
HOT_FEATURES = frozenset(HOT_CAPACITY_FEATURE_UNITS) | {"rank_count", "top_k", "kv_row_bytes"}
BACKING_FEATURES = frozenset(BACKING_MODE_FEATURE_UNITS) | {"candidate_hot_slots", "case_max_request_tokens",
                                                         "rank_count", "top_k", "kv_row_bytes"}
RESIDENT_HOT_SLOTS = sglang_resident.HOT_SLOTS
RESIDENT_SENTINEL_LOGICAL_BYTES = sglang_resident.RESIDENT_SENTINEL_LOGICAL_BYTES
RESIDENT_ALLOCATOR_CHARGE_BYTES = sglang_resident.RESIDENT_ALLOCATOR_CHARGE_BYTES
PROFILE_ADAPTER_NAMES = ("sglang_serving.py", "sglang_smoke.py", "sglang_backing.py", "sglang_relay.py",
                         "sglang_observe.py", "sglang_dependency.py", "sglang_sources.py",
                         "sglang_ingress_gate.py", "sglang_global_ticks.py", "sglang_resident.py",
                         "sglang_resident_oracle.py")
PROFILE_KIND = "sglang_native_hot_metadata"
MEASURED_FIELDS = ("packed_device_payload_bytes", "device_physical_rows", "device_logical_row_capacity",
                   "request_metadata_slots_including_pad", "hot_metadata_bytes")
PARITY_KEYS = ("effective_schedule_sha256", "dispatch_sha256", "output_ids_sha256", "packed_kv_sha256_by_rank",
               "runtime_sha256", "global_ticks_sha256")


def stock_control_program():
    """Represent the strongest predeclared short-context stock all-host control."""
    one, zero = {"op": "const", "value": 1}, {"op": "const", "value": 0}
    slots = {"op": "feature", "name": "candidate_hot_slots"}
    expected = {"op": "const", "value": 6144}
    exact = {"op": "if_lt", "args": [slots, expected, zero, {"op": "if_lt", "args": [expected, slots, zero, one]}]}
    return MemoryProgram.from_dict({"schema": 1, "kind": "kv_memory_program", "rules": [
        {"event": "configure", "action": "hot_capacity", "when": exact, "score": zero},
        {"event": "allocate", "action": "place", "when": {"op": "feature", "name": "target_is_host"}, "score": zero}]})


def resident_control_program():
    raw = stock_control_program().to_dict()
    raw["rules"][1] = {"event": "configure", "action": "backing_mode",
                       "when": {"op": "feature", "name": "candidate_backing_is_resident"},
                       "score": {"op": "const", "value": 0}}
    return MemoryProgram.from_dict(raw)


def case_token_bounds(external):
    groups = ingress.checked_contract(external.get(ingress.EXTERNAL_FIELD))
    tokens, maxima = {}, []
    for phase, members in zip(ingress.PHASES, groups, strict=True):
        rows = external[phase]
        if len(rows) != len(members) or {row["rid"] for row in rows} != {member.rid for member in members}:
            raise ValueError("fixed ingress commitments differ from scheduled requests")
        by_rid = {row["rid"]: row for row in rows}
        for member in members:
            row = by_rid[member.rid]
            allowance = row["max_new_tokens"]
            if (type(allowance) is not int or allowance < 1
                    or type(row["routed_dp_rank"]) is not int or row["routed_dp_rank"] != member.rank):
                raise ValueError("fixed ingress rank or decode allowance differs")
            tokens[member.rid] = {"input_tokens": member.input_tokens, "input_ids_sha256": member.input_ids_sha256}
            maxima.append(member.input_tokens + allowance)
    return tokens, max(maxima)


def normalized_schedule(schedule):
    result = deepcopy(schedule)
    config = json.loads(result["engine"]["hisparse_config"])
    if type(config.get("device_buffer_size")) is not int or config["device_buffer_size"] not in HOT_SLOTS:
        raise ValueError("cannot normalize an unbound HiSparse hot setting")
    del config["device_buffer_size"]
    result["engine"]["hisparse_config"] = config
    return result


def schedule_at_capacity(schedule, slots):
    result = deepcopy(schedule)
    config = json.loads(result["engine"]["hisparse_config"])
    config["device_buffer_size"] = slots
    result["engine"]["hisparse_config"] = json.dumps(config)
    if normalized_schedule(result) != normalized_schedule(schedule):
        raise ValueError("startup configuration changed the fixed external schedule")
    return result


def profile_runtime_identity(runtime):
    return fingerprint(runtime)


def profile_adapter_sources():
    directory = Path(serving.__file__).parent
    return {name: sha256((directory / name).read_bytes()).hexdigest() for name in PROFILE_ADAPTER_NAMES}


def observed_rank_geometry(result):
    rows = []
    for rank, record in enumerate(result["ranks"]):
        backing = record["backing"]
        if backing["rank"] != rank or backing["cuda_device"] != rank:
            raise ValueError("hot metadata measurements use a different GPU ordinal or rank")
        values = {name: backing[name] for name in MEASURED_FIELDS}
        if any(type(value) is not int or value <= 0 for value in values.values()):
            raise ValueError("real backing report lacks positive observed geometry")
        rows.append(values)
    if len(rows) != 2:
        raise ValueError("hot metadata profile requires both measured DP ranks")
    return rows


def validate_profile(profile, target, topology, external):
    fields = {"schema", "kind", "status", "base_target_sha256", "base_topology_sha256", "calibration_workload_sha256",
              "runtime_identity_sha256", "calibration_adapter_sources_sha256", "observations"}
    if (type(profile) is not dict or profile.keys() != fields or type(profile["schema"]) is not int or profile["schema"] != 1
            or profile["kind"] != PROFILE_KIND or profile["status"] != "observed_native_parity"):
        raise ValueError("blocked until measured native hot metadata profile is supplied")
    if (profile["base_target_sha256"] != fingerprint(target) or profile["base_topology_sha256"] != topology.fingerprint()
            or type(profile["observations"]) is not dict or profile["observations"].keys() != set(map(str, HOT_SLOTS))):
        raise ValueError("measured metadata must bind this target, topology and all three native capacities")
    checked_sha(profile["calibration_workload_sha256"])
    checked_sha(profile["runtime_identity_sha256"])
    calibration_sources = profile["calibration_adapter_sources_sha256"]
    if calibration_sources != profile_adapter_sources():
        raise ValueError("hot profile must come from the same qualified serving, ingress and global observer sources")
    pool = target["pool"]
    expected_rank_concurrency = external["engine"]["max_running_requests"] // external["engine"]["dp_size"]
    observations = []
    for slots in HOT_SLOTS:
        item = profile["observations"][str(slots)]
        if type(item) is not dict or item.keys() != {"result_sha256", "case_manifest_sha256", "ranks"}:
            raise ValueError("each hot capacity requires a real native receipt and both rank observations")
        checked_sha(item["result_sha256"])
        checked_sha(item["case_manifest_sha256"])
        ranks = item["ranks"]
        if type(ranks) is not list or len(ranks) != 2:
            raise ValueError("each measured capacity must cover both attention ranks")
        for row in ranks:
            if type(row) is not dict or row.keys() != set(MEASURED_FIELDS) or any(type(value) is not int or value <= 0 for value in row.values()):
                raise ValueError("hot metadata geometry must be reported by the native runtime")
            if (row["device_logical_row_capacity"] != external["engine"]["max_total_tokens"]
                    or row["device_physical_rows"] != row["device_logical_row_capacity"] + pool["page_size"]
                    or row["request_metadata_slots_including_pad"] - 1 != expected_rank_concurrency
                    or row["device_physical_rows"] * pool["layers"] * 656 != row["packed_device_payload_bytes"]
                    or (slots + pool["page_size"]) * expected_rank_concurrency > row["device_logical_row_capacity"]):
                raise ValueError("observed hot capacity conflicts with the fixed physical pool or external concurrency")
        observations.append(ranks)
    baseline = observations[0]
    for index, ranks in enumerate(observations[1:], 1):
        for rank, row in enumerate(ranks):
            if (any(row[field] != baseline[rank][field] for field in MEASURED_FIELDS[:-1])
                    or row["hot_metadata_bytes"] <= observations[index - 1][rank]["hot_metadata_bytes"]):
                raise ValueError("hot probe changed physical rows, concurrency, or monotonic metadata")
    if len({item["result_sha256"] for item in profile["observations"].values()}) != len(HOT_SLOTS):
        raise ValueError("hot capacities must be measured in separate real receipts")
    return profile


@dataclass(frozen=True)
class CompiledMemory:
    target: dict
    topology: HardwareTopology
    scenario: MemoryScenario
    external: dict
    cold_program: MemoryProgram | None
    plan: dict
    workers: tuple
    audit: dict


def compile_memory(target, topology, scenario, external, profile, program, *, native_hot=None):
    """Apply a bounded startup decision, reserve observed metadata, and lower cold placement."""
    if (target["pool"]["hot_slots"] != 4096 or fingerprint(target) != scenario.target_sha256
            or fingerprint(external) != scenario.schedule_sha256 or len(scenario.rank_placement) != 2
            or json.loads(external["engine"]["hisparse_config"])["device_buffer_size"] != 4096
            or [node.id for node in topology.nodes if node.kind == "host"] != [target["native_host"]]):
        raise ValueError("configurable memory must start from the frozen 4096-slot DP2 case")
    commitments, maximum = case_token_bounds(external)
    validate_profile(profile, target, topology, external)
    if native_hot is not None and (program is not None or type(native_hot) is not int or native_hot not in HOT_SLOTS):
        raise ValueError("only the trusted native reference can override startup capacity")
    baseline = profile["observations"]["4096"]["ranks"]
    deltas = {}
    options = []
    for slots in HOT_SLOTS:
        ranks = profile["observations"][str(slots)]["ranks"]
        delta = {node: ranks[rank]["hot_metadata_bytes"] - baseline[rank]["hot_metadata_bytes"] for rank, node in scenario.rank_placement}
        deltas[slots] = delta
        remaining = min(topology.node(node).capacity_bytes - extra for node, extra in delta.items())
        if remaining < 0:
            continue
        options.append((slots, {"candidate_hot_slots": slots, "candidate_max_extra_metadata_bytes": max(delta.values()),
                                "candidate_total_extra_metadata_bytes": sum(delta.values()), "candidate_min_remaining_cold_bytes": remaining,
                                "max_context_tokens": external["engine"]["context_length"],
                                "case_max_request_tokens": maximum,
                                "max_running_requests_per_rank": ranks[0]["request_metadata_slots_including_pad"] - 1,
                                "rank_count": len(scenario.rank_placement), "top_k": scenario.top_k, "kv_row_bytes": scenario.kv_row_bytes}))
    slots, decision_index, cold = native_hot or 4096, None, None
    mode, mode_index = "tiered", None
    rule_indices = []
    if program is not None:
        program.validate_actions(ACTIONS)
        startup = [(index, rule) for index, rule in enumerate(program.rules) if (rule.event, rule.action) == HOT_ACTION]
        placement = [(index, rule) for index, rule in enumerate(program.rules) if (rule.event, rule.action) in COLD_ACTIONS]
        backing = [(index, rule) for index, rule in enumerate(program.rules) if (rule.event, rule.action) == BACKING_ACTION]
        if placement:
            cold = MemoryProgram(tuple(rule for _, rule in placement))
            cold.validate_features(COLD_FEATURES)
            rule_indices = [index for index, _ in placement]
        if startup:
            config = MemoryProgram(tuple(rule for _, rule in startup))
            config.validate_features(HOT_FEATURES)
            selected = config.choose(*HOT_ACTION, options)
            if selected is None:
                raise MemoryError("startup rules selected no measured, capacity-safe hot configuration")
            index, slots = selected
            decision_index = startup[index][0]
        if backing:
            chooser = MemoryProgram(tuple(rule for _, rule in backing))
            chooser.validate_features(BACKING_FEATURES)
            shared = {"candidate_hot_slots": slots, "case_max_request_tokens": maximum,
                      "resident_hot_slot_margin": RESIDENT_HOT_SLOTS - maximum,
                      "rank_count": len(scenario.rank_placement), "top_k": scenario.top_k, "kv_row_bytes": scenario.kv_row_bytes}
            choices = [(value, {**shared, "candidate_backing_is_resident": int(value == "resident")})
                       for value in ("tiered", "resident")]
            selected = chooser.choose(*BACKING_ACTION, choices)
            if selected is None:
                raise MemoryError("backing-mode rules selected no explicit storage mode")
            index, mode = selected
            mode_index = backing[index][0]
        if mode == "resident" and placement:
            raise ValueError("resident backing cannot include cold placement rules")
        if mode == "tiered" and not placement:
            raise ValueError("tiered backing requires an executable cold placement rule")
    if slots not in {value for value, _ in options}:
        raise MemoryError("native or default hot configuration exceeds the common HBM budget")
    delta = deltas[slots]
    remaining = topology.to_dict()
    for node in remaining["nodes"]:
        node["capacity_bytes"] -= delta.get(node["id"], 0)
    derived_graph = HardwareTopology.from_dict(remaining)
    derived_target = deepcopy(target)
    derived_target["pool"]["hot_slots"] = slots
    derived_external = schedule_at_capacity(external, slots)
    derived_scenario = MemoryScenario.from_dict({**scenario.to_dict(), "target_sha256": fingerprint(derived_target),
                                                 "schedule_sha256": fingerprint(derived_external)})
    plan, workers = compile_layout(derived_target, derived_graph, derived_scenario, cold)
    resident, logical, allocator = None, {}, {}
    if mode == "resident":
        if (slots != RESIDENT_HOT_SLOTS or target["pool"]["layers"] != 6 or target["pool"]["page_size"] != 64
                or scenario.kv_row_bytes != 656 or scenario.top_k != 2048):
            raise ValueError("resident backing requires the qualified 6144-slot packed GLM pool")
        if maximum > slots:
            raise MemoryError("a committed request can outgrow resident hot KV before its decode limit")
        resident = sglang_resident.proof_from_commitments(derived_external["engine"], derived_external, commitments,
                                                          device_rows=derived_external["engine"]["max_total_tokens"])
        logical = {node: RESIDENT_SENTINEL_LOGICAL_BYTES for _, node in scenario.rank_placement}
        allocator = {node: RESIDENT_ALLOCATOR_CHARGE_BYTES for _, node in scenario.rank_placement}
        # Native layout remains as the comparison geometry. No shadow cold slab is allocated.
        ledger = {node.id: allocator.get(node.id, 0) for node in topology.nodes}
    else:
        ledger = dict(plan["used_bytes_by_node"])
        ordinals = {row["ordinal"]: node for node, row in target["gpu_bindings"].items()}
        for worker in workers:
            scratch = worker.relay_workspace_bytes
            ledger[ordinals[worker.compute_device]] += scratch["compute"]
            if worker.peer_device is not None:
                ledger[ordinals[worker.peer_device]] += scratch["peer"]
            ledger[target["native_host"]] += scratch["host"]
    for node, extra in delta.items():
        ledger[node] += extra
    if any(used > topology.node(node).capacity_bytes for node, used in ledger.items()):
        raise MemoryError("combined native metadata and backing allocations exceed the common budget")
    audit = {"schema": 1, "full_program_sha256": None if program is None else program.fingerprint(),
             "cold_program_sha256": plan["program_sha256"], "placement_rule_indices_in_full_program": rule_indices,
             "metadata_profile_sha256": fingerprint(profile), "base_target_sha256": scenario.target_sha256,
             "base_topology_sha256": topology.fingerprint(), "external_scheduler_sha256": fingerprint(normalized_schedule(external)),
             "ingress_contract_sha256": fingerprint(external[ingress.EXTERNAL_FIELD]),
             "startup": {"hot_slots": slots, "rule_index": decision_index, "native_reference_override": native_hot,
                         "metadata_bytes_reserved_by_node": delta, "eligible_hot_slots": [slot for slot, _ in options],
                         "backing_mode": mode, "backing_rule_index": mode_index, "resident": resident,
                         "case_max_request_tokens": maximum,
                         "resident_sentinel_logical_bytes_by_node": logical,
                         "resident_sentinel_allocator_charge_bytes_by_node": allocator},
             "derived_target_sha256": derived_scenario.target_sha256, "derived_schedule_sha256": derived_scenario.schedule_sha256,
             "derived_topology_sha256": derived_graph.fingerprint(), "derived_scenario_sha256": derived_scenario.fingerprint(),
             "cold_plan_sha256": fingerprint(plan),
             "cold_plan_role": "native_comparison_geometry_only" if mode == "resident" else "runtime_cold_backing",
             "total_incremental_bytes_by_node": ledger,
             "equivalent_to_strong_stock_control": mode == "tiered" and slots == 6144 and all(
                 not worker.local_hbm_bytes and not worker.peer_hbm_bytes for worker in workers)}
    return CompiledMemory(derived_target, derived_graph, derived_scenario, derived_external, cold, plan, workers, audit)


def check_live_metadata(result, profile, slots):
    expected = profile["observations"][str(slots)]["ranks"]
    if (observed_rank_geometry(result) != expected
            or any(row["backing"]["hot_slots"] != slots for row in result["ranks"])
            or profile_runtime_identity(result["runtime"]) != profile["runtime_identity_sha256"]):
        raise ValueError("live runtime geometry or environment differs from measured hot metadata")
    for row, observed in zip(result["ranks"], expected, strict=True):
        limits = row["schedule"]["worker_limits"]
        if (limits["max_running_requests"] != observed["request_metadata_slots_including_pad"] - 1
                or limits["max_total_num_tokens"] != observed["device_logical_row_capacity"]):
            raise ValueError("effective scheduler concurrency or token limit changed from the measured target")
