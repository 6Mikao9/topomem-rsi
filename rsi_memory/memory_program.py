"""Bounded generated rules for KV placement and movement on a hardware graph."""

from __future__ import annotations

from dataclasses import dataclass
import json

from .scalar_program import evaluate_scalar, validate_program
from .topology import HardwareTopology, _digest, _fields, _id, _integer, _list, fingerprint


ACTION_DOMAIN = {
    "allocate": ("place", "replicate", "prefetch", "evict", "reclaim", "demote"),
    "access": ("replicate", "migrate", "prefetch", "evict", "reclaim", "demote"),
    "hint": ("replicate", "migrate", "prefetch", "evict", "reclaim"),
    "release": ("reclaim",),
    "idle": ("evict", "reclaim"),
}
_PROGRAM_ACTION_DOMAIN = {**ACTION_DOMAIN, "configure": ("hot_capacity", "backing_mode")}
FEATURE_UNITS = {
    "rank": "rank ordinal", "rank_count": "count", "layer": "layer ordinal", "block": "block ordinal",
    "bytes": "bytes", "top_k": "selection units", "kv_row_bytes": "bytes",
    "time_ns": "ns", "age_ns": "ns", "accesses": "count", "replicas": "count",
    "readers": "count", "inflight": "count", "target_index": "sorted logical node ordinal",
    "target_is_local": "0/1", "target_is_gpu": "0/1", "target_is_host": "0/1",
    "target_capacity_bytes": "bytes", "target_used_bytes": "bytes", "target_free_bytes": "bytes",
    "target_utilization": "fraction", "target_has_copy": "0/1", "source_is_local": "0/1",
    "source_is_gpu": "0/1", "source_is_host": "0/1", "source_used_bytes": "bytes",
    "route_hops": "count", "route_latency_ns": "ns", "route_bandwidth_bytes_per_s": "bytes/s",
    "route_transfer_ns": "ns", "route_queue_ns": "ns", "shared_bottlenecks": "count",
    "write_route_hops": "count", "write_route_transfer_ns": "ns", "write_route_bandwidth_bytes_per_s": "bytes/s",
}
BASE_FEATURES = frozenset(FEATURE_UNITS)
LIFECYCLE_FEATURE_UNITS = {
    "valid_rows": "rows in this physical KV block version", "block_rows": "rows per physical block slot",
    "prior_version_rows": "rows physically preserved from the previous block version",
    "logical_refs": "live session references to this immutable block version",
    "page_accesses": "observed attention reads of this layer and logical block index across versions",
    "last_use_step": "external step ordinal of last producer publication or attention use",
    "recency_steps": "external steps since last producer publication or attention use",
}
FEATURE_UNITS.update(LIFECYCLE_FEATURE_UNITS)
HOT_CAPACITY_FEATURE_UNITS = {
    "candidate_hot_slots": "resident token slots per live request",
    "candidate_max_extra_metadata_bytes": "maximum measured extra bytes on any GPU versus the 4096-slot native reference",
    "candidate_total_extra_metadata_bytes": "sum of measured extra bytes across the compute GPUs versus the 4096-slot native reference",
    "candidate_min_remaining_cold_bytes": "minimum device cold-storage budget after reserving extra metadata",
    "max_context_tokens": "fixed external context token limit",
    "case_max_request_tokens": "largest committed prompt length plus its external decode allowance",
    "max_running_requests_per_rank": "fixed external effective concurrent requests per attention rank",
}
BACKING_MODE_FEATURE_UNITS = {
    "candidate_backing_is_resident": "0/1",
    "resident_hot_slot_margin": "qualified resident hot slots minus the longest bound request",
}
_PROGRAM_FEATURE_UNITS = {**FEATURE_UNITS, **HOT_CAPACITY_FEATURE_UNITS, **BACKING_MODE_FEATURE_UNITS}
MAX_RULES = 24


@dataclass(frozen=True)
class MemoryScenario:
    rank_placement: tuple[tuple[int, str], ...]
    family: str
    selector: str
    selection_unit: str
    top_k: int
    kv_row_bytes: int
    target_sha256: str
    workload_sha256: str
    schedule_sha256: str

    @classmethod
    def from_dict(cls, data):
        _fields(data, ("schema", "rank_placement", "attention", "target_sha256", "workload_sha256", "schedule_sha256"))
        if type(data["schema"]) is not int or data["schema"] != 1:
            raise ValueError("unsupported KV memory scenario schema")
        ranks = []
        for binding in _list(data["rank_placement"], "rank placement", limit=32):
            _fields(binding, ("rank", "node"))
            ranks.append((_integer(binding["rank"], "rank"), _id(binding["node"])))
        ranks.sort()
        if not ranks or [rank for rank, _ in ranks] != list(range(len(ranks))):
            raise ValueError("rank placement requires unique, contiguous rank ordinals")
        attention = data["attention"]
        _fields(attention, ("family", "selector", "selection_unit", "top_k", "kv_row_bytes"))
        unit = attention["selection_unit"]
        if unit not in ("token", "block"):
            raise ValueError("unsupported attention selection unit")
        return cls(tuple(ranks), _id(attention["family"]), _id(attention["selector"]), unit,
                   _integer(attention["top_k"], "Top-K", 1), _integer(attention["kv_row_bytes"], "packed KV row width", 1),
                   _digest(data["target_sha256"]), _digest(data["workload_sha256"]), _digest(data["schedule_sha256"]))

    def to_dict(self):
        return {"schema": 1, "rank_placement": [{"rank": rank, "node": node} for rank, node in self.rank_placement],
                "attention": {"family": self.family, "selector": self.selector, "selection_unit": self.selection_unit,
                              "top_k": self.top_k, "kv_row_bytes": self.kv_row_bytes},
                "target_sha256": self.target_sha256, "workload_sha256": self.workload_sha256, "schedule_sha256": self.schedule_sha256}

    def fingerprint(self):
        return fingerprint(self.to_dict())

    def rank_node(self, rank):
        if type(rank) is not int or not 0 <= rank < len(self.rank_placement):
            raise ValueError("unknown external attention rank")
        return self.rank_placement[rank][1]

    def validate_topology(self, topology):
        if any(topology.node(node).kind != "gpu" for _, node in self.rank_placement):
            raise ValueError("external ranks must bind actual GPU nodes")


@dataclass(frozen=True)
class Rule:
    event: str
    action: str
    when: dict
    score: dict


@dataclass(frozen=True)
class MemoryProgram:
    rules: tuple[Rule, ...]

    @classmethod
    def from_dict(cls, data):
        _fields(data, ("schema", "kind", "rules"))
        if type(data["schema"]) is not int or data["schema"] != 1 or data["kind"] != "kv_memory_program":
            raise ValueError("unsupported KV memory program schema")
        rules = []
        for rule in _list(data["rules"], "memory rules", limit=MAX_RULES):
            _fields(rule, ("event", "action", "when", "score"))
            event, action = rule["event"], rule["action"]
            if type(event) is not str or action not in _PROGRAM_ACTION_DOMAIN.get(event, ()):
                raise ValueError("memory rule requests an unsupported event/action pair")
            rules.append(Rule(event, action, validate_program(rule["when"], _PROGRAM_FEATURE_UNITS),
                              validate_program(rule["score"], _PROGRAM_FEATURE_UNITS)))
        if not any((rule.event, rule.action) in {("allocate", "place"), ("configure", "backing_mode")} for rule in rules):
            raise ValueError("memory program requires a generated allocation or backing-mode rule")
        return cls(tuple(rules))

    def to_dict(self):
        data = {"schema": 1, "kind": "kv_memory_program", "rules": [
            {"event": r.event, "action": r.action, "when": r.when, "score": r.score} for r in self.rules]}
        return json.loads(json.dumps(data, allow_nan=False))

    def fingerprint(self):
        return fingerprint(self.to_dict())

    def validate_actions(self, supported):
        unexpected = {(rule.event, rule.action) for rule in self.rules} - set(supported)
        if unexpected:
            raise ValueError(f"adapter cannot execute generated actions: {sorted(unexpected)}")

    def validate_features(self, supported):
        def used(node):
            if node["op"] == "feature":
                yield node["name"]
            for child in node.get("args", ()):
                yield from used(child)

        names = {name for rule in self.rules for tree in (rule.when, rule.score) for name in used(tree)}
        unsupported = names - set(supported)
        if unsupported:
            raise ValueError(f"adapter cannot observe generated features: {sorted(unsupported)}")

    def choose(self, event, action, candidates):
        """First eligible rule wins. Candidate ordering supplies stable score ties."""
        candidates = tuple(candidates)
        for index, rule in enumerate(self.rules):
            if (rule.event, rule.action) != (event, action):
                continue
            scored = [(evaluate_scalar(rule.score, features), ordinal, candidate)
                      for ordinal, (candidate, features) in enumerate(candidates) if evaluate_scalar(rule.when, features) > 0]
            if scored:
                return index, min(scored, key=lambda item: item[:2])[2]
        return None


def transfer_duration_ns(topology, route, size):
    return sum(link.latency_ns + (size * 1_000_000_000 + topology.link_bandwidth(link) - 1) // topology.link_bandwidth(link)
               for link in route.links)


def candidate_features(topology, scenario, *, rank, layer, block, size, target, source, route, usage,
                       write_route=None, time_ns=0, age_ns=0, accesses=0, replicas=0, readers=0, inflight=0,
                       target_has_copy=False, route_queue_ns=0):
    local = scenario.rank_node(rank)
    target_node, source_node = topology.node(target), topology.node(source)
    cap, used = target_node.capacity_bytes, usage[target]
    write_links = () if write_route is None else write_route.links
    return {
        "rank": rank, "rank_count": len(scenario.rank_placement), "layer": layer, "block": block, "bytes": size,
        "top_k": scenario.top_k, "kv_row_bytes": scenario.kv_row_bytes, "time_ns": time_ns, "age_ns": age_ns,
        "accesses": accesses, "replicas": replicas, "readers": readers, "inflight": inflight,
        "target_index": topology.nodes.index(target_node), "target_is_local": int(target == local),
        "target_is_gpu": int(target_node.kind == "gpu"), "target_is_host": int(target_node.kind == "host"),
        "target_capacity_bytes": cap, "target_used_bytes": used, "target_free_bytes": cap - used,
        "target_utilization": used / cap if cap else 0, "target_has_copy": int(target_has_copy),
        "source_is_local": int(source == local), "source_is_gpu": int(source_node.kind == "gpu"),
        "source_is_host": int(source_node.kind == "host"), "source_used_bytes": usage[source],
        "route_hops": len(route.links), "route_latency_ns": sum(link.latency_ns for link in route.links),
        "route_bandwidth_bytes_per_s": min((topology.link_bandwidth(link) for link in route.links), default=0),
        "route_transfer_ns": transfer_duration_ns(topology, route, size), "route_queue_ns": route_queue_ns,
        "shared_bottlenecks": len({group for link in route.links for group in link.bottlenecks}),
        "write_route_hops": len(write_links),
        "write_route_transfer_ns": 0 if write_route is None else transfer_duration_ns(topology, write_route, size),
        "write_route_bandwidth_bytes_per_s": min((topology.link_bandwidth(link) for link in write_links), default=0),
    }


def plan_backing(topology, scenario, program, slabs, *, initial_used=None, transports=None):
    """Emit capacity-safe static slab reservations for real adapter consumption."""
    scenario.validate_topology(topology)
    program.validate_actions({("allocate", "place")})
    program.validate_features(BASE_FEATURES)
    usage = {node.id: 0 for node in topology.nodes}
    for node, size in ({} if initial_used is None else initial_used).items():
        if _integer(size, "initial occupied bytes") > topology.node(node).capacity_bytes:
            raise ValueError("initial backing exceeds the node capacity")
        usage[node] = size
    seen, actions = set(), []
    for slab in slabs:
        _fields(slab, ("key", "rank", "layer", "bytes", "allowed_nodes"))
        key, rank = _id(slab["key"]), slab["rank"]
        if key in seen:
            raise ValueError("duplicate logical slab key")
        seen.add(key)
        local = scenario.rank_node(rank)
        layer, size = _integer(slab["layer"], "layer"), _integer(slab["bytes"], "slab bytes", 1)
        if size % scenario.kv_row_bytes:
            raise ValueError("slab bytes must cover a whole number of packed KV rows")
        allowed = sorted(_id(node) for node in _list(slab["allowed_nodes"], "adapter destinations", limit=32))
        if not allowed or len(allowed) != len(set(allowed)):
            raise ValueError("adapter must supply unique allowed backing destinations")
        candidates = []
        for target in allowed:
            node = topology.node(target)
            if node.kind == "fabric":
                raise ValueError("fabric nodes cannot back KV slabs")
            if usage[target] + size > node.capacity_bytes:
                continue
            for read in topology.routes(target, local, transports=transports):
                for write in topology.routes(local, target, transports=transports):
                    candidate = (target, read, write)
                    features = candidate_features(topology, scenario, rank=rank, layer=layer, block=0, size=size,
                                                  target=target, source=target, route=read, write_route=write, usage=usage)
                    candidates.append((candidate, features))
        selected = program.choose("allocate", "place", candidates)
        if selected is None:
            raise MemoryError(f"no capacity-safe generated backing for slab {key!r}")
        index, (target, read, write) = selected
        usage[target] += size
        actions.append({"action": "reserve_backing", "key": key, "rank": rank, "layer": layer, "bytes": size,
                        "node": target, "read_links": list(read.ids), "write_links": list(write.ids), "rule_index": index})
    if not actions:
        raise ValueError("adapter supplied no KV slabs")
    return {"schema": 1, "topology_sha256": topology.fingerprint(), "scenario_sha256": scenario.fingerprint(),
            "program_sha256": program.fingerprint(), "actions": actions, "used_bytes_by_node": usage}


def generator_context(topology, scenario, feedback=(), *, supported_actions=None, supported_features=BASE_FEATURES):
    """Show the proposer its fixed physical and attention inputs plus measured feedback."""
    scenario.validate_topology(topology)
    domain_schema = ACTION_DOMAIN
    if supported_actions is not None:
        supported_actions = frozenset(supported_actions)
        if not supported_actions <= {(event, action) for event, actions in _PROGRAM_ACTION_DOMAIN.items() for action in actions}:
            raise ValueError("adapter advertised unknown memory actions")
        if any(event == "configure" for event, _ in supported_actions):
            domain_schema = _PROGRAM_ACTION_DOMAIN
    domain = {event: [action for action in actions if supported_actions is None or (event, action) in supported_actions]
              for event, actions in domain_schema.items()}
    features = frozenset(supported_features)
    if not features <= _PROGRAM_FEATURE_UNITS.keys():
        raise ValueError("adapter advertised unknown memory features")
    return {"topology": topology.to_dict(), "topology_sha256": topology.fingerprint(), "scenario": scenario.to_dict(),
            "scenario_sha256": scenario.fingerprint(), "program": {"schema": 1, "kind": "kv_memory_program",
                "events_and_actions": domain, "max_rules": MAX_RULES,
                "features": {name: unit for name, unit in _PROGRAM_FEATURE_UNITS.items() if name in features},
                "when": "eligible if positive", "score": "lower wins within the first eligible rule"},
            "measured_feedback": json.loads(json.dumps(list(feedback), allow_nan=False))}
