"""Run generated KV rules against real CUDA storage and fixed block attention."""

from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from hashlib import sha256
import heapq
from time import perf_counter_ns

from rsi_memory.compiled_memory import CompiledProgram
from rsi_memory.memory_program import ACTION_DOMAIN, FEATURE_UNITS, candidate_features, transfer_duration_ns
from rsi_memory.topology import Route, fingerprint
from .torch_pressure_lowering import lower_pressure_keys
from .torch_route_lowering import lower_static_routes
from .torch_work_counts import WorkCounts


SUPPORTED_ACTIONS = frozenset((event, action) for event, actions in ACTION_DOMAIN.items() for action in actions)
SUPPORTED_FEATURES = frozenset(name for name in FEATURE_UNITS if not name.startswith("write_route_"))
ARMS = frozenset({"native_lru", "manual_topology", "generated"})
def _used_bytes(runtime, node):
    return len(runtime.pools[node].used)*runtime.workload.block_bytes


_FEATURE_READERS = {
    "layer": lambda runtime, state, src, dst, cap: state.spec.layer,
    "block": lambda runtime, state, src, dst, cap: state.spec.index,
    "time_ns": lambda runtime, state, src, dst, cap: runtime.clock,
    "age_ns": lambda runtime, state, src, dst, cap: runtime.clock-state.born_ns,
    "accesses": lambda runtime, state, src, dst, cap: state.reads,
    "replicas": lambda runtime, state, src, dst, cap: max(0, len(state.copies)-1),
    "target_has_copy": lambda runtime, state, src, dst, cap: int(dst in state.copies),
    "target_used_bytes": lambda runtime, state, src, dst, cap: _used_bytes(runtime, dst),
    "target_free_bytes": lambda runtime, state, src, dst, cap: cap-_used_bytes(runtime, dst),
    "target_utilization": lambda runtime, state, src, dst, cap: _used_bytes(runtime, dst)/cap if cap else 0,
    "source_used_bytes": lambda runtime, state, src, dst, cap: _used_bytes(runtime, src),
    "valid_rows": lambda runtime, state, src, dst, cap: state.spec.valid_rows,
    "logical_refs": lambda runtime, state, src, dst, cap: state.refs,
    "prior_version_rows": lambda runtime, state, src, dst, cap: runtime.versions.get(state.spec.id, (0, 0))[1],
    "page_accesses": lambda runtime, state, src, dst, cap: runtime.reads[state.spec.layer, state.spec.index],
    "last_use_step": lambda runtime, state, src, dst, cap: state.last_use,
    "recency_steps": lambda runtime, state, src, dst, cap: runtime.ordinal-state.last_use,
}
_DYNAMIC_FEATURES = frozenset(_FEATURE_READERS)


def _lower_producer_retain(program, local, rule_routes, rule_features):
    """Prove the existing producer copy is the only possible allocation replica."""
    group = "allocate", "replicate"
    if group not in program.rules or program.features[group] & _DYNAMIC_FEATURES:
        return None
    if any(("allocate", action) in program.rules for action in ("migrate", "prefetch")):
        return None
    if any(routes for (key, _, dst), routes in rule_routes.items() if key == group and dst != local):
        return None
    local_routes = rule_routes[group, local, local]
    if len(local_routes) != 1 or local_routes[0].nodes != (local,) or local_routes[0].links:
        return None
    route = local_routes[0]
    static, _, _ = rule_features[group, route]
    try:
        choice = program.choose(*group, ((route, static),))
    except ValueError:
        # Keep errors at the point this candidate is actually evaluated.
        return None
    if choice is None:
        return None
    return choice[0], "replicate", route


class InfeasiblePlan(RuntimeError):
    """A policy has no safe decision under the fixed capacities or routes."""


@dataclass
class _Block:
    spec: object
    owner: str
    born_ns: int
    copies: dict[str, int] = field(default_factory=dict)
    refs: int = 1
    reads: int = 0
    last_use: int = 0


@dataclass(frozen=True)
class _Move:
    block: _Block
    route: object
    slots: tuple[int, ...]
    mode: str
    rule: int | None
    pressure_event: str | None = None


class _Pool:
    def __init__(self, torch, node, device, block_rows, head_dim, *, pin_host):
        size = block_rows * head_dim * 2 * 4
        self.capacity = node.capacity_bytes // size
        self.tensor = torch.empty((self.capacity, block_rows, 2, head_dim), dtype=torch.float32,
                                  device=device, pin_memory=pin_host and node.kind == "host")
        self.free = list(range(self.capacity))
        self.used = set()
        self.peak = 0

    def take(self):
        if not self.free:
            raise InfeasiblePlan("managed pool capacity exhausted")
        slot = heapq.heappop(self.free)
        self.used.add(slot)
        self.peak = max(self.peak, len(self.used))
        return slot

    def release(self, slot):
        self.used.remove(slot)
        heapq.heappush(self.free, slot)


class _Runtime:
    def __init__(self, torch, topology, scenario, packed, program, arm, devices, verify, *, pin_host=True, collect_work=False):
        if type(collect_work) is not bool or collect_work and not verify:
            raise ValueError("work counters are available only during verified diagnostics")
        self.torch, self.topology, self.scenario, self.workload = torch, topology, scenario, packed
        self.program, self.arm, self.verify = None if program is None else CompiledProgram(program), arm, verify
        self.work_counts = WorkCounts() if collect_work else None
        self.local = scenario.rank_node(0)
        self.devices = devices
        self.hosts = sorted(node for node in devices if topology.node(node).kind == "host")
        self.routes = {(src, dst): tuple(route for route in topology.routes(src, dst) if self._supported_route(route))
                       for src in self.devices for dst in self.devices}
        if not self.hosts or not any(self.routes[self.local, host] and self.routes[host, self.local] for host in self.hosts):
            raise ValueError("runtime requires a bidirectional pinned-host backing route")
        self.pools = {node: _Pool(torch, topology.node(node), device, packed.block_rows, packed.head_dim, pin_host=pin_host)
                      for node, device in self.devices.items()}
        self.rule_routes, self.rule_features, self.rule_handoffs = {}, {}, {}
        self.rule_nodes = {}
        self.pressure_keys = {}
        self.static_routes = {}
        self.producer_retain = None
        if self.program is not None:
            empty = dict.fromkeys(devices, 0)
            supports_handoff = any(action == "demote" for _, action in self.program.rules)
            for (source, target), paths in self.routes.items():
                statics = {}
                handoff = Route((source, target), ()) if supports_handoff and source != target else None
                candidate_paths = paths + (handoff,) if handoff is not None else paths
                for route in candidate_paths:
                    static = candidate_features(
                        topology, scenario, rank=0, layer=0, block=0, size=packed.block_bytes,
                        target=target, source=source, route=route, usage=empty)
                    static["block_rows"] = packed.block_rows
                    statics[route] = static
                for group, names in self.program.features.items():
                    eligible = []
                    dynamic = names & _DYNAMIC_FEATURES
                    readers = tuple((name, _FEATURE_READERS[name]) for name in sorted(dynamic))
                    for route, static in statics.items():
                        if route == handoff and group[1] != "demote":
                            continue
                        if self.program.possibly_eligible(group, static, _DYNAMIC_FEATURES):
                            subset = {name: static[name] for name in names - dynamic}
                            self.rule_features[group, route] = (subset, readers, topology.node(target).capacity_bytes)
                            if route == handoff:
                                self.rule_handoffs[group, source, target] = route
                            else:
                                eligible.append(route)
                    self.rule_routes[group, source, target] = tuple(eligible)
            self.rule_nodes = {group: frozenset(node for node in devices if self.rule_routes[group, node, node])
                               for group in self.program.rules if group[1] in ("evict", "reclaim")}
            self.pressure_keys = lower_pressure_keys(program, self.program, self.rule_features, _DYNAMIC_FEATURES, packed)
            self.static_routes = lower_static_routes(self.program, self.rule_routes, self.rule_features,
                                                     _DYNAMIC_FEATURES, self.pools)
            self.producer_retain = _lower_producer_retain(
                self.program, self.local, self.rule_routes, self.rule_features)
            if self.work_counts is not None:
                self.work_counts.instrument(self.program)
        self.blocks, self.sessions = {}, {}
        self.versions, self.previous_attention = {}, {}
        self.local_owners = set()
        self.residents = {node: set() for node in devices}
        self.replicas = {node: set() for node in devices}
        self.pool_nodes = frozenset(devices)
        self.retired = set()
        self.timeline, self.access_log, self.semantics = [], [], []
        self.link_bytes, self.group_bytes = Counter(), Counter()
        self.misses, self.reads = Counter(), Counter()
        self.transfer_wall_ns = 0
        self.clock, self.ordinal = 0, 0

    def _supported_route(self, route):
        if any(node not in self.devices for node in route.nodes):
            return False
        for link in route.links:
            src_kind, dst_kind = self.topology.node(link.src).kind, self.topology.node(link.dst).kind
            if src_kind == dst_kind == "gpu":
                src, dst = self.devices[link.src], self.devices[link.dst]
                if link.transport != "cuda_peer" or not self.torch.cuda.can_device_access_peer(src.index, dst.index):
                    return False
            elif src_kind == dst_kind or link.transport != "pinned_host":
                return False
        return True

    def _sync(self, nodes):
        for node in sorted(set(nodes)):
            device = self.devices[node]
            if device.type == "cuda":
                self.torch.cuda.synchronize(device)

    def _tensor(self, state, node):
        return self.pools[node].tensor[state.copies[node]]

    def _source_tensor(self, raw):
        return self.torch.frombuffer(bytearray(raw), dtype=self.torch.float32)

    def _payload(self, tensor):
        return tensor.detach().cpu().contiguous().numpy().tobytes()

    def _features(self, state, route, group):
        static, readers, cap = self.rule_features[group, route]
        if not readers:
            return static
        source, target = route.nodes[0], route.nodes[-1]
        fields = static.copy()
        for name, read in readers:
            fields[name] = read(self, state, source, target, cap)
        return fields

    def _choices(self, state, sources, targets, *, allow_local=False, group=None, blocked=None):
        candidates = []
        for src in sorted(sources):
            for dst in sorted(targets):
                if (src == dst and not allow_local) or (dst in state.copies and dst != src):
                    continue
                routes = self.routes[src, dst] if group is None else self.rule_routes[group, src, dst]
                for route in routes:
                    if all(self.pools[node].free for node in route.nodes[1:]):
                        candidates.append((route, None if group is None else self._features(state, route, group)))
                    elif blocked is not None:
                        blocked.update(node for node in route.nodes[1:] if not self.pools[node].free)
        return candidates

    def _choose_static(self, group, state, sources, targets, *, allow_local=False, blocked=None):
        if self.work_counts is not None:
            self.work_counts.add(group, "static_route_choices", 1)
        return self.static_routes[group].choose(state.copies, sources, targets, allow_local=allow_local, blocked=blocked)

    def _drop(self, state, node, action, rule=None):
        if state.refs and state.owner == node:
            raise RuntimeError("attempted to release a live authoritative KV block")
        self.pools[node].release(state.copies.pop(node))
        self.residents[node].remove(state.spec.id)
        self.replicas[node].discard(state.spec.id)
        if node == self.local:
            self.local_owners.discard(state.spec.id)
        if not state.copies:
            self.retired.discard(state.spec.id)
        self.timeline.append({"step": self.ordinal, "action": action, "block": state.spec.id, "node": node, "rule": rule})

    def _relieve(self, event, *, nodes=None, protected=(), demote_nodes=(), demoted=None):
        reclaim = self.program is None or (event, "reclaim") in self.program.rules
        evict = self.program is None or (event, "evict") in self.program.rules
        demote = bool(demote_nodes) and self.program is not None and (event, "demote") in self.program.rules
        if not reclaim and not evict and not demote:
            return False
        scope = self.pool_nodes if nodes is None else nodes
        protected = set(protected)
        options = defaultdict(list)
        choices = []
        if reclaim:
            group = event, "reclaim"
            eligible = scope if self.program is None else scope & self.rule_nodes[event, "reclaim"]
            scanned = self.retired - protected if eligible else ()
            if self.work_counts is not None:
                self.work_counts.add(group, "rows_scanned", len(scanned))
            if group in self.pressure_keys:
                choice = self._indexed_victim(group, scanned, eligible)
                if choice is not None:
                    choices.append(choice)
            else:
                for block_id in sorted(scanned):
                    state = self.blocks[block_id]
                    for node in sorted(eligible.intersection(state.copies)):
                        options["reclaim"].append((state, node))
        if evict:
            group = event, "evict"
            eligible = scope if self.program is None else scope & self.rule_nodes[event, "evict"]
            candidates = set().union(*(self.replicas[node] for node in eligible)) - protected
            if self.work_counts is not None:
                self.work_counts.add(group, "rows_scanned", len(candidates))
            if group in self.pressure_keys:
                choice = self._indexed_victim(group, candidates, eligible)
                if choice is not None:
                    choices.append(choice)
            else:
                for block_id in sorted(candidates):
                    state = self.blocks[block_id]
                    for node in sorted(eligible.intersection(state.copies)):
                        if node != state.owner:
                            options["evict"].append((state, node))
        if self.program is None:
            if not any(options.values()):
                return self.arm == "manual_topology" and (nodes is None or self.local in nodes) and self._demote_manual(protected)
            action = "reclaim" if options["reclaim"] else "evict"
            state, node = min(options[action], key=lambda row: (self._cache_priority(row[0]), row[0].spec.id, row[1]))
            self._drop(state, node, action)
            return True
        for action, objects in options.items():
            group = event, action
            candidates = [((state, node), self._features(state, route, group)) for state, node in objects
                          for route in self.rule_routes[group, node, node]]
            choice = self.program.choose(event, action, candidates)
            if choice is not None:
                index, (state, node) = choice
                choices.append((index, action, state, node))
        if demote:
            group = event, "demote"
            first_demotion_rule = self.program.rules[group][0][0]
            if not choices or first_demotion_rule < min(choice[0] for choice in choices):
                demoted = set() if demoted is None else demoted
                source_options = self._demotion_sources(group, demote_nodes)
                scanned = set().union(*(self.residents[node] for node in source_options)) - protected - demoted
                if self.work_counts is not None:
                    self.work_counts.add(group, "rows_scanned", len(scanned))
                if group in self.pressure_keys:
                    choice = self._indexed_demotion(group, scanned, source_options)
                    if choice is not None:
                        choices.append(choice)
                else:
                    candidates = self._demotion_choices(group, scanned, source_options)
                    choice = self.program.choose(*group, candidates)
                    if choice is not None:
                        index, (state, route) = choice
                        choices.append((index, "demote", state, route))
        if not choices:
            return False
        rule, action, state, target = min(choices, key=lambda item: item[0])
        if action == "demote":
            self._demote_owner(state, target, rule, event)
            demoted.add(state.spec.id)
        else:
            self._drop(state, target, action, rule)
        return True

    def _indexed_victim(self, group, blocks, eligible):
        key = self.pressure_keys[group]
        action = group[1]
        if self.work_counts is not None:
            self.work_counts.add(group, "indexed_choices", 1)
        best = None
        for block_id in blocks:
            state = self.blocks[block_id]
            nodes = eligible.intersection(state.copies)
            if action == "evict":
                nodes = nodes - {state.owner}
            if nodes:
                order = key.value(self, state), block_id, min(nodes)
                if best is None or order < best:
                    best = order
        return None if best is None else (key.index, action, self.blocks[best[1]], best[2])

    def _demotion_sources(self, group, blocked):
        source_options = {}
        for source in blocked:
            if self.pools[source].free:
                continue
            destinations = []
            for target in sorted(self.pools):
                if source == target:
                    continue
                handoff = self.rule_handoffs.get((group, source, target))
                routes = tuple(route for route in self.rule_routes[group, source, target]
                               if all(self.pools[node].free for node in route.nodes[1:]))
                if handoff is not None or routes:
                    destinations.append((target, handoff, routes))
            if destinations:
                source_options[source] = destinations
        return source_options

    @staticmethod
    def _demotion_routes(state, source_options):
        for target, handoff, routes in source_options.get(state.owner, ()):
            if target in state.copies:
                if handoff is not None:
                    yield handoff
            else:
                yield from routes

    def _demotion_choices(self, group, blocks, source_options):
        candidates = []
        for block_id in sorted(blocks):
            state = self.blocks[block_id]
            if state.refs:
                for route in self._demotion_routes(state, source_options):
                    candidates.append(((state, route), self._features(state, route, group)))
        return candidates

    def _indexed_demotion(self, group, blocks, source_options):
        key = self.pressure_keys[group]
        if self.work_counts is not None:
            self.work_counts.add(group, "indexed_choices", 1)
        best, selected, selected_route = None, None, None
        for block_id in blocks:
            state = self.blocks[block_id]
            if not state.refs:
                continue
            order = key.value(self, state), block_id
            if best is not None and order >= best:
                continue
            route = next(self._demotion_routes(state, source_options), None)
            if route is not None:
                best, selected, selected_route = order, state, route
        return None if best is None else (key.index, "demote", selected, selected_route)

    def _demote_owner(self, state, route, rule, event):
        source, target = route.nodes[0], route.nodes[-1]
        if source != state.owner or not state.refs or self.pools[source].free:
            raise RuntimeError("owner demotion requires a live block in a full source pool")
        if route.links:
            self._execute([self._reserve(state, route, "demote", rule, pressure_event=event)])
        else:
            if target not in state.copies or source == target:
                raise RuntimeError("zero-copy owner handoff requires an existing replica")
            if self.verify and sha256(self._payload(self._tensor(state, target))).hexdigest() != state.spec.digest:
                raise RuntimeError("owner handoff found corrupt replica bytes")
            state.owner = target
            self.replicas[target].remove(state.spec.id)
            self._drop(state, source, "demoted_source", rule)
            self.timeline.append({"step": self.ordinal, "action": "demote_owner_handoff", "block": state.spec.id,
                                  "src": source, "dst": target, "links": [], "rule": rule, "pressure_event": event})
        if not self.pools[source].free:
            raise RuntimeError("owner demotion did not relieve its full source pool")

    def _reserve(self, state, route, mode, rule, *, pressure_event=None):
        if not route.links:
            return None
        if state.refs <= 0 or route.nodes[0] not in state.copies or route.nodes[-1] in state.copies:
            raise RuntimeError("invalid transfer endpoints or ownership")
        return _Move(state, route, tuple(self.pools[node].take() for node in route.nodes[1:]), mode, rule, pressure_event)

    def _execute(self, moves):
        if not moves:
            return
        started = perf_counter_ns()
        for stage in range(max(len(move.route.links) for move in moves)):
            synced = set()
            for move in moves:
                if stage >= len(move.route.links):
                    continue
                link = move.route.links[stage]
                source_slot = move.block.copies[link.src] if stage == 0 else move.slots[stage-1]
                self.pools[link.dst].tensor[move.slots[stage]].copy_(self.pools[link.src].tensor[source_slot], non_blocking=True)
                synced.update((link.src, link.dst))
                self.link_bytes[link.id] += self.workload.block_bytes
                for group in link.bottlenecks:
                    self.group_bytes[group] += self.workload.block_bytes
            # A relay hop can consume its source only after prior DMA completes.
            self._sync(synced)
        self.transfer_wall_ns += perf_counter_ns() - started
        for move in moves:
            src, dst = move.route.nodes[0], move.route.nodes[-1]
            state = move.block
            state.copies[dst] = move.slots[-1]
            self.residents[dst].add(state.spec.id)
            for node, slot in zip(move.route.nodes[1:-1], move.slots[:-1]):
                self.pools[node].release(slot)
            if move.mode in ("place", "migrate", "baseline_place", "demote"):
                if state.owner != src:
                    raise RuntimeError("migration lost its authoritative source")
                state.owner = dst
                if move.mode in ("migrate", "demote"):
                    self._drop(state, src, "demoted_source" if move.mode == "demote" else "migrated_source", move.rule)
                else:
                    self.replicas[src].add(state.spec.id)
            else:
                self.replicas[dst].add(state.spec.id)
            entry = {"step": self.ordinal, "action": move.mode, "block": state.spec.id, "src": src,
                     "dst": dst, "links": list(move.route.ids), "rule": move.rule}
            if move.pressure_event is not None:
                entry["pressure_event"] = move.pressure_event
            self.timeline.append(entry)

    def _best_route(self, candidates):
        return min((route for route, _ in candidates), key=lambda route: (
            transfer_duration_ns(self.topology, route, self.workload.block_bytes), route.ids), default=None)

    def _cache_priority(self, state):
        frequency = self.reads[state.spec.layer, state.spec.index] if self.arm == "manual_topology" else 0
        return frequency, state.last_use

    def _best_backing(self, state):
        targets = tuple(node for node in self.pools if node != self.local)
        routes = [route for route, _ in self._choices(state, (self.local,), targets)
                  if self.routes[route.nodes[-1], self.local]]

        def cost(route):
            size = self.workload.block_bytes
            back = self.routes[route.nodes[-1], self.local]
            return transfer_duration_ns(self.topology, route, size) + min(transfer_duration_ns(self.topology, r, size) for r in back)

        return min(routes, key=lambda route: (cost(route), route.ids), default=None)

    def _demote_manual(self, protected):
        states = (self.blocks[pid] for pid in self.local_owners if pid not in protected)
        for state in sorted(states, key=lambda item: (self._cache_priority(item), item.spec.id)):
            route = self._best_backing(state)
            if route is not None:
                self._execute([self._reserve(state, route, "migrate", None)])
                return True
        return False

    def _choose_place(self, state, *, blocked=None):
        if self.arm == "generated":
            group = "allocate", "place"
            if group in self.static_routes:
                return self._choose_static(group, state, (self.local,), self.pools, allow_local=True, blocked=blocked)
            choices = self._choices(state, (self.local,), self.pools, allow_local=True, group=group, blocked=blocked)
            return self.program.choose("allocate", "place", choices)
        if self.arm == "native_lru":
            route = self._best_route(self._choices(state, (self.local,), self.hosts))
        else:
            owner_limit = max(0, self.pools[self.local].capacity - self.workload.top_k - 2)
            hot = state.spec.valid_rows < self.workload.block_rows or self.reads[state.spec.layer, state.spec.index] >= 2
            if hot and len(self.local_owners) < owner_limit:
                route = self.routes[self.local, self.local][0]
            else:
                route = self._best_backing(state)
        return None if route is None else (None, route)

    def _place_batch(self, states):
        moves = []
        protected = {state.spec.id for state in states}
        can_demote = self.program is not None and ("allocate", "demote") in self.program.rules
        for state in states:
            demoted = set() if can_demote else None
            while True:
                blocked = set() if can_demote else None
                choice = self._choose_place(state, blocked=blocked)
                if choice is not None:
                    rule, route = choice
                    break
                if moves:
                    self._execute(moves)
                    moves = []
                elif not self._relieve("allocate", protected=protected, demote_nodes=blocked or (), demoted=demoted):
                    raise InfeasiblePlan("no safe owner placement")
            self.timeline.append({"step": self.ordinal, "action": "place_owner", "block": state.spec.id,
                                  "node": route.nodes[-1], "rule": rule})
            mode = "place" if self.program is not None else "baseline_place"
            move = self._reserve(state, route, mode, rule)
            if move is None:
                self.local_owners.add(state.spec.id)
            else:
                moves.append(move)
        self._execute(moves)
        if self.program is not None:
            for state in states:
                temporary = state.owner != self.local and self.local in state.copies
                retained = self._optional("allocate", state, retain_source=temporary)
                if temporary and not retained:
                    self._drop(state, self.local, "producer_release")

    def _make_room(self, protected):
        demoted = None
        while not self.pools[self.local].free:
            if demoted is None and self.program is not None and ("allocate", "demote") in self.program.rules:
                demoted = set()
            if not self._relieve("allocate", nodes={self.local}, protected=protected,
                                 demote_nodes=(self.local,), demoted=demoted):
                raise InfeasiblePlan("no safe capacity for the compute producer")

    def _movement(self, event, state, *, demand=False, retain_source=False, blocked=None):
        choices = []
        for action in ("replicate", "migrate", "prefetch"):
            group = event, action
            if group not in self.program.rules:
                continue
            sources = (state.owner,) if action == "migrate" else state.copies
            targets = (self.local,) if demand or action == "prefetch" else self.pools
            if group in self.static_routes:
                selected = self._choose_static(group, state, sources, targets, blocked=blocked)
            else:
                candidates = self._choices(state, sources, targets, group=group, blocked=blocked)
                if retain_source and action == "replicate":
                    for route in self.rule_routes[group, self.local, self.local]:
                        candidates.append((route, self._features(state, route, group)))
                selected = self.program.choose(event, action, candidates)
            if selected is not None:
                rule, route = selected
                choices.append((rule, action, route))
        return min(choices, key=lambda row: row[0], default=None)

    def _optional(self, event, state, *, retain_source=False):
        # _choices excludes ordinary transfers to an already resident target.
        if event == "allocate" and retain_source and self.producer_retain is not None and self.local in state.copies:
            if self.work_counts is not None:
                self.work_counts.add(("allocate", "replicate"), "static_producer_retain_choices", 1)
            choice = self.producer_retain
        else:
            choice = self._movement(event, state, retain_source=retain_source)
        if choice is not None:
            rule, mode, route = choice
            if not route.links:
                self.timeline.append({"step": self.ordinal, "action": "retain_producer_replica", "block": state.spec.id,
                                      "node": self.local, "rule": rule})
                return True
            self._execute([self._reserve(state, route, mode, rule)])
        return False

    def _fetch(self, selected):
        protected = {state.spec.id for state in selected}
        moves = []
        can_demote = self.program is not None and ("access", "demote") in self.program.rules
        for state in selected:
            if self.local in state.copies:
                continue
            self.misses[state.spec.layer, state.spec.index] += 1
            demoted = set() if can_demote else None
            while True:
                blocked = set() if can_demote else None
                if self.program is None:
                    route = self._best_route(self._choices(state, state.copies, (self.local,)))
                    choice = None if route is None else (None, "demand_replica", route)
                else:
                    choice = self._movement("access", state, demand=True, blocked=blocked)
                if choice is not None:
                    rule, mode, route = choice
                    moves.append(self._reserve(state, route, mode, rule))
                    break
                if self._relieve("access", protected=protected, demote_nodes=blocked or (), demoted=demoted):
                    continue
                if not moves:
                    raise InfeasiblePlan("no safe transfer to the attention rank")
                self._execute(moves)
                moves = []
        self._execute(moves)

    def _release_ref(self, state):
        state.refs -= 1
        if state.refs < 0:
            raise RuntimeError("logical KV reference underflow")
        if state.refs == 0:
            self.retired.add(state.spec.id)
            for node in state.copies:
                self.replicas[node].discard(state.spec.id)
            if self.program is None:
                for node in list(state.copies):
                    self._drop(state, node, "reclaim")
            else:
                while self._relieve("release"):
                    pass

    def _allocate(self, spec):
        if spec.id in self.blocks:
            raise RuntimeError("logical block was produced more than once")
        state = _Block(spec, self.local, self.clock, {self.local: self.pools[self.local].take()}, last_use=self.ordinal)
        self.blocks[spec.id] = state
        self.residents[self.local].add(spec.id)
        return state

    def _write_tail(self, session, mutation):
        spec = self.workload.blocks[mutation.block]
        old = self.blocks[mutation.replaces]
        self._fetch([old])
        self._make_room({old.spec.id})
        state = self._allocate(spec)
        dest = self._tensor(state, self.local)
        preserved = mutation.preserved_rows
        suffix = self._source_tensor(spec.payload[preserved*self.workload.row_bytes:]).view(dest[preserved:].shape)
        dest[preserved:].copy_(suffix, non_blocking=True)
        source = self._tensor(old, self.local)
        if source.data_ptr() == dest.data_ptr():
            raise RuntimeError("versioned KV writes require a disjoint destination")
        dest[:preserved].copy_(source[:preserved])
        self._sync((self.local,))
        if self.verify and sha256(self._payload(dest)).hexdigest() != spec.digest:
            raise RuntimeError("producer or physical COW bytes disagree with immutable KV")
        self.versions[spec.id] = (old.spec.id, preserved)
        if old.refs > 1:
            if self.verify and sha256(self._payload(self._tensor(old, self.local))).hexdigest() != old.spec.digest:
                raise RuntimeError("COW changed another branch's physical bytes")
            self.semantics.append({"step": self.ordinal, "action": "copy_on_write", "old": old.spec.id,
                                   "new": spec.id, "session": session, "parent_owners": old.refs,
                                   "preserved_rows": preserved, "old_sha256": old.spec.digest, "new_sha256": spec.digest})
        self._release_ref(old)
        self._place_batch([state])

    def _publish_batch(self, states):
        sources = []
        for state in states:
            dest = self._tensor(state, self.local)
            sources.append(self._source_tensor(state.spec.payload).view(dest.shape))
            dest.copy_(sources[-1], non_blocking=True)
        self._sync((self.local,))
        if self.verify and any(sha256(self._payload(self._tensor(state, self.local))).hexdigest() != state.spec.digest
                               for state in states):
            raise RuntimeError("batch producer bytes disagree with immutable KV")
        self._place_batch(states)

    def _writes(self, step):
        pages = self.sessions[step.session]
        batch = []
        for mutation in step.writes:
            if batch and (mutation.replaces is not None or not self.pools[self.local].free or len(batch) >= 32):
                self._publish_batch(batch)
                batch = []
            if mutation.replaces is not None:
                pages.remove(mutation.replaces)
                self._write_tail(step.session, mutation)
            else:
                self._make_room(())
                batch.append(self._allocate(self.workload.blocks[mutation.block]))
            pages.add(mutation.block)
        if batch:
            self._publish_batch(batch)
        if self.program is not None:
            for pid in step.hints:
                self._optional("hint", self.blocks[pid])

    def _retained_rows(self, previous, current):
        retained = 0
        for old_id, new_id in zip(previous, current):
            same_rows = self.blocks[old_id].spec.valid_rows
            while old_id != new_id and new_id in self.versions:
                new_id, copied = self.versions[new_id]
                same_rows = min(same_rows, copied)
            if old_id != new_id:
                break
            retained += same_rows
        return retained

    def _attention(self, session, request):
        torch = self.torch
        device = self.devices[self.local]
        chosen = [self.blocks[pid] for pid in request.selected]
        started = perf_counter_ns()
        self._fetch(chosen)
        for state in chosen:
            state.reads += 1
            state.last_use = self.ordinal
            self.reads[state.spec.layer, state.spec.index] += 1
        slots = torch.tensor([state.copies[self.local] for state in chosen], device=device, dtype=torch.int64)
        payload = self.pools[self.local].tensor.index_select(0, slots)
        key_values = payload.view(-1, 2, self.workload.head_dim)
        query = self._source_tensor(request.query).to(device)
        valid = torch.tensor([state.spec.valid_rows for state in chosen], device=device)
        mask = (torch.arange(self.workload.block_rows, device=device)[None, :] < valid[:, None]).flatten()
        scores = key_values[:, 0].mv(query).mul_(self.workload.head_dim ** -0.5).masked_fill_(~mask, -torch.inf)
        output = torch.softmax(scores, dim=0) @ key_values[:, 1]
        self._sync((self.local,))
        access_ns = perf_counter_ns() - started
        key = session, request.layer
        previous = self.previous_attention.get(key, ())
        old_rows = sum(self.blocks[pid].spec.valid_rows for pid in previous)
        retained_rows = self._retained_rows(previous, request.blocks)
        if previous and retained_rows != old_rows:
            raise RuntimeError("session update lost rows from its retained prefix")
        self.previous_attention[key] = request.blocks
        observation = {"step": self.ordinal, "session": session, "layer": request.layer,
                       "selected": list(request.selected), "access_ns": access_ns, "retained_rows": retained_rows,
                       "context_rows": sum(self.blocks[pid].spec.valid_rows for pid in request.blocks)}
        if self.verify:
            raw = self._payload(payload)
            if raw != b"".join(state.spec.payload for state in chosen):
                raise RuntimeError("CUDA attention consumed incorrect ordered selected KV bytes")
            dense = self._source_tensor(b"".join(self.workload.blocks[pid].payload for pid in request.blocks))
            dense = dense.view(-1, 2, self.workload.head_dim)
            selected = set(request.selected)
            dense_mask = torch.tensor([pid in selected and offset < self.workload.blocks[pid].valid_rows
                                       for pid in request.blocks for offset in range(self.workload.block_rows)])
            full = (dense[:, 0].mv(self._source_tensor(request.query)) * self.workload.head_dim ** -0.5)
            expected = torch.softmax(full.masked_fill(~dense_mask, -torch.inf), dim=0) @ dense[:, 1]
            actual = output.cpu()
            torch.testing.assert_close(actual, expected, rtol=1e-4, atol=1e-5)
            observation.update(selected_sha256=sha256(raw).hexdigest(), output_sha256=sha256(self._payload(actual)).hexdigest(),
                               max_abs_error=float((actual-expected).abs().max().item()))
        self.access_log.append(observation)
        if self.program is not None:
            for state in chosen:
                self._optional("access", state)

    def _check_state(self):
        ownership = Counter(pid for blocks in self.sessions.values() for pid in blocks)
        occupied = defaultdict(set)
        resident, replicas, retired = defaultdict(set), defaultdict(set), set()
        for pid, state in self.blocks.items():
            if state.refs != ownership[pid] or (state.refs and state.owner not in state.copies):
                raise RuntimeError("physical owner disagrees with the external logical references")
            for node, slot in state.copies.items():
                if slot in occupied[node]:
                    raise RuntimeError("independent KV versions alias a physical slot")
                occupied[node].add(slot)
                resident[node].add(pid)
                if state.refs and node != state.owner:
                    replicas[node].add(pid)
            if state.refs == 0 and state.copies:
                retired.add(pid)
        if retired != self.retired or any(occupied[node] != pool.used or resident[node] != self.residents[node]
                                          or replicas[node] != self.replicas[node]
                                          for node, pool in self.pools.items()):
            raise RuntimeError("runtime has unaccounted physical storage")

    def run(self):
        self._sync(self.devices)
        started = perf_counter_ns()
        for self.ordinal, step in enumerate(self.workload.steps):
            self.clock = step.at_ns
            if step.op == "open":
                self.sessions[step.session] = set()
            elif step.op == "fork":
                parent = self.sessions[step.parent]
                self.sessions[step.session] = parent.copy()
                for pid in parent:
                    self.blocks[pid].refs += 1
                self.semantics.append({"step": self.ordinal, "action": "fork", "parent": step.parent,
                                       "child": step.session, "shared_blocks": sorted(parent),
                                       "owners_after": {pid: self.blocks[pid].refs for pid in sorted(parent)}})
            elif step.op == "close":
                for pid in sorted(self.sessions.pop(step.session)):
                    self._release_ref(self.blocks[pid])
                if self.program is not None:
                    while self._relieve("idle"):
                        pass
            else:
                if step.call_id is not None:
                    self.semantics.append({"step": self.ordinal, "session": step.session,
                                           "action": step.segment_role, "call_id": step.call_id})
                self._writes(step)
                for request in step.attention:
                    if set(request.blocks) != self.sessions[step.session]:
                        # Compare just this layer when a session owns multiple layers.
                        layer_blocks = {pid for pid in self.sessions[step.session] if self.blocks[pid].spec.layer == request.layer}
                        if set(request.blocks) != layer_blocks:
                            raise RuntimeError("attention does not bind its exact live KV versions")
                    self._attention(step.session, request)
            if self.verify:
                self._check_state()
        self._sync(self.devices)
        if self.sessions or any(state.refs for state in self.blocks.values()):
            raise RuntimeError("runtime ended with live KV references")
        if self.workload.kind == "rollout_cow" and len({row["session"] for row in self.semantics
                                                       if row["action"] == "copy_on_write"}) < 2:
            raise RuntimeError("both rollout branches must execute physical copy-on-write")
        residual = {node: len(pool.used)*self.workload.block_bytes for node, pool in self.pools.items()}
        for state in self.blocks.values():
            for node in list(state.copies):
                self._drop(state, node, "teardown")
        if any(pool.used for pool in self.pools.values()):
            raise RuntimeError("runtime ended with outstanding reservations")
        elapsed_ns = perf_counter_ns() - started
        result = {"status": "verified" if self.verify else "timed", "elapsed_ns": elapsed_ns,
                "transfer_wall_ns": self.transfer_wall_ns, "accesses": self.access_log, "actions": self.timeline,
                "semantics": self.semantics, "link_bytes": dict(self.link_bytes), "shared_bottleneck_bytes": dict(self.group_bytes),
                "managed_peak_bytes_by_node": {node: pool.peak*self.workload.block_bytes for node, pool in self.pools.items()},
                "managed_reserved_bytes_by_node": {node: pool.capacity*self.workload.block_bytes for node, pool in self.pools.items()},
                "residual_before_teardown_bytes_by_node": residual, "live_references_at_end": 0,
                "page_activity": [{"layer": layer, "page": index, "attention_reads": self.reads[layer, index],
                                   "fetch_misses": self.misses[layer, index]}
                                  for layer, index in sorted(self.reads.keys() | self.misses.keys())],
                "total_fetch_misses": self.misses.total()}
        if self.work_counts is not None:
            result["work_counts"] = self.work_counts.report()
        return result


def run_trial(topology, scenario, packed, program=None, *, arm="generated", device_map, verify=True, collect_work=False):
    """Use fresh capacity-bounded pools; a diagnostic trial is distinct from timing."""
    import torch

    if arm not in ARMS or (arm == "generated") != (program is not None):
        raise ValueError("arm and generated MemoryProgram disagree")
    if scenario.to_dict() != packed.bind_scenario(scenario.target_sha256, scenario.rank_node(0)).to_dict():
        raise ValueError("packed KV workload or attention differs from its frozen scenario")
    scenario.validate_topology(topology)
    if program is not None:
        program.validate_actions(SUPPORTED_ACTIONS)
        program.validate_features(SUPPORTED_FEATURES)
    expected = {node.id for node in topology.nodes if node.kind == "gpu"}
    if (type(device_map) is not dict or device_map.keys() != expected or len(expected) < 2
            or any(type(index) is not int or not 0 <= index < torch.cuda.device_count() for index in device_map.values())
            or len(set(device_map.values())) != len(device_map)):
        raise ValueError("runtime requires distinct visible CUDA devices bound to all declared GPU nodes")
    before = {node: torch.cuda.memory_allocated(index) for node, index in device_map.items()}
    for index in device_map.values():
        torch.cuda.reset_peak_memory_stats(index)
    devices = {node: torch.device("cuda", index) for node, index in device_map.items()}
    devices.update({node.id: torch.device("cpu") for node in topology.nodes if node.kind == "host"})
    with torch.inference_mode():
        runtime = _Runtime(torch, topology, scenario, packed, program, arm, devices, verify, collect_work=collect_work)
        result = runtime.run()
    peaks = {node: torch.cuda.max_memory_allocated(index)-before[node] for node, index in device_map.items()}
    selected = [{key: access[key] for key in ("step", "layer", "selected", "selected_sha256", "output_sha256")}
                for access in result["accesses"]] if verify else None
    return {"schema": 1, "scope": "cuda_text_derived_fixed_block_sparse", "arm": arm,
            "schedule_execution": "ordered_logical_clock_without_wall_waits", "fixture_kind": packed.kind,
            "topology_sha256": topology.fingerprint(), "scenario_sha256": scenario.fingerprint(),
            "workload_sha256": packed.workload_sha256, "schedule_sha256": packed.schedule_sha256,
            "program_sha256": None if program is None else program.fingerprint(),
            "cuda_peak_allocated_bytes_delta_by_node": peaks,
            "selected_and_output_sha256": None if selected is None else fingerprint(selected), **result}
