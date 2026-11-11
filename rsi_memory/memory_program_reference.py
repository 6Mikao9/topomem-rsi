"""Executable CPU judge for generated KV memory rules and an immutable event trace."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from hashlib import sha256
import heapq

from .memory_program import ACTION_DOMAIN, BASE_FEATURES, candidate_features, transfer_duration_ns
from .topology import _digest, _fields, _id, _integer, _list, fingerprint


@dataclass(frozen=True, order=True)
class KVKey:
    rank: int
    request: str
    generation: int
    layer: int
    block: int

    @classmethod
    def from_dict(cls, data):
        _fields(data, ("rank", "request", "generation", "layer", "block"))
        return cls(_integer(data["rank"], "rank"), _id(data["request"]), _integer(data["generation"], "generation"),
                   _integer(data["layer"], "layer"), _integer(data["block"], "block"))

    def to_dict(self):
        return {"rank": self.rank, "request": self.request, "generation": self.generation,
                "layer": self.layer, "block": self.block}


@dataclass
class _Object:
    key: KVKey
    size: int
    digest: str
    allowed: tuple[str, ...]
    owner: str
    born_ns: int
    copies: dict[str, bytes] = field(default_factory=dict)
    accesses: int = 0
    retired: bool = False


@dataclass(frozen=True)
class _Transfer:
    id: int
    obj: _Object
    source: str
    target: str
    reservations: tuple[str, ...]
    finish_ns: int
    action: dict
    mode: str


class ReferenceExecutor:
    """Issue bounded plans, then check ownership, reservations, fences and exact payloads."""

    def __init__(self, topology, scenario, program, *, transports=None, copy_payload=None):
        scenario.validate_topology(topology)
        program.validate_features(BASE_FEATURES)
        program.validate_actions({(event, action) for event, actions in ACTION_DOMAIN.items()
                                  for action in actions if action != "demote"})
        self.topology, self.scenario, self.program = topology, scenario, program
        self.transports = None if transports is None else frozenset(transports)
        self.copy_payload = copy_payload
        self.objects = {}
        self.usage = {node.id: 0 for node in topology.nodes}
        self.peak = self.usage.copy()
        self.pending = {}
        self.readers = Counter()
        self.inflight = Counter()
        self.link_bytes = Counter()
        self.group_bytes = Counter()
        self.resource_until = {}
        self.queue = []
        self.actions = []
        self.accesses = []
        self.now = 0
        self.wait_ns = 0
        self.demand_misses = 0
        self._serial = 0
        self._started = False

    def _next_id(self):
        self._serial += 1
        return self._serial

    def _bytes(self, node, change):
        self.usage[node] += change
        if not 0 <= self.usage[node] <= self.topology.node(node).capacity_bytes:
            raise RuntimeError("physical residency escaped the fixed node capacity")
        self.peak[node] = max(self.peak[node], self.usage[node])

    def _route_schedule(self, route, size):
        ready = self.now
        stages = []
        for link in route.links:
            resources = (("link", link.id), *(("group", group) for group in link.bottlenecks))
            start = max(ready, *(self.resource_until.get(resource, 0) for resource in resources))
            bandwidth = self.topology.link_bandwidth(link)
            ready = start + link.latency_ns + (size * 1_000_000_000 + bandwidth - 1) // bandwidth
            stages.append((link, resources, start, ready))
        return ready, stages

    def _features(self, obj, source, target, route, *, queue_ns=0):
        key = obj.key
        return candidate_features(self.topology, self.scenario, rank=key.rank, layer=key.layer, block=key.block,
                                  size=obj.size, target=target, source=source, route=route, usage=self.usage,
                                  time_ns=self.now, age_ns=self.now-obj.born_ns, accesses=obj.accesses,
                                  replicas=len(obj.copies)-int(obj.owner in obj.copies),
                                  readers=self.readers[key, source], inflight=self.inflight[key, source],
                                  target_has_copy=target in obj.copies, route_queue_ns=queue_ns)

    def _path_candidates(self, obj, targets, *, sources, allow_local=False):
        candidates = []
        for source in sorted(sources):
            for target in sorted(targets):
                if (source == target and not allow_local) or (target in obj.copies and source != target) or (obj.key, target) in self.pending:
                    continue
                for route in self.topology.routes(source, target, transports=self.transports):
                    reservations = tuple(node for node in route.nodes[1:] if self.topology.node(node).kind != "fabric")
                    if any(self.usage[node] + obj.size > self.topology.node(node).capacity_bytes for node in reservations):
                        continue
                    finish, stages = self._route_schedule(route, obj.size)
                    delay = finish - self.now - transfer_duration_ns(self.topology, route, obj.size)
                    candidates.append(((source, target, route, reservations, finish, stages),
                                       self._features(obj, source, target, route, queue_ns=delay)))
        return candidates

    def _copy(self, obj, choice, mode, rule_index):
        source, target, route, reservations, finish, stages = choice
        if source == target:
            return
        if (obj.retired or source not in obj.copies or target in obj.copies or (obj.key, target) in self.pending
                or mode == "migrate" and (obj.owner != source or any(t.obj is obj and t.mode == "migrate" for t in self.pending.values()))):
            raise RuntimeError("invalid ownership or duplicate in-flight transfer")
        for node in reservations:
            self._bytes(node, obj.size)
        for link, resources, _, end in stages:
            for resource in resources:
                self.resource_until[resource] = end
            self.link_bytes[link.id] += obj.size
            for group in link.bottlenecks:
                self.group_bytes[group] += obj.size
        transfer_id = self._next_id()
        action = {"action": "copy", "mode": mode, "key": obj.key.to_dict(), "source": source, "target": target,
                  "bytes": obj.size, "sha256": obj.digest, "transfer_id": transfer_id, "rule_index": rule_index,
                  "route_links": list(route.ids), "staging_nodes": list(reservations[:-1]), "finish_ns": finish,
                  "stages": [{"link": link.id, "start_ns": start, "finish_ns": end} for link, _, start, end in stages]}
        transfer = _Transfer(transfer_id, obj, source, target, reservations, finish, action, mode)
        self.actions.append(action)
        self.pending[obj.key, target] = transfer
        self.inflight[obj.key, source] += 1
        heapq.heappush(self.queue, (finish, transfer_id, "copy", transfer))

    def _removable(self, obj, node):
        return (node in obj.copies and (obj.retired or node != obj.owner)
                and not self.readers[obj.key, node] and not self.inflight[obj.key, node])

    def _drop(self, obj, node, action, rule_index=None):
        if not self._removable(obj, node):
            raise RuntimeError("cannot discard a live owner or an in-use KV copy")
        del obj.copies[node]
        self._bytes(node, -obj.size)
        self.actions.append({"action": action, "key": obj.key.to_dict(), "node": node, "bytes": obj.size,
                             "rule_index": rule_index, "time_ns": self.now})

    def _finish_copy(self, transfer):
        obj, source, target = transfer.obj, transfer.source, transfer.target
        raw = obj.copies[source]
        if self.copy_payload is not None:
            raw = self.copy_payload(transfer.action.copy(), raw)
        if type(raw) is not bytes or len(raw) != obj.size or sha256(raw).hexdigest() != obj.digest:
            raise RuntimeError("copied payload disagrees with the published logical KV bytes")
        self.inflight[obj.key, source] -= 1
        del self.pending[obj.key, target]
        obj.copies[target] = raw
        for node in transfer.reservations[:-1]:
            self._bytes(node, -obj.size)
        if transfer.mode == "migrate":
            if obj.owner != source:
                raise RuntimeError("migration completed after its source lost ownership")
            obj.owner = target
            if self._removable(obj, source):
                self._drop(obj, source, "migration_source_release", transfer.action["rule_index"])

    def _advance(self, until):
        until = max(until, self.now)
        while self.queue and self.queue[0][0] <= until:
            self.now, _, kind, state = heapq.heappop(self.queue)
            if kind == "copy":
                self._finish_copy(state)
            else:
                for key, node in state:
                    self.readers[key, node] -= 1
        self.now = until

    def _relieve(self, event, *, nodes=None, protected=()):
        options = {"evict": [], "reclaim": []}
        for key, obj in sorted(self.objects.items()):
            if key in protected:
                continue
            for node in sorted(obj.copies):
                if (nodes is None or node in nodes) and self._removable(obj, node):
                    route, = self.topology.routes(node, node)
                    action = "reclaim" if obj.retired else "evict"
                    options[action].append(((obj, node), self._features(obj, node, node, route)))
        choices = []
        for action, candidates in options.items():
            result = self.program.choose(event, action, candidates)
            if result is not None:
                index, item = result
                choices.append((index, action, item))
        if not choices:
            return False
        index, action, (obj, node) = min(choices, key=lambda row: row[0])
        self._drop(obj, node, action, index)
        return True

    def _movement(self, event, obj, *, targets=None):
        choices = []
        for action in ("replicate", "migrate", "prefetch"):
            allowed = set(obj.allowed) if targets is None else set(targets)
            if action == "prefetch":
                allowed = {self.scenario.rank_node(obj.key.rank)}
            if action == "migrate" and any(t.obj is obj and t.mode == "migrate" for t in self.pending.values()):
                continue
            sources = (obj.owner,) if action == "migrate" else obj.copies
            result = self.program.choose(event, action, self._path_candidates(obj, allowed, sources=sources))
            if result is not None:
                index, choice = result
                choices.append((index, action, choice))
        if not choices:
            return False
        index, mode, choice = min(choices, key=lambda row: row[0])
        self._copy(obj, choice, mode, index)
        return True

    def _publish(self, event):
        _fields(event, ("event", "time_ns", "key", "payload_hex", "allowed_nodes"))
        key = KVKey.from_dict(event["key"])
        local = self.scenario.rank_node(key.rank)
        if key in self.objects:
            raise ValueError("logical KV generation was published more than once")
        if type(event["payload_hex"]) is not str:
            raise ValueError("reference trace requires actual hexadecimal payload bytes")
        raw = bytes.fromhex(event["payload_hex"])
        if not raw or len(raw) % self.scenario.kv_row_bytes or (self.scenario.selection_unit == "token" and len(raw) != self.scenario.kv_row_bytes):
            raise ValueError("payload width disagrees with the fixed attention selection unit")
        allowed = tuple(sorted(_id(node) for node in _list(event["allowed_nodes"], "backing destinations", limit=32)))
        if (not allowed or len(set(allowed)) != len(allowed)
                or any(self.topology.node(node).kind == "fabric" for node in allowed)):
            raise ValueError("invalid fixed backing destination set")
        while self.usage[local] + len(raw) > self.topology.node(local).capacity_bytes:
            if not self._relieve("allocate", nodes={local}):
                raise MemoryError("producer cannot reserve capacity for a published KV row")
        obj = _Object(key, len(raw), sha256(raw).hexdigest(), allowed, local, self.now, {local: raw})
        self.objects[key] = obj
        self._bytes(local, obj.size)
        self.actions.append({"action": "publish", "key": key.to_dict(), "node": local, "bytes": obj.size, "sha256": obj.digest})
        while True:
            placement = self.program.choose("allocate", "place", self._path_candidates(obj, allowed, sources=(local,), allow_local=True))
            if placement is not None:
                break
            if not self._relieve("allocate", protected={key}):
                raise MemoryError("no capacity-safe generated owner placement")
        index, choice = placement
        self.actions.append({"action": "place_owner", "key": key.to_dict(), "node": choice[1], "rule_index": index})
        self._copy(obj, choice, "migrate", index)
        self._movement("allocate", obj)

    def _get(self, data):
        key = KVKey.from_dict(data)
        if key not in self.objects or self.objects[key].retired:
            raise ValueError("trace used an unknown or retired logical KV generation")
        return self.objects[key]

    def _access(self, event):
        _fields(event, ("event", "time_ns", "rank", "selected", "hold_ns"))
        rank, hold = event["rank"], _integer(event["hold_ns"], "attention hold time")
        local = self.scenario.rank_node(rank)
        selected = []
        for row in _list(event["selected"], "ordered selection", limit=self.scenario.top_k):
            _fields(row, ("key", "sha256"))
            obj, expected = self._get(row["key"]), _digest(row["sha256"])
            if obj.key.rank != rank or obj.digest != expected:
                raise ValueError("ordered selection disagrees with its published rank-local KV identity")
            selected.append(obj)
        protected = {obj.key for obj in selected}
        if (not selected or len(protected) != len(selected)
                or len({(key.request, key.generation, key.layer) for key in protected}) != 1):
            raise ValueError("selection must bind one request generation and layer, with no duplicate KV keys")
        before = self.now
        fences, pins = [], []
        for obj in selected:
            if local not in obj.copies:
                if (obj.key, local) not in self.pending:
                    self.demand_misses += 1
                    while not self._movement("access", obj, targets={local}):
                        if not self._relieve("access", protected=protected):
                            raise MemoryError("no capacity-safe generated transfer for selected KV")
                pending = self.pending[obj.key, local]
                fences.append(pending.id)
                self._advance(pending.finish_ns)
            if sha256(obj.copies[local]).hexdigest() != obj.digest:
                raise RuntimeError("attention selected bytes disagree with the trace")
            obj.accesses += 1
            self.readers[obj.key, local] += 1
            pins.append((obj.key, local))
        self.wait_ns += self.now - before
        until = self.now + hold
        heapq.heappush(self.queue, (until, self._next_id(), "read", pins))
        ordered = [{"key": obj.key.to_dict(), "sha256": obj.digest} for obj in selected]
        self.accesses.append({"rank": rank, "selected": ordered})
        self.actions.append({"action": "consume", "rank": rank, "node": local, "selected": ordered,
                             "wait_for_transfers": fences, "ready_ns": self.now, "read_done_ns": until})

    def run(self, events):
        if self._started:
            raise ValueError("reference executor must be fresh for each trial")
        self._started = True
        if type(events) is not list or not events or fingerprint(events) != self.scenario.workload_sha256:
            raise ValueError("external workload trace does not match the fixed scenario fingerprint")
        previous = 0
        for event in events:
            time_ns = _integer(event["time_ns"], "external timestamp")
            if time_ns < previous:
                raise ValueError("external events are out of order")
            previous = time_ns
            self._advance(time_ns)
            name = event["event"]
            if name == "allocate":
                self._publish(event)
            elif name == "access":
                self._access(event)
            elif name in ("hint", "release"):
                _fields(event, ("event", "time_ns", "keys"))
                objects = [self._get(key) for key in _list(event["keys"], "lifecycle keys", limit=4096)]
                if not objects or len({obj.key for obj in objects}) != len(objects):
                    raise ValueError("lifecycle event requires distinct published KV keys")
                if name == "hint":
                    for obj in objects:
                        self._movement("hint", obj)
                else:
                    for obj in objects:
                        obj.retired = True
                    while self._relieve("release"):
                        pass
            elif name == "idle":
                _fields(event, ("event", "time_ns"))
                while self._relieve("idle"):
                    pass
            else:
                raise ValueError(f"unsupported external event {name!r}")
        if any(not obj.retired for obj in self.objects.values()):
            raise ValueError("trace ended with live logical KV ownership")
        if not self.accesses:
            raise ValueError("trace did not observe any attention selections")
        while self.queue:
            self._advance(self.queue[0][0])
        residual = self.usage.copy()
        for _, obj in sorted(self.objects.items()):
            for node in sorted(obj.copies):
                self._drop(obj, node, "teardown_release")
        if self.pending or any(self.usage.values()) or any(self.readers.values()) or any(self.inflight.values()):
            raise RuntimeError("reference executor ended with unbalanced reservations")
        return {"schema": 1, "scope": "cpu_reference", "status": "verified", "topology_sha256": self.topology.fingerprint(),
                "scenario_sha256": self.scenario.fingerprint(), "program_sha256": self.program.fingerprint(),
                "ordered_selected_sha256": fingerprint(self.accesses), "selected_accesses": len(self.accesses),
                "peak_bytes_by_node": self.peak, "residual_before_teardown_bytes_by_node": residual,
                "link_bytes": dict(self.link_bytes), "shared_bottleneck_bytes": dict(self.group_bytes),
                "simulated_access_wait_ns": self.wait_ns, "demand_misses": self.demand_misses, "actions": self.actions}
