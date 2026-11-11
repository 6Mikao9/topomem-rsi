"""Directed hardware resources and transfer routes for KV memory programs."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
import json
import re


NODE_KINDS = frozenset({"gpu", "host", "fabric"})
MAX_ROUTE_HOPS = 4
MAX_ROUTES = 64
_IDENTIFIER = re.compile(r"[a-z][a-z0-9_.-]{0,47}")
_SHA256 = re.compile(r"[0-9a-f]{64}")


def _fields(data, required, optional=()):
    if type(data) is not dict or not set(required) <= data.keys() or data.keys() - set(required) - set(optional):
        raise ValueError(f"expected fields {', '.join(required)}")


def _id(value):
    if type(value) is not str or not _IDENTIFIER.fullmatch(value):
        raise ValueError("expected a logical hardware identifier")
    return value


def _digest(value):
    if type(value) is not str or not _SHA256.fullmatch(value):
        raise ValueError("expected a SHA-256 digest")
    return value


def _integer(value, label, minimum=0):
    if type(value) is not int or not minimum <= value < 2**63:
        raise ValueError(f"invalid {label}")
    return value


def _list(value, label, *, limit):
    if type(value) is not list or len(value) > limit:
        raise ValueError(f"invalid {label}")
    return value


def fingerprint(data):
    raw = json.dumps(data, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode("utf-8")
    return sha256(raw).hexdigest()


def _optional_evidence(data):
    return _digest(data["evidence_sha256"]) if "evidence_sha256" in data else None


def _with_evidence(record, evidence):
    return record if evidence is None else {**record, "evidence_sha256": evidence}


@dataclass(frozen=True)
class Node:
    id: str
    kind: str
    capacity_bytes: int
    physical_bytes: int | None = None
    evidence_sha256: str | None = None

    @classmethod
    def from_dict(cls, data):
        _fields(data, ("id", "kind", "capacity_bytes"), ("physical_bytes", "evidence_sha256"))
        kind = data["kind"]
        if type(kind) is not str or kind not in NODE_KINDS:
            raise ValueError("unknown hardware node kind")
        cap = _integer(data["capacity_bytes"], "node capacity")
        physical = _integer(data["physical_bytes"], "physical capacity", 1) if "physical_bytes" in data else None
        if (kind == "fabric" and (cap != 0 or physical is not None)) or (physical is not None and cap > physical):
            raise ValueError("node budget exceeds its physical capacity or fabric cannot store KV")
        return cls(_id(data["id"]), kind, cap, physical, _optional_evidence(data))

    def to_dict(self):
        data = {"id": self.id, "kind": self.kind, "capacity_bytes": self.capacity_bytes}
        if self.physical_bytes is not None:
            data["physical_bytes"] = self.physical_bytes
        return _with_evidence(data, self.evidence_sha256)


@dataclass(frozen=True)
class Bottleneck:
    id: str
    bandwidth_bytes_per_s: int
    evidence_sha256: str | None = None

    @classmethod
    def from_dict(cls, data):
        _fields(data, ("id", "bandwidth_bytes_per_s"), ("evidence_sha256",))
        return cls(_id(data["id"]), _integer(data["bandwidth_bytes_per_s"], "shared bandwidth", 1), _optional_evidence(data))

    def to_dict(self):
        return _with_evidence({"id": self.id, "bandwidth_bytes_per_s": self.bandwidth_bytes_per_s}, self.evidence_sha256)


@dataclass(frozen=True)
class Link:
    id: str
    src: str
    dst: str
    bandwidth_bytes_per_s: int
    latency_ns: int
    bottlenecks: tuple[str, ...]
    transport: str
    evidence_sha256: str | None = None

    @classmethod
    def from_dict(cls, data):
        _fields(data, ("id", "src", "dst", "bandwidth_bytes_per_s", "latency_ns", "bottlenecks", "transport"), ("evidence_sha256",))
        groups = tuple(sorted(_id(name) for name in _list(data["bottlenecks"], "link bottlenecks", limit=16)))
        if len(set(groups)) != len(groups):
            raise ValueError("duplicate shared bottleneck on a directed link")
        return cls(_id(data["id"]), _id(data["src"]), _id(data["dst"]),
                   _integer(data["bandwidth_bytes_per_s"], "link bandwidth", 1),
                   _integer(data["latency_ns"], "link latency"), groups,
                   _id(data["transport"]), _optional_evidence(data))

    def to_dict(self):
        return _with_evidence({"id": self.id, "src": self.src, "dst": self.dst,
                               "bandwidth_bytes_per_s": self.bandwidth_bytes_per_s, "latency_ns": self.latency_ns,
                               "bottlenecks": list(self.bottlenecks), "transport": self.transport}, self.evidence_sha256)


@dataclass(frozen=True)
class Route:
    nodes: tuple[str, ...]
    links: tuple[Link, ...]

    @property
    def ids(self):
        return tuple(link.id for link in self.links)


@dataclass(frozen=True)
class HardwareTopology:
    nodes: tuple[Node, ...]
    links: tuple[Link, ...]
    bottlenecks: tuple[Bottleneck, ...]

    @classmethod
    def from_dict(cls, data):
        _fields(data, ("schema", "nodes", "links", "bottlenecks"))
        if type(data["schema"]) is not int or data["schema"] != 1:
            raise ValueError("unsupported hardware topology schema")
        nodes = tuple(sorted((Node.from_dict(n) for n in _list(data["nodes"], "nodes", limit=32)), key=lambda n: n.id))
        links = tuple(sorted((Link.from_dict(n) for n in _list(data["links"], "links", limit=128)), key=lambda n: n.id))
        groups = tuple(sorted((Bottleneck.from_dict(n) for n in _list(data["bottlenecks"], "bottlenecks", limit=64)), key=lambda n: n.id))
        for records in (nodes, links, groups):
            if len({record.id for record in records}) != len(records):
                raise ValueError("duplicate hardware resource ID")
        node_ids, group_ids = {n.id for n in nodes}, {g.id for g in groups}
        if not any(n.kind == "gpu" for n in nodes):
            raise ValueError("topology needs a GPU")
        if any(link.src == link.dst or not {link.src, link.dst} <= node_ids or not set(link.bottlenecks) <= group_ids
               for link in links):
            raise ValueError("directed link refers to invalid hardware resources")
        return cls(nodes, links, groups)

    def to_dict(self):
        return {"schema": 1, "nodes": [n.to_dict() for n in self.nodes], "links": [l.to_dict() for l in self.links],
                "bottlenecks": [g.to_dict() for g in self.bottlenecks]}

    def fingerprint(self):
        return fingerprint(self.to_dict())

    def node(self, name):
        for node in self.nodes:
            if node.id == name:
                return node
        raise ValueError(f"unknown hardware node {name!r}")

    def link_bandwidth(self, link):
        return min([link.bandwidth_bytes_per_s, *(group.bandwidth_bytes_per_s for group in self.bottlenecks
                                                   if group.id in link.bottlenecks)])

    def routes(self, src, dst, *, transports=None):
        self.node(src)
        self.node(dst)
        found = []

        def visit(nodes, links):
            if nodes[-1] == dst:
                found.append(Route(nodes, links))
                if len(found) > MAX_ROUTES:
                    raise ValueError("hardware route count exceeds the bounded reference executor")
            elif len(links) < MAX_ROUTE_HOPS:
                for link in self.links:
                    if link.src == nodes[-1] and link.dst not in nodes and (transports is None or link.transport in transports):
                        visit((*nodes, link.dst), (*links, link))

        visit((src,), ())
        return tuple(found)
