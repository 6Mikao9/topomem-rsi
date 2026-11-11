"""Compile fixed semantic workloads into immutable block attention fixtures."""

from __future__ import annotations

from array import array
from dataclasses import dataclass
from hashlib import sha256, shake_256
import sys

from rsi_memory.memory_program import MemoryScenario
from rsi_memory.scenarios import ScenarioWorkload
from rsi_memory.topology import fingerprint


@dataclass(frozen=True)
class PackedBlock:
    id: int
    layer: int
    index: int
    valid_rows: int
    payload: bytes
    digest: str


@dataclass(frozen=True)
class Mutation:
    block: int
    replaces: int | None
    preserved_rows: int


@dataclass(frozen=True)
class Attention:
    layer: int
    blocks: tuple[int, ...]
    selected: tuple[int, ...]
    query: bytes


@dataclass(frozen=True)
class Step:
    op: str
    session: str
    at_ns: int
    parent: str | None = None
    writes: tuple[Mutation, ...] = ()
    attention: tuple[Attention, ...] = ()
    hints: tuple[int, ...] = ()
    segment_role: str | None = None
    call_id: str | None = None
    max_new_tokens: int | None = None
    decode_index: int | None = None


@dataclass(frozen=True)
class PackedWorkload:
    kind: str
    block_rows: int
    head_dim: int
    top_k: int
    layers: int
    text_bytes_per_row: int
    blocks: tuple[PackedBlock, ...]
    steps: tuple[Step, ...]
    source_workload_sha256: str
    workload_sha256: str
    schedule_sha256: str

    @property
    def row_bytes(self):
        return self.head_dim * 2 * 4

    @property
    def block_bytes(self):
        return self.row_bytes * self.block_rows

    def bind_scenario(self, target_sha256, compute_node):
        return MemoryScenario.from_dict({
            "schema": 1, "rank_placement": [{"rank": 0, "node": compute_node}],
            "attention": {"family": "torch_fixed_block_sparse", "selector": "sealed_ordered_fixture",
                          "selection_unit": "block", "top_k": self.top_k, "kv_row_bytes": self.row_bytes},
            "target_sha256": target_sha256, "workload_sha256": self.workload_sha256,
            "schedule_sha256": self.schedule_sha256,
        })


@dataclass
class _Session:
    blocks: list[list[int]]
    byte_rows: list[int]
    row_count: int = 0
    previous_selected: tuple[int, ...] = ()


def _floats(domain, width):
    raw = shake_256(domain).digest(width)
    values = array("f", ((value - 127.5) / 127.5 for value in raw))
    if sys.byteorder != "little":
        values.byteswap()
    return values.tobytes()


def _selection(indices, *, count, top_k, seed, step):
    selected = list(dict.fromkeys(indices))[:top_k]
    remainder = sorted((i for i in range(count) if i not in selected),
                       key=lambda i: sha256(f"select:{seed}:{step}:{i}".encode()).digest())
    return selected + remainder[:max(0, top_k - len(selected))]


def compile_workload(workload, *, block_rows=16, head_dim=32, top_k=4, layers=1, text_bytes_per_row=2, seed=0):
    if not isinstance(workload, ScenarioWorkload):
        raise ValueError("expected a validated semantic workload")
    for value, low, high in ((block_rows, 2, 256), (head_dim, 4, 256), (top_k, 1, 64), (layers, 1, 32),
                             (text_bytes_per_row, 1, 32), (seed, 0, 2**63-1)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("invalid block fixture geometry or seed")
    data = workload.to_dict()
    segments = {segment["id"]: segment for segment in data["segments"]}
    sessions, blocks, steps = {}, [], []
    forked_tails, cow_children = {}, set()
    row_bytes = head_dim * 2 * 4

    def publish(layer, index, valid, payload):
        pid = len(blocks)
        raw = payload.ljust(block_rows * row_bytes, b"\x00")
        blocks.append(PackedBlock(pid, layer, index, valid, raw, sha256(raw).hexdigest()))
        return pid

    def append_rows(session_id, chunks):
        session = sessions[session_id]
        writes = []
        for layer, layer_blocks in enumerate(session.blocks):
            rows = [_floats(f"kv:{seed}:{layer}:{session.row_count+i}:".encode() + chunk, head_dim*2)
                    for i, chunk in enumerate(chunks)]
            tail = session.row_count % block_rows
            if tail:
                old_id = layer_blocks[-1]
                appended = min(block_rows-tail, len(rows))
                prefix = blocks[old_id].payload[:tail*row_bytes]
                new_id = publish(layer, len(layer_blocks)-1, tail+appended, prefix+b"".join(rows[:appended]))
                writes.append(Mutation(new_id, old_id, tail))
                if old_id in forked_tails.get(session_id, ()):
                    cow_children.add(session_id)
                layer_blocks[-1] = new_id
                rows = rows[appended:]
            for offset in range(0, len(rows), block_rows):
                piece = rows[offset:offset+block_rows]
                pid = publish(layer, len(layer_blocks), len(piece), b"".join(piece))
                writes.append(Mutation(pid, None, 0))
                layer_blocks.append(pid)
        session.row_count += len(chunks)
        return tuple(writes)

    def prior_hints(session):
        live = {pid for layer_blocks in session.blocks for pid in layer_blocks}
        return tuple(pid for pid in session.previous_selected if pid in live)

    for ordinal, event in enumerate(data["events"]):
        action, session_id, at_ns = event["op"], event["session"], event["at_ns"]
        if action == "open":
            sessions[session_id] = _Session([[] for _ in range(layers)], [])
            steps.append(Step(action, session_id, at_ns))
        elif action == "fork":
            parent = sessions[event["parent"]]
            sessions[session_id] = _Session([pages.copy() for pages in parent.blocks], parent.byte_rows.copy(),
                                            parent.row_count, parent.previous_selected)
            forked_tails[session_id] = {pages[-1] for pages in parent.blocks}
            steps.append(Step(action, session_id, at_ns, parent=event["parent"]))
        elif action == "close":
            del sessions[session_id]
            steps.append(Step(action, session_id, at_ns))
        elif action == "append":
            session, segment = sessions[session_id], segments[event["segment"]]
            text = segment["text"].encode("utf-8")
            chunks = [text[i:i+text_bytes_per_row] for i in range(0, len(text), text_bytes_per_row)]
            for i, chunk in enumerate(chunks):
                session.byte_rows.extend([session.row_count+i] * len(chunk))
            writes = append_rows(session_id, chunks)
            steps.append(Step(action, session_id, at_ns, writes=writes, hints=prior_hints(session),
                              segment_role=segment["role"], call_id=segment.get("call_id")))
        else:
            session = sessions[session_id]
            budget = event["max_new_tokens"]
            if budget > 4096:
                raise ValueError("fixed decode budget exceeds the bounded fixture")
            fixed = tuple(session.byte_rows[pos] // block_rows for pos in event["select_rows"])
            writes = ()
            for token in range(budget):
                count = len(session.blocks[0])
                indices = fixed if token == 0 else (count-1, *fixed[:min(2, top_k-1)])
                selected = _selection(indices, count=count, top_k=top_k, seed=seed, step=f"{ordinal}:{token}")
                requests = tuple(Attention(layer, tuple(ids), tuple(ids[index] for index in selected),
                                           _floats(f"query:{seed}:{ordinal}:{token}:{layer}".encode(), head_dim))
                                 for layer, ids in enumerate(session.blocks))
                steps.append(Step(action if token == 0 else "decode", session_id, at_ns, writes=writes,
                                  attention=requests, hints=prior_hints(session), max_new_tokens=budget, decode_index=token))
                session.previous_selected = tuple(pid for request in requests for pid in request.selected)
                # The continuation is sealed input to the next query, independent of policy and output.
                writes = append_rows(session_id, [f"fixed_decode:{seed}:{session_id}:{ordinal}:{token}".encode()])
            steps.append(Step("commit", session_id, at_ns, writes=writes, decode_index=budget))

    if workload.kind == "rollout_cow" and len(cow_children) < 2:
        raise ValueError("packed rollout geometry must retain a partial tail for actual COW in both branches")
    schedule_sha = fingerprint({"semantic_schedule_sha256": workload.schedule_sha256,
                                "lifecycle": [[step.op, step.session, step.at_ns, step.parent, step.decode_index] for step in steps]})
    fixture = {"semantic_workload_sha256": workload.workload_sha256, "schedule_sha256": schedule_sha,
               "geometry": [block_rows, head_dim, top_k, layers, text_bytes_per_row, seed],
               "blocks": [[b.layer, b.index, b.valid_rows, b.digest] for b in blocks],
               "writes": [[[w.block, w.replaces, w.preserved_rows] for w in step.writes] for step in steps],
               "attention": [[[r.layer, list(r.blocks), list(r.selected), sha256(r.query).hexdigest()]
                              for r in step.attention] for step in steps], "hints": [step.hints for step in steps]}
    return PackedWorkload(workload.kind, block_rows, head_dim, top_k, layers, text_bytes_per_row, tuple(blocks),
                          tuple(steps), workload.workload_sha256, fingerprint(fixture), schedule_sha)
