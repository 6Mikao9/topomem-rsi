"""Fixed external arrival groups for one SGLang 0.5.20 TP2/attention-DP2 replay."""

from copy import deepcopy
from dataclasses import dataclass
from datetime import timedelta
from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path
import re
from time import monotonic_ns


EXTERNAL_FIELD = "ingress_rendezvous"
TICK_POLICY = "all_global_forwards_including_local_idle_v1"
SOURCES = {
    "srt/managers/scheduler.py": "439ee96dd785fbb63a9f139b192cec984c252df16a573a436877f0ef9a4ca045",
    "srt/managers/scheduler_components/dp_attn.py": "5e6b1ce9db4b5bf0d255686226e95caf96398d4050965b72295d53dbd5e965cc",
    "srt/managers/scheduler_components/request_receiver.py": "245bbca67d02b18b334a2ddd6f2c2d21a0dd4cd57d86d7082e612fa026b8fa8a",
    "srt/managers/io_struct.py": "cc4e6a88ddfeaa78fc4518990013859168d5bc141905cd66252f9d020aa86cd5",
    "srt/model_executor/forward_batch_info.py": "6da11b9fda45d2589e409de77707562b744be60a6092e5dc05cfd2db7ab41f6d",
}
PHASES = ("warmup", "requests")
REASONS = ("none", "unexpected_work_or_control", "unexpected_rid", "duplicate_rid", "wrong_route",
           "wrong_input_tokens", "native_idle_probe_failed", "group_timeout")
SHA = re.compile(r"[0-9a-f]{64}")
RID = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


def fingerprint(value):
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _integer(value, lower, upper):
    return type(value) is int and lower <= value <= upper


def _fields(value, keys):
    if type(value) is not dict or value.keys() != set(keys.split()):
        raise ValueError("unexpected ingress rendezvous fields")


@dataclass(frozen=True)
class Member:
    rank: int
    rid: str
    input_tokens: int
    input_ids_sha256: str


def checked_contract(value):
    _fields(value, "schema kind global_tick_policy gate_timeout_ms collective_timeout_ms groups")
    if (type(value["schema"]) is not int or value["schema"] != 1 or value["kind"] != "sglang_dp2_ingress_rendezvous"
            or value["global_tick_policy"] != TICK_POLICY or not _integer(value["gate_timeout_ms"], 1, 120000)
            or not _integer(value["collective_timeout_ms"], 1, value["gate_timeout_ms"])
            or type(value["groups"]) is not list or len(value["groups"]) != len(PHASES)):
        raise ValueError("unsupported fixed ingress rendezvous")
    groups = []
    rids = set()
    for phase, group in zip(PHASES, value["groups"], strict=True):
        _fields(group, "phase members")
        if group["phase"] != phase or type(group["members"]) is not list or len(group["members"]) != 2:
            raise ValueError("expected one frozen request per rank and phase")
        members = []
        for rank, row in enumerate(group["members"]):
            _fields(row, "rank rid input_tokens input_ids_sha256")
            rid, digest = row["rid"], row["input_ids_sha256"]
            if (type(row["rank"]) is not int or row["rank"] != rank or not isinstance(rid, str) or not RID.fullmatch(rid)
                    or rid in rids or not _integer(row["input_tokens"], 1, 8192) or not isinstance(digest, str)
                    or not SHA.fullmatch(digest)):
                raise ValueError("invalid, duplicate, or unbound ingress member")
            rids.add(rid)
            members.append(Member(**row))
        groups.append(tuple(members))
    return tuple(groups)


def bind_schedule(schedule, expected_inputs, *, gate_timeout_ms=10000, collective_timeout_ms=5000):
    """Bind the gate to the same committed schedule and token identities as every arm."""
    if type(schedule) is not dict or EXTERNAL_FIELD in schedule:
        raise ValueError("require the ungated fixed schedule for an explicit new binding")
    engine = schedule["engine"]
    required = {"tp_size": 2, "pp_size": 1, "dp_size": 2, "enable_dp_attention": True,
                "disable_overlap_schedule": True, "disable_cuda_graph": True, "disable_radix_cache": True,
                "enable_hisparse": True}
    if any(type(engine.get(key)) is not type(value) or engine[key] != value for key, value in required.items()):
        raise ValueError("ingress rendezvous requires the fixed eager HiSparse TP2/attention-DP2 schedule")
    if engine.get("sleep_on_idle", False) is not False:
        raise ValueError("ingress rendezvous requires nonblocking native receive polling")
    result = deepcopy(schedule)
    groups = []
    for phase in PHASES:
        requests = result[phase]
        if (len(requests) != 2 or {row["routed_dp_rank"] for row in requests} != {0, 1}
                or len({row["arrival_ms"] for row in requests}) != 1):
            raise ValueError("each external phase must contain one simultaneous request per rank")
        members = []
        for row in sorted(requests, key=lambda item: item["routed_dp_rank"]):
            rid = row["rid"]
            tokenized = expected_inputs[rid]
            _fields(tokenized, "input_tokens input_ids_sha256")
            members.append({"rank": row["routed_dp_rank"], "rid": rid, **tokenized})
        groups.append({"phase": phase, "members": members})
    if set(expected_inputs) != {row["rid"] for group in groups for row in group["members"]}:
        raise ValueError("token commitments differ from the two external phases")
    gate = {"schema": 1, "kind": "sglang_dp2_ingress_rendezvous", "global_tick_policy": TICK_POLICY,
            "gate_timeout_ms": gate_timeout_ms, "collective_timeout_ms": collective_timeout_ms, "groups": groups}
    checked_contract(gate)
    result[EXTERNAL_FIELD] = gate
    return result


class GateFailure(RuntimeError):
    """The fixed arrival contract failed before issuing another forward."""


class IngressGate:
    """Pure phase state, using the same gathered vote on both scheduler workers."""

    def __init__(self, contract, rank, *, work_type, controls, teardown_controls=(), now=monotonic_ns, report=lambda event: None):
        groups = checked_contract(contract)
        if type(rank) is not int or rank not in (0, 1):
            raise ValueError("invalid ingress rank")
        self.members = tuple(group[rank] for group in groups)
        self.rank = rank
        self.contract_sha256 = fingerprint(contract)
        self.work_type = work_type
        self.controls = tuple(controls)
        self.teardown_controls = tuple(teardown_controls)
        self.now, self.report = now, report
        self.timeout_ns = contract["gate_timeout_ms"] * 1_000_000
        self.phase = 0
        self.vote_index = 0
        self.rejected = None
        self.held = self.held_ns = None

    def _event(self, name, **fields):
        self.report({"event": name, "rank": self.rank, "contract_sha256": self.contract_sha256, **fields})

    def _reject(self, reason, votes=None):
        self.rejected = reason
        self._event("ingress_gate_rejected", phase_index=self.phase, vote_index=self.vote_index, reason=reason, votes=votes)
        raise GateFailure(reason)

    def _validate(self, request, member):
        if getattr(request, "rid", None) != member.rid:
            return "unexpected_rid"
        if self.held is not None:
            return "duplicate_rid"
        route = getattr(request, "routed_dp_rank", None)
        if type(route) is not int or route != self.rank:
            return "wrong_route"
        ids = getattr(request, "input_ids", None)
        try:
            if ids is None or len(ids) != member.input_tokens:
                return "wrong_input_tokens"
            ids = list(ids)
        except (TypeError, ValueError):
            return "wrong_input_tokens"
        if any(not _integer(token, 0, (1 << 32) - 1) for token in ids) or fingerprint(ids) != member.input_ids_sha256:
            return "wrong_input_tokens"
        return "none"

    def step(self, incoming, native_idle, vote):
        """Retain work, forward read-only control traffic, and jointly release one group."""
        if self.rejected is not None:
            raise GateFailure(self.rejected)
        if self.phase == len(self.members):
            allowed = self.controls + self.teardown_controls
            if any(type(request) not in allowed for request in incoming):
                self._reject("unexpected_message_after_final_release")
            return incoming
        member = self.members[self.phase]
        controls = []
        failure = "none"
        for request in incoming:
            if type(request) in self.controls:
                controls.append(request)
                continue
            error = self._validate(request, member) if type(request) is self.work_type else "unexpected_work_or_control"
            if error == "none":
                self.held, self.held_ns = request, self.now()
            elif failure == "none":
                failure = error
        if self.held is not None and failure == "none":
            if self.now() - self.held_ns >= self.timeout_ns:
                failure = "group_timeout"
        try:
            idle = self.held is not None and bool(native_idle())
        except Exception:
            failure, idle = "native_idle_probe_failed", False
        local = (self.phase, int(self.held is not None), int(idle), REASONS.index(failure))
        self.vote_index += 1
        try:
            rows = tuple(tuple(row) for row in vote(local))
        except Exception:
            self._reject("ingress_collective_failed")
        if (len(rows) != 2 or any(len(row) != 4 or row[0] != self.phase or row[1] not in (0, 1)
                                  or row[2] not in (0, 1) or not _integer(row[3], 0, len(REASONS)-1) for row in rows)
                or rows[self.rank] != local):
            self._reject("ingress_collective_state_mismatch", rows)
        if any(row[3] for row in rows):
            self._reject(REASONS[next(row[3] for row in rows if row[3])], rows)
        if all(row[1] and row[2] for row in rows):
            request, waited = self.held, self.now() - self.held_ns
            self._event("ingress_group_released", phase=PHASES[self.phase], phase_index=self.phase, rid=member.rid,
                        input_tokens=member.input_tokens, input_ids_sha256=member.input_ids_sha256, wait_ns=waited,
                        vote_index=self.vote_index)
            self.phase += 1
            self.held = self.held_ns = None
            controls.append(request)
        return controls


def _collective(torch, group, timeout_ms):
    payload = torch.empty(4, dtype=torch.int64, device="cpu")
    output = torch.empty(8, dtype=torch.int64, device="cpu")
    timeout = timedelta(milliseconds=timeout_ms)

    def gather(row):
        for index, value in enumerate(row):
            payload[index] = value
        work = torch.distributed.all_gather_into_tensor(output, payload, group=group, async_op=True)
        if not work.wait(timeout=timeout):
            raise TimeoutError("ingress vote timed out")
        return output.view(2, 4).tolist()

    return gather


def _verify_live(scheduler, rank, distributed, envs, disaggregation_null):
    ps = scheduler.ps
    group = scheduler.tp_cpu_group
    if ((ps.tp_size, ps.pp_size, ps.pp_rank, ps.attn_tp_size, ps.attn_cp_size, ps.attn_dp_size, ps.tp_rank, ps.attn_dp_rank)
            != (2, 1, 0, 1, 1, 2, rank, rank) or not scheduler.enable_dp_attention or not scheduler.enable_hisparse
            or not scheduler.require_mlp_sync or scheduler.enable_overlap or scheduler._engine_paused
            or scheduler.idle_sleeper is not None or scheduler.recv_skipper is not None
            or scheduler.scripted_scheduler_hook is not None or scheduler.dllm_config is not None
            or scheduler.disaggregation_mode != disaggregation_null or not scheduler.spec_algorithm.is_none()
            or envs.SGLANG_SCHEDULER_SKIP_ALL_GATHER.get() or distributed.get_backend(group) != "gloo"
            or distributed.get_world_size(group) != 2 or distributed.get_rank(group) != rank):
        raise GateFailure("unsupported live scheduler or Gloo rank binding")
    return group


def _wrap_processor(original, gate, initialize_vote):
    vote = None

    @wraps(original)
    def process(scheduler, recv_reqs):
        nonlocal vote
        if vote is None:
            vote = initialize_vote(scheduler)
            gate._event("ingress_gate_installed")
        admitted = gate.step(recv_reqs, scheduler.is_fully_idle, vote)
        return original(scheduler, admitted)

    return process


def verify_ingress_events(rows_by_rank, contract):
    groups = checked_contract(contract)
    digest = fingerprint(contract)
    if len(rows_by_rank) != 2:
        raise ValueError("missing rank ingress records")
    waits, release_votes = [], []
    for rank, rows in enumerate(rows_by_rank):
        if ([row.get("event") for row in rows] != ["ingress_gate_installed", "ingress_group_released", "ingress_group_released"]
                or any(type(row.get("rank")) is not int or row["rank"] != rank or row.get("contract_sha256") != digest for row in rows)
                or not _integer(rows[0].get("pid"), 1, (1 << 31) - 1)
                or any(row.get("pid") != rows[0]["pid"] for row in rows)):
            raise ValueError("worker did not complete both committed ingress groups")
        rank_waits, rank_votes = {}, []
        for index, (phase, group, row) in enumerate(zip(PHASES, groups, rows[1:], strict=True)):
            member = group[rank]
            if ((row.get("phase"), row.get("phase_index"), row.get("rid"), row.get("input_tokens"), row.get("input_ids_sha256"))
                    != (phase, index, member.rid, member.input_tokens, member.input_ids_sha256)
                    or not _integer(row.get("wait_ns"), 0, (1 << 63)-1)
                    or not _integer(row.get("vote_index"), rank_votes[-1]+1 if rank_votes else 1, (1 << 63)-1)):
                raise ValueError("worker ingress release does not match the fixed tokenized input")
            rank_waits[phase] = row["wait_ns"]
            rank_votes.append(row["vote_index"])
        waits.append(rank_waits)
        release_votes.append(rank_votes)
    if rows_by_rank[0][0]["pid"] == rows_by_rank[1][0]["pid"]:
        raise ValueError("two distinct scheduler workers are required")
    if release_votes[0] != release_votes[1]:
        raise ValueError("ranks did not release the frozen groups on the same ingress votes")
    return {"contract_sha256": digest, "completed_groups": list(PHASES), "wait_ns_by_rank": waits,
            "release_vote_indices": release_votes[0]}


def install(contract, rank, directory):
    """Install in the spawned rank before constructing the native Scheduler."""
    import sglang
    import torch
    from sglang.srt.environ import envs
    from sglang.srt.managers import io_struct, scheduler as module

    if sglang.__version__ != "0.5.20":
        raise ValueError("unsupported SGLang version for the ingress rendezvous")
    root = Path(sglang.__file__).resolve().parent
    for name, digest in SOURCES.items():
        if sha256((root / name).read_bytes()).hexdigest() != digest:
            raise ValueError(f"installed ingress dependency fingerprint differs: {name}")
    cls = module.Scheduler
    if getattr(cls, "_rsi_ingress_rendezvous", False):
        raise RuntimeError("ingress rendezvous is already installed in this worker")
    path = Path(directory) / f"ingress-rank-{rank}.jsonl"
    with path.open("xb"):
        pass

    def report(event):
        with path.open("ab") as stream:
            stream.write((json.dumps({**event, "pid": os.getpid()}, sort_keys=True, separators=(",", ":")) + "\n").encode())

    gate = IngressGate(contract, rank, work_type=io_struct.TokenizedGenerateReqInput,
                       controls=(io_struct.GetInternalStateReq,), teardown_controls=(io_struct.ShutdownReq,), report=report)
    def initialize_vote(scheduler):
        group = _verify_live(scheduler, rank, torch.distributed, envs, module.DisaggregationMode.NULL)
        return _collective(torch, group, contract["collective_timeout_ms"])

    cls.process_input_requests = _wrap_processor(cls.process_input_requests, gate, initialize_vote)
    cls._rsi_ingress_rendezvous = True
