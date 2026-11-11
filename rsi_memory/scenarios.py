"""Content and externally scheduled lifecycles for four separate KV workloads."""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import sha256
import json
from pathlib import Path

from .topology import _digest, _fields, _id, _integer, _list, fingerprint


CAPABILITIES = {
    "agent_tools": ("tool_roundtrip", "retained_prefix"),
    "long_document_qa": ("long_document_qa",),
    "chatbot_growth": ("growing_conversation", "retained_prefix"),
    "rollout_cow": ("shared_prefix_cow",),
}
_EVENT_FIELDS = {
    "open": (), "append": ("segment",), "infer": ("max_new_tokens", "select_rows"),
    "fork": ("parent",), "close": (),
}
_ROLES = frozenset({"system", "user", "assistant", "tool_call", "tool_result", "document", "question"})
_MAX_CONTENT_BYTES = 8 * 1024 * 1024


def _text(value, label):
    if type(value) is not str or not value.strip():
        raise ValueError(f"empty {label}")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise ValueError(f"invalid Unicode in {label}") from exc
    return value


def content_fingerprint(data):
    return fingerprint({"segments": data["segments"], "task": data["task"]})


def _validate(data):
    _fields(data, ("schema", "kind", "name", "source", "page_rows", "segments", "task", "events"))
    kind = data["kind"]
    if type(data["schema"]) is not int or data["schema"] != 1 or type(kind) is not str or kind not in CAPABILITIES:
        raise ValueError("unsupported workload schema or kind")
    _id(data["name"])
    if _integer(data["page_rows"], "reference page rows", 2) > 65536:
        raise ValueError("reference page exceeds limit")
    source = data["source"]
    _fields(source, ("kind", "locator", "revision", "license", "content_sha256"))
    if source["kind"] not in ("authored_fixture", "published_example"):
        raise ValueError("unsupported workload source")
    for field_name in ("locator", "revision", "license"):
        _text(source[field_name], field_name)
    if source["kind"] == "published_example" and not source["locator"].startswith("https://"):
        raise ValueError("published examples require a source URL")
    segments = {}
    total_bytes = 0
    for segment in _list(data["segments"], "segments", limit=256):
        _fields(segment, ("id", "role", "text"), ("call_id",))
        segment_id, role = _id(segment["id"]), segment["role"]
        if segment_id in segments or type(role) is not str or role not in _ROLES:
            raise ValueError("duplicate segment or unknown role")
        if role in ("tool_call", "tool_result"):
            _id(segment.get("call_id"))
        elif "call_id" in segment:
            raise ValueError("call ID requires a tool role")
        total_bytes += len(_text(segment["text"], "segment text").encode("utf-8"))
        segments[segment_id] = segment
    if not segments or total_bytes > _MAX_CONTENT_BYTES:
        raise ValueError("workload content missing or exceeds limit")
    if kind == "long_document_qa":
        _fields(data["task"], ("minimum_document_bytes", "expected_answers"))
        minimum = _integer(data["task"]["minimum_document_bytes"], "minimum document bytes", 4096)
        answers = _list(data["task"]["expected_answers"], "expected answers", limit=16)
        if minimum > _MAX_CONTENT_BYTES or not answers:
            raise ValueError("document task lacks a usable QA contract")
        for answer in answers:
            if len(_text(answer, "expected answer").encode("utf-8")) > 4096:
                raise ValueError("expected answer exceeds limit")
    else:
        _fields(data["task"], ())
    if _digest(source["content_sha256"]) != content_fingerprint(data):
        raise ValueError("workload content hash differs from source binding")

    previous = -1
    for event in _list(data["events"], "external schedule", limit=1024):
        action = event.get("op") if type(event) is dict else None
        if type(action) is not str or action not in _EVENT_FIELDS:
            raise ValueError("unknown external schedule operation")
        _fields(event, ("op", "at_ns", "session", *_EVENT_FIELDS[action]))
        at_ns = _integer(event["at_ns"], "event timestamp")
        if at_ns < previous:
            raise ValueError("external schedule timestamps are out of order")
        previous = at_ns
        _id(event["session"])
        if action == "fork":
            _id(event["parent"])
        if action == "append" and _id(event["segment"]) not in segments:
            raise ValueError("external schedule references an unknown segment")
        if action == "infer":
            _integer(event["max_new_tokens"], "inference output budget", 1)
            positions = _list(event["select_rows"], "ordered reference selection", limit=16384)
            if not positions or len(set(_integer(pos, "selected position") for pos in positions)) != len(positions):
                raise ValueError("reference inference needs nonduplicate selected positions")
    return segments


@dataclass(frozen=True)
class ScenarioWorkload:
    """Immutable manifest; schedule and content can be pinned independently."""

    _manifest: str
    kind: str
    schedule_sha256: str
    workload_sha256: str

    @classmethod
    def from_dict(cls, data):
        segments = _validate(data)
        _replay(data, segments)
        return cls(json.dumps(data, sort_keys=True, ensure_ascii=True, allow_nan=False), data["kind"],
                   fingerprint(data["events"]), fingerprint(data))

    @classmethod
    def read(cls, path):
        return cls.from_dict(json.loads(Path(path).read_text(encoding="utf-8")))

    def to_dict(self):
        return json.loads(self._manifest)

    @property
    def required_capabilities(self):
        return CAPABILITIES[self.kind]

    def require_backend_capabilities(self, implemented):
        """Reject an unsupported backend before execution; this gate is not runtime evidence."""
        missing = set(self.required_capabilities) - set(implemented)
        if missing:
            raise ValueError(f"backend has no implementation for workload capabilities: {sorted(missing)}")

    def reference_report(self):
        """Verify real UTF-8 content with a byte-row page store, separate from model KV."""
        data = self.to_dict()
        return {"execution_scope": "cpu_utf8_page_reference", "workload_sha256": self.workload_sha256,
                "schedule_sha256": self.schedule_sha256, **_replay(data, {s["id"]: s for s in data["segments"]})}


@dataclass
class _Page:
    rows: bytearray
    owners: int = 1


@dataclass
class _Session:
    pages: list[int] = field(default_factory=list)
    segments: list[str] = field(default_factory=list)
    last_inference: tuple[int, tuple[int, ...], int, int] | None = None
    pending_call: str | None = None


def _append(pages, session, rows, page_rows, new_page):
    copied = False
    if session.pages:
        tail = pages[session.pages[-1]]
        remaining = page_rows - len(tail.rows)
        if remaining:
            if tail.owners > 1:
                tail.owners -= 1
                session.pages[-1] = new_page(tail.rows)
                tail = pages[session.pages[-1]]
                copied = True
            tail.rows.extend(rows[:remaining])
            rows = rows[remaining:]
    for offset in range(0, len(rows), page_rows):
        session.pages.append(new_page(rows[offset:offset + page_rows]))
    return copied


def _replay(data, segments):
    sessions, pages = {}, {}
    used_sessions, used_segments = set(), set()
    rows = {segment_id: segment["text"].encode("utf-8") for segment_id, segment in segments.items()}
    witnesses, observations, forks, branch_writes = {}, [], {}, {}
    tool_calls = set()
    next_page = 0

    def new_page(payload):
        nonlocal next_page
        next_page += 1
        pages[next_page] = _Page(bytearray(payload))
        return next_page

    def materialize(session):
        actual = b"".join(pages[pid].rows for pid in session.pages)
        expected = b"".join(rows[segment_id] for segment_id in session.segments)
        if actual != expected:
            raise ValueError("page contents differ from the immutable logical transcript")
        return actual

    for ordinal, event in enumerate(data["events"]):
        action, session_id = event["op"], event["session"]
        if action in ("open", "fork"):
            if session_id in used_sessions:
                raise ValueError("session identity reused")
            used_sessions.add(session_id)
            sessions[session_id] = _Session()
            if action == "fork":
                parent_id = event["parent"]
                if parent_id not in sessions or parent_id == session_id or not sessions[parent_id].pages:
                    raise ValueError("fork requires a live nonempty parent")
                parent = sessions[parent_id]
                if parent.pending_call is not None:
                    raise ValueError("cannot fork while a tool call is pending")
                shared = parent.pages.copy()
                sessions[session_id] = _Session(shared, parent.segments.copy())
                for pid in shared:
                    pages[pid].owners += 1
                forks[session_id] = (parent_id, ordinal, materialize(parent), len(shared))
            continue
        if session_id not in sessions:
            raise ValueError("schedule operation requires a live session")
        session = sessions[session_id]
        if action == "close":
            if session.pending_call is not None:
                raise ValueError("cannot close while a tool call is pending")
            materialize(session)
            for pid in session.pages:
                pages[pid].owners -= 1
                if pages[pid].owners == 0:
                    del pages[pid]
            del sessions[session_id]
            continue
        if action == "append":
            segment_id = event["segment"]
            segment = segments[segment_id]
            if session.pending_call is not None:
                if segment["role"] != "tool_result" or segment["call_id"] != session.pending_call:
                    raise ValueError("tool response does not match its pending call")
                session.pending_call = None
            elif segment["role"] == "tool_result":
                raise ValueError("tool response requires a pending call")
            elif segment["role"] == "tool_call":
                if session.last_inference is None:
                    raise ValueError("tool call requires a preceding inference")
                session.pending_call = segment["call_id"]
            if _append(pages, session, rows[segment_id], data["page_rows"], new_page) and session_id in forks:
                branch_writes[session_id] = ordinal
            session.segments.append(segment_id)
            used_segments.add(segment_id)
            continue

        if session.pending_call is not None:
            raise ValueError("inference cannot start before a tool response")
        payload = materialize(session)
        positions = event["select_rows"]
        if max(positions) >= len(payload):
            raise ValueError("reference selection exceeds the current context")
        previous = session.last_inference
        retained = 0
        if previous is not None:
            previous_size, previous_pages, previous_segments, previous_ordinal = previous
            for page_id, old_id in zip(session.pages, previous_pages):
                if page_id != old_id:
                    break
                retained = min(previous_size, retained + data["page_rows"])
            if retained == previous_size and len(payload) > previous_size:
                witnesses["retained_prefix"] = [previous_ordinal, ordinal]
                new_segments = [segments[segment_id] for segment_id in session.segments[previous_segments:]]
                if (len(new_segments) == 2 and [s["role"] for s in new_segments] == ["tool_call", "tool_result"]
                        and new_segments[0]["call_id"] == new_segments[1]["call_id"]):
                    tool_calls.add(new_segments[0]["call_id"])
                    witnesses["tool_roundtrip"] = [previous_ordinal, ordinal]
                if ([s["role"] for s in new_segments] == ["assistant", "user"]
                        and segments[session.segments[previous_segments - 1]]["role"] == "user"):
                    witnesses["growing_conversation"] = [previous_ordinal, ordinal]

        transcript = [segments[segment_id] for segment_id in session.segments]
        if data["kind"] == "long_document_qa" and transcript[-1]["role"] == "question":
            document_bytes = sum(len(s["text"].encode("utf-8")) for s in transcript[:-1] if s["role"] == "document")
            if document_bytes >= data["task"]["minimum_document_bytes"]:
                witnesses["long_document_qa"] = [ordinal]
        if session_id in branch_writes:
            parent_id, fork_ordinal, prefix, shared_pages = forks[session_id]
            parent = sessions.get(parent_id)
            if parent is None or not materialize(parent).startswith(prefix) or not payload.startswith(prefix):
                raise ValueError("copy-on-write did not preserve the live parent and child prefix")
            if len(payload) <= len(prefix):
                raise ValueError("copy-on-write branch lacks a new suffix")
            cow = {"parent": parent_id, "fork_event": fork_ordinal, "write_event": branch_writes[session_id],
                   "shared_pages_at_fork": shared_pages, "prefix_sha256": sha256(prefix).hexdigest(),
                   "suffix_sha256": sha256(payload[len(prefix):]).hexdigest()}
        else:
            cow = None
        observations.append({"event": ordinal, "session": session_id, "context_rows": len(payload),
                             "context_sha256": sha256(payload).hexdigest(), "retained_rows": retained,
                             "selected_sha256": sha256(bytes(payload[pos] for pos in positions)).hexdigest(),
                             **({"cow": cow} if cow else {})})
        session.last_inference = (len(payload), tuple(session.pages), len(session.segments), ordinal)

    if sessions or pages or used_segments != segments.keys():
        raise ValueError("workload must consume every segment and close every page owner")
    branches = [obs for obs in observations if "cow" in obs]
    for ordinal, first in enumerate(branches):
        for second in branches[ordinal + 1:]:
            a, b = first["cow"], second["cow"]
            if (first["session"] != second["session"] and a["parent"] == b["parent"]
                    and a["prefix_sha256"] == b["prefix_sha256"] and a["suffix_sha256"] != b["suffix_sha256"]):
                witnesses["shared_prefix_cow"] = [a["fork_event"], b["fork_event"], first["event"], second["event"]]
    missing = set(CAPABILITIES[data["kind"]]) - witnesses.keys()
    if missing:
        raise ValueError(f"external schedule has no evidence for workload capabilities: {sorted(missing)}")
    return {"witnesses": {key: witnesses[key] for key in CAPABILITIES[data["kind"]]}, "inferences": observations,
            "distinct_tool_calls": len(tool_calls), "physical_pages_created": next_page, "live_pages_at_end": len(pages)}
