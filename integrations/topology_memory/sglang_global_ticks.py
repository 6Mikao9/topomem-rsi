"""Compare rank-aligned DP forwards without dropping one-rank idle work."""

from .sglang_ingress_gate import PHASES, TICK_POLICY, checked_contract, fingerprint


_COMMON = ("global_forward_mode", "global_num_tokens", "global_num_tokens_for_logprob", "dp_padding_mode",
           "padded_global_num_tokens", "global_dp_buffer_len")
_LOCAL = ("mode", "forward_mode", "rids", "seq_lens", "extend_lens", "prefix_lens")
_FIELDS = {"rank", "forward_iter", *_COMMON, *_LOCAL}
_MODES = {"EXTEND", "DECODE", "MIXED", "IDLE"}


def scheduler_tick(rank, forward_iter, batch):
    """Capture original schedule fields before a real forward. All reads are CPU-only."""
    mode = batch.forward_mode.name
    rids = [req.rid for req in batch.reqs]
    seq_lens = []
    if rids:
        if batch.seq_lens_cpu is None or batch.seq_lens_cpu.device.type != "cpu":
            raise ValueError("missing CPU sequence lengths for the active DP tick")
        seq_lens = batch.seq_lens_cpu.tolist()
    is_extend = mode in {"EXTEND", "MIXED"}
    global_mode = batch.global_forward_mode
    if type(global_mode) is int:
        global_mode = type(batch.forward_mode)(global_mode)
    return {"rank": rank, "forward_iter": forward_iter, "mode": mode, "rids": rids, "seq_lens": seq_lens,
            "extend_lens": list(batch.extend_lens) if is_extend else [],
            "prefix_lens": list(batch.prefix_lens) if is_extend else [],
            "global_forward_mode": global_mode.name if global_mode is not None else None,
            "global_num_tokens": list(batch.global_num_tokens),
            "global_num_tokens_for_logprob": list(batch.global_num_tokens_for_logprob)}


def prepared_tick(scheduler_row, forward_batch):
    """Join a captured scheduler tick to its actual eager model padding result."""
    if (scheduler_row["global_num_tokens"] != forward_batch.original_global_num_tokens_cpu
            or scheduler_row["global_num_tokens_for_logprob"] != forward_batch.global_num_tokens_for_logprob_cpu):
        raise ValueError("model forward changed the scheduler's original global token vectors")
    return {**scheduler_row, "forward_mode": forward_batch.forward_mode.name,
            "dp_padding_mode": forward_batch.dp_padding_mode.name,
            "padded_global_num_tokens": list(forward_batch.global_num_tokens_cpu),
            "global_dp_buffer_len": forward_batch.global_dp_buffer_len}


def _vector(value, length, upper):
    return type(value) is list and len(value) == length and all(type(item) is int and 0 <= item <= upper for item in value)


def _active_rows(rows, rank):
    active = []
    for row in rows:
        if type(row) is not dict or row.keys() != _FIELDS or type(row["rank"]) is not int or row["rank"] != rank:
            raise ValueError("unexpected global tick fields or rank")
        counts, logprob = row["global_num_tokens"], row["global_num_tokens_for_logprob"]
        if not _vector(counts, 2, 16384) or not _vector(logprob, 2, 16384):
            raise ValueError("invalid original global token vectors")
        if counts == [0, 0]:
            if (row["mode"] != "NO_FORWARD" or row["forward_mode"] is not None or row["global_forward_mode"] is not None
                    or row["forward_iter"] is not None or row["dp_padding_mode"] is not None
                    or row["padded_global_num_tokens"] != [0, 0] or row["global_dp_buffer_len"] != 0
                    or logprob != [0, 0] or any(row[key] != [] for key in ("rids", "seq_lens", "extend_lens", "prefix_lens"))):
                raise ValueError("zero-token row is not a verified no-forward poll")
            continue
        idx = row["forward_iter"]
        if type(idx) is not int or idx != len(active) + 1:
            raise ValueError("missing or duplicated global forward iteration")
        if (row["mode"] not in _MODES or row["forward_mode"] not in _MODES
                or (row["global_forward_mode"] is not None and row["global_forward_mode"] not in _MODES)
                or (row["mode"] == "IDLE") != (counts[rank] == 0)):
            raise ValueError("invalid rank-local mode, idle state, or global mode")
        rids = row["rids"]
        expected_length = int(counts[rank] != 0)
        if type(rids) is not list or len(rids) != expected_length or not _vector(row["seq_lens"], expected_length, 8192):
            raise ValueError("global tick lost its real rank-local request shape")
        expected_length = expected_length if row["mode"] in {"EXTEND", "MIXED"} else 0
        if (not _vector(row["extend_lens"], expected_length, 8192) or not _vector(row["prefix_lens"], expected_length, 8192)
                or (expected_length and sum(row["extend_lens"]) != counts[rank])):
            raise ValueError("invalid original rank-local prefill shape")
        padding = row["dp_padding_mode"]
        if padding not in {"SUM_LEN", "MAX_LEN"}:
            raise ValueError("missing actual DP padding mode")
        expected_padding = counts if padding == "SUM_LEN" else [max(counts)] * 2
        if (not _vector(row["padded_global_num_tokens"], 2, 16384) or row["padded_global_num_tokens"] != expected_padding
                or type(row["global_dp_buffer_len"]) is not int or row["global_dp_buffer_len"] != sum(expected_padding)):
            raise ValueError("actual DP padding shape differs from its token vectors")
        active.append(row)
    return active


def verify_global_ticks(rows_by_rank, contract, *, expected_sha256=None):
    """Require complete dual-rank traces, then compare their frozen forward signature."""
    groups = checked_contract(contract)
    if len(rows_by_rank) != 2:
        raise ValueError("global forward proof requires both attention ranks")
    active = [_active_rows(rows, rank) for rank, rows in enumerate(rows_by_rank)]
    if not active[0] or len(active[0]) != len(active[1]):
        raise ValueError("global forward proof lost a participating rank or iteration")
    phases = {member.rid: (index, member.rank) for index, group in enumerate(groups) for member in group}
    phase = -1
    canonical = []
    for left, right in zip(*active, strict=True):
        pair = (left, right)
        if any(left[key] != right[key] for key in _COMMON):
            raise ValueError("ranks disagree on the common global forward shape")
        assigned = []
        for rank, row in enumerate(pair):
            for rid in row["rids"]:
                if not isinstance(rid, str) or rid not in phases or phases[rid][1] != rank:
                    raise ValueError("global forward used a request outside its committed rank")
                assigned.append(phases[rid][0])
        if len(set(assigned)) != 1 or assigned[0] not in (phase, phase+1):
            raise ValueError("global forward reordered or crossed frozen arrival groups")
        if assigned[0] != phase:
            phase += 1
            if any(row["mode"] != "EXTEND" or row["rids"] != [groups[phase][rank].rid] for rank, row in enumerate(pair)):
                raise ValueError("arrival group did not start both ranks on the same global prefill tick")
        canonical.append({"forward_iter": left["forward_iter"], "phase": PHASES[phase], **{key: left[key] for key in _COMMON},
                          "ranks": [{key: row[key] for key in _LOCAL} for row in pair]})
    if phase != len(PHASES) - 1:
        raise ValueError("missing measured global forward group")
    digest = fingerprint({"schema": 1, "policy": TICK_POLICY, "ticks": canonical})
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError("global forward signature differs from the frozen native reference")
    return {"schema": 1, "policy": TICK_POLICY, "global_ticks_sha256": digest, "active_forward_count": len(canonical),
            "rank_idle_forward_count": sum(row["mode"] == "IDLE" for rows in active for row in rows)}
