"""Read-only scheduler and packed-KV observations for the TP/relay runner."""

from functools import wraps
from hashlib import sha256
import json
import os
from pathlib import Path

from .sglang_global_ticks import prepared_tick, scheduler_tick

SCHEDULER_FIELDS = (
    "schedule_policy", "schedule_conservativeness", "max_running_requests", "max_total_tokens", "max_prefill_tokens",
    "chunked_prefill_size", "tp_size", "pp_size", "dp_size", "disaggregation_mode", "disable_overlap_schedule",
    "disable_radix_cache", "disable_cuda_graph", "enable_priority_scheduling", "schedule_low_priority_values_first",
)


def append(path, event):
    line = (json.dumps(event, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n").encode()
    with Path(path).open("ab") as out:
        out.write(line)


def save(path, value):
    with Path(path).open("x", encoding="utf-8") as out:
        json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
        out.write("\n")


def publish(path, value):
    """Expose a complete worker receipt with exclusive-create semantics."""
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    pending = path.with_name(path.name + ".pending")
    out = pending.open("x", encoding="utf-8")
    try:
        with out:
            json.dump(value, out, sort_keys=True, indent=2, allow_nan=False)
            out.write("\n")
        # A same-directory hard link atomically creates the final name without
        # replacing another writer's receipt on POSIX.
        os.link(pending, path)
    finally:
        pending.unlink()


def _save_schedule(path, rank, scheduler):
    effective = {key: getattr(scheduler.server_args, key) for key in SCHEDULER_FIELDS}
    effective["worker_limits"] = {key: getattr(scheduler, key) for key in (
        "max_running_requests", "max_total_num_tokens", "max_prefill_tokens", "chunked_prefill_size",
        "max_req_len", "max_req_input_len")}
    ps = scheduler.ps
    parallel = {"backbone_tp_size": ps.tp_size, "attention_tp_size": ps.attn_tp_size,
                "attention_dp_size": ps.attn_dp_size, "attention_dp_rank": ps.attn_dp_rank}
    publish(path, {"rank": rank, "pid": os.getpid(), "effective": effective, "parallel": parallel})


def install_schedule_snapshot(path, rank):
    from sglang.srt.managers.scheduler import Scheduler

    original = Scheduler.__init__

    @wraps(original)
    def initialize(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _save_schedule(path, rank, self)

    Scheduler.__init__ = initialize


def _payload(torch, slab, positions):
    ids = torch.tensor(positions, dtype=torch.int64, device=slab.device)
    return slab.index_select(0, ids).cpu().numpy().tobytes()


def _packed_row(request, length, selected, history, newest, misses):
    return {"request_slot": request, "seq_len": length, "selected_ids": selected,
            "history_bytes": len(history), "history_sha256": sha256(history).hexdigest(),
            "newest_sha256": sha256(newest).hexdigest(), "miss_rows": misses}


def _packed_event(path, rank, layer, rows):
    append(path, {"event": "kv_selected_exact", "rank": rank, "layer": layer, "rows": rows})


def install_resident_packed_kv_probe(path, rank):
    """Emit the stock diagnostic format using the independently observed producer."""
    import torch
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator, HiSparseDSATokenToKVPool

    from .sglang_resident_oracle import PackedProducerOracle, attach_oracle, torch_row_reader

    path = Path(path)
    with path.open("xb"):
        pass
    oracle = PackedProducerOracle(torch_row_reader(torch))

    def emit(layer, records):
        _packed_event(path, rank, layer, [_packed_row(*record, 0) for record in records])

    def on_mismatch(layer, reason):
        append(path, {"event": "kv_mismatch", "rank": rank, "layer": layer, "reason": reason})

    attach_oracle(HiSparseCoordinator, HiSparseDSATokenToKVPool, oracle, emit=emit, on_mismatch=on_mismatch)
    return oracle


def install_packed_kv_probe(path, plan):
    """Compare the bytes in attention's selected native hot slots to their backing."""
    import torch
    from sglang.srt.managers.hisparse_coordinator import HiSparseCoordinator

    path = Path(path)
    with path.open("xb"):
        pass
    original = HiSparseCoordinator._run_swap_in_kernel
    usage = {tier: 0 for tier in plan.layer_tiers}

    @wraps(original)
    def checked(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id, record_plan=False):
        real = int(self.num_real_reqs.item())
        if not 0 < real <= req_pool_indices.numel() or self.skip_io or self.enable_prefetch or record_plan:
            raise RuntimeError("packed probe needs an ordinary live eager decode batch")
        requests = req_pool_indices[:real].tolist()
        lengths = compressed_seq_lens[:real].tolist()
        selections = top_k_result[:real].tolist()
        tagged = self.req_device_buffer_tokens[layer_id, requests, :plan.hot_slots].tolist()
        output = original(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id, record_plan=record_plan)
        locations = output[:real].tolist()
        cold = self.mem_pool_host.kv_buffer[layer_id]
        hot = self.mem_pool_device.kv_buffer[layer_id]
        rows = []
        for request, length, selected, slots, previous in zip(requests, lengths, selections, locations, tagged, strict=True):
            lanes = [(token, slot) for token, slot in zip(selected, slots, strict=True) if token >= 0]
            tokens, used_slots = ({lane[index] for lane in lanes} for index in (0, 1))
            if (not lanes or len(tokens) != len(lanes) or len(used_slots) != len(lanes)
                    or any(not 0 <= token < length or not 0 < slot < hot.shape[0] for token, slot in lanes)):
                raise RuntimeError("invalid native sparse-attention lane or physical hot slot")
            history = [(token, slot) for token, slot in lanes if token != length - 1]
            history_ids = [token for token, _ in history]
            backing_rows = self.req_to_host_pool[request, history_ids].tolist()
            if len(set(backing_rows)) != len(backing_rows) or any(not 0 <= row < cold.shape[0] for row in backing_rows):
                raise RuntimeError("invalid or aliased cold KV backing location")
            expected = _payload(torch, cold, backing_rows)
            observed = _payload(torch, hot, [slot for _, slot in history])
            if expected != observed:
                offset = next(i for i, (left, right) in enumerate(zip(expected, observed, strict=True)) if left != right)
                append(path, {"event": "kv_mismatch", "rank": plan.rank, "layer": layer_id, "request_slot": request,
                              "seq_len": length, "logical_token": history[offset // plan.row_bytes][0],
                              "byte_offset": offset % plan.row_bytes})
                raise RuntimeError("selected sparse-attention KV differs from its cold backing")
            newest_slot = int(self.req_device_buffer_token_locs[layer_id, request, min(length - 1, plan.hot_slots)].item())
            if not 0 < newest_slot < hot.shape[0] or any(slot != newest_slot for token, slot in lanes if token == length - 1):
                raise RuntimeError("newest logical KV used an invalid reserved slot")
            newest = _payload(torch, hot, [newest_slot])
            misses = len(set(history_ids) - set(previous)) if length > plan.hot_slots else 0
            usage[plan.layer_tiers[layer_id]] += misses * plan.row_bytes
            rows.append(_packed_row(request, length, [token for token, _ in lanes], expected, newest, misses))
        _packed_event(path, plan.rank, layer_id, rows)
        return output

    HiSparseCoordinator._run_swap_in_kernel = checked
    return usage


def _install_global_observer(directory, rank, scheduler_class, forward_batch_class, local_finished):
    original_run = scheduler_class.run_batch
    original_next = scheduler_class.get_next_batch_to_run
    original_prepare = forward_batch_class.prepare_mlp_sync_batch
    rows = []
    pending = actual = None
    closed = False
    finished_paths = [directory / f"dispatch-rank-{index}.json" for index in range(2)]

    @wraps(original_prepare)
    def prepare(batch, *args, **kwargs):
        nonlocal actual
        result = original_prepare(batch, *args, **kwargs)
        if pending is not None:
            if actual is not None:
                raise RuntimeError("one scheduler tick prepared multiple eager DP forwards")
            actual = prepared_tick(pending, batch)
        return result

    @wraps(original_run)
    def run(self, batch, *args, **kwargs):
        nonlocal pending, actual
        if pending is not None or closed:
            raise RuntimeError("unexpected nested or post-completion global forward")
        pending = scheduler_tick(rank, self.forward_ct + 1, batch)
        try:
            result = original_run(self, batch, *args, **kwargs)
            if actual is None or batch.forward_iter != pending["forward_iter"]:
                raise RuntimeError("scheduler forward lacked its matching actual DP padding capture")
            rows.append(actual)
            return result
        finally:
            pending = actual = None

    @wraps(original_next)
    def plan(self, *args, **kwargs):
        nonlocal closed
        value = original_next(self, *args, **kwargs)
        # The pinned DP sync represents a peer's forward as a local IDLE batch.
        if not closed and value.batch_to_run is None and local_finished() and all(path.is_file() for path in finished_paths):
            publish(directory / f"global-ticks-rank-{rank}.json", {"schema": 1, "rank": rank, "pid": os.getpid(), "ticks": rows})
            closed = True
        return value

    forward_batch_class.prepare_mlp_sync_batch = prepare
    scheduler_class.run_batch = run
    scheduler_class.get_next_batch_to_run = plan


def install_scheduler_observer(directory, plan, expected_rids, selected_bytes, *, record_global_ticks=False,
                               layer_tiers=None, resident_oracle=None, expected_decode_steps=None):
    import torch
    from sglang.srt.managers.scheduler import Scheduler

    directory = Path(directory)
    expected = set(expected_rids)
    tiers = plan.layer_tiers if layer_tiers is None else layer_tiers
    if resident_oracle is not None and (type(expected_decode_steps) is not dict or expected_decode_steps.keys() != expected):
        raise ValueError("resident packed coverage needs the expected decode count for each request")
    original_init, original_run, original_process = Scheduler.__init__, Scheduler.run_batch, Scheduler.process_batch_result
    batches, finished, backup_bytes = [], set(), {tier: 0 for tier in tiers}
    saved = False

    @wraps(original_init)
    def initialize(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        _save_schedule(directory / f"schedule-rank-{plan.rank}.json", plan.rank, self)
        pool = self.hisparse_coordinator.mem_pool_host
        original_backup = pool.backup_from_device_all_layer

        @wraps(original_backup)
        def backup(device_pool, host_indices, device_indices, io_backend):
            result = original_backup(device_pool, host_indices, device_indices, io_backend)
            for tier in backup_bytes:
                backup_bytes[tier] += host_indices.numel() * plan.row_bytes * tiers.count(tier)
            return result

        pool.backup_from_device_all_layer = backup

    @wraps(original_run)
    def run(self, batch, *args, **kwargs):
        if not saved and batch.reqs:
            batches.append({"request_ids": [req.rid for req in batch.reqs], "mode": batch.forward_mode.name})
        return original_run(self, batch, *args, **kwargs)

    @wraps(original_process)
    def complete(self, batch, *args, **kwargs):
        nonlocal saved
        requests = tuple(batch.reqs)
        result = original_process(self, batch, *args, **kwargs)
        if not saved:
            finished.update(req.rid for req in requests if req.finished())
            if expected <= finished:
                pool = self.hisparse_coordinator.mem_pool_host
                relay = getattr(pool, "_relay", None)
                ordinals = {plan.compute_device, plan.peer_device} - {None}
                memory = {index: {"allocated_bytes": torch.cuda.memory_allocated(index),
                                  "reserved_bytes": torch.cuda.memory_reserved(index),
                                  "peak_allocated_bytes": torch.cuda.max_memory_allocated(index),
                                  "peak_reserved_bytes": torch.cuda.max_memory_reserved(index)} for index in sorted(ordinals)}
                record = {"rank": plan.rank, "finished": sorted(finished), "batches": batches,
                          "backup_payload_bytes_by_tier": backup_bytes, "selected_miss_payload_bytes_by_tier": selected_bytes,
                          "relay": None if relay is None else relay.statistics(), "torch_cuda_memory": memory}
                if resident_oracle is not None:
                    record["resident_producer_oracle"] = resident_oracle.result(expected_decode_steps)
                publish(directory / f"dispatch-rank-{plan.rank}.json", record)
                saved = True
        return result

    Scheduler.__init__, Scheduler.run_batch, Scheduler.process_batch_result = initialize, run, complete
    if record_global_ticks:
        from sglang.srt.model_executor.forward_batch_info import ForwardBatch

        _install_global_observer(directory, plan.rank, Scheduler, ForwardBatch, lambda: saved)
