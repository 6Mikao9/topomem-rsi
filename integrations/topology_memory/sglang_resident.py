"""Proof and worker hooks for the fixed CUDA GLM HiSparse short path."""

from dataclasses import asdict, dataclass
from functools import wraps
from hashlib import sha256
import json
from pathlib import Path
import re


HOT_SLOTS = 6144
TOP_K = 2048
PAGE_ROWS = 64
ROW_BYTES = 656
LAYERS = 6
PHASES = ("warmup", "requests")
RESIDENT_TIER = "resident"
RESIDENT_SENTINEL_LOGICAL_BYTES = LAYERS * ROW_BYTES
RESIDENT_ALLOCATOR_CHARGE_BYTES = LAYERS * 2 * 1024**2
ENGINE_FIELDS = {
    "tp_size": 2, "dp_size": 2, "pp_size": 1, "enable_dp_attention": True,
    "enable_hisparse": True, "disable_radix_cache": True,
    "disable_cuda_graph": True, "disable_overlap_schedule": True,
    "kv_cache_dtype": "fp8_e4m3", "dsa_prefill_backend": "flashinfer_sparse_mla",
    "dsa_decode_backend": "flashinfer_sparse_mla", "dsa_topk_backend": "torch",
}
EXTRA_SOURCES = {
    "srt/mem_cache/allocator/hisparse.py": "7905663ae85954276fea4e752d2ce5caa3cb9434f450bb68a82ea72a3d4db712",
    "srt/mem_cache/hisparse_memory_pool.py": "fb266ef79ef763b0a2401112a5e9cd696bb8dac4421f8c2f6c909d4521fb2118",
    "srt/mem_cache/pool_host/hisparse.py": "0f94fd23d74613556ac5dbca59d8399fd3c5f6cbfc8758775bedb5fb5b8f9e01",
}


def fingerprint(value):
    raw = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    return sha256(raw.encode()).hexdigest()


def _round_page(length):
    return (length + PAGE_ROWS - 1) // PAGE_ROWS * PAGE_ROWS


def reserved_rows(length):
    if type(length) is not int or not 1 <= length <= HOT_SLOTS:
        raise ValueError("request exceeds the resident-only hot range")
    rounded = _round_page(length)
    return rounded + (PAGE_ROWS if rounded == HOT_SLOTS else 0)


def _tokens(ids):
    values = list(ids)
    if not values or any(type(token) is not int or token < 0 for token in values):
        raise ValueError("resident-only requires exact nonnegative token IDs")
    return values


@dataclass(frozen=True)
class RequestBound:
    phase: str
    rid: str
    rank: int
    prompt_length: int
    prompt_sha256: str
    max_new_tokens: int

    @property
    def length(self):
        return self.prompt_length + self.max_new_tokens


@dataclass(frozen=True)
class ResidentProof:
    engine_sha256: str
    device_rows: int
    reserved_rows_by_rank: tuple[int, int]
    requests: tuple[RequestBound, ...]

    def to_dict(self):
        return {"schema": 1, "engine_sha256": self.engine_sha256, "device_rows": self.device_rows,
                "reserved_rows_by_rank": list(self.reserved_rows_by_rank),
                "requests": [asdict(request) for request in self.requests]}

    def fingerprint(self):
        return fingerprint(self.to_dict())

    def for_rank(self, rank):
        if type(rank) is not int or rank not in (0, 1):
            raise ValueError("resident-only worker requires a fixed rank 0 or 1")
        return {request.rid: request for request in self.requests if request.rank == rank}


def proof_from_commitments(engine, fixture, token_commitments, *, device_rows):
    """Bind the case's actual tokenizer commitments to measured native hot capacity."""
    if type(engine) is not dict or any(type(engine.get(key)) is not type(value) or engine[key] != value
                                       for key, value in ENGINE_FIELDS.items()):
        raise ValueError("resident-only requires the fixed DP2 GLM DSA engine")
    if (engine.get("speculative_algorithm") is not None or engine.get("disaggregation_mode", "null") != "null"
            or engine.get("enable_prefill_cp", False) or engine.get("attn_dcp_size", 1) != 1
            or engine.get("enable_hierarchical_cache", False) or engine.get("disaggregation_decode_enable_offload_kvcache", False)
            or engine.get("sleep_on_idle", False) or engine.get("scheduler_recv_interval", 1) != 1):
        raise ValueError("resident-only forbids speculative, prefix/offload and alternate scheduler paths")
    try:
        pool = json.loads(engine["hisparse_config"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("resident-only requires a bound native HiSparse configuration") from exc
    if pool != {"device_buffer_size": HOT_SLOTS, "top_k": TOP_K, "host_to_device_ratio": 2}:
        raise ValueError("resident-only requires unchanged 6144-slot native hot geometry")
    if (type(device_rows) is not int or device_rows <= 0 or device_rows % PAGE_ROWS
            or engine.get("max_total_tokens") != device_rows):
        raise ValueError("resident-only requires the measured native device allocator size")

    if type(fixture) is not dict or type(token_commitments) is not dict:
        raise ValueError("resident-only requires a fixed case and tokenization commitments")
    bounds = []
    for phase in PHASES:
        for row in fixture.get(phase, ()):
            rid, rank, maximum = (row[key] for key in ("rid", "routed_dp_rank", "max_new_tokens"))
            if (type(rid) is not str or not 1 <= len(rid) <= 160 or type(rank) is not int or rank not in (0, 1)
                    or type(maximum) is not int or maximum < 1):
                raise ValueError("resident-only requires four fixed routed text requests")
            commitment = token_commitments.get(rid)
            if (type(commitment) is not dict or commitment.keys() != {"input_tokens", "input_ids_sha256"}
                    or type(commitment["input_tokens"]) is not int or commitment["input_tokens"] < 1
                    or type(commitment["input_ids_sha256"]) is not str
                    or not re.fullmatch(r"[0-9a-f]{64}", commitment["input_ids_sha256"])):
                raise ValueError("invalid resident-only tokenizer commitment")
            bounds.append(RequestBound(phase, rid, rank, commitment["input_tokens"], commitment["input_ids_sha256"], maximum))
    if (len(bounds) != 4 or len({bound.rid for bound in bounds}) != 4
            or {bound.rid for bound in bounds} != token_commitments.keys()
            or {(bound.phase, bound.rank) for bound in bounds} != {(phase, rank) for phase in PHASES for rank in (0, 1)}):
        raise ValueError("resident-only requires one unique request per phase and rank")
    # Do not rely on warmup cleanup racing measured arrival. Charge both requests at their maximum.
    rows = tuple(sum(reserved_rows(bound.length) for bound in bounds if bound.rank == rank) for rank in (0, 1))
    if any(count > device_rows for count in rows):
        raise ValueError("resident-only proof exceeds the existing physical page budget")
    return ResidentProof(fingerprint(engine), device_rows, rows, tuple(bounds)).to_dict()


def checked_binding(engine, fixture, token_commitments, binding):
    if (type(binding) is not dict or binding.keys() != {"proof", "allocator_charge_bytes_by_rank"}
            or binding["allocator_charge_bytes_by_rank"] != [RESIDENT_ALLOCATOR_CHARGE_BYTES] * 2):
        raise ValueError("invalid resident-only allocator charge")
    proof = proof_from_commitments(engine, fixture, token_commitments, device_rows=engine["max_total_tokens"])
    if binding["proof"] != proof:
        raise ValueError("resident-only proof differs from actual tokenized inputs and native geometry")
    return ResidentProof(proof["engine_sha256"], proof["device_rows"], tuple(proof["reserved_rows_by_rank"]),
                         tuple(RequestBound(**row) for row in proof["requests"]))


def check_request(bounds, rid, ids, max_new_tokens):
    bound = bounds.get(rid)
    if bound is None or type(max_new_tokens) is not int or max_new_tokens != bound.max_new_tokens:
        raise ValueError("request differs from the sealed resident-only input")
    values = _tokens(ids)
    if len(values) != bound.prompt_length or fingerprint(values) != bound.prompt_sha256:
        raise ValueError("request token IDs differ from the sealed resident-only input")
    return bound


def check_length(bound, length):
    if type(length) is not int or not 1 <= length <= bound.length or length > HOT_SLOTS:
        raise ValueError("live request exceeds its resident-only proof")


def _no_cold_allocations(coordinator):
    if coordinator._has_pending_backup or not coordinator.mem_pool_host._is_dummy:
        raise RuntimeError("resident-only encountered an active cold backup path")


def _stage_without_backup(coordinator, req, device_module, staging_record):
    _no_cold_allocations(coordinator)
    req.hisparse_staging = True
    start_event = device_module.Event()
    finish_event = device_module.Event()
    start_event.record()
    with device_module.stream(coordinator.write_staging_stream):
        start_event.wait(coordinator.write_staging_stream)
        finish_event.record()
    coordinator.ack_staging_queue.append(staging_record(start_event, finish_event, req))


def _patched_classes(proof, rank, torch, coordinator_module, scheduler, schedule_batch, report):
    """Install only on a dedicated worker before any coordinator is initialized."""
    bounds = proof.for_rank(rank)
    admitted = set()
    coordinator = coordinator_module.HiSparseCoordinator
    if (getattr(coordinator, "_resident_only_installed", False)
            or getattr(coordinator, "_rsi_cold_backing", False)):
        raise RuntimeError("resident-only requires a fresh non-tiered scheduler worker")
    host_cls = coordinator_module.MLATokenToKVPoolHost
    allocation = {}

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("resident-only worker cannot allocate or transfer cold KV")

    def dummy_host(*args, **kwargs):
        pool = kwargs.get("device_pool")
        if (args or pool is None or kwargs.get("layout") != "layer_first" or kwargs.get("page_size") != PAGE_ROWS
                or kwargs.get("host_to_device_ratio") != 2 or kwargs.get("host_size") != 0
                or pool.size != proof.device_rows or pool.page_size != PAGE_ROWS or pool.layer_num != LAYERS
                or pool.kv_cache_dim * pool.store_dtype.itemsize != ROW_BYTES or pool.layer_shard_enabled):
            raise RuntimeError("resident-only worker differs from the proved native device geometry")
        if len(pool.kv_buffer) != LAYERS or any(buffer.device.type != "cuda" for buffer in pool.kv_buffer):
            raise RuntimeError("resident-only requires six CUDA packed KV slabs")
        host = host_cls(**kwargs, is_dummy=True, pin_memory=False)
        # The stock launch resolves a host_cache TensorView pointer even on its no-miss path.
        # Independent CUDA sentinels keep that ABI valid without allocating cold KV pages.
        device = pool.kv_buffer[0].device
        if (torch.cuda.get_allocator_backend() != "native" or device.index != rank
                or any(buffer.device != device for buffer in pool.kv_buffer)):
            raise RuntimeError("resident-only requires the qualified native CUDA allocator and rank")
        allocated_before, reserved_before = torch.cuda.memory_allocated(device), torch.cuda.memory_reserved(device)
        host.kv_buffer = [torch.empty((1, 1, pool.kv_cache_dim), dtype=pool.store_dtype,
                                      device=buffer.device) for buffer in pool.kv_buffer]
        allocated = torch.cuda.memory_allocated(device) - allocated_before
        reserved = torch.cuda.memory_reserved(device) - reserved_before
        if not (RESIDENT_SENTINEL_LOGICAL_BYTES <= allocated <= RESIDENT_ALLOCATOR_CHARGE_BYTES
                and 0 <= reserved <= RESIDENT_ALLOCATOR_CHARGE_BYTES):
            raise RuntimeError("resident-only sentinel allocations exceeded the frozen charge")
        allocation.update(resident_sentinel_allocated_delta_bytes=allocated, resident_sentinel_reserved_delta_bytes=reserved)
        host.alloc_paged_token_slots = unavailable
        host.backup_from_device_all_layer = unavailable
        host.load_to_device_per_layer = unavailable
        return host

    init = coordinator.__init__

    @wraps(init)
    def initialize(self, *args, **kwargs):
        if coordinator_module.MLATokenToKVPoolHost is not host_cls:
            raise RuntimeError("resident-only cold pool constructor is already altered")
        coordinator_module.MLATokenToKVPoolHost = dummy_host
        try:
            init(self, *args, **kwargs)
        finally:
            coordinator_module.MLATokenToKVPoolHost = host_cls
        if (self.is_dsv4_hisparse or self.compress_ratio != 1 or self.page_size != PAGE_ROWS or self.item_size_bytes != ROW_BYTES
                or self.device_buffer_size != HOT_SLOTS or self.top_k != TOP_K
                or self.enable_prefetch or self.skip_io or not self.mem_pool_host._is_dummy):
            raise RuntimeError("resident-only requires unchanged stock short-path dispatch")
        self._resident_bounds_by_slot = {}
        self._resident_proof = proof
        pool = self.mem_pool_device
        hot_metadata = (self.req_to_device_buffer, self.req_device_buffer_tokens, self.req_device_buffer_token_locs,
                        self._lru_init, self.lru_slots, self._device_buffer_arange_i32)
        if (self.tp_world_size != 1 or pool.kv_buffer[0].shape[0] != proof.device_rows + PAGE_ROWS
                or self.mem_pool_host.pin_memory or self.mem_pool_host.size != 2 * proof.device_rows + PAGE_ROWS):
            raise RuntimeError("resident-only changed the measured hot geometry or attention partition")
        report({"event": "hisparse_cold_backing_bound", "rank": rank, "cuda_device": rank, "backing_mode": RESIDENT_TIER,
                "packed_device_payload_bytes": sum(buffer.nbytes for buffer in pool.kv_buffer),
                "device_physical_rows": pool.kv_buffer[0].shape[0], "device_logical_row_capacity": pool.size,
                "request_metadata_slots_including_pad": self.req_to_token_pool.req_to_token.shape[0],
                "hot_metadata_bytes": sum(tensor.nbytes for tensor in hot_metadata),
                "rows": self.mem_pool_host.size, "row_bytes": ROW_BYTES, "top_k": TOP_K, "hot_slots": HOT_SLOTS,
                "layer_tiers": [RESIDENT_TIER] * LAYERS, "pinned_host_bytes": 0, "local_hbm_bytes": 0,
                "peer_hbm_bytes": 0, "peer_device": None, "relay_workspace_bytes": {"compute": 0, "host": 0, "peer": 0},
                "attention_tp_size": self.tp_world_size, "global_compute_ranks": 2,
                "resident_proof_sha256": proof.fingerprint(), "resident_sentinel_logical_bytes": RESIDENT_SENTINEL_LOGICAL_BYTES,
                "resident_sentinel_allocator_charge_bytes": RESIDENT_ALLOCATOR_CHARGE_BYTES, **allocation})

    add_request = scheduler._add_request_to_queue

    @wraps(add_request)
    def queue_request(self, req, is_retracted=False):
        if (is_retracted or req.rid in admitted or self.ps.tp_rank != rank or self.enable_overlap
                or self.hisparse_coordinator._resident_proof != proof):
            raise RuntimeError("unexpected retraction, duplicate or scheduler in resident-only replay")
        check_request(bounds, req.rid, req.origin_input_ids, req.sampling_params.max_new_tokens)
        admitted.add(req.rid)
        return add_request(self, req, is_retracted=is_retracted)

    prepare_extend = schedule_batch.prepare_for_extend
    prepare_decode = schedule_batch.prepare_for_decode

    @wraps(prepare_extend)
    def extend(self):
        for req in self.reqs:
            bound = bounds.get(req.rid)
            if req.rid not in admitted or bound is None:
                raise RuntimeError("unproved request reached resident prefill allocation")
            check_length(bound, req.extend_range.end)
        return prepare_extend(self)

    @wraps(prepare_decode)
    def decode(self):
        lengths = self.seq_lens_cpu.tolist()
        if len(lengths) != len(self.reqs):
            raise RuntimeError("resident decode request/length cardinality differs")
        for req, length in zip(self.reqs, lengths):
            bound = bounds.get(req.rid)
            if req.rid not in admitted or bound is None:
                raise RuntimeError("unproved request reached resident decode allocation")
            check_length(bound, length + 1)
        return prepare_decode(self)

    def stage(self, req):
        slot = req.kv.req_pool_idx
        bound = bounds.get(req.rid)
        if req.rid not in admitted or bound is None or slot in self._resident_bounds_by_slot:
            raise RuntimeError("unproved or reused request slot reached resident staging")
        check_length(bound, req.extend_range.end)
        check_length(bound, req.kv.kv_allocated_len)
        self._resident_bounds_by_slot[slot] = bound
        _stage_without_backup(self, req, coordinator_module.device_module, coordinator_module.HiSparseAct)

    def omit_decode_backup(self, _seq_lens, _req_pool_indices, seq_lens_cpu, req_pool_indices_cpu):
        _no_cold_allocations(self)
        lengths, slots = seq_lens_cpu.tolist(), req_pool_indices_cpu.tolist()
        if len(lengths) != len(slots):
            raise RuntimeError("resident backup metadata cardinality differs")
        for slot, length in zip(slots, lengths):
            check_length(self._resident_bounds_by_slot[slot], length)
            if self.req_to_host_pool_allocated_len[slot]:
                raise RuntimeError("resident-only request acquired cold slots")
            self._skip_first_backup[slot] = False

    for method in ("request_finished", "abort_staging_request"):
        original = getattr(coordinator, method)

        def wrap_release(native):
            @wraps(native)
            def release(self, req):
                _no_cold_allocations(self)
                slot = req.kv.req_pool_idx
                outcome = native(self, req)
                self._resident_bounds_by_slot.pop(slot, None)
                return outcome
            return release

        setattr(coordinator, method, wrap_release(original))

    coordinator.__init__ = initialize
    coordinator.admit_request_into_staging = stage
    coordinator._eager_backup_previous_token = omit_decode_backup
    coordinator.admit_request_direct = unavailable
    coordinator._preload_to_device_buffer = unavailable
    coordinator.naive_load_topk = unavailable
    coordinator._resident_only_installed = True
    scheduler._add_request_to_queue = queue_request
    schedule_batch.prepare_for_extend = extend
    schedule_batch.prepare_for_decode = decode


def install(proof, *, engine, rank, report):
    """Bind the draft in an isolated worker. The launcher must supply the proved engine."""
    import sglang
    import torch
    from .sglang_sources import check_sources
    from sglang.srt.environ import envs
    from sglang.srt.managers import hisparse_coordinator
    from sglang.srt.managers.schedule_batch import ScheduleBatch
    from sglang.srt.managers.scheduler import Scheduler

    if type(proof) is not ResidentProof or not callable(report):
        raise TypeError("resident-only requires a compiler proof and trusted reporting callback")
    if fingerprint(engine) != proof.engine_sha256:
        raise ValueError("resident-only engine differs from the compiled proof")
    root = check_sources(sglang)
    for relative, expected in EXTRA_SOURCES.items():
        if sha256((Path(root) / relative).read_bytes()).hexdigest() != expected:
            raise ValueError(f"resident-only source identity differs for {relative}")
    if envs.SGLANG_DSA_FUSE_TOPK.get() or not envs.SGLANG_DISABLE_HISPARSE_PREFETCH.get():
        raise ValueError("resident-only requires the already qualified unfused selector and no prefetch")
    if getattr(hisparse_coordinator, "MLATokenToKVPoolHost", None) is None:
        raise ValueError("resident-only requires the stock CUDA MLA host constructor")
    _patched_classes(proof, rank, torch, hisparse_coordinator, Scheduler, ScheduleBatch, report)
