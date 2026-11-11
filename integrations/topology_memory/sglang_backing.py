"""Tiered cold backing for native SGLang HiSparse MLA.

The tier choice applies to the complete cold KV slab of one layer. Logical
request slots, the native hot pool, swap/eviction, and attention stay native.
Peer backing requires direct CUDA peer addressing from the compute worker.
"""

from dataclasses import dataclass
from functools import wraps

from .sglang_relay import StagedPeerRelay, transfer_all_layers, workspace_bytes
from .sglang_sources import SOURCES, check_sources


HOST = "host"
LOCAL_HBM = "local_hbm"
PEER_HBM = "peer_hbm"
PEER_RELAY = "peer_relay"
NATIVE_TRANSPORTS = frozenset({"pinned_h2d", "pinned_d2h", "cuda_p2p"})
@dataclass(frozen=True)
class RankBacking:
    rank: int
    rows: int
    row_bytes: int
    page_size: int
    top_k: int
    hot_slots: int
    layer_tiers: tuple[str, ...]
    local_hbm_cap_bytes: int
    pinned_host_cap_bytes: int
    world_size: int = 2
    peer_device: int | None = None
    peer_hbm_cap_bytes: int = 0
    compute_device: int | None = None
    attention_tp_size: int | None = None

    def __post_init__(self):
        if (type(self.world_size) is not int or self.world_size not in (1, 2)
                or type(self.rank) is not int or not 0 <= self.rank < self.world_size):
            raise ValueError("ordinary HiSparse backing supports one or two fixed compute ranks")
        if self.attention_tp_size is not None and (type(self.attention_tp_size) is not int
                                                   or self.attention_tp_size not in (1, self.world_size)):
            raise ValueError("invalid local attention TP width for the fixed global ranks")
        dimensions = (self.rows, self.row_bytes, self.page_size, self.top_k, self.hot_slots)
        if any(type(value) is not int or value <= 0 for value in dimensions):
            raise ValueError("HiSparse backing dimensions must be positive integers")
        if self.rows % self.page_size or self.hot_slots % self.page_size or not self.top_k <= self.hot_slots < self.rows:
            raise ValueError("incompatible HiSparse backing geometry")
        if (type(self.layer_tiers) is not tuple or not self.layer_tiers
                or any(type(tier) is not str or tier not in (HOST, LOCAL_HBM, PEER_HBM, PEER_RELAY) for tier in self.layer_tiers)):
            raise ValueError("each local layer requires exactly one supported backing tier")
        if self.peer_hbm_bytes and (type(self.peer_device) is not int or self.peer_device < 0):
            raise ValueError("peer backing requires an explicit visible CUDA device ordinal")
        if self.compute_device is not None and (type(self.compute_device) is not int or self.compute_device < 0
                                                or self.peer_hbm_bytes and self.peer_device == self.compute_device):
            raise ValueError("invalid fixed compute and peer CUDA device assignment")
        relay = self.relay_workspace_bytes
        for cap, allocated in ((self.local_hbm_cap_bytes, self.local_hbm_bytes + relay["compute"]),
                               (self.pinned_host_cap_bytes, self.pinned_host_bytes + relay["host"]),
                               (self.peer_hbm_cap_bytes, self.peer_hbm_bytes + relay["peer"])):
            if type(cap) is not int or allocated > cap:
                raise ValueError("HiSparse cold backing exceeds its tier cap")

    @property
    def layer_bytes(self):
        return self.rows * self.row_bytes

    @property
    def local_hbm_bytes(self):
        return self.layer_tiers.count(LOCAL_HBM) * self.layer_bytes

    @property
    def pinned_host_bytes(self):
        return self.layer_tiers.count(HOST) * self.layer_bytes

    @property
    def peer_hbm_bytes(self):
        return (self.layer_tiers.count(PEER_HBM) + self.layer_tiers.count(PEER_RELAY)) * self.layer_bytes

    @property
    def relay_workspace_bytes(self):
        if PEER_RELAY in self.layer_tiers:
            native_layers = len(self.layer_tiers) - self.layer_tiers.count(PEER_RELAY)
            return workspace_bytes(self.top_k, self.row_bytes, native_layers)
        return {"compute": 0, "peer": 0, "host": 0}


def _check_route(links, path, source, destination, transport):
    if source == destination and path == []:
        return
    if type(path) is not list or len(path) != 1 or type(path[0]) is not str:
        raise ValueError("native backing requires a single measured directed link per nonlocal transfer")
    link = links.get(path[0])
    if (link is None or (link.src, link.dst, link.transport) != (source, destination, transport)
            or link.evidence_sha256 is None):
        raise ValueError("native backing transfer differs from its measured topology link")


def _relay_host(topology, links, read, write, peer, local):
    if not all(type(path) is list and len(path) == 2 and all(type(link) is str and link in links for link in path)
               for path in (read, write)):
        raise ValueError("staged peer backing requires two measured links per direction")
    host = links[read[0]].dst
    if topology.node(host).kind != "host":
        raise ValueError("staged peer backing requires a pinned host relay node")
    for path, start, end in ((read, peer, local), (write, local, peer)):
        _check_route(links, path[:1], start, host, "pinned_d2h")
        _check_route(links, path[1:], host, end, "pinned_h2d")
    return host


def bind_plans(topology, scenario, plan, *, gpu_ordinals, rows, page_size, layers, hot_slots, initial_used=None,
               attention_tp_size=None):
    """Check generated slab decisions and convert them to executable worker plans.

    The adapter supports local, pinned host, direct peer, and two-hop pinned
    relay transfers. The requested edges must have recorded measurements.
    """
    scenario.validate_topology(topology)
    ranks = tuple(node for _, node in scenario.rank_placement)
    if len(ranks) not in (1, 2) or len(set(ranks)) != len(ranks):
        raise ValueError("the adapter requires one fixed compute rank per CUDA device")
    if scenario.family != "dsa" or scenario.selection_unit != "token" or scenario.kv_row_bytes != 656:
        raise ValueError("the adapter requires the audited ordinary DSA packed KV layout")
    if any(type(value) is not int or value <= 0 for value in (rows, page_size, layers, hot_slots)):
        raise ValueError("invalid external cold-pool geometry")
    if (type(gpu_ordinals) is not dict or any(type(i) is not int or i < 0 for i in gpu_ordinals.values())
            or len(gpu_ordinals) != len(set(gpu_ordinals.values())) or not set(ranks) <= gpu_ordinals.keys()
            or any(topology.node(node).kind != "gpu" for node in gpu_ordinals)):
        raise ValueError("visible CUDA ordinals do not bind the target's GPU nodes")
    if (type(plan) is not dict or type(plan.get("schema")) is not int or plan["schema"] != 1
            or plan.get("topology_sha256") != topology.fingerprint() or plan.get("scenario_sha256") != scenario.fingerprint()
            or type(plan.get("actions")) is not list or len(plan["actions"]) != len(ranks) * layers):
        raise ValueError("generated plan does not bind the topology, scenario, and all KV slabs")

    capacities = {node.id: node.capacity_bytes for node in topology.nodes}
    for node, bytes_in_use in (initial_used or {}).items():
        if node not in capacities or type(bytes_in_use) is not int or not 0 <= bytes_in_use <= capacities[node]:
            raise ValueError("invalid fixed initial occupancy")
        capacities[node] -= bytes_in_use
    used = {node.id: 0 for node in topology.nodes}
    links = {link.id: link for link in topology.links}
    key_fields = {"action", "key", "rank", "layer", "bytes", "node", "read_links", "write_links", "rule_index"}
    selected, keys = {}, set()
    for action in plan["actions"]:
        if (type(action) is not dict or action.keys() != key_fields or action["action"] != "reserve_backing"
                or type(action["key"]) is not str or action["key"] in keys
                or type(action["rank"]) is not int or not 0 <= action["rank"] < len(ranks)
                or type(action["layer"]) is not int or not 0 <= action["layer"] < layers
                or type(action["read_links"]) is not list or type(action["write_links"]) is not list
                or type(action["bytes"]) is not int or action["bytes"] != rows * scenario.kv_row_bytes):
            raise ValueError("invalid generated KV slab reservation")
        rank, layer, node = action["rank"], action["layer"], action["node"]
        if (rank, layer) in selected or type(node) is not str:
            raise ValueError("duplicate or malformed rank/layer reservation")
        local = ranks[rank]
        stage = None
        if node == local:
            tier, read, write = LOCAL_HBM, "", ""
        elif topology.node(node).kind == "host":
            tier, read, write = HOST, "pinned_h2d", "pinned_d2h"
        elif node in gpu_ordinals:
            tier, read, write = PEER_HBM, "cuda_p2p", "cuda_p2p"
            if len(action["read_links"]) == 2 or len(action["write_links"]) == 2:
                stage = _relay_host(topology, links, action["read_links"], action["write_links"], node, local)
                tier = PEER_RELAY
        else:
            raise ValueError("generated backing selected an unsupported storage node")
        if stage is None:
            _check_route(links, action["read_links"], node, local, read)
            _check_route(links, action["write_links"], local, node, write)
        keys.add(action["key"])
        used[node] += action["bytes"]
        selected[rank, layer] = (tier, node, stage)
    if any(used[node] > capacities[node] for node in used):
        raise ValueError("generated cold backing exceeds the shared topology budget")
    reported_usage = {node: used[node] + (initial_used or {}).get(node, 0) for node in used}
    if reported_usage != plan.get("used_bytes_by_node"):
        raise ValueError("generated plan backing totals do not match its reservations")

    workers = []
    for rank, local in enumerate(ranks):
        placements = [selected[rank, layer] for layer in range(layers)]
        host_nodes = {node for tier, node, _ in placements if tier == HOST}
        host_nodes.update(stage for _, _, stage in placements if stage is not None)
        peer_nodes = {node for tier, node, _ in placements if tier in (PEER_HBM, PEER_RELAY)}
        if len(host_nodes) > 1 or len(peer_nodes) > 1:
            raise ValueError("the native worker supports at most one host pool and one peer GPU")
        host = next(iter(host_nodes), None)
        peer = next(iter(peer_nodes), None)
        worker = RankBacking(rank=rank, rows=rows, row_bytes=scenario.kv_row_bytes, page_size=page_size,
                             top_k=scenario.top_k, hot_slots=hot_slots, layer_tiers=tuple(tier for tier, _, _ in placements),
                             local_hbm_cap_bytes=capacities[local], pinned_host_cap_bytes=capacities.get(host, 0),
                             world_size=len(ranks), peer_device=gpu_ordinals.get(peer), peer_hbm_cap_bytes=capacities.get(peer, 0),
                             compute_device=gpu_ordinals[local], attention_tp_size=attention_tp_size)
        workspace = worker.relay_workspace_bytes
        for node, amount in ((local, workspace["compute"]), (peer, workspace["peer"]), (host, workspace["host"])):
            if amount:
                used[node] += amount
        workers.append(worker)
    if any(used[node] > capacities[node] for node in used):
        raise ValueError("cold backing plus relay workspaces exceeds the shared topology budget")
    return tuple(workers)


def _check_worker_layout(coordinator, plan):
    local_tp = plan.world_size if plan.attention_tp_size is None else plan.attention_tp_size
    if (coordinator.tp_world_size != local_tp or coordinator.is_dsv4_hisparse or coordinator.enable_prefetch
            or coordinator.skip_io or coordinator.compress_ratio != 1):
        raise ValueError("tiered backing requires synchronous ordinary MLA with live KV copying")
    pool = coordinator.mem_pool_host
    if (pool.layout != "layer_first" or pool.layer_num != len(plan.layer_tiers) or pool.size != plan.rows
            or coordinator.item_size_bytes != plan.row_bytes or coordinator.page_size != plan.page_size
            or coordinator.top_k != plan.top_k or coordinator.device_buffer_size != plan.hot_slots):
        raise ValueError("native HiSparse geometry differs from the bound cold backing plan")


def _check_pool_buffers(pool, device, torch, plan):
    buffers = [pool.kv_buffer[layer] for layer in range(len(plan.layer_tiers))]
    for buffer, tier in zip(buffers, plan.layer_tiers, strict=True):
        if (tuple(buffer.shape) != (plan.rows, 1, plan.row_bytes) or buffer.dtype != torch.uint8
                or not buffer.is_contiguous() or buffer.nbytes != plan.layer_bytes):
            raise ValueError("cold backing must use contiguous packed uint8 rows")
        if tier == HOST:
            if buffer.device.type != "cpu" or not buffer.is_pinned():
                raise ValueError("host cold backing must be pinned and device-accessible")
        elif tier == LOCAL_HBM and buffer.device != device:
            raise ValueError("HBM cold backing is not on the rank's compute GPU")
        elif tier in (PEER_HBM, PEER_RELAY) and (buffer.device.type != "cuda" or buffer.device.index != plan.peer_device
                                               or buffer.device == device):
            raise ValueError("peer cold backing is not on the bound remote CUDA device")


def _check_rank_device(torch, coordinator):
    device = coordinator.mem_pool_device.kv_buffer[0].device
    if device.type != "cuda" or device.index != torch.cuda.current_device():
        raise ValueError("HiSparse backing must be initialized on the active rank CUDA device")
    return device


def enable_direct_peer(torch, compute_device, peer_device):
    """Enable raw peer addressing in the worker. Runtime probes still verify copies."""
    if (compute_device == peer_device or not 0 <= peer_device < torch.cuda.device_count()
            or not torch.cuda.can_device_access_peer(compute_device, peer_device)):
        raise RuntimeError("direct CUDA peer access is unavailable for the bound device pair")
    if torch.cuda.get_allocator_backend() != "native":
        raise RuntimeError("direct backing currently requires the native CUDA allocator")
    from cupy.cuda import runtime

    with torch.cuda.device(compute_device):
        try:
            runtime.deviceEnablePeerAccess(peer_device, 0)
        except runtime.CUDARuntimeError as exc:
            if exc.status != 704:  # cudaErrorPeerAccessAlreadyEnabled
                raise


def _pointer_table(torch, original, buffers, device):
    cpu_indices = [index for index, buffer in enumerate(buffers) if buffer.device.type == "cpu"]
    gpu_indices = [index for index, buffer in enumerate(buffers) if buffer.device.type == "cuda"]
    if len(cpu_indices) + len(gpu_indices) != len(buffers):
        raise ValueError("unexpected backing tensor device")
    parts = []
    for indices, registered in ((cpu_indices, True), (gpu_indices, False)):
        if indices:
            refs = [buffers[index] for index in indices]
            parts.append(original(refs, device, host_memory_registered=registered))
    if len(parts) == 1:
        return parts[0]
    order = {layer: offset for offset, layer in enumerate(cpu_indices + gpu_indices)}
    indices = torch.tensor([order[layer] for layer in range(len(buffers))], dtype=torch.long, device=device)
    # Index an int64 view because some Torch builds lack uint64 gather kernels.
    return torch.cat([part.view(torch.int64) for part in parts]).index_select(0, indices).view(parts[0].dtype)


def _tiered_pool_type(native_module, torch, plan, report=lambda _event: None, *, count_relay_misses=False):
    native_type = native_module.MLATokenToKVPoolHost

    class TieredMLABacking(native_type):
        def __init__(self, *args, **kwargs):
            original = native_module.make_kernel_ptr_table

            def build_ptrs(buffers, device, *, host_memory_registered):
                if (len(buffers) == len(self._cold_slabs)
                        and all(left is right for left, right in zip(buffers, self._cold_slabs, strict=True))):
                    return _pointer_table(torch, original, buffers, device)
                return original(buffers, device, host_memory_registered=host_memory_registered)

            self._cold_slabs = ()
            self._relay = None
            native_module.make_kernel_ptr_table = build_ptrs
            try:
                super().__init__(*args, **kwargs)
            finally:
                native_module.make_kernel_ptr_table = original
            if not self._cold_slabs:
                raise ValueError("native pool did not initialize the planned cold backing")
            if PEER_RELAY in plan.layer_tiers:
                compute = self.device_pool.kv_buffer[0].device
                peer = torch.device("cuda", plan.peer_device)
                self._relay = StagedPeerRelay(torch, compute, peer, plan.top_k, plan.row_bytes, original,
                                              count_misses=count_relay_misses)
                self._native_layers = tuple(i for i, tier in enumerate(plan.layer_tiers) if tier != PEER_RELAY)
                indices = torch.tensor(self._native_layers, dtype=torch.long, device=compute)
                self._native_sources = self.device_pool.data_ptrs.view(torch.int64).index_select(0, indices).view(torch.uint64)
                self._native_destinations = self.data_ptrs.view(torch.int64).index_select(0, indices).view(torch.uint64)

        def init_kv_buffer(self):
            if (self.layout != "layer_first" or self.size != plan.rows or self.layer_num != len(plan.layer_tiers)
                    or self.kv_cache_dim != plan.row_bytes or self.dtype != torch.uint8 or self.dcp_size != 1
                    or self.mtp_draft_device_pools or self._is_dummy or not self.pin_memory
                    or self.device_pool.layer_shard_enabled):
                raise ValueError("native host allocator geometry differs from the tiered plan")
            compute = self.device_pool.kv_buffer[0].device
            if (compute.type != "cuda" or compute.index != torch.cuda.current_device()
                    or plan.compute_device is not None and compute.index != plan.compute_device):
                raise ValueError("HiSparse backing received the wrong compute device")
            if PEER_HBM in plan.layer_tiers:
                enable_direct_peer(torch, compute.index, plan.peer_device)
            peer = None if plan.peer_device is None else torch.device("cuda", plan.peer_device)
            destinations = {HOST: torch.device("cpu"), LOCAL_HBM: compute, PEER_HBM: peer, PEER_RELAY: peer}
            self.token_stride_size = plan.row_bytes
            self.layout_dim = plan.row_bytes * self.layer_num
            shape = (plan.rows, 1, plan.row_bytes)
            self._cold_slabs = tuple(torch.empty(shape, dtype=self.dtype, device=destinations[tier],
                                                 pin_memory=tier == HOST) for tier in plan.layer_tiers)
            return self._cold_slabs

        def backup_from_device_all_layer(self, device_pool, host_indices, device_indices, io_backend):
            if self._relay is None:
                return super().backup_from_device_all_layer(device_pool, host_indices, device_indices, io_backend)
            if device_pool is not self.device_pool or io_backend != "kernel":
                raise ValueError("peer relay accepts only native staged MLA write-back")
            if self._native_layers:
                transfer_all_layers(self._relay.hicache, self._relay.fallback, self._native_sources,
                                    self._native_destinations, device_indices, host_indices, plan.row_bytes,
                                    len(self._native_layers), self.can_use_jit)
            for layer, tier in enumerate(plan.layer_tiers):
                if tier == PEER_RELAY:
                    self._relay.store(self.kv_buffer[layer], device_pool.data_ptrs[layer:layer + 1],
                                      host_indices, device_indices, use_jit=self.can_use_jit)

        def load_to_device_per_layer(self, device_pool, host_indices, device_indices, layer_id, io_backend, **kwargs):
            if plan.layer_tiers[layer_id] == PEER_RELAY:
                raise ValueError("peer relay is only supported by the ordinary decode swap path")
            return super().load_to_device_per_layer(device_pool, host_indices, device_indices, layer_id, io_backend, **kwargs)

        def destroy(self):
            if self._cold_slabs:
                # The native coordinator drains its backup streams first.
                # These slabs use Torch's owning allocators, not explicit host registration.
                if self._relay is not None:
                    self._relay.close()
                    report({"event": "hisparse_cold_relay_usage", "rank": plan.rank, **self._relay.statistics()})
                    self._relay = None
                torch.cuda.synchronize(self.device_pool.kv_buffer[0].device)
                self.data_ptrs = None
                self.data_refs = []
                self.kv_buffer = None
                self._cold_slabs = ()

    return TieredMLABacking


def _wrap_coordinator(original, torch, plan, report):
    @wraps(original)
    def checked(self, *args, **kwargs):
        original(self, *args, **kwargs)
        _check_worker_layout(self, plan)
        device = _check_rank_device(torch, self)
        if plan.compute_device is not None and plan.compute_device != device.index:
            raise ValueError("worker CUDA ordinal differs from the frozen compute placement")
        _check_pool_buffers(self.mem_pool_host, device, torch, plan)
        hot_metadata = (self.req_to_device_buffer, self.req_device_buffer_tokens,
                        self.req_device_buffer_token_locs, self._lru_init, self.lru_slots,
                        self._device_buffer_arange_i32)
        physical = {"packed_device_payload_bytes": sum(buffer.nbytes for buffer in self.mem_pool_device.kv_buffer),
                    "device_physical_rows": self.mem_pool_device.kv_buffer[0].shape[0],
                    "device_logical_row_capacity": self.mem_pool_device.size,
                    "request_metadata_slots_including_pad": self.req_to_token_pool.req_to_token.shape[0],
                    "hot_metadata_bytes": sum(tensor.nbytes for tensor in hot_metadata)}
        report({"event": "hisparse_cold_backing_bound", "rank": plan.rank, "cuda_device": device.index,
                **physical,
                "rows": plan.rows, "row_bytes": plan.row_bytes, "top_k": plan.top_k, "hot_slots": plan.hot_slots,
                "layer_tiers": plan.layer_tiers, "pinned_host_bytes": plan.pinned_host_bytes,
                "local_hbm_bytes": plan.local_hbm_bytes, "local_hbm_cap_bytes": plan.local_hbm_cap_bytes,
                "pinned_host_cap_bytes": plan.pinned_host_cap_bytes, "peer_device": plan.peer_device,
                "peer_hbm_bytes": plan.peer_hbm_bytes, "peer_hbm_cap_bytes": plan.peer_hbm_cap_bytes,
                "relay_workspace_bytes": plan.relay_workspace_bytes, "attention_tp_size": self.tp_world_size,
                "global_compute_ranks": plan.world_size})

    return checked


def _wrap_swap(original, plan):
    @wraps(original)
    def swap(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id, record_plan=False):
        if plan.layer_tiers[layer_id] == PEER_RELAY:
            return self.mem_pool_host._relay.swap(self, req_pool_indices, compressed_seq_lens, top_k_result,
                                                  layer_id, record_plan=record_plan)
        return original(self, req_pool_indices, compressed_seq_lens, top_k_result, layer_id, record_plan=record_plan)

    return swap


def install(plan, report, *, count_relay_misses=False):
    """Install the selected backing in one spawned rank before model creation."""
    import sglang
    import torch

    check_sources(sglang)
    from sglang.srt.managers import hisparse_coordinator
    from sglang.srt.mem_cache.pool_host import mla

    if not isinstance(plan, RankBacking) or not callable(report):
        raise ValueError("tiered backing requires a bound rank plan and reporting sink")
    cls = hisparse_coordinator.HiSparseCoordinator
    if getattr(cls, "_rsi_cold_backing", False) or hisparse_coordinator.MLATokenToKVPoolHost is not mla.MLATokenToKVPoolHost:
        raise RuntimeError("HiSparse backing is already modified")
    if plan.local_hbm_bytes or plan.peer_hbm_bytes:
        hisparse_coordinator.MLATokenToKVPoolHost = _tiered_pool_type(mla, torch, plan, report, count_relay_misses=count_relay_misses)
    if PEER_RELAY in plan.layer_tiers:
        cls._run_swap_in_kernel = _wrap_swap(cls._run_swap_in_kernel, plan)
    cls.__init__ = _wrap_coordinator(cls.__init__, torch, plan, report)
    cls._rsi_cold_backing = True
