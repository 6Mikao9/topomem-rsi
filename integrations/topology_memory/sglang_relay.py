"""Explicit pinned-host relay between CUDA devices without peer addressing."""


RELAY_REQUESTS = 2
RELAY_BACKUP_ROWS = 2048


def workspace_bytes(top_k, row_bytes, native_layers=0):
    read_rows = RELAY_REQUESTS * top_k
    staging = (read_rows + RELAY_BACKUP_ROWS) * (row_bytes + 8)
    compute = read_rows * (8 + 4 + 8) + RELAY_REQUESTS * 4 + RELAY_BACKUP_ROWS * 8 + 8 + 2 * native_layers * 8
    return {"compute": compute, "peer": staging, "host": staging}


def transfer_all_layers(hicache, fallback, src_ptr, dst_ptr, src_ids, dst_ids, row_bytes, layers, use_jit):
    if use_jit:
        hicache.transfer_hicache_all_layer_mla(ptr_dst=dst_ptr, indices_dst=dst_ids, ptr_src=src_ptr,
                                              indices_src=src_ids, cache_dst_stride_bytes=row_bytes,
                                              cache_src_stride_bytes=row_bytes, element_size=row_bytes)
    else:
        fallback(src_layers=src_ptr, dst_layers=dst_ptr, src_indices=src_ids, dst_indices=dst_ids,
                 item_size=row_bytes, num_layers=layers)


class StagedPeerRelay:
    """Move actual KV through pinned slabs, using explicit cross-device event edges."""

    def __init__(self, torch, compute, peer, top_k, row_bytes, make_kernel_ptr_table, *, count_misses=False):
        if compute == peer:
            raise ValueError("relay compute and storage CUDA devices must differ")
        from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla
        from sglang.kernels.ops.kvcache import hicache, hisparse

        self.torch, self.compute, self.peer = torch, compute, peer
        self.top_k, self.row_bytes, self.count_misses = top_k, row_bytes, count_misses
        self.hicache, self.hisparse, self.fallback = hicache, hisparse, transfer_kv_all_layer_mla
        self.read_rows = RELAY_REQUESTS * top_k
        self.stream = torch.cuda.Stream(device=peer)
        self.read_input, self.read_ready, self.read_reuse = (torch.cuda.Event() for _ in range(3))
        self.write_input, self.write_ready, self.write_reuse = (torch.cuda.Event() for _ in range(3))
        self.read_active = self.write_active = False

        self.read_payload = torch.empty((self.read_rows, 1, row_bytes), dtype=torch.uint8, device="cpu", pin_memory=True)
        self.read_indices = torch.empty(self.read_rows, dtype=torch.int64, device="cpu", pin_memory=True)
        self.peer_read_payload = torch.empty_like(self.read_payload, device=peer)
        self.peer_read_indices = torch.empty_like(self.read_indices, device=peer)
        self.write_payload = torch.empty((RELAY_BACKUP_ROWS, 1, row_bytes), dtype=torch.uint8, device="cpu", pin_memory=True)
        self.write_indices = torch.empty(RELAY_BACKUP_ROWS, dtype=torch.int64, device="cpu", pin_memory=True)
        self.peer_write_payload = torch.empty_like(self.write_payload, device=peer)
        self.peer_write_indices = torch.empty_like(self.write_indices, device=peer)
        self.write_ptrs = make_kernel_ptr_table([self.write_payload], compute, host_memory_registered=True)
        self.write_order = torch.arange(RELAY_BACKUP_ROWS, dtype=torch.int64, device=compute)

        shape = (RELAY_REQUESTS, top_k)
        self.miss_src = torch.empty(shape, dtype=torch.int64, device=compute)
        self.miss_dst = torch.empty(shape, dtype=torch.int32, device=compute)
        self.miss_count = torch.empty(RELAY_REQUESTS, dtype=torch.int32, device=compute)
        self.relay_src = torch.arange(self.read_rows, dtype=torch.int64, device=compute).view(shape)
        self.read_calls = self.write_calls = self.read_payload_bytes_per_hop = self.write_payload_bytes_per_hop = 0
        self.read_index_bytes_per_hop = self.write_index_bytes_per_hop = 0
        self.logical_read_bytes = 0

    def store(self, backing, source_ptrs, host_indices, device_indices, *, use_jit):
        current = self.torch.cuda.current_stream(self.compute)
        for start in range(0, host_indices.numel(), RELAY_BACKUP_ROWS):
            end = min(start + RELAY_BACKUP_ROWS, host_indices.numel())
            rows = end - start
            if self.write_active:
                current.wait_event(self.write_reuse)
            host_rows, device_rows = host_indices[start:end], device_indices[start:end]
            self.write_indices[:rows].copy_(host_rows, non_blocking=True)
            transfer_all_layers(self.hicache, self.fallback, source_ptrs, self.write_ptrs,
                                device_rows, self.write_order[:rows], self.row_bytes, 1, use_jit)
            self.write_input.record(current)
            with self.torch.cuda.stream(self.stream):
                self.stream.wait_event(self.write_input)
                self.peer_write_indices[:rows].copy_(self.write_indices[:rows], non_blocking=True)
                self.peer_write_payload[:rows].copy_(self.write_payload[:rows], non_blocking=True)
                backing.index_copy_(0, self.peer_write_indices[:rows], self.peer_write_payload[:rows])
                self.write_ready.record(self.stream)
            current.wait_event(self.write_ready)
            self.write_reuse.record(current)
            self.write_active = True
            self.write_calls += 1
            self.write_payload_bytes_per_hop += rows * self.row_bytes
            self.write_index_bytes_per_hop += rows * 8

    def swap(self, coordinator, request_indices, seq_lens, selected, layer, *, record_plan):
        if record_plan or coordinator.skip_io or coordinator.enable_prefetch:
            raise ValueError("relay needs the ordinary synchronous HiSparse swap plan")
        requests = request_indices.size(0)
        if not 0 < requests <= RELAY_REQUESTS:
            raise ValueError("relay request batch exceeds its fixed scratch envelope")
        rows = requests * self.top_k
        result = coordinator.top_k_device_locs_buffer[:requests]
        cold = coordinator.mem_pool_host.kv_buffer[layer]
        hot = coordinator.mem_pool_device.kv_buffer[layer]
        current = self.torch.cuda.current_stream(self.compute)
        if self.read_active:
            current.wait_event(self.read_reuse)
        src, dst, count = self.miss_src[:requests], self.miss_dst[:requests], self.miss_count[:requests]
        src.zero_()
        self.hisparse.load_cache_to_device_buffer_mla(
            top_k_tokens=selected, device_buffer_tokens=coordinator.req_device_buffer_tokens[layer],
            host_cache_locs=coordinator.req_to_host_pool, device_buffer_locs=coordinator.req_device_buffer_token_locs[layer],
            host_cache=self.read_payload, device_buffer=hot, top_k_device_locs=result,
            req_pool_indices=request_indices, seq_lens=seq_lens, lru_slots=coordinator.lru_slots[layer],
            item_size_bytes=self.row_bytes, num_top_k=self.top_k, hot_buffer_size=coordinator.device_buffer_size,
            page_size=1, block_size=coordinator.swap_in_block_size, num_real_reqs=coordinator.num_real_reqs,
            miss_src=src, miss_dst=dst, miss_count=count, skip_io=True)
        self.read_indices[:rows].copy_(src.view(-1), non_blocking=True)
        self.read_input.record(current)
        with self.torch.cuda.stream(self.stream):
            self.stream.wait_event(self.read_input)
            self.peer_read_indices[:rows].copy_(self.read_indices[:rows], non_blocking=True)
            self.torch.index_select(cold, 0, self.peer_read_indices[:rows], out=self.peer_read_payload[:rows])
            self.read_payload[:rows].copy_(self.peer_read_payload[:rows], non_blocking=True)
            self.read_ready.record(self.stream)
        current.wait_event(self.read_ready)
        self.hisparse.copy_cache_planned_mla(miss_src=self.relay_src[:requests], miss_dst=dst, miss_count=count,
                                             num_real_reqs=coordinator.num_real_reqs, host_cache=self.read_payload,
                                             device_buffer=hot, item_size_bytes=self.row_bytes)
        self.read_reuse.record(current)
        self.read_active = True
        self.read_calls += 1
        self.read_payload_bytes_per_hop += rows * self.row_bytes
        self.read_index_bytes_per_hop += rows * 8
        if self.count_misses:
            real = int(coordinator.num_real_reqs.item())
            self.logical_read_bytes += int(count[:real].sum().item()) * self.row_bytes
        return result

    def close(self):
        self.stream.synchronize()
        self.torch.cuda.synchronize(self.compute)

    def statistics(self):
        return {"read_calls": self.read_calls, "write_chunks": self.write_calls,
                "peer_to_host_read_payload_bytes": self.read_payload_bytes_per_hop,
                "host_to_compute_read_payload_bytes": self.logical_read_bytes if self.count_misses else None,
                "host_to_compute_read_payload_upper_bound_bytes": self.read_payload_bytes_per_hop,
                "compute_to_host_write_payload_bytes": self.write_payload_bytes_per_hop,
                "host_to_peer_write_payload_bytes": self.write_payload_bytes_per_hop,
                "read_index_bytes_per_hop": self.read_index_bytes_per_hop,
                "write_index_bytes_per_hop": self.write_index_bytes_per_hop,
                "read_payload_scope": "fixed-size per-request relay slots, including inactive miss lanes"}
