"""Qualify physical cold-KV routes with the native SGLang write and swap kernels."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median
from types import SimpleNamespace

from .sglang_backing import check_sources, enable_direct_peer
from .sglang_relay import StagedPeerRelay, workspace_bytes


HOT = 4096
TOP_K = 2048
ROWS = 8192
ROW_BYTES = 656


def _samples(torch, operation, before, *, warmup, count):
    for _ in range(warmup):
        before()
        operation()
    torch.cuda.synchronize()
    measurements = []
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(count):
        before()
        start.record()
        operation()
        end.record()
        end.synchronize()
        measurements.append(start.elapsed_time(end) * 1000)
    return {"samples_us": measurements, "median_us": median(measurements),
            "p95_us": sorted(measurements)[(95 * count + 99) // 100 - 1]}


def _route(torch, hisparse, hicache, pointer_table, destination, *, warmup, count, staged=False):
    from sgl_kernel.kvcacheio import transfer_kv_all_layer_mla

    compute = torch.device("cuda", torch.cuda.current_device())
    host = destination.type == "cpu"
    generator = torch.Generator().manual_seed(19)
    reference = torch.randint(0, 256, (ROWS, 1, ROW_BYTES), dtype=torch.uint8, generator=generator)
    source = reference.to(compute)
    cold = torch.empty(source.shape, dtype=source.dtype, device=destination, pin_memory=host)
    ids = torch.arange(ROWS, dtype=torch.int64, device=compute)
    src = pointer_table([source], compute, host_memory_registered=False)
    relay = StagedPeerRelay(torch, compute, destination, TOP_K, ROW_BYTES, pointer_table) if staged else None
    dst = None if staged else pointer_table([cold], compute, host_memory_registered=host)

    def backup():
        if relay is not None:
            relay.store(cold, src, ids, ids, use_jit=use_jit)
        elif use_jit:
            hicache.transfer_hicache_all_layer_mla(ptr_dst=dst, indices_dst=ids, ptr_src=src, indices_src=ids,
                                                  cache_dst_stride_bytes=ROW_BYTES, cache_src_stride_bytes=ROW_BYTES,
                                                  element_size=ROW_BYTES)
        else:
            transfer_kv_all_layer_mla(src_layers=src, dst_layers=dst, src_indices=ids, dst_indices=ids,
                                      item_size=ROW_BYTES, num_layers=1)

    use_jit = hicache.can_use_hicache_jit_kernel(element_size=ROW_BYTES)
    backup()
    torch.cuda.synchronize()
    if not torch.equal(cold.cpu(), reference):
        raise RuntimeError("native KV backup bytes differ on the selected route")
    backup_timing = _samples(torch, backup, lambda: None, warmup=warmup, count=count)

    logical_hot = torch.cat((ids[:HOT], ids[-1:]))
    hot = source.index_select(0, logical_hot)
    initial_slots = logical_hot.to(torch.int32).view(1, -1)
    slots = initial_slots.clone()
    initial_lru = ids[:HOT].to(torch.int16).view(1, -1)
    lru = initial_lru.clone()
    selected = ids[HOT:HOT + TOP_K].to(torch.int32).view(1, -1)
    topk_locations = torch.empty_like(selected)
    slot_locations = torch.arange(HOT + 1, dtype=torch.int32, device=compute).view(1, -1)
    req = torch.zeros(1, dtype=torch.int64, device=compute)
    seq = torch.full((1,), ROWS, dtype=torch.int32, device=compute)
    active = torch.ones(1, dtype=torch.int32, device=compute)
    coordinator = SimpleNamespace(skip_io=False, enable_prefetch=False, top_k_device_locs_buffer=topk_locations,
                                  mem_pool_host=SimpleNamespace(kv_buffer=[cold]),
                                  mem_pool_device=SimpleNamespace(kv_buffer=[hot]),
                                  req_device_buffer_tokens=[slots], req_to_host_pool=ids.view(1, -1),
                                  req_device_buffer_token_locs=[slot_locations], lru_slots=[lru],
                                  device_buffer_size=HOT, swap_in_block_size=960, num_real_reqs=active)

    def reset():
        slots.copy_(initial_slots)
        lru.copy_(initial_lru)

    def swap():
        if relay is not None:
            relay.swap(coordinator, req, seq, selected, 0, record_plan=False)
            return
        hisparse.load_cache_to_device_buffer_mla(
            top_k_tokens=selected, device_buffer_tokens=slots, host_cache_locs=ids.view(1, -1),
            device_buffer_locs=slot_locations, host_cache=cold, device_buffer=hot, top_k_device_locs=topk_locations,
            req_pool_indices=req, seq_lens=seq, lru_slots=lru, item_size_bytes=ROW_BYTES, num_top_k=TOP_K,
            hot_buffer_size=HOT, block_size=960, num_real_reqs=active)

    swap_timing = _samples(torch, swap, reset, warmup=warmup, count=count)
    locations = topk_locations.flatten().long()
    selected_payload = hot.index_select(0, locations).cpu()
    if (locations.unique().numel() != TOP_K or not torch.equal(slots.flatten().index_select(0, locations), selected.flatten())
            or not torch.equal(selected_payload, reference[HOT:HOT + TOP_K])):
        raise RuntimeError("native HiSparse swap did not deliver the expected logical KV rows")
    if relay is not None:
        relay.close()
    return {"compute_device": compute.index, "backing_device": str(destination), "backup_jit": use_jit,
            "transport": "pinned_relay" if staged else "pinned_host" if host else "direct_peer",
            "backup": {"payload_bytes": ROWS * ROW_BYTES, **backup_timing},
            "all_miss_swap": {"payload_bytes": TOP_K * ROW_BYTES, **swap_timing},
            "selected_sha256": sha256(selected_payload.numpy().tobytes()).hexdigest(), "exact_bytes": True,
            "relay_workspace_bytes": workspace_bytes(TOP_K, ROW_BYTES) if staged else None}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--compute-device", type=int, default=0)
    parser.add_argument("--peer-device", type=int, default=1)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--samples", type=int, default=30)
    parser.add_argument("--mode", choices=("direct", "staged"), default="direct")
    args = parser.parse_args()
    if args.warmup < 1 or args.samples < 1 or args.compute_device == args.peer_device:
        parser.error("expected positive samples and two distinct CUDA ordinals")

    report = {"schema": 1, "status": "started", "rows": ROWS, "row_bytes": ROW_BYTES, "hot_slots": HOT,
              "top_k": TOP_K, "warmup": args.warmup, "samples": args.samples,
              "scope": "Native GPU KV transfer and swap kernel timings, with CPU byte oracle. No serving timing."}
    with args.output.open("x", encoding="utf-8") as out:
        try:
            import sglang
            import torch

            check_sources(sglang)
            from sglang.kernels.ops.kvcache import hicache, hisparse
            from sglang.srt.mem_cache.pool_host.common import make_kernel_ptr_table

            visible = torch.cuda.device_count()
            if not 0 <= args.compute_device < visible or not 0 <= args.peer_device < visible:
                raise ValueError("both CUDA ordinals must be visible")
            report["devices"] = {index: {"name": torch.cuda.get_device_name(index),
                                         "capability": torch.cuda.get_device_capability(index)}
                                 for index in (args.compute_device, args.peer_device)}
            report["peer_capability"] = [torch.cuda.can_device_access_peer(a, b) for a, b in
                                         ((args.compute_device, args.peer_device), (args.peer_device, args.compute_device))]
            with torch.cuda.device(args.compute_device):
                report["host"] = _route(torch, hisparse, hicache, make_kernel_ptr_table, torch.device("cpu"),
                                        warmup=args.warmup, count=args.samples)
                if args.mode == "direct" and not report["peer_capability"][0]:
                    report["status"] = "peer_unavailable"
                    return 2
                if args.mode == "direct":
                    enable_direct_peer(torch, args.compute_device, args.peer_device)
                report["peer"] = _route(torch, hisparse, hicache, make_kernel_ptr_table, torch.device("cuda", args.peer_device),
                                        warmup=args.warmup, count=args.samples, staged=args.mode == "staged")
            if report["host"]["selected_sha256"] != report["peer"]["selected_sha256"]:
                raise RuntimeError("peer and native host swap returned different selected KV bytes")
            report["status"] = "verified"
        except Exception as exc:
            report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
            raise
        finally:
            json.dump(report, out, indent=2, allow_nan=False)
            out.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
