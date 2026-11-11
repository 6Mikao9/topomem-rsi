"""Measure directed pinned-host links on the selected physical CUDA devices."""

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
from statistics import median
from time import perf_counter_ns


def profile(*, sizes=(4096, 65536, 4194304), samples=30, warmup=10):
    import torch

    records, devices = [], []
    for index in range(torch.cuda.device_count()):
        properties = torch.cuda.get_device_properties(index)
        devices.append({"ordinal": index, "name": properties.name, "uuid": str(properties.uuid),
                        "total_memory": properties.total_memory, "capability": [properties.major, properties.minor]})
        for size in sizes:
            host = torch.empty(size, dtype=torch.uint8, pin_memory=True)
            host.copy_(torch.arange(size, dtype=torch.int64).remainder(251).byte())
            device = torch.empty(size, dtype=torch.uint8, device=f"cuda:{index}")
            for direction in ("h2d", "d2h"):
                if direction == "d2h":
                    device.copy_(host)
                source, target = (host, device) if direction == "h2d" else (device, host)
                times = []
                for repeat in range(warmup + samples):
                    torch.cuda.synchronize(index)
                    before = perf_counter_ns()
                    target.copy_(source, non_blocking=True)
                    torch.cuda.synchronize(index)
                    elapsed = perf_counter_ns() - before
                    if repeat >= warmup:
                        times.append(elapsed)
                if not torch.equal(device.cpu(), host):
                    raise RuntimeError("directed link failed its byte oracle")
                records.append({"gpu_ordinal": index, "direction": direction, "bytes": size,
                                "samples_ns": times, "median_ns": median(times), "exact_bytes": True})
    edges = []
    for device in devices:
        for direction in ("h2d", "d2h"):
            points = [row for row in records if row["gpu_ordinal"] == device["ordinal"] and row["direction"] == direction]
            small, large = min(points, key=lambda x: x["bytes"]), max(points, key=lambda x: x["bytes"])
            slope = max(1e-6, (large["median_ns"]-small["median_ns"]) / (large["bytes"]-small["bytes"]))
            edges.append({"gpu_ordinal": device["ordinal"], "direction": direction,
                          "bandwidth_bytes_per_s": max(1, int(1e9/slope)),
                          "latency_ns": max(0, int(small["median_ns"]-small["bytes"]*slope))})
    return {"schema": 1, "utc": datetime.now(timezone.utc).isoformat(),
            "scope": "synchronized_directed_pinned_host_copy_not_attention_latency",
            "torch_version": torch.__version__, "devices": devices,
            "host_cpu_affinity": sorted(os.sched_getaffinity(0)),
            "peer_capability": [[torch.cuda.can_device_access_peer(a, b) if a != b else True
                                 for b in range(len(devices))] for a in range(len(devices))],
            "warmup": warmup, "samples": samples, "records": records, "affine_link_fits": edges,
            "fit_limitation": "two endpoint robust medians; middle size is retained as a diagnostic; contention not inferred"}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cpu", type=int)
    args = parser.parse_args()
    if args.cpu is not None:
        if args.cpu not in os.sched_getaffinity(0):
            raise ValueError("requested profiling CPU is outside the container cpuset")
        os.sched_setaffinity(0, {args.cpu})
    result = profile()
    with args.output.open("x") as stream:
        json.dump(result, stream, indent=2, allow_nan=False)
        stream.write("\n")


if __name__ == "__main__":
    main()
