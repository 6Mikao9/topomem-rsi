"""Start native HiSparse on a fixed one- or two-rank CUDA placement."""

import argparse
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import time

from .sglang_backing import check_sources
from .sglang_observe import install_schedule_snapshot


def engine_settings(model, tp_size, selector):
    return {"model_path": str(model), "tp_size": tp_size, "pp_size": 1, "dp_size": 1,
            "base_gpu_id": 0, "disable_custom_all_reduce": True,
            "context_length": 8192, "max_total_tokens": 16384, "max_running_requests": 2,
            "max_prefill_tokens": 16384, "chunked_prefill_size": 4096, "mem_fraction_static": .4,
            "enable_hisparse": True, "kv_cache_dtype": "fp8_e4m3", "disable_cuda_graph": True,
            "disable_overlap_schedule": True, "disable_radix_cache": True, "schedule_policy": "fcfs",
            "dsa_prefill_backend": "flashinfer_sparse_mla", "dsa_decode_backend": "flashinfer_sparse_mla",
            "dsa_topk_backend": selector, "random_seed": 17, "stream_interval": 1, "log_level": "info",
            "hisparse_config": json.dumps({"top_k": 2048, "device_buffer_size": 4096, "host_to_device_ratio": 2})}


def _worker(*args, **kwargs):
    from . import sglang_dependency
    from sglang.srt.managers.scheduler import run_scheduler_process

    output = Path(os.environ["RSI_TOPOLOGY_SMOKE_DIR"])
    bound = inspect.signature(run_scheduler_process).bind(*args, **kwargs).arguments
    rank, gpu = bound["tp_rank"], bound["gpu_id"]
    with (output / f"worker-rank-{rank}.json").open("x", encoding="utf-8") as stream:
        json.dump({"rank": rank, "gpu_id": gpu, "pid": os.getpid()}, stream)
        stream.write("\n")
    install_schedule_snapshot(output / f"schedule-rank-{rank}.json", rank)
    if os.environ.get("RSI_TOPOLOGY_SMOKE_BACKUP") == "1":
        sglang_dependency.install(output / f"backup-rank-{rank}.jsonl")
    return run_scheduler_process(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=2)
    parser.add_argument("--selector", choices=("torch", "sgl-kernel"), default="torch")
    parser.add_argument("--backup-dependency", action="store_true")
    parser.add_argument("--input-tokens", type=int, default=6144)
    parser.add_argument("--output-tokens", type=int, default=32)
    args = parser.parse_args()
    if not 4096 < args.input_tokens <= 8192 - args.output_tokens or args.output_tokens < 2:
        parser.error("the smoke must exercise long HiSparse decode within 8192 tokens")
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    os.environ["RSI_TOPOLOGY_SMOKE_DIR"] = str(output)
    os.environ["RSI_TOPOLOGY_SMOKE_BACKUP"] = str(int(args.backup_dependency))
    report = {"schema": 1, "status": "started", "mode": "native", "tp_size": args.tp_size,
              "fixed_backup_dependency": args.backup_dependency}
    engine = None
    start = time.perf_counter()
    try:
        import sglang
        import torch
        from sglang.srt.entrypoints.engine import Engine
        from transformers import AutoTokenizer

        check_sources(sglang)
        visible = torch.cuda.device_count()
        if visible < args.tp_size:
            raise RuntimeError("too few visible GPUs for the fixed attention placement")
        report["visible_gpu_devices"] = [{"index": index, "name": torch.cuda.get_device_name(index),
                                          "capability": torch.cuda.get_device_capability(index)} for index in range(visible)]

        class SmokeEngine(Engine):
            run_scheduler_process_func = staticmethod(_worker)

        model = args.model.resolve(strict=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        tokens = tokenizer.encode("The archive records the sequence of requests and the verified results. ",
                                  add_special_tokens=False)
        if not tokens:
            raise ValueError("native smoke produced no input tokens")
        ids = (tokens * ((args.input_tokens + len(tokens) - 1) // len(tokens)))[:args.input_tokens]
        settings = engine_settings(model, args.tp_size, args.selector)
        report["engine_args"] = {key: value for key, value in settings.items() if key != "model_path"}
        report["input_ids_sha256"] = sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        report["input_tokens"] = len(ids)
        engine = SmokeEngine(**settings)
        ready = time.perf_counter()
        response = engine.generate(input_ids=ids, rid="native-smoke",
                                   sampling_params={"temperature": 0, "ignore_eos": True, "max_new_tokens": args.output_tokens})
        output_ids = response["output_ids"]
        if (type(output_ids) is not list or len(output_ids) != args.output_tokens
                or any(type(token) is not int or token < 0 for token in output_ids)):
            raise RuntimeError("native smoke did not generate the required output token sequence")
        ranks = [json.loads((output / f"worker-rank-{rank}.json").read_bytes()) for rank in range(args.tp_size)]
        if len({rank["pid"] for rank in ranks}) != args.tp_size or [rank["gpu_id"] for rank in ranks] != list(range(args.tp_size)):
            raise RuntimeError("native engine workers differ from the fixed CUDA placement")
        report.update(status="completed", worker_ranks=ranks, output_ids=output_ids,
                      init_seconds=ready - start, request_seconds=time.perf_counter() - ready)
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            with (output / "result.json").open("x", encoding="utf-8") as stream:
                json.dump(report, stream, indent=2, allow_nan=False)
                stream.write("\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
