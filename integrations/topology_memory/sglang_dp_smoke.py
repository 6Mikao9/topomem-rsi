"""Qualify stock GLM HiSparse with backbone TP2 and independent attention DP2."""

import argparse
from functools import wraps
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import time
from uuid import UUID

from .sglang_dependency import install as install_backup
from .sglang_observe import append, install_schedule_snapshot, save
from .sglang_smoke import engine_settings
from .sglang_sources import check_sources


def _uuid(value):
    return f"GPU-{UUID(str(value).removeprefix('GPU-'))}"


def _worker(*args, **kwargs):
    from sglang.srt.layers.attention.dsa_backend import DeepseekSparseAttnBackend
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process

    directory = Path(os.environ["RSI_ATTENTION_DP_OUTPUT"])
    bound = inspect.signature(run_scheduler_process).bind(*args, **kwargs).arguments
    rank = bound["tp_rank"]
    install_backup(directory / f"backup-rank-{rank}.jsonl")
    install_schedule_snapshot(directory / f"schedule-rank-{rank}.json", rank)
    original_init, original_decode = Scheduler.__init__, DeepseekSparseAttnBackend.forward_decode
    original_run = Scheduler.run_batch
    seen_decode = False
    requests = set()

    @wraps(original_init)
    def init(self, *rest, **kw):
        original_init(self, *rest, **kw)
        ps = self.ps
        save(directory / f"worker-rank-{rank}.json", {"rank": rank, "gpu_ordinal": bound["gpu_id"], "pid": os.getpid(),
             "backbone_tp_size": ps.tp_size, "attention_tp_size": ps.attn_tp_size, "attention_dp_size": ps.attn_dp_size,
             "attention_dp_rank": ps.attn_dp_rank, "coordinator_tp_size": self.hisparse_coordinator.tp_world_size})

    @wraps(original_decode)
    def decode(self, q, *rest, **kw):
        nonlocal seen_decode
        result = original_decode(self, q, *rest, **kw)
        if not seen_decode and q.numel():
            save(directory / f"decode-rank-{rank}.json", {"rank": rank, "pid": os.getpid(),
                 "local_query_heads": self.num_q_heads, "decode_backend": self.dsa_decode_impl,
                 "device": str(q.device), "returned_shape": list(result.shape)})
            seen_decode = True
        return result

    @wraps(original_run)
    def run(self, batch, *rest, **kw):
        result = original_run(self, batch, *rest, **kw)
        if batch.forward_mode.is_decode():
            for request in batch.reqs:
                if request.rid.startswith("attention-dp-") and request.rid not in requests:
                    requests.add(request.rid)
                    append(directory / f"dispatch-rank-{rank}.jsonl", {"rid": request.rid, "rank": self.ps.attn_dp_rank})
        return result

    Scheduler.__init__, Scheduler.run_batch, DeepseekSparseAttnBackend.forward_decode = init, run, decode
    return run_scheduler_process(*args, **kwargs)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-uuid", action="append", required=True)
    parser.add_argument("--input-tokens", type=int, default=6144)
    parser.add_argument("--output-tokens", type=int, default=32)
    args = parser.parse_args()
    if (len(args.expected_uuid) != 2 or len(set(map(_uuid, args.expected_uuid))) != 2
            or args.output_tokens < 3 or not 4096 < args.input_tokens <= 8192 - args.output_tokens):
        parser.error("expected two distinct GPU identities and a long-context request with multiple decode steps")
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["RSI_ATTENTION_DP_OUTPUT"] = str(directory)
    report = {"schema": 1, "status": "started", "scope": "independent attention DP2 on a TP2 backbone, one routed request per attention rank"}
    engine = None
    try:
        import sglang
        import torch
        from sglang.srt.entrypoints.engine import Engine
        from transformers import AutoTokenizer

        check_sources(sglang)
        if torch.cuda.device_count() != 2:
            raise RuntimeError("the bound attention-DP fixture requires exactly two visible GPUs")
        observed = [_uuid(torch.cuda.get_device_properties(index).uuid) for index in range(2)]
        if observed != list(map(_uuid, args.expected_uuid)):
            raise RuntimeError("the visible CUDA order differs from the bound devices")
        report["device_uuids"] = observed
        override = os.environ.get("SGLANG_NCCL_SO_PATH")
        report["nccl_override_sha256"] = sha256(Path(override).read_bytes()).hexdigest() if override else None

        class AttentionDpEngine(Engine):
            run_scheduler_process_func = staticmethod(_worker)

        model = args.model.resolve(strict=True)
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)
        seed = tokenizer.encode("A fixed record of the verified attention routes and request results. ", add_special_tokens=False)
        if not seed:
            raise ValueError("fixed smoke text produced no tokens")
        ids = (seed * ((args.input_tokens + len(seed) - 1) // len(seed)))[:args.input_tokens]
        settings = engine_settings(model, 2, "torch")
        settings.update(dp_size=2, enable_dp_attention=True)
        report["engine_args"] = {key: value for key, value in settings.items() if key != "model_path"}
        report["input_ids_sha256"] = sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        engine = AttentionDpEngine(**settings)
        report["requests"] = outputs = []
        for routed_rank in range(2):
            start = time.monotonic()
            output = engine.generate(input_ids=ids, rid=f"attention-dp-{routed_rank}", routed_dp_rank=routed_rank,
                                     sampling_params={"temperature": 0, "ignore_eos": True, "max_new_tokens": args.output_tokens})
            tokens = output["output_ids"]
            if type(tokens) is not list or len(tokens) != args.output_tokens or any(type(token) is not int or token < 0 for token in tokens):
                raise RuntimeError("the routed stock attention request did not finish at its fixed length")
            outputs.append({"routed_attention_dp_rank": routed_rank, "output_ids": tokens,
                            "seconds": time.monotonic() - start})
        rank_records, decodes = [], []
        for rank in range(2):
            worker = json.loads((directory / f"worker-rank-{rank}.json").read_bytes())
            decode = json.loads((directory / f"decode-rank-{rank}.json").read_bytes())
            dispatch = [json.loads(line) for line in (directory / f"dispatch-rank-{rank}.jsonl").read_bytes().splitlines()]
            if (worker["attention_tp_size"], worker["attention_dp_size"], worker["coordinator_tp_size"],
                    decode["local_query_heads"], decode["decode_backend"], decode["pid"]) != (
                        1, 2, 1, 8, "flashinfer_sparse_mla", worker["pid"]):
                raise RuntimeError("actual rank attention differs from the fixed full-head DP scenario")
            local_rank = worker["attention_dp_rank"]
            if dispatch != [{"rid": f"attention-dp-{local_rank}", "rank": local_rank}]:
                raise RuntimeError("the actual native scheduler dispatch differs from external rank routing")
            worker["decode_dispatch"] = dispatch
            rank_records.append(worker)
            decodes.append(decode)
        if sorted(row["attention_dp_rank"] for row in rank_records) != [0, 1]:
            raise RuntimeError("native attention DP did not execute on both independent ranks")
        report.update(rank_placement=rank_records, actual_decode=decodes,
                      exact_output_match=outputs[0]["output_ids"] == outputs[1]["output_ids"])
        if not report["exact_output_match"]:
            raise RuntimeError("identical fixed inputs produced different outputs across attention DP ranks")
        report["status"] = "verified"
    except Exception as exc:
        report.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            save(directory / "result.json", report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
