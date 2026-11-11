"""Observe native DSA session branches and physical UnifiedRadixCache reuse."""

import argparse
from functools import wraps
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
import time

from rsi_memory.topology import fingerprint

from .sglang_backing import check_sources
from .sglang_observe import save
from .sglang_smoke import engine_settings
from .sglang_sources import file_sha256


SOURCES = {
    "srt/mem_cache/kv_cache_builder.py": "1512eae057eedb7141ba4d6f4dcd62fb54ac49d5aca4c3e29ef2ea2889dd5107",
    "srt/mem_cache/registry.py": "a7f03456d406529d734ebeded7840b894b58f8eb193c048629f2f0e53b51acba",
    "srt/mem_cache/radix_cache.py": "150818b0a4eaa05171ca2e485698fa455608c13ee6b07f3945ac2095fa294de3",
    "srt/mem_cache/unified_radix_cache.py": "df8b72f0382c452d58497d3020c9f666f03ec1d6e5979ec4ea6efc3dbafc9b41",
    "srt/mem_cache/unified_cache/tree_core_registry.py": "c2b372b406fbead89e5762245486bfff2133cfcd7ba91192ae11e64536a64dbb",
    "srt/mem_cache/unified_cache/unified_tree_core.py": "32fc069026272c620d3fd007dc27eb311685bbe0448f96457a67a84181dabb56",
    "srt/mem_cache/unified_cache/unified_tree_core_interface.py": "6257f16699c76eccb494517315e09955afa58302c91e8772ee643c3b9357edf7",
    "srt/mem_cache/unified_cache/component_type.py": "ea86556b819659c0ec14f6d774d20ec01f4577b546bd669c20f4e7a602580cac",
    "srt/mem_cache/unified_cache/components/__init__.py": "885dac63e0cdb679f5b0a1abf116a45a93798361514e8eb95a5c657dd644c6a0",
    "srt/mem_cache/unified_cache/components/base.py": "4dfdde9d4b6265ca817106fc3ebe8542b218ec367c6b848b47d231f66b1ac5ac",
    "srt/mem_cache/unified_cache/components/full.py": "6380586e6d84ef37c2e2244c807a1c7b5a2e84cb78d0a473f3d34cadd6c673bf",
    "srt/session/session_controller.py": "5e92045afba811a1b3520fa121b83d85f9d64cb03a12f3aa42e47e88e2211be5",
    "srt/session/streaming_session.py": "3de6ccd81c151af0ad5c920b7aac460d95f3fba0b1873bc5a2b30cfb19c046aa",
    "srt/arg_groups/hisparse_hook.py": "2b44dc17c0a76863880b0b68fd9a20a5653fc4f598f4d817b27216cca6eb88eb",
    "srt/managers/io_struct.py": "cc4e6a88ddfeaa78fc4518990013859168d5bc141905cd66252f9d020aa86cd5",
}


def _validate(fixture):
    if (type(fixture) is not dict or fixture.keys() != {"schema", "name", "prefix", "branches", "max_new_tokens"}
            or type(fixture["schema"]) is not int or fixture["schema"] != 1
            or type(fixture["name"]) is not str or not fixture["name"]
            or type(fixture["branches"]) is not list or len(fixture["branches"]) != 2
            or type(fixture["max_new_tokens"]) is not int or not 2 <= fixture["max_new_tokens"] <= 128):
        raise ValueError("expected one fixed root and two native session branches")
    for index, row in enumerate([fixture["prefix"], *fixture["branches"]]):
        branch = index > 0
        if type(row) is not dict:
            raise ValueError("prefix and branches must be request records")
        expected = {"rid", "input_ids" if "input_ids" in row else "prompt"}
        if branch:
            expected.add("drop_parent_output")
        if row.keys() != expected or type(row["rid"]) is not str or not row["rid"] or len(row["rid"]) > 96:
            raise ValueError("each request requires an ID and exactly one fixed input")
        if branch and type(row["drop_parent_output"]) is not bool:
            raise ValueError("branch must explicitly choose whether to append the parent's output")
        if "input_ids" in row and (type(row["input_ids"]) is not list or not row["input_ids"]
                                   or any(type(token) is not int or token < 0 for token in row["input_ids"])):
            raise ValueError("invalid native session token IDs")
        if "prompt" in row and (type(row["prompt"]) is not str or not row["prompt"].strip()):
            raise ValueError("invalid native session prompt")
    ids = [row["rid"] for row in [fixture["prefix"], *fixture["branches"]]]
    if len(set(ids)) != len(ids) or any(rid.startswith("reference-") for rid in ids):
        raise ValueError("session fixture request IDs must be unique and avoid reserved prefixes")
    return fixture


def _read_radix_mapping(core, key, full):
    """Read the matched FULL device prefix without touching recency or tree structure."""
    matched, node_id, pinned = core.match_full_device_prefix(key)
    node = core.node_by_id(node_id)
    segments = []
    while node is not core.root_node:
        value = core.get_component_device_value(node.id, full)
        if value is None or len(value) != len(node.key):
            raise RuntimeError("matched UnifiedRadixCache node is missing its FULL device rows")
        segments.append(value.tolist())
        node = node.parent
    indices = [index for segment in reversed(segments) for index in segment]
    if len(indices) != pinned or not 0 <= matched <= pinned or matched % core.page_size:
        raise RuntimeError("UnifiedRadixCache FULL device match has inconsistent aligned geometry")
    return indices[:matched]


def _worker(*args, **kwargs):
    from sglang.srt.managers.scheduler import Scheduler, run_scheduler_process
    from sglang.srt.mem_cache.radix_cache import RadixKey
    from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
    from sglang.srt.mem_cache.unified_cache.components.full import FullComponent
    from sglang.srt.mem_cache.unified_cache.unified_tree_core import UnifiedTreeCore
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

    directory = Path(os.environ["RSI_NATIVE_SESSION_OUTPUT"])
    config = json.loads((directory / "worker_input.json").read_bytes())
    bindings = inspect.signature(run_scheduler_process).bind(*args, **kwargs).arguments
    rank = bindings["tp_rank"]
    expected = config["request_ids"]
    branch_ids = expected[1:]
    launched, finished = {}, {}
    original_run, original_result = Scheduler.run_batch, Scheduler.process_batch_result
    reported = False

    @wraps(original_run)
    def run(self, batch, *rest, **kw):
        if batch.forward_mode.is_extend():
            for request in batch.reqs:
                if request.rid in expected:
                    snapshot = launched.setdefault(request.rid, [])
                    # Store every extend segment so chunked prefill cannot look like a cross-request cache hit.
                    slots = self.req_to_token_pool.req_to_token[request.kv.req_pool_idx, :request.extend_range.end].tolist()
                    snapshot.append({"mode": batch.forward_mode.name, "matched_prefix_rows": len(request.prefix_indices),
                                     "input_ids_sha256": fingerprint(list(request.origin_input_ids)), "physical_slots": slots})
        return original_run(self, batch, *rest, **kw)

    @wraps(original_result)
    def result(self, batch, *rest, **kw):
        nonlocal reported
        requests = tuple(batch.reqs)
        response = original_result(self, batch, *rest, **kw)
        for request in requests:
            if request.rid in expected and request.finished():
                finished[request.rid] = request
        if not reported and all(rid in finished for rid in expected):
            cache = self.tree_cache
            full = ComponentType.FULL
            if (type(cache) is not UnifiedRadixCache or cache.disable or self.enable_hisparse
                    or type(cache.tree_core) is not UnifiedTreeCore or cache.tree_core.is_eagle
                    or cache.tree_core.enable_hicache or cache.linker is not None or cache.cache_controller is not None
                    or cache.enable_session_radix_cache or cache.tree_components != (full,)
                    or type(cache.components[full]) is not FullComponent):
                raise RuntimeError("expected stock UnifiedRadixCache with the Python FULL-device tree on DSA")
            mappings = {}
            for rid in expected:
                request = finished[rid]
                key = RadixKey(request.origin_input_ids, request.extra_key, cache_salt=request.cache_salt)
                mappings[rid] = _read_radix_mapping(cache.tree_core, key, full)
            rank_path = directory / f"rank-{rank}.json"
            pending = rank_path.with_suffix(".pending")
            save(pending, {"rank": rank, "gpu_ordinal": bindings["gpu_id"], "pid": os.getpid(),
                 "cache_class": type(cache).__name__, "tree_core_class": type(cache.tree_core).__name__, "component": full.name,
                 "page_size": cache.page_size, "launched": launched, "retained_mappings_after_both_branches": mappings,
                 "branch_request_ids": branch_ids})
            os.replace(pending, rank_path)
            reported = True
        return response

    Scheduler.run_batch, Scheduler.process_batch_result = run, result
    return run_scheduler_process(*args, **kwargs)


def _proof(rank, root, branches, *, root_length, common_length, input_hashes):
    page_size = rank["page_size"]
    root_aligned = root_length // page_size * page_size
    shared_aligned = common_length // page_size * page_size
    if not root_aligned or shared_aligned < root_aligned:
        raise ValueError("session root did not cover a reusable native page")
    maps = rank["retained_mappings_after_both_branches"]
    root_slots = maps[root][:root_aligned]
    if len(root_slots) != root_aligned or len(set(root_slots)) != root_aligned or min(root_slots) <= 0:
        raise ValueError("native radix did not retain an unaliased full-page root prefix")
    shared = maps[branches[0]][:shared_aligned]
    if (len(shared) != shared_aligned or len(set(shared)) != shared_aligned or min(shared) <= 0
            or maps[branches[1]][:shared_aligned] != shared):
        raise ValueError("branches do not share their full-page common mapping")
    tails = []
    for branch in branches:
        launches = rank["launched"][branch]
        first = launches[0]
        mapping = maps[branch]
        if (first["matched_prefix_rows"] < root_aligned or first["physical_slots"][:root_aligned] != root_slots
                or first["input_ids_sha256"] != input_hashes[branch] or mapping[:root_aligned] != root_slots):
            raise ValueError("native session branch did not consume the retained physical root prefix")
        tail = mapping[shared_aligned:]
        if len(tail) < page_size or len(set(tail)) != len(tail) or set(tail).intersection(shared) or min(tail) <= 0:
            raise ValueError("branch suffix is missing or aliases its immutable parent pages")
        tails.append(set(tail))
    if tails[0].intersection(tails[1]):
        raise ValueError("diverging native branches share physical suffix slots")
    return {"shared_root_prefix_rows": root_aligned, "shared_prefix_rows": shared_aligned,
            "shared_prefix_sha256": fingerprint(shared),
            "retained_suffix_rows": [len(tail) for tail in tails], "disjoint_suffixes": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("model", "fixture", "output"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--model-weights-sha256", required=True)
    parser.add_argument("--tp-size", type=int, choices=(1, 2), default=1)
    args = parser.parse_args()
    fixture = _validate(json.loads(args.fixture.read_bytes()))
    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    os.environ["RSI_NATIVE_SESSION_OUTPUT"] = str(directory)
    root, branches = fixture["prefix"], fixture["branches"]
    root_id, branch_ids = root["rid"], [branch["rid"] for branch in branches]
    save(directory / "worker_input.json", {"request_ids": [root_id, *branch_ids]})
    engine = None
    report = {"schema": 1, "status": "started", "fixture_sha256": fingerprint(fixture), "tp_size": args.tp_size,
              "scope": "native DSA UnifiedRadixCache page-aligned shared prefix and separate suffixes; separate from HiSparse relay"}
    try:
        import sglang
        import torch
        from sglang.srt.entrypoints.engine import Engine
        from transformers import AutoTokenizer

        wheel = check_sources(sglang)
        for name, expected in SOURCES.items():
            if sha256((wheel / name).read_bytes()).hexdigest() != expected:
                raise ValueError(f"native session source differs for {name}")
        if torch.cuda.device_count() < args.tp_size:
            raise RuntimeError("not enough visible GPUs for the bound session placement")
        report["devices"] = [str(torch.cuda.get_device_properties(index).uuid) for index in range(args.tp_size)]
        report["source_sha256"] = SOURCES
        model = args.model.resolve(strict=True)
        model_digests = {"config": file_sha256(model / "config.json"), "weights": file_sha256(model / "model.safetensors")}
        if model_digests != {"config": args.model_config_sha256, "weights": args.model_weights_sha256}:
            raise ValueError("native session probe model files differ from the bound checkpoint")
        report["model_sha256"] = model_digests
        tokenizer = AutoTokenizer.from_pretrained(str(model), local_files_only=True)

        def tokens(row):
            return row["input_ids"] if "input_ids" in row else tokenizer.encode(row["prompt"], add_special_tokens=False)

        root_tokens, suffixes = tokens(root), [tokens(branch) for branch in branches]
        report["fixture_token_lengths"] = {root_id: len(root_tokens), **dict(zip(branch_ids, map(len, suffixes), strict=True))}
        if (len(root_tokens) < 64 or any(len(suffix) < 128 or len(root_tokens) + len(suffix) + 2 * fixture["max_new_tokens"] > 8192
                                        for suffix in suffixes) or suffixes[0][0] == suffixes[1][0]
                or any(suffix[0] == tokenizer.bos_token_id for suffix in suffixes)):
            raise ValueError("need bounded non-overlapping branch suffixes and a reusable root prefix")
        if len({branch["drop_parent_output"] for branch in branches}) != 1:
            raise ValueError("both branch replays must start from the same parent state")

        class RadixProbeEngine(Engine):
            run_scheduler_process_func = staticmethod(_worker)

        settings = engine_settings(model, args.tp_size, "torch")
        settings.pop("hisparse_config")
        settings.update(enable_hisparse=False, disable_radix_cache=False, enable_session_radix_cache=False)
        report["engine_args"] = {key: value for key, value in settings.items() if key != "model_path"}
        engine = RadixProbeEngine(**settings)
        params = {"temperature": 0, "ignore_eos": True, "max_new_tokens": fixture["max_new_tokens"]}
        session_id = engine.open_session(capacity_of_str_len=128 * 1024, session_id="radix-proof", streaming=False)
        base = engine.generate(input_ids=root_tokens, rid=root_id, sampling_params=params, session_params={"id": session_id})
        report["root_output_ids"] = base["output_ids"]
        if len(base["output_ids"]) != fixture["max_new_tokens"]:
            raise RuntimeError("native session root output did not reach its fixed length")
        report["branches"] = outputs = {}
        full_inputs = {}
        for branch, suffix in zip(branches, suffixes, strict=True):
            rid, drop = branch["rid"], branch["drop_parent_output"]
            full_inputs[rid] = [*root_tokens, *([] if drop else base["output_ids"]), *suffix]
            response = engine.generate(input_ids=suffix, rid=rid, sampling_params=params,
                                       session_params={"id": session_id, "rid": root_id, "drop_previous_output": drop})
            outputs[rid] = {"output_ids": response["output_ids"], "text": response.get("text"),
                            "cached_tokens": response["meta_info"].get("cached_tokens")}
            if len(response["output_ids"]) != fixture["max_new_tokens"]:
                raise RuntimeError("native session branch output did not reach its fixed length")
        left, right = (full_inputs[rid] for rid in branch_ids)
        common = next((i for i, (a, b) in enumerate(zip(left, right)) if a != b), min(len(left), len(right)))
        limit = time.monotonic() + 15
        rank_paths = [directory / f"rank-{index}.json" for index in range(args.tp_size)]
        while time.monotonic() < limit and not all(path.is_file() for path in rank_paths):
            time.sleep(.02)
        if not all(path.is_file() for path in rank_paths):
            raise TimeoutError("the scheduler did not publish complete radix rank receipts")
        report["rank_proofs"] = [_proof(json.loads(path.read_bytes()), root_id, branch_ids, root_length=len(root_tokens), common_length=common,
                                       input_hashes={rid: fingerprint(ids) for rid, ids in full_inputs.items()}) for path in rank_paths]
        engine.close_session(session_id)
        for index, rid in enumerate(branch_ids):
            reference = engine.generate(input_ids=full_inputs[rid], rid=f"reference-{index}", sampling_params=params,
                                        cache_salt=f"uncached-reference-{index}")
            outputs[rid]["uncached_output_ids"] = reference["output_ids"]
            outputs[rid]["uncached_cached_tokens"] = reference["meta_info"].get("cached_tokens")
            if outputs[rid]["uncached_cached_tokens"] != 0:
                raise RuntimeError("explicit full-context reference unexpectedly reused cached KV")
            if reference["output_ids"] != outputs[rid]["output_ids"]:
                raise RuntimeError("native session reuse changed the model's deterministic output")
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
