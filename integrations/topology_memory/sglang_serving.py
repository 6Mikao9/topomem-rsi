"""Replay fixed requests on native, hand-written, or generated HiSparse backing."""

import argparse
import asyncio
from dataclasses import asdict
from hashlib import sha256
import inspect
import json
import os
from pathlib import Path
from statistics import median
import time
from uuid import UUID

from rsi_memory.memory_program import MemoryProgram, MemoryScenario, plan_backing
from rsi_memory.topology import HardwareTopology, fingerprint

from . import sglang_backing as backing, sglang_observe as observe
from . import sglang_resident as resident
from . import sglang_dependency, sglang_global_ticks as global_ticks, sglang_ingress_gate as ingress
from .sglang_smoke import engine_settings


SELECTORS = {"torch-unfused": "torch", "sgl-kernel": "sgl-kernel"}
_TARGET = {"schema", "gpu_bindings", "model_config_sha256", "model_weights_sha256", "pool", "native_host", "allowed_nodes"}
_PROGRAM = {"schema": 1, "kind": "kv_memory_program", "rules": [
    {"event": "allocate", "action": "place", "when": {"op": "const", "value": 1}, "score": {"op": "const", "value": 0}}]}


def _load(path):
    return json.loads(Path(path).read_bytes())


def _hash_file(path):
    digest = sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024**2), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _uuid(value):
    return f"GPU-{UUID(str(value).removeprefix('GPU-'))}"


def validate_fixture(fixture):
    """Validate an immutable replay of independent model requests."""
    if (type(fixture) is not dict or fixture.keys() != {"schema", "name", "warmup", "requests"}
            or type(fixture["schema"]) is not int or fixture["schema"] != 1
            or type(fixture["name"]) is not str or not fixture["name"]):
        raise ValueError("expected a schema-1 independent request replay")
    rids = set()
    for name in ("warmup", "requests"):
        group = fixture[name]
        if type(group) is not list or not 0 < len(group) <= 128:
            raise ValueError(f"fixture requires a bounded nonempty {name} list")
        previous = -1
        for row in group:
            required = {"rid", "arrival_ms", "max_new_tokens"}
            allowed = required | {"input_ids", "prompt", "session", "branch", "routed_dp_rank"}
            if (type(row) is not dict or not required <= row.keys() or row.keys() - allowed
                    or ("input_ids" in row) == ("prompt" in row)):
                raise ValueError("each fixed request needs exactly one prompt or input_ids")
            rid = row["rid"]
            if type(rid) is not str or not rid or len(rid) > 128 or rid in rids:
                raise ValueError("fixture request IDs must be unique and bounded")
            rids.add(rid)
            arrival, budget = row["arrival_ms"], row["max_new_tokens"]
            if (type(arrival) is not int or not 0 <= arrival <= 600_000 or arrival < previous
                    or type(budget) is not int or not 2 <= budget <= 1024):
                raise ValueError("fixture arrival order or decode budget is outside its envelope")
            previous = arrival
            for field in row.keys() & {"session", "branch"}:
                if type(row[field]) is not str or not row[field] or len(row[field]) > 128:
                    raise ValueError("session and branch are bounded passive labels")
            if "routed_dp_rank" in row and (type(row["routed_dp_rank"]) is not int or row["routed_dp_rank"] not in (0, 1)):
                raise ValueError("external attention DP rank must be 0 or 1")
            if "input_ids" in row and (type(row["input_ids"]) is not list or not 0 < len(row["input_ids"]) <= 8192
                                       or any(type(token) is not int or token < 0 for token in row["input_ids"])):
                raise ValueError("invalid fixed input token IDs")
            if "prompt" in row and (type(row["prompt"]) is not str or not 0 < len(row["prompt"].encode()) <= 256 * 1024):
                raise ValueError("invalid fixed request text")
    return fixture


def schedule_inputs(fixture, tp_size, selector, *, attention_dp_size=1, hot_slots=4096):
    if type(hot_slots) is not int or hot_slots not in (4096, 6144, 8192):
        raise ValueError("unsupported native HiSparse hot capacity")
    if type(attention_dp_size) is not int or attention_dp_size not in (1, 2) or attention_dp_size == 2 and tp_size != 2:
        raise ValueError("attention DP2 requires two fixed backbone TP workers")
    settings = engine_settings("", tp_size, SELECTORS[selector])
    settings.pop("model_path")
    if hot_slots != 4096:
        hisparse = json.loads(settings["hisparse_config"])
        hisparse["device_buffer_size"] = hot_slots
        settings["hisparse_config"] = json.dumps(hisparse)
    request_keys = ("rid", "arrival_ms", "max_new_tokens")
    if attention_dp_size == 2:
        settings.update(dp_size=2, enable_dp_attention=True)
        request_keys += ("routed_dp_rank",)
        for group in ("warmup", "requests"):
            if {row.get("routed_dp_rank") for row in fixture[group]} != {0, 1}:
                raise ValueError("attention DP2 requires explicit warmup and measured requests on both ranks")
    elif any("routed_dp_rank" in row for group in ("warmup", "requests") for row in fixture[group]):
        raise ValueError("request rank routing is available only in the fixed attention DP2 mode")
    return {"engine": settings, "backup_dependency": True,
            "warmup": [{key: row[key] for key in request_keys} for row in fixture["warmup"]],
            "requests": [{key: row[key] for key in request_keys} for row in fixture["requests"]]}


def _local_tokenizer(model):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(model), local_files_only=True)


def _tokenize(fixture, tokenizer):
    tokenized = {}
    for phase in ingress.PHASES:
        for spec in fixture[phase]:
            ids = spec["input_ids"] if "input_ids" in spec else tokenizer.encode(spec["prompt"], add_special_tokens=False)
            if (type(ids) is not list or not ids or len(ids) + spec["max_new_tokens"] > 8192
                    or any(type(token) is not int or token < 0 for token in ids)):
                raise ValueError("tokenized request differs from the frozen 8192-token context contract")
            tokenized[spec["rid"]] = ids
    return tokenized


def _token_commitments(tokenized):
    return {rid: {"input_tokens": len(ids), "input_ids_sha256": fingerprint(ids)} for rid, ids in tokenized.items()}


def _bound_external(external, committed, tokenized):
    gate = committed[ingress.EXTERNAL_FIELD]
    ingress.checked_contract(gate)
    external = ingress.bind_schedule(external, _token_commitments(tokenized), gate_timeout_ms=gate["gate_timeout_ms"],
                                     collective_timeout_ms=gate["collective_timeout_ms"])
    if external != committed:
        raise ValueError("live tokenizer or ingress schedule differs from the frozen commitment")
    return external


def _compile_plan(target, topology, scenario, program, *, native):
    pool = target["pool"]
    attention_dp_size = target.get("attention_dp_size", 1)
    ordinals = {node: binding["ordinal"] for node, binding in target["gpu_bindings"].items()}
    slabs = [{"key": f"r{rank}-l{layer}", "rank": rank, "layer": layer, "bytes": pool["rows"] * scenario.kv_row_bytes,
              "allowed_nodes": [target["native_host"]] if native else target["allowed_nodes"][node]}
             for rank, node in scenario.rank_placement for layer in range(pool["layers"])]
    compiled = plan_backing(topology, scenario, program, slabs, transports=backing.NATIVE_TRANSPORTS)
    plans = backing.bind_plans(topology, scenario, compiled, gpu_ordinals=ordinals,
                               attention_tp_size=1 if attention_dp_size == 2 else None, **pool)
    return compiled, plans


def _bind(args, *, load_tokenizer=_local_tokenizer):
    target = _load(args.target)
    fixture = validate_fixture(_load(args.fixture))
    topology = HardwareTopology.from_dict(_load(args.topology))
    scenario = MemoryScenario.from_dict(_load(args.scenario))
    if (type(target) is not dict or not _TARGET <= target.keys() or target.keys() - _TARGET - {"attention_dp_size"}
            or type(target["schema"]) is not int or target["schema"] != 1):
        raise ValueError("unknown topology serving target schema")
    tp_size = len(scenario.rank_placement)
    if scenario.selector not in SELECTORS or tp_size not in (1, 2):
        raise ValueError("runtime supports one or two fixed DSA attention ranks")
    attention_dp_size = target.get("attention_dp_size", 1)
    pool = target["pool"]
    hot_slots = pool.get("hot_slots") if type(pool) is dict else None
    external = schedule_inputs(fixture, tp_size, scenario.selector, attention_dp_size=attention_dp_size, hot_slots=hot_slots)
    if (scenario.target_sha256, scenario.workload_sha256) != (fingerprint(target), fingerprint(fixture)):
        raise ValueError("scenario does not bind the actual target and replay requests")
    if (attention_dp_size == 2) != (args.external_schedule is not None):
        raise ValueError("fixed attention DP2 requires a committed gated external schedule")
    model = args.model.resolve(strict=True)
    for name, key in (("config.json", "model_config_sha256"), ("model.safetensors", "model_weights_sha256")):
        if _hash_file(model / name) != target[key]:
            raise ValueError("model files differ from the bound serving target")
    if sorted(path.name for path in model.glob("*.safetensors")) != ["model.safetensors"]:
        raise ValueError("topology runner supports the bound unsharded checkpoint")
    tokenized = _tokenize(fixture, load_tokenizer(model))
    if attention_dp_size == 2:
        external = _bound_external(external, _load(args.external_schedule), tokenized)
    if scenario.schedule_sha256 != fingerprint(external):
        raise ValueError("scenario does not bind the actual gated external schedule")
    if (type(target["pool"]) is not dict or target["pool"].keys() != {"rows", "layers", "page_size", "hot_slots"}
            or any(type(value) is not int or value <= 0 for value in target["pool"].values())):
        raise ValueError("invalid fixed cold backing pool geometry")
    gpu_nodes = {node.id for node in topology.nodes if node.kind == "gpu"}
    bindings = target["gpu_bindings"]
    if type(bindings) is not dict or bindings.keys() != gpu_nodes or len(bindings) < 2:
        raise ValueError("target needs an explicit physical binding for every GPU node")
    ordinals = {}
    for node, binding in bindings.items():
        if type(binding) is not dict or binding.keys() != {"ordinal", "uuid", "name", "capability"}:
            raise ValueError("malformed GPU binding")
        _uuid(binding["uuid"])
        ordinals[node] = binding["ordinal"]
    if (any(type(value) is not int for value in ordinals.values())
            or sorted(ordinals.values()) != list(range(len(ordinals)))
            or [ordinals[node] for _, node in scenario.rank_placement] != list(range(tp_size))):
        raise ValueError("set CUDA visibility so the fixed compute ranks are first in ordinal order")
    if topology.node(target["native_host"]).kind != "host":
        raise ValueError("native control must back cold KV in pinned host memory")
    allowed = target["allowed_nodes"]
    if type(allowed) is not dict or allowed.keys() != {node for _, node in scenario.rank_placement}:
        raise ValueError("allowed backing nodes must be supplied for each fixed compute node")
    shadow_native = args.arm == "native" or getattr(args, "resident_proof", None) is not None
    program = MemoryProgram.from_dict(_PROGRAM if shadow_native else _load(args.program))
    compiled, plans = _compile_plan(target, topology, scenario, program, native=shadow_native)
    evidence = {_hash_file(path) for path in args.link_evidence}
    used_links = {edge for action in compiled["actions"] for key in ("read_links", "write_links") for edge in action[key]}
    if any(link.evidence_sha256 not in evidence for link in topology.links if link.id in used_links):
        raise ValueError("a deployed route is missing the bound link measurement file")
    if args.require_remote and not any(plan.peer_hbm_bytes for plan in plans):
        raise ValueError("this trial requires actual placement on a separate GPU")
    return target, fixture, scenario, external, model, compiled, plans, tokenized


def _worker(*args, **kwargs):
    from sglang.srt.managers.scheduler import run_scheduler_process

    directory = Path(os.environ["RSI_TOPOLOGY_SERVING_DIR"])
    job = _load(directory / "worker_input.json")
    bound = inspect.signature(run_scheduler_process).bind(*args, **kwargs).arguments
    rank, gpu_id = bound["tp_rank"], bound["gpu_id"]
    info = job["plans"][rank]
    plan = backing.RankBacking(**{**info, "layer_tiers": tuple(info["layer_tiers"])})
    if plan.rank != rank or plan.compute_device != gpu_id:
        raise RuntimeError("worker launch changed the fixed rank placement")
    sglang_dependency.install(directory / f"backup-rank-{rank}.jsonl")
    diagnostic = job["phase"] == "parity"
    report = lambda event: observe.append(directory / f"backing-rank-{rank}.jsonl", event)
    path = directory / f"kv-rank-{rank}.jsonl"
    binding = job.get("resident_binding")
    observation = {}
    if binding is None:
        backing.install(plan, report, count_relay_misses=diagnostic)
        usage = observe.install_packed_kv_probe(path, plan) if diagnostic else None
    else:
        external = job["external_schedule"]
        proof = resident.checked_binding(external["engine"], external, job["token_commitments"], binding)
        resident.install(proof, engine=external["engine"], rank=rank, report=report)
        usage = {resident.RESIDENT_TIER: 0} if diagnostic else None
        observation["layer_tiers"] = (resident.RESIDENT_TIER,) * resident.LAYERS
        if diagnostic:
            observation["resident_oracle"] = observe.install_resident_packed_kv_probe(path, rank)
            observation["expected_decode_steps"] = {rid: bound.max_new_tokens - 1 for rid, bound in proof.for_rank(rank).items()}
    gate = job.get(ingress.EXTERNAL_FIELD)
    if gate is not None:
        ingress.install(gate, rank, directory)
    observe.install_scheduler_observer(directory, plan, job["rids_by_rank"][rank], usage,
                                       record_global_ticks=gate is not None, **observation)
    return run_scheduler_process(*args, **kwargs)


async def _one(engine, spec, ids, start_ns):
    await asyncio.sleep(max(0, (start_ns + spec["arrival_ms"] * 10**6 - time.monotonic_ns()) / 10**9))
    sent = time.monotonic_ns()
    previous, chunks, text, finish = [], [], "", None
    settings = {"temperature": 0, "ignore_eos": True, "max_new_tokens": spec["max_new_tokens"]}
    routing = {"routed_dp_rank": spec["routed_dp_rank"]} if "routed_dp_rank" in spec else {}
    stream = await engine.async_generate(input_ids=ids, rid=spec["rid"], sampling_params=settings, stream=True, **routing)
    async for item in stream:
        now = time.monotonic_ns()
        output = item["output_ids"]
        meta = item["meta_info"]
        if (type(output) is not list or meta.get("id") != spec["rid"] or output[:len(previous)] != previous
                or meta.get("completion_tokens") != len(output)
                or len(output) > spec["max_new_tokens"] or any(type(token) is not int or token < 0 for token in output)):
            raise RuntimeError("engine changed the cumulative deterministic token stream")
        if len(output) > len(previous):
            chunks.append({"at_ns": now, "first_index": len(previous), "count": len(output) - len(previous)})
        previous = output
        text, finish = item.get("text", ""), meta.get("finish_reason")
    end = time.monotonic_ns()
    if len(previous) != spec["max_new_tokens"] or type(finish) is not dict or finish.get("type") != "length":
        raise RuntimeError("engine did not complete the bound request")
    first, last = (chunks[index]["at_ns"] for index in (0, -1))
    return {**{key: spec[key] for key in ("rid", "arrival_ms", "session", "branch", "routed_dp_rank") if key in spec},
            "input_ids_sha256": fingerprint(ids), "input_tokens": len(ids), "output_ids": previous,
            "output_text": text, "submitted_ns": sent, "completed_ns": end, "chunks": chunks,
            "ttft_ms": (first - sent) / 10**6,
            "tpot_ms": (last - first) / ((len(previous) - 1) * 10**6) if chunks[0]["count"] == 1 else None}


async def _replay(engine, requests, tokenized, deadline):
    start = time.monotonic_ns()
    return await asyncio.wait_for(asyncio.gather(*(_one(engine, spec, tokenized[spec["rid"]], start) for spec in requests)), deadline)


def _metrics(rows):
    def summarize(key):
        values = sorted(row[key] for row in rows if row[key] is not None)
        return {"samples": len(values), "median": median(values) if values else None,
                "p95": values[(95 * len(values) + 99) // 100 - 1] if values else None}
    wall_seconds = (max(row["completed_ns"] for row in rows) - min(row["submitted_ns"] for row in rows)) / 10**9
    if wall_seconds <= 0:
        raise ValueError("wall clock did not resolve the fixed request replay")
    return {"ttft_ms": summarize("ttft_ms"), "tpot_ms": summarize("tpot_ms"),
            "window_seconds": wall_seconds, "output_tokens_per_second": sum(len(row["output_ids"]) for row in rows) / wall_seconds}


def _observed_gpus(torch, target):
    bindings = target["gpu_bindings"]
    if torch.cuda.device_count() != len(bindings):
        raise ValueError("visible GPU count differs from the target")
    result = {}
    for node, spec in bindings.items():
        index = spec["ordinal"]
        props = torch.cuda.get_device_properties(index)
        actual = {"ordinal": index, "uuid": _uuid(props.uuid), "name": props.name, "capability": [props.major, props.minor]}
        if actual != {**spec, "uuid": _uuid(spec["uuid"])}:
            raise ValueError("visible GPU identity or placement differs from the target")
        result[node] = {**actual, "total_memory_bytes": props.total_memory}
    return result


def _load_proof(path, result, *, same_arm):
    proof = _load(path)
    if proof.get("status") != "completed" or proof.get("phase") != "parity":
        raise ValueError("required qualification is not a completed parity run")
    for key in ("target_sha256", "scenario_sha256", "topology_sha256", "runtime", "schedule_inputs_sha256",
                "ingress_contract_sha256", "input_tokenization_sha256"):
        if proof[key] != result[key]:
            raise ValueError(f"qualification differs on {key}")
    if (result["ingress_contract_sha256"] is not None and
            (not isinstance(proof.get("global_ticks_sha256"), str) or not ingress.SHA.fullmatch(proof["global_ticks_sha256"]))):
        raise ValueError("qualification lacks the complete bound global forward signature")
    if same_arm:
        if (proof["arm"], proof["compiled_plan_sha256"]) != (result["arm"], result["compiled_plan_sha256"]):
            raise ValueError("timed trial differs from its qualified arm or generated layout")
        if any(proof.get(key) != result.get(key) for key in ("backing_mode", "resident_binding_sha256")):
            raise ValueError("timed trial differs from its qualified backing mode or resident proof")
    elif proof["arm"] != "native" or proof.get("backing_mode", "tiered") != "tiered":
        raise ValueError("generated and hand-written parity must compare to the native control")
    return proof


def _verify_resident_rank(dispatch, bound, rank, binding, *, diagnostic):
    proof = binding["proof"]
    allocated = bound.get("resident_sentinel_allocated_delta_bytes")
    reserved = bound.get("resident_sentinel_reserved_delta_bytes")
    limit = binding["allocator_charge_bytes_by_rank"][rank]
    logical = resident.RESIDENT_SENTINEL_LOGICAL_BYTES
    if (bound.get("backing_mode") != resident.RESIDENT_TIER or bound.get("resident_proof_sha256") != fingerprint(proof)
            or bound.get("resident_sentinel_logical_bytes") != logical or bound.get("resident_sentinel_allocator_charge_bytes") != limit
            or type(allocated) is not int or not logical <= allocated <= limit or type(reserved) is not int or not 0 <= reserved <= limit
            or any(bound.get(key) != 0 for key in ("pinned_host_bytes", "local_hbm_bytes", "peer_hbm_bytes"))
            or bound.get("relay_workspace_bytes") != {"compute": 0, "host": 0, "peer": 0}
            or dispatch.get("backup_payload_bytes_by_tier") != {resident.RESIDENT_TIER: 0}
            or dispatch.get("relay") is not None):
        raise ValueError("resident backing did not meet its exact no-cold allocation and transfer contract")
    diagnostic_record = dispatch.get("resident_producer_oracle")
    usage = dispatch.get("selected_miss_payload_bytes_by_tier")
    if not diagnostic:
        if diagnostic_record is not None or usage is not None:
            raise ValueError("resident timing included a diagnostic shadow")
        return
    bounds = [item for item in proof["requests"] if item["rank"] == rank]
    expected = {item["rid"]: item["max_new_tokens"] - 1 for item in bounds}
    count = resident.LAYERS * sum(expected.values())
    if (type(diagnostic_record) is not dict or diagnostic_record.get("decode_steps_by_rid") != expected
            or diagnostic_record.get("verified_swaps") != count or diagnostic_record.get("verified_newest_rows") != count
            or diagnostic_record.get("captured_prefill_rows") != resident.LAYERS * sum(item["prompt_length"] for item in bounds)
            or type(diagnostic_record.get("verified_selected_history_rows")) is not int
            or diagnostic_record["verified_selected_history_rows"] <= 0
            or not isinstance(diagnostic_record.get("producer_and_selected_sha256"), str)
            or not ingress.SHA.fullmatch(diagnostic_record["producer_and_selected_sha256"])
            or usage != {resident.RESIDENT_TIER: 0}):
        raise ValueError("resident parity lacks complete independent packed producer evidence")


def _verify_run(directory, rows, plans, *, diagnostic, attention_dp_size, resident_binding=None):
    rank_records, kv_digests = [], []
    for plan in plans:
        rank = plan.rank
        assigned = [row for row in rows if attention_dp_size == 1 or row["routed_dp_rank"] == rank]
        rids = sorted(row["rid"] for row in assigned)
        total_decode = sum(len(row["output_ids"]) - 1 for row in assigned)
        schedule = _load(directory / f"schedule-rank-{rank}.json")
        dispatch = _load(directory / f"dispatch-rank-{rank}.json")
        if dispatch["finished"] != rids or dispatch["rank"] != rank:
            raise ValueError("actual worker did not complete the fixed request set")
        if {rid for batch in dispatch["batches"] for rid in batch["request_ids"]} != set(rids):
            raise ValueError("worker dispatch does not match its fixed request placement")
        backup_events = [json.loads(line)["event"] for line in (directory / f"backup-rank-{rank}.jsonl").read_bytes().splitlines()]
        expected_backup = ["backup_dependency_installed"]
        if resident_binding is None and any(len(row["output_ids"]) >= 3 for row in assigned):
            expected_backup.append("backup_dependency_exercised")
        if backup_events != expected_backup:
            raise ValueError("the worker did not exercise the bound backup-ordering dependency")
        events = [json.loads(line) for line in (directory / f"backing-rank-{rank}.jsonl").read_bytes().splitlines()]
        bound = [event for event in events if event["event"] == "hisparse_cold_backing_bound"]
        tiers = plan.layer_tiers if resident_binding is None else (resident.RESIDENT_TIER,) * resident.LAYERS
        if (len(bound) != 1 or bound[0]["cuda_device"] != plan.compute_device or tuple(bound[0]["layer_tiers"]) != tiers
                or bound[0]["attention_tp_size"] != plan.world_size // attention_dp_size):
            raise ValueError("actual worker backing did not match the compiled tier assignment")
        if resident_binding is not None:
            _verify_resident_rank(dispatch, bound[0], rank, resident_binding, diagnostic=diagnostic)
        elif bound[0].get("backing_mode", "tiered") != "tiered" or "resident_producer_oracle" in dispatch:
            raise ValueError("tiered arm unexpectedly used the resident backing")
        if plan.peer_hbm_bytes and backing.PEER_RELAY in plan.layer_tiers:
            relay = dispatch["relay"]
            if not relay or not relay["peer_to_host_read_payload_bytes"] or not relay["host_to_peer_write_payload_bytes"]:
                raise ValueError("the bound staged route did not carry real cold KV")
            if diagnostic:
                consumed = dispatch["selected_miss_payload_bytes_by_tier"][backing.PEER_RELAY]
                if not consumed or relay["host_to_compute_read_payload_bytes"] != consumed:
                    raise ValueError("the staged route did not supply verified selected KV misses to attention")
        if diagnostic and backing.PEER_HBM in plan.layer_tiers:
            if not dispatch["selected_miss_payload_bytes_by_tier"][backing.PEER_HBM]:
                raise ValueError("direct peer backing did not supply selected KV misses to attention")
        if diagnostic:
            counts = [0] * len(plan.layer_tiers)
            comparable = []
            for line in (directory / f"kv-rank-{rank}.jsonl").read_bytes().splitlines():
                event = json.loads(line)
                if event["event"] != "kv_selected_exact" or event["rank"] != rank:
                    raise ValueError("packed KV parity contains an unverified decode event")
                counts[event["layer"]] += len(event["rows"])
                comparable.append({**event, "rows": [{k: v for k, v in row.items() if k != "miss_rows"} for row in event["rows"]]})
            if counts != [total_decode] * len(counts):
                raise ValueError("packed KV parity did not cover each real decode row and layer")
            kv_digests.append(fingerprint(comparable))
        parallel = schedule["parallel"]
        if (parallel["attention_dp_size"] != attention_dp_size or parallel["attention_tp_size"] != plan.world_size // attention_dp_size
                or attention_dp_size == 2 and parallel["attention_dp_rank"] != rank):
            raise ValueError("actual worker attention partition differs from the external rank binding")
        rank_records.append({"schedule": schedule["effective"], "dispatch": dispatch, "backing": bound[0], "parallel": parallel,
                             "backup_dependency": backup_events})
    return rank_records, kv_digests


def _compare(proof, result, *, parity):
    fields = ("effective_schedule_sha256", "dispatch_sha256", "output_ids_sha256", "global_ticks_sha256")
    if parity:
        fields += ("packed_kv_sha256_by_rank",)
    if any(proof[key] != result[key] for key in fields):
        raise ValueError("live output, selected KV, or actual dispatch differs from qualification")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("model", "target", "topology", "scenario", "fixture", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--arm", choices=("native", "manual", "generated"), required=True)
    parser.add_argument("--phase", choices=("parity", "performance"), required=True)
    parser.add_argument("--program", type=Path)
    parser.add_argument("--resident-proof", type=Path)
    parser.add_argument("--external-schedule", type=Path)
    parser.add_argument("--link-evidence", type=Path, action="append", default=[])
    parser.add_argument("--qualification", type=Path)
    parser.add_argument("--require-remote", action="store_true")
    parser.add_argument("--deadline", type=int, default=600)
    args = parser.parse_args()
    if (args.resident_proof is not None and (args.arm == "native" or args.program is not None)
            or args.resident_proof is None and (args.arm == "native") == (args.program is not None)):
        parser.error("only the manual and generated arms take a frozen program")
    required = args.phase == "performance" or args.arm != "native"
    if required != (args.qualification is not None) or not 1 <= args.deadline <= 3600:
        parser.error("candidate parity needs native qualification and timing needs same-arm parity")

    directory = args.output.resolve()
    directory.mkdir(parents=True, exist_ok=False)
    result = {"schema": 1, "status": "started", "arm": args.arm, "phase": args.phase,
              "scope": "independent real model requests; session and branch labels do not imply prefix retention or COW"}
    engine = None
    try:
        target, fixture, scenario, external, model, compiled, plans, tokenized = _bind(args)
        import sglang
        import torch
        from sglang.srt.entrypoints.engine import Engine

        backing.check_sources(sglang)
        gate = external.get(ingress.EXTERNAL_FIELD)
        ingress_sha = fingerprint(gate) if gate is not None else None
        commitments = _token_commitments(tokenized)
        resident_binding = _load(args.resident_proof) if args.resident_proof is not None else None
        if resident_binding is not None:
            if (gate is None or target.get("attention_dp_size") != 2 or target["pool"] != {
                    "hot_slots": resident.HOT_SLOTS, "layers": resident.LAYERS, "page_size": resident.PAGE_ROWS,
                    "rows": external["engine"]["max_total_tokens"] * 2 + resident.PAGE_ROWS}
                    or scenario.kv_row_bytes != resident.ROW_BYTES or scenario.top_k != resident.TOP_K):
                raise ValueError("resident proof requires the fixed native DP2 pool geometry")
            resident.checked_binding(external["engine"], fixture, commitments, resident_binding)
        fixed_env = {name: os.environ.get(name) for name in ("SGLANG_DSA_FUSE_TOPK", "SGLANG_DISABLE_HISPARSE_PREFETCH")}
        if gate is not None and fixed_env != {"SGLANG_DSA_FUSE_TOPK": "0", "SGLANG_DISABLE_HISPARSE_PREFETCH": "1"}:
            raise ValueError("gated replay requires the frozen unfused selector and disabled HiSparse prefetch")
        runtime = {"gpus": _observed_gpus(torch, target), "sglang": sglang.__version__, "torch": str(torch.__version__),
                   "cuda_build": torch.version.cuda, "sglang_source_sha256": backing.SOURCES,
                   "ingress_dependency_source_sha256": ingress.SOURCES if gate is not None else None,
                   "fixed_environment": fixed_env,
                   "backup_dependency_source_sha256": _hash_file(sglang_dependency.__file__),
                   "adapter_source_sha256": {name: _hash_file(Path(__file__).with_name(name)) for name in
                                             ("sglang_serving.py", "sglang_smoke.py", "sglang_backing.py", "sglang_relay.py",
                                              "sglang_observe.py", "sglang_dependency.py", "sglang_sources.py",
                                              "sglang_ingress_gate.py", "sglang_global_ticks.py", "sglang_resident.py",
                                              "sglang_resident_oracle.py")}}
        nccl_override = os.environ.get("SGLANG_NCCL_SO_PATH")
        runtime["nccl_override_sha256"] = _hash_file(nccl_override) if nccl_override else None
        result.update(runtime=runtime, target_sha256=scenario.target_sha256, scenario_sha256=scenario.fingerprint(),
                      topology_sha256=compiled["topology_sha256"], schedule_inputs_sha256=fingerprint(external),
                      compiled_plan_sha256=fingerprint(compiled), program_sha256=compiled["program_sha256"],
                      ingress_contract_sha256=ingress_sha, input_tokenization_sha256=fingerprint(commitments),
                      backing_mode="tiered" if resident_binding is None else resident.RESIDENT_TIER,
                      resident_binding_sha256=fingerprint(resident_binding) if resident_binding is not None else None,
                      qualification_sha256=_hash_file(args.qualification) if args.qualification else None,
                      attention_dp_size=target.get("attention_dp_size", 1))
        proof = _load_proof(args.qualification, result, same_arm=args.phase == "performance") if args.qualification else None
        observe.save(directory / "compiled_plan.json", compiled)
        observe.save(directory / "external_schedule.json", external)
        attention_dp_size = target.get("attention_dp_size", 1)
        observe.save(directory / "worker_input.json", {"phase": args.phase, "plans": [asdict(plan) for plan in plans],
                     ingress.EXTERNAL_FIELD: gate,
                      "resident_binding": resident_binding, "external_schedule": external, "token_commitments": commitments,
                     "rids_by_rank": [[row["rid"] for name in ("warmup", "requests") for row in fixture[name]
                                       if attention_dp_size == 1 or row["routed_dp_rank"] == plan.rank] for plan in plans]})
        os.environ["RSI_TOPOLOGY_SERVING_DIR"] = str(directory)

        class TopologyEngine(Engine):
            run_scheduler_process_func = staticmethod(_worker)

        start = time.monotonic()
        engine = TopologyEngine(model_path=str(model), **external["engine"])
        result["engine_init_seconds"] = time.monotonic() - start
        if engine.tokenizer_manager.incremental_streaming_output:
            raise ValueError("topology runner requires the audited cumulative streaming interface")

        async def replay():
            warmup = await _replay(engine, fixture["warmup"], tokenized, args.deadline)
            measured = await _replay(engine, fixture["requests"], tokenized, args.deadline)
            return warmup, measured

        warmup, measured = engine.loop.run_until_complete(replay())
        limit = time.monotonic() + 15
        dispatch_paths = [directory / f"dispatch-rank-{plan.rank}.json" for plan in plans]
        if gate is not None:
            dispatch_paths += [directory / f"global-ticks-rank-{plan.rank}.json" for plan in plans]
        while time.monotonic() < limit and not all(path.is_file() for path in dispatch_paths):
            time.sleep(.02)
        if not all(path.is_file() for path in dispatch_paths):
            raise TimeoutError("the scheduler did not publish complete dispatch receipts")
        rank_records, kv_digests = _verify_run(directory, [*warmup, *measured], plans, diagnostic=args.phase == "parity",
                                               attention_dp_size=attention_dp_size, resident_binding=resident_binding)
        dispatch_global = None
        if gate is not None:
            releases = [[json.loads(line) for line in (directory / f"ingress-rank-{rank}.jsonl").read_bytes().splitlines()]
                        for rank in range(2)]
            result["ingress"] = ingress.verify_ingress_events(releases, gate)
            tick_files = [directory / f"global-ticks-rank-{rank}.json" for rank in range(2)]
            observations = [_load(path) for path in tick_files]
            result["global_tick_receipt_sha256_by_rank"] = [_hash_file(path) for path in tick_files]
            if any(row.get("schema") != 1 or row.get("rank") != rank or row.get("pid") != releases[rank][0]["pid"]
                   for rank, row in enumerate(observations)):
                raise ValueError("global forward traces did not originate in the bound ingress workers")
            dispatch_global = global_ticks.verify_global_ticks([row["ticks"] for row in observations], gate)
        result.update(warmup=warmup, requests=measured, ranks=rank_records, packed_kv_sha256_by_rank=kv_digests,
                      global_ticks=dispatch_global, global_ticks_sha256=dispatch_global["global_ticks_sha256"] if dispatch_global else None,
                      effective_schedule_sha256=fingerprint([rank["schedule"] for rank in rank_records]),
                      dispatch_sha256=fingerprint([rank["dispatch"]["batches"] for rank in rank_records]),
                      output_ids_sha256=fingerprint([(row["rid"], row["input_ids_sha256"], row["output_ids"]) for row in [*warmup, *measured]]))
        if proof is not None:
            _compare(proof, result, parity=args.phase == "parity")
        if args.phase == "performance":
            result["metrics"] = _metrics(measured)
        result["status"] = "completed"
    except Exception as exc:
        result.update(status="failed", error_type=type(exc).__name__, error=str(exc))
        raise
    finally:
        try:
            if engine is not None:
                engine.shutdown()
        finally:
            observe.save(directory / "result.json", result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
