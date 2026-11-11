"""Build an unverified HiSparse attention-DP2 case from an actual pinned-copy profile."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from statistics import median

from rsi_memory.memory_program import MemoryProgram, MemoryScenario, plan_backing
from rsi_memory.topology import HardwareTopology, fingerprint

from . import sglang_backing as backing, sglang_serving as serving
from .generate_serving_case import build_case
from .sglang_observe import save


POOL = {"rows": 32832, "layers": 6, "page_size": 64, "hot_slots": 4096}
TOP_K = 2048
ROW_BYTES = 656
NEW_TOKENS = 64
COMPUTE = ["gpu-a", "gpu-b"]
HOST = "host"
ALLOWED = {node: [*COMPUTE, HOST] for node in COMPUTE}
PROFILE_SCOPE = "synchronized_directed_pinned_host_copy_not_attention_latency"


def measured_topology(raw):
    """Convert the first two measured CUDA ordinals without inventing peer edges."""
    profile = json.loads(raw)
    if profile.get("schema") != 1 or profile.get("scope") != PROFILE_SCOPE:
        raise ValueError("requires the directed torch pinned-copy profile")
    if profile["peer_capability"][0][1] is not False or profile["peer_capability"][1][0] is not False:
        raise ValueError("this case requires a measured non-peer-addressable CUDA pair")
    evidence = sha256(raw).hexdigest()
    slab = POOL["rows"] * ROW_BYTES
    workspace = backing.workspace_bytes(TOP_K, ROW_BYTES, POOL["layers"] - 1)
    gpu_budget = slab + workspace["compute"] + workspace["peer"]
    host_budget = len(COMPUTE) * (POOL["layers"] * slab + workspace["host"])
    graph = {"schema": 1, "nodes": [{"id": HOST, "kind": "host", "capacity_bytes": host_budget}],
             "links": [], "bottlenecks": []}
    bindings = {}
    for ordinal, node in enumerate(COMPUTE):
        observed = [row for row in profile["devices"] if row["ordinal"] == ordinal]
        if len(observed) != 1:
            raise ValueError(f"missing unique profiled device at ordinal {ordinal}")
        device = observed[0]
        bindings[node] = {key: device[key] for key in ("ordinal", "uuid", "name", "capability")}
        bindings[node]["uuid"] = serving._uuid(device["uuid"])
        graph["nodes"].append({"id": node, "kind": "gpu", "capacity_bytes": gpu_budget,
                               "physical_bytes": device["total_memory"], "evidence_sha256": evidence})
        for direction, source, destination in (("h2d", HOST, node), ("d2h", node, HOST)):
            def matching(row):
                return row["gpu_ordinal"] == ordinal and row["direction"] == direction

            fits = [row for row in profile["affine_link_fits"] if matching(row)]
            records = [row for row in profile["records"] if matching(row)]
            if (len(fits) != 1 or len(records) < 3 or len({row["bytes"] for row in records}) != len(records)
                    or type(profile["samples"]) is not int or profile["samples"] < 2
                    or any(row["exact_bytes"] is not True or len(row["samples_ns"]) != profile["samples"]
                           or any(type(value) is not int or value <= 0 for value in row["samples_ns"])
                           or type(row["bytes"]) is not int or row["bytes"] <= 0
                           or row["median_ns"] != median(row["samples_ns"]) for row in records)):
                raise ValueError(f"{node} {direction} needs verified raw directed samples and a unique fit")
            fit = fits[0]
            small, large = min(records, key=lambda row: row["bytes"]), max(records, key=lambda row: row["bytes"])
            slope = max(1e-6, (large["median_ns"] - small["median_ns"]) / (large["bytes"] - small["bytes"]))
            if (fit["bandwidth_bytes_per_s"], fit["latency_ns"]) != (
                max(1, int(1e9 / slope)), max(0, int(small["median_ns"] - small["bytes"] * slope))
            ):
                raise ValueError(f"{node} {direction} fit differs from the recorded measurements")
            graph["links"].append({"id": f"{node}-{direction}", "src": source, "dst": destination,
                "bandwidth_bytes_per_s": fit["bandwidth_bytes_per_s"], "latency_ns": fit["latency_ns"],
                "bottlenecks": [], "transport": f"pinned_{direction}", "evidence_sha256": evidence})
    return graph, bindings


def make_fixture(briefing, supplement):
    prefix = briefing["prefix"]["prompt"].rsplit("\n\n", 1)[0]
    if not prefix or not supplement.strip() or len(briefing["branches"]) != 2:
        raise ValueError("requires the complete incident brief, supplement and two questions")
    context = f"{prefix}\n\n{supplement.strip()}"

    def request(rank, phase, question):
        return {"rid": f"archive-{phase}-r{rank}", "arrival_ms": 0, "max_new_tokens": NEW_TOKENS,
                "routed_dp_rank": rank, "prompt": f"{context}\n\nRequest\n{question}"}

    return serving.validate_fixture({"schema": 1, "name": "authored-archive-attention-dp2-v1",
        "warmup": [request(0, "warm", "Summarize the conditions under which an ambiguous date should remain blank. Include one example."),
                   request(1, "warm", "Describe the diagnostic comparison that makes a proxy issue unlikely. Give a concise explanation for an operator.")],
        "requests": [request(rank, "measured", item["prompt"]) for rank, item in enumerate(briefing["branches"])]})


def _controls():
    one = {"op": "const", "value": 1}
    two = {"op": "const", "value": 2}
    local = {"op": "feature", "name": "target_is_local"}
    host = {"op": "feature", "name": "target_is_host"}

    def program(score):
        return {"schema": 1, "kind": "kv_memory_program", "rules": [
            {"event": "allocate", "action": "place", "when": one, "score": score}]}

    manual = program({"op": "mul", "args": [{"op": "sub", "args": [one, local]}, {"op": "sub", "args": [two, host]}]})
    witness = program({"op": "add", "args": [{"op": "mul", "args": [two, local]}, host]})
    return manual, witness


def _preview(topology, scenario, bindings, raw_program, expected_tier):
    slabs = [{"key": f"r{rank}-l{layer}", "rank": rank, "layer": layer, "bytes": POOL["rows"] * ROW_BYTES,
              "allowed_nodes": ALLOWED[node]} for rank, node in scenario.rank_placement for layer in range(POOL["layers"])]
    plan = plan_backing(topology, scenario, MemoryProgram.from_dict(raw_program), slabs, transports=backing.NATIVE_TRANSPORTS)
    workers = backing.bind_plans(topology, scenario, plan, gpu_ordinals={node: row["ordinal"] for node, row in bindings.items()},
                                 attention_tp_size=1, **POOL)
    if any(worker.layer_tiers.count(expected_tier) != 1 or worker.layer_tiers.count(backing.HOST) != POOL["layers"] - 1
           for worker in workers):
        raise ValueError(f"the common budget did not produce exactly one {expected_tier} slab per rank")
    return plan


def build_dp2_case(output, *, profile, briefing, supplement, config_sha256, weights_sha256, tokenizer=None):
    raw = Path(profile).read_bytes()
    briefing_raw, supplement_raw = Path(briefing).read_bytes(), Path(supplement).read_bytes()
    topology, bindings = measured_topology(raw)
    fixture = make_fixture(json.loads(briefing_raw), supplement_raw.decode("utf-8"))
    lengths = None
    if tokenizer is not None:
        lengths = {row["rid"]: len(tokenizer.encode(row["prompt"], add_special_tokens=False))
                   for group in ("warmup", "requests") for row in fixture[group]}
        if any(not POOL["hot_slots"] < length <= 8192 - NEW_TOKENS for length in lengths.values()):
            raise ValueError(f"real tokenizer lengths outside the long decode envelope: {lengths}")
    manual, witness = _controls()
    manifest = build_case(output, topology=topology, gpu_bindings=bindings, fixture=fixture,
                          model_config_sha256=config_sha256, model_weights_sha256=weights_sha256,
                          pool=POOL, native_host=HOST, compute_nodes=COMPUTE, allowed_nodes=ALLOWED,
                          selector="torch-unfused", top_k=TOP_K, link_evidence=[profile], manual=manual,
                          attention_dp_size=2)
    output = Path(output)
    graph = HardwareTopology.from_dict(topology)
    scenario = MemoryScenario.from_dict(json.loads((output / "scenario.json").read_bytes()))
    _preview(graph, scenario, bindings, manual, backing.LOCAL_HBM)
    relay = _preview(graph, scenario, bindings, witness, backing.PEER_RELAY)
    save(output / "relay_witness_program.json", witness)
    save(output / "relay_witness_plan_preview.json", relay)
    source = json.loads(raw)
    scope = {"schema": 1, "status": "bound_unverified", "model_config_sha256": config_sha256,
             "model_weights_sha256": weights_sha256, "briefing_sha256": sha256(briefing_raw).hexdigest(),
             "supplement_sha256": sha256(supplement_raw).hexdigest(), "token_lengths": lengths,
             "utf8_prompt_bytes": {row["rid"]: len(row["prompt"].encode()) for group in ("warmup", "requests") for row in fixture[group]},
             "link_profile": {key: source[key] for key in ("scope", "utc", "torch_version", "host_cpu_affinity", "fit_limitation")},
             "link_profile_sha256": sha256(raw).hexdigest(), "cold_budget_bytes": {row["id"]: row["capacity_bytes"] for row in topology["nodes"]},
             "scope": "authored independent requests for public serving qualification; no measured serving scores",
             "native": "stock host-only cold backing with common external backup dependency",
             "manual_heuristic": "one first local slab per rank; remaining layers use pinned host; no trace tuning",
             "manual_tuned": None, "generated": None,
             "relay_witness": "hand-authored one peer slab per rank for transfer and attention parity qualification",
             "relay_witness_program_sha256": relay["program_sha256"], "relay_witness_plan_sha256": fingerprint(relay)}
    save(output / "case_scope.json", scope)
    return {**manifest, "case_scope": scope}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("profile", "briefing", "supplement", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--model-weights-sha256", required=True)
    parser.add_argument("--tokenizer", type=Path, help="optional local bound model directory for real token lengths")
    args = parser.parse_args()
    tokenizer = None
    if args.tokenizer is not None:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, local_files_only=True)
    result = build_dp2_case(args.output, profile=args.profile, briefing=args.briefing, supplement=args.supplement,
                            config_sha256=args.model_config_sha256, weights_sha256=args.model_weights_sha256,
                            tokenizer=tokenizer)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
