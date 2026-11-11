"""Bind one immutable request replay to a measured HiSparse topology and target."""

import argparse
from hashlib import sha256
import json
from pathlib import Path

from rsi_memory.memory_program import MemoryProgram, MemoryScenario, plan_backing
from rsi_memory.topology import HardwareTopology, fingerprint

from . import sglang_backing as backing, sglang_serving as serving
from .sglang_observe import save


def _digest(text):
    if type(text) is not str or len(text) != 64 or any(char not in "0123456789abcdef" for char in text):
        raise ValueError("model identities require lowercase SHA-256 digests")
    return text


def build_case(output, *, topology, gpu_bindings, fixture, model_config_sha256, model_weights_sha256,
               pool, native_host, compute_nodes, allowed_nodes, selector, top_k, link_evidence, manual=None,
               attention_dp_size=1):
    """Write a new case without changing the replay or inventing any route profile."""
    graph = HardwareTopology.from_dict(topology)
    serving.validate_fixture(fixture)
    if (type(compute_nodes) is not list or not 1 <= len(compute_nodes) <= 2
            or any(type(node) is not str for node in compute_nodes) or len(set(compute_nodes)) != len(compute_nodes)
            or selector not in serving.SELECTORS or type(top_k) is not int or top_k != 2048):
        raise ValueError("invalid external rank placement or fixed attention selector")
    if type(pool) is not dict or set(pool) != {"rows", "layers", "page_size", "hot_slots"} or any(
            type(value) is not int or value <= 0 for value in pool.values()) or pool["hot_slots"] != 4096:
        raise ValueError("provide the audited cold KV pool shape")
    nodes = {node.id for node in graph.nodes if node.kind == "gpu"}
    if type(gpu_bindings) is not dict or set(gpu_bindings) != nodes or len(nodes) < 2 or not set(compute_nodes) <= nodes:
        raise ValueError("bind every GPU node, including any separate memory GPU")
    ordinals = {}
    uuids = set()
    for node, spec in gpu_bindings.items():
        if (type(spec) is not dict or set(spec) != {"ordinal", "uuid", "name", "capability"}
                or type(spec["ordinal"]) is not int or type(spec["name"]) is not str or not spec["name"]
                or type(spec["capability"]) is not list or len(spec["capability"]) != 2
                or any(type(value) is not int or value < 0 for value in spec["capability"])):
            raise ValueError("invalid observed GPU binding")
        ordinals[node] = spec["ordinal"]
        uuids.add(serving._uuid(spec["uuid"]))
    if (len(uuids) != len(nodes) or sorted(ordinals.values()) != list(range(len(nodes)))
            or [ordinals[node] for node in compute_nodes] != list(range(len(compute_nodes)))):
        raise ValueError("CUDA ordinals must identify unique GPUs with compute ranks first")
    if graph.node(native_host).kind != "host" or type(allowed_nodes) is not dict or set(allowed_nodes) != set(compute_nodes):
        raise ValueError("provide the common native pinned host and allowed storage for each rank")
    for allowed in allowed_nodes.values():
        if (type(allowed) is not list or not allowed or any(type(node) is not str for node in allowed)
                or len(set(allowed)) != len(allowed) or any(graph.node(node).kind == "fabric" for node in allowed)):
            raise ValueError("allowed storage nodes must be distinct GPU or host nodes")

    target = {"schema": 1, "gpu_bindings": gpu_bindings, "model_config_sha256": _digest(model_config_sha256),
              "model_weights_sha256": _digest(model_weights_sha256), "pool": pool, "native_host": native_host,
              "allowed_nodes": allowed_nodes}
    schedule = serving.schedule_inputs(fixture, len(compute_nodes), selector, attention_dp_size=attention_dp_size)
    if attention_dp_size != 1:
        target["attention_dp_size"] = attention_dp_size
    scenario = MemoryScenario.from_dict({"schema": 1, "target_sha256": fingerprint(target),
        "workload_sha256": fingerprint(fixture), "schedule_sha256": fingerprint(schedule),
        "rank_placement": [{"rank": rank, "node": node} for rank, node in enumerate(compute_nodes)],
        "attention": {"family": "dsa", "selector": selector, "selection_unit": "token", "top_k": top_k, "kv_row_bytes": 656}})

    evidence = {}
    for path in link_evidence:
        raw = Path(path).read_bytes()
        evidence[sha256(raw).hexdigest()] = raw
    if any(link.evidence_sha256 not in evidence for link in graph.links):
        raise ValueError("all exposed directed links require the matching measured evidence bytes")
    placements = {}
    if manual is not None:
        placements["manual"] = (MemoryProgram.from_dict(manual), allowed_nodes)
    placements["native"] = (MemoryProgram.from_dict(serving._PROGRAM), {node: [native_host] for node in compute_nodes})
    compiled = {}
    for arm, (program, allowed) in placements.items():
        slabs = [{"key": f"r{rank}-l{layer}", "rank": rank, "layer": layer, "bytes": pool["rows"] * 656,
                  "allowed_nodes": allowed[node]} for rank, node in scenario.rank_placement for layer in range(pool["layers"])]
        plan = plan_backing(graph, scenario, program, slabs, transports=backing.NATIVE_TRANSPORTS)
        backing.bind_plans(graph, scenario, plan, gpu_ordinals=ordinals,
                           attention_tp_size=1 if attention_dp_size == 2 else None, **pool)
        compiled[arm] = plan

    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    evidence_dir = output / "link_evidence"
    evidence_dir.mkdir()
    evidence_files = {}
    for digest in sorted({link.evidence_sha256 for link in graph.links}):
        name = f"{digest}.data"
        with (evidence_dir / name).open("xb") as stream:
            stream.write(evidence[digest])
        evidence_files[digest] = f"link_evidence/{name}"
    for name, record in (("target", target), ("topology", graph.to_dict()), ("scenario", scenario.to_dict()), ("fixture", fixture),
                         ("external_schedule", schedule), ("native_plan_preview", compiled["native"])):
        save(output / f"{name}.json", record)
    if manual is not None:
        save(output / "manual_program.json", manual)
        save(output / "manual_plan_preview.json", compiled["manual"])
    manifest = {"schema": 1, "status": "bound_unverified", "target_sha256": scenario.target_sha256,
                "topology_sha256": graph.fingerprint(), "scenario_sha256": scenario.fingerprint(),
                "workload_sha256": scenario.workload_sha256, "schedule_sha256": scenario.schedule_sha256,
                "link_evidence": evidence_files, "compiled_preview_sha256": {key: fingerprint(value) for key, value in compiled.items()}}
    save(output / "manifest.json", manifest)
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "topology", "gpu-bindings", "fixture", "pool", "allowed-nodes"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--model-config-sha256", required=True)
    parser.add_argument("--model-weights-sha256", required=True)
    parser.add_argument("--native-host", required=True)
    parser.add_argument("--compute-node", action="append", required=True)
    parser.add_argument("--link-evidence", type=Path, action="append", required=True)
    parser.add_argument("--selector", choices=serving.SELECTORS, default="torch-unfused")
    parser.add_argument("--top-k", type=int, default=2048)
    parser.add_argument("--attention-dp-size", type=int, choices=(1, 2), default=1)
    parser.add_argument("--manual-program", type=Path)
    args = parser.parse_args()

    def read(path):
        return json.loads(path.read_bytes())

    result = build_case(args.output, topology=read(args.topology), gpu_bindings=read(args.gpu_bindings),
                        fixture=read(args.fixture), pool=read(args.pool), allowed_nodes=read(args.allowed_nodes),
                        model_config_sha256=args.model_config_sha256, model_weights_sha256=args.model_weights_sha256,
                        native_host=args.native_host, compute_nodes=args.compute_node, selector=args.selector,
                        top_k=args.top_k, link_evidence=args.link_evidence, manual=read(args.manual_program) if args.manual_program else None,
                        attention_dp_size=args.attention_dp_size)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
