"""Build four independent CUDA RSI configurations from sealed workload content and measurements."""

from __future__ import annotations

import argparse
from hashlib import sha256
import json
from pathlib import Path

from .memory_program import MemoryProgram
from .scenarios import ScenarioWorkload
from .topology import HardwareTopology, _digest, _fields, _id, fingerprint


VARIANTS = {
    "agent_tools": {
        "train": ("agent-water-inspection", "agent-water-inspection"),
        "validation": ("agent-museum-access", "agent-observatory-handoff"),
        "test": ("agent-grid-recovery", "agent-clinic-supply"),
    },
    "chatbot_growth": {
        "train": ("chatbot-trip-planning", "chatbot-trip-planning"),
        "validation": ("chatbot-wetland-tour", "chatbot-pottery-course"),
        "test": ("chatbot-library-project", "chatbot-field-notebook"),
    },
    "long_document_qa": {
        "train": ("long-document-estuary-qa", "long-document-estuary-qa"),
        "validation": ("long-document-rail-qa", "long-document-archive-qa"),
        "test": ("long-document-flood-ledger", "long-document-optics-qa"),
    },
    "rollout_cow": {
        "train": ("rollout-bridge-options", "rollout-bridge-options"),
        "validation": ("rollout-warehouse-routes", "rollout-gallery-layouts"),
        "test": ("rollout-ferry-recovery", "rollout-dune-survey"),
    },
}
_GEOMETRY = frozenset({"block_rows", "head_dim", "top_k", "layers", "text_bytes_per_row"})
_PROFILE_SCOPE = "synchronized_directed_pinned_host_copy_not_attention_latency"


def _uuid(value):
    if type(value) is not str:
        raise ValueError("expected a physical GPU UUID")
    return value.lower().removeprefix("gpu-")


def _measurement(path):
    raw = Path(path).read_bytes()
    data = json.loads(raw)
    if data.get("schema") != 1 or data.get("scope") != _PROFILE_SCOPE:
        raise ValueError("expected a measured synchronized pinned-host copy profile")
    devices = {_uuid(row["uuid"]): row for row in data["devices"]}
    fits = {(row["gpu_ordinal"], row["direction"]): row for row in data["affine_link_fits"]}
    if len(devices) != len(data["devices"]) or len(fits) != len(data["affine_link_fits"]):
        raise ValueError("measured profile has ambiguous device or route records")
    return sha256(raw).hexdigest(), data, devices, fits


def _profile(profile, executor, measured):
    _fields(profile, ("topology", "target", "compute_node", "device_map", "compile"))
    digest, calibration, devices, fits = measured
    graph = HardwareTopology.from_dict(profile["topology"])
    if graph.bottlenecks or any(node.kind == "fabric" for node in graph.nodes):
        raise ValueError("this calibration has directed copy fits and no fabric contention measurements")
    mapping, target, geometry = profile["device_map"], dict(profile["target"]), profile["compile"]
    if (type(geometry) is not dict or not geometry.keys() <= _GEOMETRY
            or type(mapping) is not dict or mapping.keys() != {node.id for node in graph.nodes if node.kind == "gpu"}
            or len(mapping) < 2 or profile["compute_node"] not in mapping):
        raise ValueError("profile lacks the fixed compute placement or supported block geometry")
    uuids = executor["gpu_uuids"]
    if (type(uuids) is not list or len(uuids) != len(mapping) or any(type(index) is not int for index in mapping.values())
            or sorted(mapping.values()) != list(range(len(uuids))) or len(set(map(_uuid, uuids))) != len(uuids)):
        raise ValueError("profile device map must uniquely bind the executor's visible GPU UUIDs")
    bound = {}
    for node_id, index in mapping.items():
        row = devices.get(_uuid(uuids[index]))
        if row is None or graph.node(node_id).physical_bytes != row["total_memory"]:
            raise ValueError("GPU capacity does not match the selected calibrated physical GPU")
        bound[node_id] = row["ordinal"]
    hosts = [node.id for node in graph.nodes if node.kind == "host"]
    if len(hosts) != 1:
        raise ValueError("this measured profile requires one pinned host pool")
    host, qualified = hosts[0], set()
    for link in graph.links:
        h2d = link.src == host and link.dst in bound
        d2h = link.dst == host and link.src in bound
        if link.transport != "pinned_host" or not (h2d or d2h) or link.evidence_sha256 != digest:
            raise ValueError("profile contains a route without matching pinned-host measurement")
        device = link.dst if h2d else link.src
        direction = "h2d" if h2d else "d2h"
        fit = fits.get((bound[device], direction))
        if fit is None or (link.bandwidth_bytes_per_s, link.latency_ns) != (fit["bandwidth_bytes_per_s"], fit["latency_ns"]):
            raise ValueError("profile route parameters differ from the measured directed fit")
        qualified.add((device, direction))
    if qualified != {(node, direction) for node in bound for direction in ("h2d", "d2h")} or len(graph.links) != len(qualified):
        raise ValueError("each visible GPU must have exactly one measured route per host direction")
    for key in ("profile_sha256", "topology_evidence_sha256"):
        if key in target and target[key] != digest:
            raise ValueError("target calibration fingerprint disagrees with measured profile")
        target[key] = digest
    for key in ("host_cpu_affinity", "torch_version"):
        observed = calibration[key]
        if key in target and target[key] != observed:
            raise ValueError("target software or host affinity differs from its calibration")
        target[key] = observed
    return graph, host, target


def _live_blocks(packed):
    sessions = {}
    peak = 0
    for step in packed.steps:
        if step.op == "open":
            sessions[step.session] = set()
        elif step.op == "fork":
            sessions[step.session] = sessions[step.parent].copy()
        elif step.op == "close":
            del sessions[step.session]
        else:
            held = sessions[step.session]
            for mutation in step.writes:
                if mutation.replaces is not None:
                    held.remove(mutation.replaces)
                held.add(mutation.block)
        peak = max(peak, len(set().union(*sessions.values())))
    return peak


def build_configs(*, experiment_id, profiles, executor, calibration, source_manifest_sha256, initial_program,
                  manual_program, workload_directory, output_root, warmup=2, samples=3, paired_blocks=12, timeout=900):
    """Bind authored content splits and measured topology without starting a model or worker."""
    from integrations.topology_memory.torch_runtime import SUPPORTED_ACTIONS, SUPPORTED_FEATURES
    from integrations.topology_memory.torch_workload import compile_workload

    _id(experiment_id)
    _digest(source_manifest_sha256)
    if set(profiles) != set(VARIANTS):
        raise ValueError("one measured profile is required for each of four distinct scenarios")
    for value, low, high in ((warmup, 1, 10), (samples, 1, 100), (paired_blocks, 1, 100), (timeout, 1, 3600)):
        if type(value) is not int or not low <= value <= high:
            raise ValueError("invalid evaluation budget")
    measured = _measurement(calibration)
    policy_data = {}
    for name, data in (("initial_program", initial_program), ("manual_program", manual_program)):
        program = MemoryProgram.from_dict(data)
        program.validate_actions(SUPPORTED_ACTIONS)
        program.validate_features(SUPPORTED_FEATURES)
        policy_data[name] = program.to_dict()
    directories = Path(workload_directory), Path(output_root).resolve()
    configs = {}
    for kind, split_names in VARIANTS.items():
        profile = profiles[kind]
        active_executor = executor[kind] if set(executor) == set(VARIANTS) else executor
        graph, host, target = _profile(profile, active_executor, measured)
        cases, seen = {}, set()
        for split, names in split_names.items():
            cases[split] = []
            split_content = set()
            for ordinal, name in enumerate(names):
                workload = ScenarioWorkload.read(directories[0] / ("" if split == "train" else "variants") / f"{name}.json")
                content = workload.to_dict()["source"]["content_sha256"]
                if workload.kind != kind or content in seen or (split != "train" and content in split_content):
                    raise ValueError("workload content is shared across scenario splits")
                split_content.add(content)
                seed = int(fingerprint([experiment_id, kind, split, ordinal, workload.workload_sha256])[:15], 16)
                geometry = {**profile["compile"], "seed": seed}
                packed = compile_workload(workload, **geometry)
                compute = profile["compute_node"]
                if graph.node(compute).capacity_bytes < (packed.top_k+2)*packed.block_bytes:
                    raise ValueError(f"compute capacity cannot hold a full selection and producer workspace for {kind}")
                if graph.node(host).capacity_bytes < _live_blocks(packed)*packed.block_bytes:
                    raise ValueError(f"pinned host cap cannot hold the native baseline's live backing for {kind}")
                cases[split].append({"name": f"{name}-{ordinal:02d}", "workload": workload.to_dict(),
                                     "compile": geometry, "topology": graph.to_dict(), "target": target.copy(),
                                     "compute_node": compute, "device_map": profile["device_map"].copy()})
            seen.update(split_content)
        configs[kind] = {"experiment_id": f"{experiment_id}-{kind.replace('_', '-')}",
                         "source_manifest_sha256": source_manifest_sha256, "cases": cases,
                         "executor": active_executor.copy(), "output_directory": str(directories[1] / kind),
                         "paired_block_ids": {split: [f"{kind}-{split}-{i:03d}" for i in range(paired_blocks)]
                                              for split in ("validation", "test")},
                         "warmup": warmup, "samples": samples, "timeout": timeout, **policy_data}
    return configs


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inputs", type=Path, required=True, help="trusted JSON keyword arguments for build_configs")
    parser.add_argument("--output", type=Path, required=True, help="new directory for private, sealed configs")
    args = parser.parse_args()
    config = build_configs(**json.loads(args.inputs.read_text(encoding="utf-8")))
    args.output.mkdir(exist_ok=False)
    for kind, value in config.items():
        with (args.output / f"{kind}.json").open("x", encoding="utf-8", newline="\n") as stream:
            stream.write(json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
