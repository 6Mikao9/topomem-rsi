"""Verify and preregister a train-selected SGLang control from archived leased jobs."""

import argparse
import atexit
from hashlib import sha256
import json
from pathlib import Path
import tempfile

from rsi_memory import topology_rsi
from rsi_memory.execution_client import PersistentExecutor
from rsi_memory.execution_worker import validate_job_id
from rsi_memory.memory_program import MemoryProgram
from rsi_memory.topology import fingerprint

from .contract import read_json
from .manual_sweep import DEFAULT, execute, prepare


ARTIFACTS = ("manifest.json", "jobs.jsonl", "result.json", "selected_program.json")


def _read(path):
    path = Path(path)
    if path.is_symlink() or not path.is_file():
        raise ValueError("missing or symlinked hand-sweep evidence " + path.name)
    return path.read_bytes()


def _sha(raw):
    return sha256(raw).hexdigest()


def _inputs(source_config, sweep_dir):
    source_config, sweep_dir = Path(source_config), Path(sweep_dir)
    if sweep_dir.is_symlink() or not sweep_dir.is_dir():
        raise ValueError("hand sweep requires a real evidence directory")
    old_bytes = _read(source_config)
    old_config = json.loads(old_bytes)
    if type(old_config) is not dict or old_config.get("schema") != 2:
        raise ValueError("hand-sweep evidence must originate from the frozen schema-2 adapter")
    return old_config, _sha(old_bytes), {name: _read(sweep_dir / name) for name in ARTIFACTS}


def _receipts(source_config, old_config, events):
    directory = Path(source_config).parent / old_config["receipt_directory"]
    if directory.is_symlink() or not directory.is_dir():
        raise ValueError("hand sweep requires its original leased receipt directory")
    ids = set()
    for line in events.splitlines():
        row = json.loads(line)
        ids.add(validate_job_id(row["job_id"]))
        if "repeat_job_id" in row:
            ids.add(validate_job_id(row["repeat_job_id"]))
    return directory, {job_id: _sha(_read(directory / f"{job_id}.json")) for job_id in sorted(ids)}


def _preregister(result):
    scores = result["sum_window_ms_by_block"][DEFAULT]
    if len(scores) != 3:
        raise ValueError("the preregistration requires three measured default native training blocks")
    spread = max(scores) - min(scores)
    return {"rounds": 2, "slots": 2, "max_api_calls_per_arm_per_round": 5, "max_output_tokens": 65536,
            "reasoning_effort": "high", "validation_blocks": 6, "test_blocks": 12, "wall_seconds": 14400,
            "minimum_paired_blocks": 6, "native_default_training_scores_ms": scores,
            "min_improvement_formula": "max(5 ms, 2 * range(default native training scores))",
            "min_improvement_ms": max(5.0, 2 * spread)}


def pin_selection(source_config, sweep_dir):
    """Prepare local config fields. The adapter verifies them by offline replay before use."""
    old_config, old_sha, raws = _inputs(source_config, sweep_dir)
    _, receipt_hashes = _receipts(source_config, old_config, raws["jobs.jsonl"])
    proof = {"schema": 1, "sweep_dir": str(Path(sweep_dir).absolute()), "source_config": str(Path(source_config).absolute()),
             "source_config_sha256": old_sha, "artifact_sha256": {name: _sha(raw) for name, raw in raws.items()},
             "receipts_sha256": fingerprint(receipt_hashes)}
    return {"manual_program": json.loads(raws["selected_program.json"]), "manual_selection": proof,
            "controller_preregistration": _preregister(json.loads(raws["result.json"]))}


class _ArchivedExecutor:
    def __init__(self, directory, hashes, *, source, source_manifest_sha256, gpu_indices, gpu_uuids, environment=None, **unused):
        self.directory, self.hashes = directory, hashes
        self.seen = set()
        self.config = {"source": source, "source_manifest_sha256": source_manifest_sha256, "gpu_indices": gpu_indices,
                       "gpu_uuids": gpu_uuids, "environment": environment or {}}

    def status(self, job_id):
        validate_job_id(job_id)
        raw = _read(self.directory / f"{job_id}.json")
        if _sha(raw) != self.hashes.get(job_id):
            raise ValueError("hand-sweep leased receipt is unbound or changed")
        receipt = json.loads(raw)
        PersistentExecutor._check_receipt(self, receipt, job_id)
        if receipt["status"] != "completed" or receipt.get("process_group_cleanup_required") is not False:
            raise ValueError("hand-sweep job lacked verified clean completion")
        self.seen.add(job_id)
        return receipt

    def run(self, *args, **kwargs):
        raise RuntimeError("offline hand-sweep verification cannot submit jobs")

    def close(self):
        pass


def verify_selection(adapter):
    """Replay the fixed training matrix and reconstruct selection and preregistration."""
    from .domain import SGLangDomain

    config = adapter.config
    proof = config["manual_selection"]
    fields = {"schema", "sweep_dir", "source_config", "source_config_sha256", "artifact_sha256", "receipts_sha256"}
    if type(proof) is not dict or proof.keys() != fields or type(proof["schema"]) is not int or proof["schema"] != 1:
        raise ValueError("invalid hand-sweep selection provenance")
    source_config = Path(proof["source_config"])
    old_config, old_sha, raws = _inputs(source_config, proof["sweep_dir"])
    if old_sha != proof["source_config_sha256"] or {name: _sha(raw) for name, raw in raws.items()} != proof["artifact_sha256"]:
        raise ValueError("hand-sweep artifact or source config digest changed")
    directory, receipt_hashes = _receipts(source_config, old_config, raws["jobs.jsonl"])
    if fingerprint(receipt_hashes) != proof["receipts_sha256"]:
        raise ValueError("hand-sweep leased receipt inventory changed")
    if (any(old_config[key] != config[key] for key in ("source_manifest_sha256", "launch_sha256", "hot_profile"))
            or old_config["cases"]["train"] != adapter.cases["train"]
            or any(old_config["executor"].get(key) != config["executor"].get(key) for key in ("gpu_indices", "gpu_uuids", "environment"))
            or any(old_config["token_inputs"][case["id"]] != adapter.token_inputs[case["id"]] for case in adapter.cases["train"])):
        raise ValueError("hand-sweep evidence binds a different training target, profile, or deployment")
    manifest = json.loads(raws["manifest.json"])

    def archived_executor(**kwargs):
        return _ArchivedExecutor(directory, receipt_hashes, **kwargs)

    replay = SGLangDomain(old_config, source_config.parent, executor_factory=archived_executor)
    try:
        expected = prepare(replay, manifest["sweep_id"], 3, 5.0)
        if manifest != expected:
            raise ValueError("hand-sweep policy set, training inputs or order differs from preregistered design")
        with tempfile.TemporaryDirectory(prefix="topomem-hand-proof-") as temp:
            result = execute(replay, expected, Path(temp))
            if (result != json.loads(raws["result.json"]) or (Path(temp) / "jobs.jsonl").read_bytes().splitlines() != raws["jobs.jsonl"].splitlines()
                    or replay.executor.seen != set(receipt_hashes)):
                raise ValueError("hand-sweep selection or job log disagrees with replay of leased receipts")
    finally:
        replay.close()
        atexit.unregister(replay.close)
    selected = manifest["policies"][result["selected"]]
    program = MemoryProgram.from_dict(config["manual_program"])
    if (program.to_dict() != selected["program"] or program.to_dict() != json.loads(raws["selected_program.json"])
            or program.fingerprint() != result["selected_program_sha256"]
            or adapter.effective_program_fingerprint(program) != selected["effective_sha256"]):
        raise ValueError("manual program differs from the verified training selection")
    preregistration = _preregister(result)
    if (config["controller_preregistration"] != preregistration
            or len(adapter.blocks["validation"]) != preregistration["validation_blocks"]
            or len(adapter.blocks["test"]) != preregistration["test_blocks"]):
        raise ValueError("controller budget or improvement threshold differs from the frozen training preregistration")
    evidence = {"schema": 1, "scope": "verified training-only hand sweep", "selected": result["selected"],
                "selected_program_sha256": program.fingerprint(), "selected_effective_sha256": selected["effective_sha256"],
                "hot_slots": selected["compilation"]["startup"]["hot_slots"], "layer_tiers_by_rank": selected["layer_tiers_by_rank"],
                "backing_mode": selected["backing_mode"],
                "sweep_sha256": result["sweep_sha256"], "source_config_sha256": old_sha,
                "artifact_sha256": proof["artifact_sha256"], "receipts_sha256": proof["receipts_sha256"],
                "receipt_count": len(receipt_hashes)}
    return program, evidence, preregistration


def main():
    from .domain import SGLangDomain

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("run", "freeze", "test"))
    parser.add_argument("--adapter-config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--key-file", type=Path)
    parser.add_argument("--round-limit", type=int)
    args = parser.parse_args()
    config = read_json(args.adapter_config)
    if type(config) is not dict or config.get("schema") != 3:
        parser.error("require a schema-3 adapter with a verified hand control")
    adapter = SGLangDomain(config, args.adapter_config.resolve().parent)
    budget = adapter.contract["controller_preregistration"]
    controller_config = topology_rsi.Config(rounds=budget["rounds"], slots=budget["slots"],
                                            max_output_tokens=budget["max_output_tokens"], min_improvement=budget["min_improvement_ms"],
                                            wall_seconds=budget["wall_seconds"], min_paired_blocks=budget["minimum_paired_blocks"])
    request = (topology_rsi.make_requester(topology_rsi.load_api_key(args.key_file)) if args.command == "run" else topology_rsi._no_api)
    controller = topology_rsi.TopologyRSI(args.output, controller_config, adapter.domain(), request)
    try:
        if args.command == "run":
            summary = {"status": controller.run(args.round_limit)["status"]}
        elif args.command == "freeze":
            summary = {"status": "frozen", "rounds": controller.freeze()["rounds"]}
        else:
            summary = {"status": controller.test()["status"]}
        print(json.dumps(summary, sort_keys=True))
    finally:
        controller.close()
        adapter.close()


if __name__ == "__main__":
    main()
