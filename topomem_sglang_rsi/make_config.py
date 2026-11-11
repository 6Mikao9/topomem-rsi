"""Bind local case descriptors and an isolated remote launcher to an RSI run."""

import argparse
from hashlib import sha256
from pathlib import Path

from integrations.topology_memory.sglang_observe import save

from .contract import read_json
from .domain import SGLangDomain
from .hot_config import profile_adapter_sources
from .rebind_cases import formal_subset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for key in ("output", "case-set", "launch-config", "source-manifest", "initial-program", "tokenization-receipt", "hot-profile"):
        parser.add_argument(f"--{key}", type=Path, required=True)
    for key in ("experiment-id", "host", "remote-root", "remote-source", "remote-worker", "remote-launcher", "receipt-directory"):
        parser.add_argument(f"--{key}", required=True)
    parser.add_argument("--validation-blocks", type=int, default=6)
    parser.add_argument("--test-blocks", type=int)
    parser.add_argument("--manual-sweep", type=Path, help="completed training-only hand sweep evidence directory")
    parser.add_argument("--manual-sweep-config", type=Path, help="original frozen schema-2 adapter config used for the sweep")
    parser.add_argument("--deadline", type=int, default=300)
    parser.add_argument("--job-timeout", type=int, default=780)
    args = parser.parse_args()
    if (args.manual_sweep is None) != (args.manual_sweep_config is None):
        parser.error("provide both the hand-sweep directory and its original schema-2 adapter config")
    if args.test_blocks is None:
        args.test_blocks = 12 if args.manual_sweep is not None else 4
    if not 1 <= args.validation_blocks <= 100 or not 1 <= args.test_blocks <= 100:
        parser.error("paired block counts must be in [1,100]")
    launch = read_json(args.launch_config)
    if launch.get("schema") != 2 or not launch.get("cuda_home") or not launch.get("cache_root"):
        parser.error("require the schema-2 qualified CUDA launcher configuration")
    source_raw = args.source_manifest.read_bytes()
    source_files = read_json(args.source_manifest).get("files", {})
    launch_sha = sha256(args.launch_config.read_bytes()).hexdigest()
    expected_files = {f"integrations/topology_memory/{name}": digest for name, digest in profile_adapter_sources().items()}
    for relative in ("rsi_memory/memory_program.py", "topomem_sglang_rsi/contract.py",
                     "topomem_sglang_rsi/hot_config.py", "topomem_sglang_rsi/sglang_eval.py"):
        expected_files[relative] = sha256((Path(__file__).resolve().parents[1] / relative).read_bytes()).hexdigest()
    expected_files["sglang_launch.json"] = launch_sha
    if any(source_files.get(name) != digest for name, digest in expected_files.items()):
        parser.error("require the local resident-v2 evaluator and exact launcher in the worker source manifest")
    tokens = read_json(args.tokenization_receipt)
    if (tokens.get("schema") != 1 or tokens.get("status") != "tokenized"
            or tokens.get("case_set_manifest_sha256") != sha256(args.case_set.read_bytes()).hexdigest()):
        parser.error("tokenization receipt must bind the declared case set")
    cases = formal_subset(args.case_set)["cases"]
    config = {"schema": 2, "experiment_id": args.experiment_id, "source_manifest_sha256": sha256(source_raw).hexdigest(),
              "launch_sha256": launch_sha, "receipt_directory": args.receipt_directory,
              "local_cases": str(args.case_set.resolve(strict=True).parent), "cases": cases,
              "token_inputs": tokens["tokens_by_case"],
              "executor": {"host": args.host, "worker": args.remote_worker, "root": args.remote_root, "python": args.remote_launcher,
                           "source": args.remote_source, "gpu_indices": launch["gpu_indices"], "gpu_uuids": launch["gpu_uuids"]},
              "paired_block_ids": {"validation": [f"v{n+1}" for n in range(args.validation_blocks)],
                                   "test": [f"t{n+1}" for n in range(args.test_blocks)]},
              "initial_program": read_json(args.initial_program), "hot_profile": read_json(args.hot_profile),
              "deadline": args.deadline, "job_timeout": args.job_timeout}
    if args.manual_sweep is not None:
        from .manual_selection import pin_selection

        config.update(pin_selection(args.manual_sweep_config, args.manual_sweep), schema=3)
    SGLangDomain(config, args.output.resolve().parent)
    save(args.output, config)


if __name__ == "__main__":
    main()
