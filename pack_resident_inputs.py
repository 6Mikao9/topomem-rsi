"""Stage the unchanged gated workloads and the exact resident-v2 native evaluator."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


BASE_CALIBRATION_SHA = "110d2809fc19aa21b4d2251798833c3816cb6c9ca48c215aa085b0882f6eccc7"
BASE_CASES_SHA = "53b30f856a050d041e94bd84992f4e41b36bbcac5d280446488ddb5832e5c137"
CAPACITIES = (4096, 6144, 8192)


def _sha(raw):
    return sha256(raw).hexdigest()


def _json(value):
    return (json.dumps(value, sort_keys=True, indent=2, allow_nan=False) + "\n").encode()


def _read(path, expected=None):
    path = Path(path)
    if expected is not None and _sha(path.read_bytes()) != expected:
        raise ValueError("staging archive differs from the precommitted input")
    with ZipFile(path) as archive:
        names = archive.namelist()
        if len(set(names)) != len(names) or any(
            PurePosixPath(name).is_absolute() or PurePosixPath(name).as_posix() != name
            or ".." in PurePosixPath(name).parts or "\\" in name for name in names
        ):
            raise ValueError("unexpected archive member")
        return {name: archive.read(name) for name in names}


def _write(path, files, *, private=False):
    with ZipFile(path, "x") as archive:
        for name, raw in sorted(files.items()):
            member = ZipInfo(name)
            member.create_system = 3
            mode = 0o100755 if name.endswith("/sglang_launch.py") else 0o100644
            if private and name.endswith("/sglang_launch.json"):
                mode = 0o100600
            member.external_attr = mode << 16
            archive.writestr(member, raw, compress_type=ZIP_DEFLATED)


def pack(output, worker, calibration, cases):
    output = Path(output)
    worker = _read(worker)
    source = json.loads(worker["SOURCE_MANIFEST.json"])
    if source != {"files": {name: _sha(raw) for name, raw in worker.items() if name != "SOURCE_MANIFEST.json"}}:
        raise ValueError("calibration must execute the exact manifested production worker source")
    expected_helpers = {"integrations/topology_memory/sglang_resident.py", "integrations/topology_memory/sglang_resident_oracle.py"}
    if not expected_helpers <= source["files"].keys():
        raise ValueError("calibration worker lacks the resident v2 evaluator")
    worker_sha = _sha(worker["SOURCE_MANIFEST.json"])
    original = _read(calibration, BASE_CALIBRATION_SHA)
    stage = {name: raw for name, raw in original.items() if name.startswith("cases/") or name == "run_hot_native_parity.py"}
    if not stage or "run_hot_native_parity.py" not in stage:
        raise ValueError("precommitted native matrix is missing")
    for directory in ("hot-control-src", "profile-tools"):
        stage.update({f"{directory}/{name}": raw for name, raw in worker.items()})
    binding = {
        "schema": 2, "kind": "gated_native_dp2_three_capacity_probes",
        "capacities": list(CAPACITIES), "repeat_orders": [list(CAPACITIES), list(reversed(CAPACITIES))],
        "native_source_manifest_sha256": worker_sha, "profile_tools_source_manifest_sha256": worker_sha,
        "original_calibration_archive_sha256": BASE_CALIBRATION_SHA,
        "case_manifest_sha256": {str(slots): _sha(stage[f"cases/{slots}/manifest.json"]) for slots in CAPACITIES},
        "files": {name: _sha(raw) for name, raw in stage.items()},
    }
    stage["TRANSFER_MANIFEST.json"] = _json(binding)
    formal = _read(cases, BASE_CASES_SHA)
    prior_case_binding = formal.pop("CASE_TRANSFER_MANIFEST.json")
    case_binding = {
        "schema": 1, "kind": "resident_v2_gated_cases",
        "worker_source_manifest_sha256": worker_sha, "parent_case_archive_sha256": BASE_CASES_SHA,
        "parent_case_transfer_manifest_sha256": _sha(prior_case_binding),
        "formal_case_set_sha256": _sha(formal["formal_case_set_manifest.json"]),
        "formal_tokenization_receipt_sha256": _sha(formal["formal_tokenization_receipt.json"]),
        "files": {name: _sha(raw) for name, raw in formal.items()},
    }
    formal["CASE_TRANSFER_MANIFEST.json"] = _json(case_binding)
    output.mkdir(parents=True, exist_ok=False)
    calibration_zip = output / "resident-v2-native-calibration.zip"
    cases_zip = output / "resident-v2-formal-cases.zip"
    _write(calibration_zip, stage, private=True)
    _write(cases_zip, formal)
    receipt = {
        "calibration_zip_sha256": _sha(calibration_zip.read_bytes()),
        "calibration_manifest_sha256": _sha(stage["TRANSFER_MANIFEST.json"]),
        "cases_zip_sha256": _sha(cases_zip.read_bytes()),
        "cases_manifest_sha256": _sha(formal["CASE_TRANSFER_MANIFEST.json"]),
        "worker_source_manifest_sha256": worker_sha,
    }
    with (output / "PACK_BINDING.json").open("xb") as stream:
        stream.write(_json(receipt))
    return receipt


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("output", "worker", "calibration", "cases"):
        parser.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    print(json.dumps(pack(args.output, args.worker, args.calibration, args.cases), sort_keys=True))


if __name__ == "__main__":
    main()
