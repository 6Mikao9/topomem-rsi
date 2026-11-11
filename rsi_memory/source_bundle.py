"""Create a reproducible, source-only deployment archive for the CUDA worker."""

import argparse
from hashlib import sha256
import json
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile, ZipInfo


def bundle(root, output):
    root, output = Path(root).resolve(strict=True), Path(output)
    files = {}
    for directory, suffix in (("rsi_memory", ".py"), ("integrations", ".py"), ("examples", ".json")):
        source = root / directory
        if source.is_symlink() or not source.is_dir():
            raise ValueError("source packages and examples must be regular directories")
        for path in sorted(source.rglob("*")):
            if path.is_symlink():
                raise ValueError("deployment sources cannot contain symlinks")
            if not path.is_file() or path.suffix != suffix:
                continue
            name = path.relative_to(root).as_posix()
            files[name] = path.read_bytes()
    if "rsi_memory/execution_worker.py" not in files or "integrations/topology_memory/torch_eval.py" not in files:
        raise ValueError("source root lacks the trusted CUDA worker and evaluator")
    manifest = (json.dumps({"files": {name: sha256(raw).hexdigest() for name, raw in sorted(files.items())}},
                           sort_keys=True, indent=2) + "\n").encode("utf-8")
    files["SOURCE_MANIFEST.json"] = manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with ZipFile(output, "x") as archive:
        for name, raw in sorted(files.items()):
            member = ZipInfo(name)
            member.create_system = 3
            member.external_attr = 0o100644 << 16
            archive.writestr(member, raw, compress_type=ZIP_DEFLATED)
    return {"archive": str(output.resolve()), "archive_sha256": sha256(output.read_bytes()).hexdigest(),
            "source_manifest_sha256": sha256(manifest).hexdigest(), "files": len(files)-1,
            "bytes": output.stat().st_size}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(bundle(args.root, args.output), sort_keys=True))


if __name__ == "__main__":
    main()
