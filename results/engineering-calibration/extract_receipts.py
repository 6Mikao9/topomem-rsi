"""Verify evidence inventories and extract exact ZIP member bytes to a new directory."""

import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile


def relative(name):
    path = PurePosixPath(name)
    if not path.parts or path.is_absolute() or path.as_posix() != name or ".." in path.parts or "\\" in name or ":" in name:
        raise ValueError("invalid evidence path " + repr(name))
    return path


def checked(raw, entry, name):
    if len(raw) != entry["bytes"] or sha256(raw).hexdigest() != entry["sha256"]:
        raise ValueError("evidence byte identity differs for " + name)
    return raw


def member_path(root, name):
    path = root / relative(name)
    if path.is_symlink() or not path.resolve(strict=True).is_relative_to(root):
        raise ValueError("evidence file escapes its manifest directory: " + name)
    return path


def verify(manifest, *, output=None, archive_names=()):
    data = json.loads(manifest.read_bytes())
    if data.get("schema") != 1 or data.get("kind") != "byte_preserving_receipts":
        raise ValueError("unexpected evidence inventory")
    root = manifest.resolve(strict=True).parent
    for name, info in data["copied_originals"].items():
        checked(member_path(root, name).read_bytes(), info, name)
    recipe = data.get("derivations", {}).get("v10-replay")
    if recipe is not None:
        name = recipe["public_parameterized_recipe"]
        if sha256(member_path(root, name).read_bytes()).hexdigest() != recipe["public_parameterized_recipe_sha256"]:
            raise ValueError("derived engineering replay recipe differs from its inventory")
    if "viewer_manifest" in data:
        entry = data["viewer_manifest"]
        views = json.loads(checked(member_path(root, entry["file"]).read_bytes(), entry, entry["file"]))
        for view in views["scenarios"].values():
            for member in view["outputs"].values():
                checked(member_path(root, member["file"]).read_bytes(), member, member["file"])
    labels = tuple(archive_names) if archive_names else tuple(data["archives"])
    if not labels or len(labels) != len(set(labels)) or set(labels) - data["archives"].keys():
        raise ValueError("unexpected or repeated evidence archive label")
    if any(len(relative(label).parts) != 1 for label in labels):
        raise ValueError("invalid evidence archive label")
    if output is not None:
        output.mkdir(parents=True, exist_ok=False)
    counts = {}
    for label in labels:
        archive = data["archives"][label]
        path = member_path(root, archive["file"])
        checked(path.read_bytes(), archive, archive["file"])
        expected = archive["members"]
        with ZipFile(path) as z:
            entries = z.infolist()
            if len(entries) != len(expected) or {entry.filename for entry in entries} != set(expected):
                raise ValueError("archive has missing, duplicate or unexpected members: " + path.name)
            for entry in entries:
                name = entry.filename
                rel = relative(name)
                info = expected[name]
                if entry.is_dir() or entry.file_size != info["bytes"]:
                    raise ValueError("invalid member geometry: " + name)
                raw = checked(z.read(entry), info, name)
                if output is not None:
                    target = output / label / rel
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with target.open("xb") as stream:
                        stream.write(raw)
                    if target.read_bytes() != raw:
                        raise ValueError("extracted member differs from its verified archive bytes")
        counts[label] = len(expected)
    return counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, help="new directory for byte-exact extraction")
    parser.add_argument("--archive", action="append", default=[], help="optional inventory label")
    args = parser.parse_args()
    print(json.dumps({"verified_members": verify(args.manifest, output=args.output, archive_names=args.archive)}, sort_keys=True))


if __name__ == "__main__":
    main()
