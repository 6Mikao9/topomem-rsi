"""Verify a public result inventory and optionally restore local viewer receipts."""
import argparse
from hashlib import sha256
import json
from pathlib import Path, PurePosixPath
import re
from zipfile import ZipFile

RECEIPT = re.compile(r"receipts/sg-[0-9a-f]{48}\.json")


def file_hash(path):
    with path.open("rb") as stream:
        return sha256_stream(stream)


def sha256_stream(stream):
    value = sha256()
    for chunk in iter(lambda: stream.read(1024*1024),b""):
        value.update(chunk)
    return value.hexdigest()


def safe_path(root,name):
    relative=PurePosixPath(name)
    if relative.is_absolute() or ".." in relative.parts or "\\" in name or ":" in name:
        raise ValueError("unsafe artifact path "+name)
    path=root/relative
    if not path.resolve().is_relative_to(root.resolve()):
        raise ValueError("artifact path leaves the result directory")
    return path


def verify(root,restore=False):
    manifest=json.loads((root/"artifact_manifest.json").read_bytes())
    files=manifest["files_sha256"]
    for name,expected in files.items():
        path=safe_path(root,name)
        if path.is_symlink() or file_hash(path)!=expected:
            raise ValueError("artifact changed "+name)
    members=manifest.get("receipt_member_sha256",manifest.get("leased_receipt_member_sha256"))
    if members is not None:
        with ZipFile(root/"leased-receipts.zip") as package:
            names=package.namelist()
            if len(names)!=len(set(names)) or set(names)!=set(members):
                raise ValueError("receipt roster differs")
            for name,expected in members.items():
                if not RECEIPT.fullmatch(name):
                    raise ValueError("unexpected receipt member "+name)
                raw=package.read(name)
                if sha256(raw).hexdigest()!=expected:
                    raise ValueError("receipt member changed "+name)
                path=safe_path(root,name)
                if path.exists():
                    if path.is_symlink() or path.read_bytes()!=raw:
                        raise ValueError("local receipt differs "+name)
                elif restore:
                    path.parent.mkdir(exist_ok=True)
                    with path.open("xb") as stream:
                        stream.write(raw)
    return {"files":len(files),"receipt_members":len(members or {}),"materialized":restore}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root",type=Path)
    parser.add_argument("--materialize-receipts",action="store_true")
    args=parser.parse_args()
    print(json.dumps(verify(args.root,args.materialize_receipts),sort_keys=True))


if __name__=="__main__":
    main()
