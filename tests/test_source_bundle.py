from hashlib import sha256
import json
from pathlib import Path
import tempfile
import unittest
from zipfile import ZipFile

from rsi_memory.source_bundle import bundle


class SourceBundleTests(unittest.TestCase):
    def test_reproducible_manifest_excludes_operator_files_and_retains_existing_output(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            members = {"rsi_memory/execution_worker.py": b"# trusted worker\n",
                       "integrations/topology_memory/torch_eval.py": b"# trusted evaluator\n",
                       "examples/workloads/task.json": b"{}\n"}
            excluded = {"private-executor.json": b"operator config", "results/key.txt": b"secret",
                        "rsi_memory/.env": b"private token", "examples/notes.md": b"local notes"}
            for name, raw in {**members, **excluded}.items():
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(raw)
            first = bundle(root, root / "a.zip")
            second = bundle(root, root / "b.zip")
            self.assertEqual(first["archive_sha256"], second["archive_sha256"])
            with ZipFile(root / "a.zip") as archive:
                manifest = json.loads(archive.read("SOURCE_MANIFEST.json"))
                self.assertEqual(set(archive.namelist()), {*members, "SOURCE_MANIFEST.json"})
                self.assertEqual(manifest["files"], {name: sha256(raw).hexdigest() for name, raw in members.items()})
                self.assertEqual(first["source_manifest_sha256"], sha256(archive.read("SOURCE_MANIFEST.json")).hexdigest())
            with self.assertRaises(FileExistsError):
                bundle(root, root / "a.zip")
            self.assertEqual(sha256((root / "a.zip").read_bytes()).hexdigest(), first["archive_sha256"])
