from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
from zipfile import ZipFile

root = Path('/home/liuyuntao/experiments/rsi-topology-20261110')
archive = root / 'topomem-worker-durability-tests-20261111.zip'
expected = 'a20ee820831934ec1d59bb02f8c782bd0219b24a2771fbcd4fcd31a3a2b09253'
assert sha256(archive.read_bytes()).hexdigest() == expected
destination = root / 'worker-durability-tests-20261111'
destination.mkdir(exist_ok=False)
with ZipFile(archive) as bundle:
    manifest = json.loads(bundle.read('TEST_SOURCE_MANIFEST.json'))
    assert set(bundle.namelist()) == set(manifest) | {'TEST_SOURCE_MANIFEST.json'}
    for name, digest in manifest.items():
        target = destination / name
        assert target.resolve().is_relative_to(destination.resolve())
        raw = bundle.read(name)
        assert sha256(raw).hexdigest() == digest
        target.parent.mkdir(exist_ok=True, parents=True)
        target.write_bytes(raw)
command = [sys.executable, '-B', '-m', 'unittest', 'discover', '-s', 'tests', '-p', 'test_execution_protocol.py', '-v']
run = subprocess.run(command, cwd=destination, capture_output=True)
log = run.stdout + run.stderr
(destination / 'tests.log').write_bytes(log)
report = {'schema': 1, 'utc': datetime.now(timezone.utc).isoformat(),
          'scope': 'Human infrastructure fix. CPU unit/fault-injection tests on POSIX; no real GPU jobs, restart or power-loss simulation.',
          'archive_sha256': expected, 'source_sha256': manifest, 'command': command,
          'returncode': run.returncode, 'log_sha256': sha256(log).hexdigest(),
          'python': sys.version, 'platform': sys.platform}
(destination / 'test-report.json').write_text(json.dumps(report, indent=2)+'\n')
print(log.decode(errors='replace'))
print(json.dumps(report))
raise SystemExit(run.returncode)
