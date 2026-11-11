"""Archive only named owned experiment jobs; no models, cache, or other jobs."""
import argparse
import hashlib
import json
from pathlib import Path
import re
from zipfile import ZipFile, ZIP_DEFLATED

p = argparse.ArgumentParser()
p.add_argument('--root', type=Path, required=True)
p.add_argument('--groups', type=Path, required=True)
p.add_argument('--output', type=Path, required=True)
a = p.parse_args()
root = a.root.resolve(strict=True)
groups = json.loads(a.groups.read_text())
a.output.mkdir(exist_ok=False)
report = {'schema': 1, 'scope': 'Explicit owned job IDs, complete text evidence, no outside paths', 'groups': {}}
for label, ids in groups.items():
    assert re.fullmatch(r'[a-z0-9-]+', label)
    files = []
    for job_id in ids:
        assert re.fullmatch(r'sg-[a-f0-9]{48}', job_id)
        job = root / 'jobs' / job_id
        assert job.is_dir() and not job.is_symlink(), job
        for f in sorted(job.rglob('*')):
            assert not f.is_symlink(), f
            if not f.is_file():
                continue
            assert f.resolve().is_relative_to(job.resolve()), f
            assert f.suffix in {'.json', '.jsonl', '.log', '.txt'}, f
            files.append(f)
    batches, batch, total = [], [], 0
    for f in files:
        size = f.stat().st_size
        assert size < 45_000_000, f
        if batch and total + size > 40_000_000:
            batches.append(batch)
            batch, total = [], 0
        batch.append(f)
        total += size
    if batch:
        batches.append(batch)
    archives = []
    for index, members in enumerate(batches):
        name = f'{label}-server-{index:02d}.zip'
        manifest = {}
        with ZipFile(a.output / name, 'x', compression=ZIP_DEFLATED, compresslevel=6) as z:
            for f in members:
                rel = f.relative_to(root / 'jobs').as_posix()
                raw = f.read_bytes()
                manifest[rel] = {'sha256': hashlib.sha256(raw).hexdigest(), 'bytes': len(raw)}
                z.writestr(rel, raw)
            z.writestr('EVIDENCE_MANIFEST.json', json.dumps(manifest, sort_keys=True, indent=2)+'\n')
        archive = a.output / name
        archives.append({'name': name, 'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(), 'bytes': archive.stat().st_size,
                         'file_count': len(manifest), 'uncompressed_bytes': sum(x['bytes'] for x in manifest.values())})
    report['groups'][label] = {'job_ids': ids, 'archives': archives}
out = a.output / 'evidence-archive-index.json'
out.write_text(json.dumps(report, sort_keys=True, indent=2)+'\n')
print(json.dumps({k: {'jobs': len(v['job_ids']), 'archives': len(v['archives']), 'bytes': sum(x['bytes'] for x in v['archives'])}
                  for k, v in report['groups'].items()}), flush=True)
