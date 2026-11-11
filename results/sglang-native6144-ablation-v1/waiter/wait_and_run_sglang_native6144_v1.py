"""Run the preregistered supplement after terminal formal execution, regardless of scores."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys
import time

base = Path(r'D:\ws\.scratch')
formal = base / 'topomem-sglang-rsi-v3'
control = base / 'topomem-native6144-poststudy-control-v1'
output = base / 'topomem-native6144-ablation-v1'
source = base / 'topomem-sglang-contract-v3-draft'
release = base / 'topomem-sglang-contract-release-v3'
driver = base / 'sglang_post_freeze_native6144.py'
prereg = base / 'sglang_post_freeze_native6144.preregistration.json'
driver_sha = '9a4eef769bc864ff45f123c4cbf7cfcb0a81881ca222abbf26ff412802247994'
prereg_sha = '47fb544e2c326d388d221101d652fb59073af1899ff69c949ffdf8bc07e7442d'
assert sha256(driver.read_bytes()).hexdigest() == driver_sha
assert sha256(prereg.read_bytes()).hexdigest() == prereg_sha
assert not output.exists()
control.mkdir(exist_ok=False)
plan = {'schema': 1, 'created_utc': datetime.now(timezone.utc).isoformat(),
        'condition': 'Only formal phase completion/exit status and preregistered UTC budget determine launch; no score condition.',
        'driver_sha256': driver_sha, 'preregistration_sha256': prereg_sha,
        'read_formal_report_only_after_all_phases_exit_zero': True,
        'latest_start_utc': '2026-11-11T00:00:00Z',
        'no_new_submissions_at_or_after_utc': '2026-11-11T00:20:00Z'}
(control/'wait-plan.json').write_text(json.dumps(plan, indent=2)+'\n', encoding='utf-8')
command = [sys.executable, '-B', str(driver), '--preregistration', str(prereg),
           '--expected-prereg-sha256', prereg_sha, '--source-root', str(source),
           '--source-manifest', str(release/'contract-v3-rc.source-manifest.json'),
           '--source-archive', str(release/'contract-v3-rc.source.zip')]
preflight = subprocess.run(command, check=True, text=True, encoding='utf-8', capture_output=True)
(control/'source-preflight.log').write_text(preflight.stdout, encoding='utf-8')
print(preflight.stdout, end='', flush=True)
cutoff = datetime(2026, 11, 11, 0, 0, tzinfo=timezone.utc)
last = None
while True:
    now = datetime.now(timezone.utc)
    status_file = formal/'pipeline-status.json'
    statuses = json.loads(status_file.read_text(encoding='utf-8')) if status_file.exists() else []
    state = [(row['phase'], row['returncode']) for row in statuses]
    if state != last:
        print(json.dumps({'utc': now.isoformat(), 'formal_phases': state, 'state': 'waiting'}), flush=True)
        last = state
    terminal = len(statuses) == 3 or any(row['returncode'] != 0 for row in statuses)
    if terminal or now > cutoff:
        break
    time.sleep(10)

command += ['--execute', '--formal-root', str(formal), '--private-adapter-config', str(formal/'schema3.json'),
            '--output', str(output)]
binding = {'utc': datetime.now(timezone.utc).isoformat(), 'formal_phases': state}
if len(statuses) == 3 and all(row['returncode'] == 0 for row in statuses) and datetime.now(timezone.utc) <= cutoff:
    report_sha = sha256((formal/'controller/report.json').read_bytes()).hexdigest()
    command += ['--expected-formal-report-sha256', report_sha]
    binding['formal_report_sha256'] = report_sha
(control/'launch-binding.json').write_text(json.dumps(binding, indent=2)+'\n', encoding='utf-8')
print(json.dumps({'state': 'launching_preregistered_driver', **binding}), flush=True)
with (control/'driver.log').open('x', encoding='utf-8') as log:
    result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
completion = {'returncode': result.returncode, 'finished_utc': datetime.now(timezone.utc).isoformat()}
if (output/'result.json').is_file():
    artifact = json.loads((output/'result.json').read_text(encoding='utf-8'))
    completion.update(status=artifact['status'], timed_jobs_completed=artifact['timed_jobs_completed'])
(control/'completion.json').write_text(json.dumps(completion, indent=2)+'\n', encoding='utf-8')
print(json.dumps(completion), flush=True)
raise SystemExit(result.returncode)
