"""Run the frozen additive v4 engineering study without retrying effects."""
from datetime import datetime, timezone
from hashlib import sha256
import json
from pathlib import Path
import subprocess
import sys

ROOT=Path(r'D:\ws\.scratch\topomem-v41-controller-source')
OUT=Path(r'D:\ws\.scratch\topomem-v41-cuda-engineering')
def digest(path):
    return sha256(path.read_bytes()).hexdigest()
def save(path,value):
    with path.open('x',encoding='utf-8',newline='\n') as stream:
        json.dump(value,stream,sort_keys=True,indent=2)
        stream.write('\n')

assert not (OUT/'controller').exists(), 'fresh controller required; inspect any existing journal'
manifest=json.loads((ROOT/'SOURCE_MANIFEST.json').read_bytes())
for name,expected in manifest['files'].items():
    assert digest(ROOT/name)==expected, 'frozen controller source differs'
config=OUT/'private-config.json'
case_plan=json.loads((OUT/'case-plan.json').read_bytes())
assert digest(config)==case_plan['private_config_sha256']
for name in case_plan['qualification_order']:
    assert json.loads((OUT/(name+'-qualification.json')).read_bytes())['result']['feasible'] is True
plan={'schema':1,'scope':case_plan['scope'],'stages':['run','freeze','test'],
      'created_utc':datetime.now(timezone.utc).isoformat(),
      'source_manifest_sha256':digest(ROOT/'SOURCE_MANIFEST.json'),
      'private_config_sha256':digest(config),'case_plan_sha256':digest(OUT/'case-plan.json'),
      'script_sha256':digest(Path(__file__)),
      'semantics':'Stop on any failed phase. Unknown external effects are not retried. No partial automatic freeze.'}
save(OUT/'pipeline-plan.json',plan)
statuses=[]
for phase in plan['stages']:
    command=[sys.executable,'-B','-m','rsi_memory.cuda_v4',phase,'--adapter-config',str(config),
             '--output',str(OUT/'controller')]
    if phase=='run':
        command+=['--key-file',r'D:\ws\05_归档\凭据\api-key.txt']
    start=datetime.now(timezone.utc).isoformat()
    print(json.dumps({'phase':phase,'status':'started','utc':start}),flush=True)
    with (OUT/('pipeline-'+phase+'.log')).open('x',encoding='utf-8') as log:
        process=subprocess.Popen(command,cwd=ROOT,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                                 text=True,encoding='utf-8',errors='replace',bufsize=1)
        for line in process.stdout:
            log.write(line);log.flush();print(line,end='',flush=True)
        code=process.wait()
    statuses.append({'phase':phase,'returncode':code,'started_utc':start,
                     'finished_utc':datetime.now(timezone.utc).isoformat()})
    temporary=OUT/'pipeline-status.pending'
    temporary.write_text(json.dumps(statuses,indent=2)+'\n',encoding='utf-8')
    temporary.replace(OUT/'pipeline-status.json')
    print(json.dumps(statuses[-1]),flush=True)
    if code:
        raise SystemExit(code)
