"""Build a standalone viewer of measured v4 iterations, without model calls."""

import argparse
from hashlib import sha256
import json
from math import isfinite
from pathlib import Path
import re
from statistics import median

from .topology import fingerprint


OUTPUTS = ('manager_best', 'admitted_output', 'fixed')
VERSIONS = {'training_paired_declared_identity_v4': '4', 'training_paired_declared_identity_v41': '4.1'}


def _number(value):
    try:
        return type(value) in (int, float) and isfinite(value)
    except OverflowError:
        return False


def _checked_test(report, manifest, threshold):
    frozen, rounds = report.get('frozen'), report['rounds']
    if report.get('status') != 'test_settled' or not isinstance(frozen, dict):
        raise ValueError('final viewer requires a frozen, test_settled report')
    if (not rounds or frozen.get('partial') is not False
            or frozen.get('rounds') != len(rounds) or len(rounds) != manifest['config']['rounds']
            or frozen.get('manifest_sha256') != fingerprint(manifest)
            or frozen.get('rounds_sha256') != fingerprint(rounds)
            or frozen.get('training_matrix_sha256') != fingerprint(rounds[-1]['v4_training'])):
        raise ValueError('frozen report does not match the completed training archive')
    contract = manifest['contract']
    block_ids = contract['paired_block_ids']['test']
    if (type(block_ids) is not list or not block_ids
            or any(type(block) is not str or not block for block in block_ids)
            or len(set(block_ids)) != len(block_ids)):
        raise ValueError('invalid predeclared test blocks')
    programs = {'native': None}
    last = rounds[-1]
    admitted = {manifest['seed_sha256'], *(row['active_after'] for row in rounds if row['decision'] == 'admitted')}
    if (frozen.get('active_harness_sha256') != last['active_after']
            or frozen.get('admitted_harness_sha256s') != sorted(admitted)):
        raise ValueError('frozen generator lineage differs from the completed rounds')
    for name in OUTPUTS:
        output = frozen.get(name, {})
        trial, choice = output.get('trial', {}), output.get('selection', {})
        program = trial.get('program')
        if (not isinstance(program, dict) or trial.get('program_sha256') != fingerprint(program)
                or choice != last['v4_selection'].get(name) or choice.get('proposal_id') != trial.get('id')
                or trial != last['v4_training']['cohort_proposals'].get(trial.get('id'))
                or choice.get('source_sha256') != trial.get('harness_sha256')):
            raise ValueError(f'frozen output {name} differs from its selected program or source')
        if ((name == 'admitted_output' and choice['source_sha256'] not in admitted)
                or (name == 'fixed' and choice['source_sha256'] != manifest['seed_sha256'])):
            raise ValueError(f'frozen output {name} has an ineligible generator source')
        programs[name] = program
    if manifest.get('manual_program') is not None:
        programs['manual'] = manifest['manual_program']
    pair = report.get('paired_evaluations', {}).get('pair/frozen/test', {})
    if pair.get('status') != 'completed':
        raise ValueError('frozen test pair did not complete')
    if pair.get('input') != {'split': 'test', 'block_ids': block_ids, 'programs': programs}:
        raise ValueError('test input differs from frozen programs or predeclared test blocks')
    result = pair.get('result', {})
    if not isinstance(result, dict) or result.keys() != programs.keys():
        raise ValueError('test results differ from the declared test arms')
    cases = None
    for name, arm in result.items():
        if not isinstance(arm, dict):
            raise ValueError(f'test arm {name} has an invalid result')
        samples = arm.get('samples')
        if (arm.get('feasible') is not True or type(samples) is not list
                or [x.get('block_id') if isinstance(x, dict) else None for x in samples] != block_ids):
            raise ValueError(f'test arm {name} lacks feasible samples for the predeclared blocks')
        for sample in samples:
            scores = sample.get('case_scores_ms')
            if (not _number(sample.get('score')) or not isinstance(scores, dict) or not scores
                    or any(not key or not _number(value) for key, value in scores.items())):
                raise ValueError(f'test arm {name} has invalid scores')
            if cases is None:
                cases = set(scores)
            if scores.keys() != cases:
                raise ValueError('test samples disagree on the case set')
    if len(cases) != len(contract['split_case_sha256']['test']):
        raise ValueError('test samples omit or add cases relative to the predeclared split')

    comparisons, case_deltas, relations = {}, [], {}
    references = ('native', 'fixed', 'manual') if 'manual' in programs else ('native', 'fixed')
    for name in OUTPUTS[:2]:
        for reference in references:
            key = f'{name}_vs_{reference}'
            samples = list(zip(result[name]['samples'], result[reference]['samples'], strict=True))
            deltas = [b['score'] - a['score'] for a, b in samples]
            if not all(_number(delta) for delta in deltas):
                raise ValueError('non-finite paired test differences')
            improvement = median(deltas)
            if not _number(improvement):
                raise ValueError('non-finite paired test median')
            passed = all(delta > 0 for delta in deltas) and improvement > threshold
            comparisons[key] = {'passed': passed,
                                'reason': 'stable_gain' if passed else 'insufficient_stable_gain',
                                'median_improvement': improvement,
                                'deltas': [{'block_id': block, 'reference_minus_candidate': delta}
                                           for block, delta in zip(block_ids, deltas, strict=True)]}
            relations[key] = programs[reference] is not None and fingerprint(programs[name]) == fingerprint(programs[reference])
            for case in sorted(cases):
                values = [b['case_scores_ms'][case] - a['case_scores_ms'][case] for a, b in samples]
                case_median = median(values)
                if not all(_number(value) for value in values) or not _number(case_median):
                    raise ValueError('non-finite paired case differences')
                case_deltas.append({'candidate': name, 'reference': reference, 'case': case,
                                    'wins': sum(value > 0 for value in values), 'blocks': len(values),
                                    'median_ms': case_median, 'deltas_ms': values})
    if comparisons != report.get('test_comparisons'):
        raise ValueError('test comparison summaries disagree with paired samples')
    recorded = report.get('test_program_relations') or {}
    if set(recorded) != set(relations) or any(recorded[key].get('same_raw_program') is not same for key, same in relations.items()):
        raise ValueError('recorded IR relationships disagree with the frozen test programs')
    return comparisons, case_deltas, relations


def bundle(report_path, harness_directory):
    raw = Path(report_path).read_bytes()
    report = json.loads(raw)
    manifest = report['manifest']
    protocol = manifest['contract']['selection_protocol']
    version = VERSIONS.get(protocol['kind'])
    if version is None or protocol.get('version', version) != version:
        raise ValueError('require an explicit v4 or v4.1 selection report')
    threshold = manifest['config'].get('min_improvement')
    if not _number(threshold):
        raise ValueError('minimum improvement must be a finite number')
    comparisons, case_deltas, relations = _checked_test(report, manifest, threshold)
    source_ids = {manifest['seed_sha256']}
    rounds = []
    for row in report['rounds']:
        source_ids.update(x for x in (row['active_before'], row['active_after'], row['proposed_child_sha256']) if x)
        matrix = row['v4_training']
        trials = {trial['id']: trial for trial in row['parent'] + row['child'] + row['fixed']}
        trials.update(matrix['cohort_proposals'])
        selected = {}
        for name, choice in row['v4_selection'].items():
            trial = trials.get(choice['proposal_id'])
            if choice['source_sha256'] is not None:
                if trial is None or trial.get('harness_sha256') != choice['source_sha256']:
                    raise ValueError(f'training selection {name} differs from its source trial')
                source_ids.add(choice['source_sha256'])
            measured = matrix['measurements'].get(choice['effect_sha256'], {})
            selected[name] = {**choice, 'median_ms': measured.get('median_ms'),
                              'block_ms': [x['window_ms'] for x in measured.get('evaluations', []) if 'window_ms' in x],
                              'program': trial['program'] if trial else None}
        rounds.append({k: row.get(k) for k in ('round', 'decision', 'reason', 'active_before', 'active_after',
                       'proposed_child_sha256', 'adaptive_meta_plan', 'controlled_behavior', 'paired_validation',
                       'previous_admission_reuse')} | {'selection': selected, 'training_plan': matrix['plan']})
    sources = {}
    if any(type(source_id) is not str or not re.fullmatch('[0-9a-f]{64}', source_id) for source_id in source_ids):
        raise ValueError('generator source ID must be a lowercase SHA-256 digest')
    root = Path(harness_directory).resolve()
    for source_id in sorted(source_ids):
        path = root / (source_id + '.py')
        if path.resolve().parent != root:
            raise ValueError('generator source is outside the harness directory')
        value = path.read_bytes()
        if sha256(value).hexdigest() != source_id:
            raise ValueError('generator source digest differs')
        sources[source_id] = value.decode('utf-8')
    return {'schema': 1, 'version': version, 'status': report['status'], 'report_sha256': sha256(raw).hexdigest(),
            'manifest': manifest, 'rounds': rounds, 'sources': sources, 'usage': report['usage'],
            'test_comparisons': comparisons, 'case_deltas': case_deltas,
            'frozen': report['frozen'], 'relations': relations}


HTML = r'''<!doctype html>
<html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>TopoMem RSI · v4 迭代证据</title>
<style>
:root{color-scheme:light;--ink:#172c37;--muted:#586b76;--green:#006f58;--red:#a9333b;--line:#d4dfe3;--card:#fff}
*{box-sizing:border-box}body{margin:0;background:#f2f5f4;color:var(--ink);font:15px/1.6 system-ui,"Microsoft Yahei",sans-serif}
main{max-width:1240px;margin:auto;padding:36px 28px}h1{font-size:34px;line-height:1.25;margin:12px 0}h2{font-size:22px;margin:8px 0 20px}
h3{font-size:16px;margin:8px 0}p{margin:10px 0}.eyebrow{font-size:12px;letter-spacing:.15em;color:var(--green);font-weight:750}
.muted,small{color:var(--muted)}.notice{background:#e6eeeb;border-left:4px solid var(--green);padding:16px 20px;margin:22px 0}
.cards{display:grid;grid-template-columns:repeat(3,1fr);gap:14px}.card,section{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:22px;margin-bottom:18px;min-width:0}
.number{display:block;font-size:32px;font-weight:700}.tabs{display:flex;gap:8px;flex-wrap:wrap;margin:24px 0 16px}button,select{font:inherit;border:1px solid var(--line);border-radius:7px;background:white;color:var(--ink);padding:9px 15px;cursor:pointer}
button[aria-selected=true]{background:var(--ink);color:white}button:focus-visible,select:focus-visible{outline:3px solid #3c9b81;outline-offset:3px}
.row{display:flex;gap:15px;align-items:center;flex-wrap:wrap}.grow{flex:1}.tag{font-size:12px;border-radius:20px;background:#edf1f2;padding:4px 10px}.ok{color:var(--green)}.bad{color:var(--red)}
.scroll{max-width:100%;overflow:auto}table{border-collapse:collapse;width:100%;white-space:nowrap;font-size:14px}th,td{text-align:left;padding:11px 14px;border-bottom:1px solid var(--line)}th{color:var(--muted);font-weight:600}
pre{white-space:pre-wrap;overflow-wrap:anywhere;max-height:560px;overflow:auto;background:#f1f4f5;padding:16px;border-radius:7px;font:12px/1.65 ui-monospace,Consolas,monospace}
code{overflow-wrap:anywhere}.two{display:grid;grid-template-columns:1fr 1fr;gap:18px}.two>*{min-width:0}.plan{overflow-wrap:anywhere}.plan p{padding-bottom:10px;border-bottom:1px solid var(--line)}details{margin:16px 0}summary{cursor:pointer;font-weight:650}.hidden{display:none!important}
a{color:var(--green)}footer{font-size:12px;color:var(--muted);overflow-wrap:anywhere;padding-bottom:25px}
@media(max-width:650px){main{padding:22px 16px}h1{font-size:27px}.cards,.two{grid-template-columns:1fr}.card,section{padding:17px}.cards{gap:0}.number{font-size:26px}.row .grow{flex-basis:100%}}
</style>
<main>
<div class="eyebrow" id="version"></div>
<h1>把内存方案与自我改进分开验收</h1>
<p class="muted">固定稀疏注意力 × 本次冻结硬件拓扑 × Agent / Chatbot / 长文 / Rollout</p>
<div class="notice">这是已公开语料在本次冻结配置下的工程复评。GPU 评估使用确定性 KV/Q 夹具，不测真实模型质量、LongBench 分数或服务吞吐。调度、批次、计算位置及显存预算均为固定输入。物理 GPU 和计算映射以本次运行的原始配置及回执为准。判断是否实际使用多卡应查阅逐设备占用与链路回执。</div>
<div class="cards" id="cards"></div>
<nav class="tabs" aria-label="证据视图"><button data-tab="rounds" aria-selected="true">逐轮自我改进</button><button data-tab="test" aria-selected="false">冻结后的测试</button><button data-tab="protocol" aria-selected="false">边界与证据</button></nav>
<div id="rounds"><section><div class="row"><h2 class="grow">先提出改进，再测量效果</h2><label>查看轮次 <select id="round-select" aria-label="查看轮次"></select></label></div><div id="round-body"></div></section></div>
<div id="test" class="hidden"><section><h2>三种产物各自保留来源</h2><p>Manager best 是训练选出的内存程序，可能来自未晋级的生成器。Admitted output 由初始或已晋级的生成器实际产出。Fixed 使用始终固定的初始生成器。生成器的晋级和实际复用证据见逐轮记录。</p><div id="final-table"></div><h3>逐场景比较</h3><p class="muted">正值表示候选更快。两臂原始 IR 相同时，仍保留原始计时和预设门槛裁决，计时差无法归因于新的内存程序。</p><div id="case-table"></div><details><summary>冻结的程序与来源</summary><pre id="frozen"></pre></details></section></div>
<div id="protocol" class="hidden"><section><h2>哪些是 RSI，哪些是工程改动</h2><p>v4 的重复测量选择器、三条产物线及槽内反馈是人写的控制器升级，不计为模型递归收益。模型必须实际修改生成与元改进函数，通过配对验证后才晋级；后续真实调用继续使用新源码，才形成可检查的递归链。</p><p>当前自动化覆盖显存管理 IR 的生成，以及任务生成和元改进入口的受限自改。完整框架、执行后端与语义知识库的自主迭代尚未实现。v13 后端优化归于人工工程。</p><p>训练矩阵使用独立作业的回执。已有最佳内存程序与本轮父/子生成器的新产物分开比较，避免它们共同继承同一程序而使比较失去意义。证据列表仅重新排序不能算结构改进。未知外部执行状态停止结算，不自动重发。</p><p>CUDA 身份目前使用精确的规范化 IR 摘要；两个不同 IR 即使恰好执行相同动作，也不会被冒认为已证明物理等价。上游旧 manifest 的 <code>task_output_protocol</code> 是兼容字段，本次权威协议在 <code>contract.selection_protocol</code>。</p><pre id="protocol-json"></pre></section></div>
<footer id="provenance"></footer></main>
<script type="application/json" id="evidence">__DATA__</script>
<script>
'use strict';const D=JSON.parse(document.getElementById('evidence').textContent);
const $=id=>document.getElementById(id),esc=x=>String(x??'—').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
const num=x=>Number.isFinite(x)?x.toFixed(3):'—', short=x=>x?x.slice(0,12):'—',pretty=x=>JSON.stringify(x,null,2);
$('version').textContent='TOPOMEM / RECURSIVE SELF-IMPROVEMENT / V'+D.version;
const names={manager_best:'Manager best',admitted_output:'Admitted output',fixed:'Fixed',parent:'本轮父生成器',child:'本轮子生成器',native:'原生',manual:'手写'};
function table(headers,rows){return '<div class="scroll"><table><thead><tr>'+headers.map(x=>'<th>'+esc(x)+'</th>').join('')+'</tr></thead><tbody>'+rows.map(row=>'<tr>'+row.map(x=>'<td>'+esc(x)+'</td>').join('')+'</tr>').join('')+'</tbody></table></div>'}
const admissions=D.rounds.filter(r=>r.decision==='admitted').length,reuses=D.rounds.filter(r=>r.previous_admission_reuse?.recursive_behavior_observed).length;
$('cards').innerHTML=[['已结算迭代',D.rounds.length+' 轮','完整记录提案、测量与去留'],['生成器晋级',admissions+' 次','通过本轮父代及已有方案的配对门槛'],['后续递归复用',reuses+' 轮','新源码参与真实任务与元改进调用']].map(x=>'<div class="card"><span class="muted">'+x[0]+'</span><span class="number">'+x[1]+'</span><small>'+x[2]+'</small></div>').join('');
for(const r of D.rounds){const o=document.createElement('option');o.value=r.round;o.textContent='R'+r.round+' · '+(r.decision==='admitted'?'晋级':'未晋级');$('round-select').append(o)}
function renderRound(){const r=D.rounds.find(x=>x.round===Number($('round-select').value));if(!r){$('round-body').textContent='尚无已结算轮次。';return}
const plan=r.adaptive_meta_plan||{},cmp=r.paired_validation?.comparisons||{};
let html='<div class="row"><span class="tag '+(r.decision==='admitted'?'ok':'bad')+'">'+(r.decision==='admitted'?'生成器晋级':'生成器未晋级')+'</span><code>'+esc(r.reason)+'</code></div><p class="muted">活动源码 '+esc(short(r.active_before))+' → '+esc(short(r.active_after))+'</p>';
html+='<div class="two"><div class="plan"><h3>评估前登记的改进方案</h3>'+[['observed_problem','观察到的问题'],['mechanism_change','拟改变的机制'],['expected_effect','预期效果'],['falsification','推翻条件']].map(([key,label])=>'<p><b>'+label+'</b><br>'+esc(plan[key])+'</p>').join('')+'</div><div><h3>验证效果</h3>'+table(['比较','配对中位改善 ms','获胜区块','通过'],Object.entries(cmp).map(([k,v])=>[k,num(v.median_improvement),v.deltas.length?v.deltas.filter(x=>x.reference_minus_candidate>0).length+'/'+v.deltas.length:'—',v.deltas.length?(v.passed?'是':'否'):'未测得可行配对']))+'<p class="muted">正值表示左侧候选更快。需要所有区块为正且中位改善超过 '+esc(D.manifest.config.min_improvement)+' ms。</p></div></div>';
html+='<h3>用重复训练测量选择产物</h3>'+table(['产物线','原始提案','训练区块 ms','训练中位 ms','生成器源码'],Object.entries(r.selection).map(([k,v])=>[names[k]||k,v.proposal_id,v.block_ms.map(num).join(' / '),num(v.median_ms),short(v.source_sha256)]));
html+='<details><summary>轮首相同输入下的结构决策重演</summary><pre>'+esc(pretty(r.controlled_behavior))+'</pre></details><details><summary>本轮对上一轮晋级源码的实际复用</summary><pre>'+esc(pretty(r.previous_admission_reuse||{status:'此轮未记录前一轮晋级源码的复用。'}))+'</pre></details>';
html+='<details><summary>父生成器与提议子生成器的源码</summary><div class="two"><div><h3>父生成器</h3><pre>'+esc(D.sources[r.active_before])+'</pre></div><div><h3>提议子生成器</h3><pre>'+esc(D.sources[r.proposed_child_sha256]||'未产生合法子源码')+'</pre></div></div></details>';
html+='<details><summary>本轮选择的内存程序</summary><pre>'+esc(pretty(Object.fromEntries(Object.entries(r.selection).map(([k,v])=>[k,v.program]))))+'</pre></details>';
$('round-body').innerHTML=html;}
$('round-select').addEventListener('change',renderRound);renderRound();
for(const button of document.querySelectorAll('[data-tab]'))button.addEventListener('click',()=>{for(const b of document.querySelectorAll('[data-tab]')){const active=b===button;b.setAttribute('aria-selected',active);$(b.dataset.tab).classList.toggle('hidden',!active)}});
$('final-table').innerHTML=table(['比较','配对中位改善 ms','获胜区块','预设门槛','程序关系'],Object.entries(D.test_comparisons).map(([k,v])=>[k,num(v.median_improvement),v.deltas.filter(x=>x.reference_minus_candidate>0).length+'/'+v.deltas.length,v.passed?'通过':'未通过',D.relations[k]?'相同原始 IR':'—']));
$('case-table').innerHTML=table(['产物','对照','场景','配对中位改善 ms','获胜区块'],D.case_deltas.map(r=>[names[r.candidate],names[r.reference],r.case,num(r.median_ms),r.wins+'/'+r.blocks]));
$('frozen').textContent=pretty(D.frozen);$('protocol-json').textContent=pretty({selection_protocol:D.manifest.contract.selection_protocol,usage:D.usage});
$('provenance').textContent='只读本地证据 · 状态 '+D.status+' · report SHA-256 '+D.report_sha256+' · 不调用模型、不上传数据。';
</script></html>'''


def render_html(data):
    encoded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    return HTML.replace('__DATA__', encoded)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--harness-directory', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    try:
        data = bundle(args.report, args.harness_directory)
        html = render_html(data)
    except (ValueError, KeyError, TypeError, OSError) as exc:
        parser.error(f'invalid final v4 report or archived source: {exc}')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open('x', encoding='utf-8', newline='\n') as stream:
        stream.write(html)
    print(json.dumps({'output': str(args.output), 'report_sha256': data['report_sha256'],
                      'html_sha256': sha256(args.output.read_bytes()).hexdigest()}))


if __name__ == '__main__':
    main()
