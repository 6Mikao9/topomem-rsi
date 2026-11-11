"""Packaged HTML, CSS and JavaScript for the offline topology evidence report."""

PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<meta name="color-scheme" content="light">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; base-uri 'none'; form-action 'none'">
<title>TopoMem RSI · 实验记录</title>
<style>
:root{--ink:#132227;--muted:#4c6065;--border:#ccd9d7;--bg:#edf3f1;--paper:#fff;--accent:#066354;--warn:#8b4f06;--fail:#943838;--mono:Consolas,"Courier New",monospace}
*{box-sizing:border-box}body{margin:0;color:var(--ink);background:var(--bg);font:15px/1.55 "Segoe UI","Microsoft YaHei",system-ui,sans-serif}main{width:min(1500px,100%);margin:auto;padding:34px 30px 65px}header{padding:22px 0 26px}h1{font-size:clamp(29px,4vw,52px);letter-spacing:-.06em;margin:4px 0 12px;line-height:1.1}h2{font-size:22px;letter-spacing:-.025em;margin:0 0 12px}h3{font-size:16px;margin:16px 0 8px}p{margin:6px 0 12px}.eyebrow{font:700 12px/1.3 var(--mono);letter-spacing:.12em;color:var(--accent)}.muted,small{color:var(--muted)}.brief{max-width:75ch}.panel{padding:22px;background:var(--paper);border:1px solid var(--border);border-radius:15px;min-width:0}.stack{display:grid;gap:16px}.grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:16px}.wide{grid-column:1/-1}.toolbar{display:flex;gap:14px;flex-wrap:wrap;align-items:end;border-top:1px solid var(--border);padding-top:16px}.toolbar label{display:grid;gap:4px;color:var(--muted);font-size:13px}.toolbar select{min-width:140px}.pills{display:flex;gap:7px;flex-wrap:wrap;margin:7px 0 13px}.pill{border-radius:100px;background:#e5eeec;padding:4px 10px;font:650 12px/1.35 "Segoe UI",system-ui,sans-serif;overflow-wrap:anywhere}.pass{background:#d9f0e8;color:#075349}.fail{background:#fae3e1;color:var(--fail)}.warn{background:#fbefd8;color:var(--warn)}select,button{font:inherit;border:1px solid #9db4b0;border-radius:8px;color:var(--ink);background:white;padding:7px 10px;max-width:100%}select:disabled{opacity:.6}button{cursor:pointer}button:hover,button[aria-pressed=true]{background:#d9f0e8;border-color:var(--accent)}button:focus-visible,select:focus-visible,summary:focus-visible{outline:3px solid var(--accent);outline-offset:2px}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(120px,1fr));gap:9px;margin:8px 0 12px}.metric{background:#f3f7f6;border-radius:8px;padding:10px;min-width:0}.metric strong{display:block;font-size:20px}.metric small{overflow-wrap:anywhere}.empty{padding:15px;border:1px dashed #aebfbc;border-radius:9px;background:#f7f9f8;color:var(--muted)}.clip{overflow:auto;max-height:380px;border:1px solid var(--border);border-radius:8px;margin:8px 0 13px}.tablewrap{overflow:auto;margin:5px 0 13px}table{width:100%;border-collapse:collapse;font-size:13px;text-align:left}th,td{padding:8px 10px;border-bottom:1px solid var(--border);vertical-align:top}th{background:#e9f1ef;font-weight:700;white-space:nowrap}td{overflow-wrap:anywhere}tbody tr:last-child td{border-bottom:0}code,pre,.mono{font-family:var(--mono);font-size:12px}pre{margin:0;padding:13px;white-space:pre-wrap;overflow-wrap:anywhere;background:#f4f8f6;color:#24383c}.diffline{display:block;white-space:pre-wrap}.add{background:#dbf1e5}.del{background:#fbe5e3}.hash{overflow-wrap:anywhere;word-break:break-word}details{border-top:1px solid var(--border);padding-top:8px;margin-top:8px}summary{cursor:pointer;color:var(--accent);font-weight:650}.timeline{display:flex;gap:8px;flex-wrap:wrap}.timeline button{font-size:13px}.scene-nav{display:flex;gap:8px;flex-wrap:wrap;margin-bottom:12px}.scene-nav button{font-size:13px}.surface{background:#f0f7f4;border-radius:12px;margin:7px 0 13px;overflow:hidden}.surface svg{display:block;width:100%;max-height:295px}.section-kicker{color:var(--accent);font-size:13px;font-weight:750;margin-bottom:3px}.legend{font-size:13px;color:var(--muted)}footer{color:var(--muted);font-size:13px;margin-top:20px}.nowrap{white-space:nowrap}
@media(max-width:900px){main{padding:20px 16px 45px}.grid{grid-template-columns:1fr}.panel{padding:17px}.wide{grid-column:auto}}@media(prefers-reduced-motion:reduce){*{scroll-behavior:auto}}
</style>
</head>
<body>
<main>
<header><div class="eyebrow">TOPOMEM RSI / OFFLINE EVIDENCE</div><h1>多 GPU 显存管理实验</h1>
<p class="brief muted">查看生成的显存程序、改写它的生成器，以及裁判记录的实际结果。比较按运行、后端和场景分别展示。</p>
<div class="toolbar"><label>运行<select id="run"></select></label><label>迭代<select id="iteration"></select></label><label>硬件拓扑<select id="topology"></select></label></div>
</header>
<noscript><p class="empty">此离线报告需要 JavaScript 来读取同一文件中的实验数据。</p></noscript>
<div class="stack">
<section class="panel" aria-labelledby="outcome-heading"><h2 id="outcome-heading">封存后的留出结果</h2><div id="outcome"></div></section>
<section class="panel" aria-labelledby="ablation-heading" id="ablation-panel" hidden><h2 id="ablation-heading">冻结方案追加对照</h2><div id="ablation"></div></section>
<section class="panel" aria-labelledby="run-heading"><h2 id="run-heading">运行状态与预算</h2><div id="overview"></div></section>
<div class="grid">
<section class="panel" aria-labelledby="manager-heading"><div class="section-kicker">任务一</div><h2 id="manager-heading">生成显存管理程序</h2><div id="manager"></div></section>
<section class="panel" aria-labelledby="generator-heading"><div class="section-kicker">任务二</div><h2 id="generator-heading">改进生成器</h2><div id="generator"></div></section>
<section class="panel wide" aria-labelledby="paired-heading"><h2 id="paired-heading">配对验证与留出测试</h2><div id="paired"></div></section>
<section class="panel" aria-labelledby="topology-heading"><h2 id="topology-heading">实际拓扑与绑定</h2><div id="graph"></div></section>
<section class="panel" aria-labelledby="scene-heading"><h2 id="scene-heading">独立场景证据</h2><div id="scene"></div></section>
<section class="panel wide" aria-labelledby="serving-heading"><h2 id="serving-heading">SGLang 固定回放回执</h2><div id="serving"></div></section>
<section class="panel wide" aria-labelledby="hand-heading"><h2 id="hand-heading">SGLang 手写策略训练对照</h2><div id="hand"></div></section>
<section class="panel wide" aria-labelledby="transfer-heading"><h2 id="transfer-heading">冻结生成器的四场景迁移</h2><div id="transfer"></div></section>
<section class="panel wide" aria-labelledby="resident-engineering-heading"><h2 id="resident-engineering-heading">全驻留后端 · 独立工程探针</h2><div id="resident-engineering"></div></section>
<section class="panel wide" aria-labelledby="engineering-heading"><h2 id="engineering-heading">执行器工程校准</h2><div id="engineering"></div></section>
<section class="panel wide" aria-labelledby="ledger-heading"><h2 id="ledger-heading">调用与裁判台账</h2><div id="ledger"></div></section>
</div>
</div>
<footer>原始记录离线嵌入。输入、源码和结果摘要的匹配状态随各记录显示。完整物理运行和计时口径以对应的裁判回执为准。</footer>
</main>
<script type="application/json" id="data">/* EMBEDDED_RUN_DATA */</script>
<script>
'use strict';
const data = JSON.parse(document.getElementById('data').textContent);
const q = id => document.getElementById(id);
const state = {run: 0, round: 0, topology: 0, scene: Object.keys(data.scenario_labels)[0]};
const labels = {parent:'父代',child:'子代',fixed:'固定生成器',fixed_a:'固定 A',fixed_b:'固定 B',adaptive:'递归生成器',native:'原生对照',manual:'手写对照',incumbent:'历史 incumbent',seed:'原始生成器',admitted:'已晋级生成器','manual-direct':'直接手写','lru-equivalent':'LRU 等价 IR',writeback:'手写写回'};
const statuses = {admitted:'晋级',rejected:'拒绝',completed:'完成',pending:'待完成',interrupted_unknown:'中断，状态未知',failed:'失败',timed_out:'超时',environment_conflict:'环境冲突',feasible:'可行',infeasible:'不可行'};
const reasons = {both_entries_need_structural_behavior_change:'任务与元入口需要可观察的结构变化',missing_feasible_matched_validation:'缺少可行的匹配验证',missing_measured_incumbent:'缺少实测历史 incumbent',child_retained_incumbent:'子代保留历史 incumbent',identical_effective_parent_and_child_outputs:'父代和子代的有效产出相同',insufficient_stable_paired_improvement:'相对父代的配对收益未达门槛',insufficient_stable_paired_incumbent_improvement:'相对历史 incumbent 的配对收益未达门槛',insufficient_matched_validation_improvement:'相对父代的匹配收益未达门槛',insufficient_matched_incumbent_improvement:'相对历史 incumbent 的匹配收益未达门槛',candidate_failed_native_packed_kv_output_or_dispatch_gate:'选中 KV、输出或实际调度未通过原生一致性门槛',selected_bytes_or_attention_mismatch:'选中字节或注意力输出不一致',native_control_unavailable:'同区块原生对照不可用'};
const number = new Intl.NumberFormat('zh-CN',{maximumFractionDigits:6});
const receiptRequests = new Map();for(const receipt of data.receipts){if(!receiptRequests.has(receipt.job_id))receiptRequests.set(receipt.job_id,new Set());receiptRequests.get(receipt.job_id).add(receipt.request_sha256);}
function txt(value) { return value == null || value === '' ? '—' : typeof value === 'boolean' ? (value?'是':'否') : String(value); }
function num(value) { return typeof value === 'number' && Number.isFinite(value) ? number.format(value) : '—'; }
function seconds(value) { return typeof value === 'number' ? `${num(value)} s` : '—'; }
function ms(value) { return typeof value === 'number' ? `${num(value/1e6)} ms` : '—'; }
function mib(value) { return typeof value === 'number' ? `${num(value/1048576)} MiB` : '—'; }
function bytes(value) { return typeof value === 'number' ? `${num(value)} B` : '—'; }
function mode(value) { return value==='resident'?'全驻留':value==='tiered'?'分层冷后备':'未记录'; }
function short(value) { return typeof value === 'string' && value.length > 14 ? value.slice(0,12)+'…' : txt(value); }
function reason(value){return reasons[value]?`${reasons[value]} · ${value}`:txt(value);}
function el(tag, content, css) {const n=document.createElement(tag);if(content!=null)n.textContent=content;if(css)n.className=css;return n;}
function pill(value, css) {return el('span',statuses[value] || txt(value),`pill ${css||''}`);}
function empty(parent,value) {parent.append(el('p',value,'empty'));}
function scalar(parent,value) {const n=el('code',txt(value),'hash');parent.append(n);}
function table(headers,rows) {const wrap=el('div',null,'tablewrap'), t=el('table'), h=el('thead'), b=el('tbody'), header=el('tr');
  for(const name of headers){const n=el('th',name);n.scope='col';header.append(n);}h.append(header);
  for(const values of rows){const tr=el('tr');for(const item of values){const td=el('td');item instanceof Node ? td.append(item) : td.textContent=txt(item);tr.append(td);}b.append(tr);}
  t.append(h,b);wrap.append(t);return wrap;}
function block(value) {const wrap=el('div',null,'clip');wrap.append(el('pre',typeof value==='string'?value:JSON.stringify(value,null,2)));return wrap;}
function details(parent,title,value) {const d=el('details');d.append(el('summary',title),block(value));parent.append(d);}
function stat(parent,label,value) {const n=el('div',null,'metric');n.append(el('strong',value),el('small',label));parent.append(n);}
function configure(select,values,active,placeholder) {
  select.replaceChildren();
  if(!values.length){select.append(new Option(placeholder,''));select.disabled=true;return;}
  select.disabled=false;values.forEach(([value,label])=>select.append(new Option(label,value)));select.value=String(active);
}
function current(){return data.runs[state.run];}
function round(){return current()?.rounds[state.round];}
function statusClass(value){return ['admitted','completed','feasible'].includes(value)?'pass':value==='pending'?'warn':'fail';}
function trialOrigin(run,trial){if(!trial)return '—';
  if(trial.origin==='initial_program_bootstrap')return '初始程序实测锚点';
  for(const row of run.rounds){if(row.fixed.some(x=>x.id===trial.id))return '固定生成器';if(row.parent.some(x=>x.id===trial.id))return '该轮活动父代';
    if(row.child.some(x=>x.id===trial.id))return row.child_fallback_to_parent?'子代槽的父代回退':row.decision==='admitted'&&row.proposed_child_sha256===trial.harness_sha256?'该轮晋级子代':'未晋级的实验子代';}
  return '来源轮次未导入';
}
function receiptIDs(run){const ids=new Set();if(!run)return ids;
  const add=x=>{if(typeof x==='string')ids.add(x);};
  for(const e of run.evaluations){add(e.metrics?.receipt_id);for(const x of e.metrics?.receipts||[])add(x);}
  for(const p of run.pairs)for(const arm of Object.values(p.arms)){add(arm.metrics?.receipt_id);for(const x of arm.metrics?.receipts||[])add(x);for(const s of arm.samples)add(s.receipt_id);}
  return ids;
}
function usage(calls){const totals={calls:calls.length,known:0,reported:{},seconds:0,unknown:0};
  for(const c of calls){if(typeof c.seconds==='number')totals.seconds+=c.seconds;if(c.status==='interrupted_unknown'||c.status==='pending')totals.unknown++;
    if(Object.keys(c.usage).length)totals.known++;
    for(const [key,value] of Object.entries(c.usage)){
      if(typeof value==='number')totals.reported[key]=(totals.reported[key]||0)+value;
      else if(value&&typeof value==='object')for(const [sub,n] of Object.entries(value))if(typeof n==='number')totals.reported[`${key}.${sub}`]=(totals.reported[`${key}.${sub}`]||0)+n;
    }}return totals;}
function isFixed(call){return call.id.split('/')[2]?.startsWith('fixed');}
function handMatches(selection,sweep){return selection?.sweep_sha256===sweep.sweep_sha256&&selection.selected===sweep.selected&&
  Object.entries(selection.artifact_sha256||{}).length>0&&Object.entries(selection.artifact_sha256).every(([key,sha])=>sweep.artifact_sha256[key]===sha);}
function showOutcome(){const p=q('outcome'),run=current(),o=run?.heldout_outcome;p.replaceChildren();
  if(o?.status==='unverified'){empty(p,'留出记录与封存方案、区块或裁决未完全匹配。');return;}
  if(o?.status==='unpaired'){empty(p,'已记录非配对留出评估。原始分值见下方留出面板。');return;}
  if(o?.status!=='settled'){
    if(o?.pair_status&&!['pending','completed'].includes(o.pair_status))p.append(pill(o.pair_status,statusClass(o.pair_status)));
    empty(p,'等待封存后的完整配对留出记录。');return;
  }
  const sg=run.sglang, native=sg?.native_comparators.default, hand=sg?.manual_selection;
  const sweep=hand&&data.manual_sweeps.find(item=>handMatches(hand,item));
  const rows={native:sg?`SGLang 原生对照${native?.hot_slots?` · ${num(native.hot_slots)} 槽`:''}`:
    o.scope==='real_cuda_fixed_block_sparse'?'CUDA LRU 参考对照':'原生对照',fixed:'固定生成器的封存方案',
    manual:hand?`手写对照 · ${sweep?`${sweep.policies.length} 项训练策略选优${sweep.policies.some(row=>row.backing_mode==='resident')?'（含全驻留）':''}`:'训练选优'} · ${hand.selected}`:'手写显存方案'};
  const scope=sg?'Tiny GLM-5.3 · SGLang 固定回放 · 架构与集成测试':
    o.scope==='real_cuda_fixed_block_sparse'?'CUDA 固定块稀疏注意力 · 实体 KV 读写与生命周期回放':'当前运行的固定评估范围';
  const status={passed:'达到门槛',rejected:'未达到门槛',infeasible:'缺少可行配对'};
  const layouts=o.recorded_configurations;
  const layoutStatus={same:'回执记录配置相同',different:'回执记录存在差异',unverified:'配置未核验'};
  p.append(el('p',scope,'muted'),table(['封存生成方案相对','更快区块 / 可比区块',`配对增益中位数${o.unit?`（${o.unit}）`:''}`,'预注册裁决',...(layouts?['物理配置核对']:[])],
    Object.entries(o.comparisons).map(([label,row])=>[rows[label],row.wins==null?'—':`${row.wins} / ${o.blocks}`,
      row.median_gain==null?'—':`${row.median_gain>0?'+':''}${num(row.median_gain)}`,
      pill(status[row.status],row.status==='passed'?'pass':row.status==='rejected'?'fail':'warn'),
      ...(layouts?[pill(layoutStatus[layouts[label]?.status||'unverified'],layouts[label]?.status==='same'?'warn':'')]:[])])));
  const threshold=o.margin==null?'':`。预注册门槛要求全部区块获胜，且中位增益严格超过 ${num(o.margin)}${o.unit?' '+o.unit:''}`;
  p.append(el('p',`增益取同区块的对照耗时减生成方案耗时。每行独立裁决${threshold}。`,'muted'));
  if(layouts){
    p.append(el('p','配置核对覆盖每个区块与用例的 GPU/rank、热池及冷热后备账本。记录配置相同时，计时差仍保留在预注册统计中，归因于新管理策略需要另有证据。','muted'));
    if(Object.values(layouts).some(row=>row.cases))details(p,'逐用例配置与通过核对的观测数',layouts);
  }
}
function showAblation(){const p=q('ablation');p.replaceChildren();q('ablation-panel').hidden=!data.ablations.length;
  if(!data.ablations.length)return;
  p.append(el('p','冻结的递归臂方案与 6,144 热槽 SGLang 原生策略配对回放。独立记录，不参与正式结果的选优或裁决。','muted'));
  const labels={skipped:'已跳过',partial:'部分执行',pending:'待执行',unverified:'证据未完整核验'};
  for(const study of data.ablations){
    if(study.status!=='complete'){
      p.append(pill(labels[study.status]||'待核验','warn'),el('p',`${num(study.completed_blocks)} / 8 个已记录区块${study.reason?' · '+study.reason:''}`,'muted'));
      continue;
    }
    p.append(pill(study.same_hot_capacity?'同容量原生对照':'调优容量原生对照','warn'),
      pill(study.passed?'达到追加对照门槛':'未达到追加对照门槛',study.passed?'pass':'fail'));
    p.append(el('p',`冻结方案实测热槽 ${study.adaptive_hot_slots.map(num).join(' / ')} · 原生 6,144 热槽 · ${num(study.case_count)} 个固定用例 · ${study.wins} / 8 个区块更快 · 配对中位增益 ${study.median_gain_ms>0?'+':''}${num(study.median_gain_ms)} ms`,'muted'));
    if(!study.same_hot_capacity)p.append(el('p','热槽容量也随比较臂变化，因此无法单独归因于 KV 后备模式。','muted'));
    p.append(table(['区块','实测顺序','冻结方案（ms）','原生 6144（ms）','原生减冻结（ms）'],study.blocks.map(row=>[
      row.block_id,row.order.map(label=>label==='frozen_adaptive'?'冻结方案':'原生 6144').join(' → '),num(row.adaptive_ms),num(row.native_ms),num(row.gain_ms)])));
    p.append(el('p',`追加对照门槛为全部区块更快，且中位增益严格超过 ${num(study.margin_ms)} ms。`,'muted'));
    details(p,'冻结程序与追加对照证据摘要',study.provenance);
  }
}
function showOverview(){const p=q('overview'),r=current();p.replaceChildren();
  if(!r){empty(p,'尚未导入拓扑 RSI 运行。此报告可以独立展示已导入的拓扑、场景和回执。');return;}
  const cfg=r.manifest.config, count=el('div',null,'metrics');
  p.append(pill(r.schema===2?'严格协议 · schema 2':'历史协议 · schema 1',r.schema===2?'pass':'warn'));
  stat(count,'已结算迭代 / 计划',`${r.rounds.length} / ${num(cfg.rounds)}`);stat(count,'生成器源码晋级次数',num(r.rounds.filter(x=>x.decision==='admitted').length));
  stat(count,'每臂每轮候选槽',num(r.manifest.candidate_slots_per_arm_per_round));stat(count,'每臂每轮 API 调用上限',num(r.manifest.provider_calls_per_arm_per_round_cap));p.append(count);
  p.append(el('p',`${r.origin} · ${r.recorded_status||'读取已落盘状态'} · ${cfg.model||'模型未记录'} · 每调用输出上限 ${num(cfg.max_output_tokens)} tokens · 发现阶段墙钟上限 ${seconds(cfg.wall_seconds)}`,'muted'));
  const arms=[['递归生成器',r.calls.filter(c=>!isFixed(c))],['固定生成器',r.calls.filter(isFixed)]];
  p.append(table(['生成器','实际调用','含用量','未知/待完成','输入 tokens','输出 tokens','总 tokens','推理 tokens','记录耗时合计'],arms.map(([label,items])=>{const u=usage(items), t=u.reported;return [label,u.calls,u.known,u.unknown,num(t.prompt_tokens??t.input_tokens),num(t.completion_tokens??t.output_tokens),num(t.total_tokens),num(t['completion_tokens_details.reasoning_tokens']),u.calls?seconds(u.seconds):'—'];})));
  const pending=r.calls.filter(c=>['pending','interrupted_unknown'].includes(c.status)).length;
  if(pending)p.append(pill(`${pending} 个未知或待完成调用`,'warn'));
  if(r.frozen)p.append(pill(r.frozen.hashes_match?'已封存，摘要匹配':'封存摘要不匹配',r.frozen.hashes_match?'pass':'fail'));
  const timeline=el('div',null,'timeline');
  for(const [index,row] of r.rounds.entries()) {const b=el('button',`R${row.round} · ${statuses[row.decision]||txt(row.decision)}`);b.type='button';b.title=txt(row.reason);b.setAttribute('aria-pressed',String(index===state.round));b.onclick=()=>{state.round=index;q('iteration').value=String(index);showRun();};timeline.append(b);}
  if(r.rounds.length)p.append(el('h3','所有已结算迭代'),timeline);
  else empty(p,'暂无已结算迭代。已有调用或裁判记录会出现在台账中。');
}
function showManager(){const p=q('manager'),r=round();p.replaceChildren();if(!r){empty(p,'暂无本轮候选程序。');return;}
  showOutputs(p,r);
  const candidates=['parent','child','fixed'].flatMap(arm=>r[arm].map(x=>[arm,x]));
  if(!candidates.length){empty(p,'本轮未记录候选槽。');return;}
  p.append(table(['候选','生成臂','操作','训练分值','状态','拒绝原因'],candidates.map(([arm,x])=>[el('code',x.id),labels[arm],x.decision?.operator,num(x.score),pill(x.status,statusClass(x.status)),reason(x.rejection?.failure_category||x.rejection?.failure||x.failure)])));
  const selector=el('select');selector.setAttribute('aria-label','候选程序');
  const initial=Math.max(0,candidates.findIndex(([arm,x])=>arm==='child'&&x.program));
  configure(selector,candidates.map(([arm,x],i)=>[i,`${labels[arm]} · ${x.id}`]),initial,'无候选');
  const body=el('div');p.append(selector,body);
  const render=()=>{const [arm,x]=candidates[Number(selector.value)];body.replaceChildren();
    body.append(el('h3','程序与生成来源'));if(!x.program){empty(body,`本槽未产出可读取的程序。${reason(x.rejection?.failure_category||x.rejection?.failure||x.failure||x.status)}`);return;}
    body.append(pill(x.program_hash_matches?'程序摘要匹配':'程序摘要不匹配',x.program_hash_matches?'pass':'fail'),pill(trialOrigin(current(),x)));
    body.append(table(['生成器 SHA-256','程序 SHA-256'],[[el('code',txt(x.harness_sha256),'hash'),el('code',txt(x.program_sha256),'hash')]]));
    const rules=x.program.rules||[];body.append(table(['规则','触发事件','动作'],rules.map((rule,i)=>[i,rule.event,rule.action])));
    details(body,'完整生成程序',x.program);if(x.decision)details(body,'父代、证据与规则编号',x.decision);
    const matches=data.receipts.flatMap(rec=>(rec.result?.cases||[]).filter(c=>c.diagnostic.program_sha256===x.program_sha256).map(c=>[rec,c]));
    if(matches.length){body.append(el('p',`有 ${matches.length} 条同程序摘要的已导入场景观测。`,'muted'),table(['回执','场景','观测动作数','计时中位数'],matches.map(([rec,c])=>[rec.job_id,c.name,c.diagnostic.action_count,ms(c.median_elapsed_ns)])));
      for(const [rec,c] of matches){const d=c.diagnostic;details(body,`${rec.job_id} · ${c.name} · 实际动作 ${d.actions.length}/${d.action_count}`,{scope:d.scope||rec.result.scope,topology_sha256:d.topology_sha256,counts:d.action_types,actions:d.actions});}}
    const serving=data.receipts.filter(rec=>rec.result?.serving?.program_sha256===x.program_sha256);
    if(serving.length)body.append(el('h3','关联 SGLang 回执'),table(['作业','场景','阶段','实际热槽'],serving.map(rec=>{const s=rec.result.serving;return [rec.job_id,s.case.id,s.phase,num(s.memory_compilation.startup.hot_slots)];})));
    if(!matches.length&&!serving.length)body.append(el('p','未导入与此程序摘要匹配的运行时回执。','muted'));
  };selector.onchange=render;render();
}
function showOutputs(parent,r){const output=r.task_output_selection;if(!output)return;
  parent.append(el('h3','本轮提交的程序与历史 incumbent'));
  const fingerprint=current().manifest.effective_program_fingerprint;
  parent.append(el('p',fingerprint==='adapter_training_only_v1'?'有效摘要来自适配器在冻结训练几何下的物理规划。':'有效摘要来自控制器的规则遮蔽分析。','muted'));
  const rows=[['共享历史 incumbent','已实测',output.shared_incumbent],['固定臂历史 incumbent','已实测',output.fixed_incumbent],
    ...Object.entries(output.selected).map(([arm,item])=>[labels[arm],{fresh:'新产出',incumbent:'保留 incumbent',unavailable:'缺失'}[item.origin],item.trial])];
  parent.append(table(['产出','来源','候选 ID','训练分值','IR 摘要','有效摘要'],rows.map(([label,origin,x])=>
    [label,x?origin:'缺失',x?.id,num(x?.score),el('code',txt(x?.program_sha256),'hash'),el('code',txt(x?.effective_program_sha256),'hash')])));
  if(output.parent_incumbent_aa)parent.append(pill('父代与 incumbent 是同一产出 · A/A','warn'));
  details(parent,'各臂最佳新程序及其有效摘要',Object.fromEntries(Object.entries(output.fresh).map(([label,x])=>[label,x&&{id:x.id,score:x.score,program_sha256:x.program_sha256,effective_program_sha256:x.effective_program_sha256}])));
  if(output.fresh_paired_comparison)details(parent,'双方均提交新程序的配对对照',output.fresh_paired_comparison);
}
function planDetails(parent,plan,caption){if(!plan)return;const d=el('details');d.open=true;
  d.append(el('summary',caption),table(['计划项','提出内容'],[['观察到的问题',plan.observed_problem],['机制变化',plan.mechanism_change],['预期效果（假设）',plan.expected_effect],['证伪条件',plan.falsification],['选择的训练证据 ID',plan.training_evidence_ids?.join(' · ')]]));parent.append(d);}
function showGenerator(){const p=q('generator'),run=current(),r=round();p.replaceChildren();if(!r){
  if(run?.meta_plans.length){for(const entry of run.meta_plans){p.append(el('h3',entry.id));planDetails(p,entry.plan,'已落盘的评估前方案');if(entry.failure)p.append(el('p',entry.failure,'muted'));}empty(p,'暂无已结算的机制比较。');}
  else empty(p,'暂无生成器版本改写。');return;}
  planDetails(p,r.adaptive_meta_plan,'评估前方案 · 递归臂');
  planDetails(p,r.fixed_meta_plan,'评估前方案 · 固定臂元产物弃用');
  p.append(el('h3','实测裁决与版本变化'));
  p.append(pill(r.decision,statusClass(r.decision)),el('p',reason(r.reason),'mono'));
  if(run.schema===1)p.append(el('p','历史晋级按当轮父子对照裁决。原生对照和历次最佳应分别查阅。','muted'));
  if(r.parent_validation?.score!=null||r.child_validation?.score!=null)
    p.append(table(['父代验证分值','子代槽验证分值','子代槽是否回退'],[[num(r.parent_validation?.score),num(r.child_validation?.score),txt(r.child_fallback_to_parent)]]));
  p.append(table(['来源','版本 SHA-256'],[['轮前活动版本',el('code',txt(r.active_before),'hash')],['提议子代',el('code',txt(r.proposed_child_sha256),'hash')],['轮后活动版本',el('code',txt(r.active_after),'hash')]]));
  if(r.child_fallback_to_parent)p.append(el('p','子代槽使用父代回退。本轮子代槽的分数应按回退证据读取。','muted'));
  const next=run.rounds.find(x=>x.round===r.round+1), incoming=r.previous_admission_reuse, outgoing=next?.previous_admission_reuse;
  const reuse=x=>`任务调用 ${num(x.task_api_completed)} 次完成 · 程序 ${num(x.task_program_trial_ids?.length)} 个 · 自修订调用${x.meta_api_completed?'完成':'未完成'}`;
  if(incoming){p.append(el('p',`本轮复用上一轮晋级版本 · ${reuse(incoming)}`));details(p,'复用的真实调用与程序 ID',incoming);}
  if(r.decision==='admitted')p.append(el('p',outgoing?.source_sha256===r.active_after?`本轮晋级版本在下一轮 · ${reuse(outgoing)}`:'尚无下一轮已结算的复用记录。','muted'));
  if(r.adaptive_meta_decision)details(p,'实际元修订模式与所选训练证据',r.adaptive_meta_decision);
  if(r.fixed_meta_decision||r.fixed_meta_failure)details(p,'固定对照的元调用槽', {decision:r.fixed_meta_decision,failure:r.fixed_meta_failure,discarded_source_sha256:r.fixed_meta_discarded_sha256});
  if(Object.keys(r.controlled_behavior).length)details(p,'同起点行为对比',r.controlled_behavior);
  const source=el('select');source.setAttribute('aria-label','生成器源码视图');configure(source,[['diff','源码差异'],['before','轮前源码'],['child','提议子代源码'],['after','轮后活动源码']],'diff','');
  const box=el('div');p.append(el('h3','版本代码'),source,box);
  const render=()=>{box.replaceChildren();if(source.value==='diff'){
    if(!r.source_diff?.length){empty(box,'本轮无已保存的源码差异。');return;}
    const content=el('pre'),wrap=el('div',null,'clip');for(const line of r.source_diff)content.append(el('span',line||' ','diffline '+(line.startsWith('+')?'add':line.startsWith('-')?'del':'')));wrap.append(content);box.append(wrap);
  }else{const id={before:r.active_before,child:r.proposed_child_sha256,after:r.active_after}[source.value], record=run.sources[id];
    if(record?.status==='verified'){box.append(block(record.source));return;}empty(box,record?.status==='hash_mismatch'?'归档源码与摘要不匹配。':'没有导入此版本的归档源码。');}};
  source.onchange=render;render();
}
function comparison(parent,values){if(!values||!Object.keys(values).length)return;
  parent.append(table(['比较','裁决','配对差中位数','原因'],Object.entries(values).map(([key,v])=>[key,pill(v.passed?'通过':'未通过',v.passed?'pass':'fail'),num(v.median_improvement),reason(v.reason)])));
  const rows=Object.entries(values).flatMap(([name,v])=>(v.deltas||[]).map(d=>[name,d.block_id,num(d.reference_minus_candidate)]));
  if(rows.length)details(parent,'逐区块差值 · 对照减候选',rows);
}
function pairedRecord(parent,record,scoreUnit){if(!record){empty(parent,'未记录配对区块。');return;}
  parent.append(pill(record.status,statusClass(record.status)));
  if(record.failure)parent.append(el('p',record.failure,'muted'));
  const entries=Object.entries(record.arms), units=scoreUnit?[scoreUnit]:[...new Set(entries.map(([,a])=>a.metrics.score_unit).filter(Boolean))];
  parent.append(el('p',`裁判原始分值，越低越好${units.length?' · '+units.join(' / '):''} · ${seconds(record.seconds)}`,'muted'));
  if(!entries.length)return;
  const order=entries.find(([,a])=>a.metrics.execution_order?.length)?.[1].metrics.execution_order;
  parent.append(table(['区块',...entries.map(([label])=>labels[label]||label)],
    [['裁判可行',...entries.map(([,arm])=>txt(arm.feasible))],['中位数',...entries.map(([,arm])=>num(arm.score))],...(record.block_ids||[]).map(blockId=>[blockId,...entries.map(([,arm])=>num(arm.samples.find(s=>s.block_id===blockId)?.score))])]));
  if(order)parent.append(table(['区块','记录的执行顺序'],order.map(row=>[row.block_id,row.order.map(label=>labels[label]||label).join(' → ')])));
  const refs=entries.flatMap(([label,arm])=>arm.samples.filter(s=>s.receipt_id).map(s=>[labels[label]||label,s.block_id,s.receipt_id]));
  if(refs.length)parent.append(table(['生成臂','区块','关联回执 ID'],refs));
}
function showPaired(){const p=q('paired'),run=current(),r=round();p.replaceChildren();if(!run){empty(p,'暂无验证或留出测试记录。');return;}
  if(run.sglang){const s=run.sglang, hand=s.manual_selection, native=s.native_comparators.default;
    p.append(table(['正式比较臂','程序来源','启动配置 / 来源摘要'],[
      ['原生对照',native?`本实验原生 ${num(native.hot_slots)} 槽对照`:'本实验原生 SGLang 对照',native?`${num(native.hot_slots)} 热槽 · ${txt(native.backing)}`:'待导入'],
      ['手写对照',hand?'训练用例选优':'已记录的手写策略',hand?`${hand.selected} · ${mode(hand.backing_mode)} · ${num(hand.hot_slots)} 热槽`:short(s.manual_program_sha256)],
      ['递归生成器','候选遵循活动生成器谱系',short(run.frozen?.adaptive?.trial?.harness_sha256)],
      ['固定生成器','候选使用固定生成器源码',short(run.frozen?.fixed?.trial?.harness_sha256)]]));
    if(hand)details(p,'训练选优、冻结程序与比较门槛',{manual_selection:hand,preregistration:s.preregistration});
  }
  p.append(el('p',txt(run.manifest.paired_gate),'muted'),el('h3','本轮验证'));
  const scoreUnit=run.sglang?'ms':null;
  if(r?.paired_validation){const v=r.paired_validation;pairedRecord(p,run.pairs.find(x=>x.id===`pair/${v.effect_id}`),scoreUnit);comparison(p,v.comparisons);}
  else if(r){const values=[r.parent_validation,r.child_validation,...r.fixed_validation];p.append(table(['生成臂','入围候选','验证分值','状态'],['父代','子代','固定 A','固定 B'].map((label,i)=>[label,values[i]?.trial?.id,num(values[i]?.score),values[i]?.status])));}
  else empty(p,'暂无本轮验证记录。');
  if(r?.incumbent_validation)p.append(el('p',`历史 incumbent 验证 · ${num(r.incumbent_validation.score)} · ${txt(r.incumbent_validation.trial?.id)}`,'muted'));
  p.append(el('h3','封存后的留出测试'));
  if(!run.frozen){empty(p,'未发现封存记录。');return;}
  const f=run.frozen;p.append(pill(f.hashes_match?'封存摘要匹配':'封存摘要不匹配',f.hashes_match?'pass':'fail'));
  if(f.partial)p.append(pill('部分轮次封存','warn'));
  if(f.admitted_harness_sha256s)details(p,'主结果的准入谱系与实际复用版本',{admitted_harness_sha256s:f.admitted_harness_sha256s,reused_harness_sha256s:f.reused_harness_sha256s});
  p.append(table(['封存臂','验证候选','验证分值','生成来源','生成器摘要','程序摘要'],['adaptive','fixed'].map(arm=>[labels[arm],f[arm]?.trial?.id,num(f[arm]?.score),trialOrigin(run,f[arm]?.trial),el('code',txt(f[arm]?.trial?.harness_sha256),'hash'),el('code',txt(f[arm]?.trial?.program_sha256),'hash')])));
  if(f.adaptive_archive_best){const best=f.adaptive_archive_best;
    p.append(el('h3','实验档案最佳 · 验证分值'),table(['候选','验证分值','生成来源','程序摘要'],[[best.trial?.id,num(best.score),trialOrigin(run,best.trial),el('code',txt(best.trial?.program_sha256),'hash')]]));}
  const test=run.pairs.find(x=>x.id==='pair/frozen/test');
  if(test){pairedRecord(p,test,scoreUnit);comparison(p,run.test_comparisons);}
  else{const rows=run.evaluations.filter(e=>e.split==='test');if(rows.length)p.append(table(['评估 ID','分值','状态'],rows.map(e=>[e.id,num(e.score),e.status])));else empty(p,'暂无封存后测试记录。');}
}
function svgEl(tag,attributes,text){const n=document.createElementNS('http://www.w3.org/2000/svg',tag);for(const [k,v] of Object.entries(attributes))n.setAttribute(k,v);if(text)n.textContent=text;return n;}
function showTopology(){const p=q('graph'),g=data.topologies[state.topology],r=current();p.replaceChildren();
  const contract=r?.contract||{};if(Object.keys(contract).length)details(p,'运行固定契约中的已投影字段',contract);
  if(!g){empty(p,'未导入拓扑图。场景回执仍可查看各自记录的拓扑摘要。');return;}
  scalar(p,g.sha256);const known=[contract.topology_sha256,...data.receipts.flatMap(rec=>(rec.result?.cases||[]).map(x=>x.diagnostic.topology_sha256))].filter(Boolean);
  p.append(el('p',known.includes(g.sha256)?'与至少一条已导入记录的拓扑摘要匹配。':'已导入拓扑文件。当前数据未提供匹配摘要。','muted'));
  if(g.nodes.length<=20&&g.nodes.length){const svg=svgEl('svg',{viewBox:'0 0 580 300',role:'img','aria-label':'已导入的有向硬件拓扑'});
    const defs=svgEl('defs',{}),arrow=svgEl('marker',{id:'arrow',viewBox:'0 0 10 10',refX:9,refY:5,markerWidth:6,markerHeight:6,orient:'auto-start-reverse'});arrow.append(svgEl('path',{d:'M 0 0 L 10 5 L 0 10 z',fill:'#558278'}));defs.append(arrow);svg.append(defs);
    const pos=new Map();g.nodes.forEach((node,i)=>{const angle=2*Math.PI*i/g.nodes.length-Math.PI/2;pos.set(node.id,[290+190*Math.cos(angle),150+95*Math.sin(angle)]);});
    for(const link of g.links){const a=pos.get(link.src),b=pos.get(link.dst);if(!a||!b)continue;const length=Math.hypot(a[0]-b[0],a[1]-b[1]);if(!length)continue;
      const dx=(b[0]-a[0])/length,dy=(b[1]-a[1])/length;svg.append(svgEl('line',{x1:a[0]+dx*24+dy*4,y1:a[1]+dy*24-dx*4,x2:b[0]-dx*28+dy*4,y2:b[1]-dy*28-dx*4,stroke:'#558278','stroke-width':2,'marker-end':'url(#arrow)'}));}
    for(const node of g.nodes){const [x,y]=pos.get(node.id);svg.append(svgEl('circle',{cx:x,cy:y,r:22,fill:node.kind==='gpu'?'#0a6656':'#49676a'}),svgEl('text',{x,y:y+4,'text-anchor':'middle',fill:'white','font-size':11},node.kind),svgEl('text',{x,y:y+38,'text-anchor':'middle',fill:'#203537','font-size':13},node.id));}
    const surface=el('div',null,'surface');surface.append(svg);p.append(surface);
  }
  p.append(table(['节点','类型','KV 容量','物理容量'],g.nodes.map(n=>[n.id,n.kind,mib(n.capacity_bytes),mib(n.physical_bytes)])));
  p.append(table(['有向连接','传输','GB/s','延迟 ns'],g.links.map(l=>[`${l.src} → ${l.dst}`,l.transport,num(l.bandwidth_bytes_per_s/1e9),num(l.latency_ns)])));
}
function showScene(){const p=q('scene'),nav=el('div',null,'scene-nav');p.replaceChildren();
  for(const [kind,name] of Object.entries(data.scenario_labels)){const b=el('button',name);b.type='button';b.setAttribute('aria-pressed',String(kind===state.scene));b.onclick=()=>{state.scene=kind;showScene();};nav.append(b);}p.append(nav);
  const inputs=data.scenarios.filter(x=>x.kind===state.scene),fixtures=inputs.filter(x=>!x.binding),matches=data.receipts.flatMap(rec=>(rec.result?.cases||[]).filter(c=>c.diagnostic.fixture_kind===state.scene).map(c=>[rec,c]));
  if(fixtures.length)p.append(el('h3','已导入输入'),table(['场景文件','内容段','外部事件'],fixtures.map(f=>[f.name||f.file,f.segments,JSON.stringify(f.events)])));
  for(const item of inputs.filter(x=>x.binding))details(p,`${item.file} · ${matches.some(([,c])=>c.diagnostic.scenario_sha256===item.binding_sha256)?'绑定摘要匹配回执':'已导入绑定'}`,{scenario_sha256:item.binding_sha256,...item.binding});
  if(!matches.length){empty(p,'此场景没有可显示的已导入 GPU 观测。');return;}
  const ids=receiptIDs(current()), select=el('select'), content=el('div');select.setAttribute('aria-label','场景 GPU 回执');
  configure(select,matches.map(([rec,c],i)=>[i,`${rec.job_id} · ${c.name} · ${c.diagnostic.arm||'臂未记录'}`]),0,'无回执');p.append(el('h3','实际后端观测'),select,content);
  const render=()=>{content.replaceChildren();const [rec,c]=matches[Number(select.value)],d=c.diagnostic,scope=d.scope||rec.result.scope;
    content.append(pill(rec.status,statusClass(rec.status)),el('p',`${txt(scope)} · ${txt(d.schedule_execution)}`,'mono'));
    if(typeof rec.result.feasible==='boolean')content.append(pill(rec.result.feasible?'裁判可行':'裁判拒绝',rec.result.feasible?'pass':'fail'));
    if(scope==='cuda_text_derived_fixed_block_sparse')content.append(el('p','文本派生的 CUDA 分块稀疏注意力测量。时间是该执行器的测量值。','muted'));
    content.append(el('p',receiptRequests.get(rec.job_id).size>1?'多个输入回执使用相同作业 ID 且请求摘要不同，无法按 ID 唯一归属。':ids.has(rec.job_id)?'当前运行台账通过回执 ID 引用此观测。':'此回执未由当前运行台账的回执 ID 引用。','muted'));
    content.append(table(['运行臂','场景中位数','动作总数','fetch misses','退出时引用'],[[d.arm,ms(c.median_elapsed_ns),d.action_count,num(d.total_fetch_misses),num(d.live_references_at_end)]]));
    const keys=[...new Set([...Object.keys(d.managed_peak_bytes_by_node||{}),...Object.keys(d.cuda_peak_allocated_bytes_delta_by_node||{})])];
    if(keys.length)content.append(table(['节点','管理 KV 峰值','CUDA 分配峰值增量'],keys.map(k=>[k,mib(d.managed_peak_bytes_by_node?.[k]),mib(d.cuda_peak_allocated_bytes_delta_by_node?.[k])])));
    details(content,'观测绑定与时间样本',{request_sha256:rec.request_sha256,source_manifest_sha256:rec.source_manifest_sha256,case_sha256:c.case_sha256,workload_sha256:d.workload_sha256,schedule_sha256:d.schedule_sha256,scenario_sha256:d.scenario_sha256,topology_sha256:d.topology_sha256,program_sha256:d.program_sha256,selected_and_output_sha256:d.selected_and_output_sha256,timings_elapsed_ns:c.timings_elapsed_ns,link_bytes:d.link_bytes,shared_bottleneck_bytes:d.shared_bottleneck_bytes});
    if(d.action_count)details(content,`实际动作 ${d.actions.length} / ${d.action_count} 条`,{counts:d.action_types,events:d.actions});
    if(d.semantics?.length)details(content,'场景语义事件',d.semantics);
  };select.onchange=render;render();
}
function showServing(){const p=q('serving');p.replaceChildren();
  p.append(pill('Tiny GLM · 架构与集成测试范围','warn'),el('p','此发布实验的目标是经玩具文本微调的 tiny GLM-5.3-0.6B-A0.4B。固定独立请求验证 packed KV、输出和实际派发。用例标签不提供前缀共享、写时复制或任务质量证据。','muted'));
  const rows=data.receipts.filter(x=>x.result?.serving);
  if(!rows.length){empty(p,'正式 SGLang 回执待导入。下方公开用例的工程探针独立展示。');return;}
  const select=el('select'),body=el('div');select.setAttribute('aria-label','SGLang 回执');
  configure(select,rows.map((rec,index)=>{const s=rec.result.serving;return [index,`${s.case.id} · ${s.arm} · ${s.phase} · ${mode(s.memory_compilation.startup.backing_mode)} · ${rec.job_id}`];}),0,'无回执');p.append(select,body);
  const render=()=>{body.replaceChildren();const rec=rows[Number(select.value)],result=rec.result,s=result.serving,c=s.memory_compilation,start=c.startup,m=s.metrics;
    body.append(pill(result.feasible?'裁判可行':'裁判拒绝',result.feasible?'pass':'fail'),el('p',`${s.case.id} · ${s.arm} · ${s.phase} · ${rec.job_id}`,'mono'));
    if(s.failure)body.append(el('p',reason(s.failure),'mono'));
    if(s.cross_capacity_native_parity.matched)body.append(pill('原生跨容量一致性已通过','pass'));
    if(c.equivalent_to_strong_stock_control)body.append(pill('编译布局等价于 6144 全主机强对照','warn'));
    const b=el('div',null,'metrics');stat(b,'启动热槽数',num(start.hot_slots));stat(b,'后备模式',mode(start.backing_mode));
    stat(b,'回放窗口',result.score==null?(s.phase==='parity'?'一致性阶段':'待观察'):`${num(result.score)} ms`);
    stat(b,'本用例最大长度上界（tokens）',num(start.case_max_request_tokens));body.append(b);
    if(start.backing_mode==='resident'){
      body.append(pill('人工受信原语 · 程序可选择','warn'),el('p','全驻留省略冷 KV 主机 payload 与备份传输，保留热设备池。CUDA 指针占位按逻辑字节、分配增量、预留增量和保守预算分别记账。','muted'));
      const proof=start.resident;
      if(proof){body.append(el('h3','本用例 6144 热槽证明 · 启动前绑定实际 token IDs'),
        table(['阶段','Rank','请求','输入 tokens','最多新 tokens','长度上界'],proof.requests.map(req=>[req.phase,req.rank,req.rid,num(req.prompt_length),num(req.max_new_tokens),num(req.prompt_length+req.max_new_tokens)])),
        el('p',`物理页预算（rows） · 每 rank 保守预留 ${proof.reserved_rows_by_rank?.map(num).join(' / ')||'—'} · 每 rank 设备行容量 ${num(proof.device_rows)}`,'muted'));
        details(body,'本用例逐请求证明与摘要',proof);
      }else empty(body,'本回执未提供可展示的用例证明。');
    }
    const base=data.topologies.find(graph=>graph.sha256===c.base_topology_sha256);
    const nodes=[...new Set([...Object.keys(c.total_incremental_bytes_by_node||{}),...Object.keys(start.metadata_bytes_reserved_by_node||{})])];
    if(nodes.length)body.append(table(['节点','相对默认热配置的元数据增量','占位保守预算','增量预算合计','共同节点上限（若导入）'],nodes.map(node=>[node,mib(start.metadata_bytes_reserved_by_node?.[node]),mib(start.resident_sentinel_allocator_charge_bytes_by_node?.[node]),mib(c.total_incremental_bytes_by_node?.[node]),mib(base?.nodes.find(x=>x.id===node)?.capacity_bytes)])));
    const keys=['base_target_sha256','base_topology_sha256','external_scheduler_sha256','metadata_profile_sha256'];
    const same=rows.filter(other=>other.source_manifest_sha256===rec.source_manifest_sha256&&other.result.serving.case.id===s.case.id&&keys.every(key=>c[key]&&other.result.serving.memory_compilation[key]===c[key]));
    if(same.length)details(body,'同一用例及共同预算口径下的已导入回执',same.map(other=>{const item=other.result.serving,boot=item.memory_compilation.startup;return {job_id:other.job_id,arm:item.arm,phase:item.phase,hot_slots:boot.hot_slots,backing_mode:boot.backing_mode,feasible:other.result.feasible,score_ms:other.result.score};}));
    if(m.ranks.length){body.append(el('h3','运行时观测 · 每个 rank 的实际后备位置'));
      body.append(table(['Rank','热槽','热设备 payload','热元数据','Pinned host 冷 KV','本卡 HBM 冷 KV','对端 HBM 冷 KV'],m.ranks.map(rank=>[rank.rank,num(rank.hot_slots),bytes(rank.packed_device_payload_bytes),bytes(rank.hot_metadata_bytes),bytes(rank.pinned_host_bytes),bytes(rank.local_hbm_bytes),bytes(rank.peer_hbm_bytes)])));
      if(start.backing_mode==='resident'){
        body.append(table(['Rank','占位逻辑字节','CUDA 已分配增量','CUDA 已预留增量','占位保守预算'],m.ranks.map(rank=>[rank.rank,bytes(rank.resident_sentinel_logical_bytes),bytes(rank.resident_sentinel_allocated_delta_bytes),bytes(rank.resident_sentinel_reserved_delta_bytes),mib(rank.resident_sentinel_allocator_charge_bytes)])));
        const parityFields=['global_ticks_sha256','effective_schedule_sha256','dispatch_sha256','output_ids_sha256','input_tokenization_sha256'];
        const stock=same.find(other=>{const item=other.result.serving;return other.result.feasible&&result.feasible&&item.arm==='native'&&item.phase==='parity'&&s.phase==='parity'&&item.memory_compilation.startup.hot_slots===start.hot_slots&&m.packed_kv_sha256_by_rank?.length&&JSON.stringify(m.packed_kv_sha256_by_rank)===JSON.stringify(item.metrics.packed_kv_sha256_by_rank)&&parityFields.every(key=>m[key]&&m[key]===item.metrics[key]);});
        if(stock)body.append(el('p',`同容量原生一致性回执 ${stock.job_id}`,'mono'),table(['Rank','原生冷主机 payload','全驻留冷主机 payload','冷主机 payload 节省'],m.ranks.map(rank=>{const host=stock.result.serving.metrics.ranks.find(row=>row.rank===rank.rank)?.pinned_host_bytes;return [rank.rank,bytes(host),bytes(rank.pinned_host_bytes),bytes(typeof host==='number'&&typeof rank.pinned_host_bytes==='number'?host-rank.pinned_host_bytes:null)];})));
      }
      body.append(table(['Rank','层','实际 tier','诊断缺页行'],m.ranks.flatMap(rank=>(rank.layer_tiers||[]).map((tier,layer)=>[rank.rank,layer,tier,num(s.feedback.cold_miss_rows_by_rank_layer?.[rank.rank]?.[layer])]))));
      details(body,'实际派发、传输字节和后端内存计数',m.ranks);}
    if(m.timing)details(body,'计时回放 · TTFT、TPOT、窗口和吞吐',m.timing);
    details(body,'用例、编译、共同预算与一致性摘要',{case:s.case,memory_compilation:c,output_ids_sha256:m.output_ids_sha256,packed_kv_sha256_by_rank:m.packed_kv_sha256_by_rank,effective_schedule_sha256:m.effective_schedule_sha256,dispatch_sha256:m.dispatch_sha256,global_ticks_sha256:m.global_ticks_sha256,input_tokenization_sha256:m.input_tokenization_sha256,raw_result_sha256:s.raw_result_sha256,source_manifest_sha256:rec.source_manifest_sha256});
  };select.onchange=render;render();
}
function showTransfer(){const p=q('transfer');p.replaceChildren();
  if(!data.transfers.length){empty(p,'尚未导入生成器迁移预注册。');return;}
  const study=el('select'),scenario=el('select'),body=el('div');study.setAttribute('aria-label','迁移试验');scenario.setAttribute('aria-label','迁移场景');
  configure(study,data.transfers.map((row,index)=>[index,row.name]),0,'无试验');
  configure(scenario,Object.entries(data.scenario_labels),Object.keys(data.scenario_labels)[0],'无场景');p.append(study,scenario,body);
  const render=()=>{body.replaceChildren();const item=data.transfers[Number(study.value)],row=item.scenarios[scenario.value],result=row.heldout;
    body.append(pill('冻结源码迁移 · 合成 CUDA 固定块稀疏注意力','warn'),el('p',`${row.scope==='same-domain-engine-transfer'?'同域引擎迁移':'跨场景迁移'} · GPU ${row.gpu_indices.join(' / ')} · 无新增 RSI 轮次`,'muted'));
    body.append(pill(item.journal_verified?'阶段摘要与台账绑定':'只显示预注册','warn'));
    if(item.phase_seals.selected)body.append(pill('候选已在训练后封存','pass'));
    body.append(table(['生成器','源码 SHA-256','封存槽','封存程序 SHA-256','训练分值'],Object.entries(item.sources).map(([label,sha])=>{const winner=row.winners[label];return [labels[label],el('code',sha,'hash'),winner?.slot_id,el('code',txt(winner?.program_sha256),'hash'),num(winner?.train_score_ms)];})));
    if(row.slots.length)body.append(table(['生成器','槽','结果','原因','API 记录用量'],row.slots.map(slot=>[labels[slot.source],slot.slot,slot.status,reason(slot.failure||slot.finish_reason),num(slot.usage?.total_tokens)])));
    details(body,'阶段封存摘要与固定用例摘要',{preregistration_sha256:item.preregistration_sha256,runtime_sha256:item.runtime_sha256,driver_sha256:item.driver_sha256,phase_seals:item.phase_seals,design:row.design,training_controls:row.controls});
    if(!result){empty(body,'此场景尚无与封存候选绑定的留出结果。');return;}
    const a=result.analysis;
    body.append(el('h3','封存后留出 · 12 个配对区块'),el('p',`每区块增益须严格超过 ${num(row.margin_ms)} ms。区间使用四场景共 12 个预注册聚合对比校正。单对比区间水平 ${num(a.aggregate_interval_level*100)}%。`,'muted'));
    body.append(table(['已晋级生成器相对','严格裁决','有效区块','配对增益中位数（ms）','校正区间（ms）'],Object.entries(a.comparisons).map(([label,contrast])=>{const m=contrast.aggregate,ci=m.paired_bootstrap_interval_ms;
      const status=contrast.identical_program_to_candidate?'封存程序相同':contrast.strict_all_blocks_above_margin?'通过':contrast.complete?'未通过':'不完整';
      return [labels[label],pill(status,status==='通过'?'pass':contrast.complete?'fail':'warn'),`${m.blocks.length} / 12`,num(m.median_gain_ms),ci?ci.map(num).join(' 至 '):'—'];})));
    body.append(table(['区块','实测顺序','各臂分值（ms）或拒绝原因'],result.blocks.map(block=>[block.block_id,block.order.map(x=>labels[x]||x).join(' → '),block.order.map(label=>{const arm=block.arms[label];return `${labels[label]} ${arm.feasible?num(arm.score):reason(arm.failure_type||arm.receipt_status)}`;}).join(' · ')])));
    details(body,'逐区块正确性摘要、计时及回执 ID',result.blocks);
    details(body,'逐块对照减候选增益与两种门槛',a.comparisons);
  };study.onchange=render;scenario.onchange=render;render();
}
function showHand(){const p=q('hand');p.replaceChildren();
  if(!data.manual_sweeps.length){empty(p,'手写训练扫参待导入。');return;}
  p.append(pill('人工策略 · 仅训练用例','warn'),el('p','训练选出的手写对照与原生默认、递归生成器及固定生成器分开计分。全驻留属于共同受信能力。','muted'));
  for(const sweep of data.manual_sweeps){const winner=sweep.selected;
    p.append(el('h3',sweep.name),el('p',`${sweep.policies.length} 项预登记策略 · ${sweep.case_count} 个训练用例 · ${sweep.block_count} 个计划区块 · 相对 ${sweep.reference} 的中位数门槛 ${num(sweep.minimum_gain_ms)} ms`,'muted'));
    if(winner)p.append(pill(`训练选优 · ${winner}`,'pass'));
    else empty(p,sweep.status==='failed'?'训练扫参已失败。':'已导入预检，计时和选优待完成。');
    const linked=data.runs.filter(run=>handMatches(run.sglang?.manual_selection,sweep));
    if(linked.length)p.append(el('p',`正式运行已绑定此训练结果 · ${linked.map(run=>run.label).join(' / ')}`,'muted'));
    p.append(table(['手写策略','后备模式','热槽','状态','逐区块窗口（ms）','较 6144 原生逐区块收益（ms）'],sweep.policies.map(row=>{
      const status=row.label===winner?(linked.length?'已绑定为正式手写对照':'训练选优'):row.failure?'一致性拒绝':row.status==='capacity_rejected'?'容量拒绝':row.window_ms_by_block?'训练计时完成':'预检通过，待计时';
      return [row.label,mode(row.backing_mode),num(row.hot_slots),pill(status,row.label===winner?'pass':row.failure||row.status==='capacity_rejected'?'fail':'warn'),row.window_ms_by_block?.map(num).join(' / '),row.gain_ms_by_block?.map(num).join(' / ')];
    })));
    details(p,'训练策略、程序摘要和证据清单',sweep);
  }
}
function showResidentEngineering(){const p=q('resident-engineering');p.replaceChildren();
  const studies=data.engineering.filter(study=>study.kind==='human_engineering_resident_backing_probe');
  if(!studies.length){empty(p,'尚未导入全驻留工程探针。');return;}
  p.append(pill('人工工程 · 独立公开用例','warn'),el('p','v3 在一个固定 DP2 架构用例上比较相同的 6144 热槽。两组计时仅用于工程接入决策。正式 RSI 的训练、验证和留出结果单独显示。','muted'));
  for(const study of studies){const parity=study.trials.parity, timing=study.trials.performance;
    p.append(el('h3',`${parity.length} 个字节资格进程 · ${timing.length} 个计时进程`),
      table(['计时配对','原生 6144（ms）','全驻留 6144（ms）','原生减全驻留（ms）'],study.pairs.map((pair,index)=>[index+1,num(pair.stock_ms),num(pair.resident_ms),num(pair.gain_ms)])));
    const stock=timing.find(row=>row.mode==='stock'), resident=timing.find(row=>row.mode==='resident');
    p.append(table(['Rank','热设备 payload','探针热元数据¹','原生冷主机 payload','全驻留冷主机 payload','全驻留占位逻辑字节'],resident.ranks.map(rank=>[rank.rank,bytes(rank.hot_geometry.packed_device_payload_bytes),bytes(rank.hot_geometry.metadata_bytes),bytes(stock.ranks.find(row=>row.rank===rank.rank)?.cold_payload_bytes),bytes(rank.cold_payload_bytes),bytes(rank.cuda_pointer_sentinel_logical_bytes)])),
      el('p','¹ v3 探针热元数据含固定请求映射。正式容量校准的 coordinator 元数据口径单列。v3 审计记录的是占位逻辑字节。','muted'));
    details(p,'原始档案摘要、每次独立资格和计时',study);
  }
}
function showEngineering(){const p=q('engineering');p.replaceChildren();
  const studies=data.engineering.filter(study=>study.kind!=='human_engineering_resident_backing_probe');
  if(!studies.length){empty(p,'尚未导入公开工程校准审计。');return;}
  p.append(pill('单卡 · 公开训练输入 · 人工工程改动','warn'),el('p','各臂展示执行器工程校准。实验版本之间的计时来自不同运行窗口。','muted'));
  for(const run of studies){p.append(el('h3',run.file),el('p',`${run.receipt_count} 份回执 · ${run.paired_blocks} 个区块 · ${run.fresh_process_count} 个独立进程`,'muted'));
    p.append(table(['用例','原生（ms）','LRU 等价（ms）','直接手写（ms）','手写写回（ms）'],Object.entries(run.scopes).map(([name,row])=>[name,...['native','lru-equivalent','manual-direct','writeback'].map(arm=>num(row.median_ms[arm]))])));
    details(p,'核对范围与审计摘要',{audit_sha256:run.file_sha256,source_manifest_sha256:run.source_manifest_sha256,environment_scope:run.environment_scope,exact_kv_and_output:run.exact_case_selected_bytes_and_outputs_across_arms,exact_native_lru_traffic_and_misses:run.native_lru_link_bytes_and_misses_exact,exact_against_v10:run.matches_v10_case_program_selected_output_actions_traffic_misses_and_managed_peaks});}
}
function showLedger(){const p=q('ledger'),r=current();p.replaceChildren();
  if(!r?.calls.length)empty(p,'没有导入 API 调用记录。');
  else{p.append(el('h3','API 调用 · 含失败与待完成'),table(['调用','状态','完成原因','来源 SHA','用量合计','耗时'],r.calls.map(c=>[el('code',c.id),pill(c.status,statusClass(c.status)),c.finish_reason||c.failure,short(c.harness_sha256),num(c.usage.total_tokens),seconds(c.seconds)])));}
  if(r&&(r.evaluations.length||r.pairs.length))p.append(el('h3','训练、验证与测试裁判'),table(['评估','拆分','状态','裁判可行','分值','耗时'],[...r.evaluations,...r.pairs].map(e=>[el('code',e.id),e.split,pill(e.status,statusClass(e.status)),txt(e.feasible),num(e.score),seconds(e.seconds)])));
  if(r?.meta_plans.length)p.append(el('h3','评估前方案台账'),table(['记录 ID','状态','前代','提议子代'],r.meta_plans.map(entry=>[el('code',entry.id),entry.status,short(entry.parent_sha256),short(entry.proposed_child_sha256)])));
  if(!data.receipts.length){p.append(el('p','未导入独立 GPU 回执文件。','muted'));return;}
  p.append(el('h3','独立 GPU 回执 · 按文件来源'),table(['作业','进程状态','裁判可行','后端范围','结果摘要','GPU UUID','运行耗时'],data.receipts.map(x=>[x.job_id,pill(x.status,statusClass(x.status)),txt(x.result?.feasible),x.result?.scope,x.canonical_result_matches==null?'无结果':x.canonical_result_matches?'匹配':'不匹配',x.gpu_uuids?.join(', '),seconds(x.elapsed_seconds)])));
}
function showRun(){showOutcome();showOverview();showManager();showGenerator();showPaired();showTopology();showScene();showLedger();}
function configureRound(){configure(q('iteration'),(current()?.rounds||[]).map((row,index)=>[index,`R${row.round} · ${statuses[row.decision]||txt(row.decision)}`]),state.round,'暂无迭代');}
configure(q('run'),data.runs.map((r,i)=>[i,r.label]),state.run,'暂无运行');configureRound();
configure(q('topology'),data.topologies.map((g,i)=>[i,g.name]),state.topology,'暂无拓扑');
q('run').onchange=()=>{state.run=Number(q('run').value);state.round=0;configureRound();showRun();};
q('iteration').onchange=()=>{state.round=Number(q('iteration').value);showRun();};
q('topology').onchange=()=>{state.topology=Number(q('topology').value);showTopology();};
showRun();showAblation();showServing();showHand();showTransfer();showResidentEngineering();showEngineering();
</script>
</body>
</html>
'''
