def build_prompt(context):
    parents = context.get('parent_ids', [])
    evidence = context.get('evidence_ids', [])
    slot = context.get('slot', 0)
    candidates = context.get('candidates', {})
    ranked = []
    for pid in parents:
        cand = candidates.get(pid, {})
        score = cand.get('score', None)
        status = cand.get('status', '')
        if score != None and status == 'feasible':
            ranked.append((score, pid))
    ranked = sorted(ranked)
    best_parent = []
    best_score = None
    if ranked:
        best_score = ranked[0][0]
        best_parent = [ranked[0][1]]
    elif parents:
        best_parent = parents[:1]
    ev = evidence[:4]
    if len(parents) >= 2 and slot % 3 == 2:
        op = 'crossover'
        ps = parents[:2]
        rule = 0
        guidance = 'Crossover the two ranked parents and keep their correct demand, eviction, and reclamation structure. Preserve explicit retain or replicate for hot immutable blocks from the best feasible training evidence; narrow prefetch and reduce reclaim churn; keep a local capacity fallback.'
    elif best_parent:
        op = 'mutate'
        ps = best_parent
        rule = 1 if best_score != None and best_score < 150 else 0
        guidance = 'Mutate the selected parent toward explicit retain or replicate of hot immutable blocks, fewer broad prefetches, and bounded eviction/reclaim. Preserve all working demand and reclamation rules and add a capacity-pressure fallback.'
    else:
        op = 'construct'
        ps = []
        rule = 0
        guidance = 'Construct a topology-aware policy with explicit retain or replicate for hot blocks, narrow prefetch, safe eviction and reclamation, and a capacity-pressure fallback. Use training evidence to avoid prefetch-heavy reclaim churn.'
    return str({'operator': op, 'parents': ps, 'evidence': ev, 'rule': rule, 'guidance': guidance})

def build_meta_prompt(context):
    evidence = context.get('evidence_ids', [])
    ev_map = context.get('evidence', {})
    candidates = context.get('candidates', {})
    rounds = context.get('round', 0)
    ev = evidence[:4]
    feasible = 0
    failed = 0
    best_score = None
    best_id = ''
    for eid in ev:
        rec = ev_map.get(eid, {})
        status = rec.get('status', '')
        score = rec.get('score', None)
        if status == 'feasible' and score != None:
            feasible += 1
            if best_score == None or score < best_score:
                best_score = score
                best_id = eid
        elif status in ('invalid_candidate', 'api_failed', 'error', 'failed'):
            failed += 1
    for pid in context.get('parent_ids', []):
        cand = candidates.get(pid, {})
        status = cand.get('status', '')
        score = cand.get('score', None)
        if status == 'feasible' and score != None:
            feasible += 1
            if best_score == None or score < best_score:
                best_score = score
                best_id = pid
        elif status in ('invalid_candidate', 'api_failed', 'error', 'failed'):
            failed += 1
    ordered = []
    if best_id:
        ordered.append(best_id)
    for eid in ev:
        if eid != best_id:
            ordered.append(eid)
    ev = ordered[:4]
    if feasible == 0 and failed > 0:
        mode = 'repair'
        guidance = 'Repair the generator by keeping its working demand, eviction, and reclamation skeleton while replacing broad prefetch with explicit retain or replicate and a safe capacity fallback. Use failed training evidence to avoid repeating invalid or API-failed structures.'
    elif failed > feasible and rounds >= 2:
        mode = 'rethink'
        guidance = 'Rethink the search structure: retain only verified feasible components, switch from broad prefetch to bounded retain or replicate, and recombine the best parent with a simple fallback before more mutation.'
    else:
        mode = 'revise'
        guidance = 'Revise the current generator with a targeted change: preserve the best feasible parent rules for demand, eviction, and reclamation; strengthen explicit retain or replicate for hot blocks; narrow prefetch; and use training evidence to reduce reclaim and eviction churn.'
    return str({'mode': mode, 'evidence': ev, 'guidance': guidance})
