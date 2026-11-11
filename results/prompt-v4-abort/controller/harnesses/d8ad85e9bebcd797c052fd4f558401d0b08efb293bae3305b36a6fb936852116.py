def build_prompt(context):
    parent_ids = list(context.get('parent_ids', []))
    evidence_ids = list(context.get('evidence_ids', []))
    slot = context.get('slot', 0)
    order = []
    for eid in ['native', 'bootstrap/initial-incumbent']:
        if eid in evidence_ids:
            order.append(eid)
    for eid in evidence_ids:
        if eid not in order:
            order.append(eid)
    evidence = order[:4]
    if slot % 2 == 1 and parent_ids:
        return str({'operator': 'mutate', 'parents': parent_ids[:1], 'evidence': evidence, 'rule': 1, 'guidance': 'Mutate the archived parent rule 1, not the placement rule. Training evidence shows bootstrap added retain_producer_replica for every placement while native recorded demand_replica equal to fetch misses. Condition or remove the unconditional retain path, preserve allocate/place, and keep eviction and reclamation demand-driven. Compare elapsed and fetch misses per fixture.'})
    return str({'operator': 'construct', 'parents': [], 'evidence': evidence, 'rule': 0, 'guidance': 'Construct from native-first evidence: required allocate/place owner routing, conditional demand replication rather than unconditional producer retention, safe pressure eviction, and zero-reference reclaim. Avoid adding retain_producer_replica for all placements; keep a feasible fallback and compare each case against the bootstrap evidence.'})

def build_meta_prompt(context):
    evidence_ids = list(context.get('evidence_ids', []))
    order = []
    for eid in ['native', 'bootstrap/initial-incumbent']:
        if eid in evidence_ids:
            order.append(eid)
    for eid in evidence_ids:
        if eid not in order:
            order.append(eid)
    evidence = order[:4]
    round_index = context.get('round', 0)
    slot = context.get('slot', 0)
    if round_index == 0:
        mode = 'repair'
        guidance = 'Repair the measured regression: native-first evidence shows demand_replica tracks fetch misses while bootstrap added an unconditional retain_producer_replica rule. Preserve successful placement, eviction, and reclamation structure; target rule 1 for conditional demand replication or removal, and use the other slot to construct a clean demand-driven fallback.'
    else:
        if slot % 2 == 0:
            mode = 'rethink'
            guidance = 'Re-think if repair did not lower elapsed time: compare native and bootstrap case evidence, prioritize eliminating unconditional retention, and test a new conditional replication or prefetch hypothesis without changing evaluator, budgets, or resource limits.'
        else:
            mode = 'repair'
            guidance = 'Repair the best archived parent by targeting the rule that produced excessive retain_producer_replica actions; keep demand replication tied to observed misses and verify all fixtures remain feasible.'
    return str({'mode': mode, 'evidence': evidence, 'guidance': guidance})
