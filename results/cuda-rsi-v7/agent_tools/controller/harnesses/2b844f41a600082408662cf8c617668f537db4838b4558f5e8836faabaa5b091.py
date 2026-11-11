def build_prompt(context):
    parents = context.get('parent_ids', [])
    evidence = context.get('evidence_ids', [])
    slot = context.get('slot', 0)
    round_index = context.get('round', 0)
    ev = []
    if len(evidence) > 0:
        ev.append(evidence[0])
    if len(evidence) > 1:
        ev.append(evidence[1])
    if len(evidence) > 2:
        ev.append(evidence[2])
    if len(evidence) > 3:
        ev.append(evidence[3])
    if slot % 2 == 0 and len(parents) >= 1:
        return str({'operator': 'mutate', 'parents': parents[:1], 'evidence': ev[:4], 'rule': 5, 'guidance': 'Mutate the top-ranked archived parent around rule 5: preserve gpu0 owner placement and low replication, keep the rule-5 reclaim fallback, and use child/00 and parent/00 as negative controls against host0 owner placement or excessive replicas.'})
    if slot % 2 == 1 and len(parents) >= 2:
        return str({'operator': 'crossover', 'parents': parents[:2], 'evidence': ev[:4], 'rule': 0, 'guidance': 'Cross the two strongest archived parents: retain gpu0 owner placement, low replication, and rule-5 reclaim from child/01 and parent/00; reject any host0 owner or high-replica pattern suggested by child/00 and parent/01 controls.'})
    return str({'operator': 'construct', 'parents': [], 'evidence': ev[:4], 'rule': 0, 'guidance': 'Construct a low-replication gpu0-owner strategy with an explicit rule-5 reclaim fallback; avoid host0 owner placement and excessive replicas seen in weaker training controls.'})

def build_meta_prompt(context):
    evidence = context.get('evidence_ids', [])
    round_index = context.get('round', 0)
    ev = []
    if len(evidence) > 1:
        ev.append(evidence[1])
    if len(evidence) > 3:
        ev.append(evidence[3])
    if len(evidence) > 0:
        ev.append(evidence[0])
    if len(evidence) > 2:
        ev.append(evidence[2])
    if round_index > 0 and round_index % 3 == 0:
        return str({'mode': 'rethink', 'evidence': ev[:4], 'guidance': 'Rethink after repeated repair: compare the best low-replication gpu0-owner and rule-5 reclaim pattern against host0/excess-replica controls, and switch archived parent lineage if repair stops improving.'})
    if round_index % 2 == 0:
        return str({'mode': 'revise', 'evidence': ev[:4], 'guidance': 'Revise the generator around the best child/01 and parent/00 evidence: preserve gpu0 ownership, low replication, and rule-5 reclaim; use child/00 and parent/01 as negative controls for host0 placement or excessive replicas.'})
    return str({'mode': 'repair', 'evidence': ev[:4], 'guidance': 'Repair after revision: keep gpu0 ownership, low replication, and rule-5 reclaim, and reject host0 owner placement or excessive replicas from the weaker evidence.'})
