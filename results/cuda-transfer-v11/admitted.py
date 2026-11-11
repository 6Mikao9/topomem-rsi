def build_prompt(context):
    parents = context.get('parent_ids', [])
    evidence = context.get('evidence_ids', [])
    slot = context.get('slot', 0)
    round_index = context.get('round', 0)
    ev = []
    if len(evidence) > 0:
        ev.append(evidence[0])
    if len(evidence) > 2:
        ev.append(evidence[2])
    if len(evidence) > 1:
        ev.append(evidence[1])
    if len(evidence) > 3:
        ev.append(evidence[3])
    if slot % 2 == 0 and round_index >= 2 and len(parents) >= 2:
        return str({'operator': 'crossover', 'parents': parents[:2], 'evidence': ev[:4], 'rule': 0, 'guidance': 'Cross the two strongest archived strategies: keep local-GPU ownership, low replication, and rule-5 reclaim from child/01 and parent/01; use child/00 and parent/00 as negative controls for host0 placement and excessive replicas.'})
    if slot % 2 == 1 and len(parents) >= 1:
        return str({'operator': 'mutate', 'parents': parents[:1], 'evidence': ev[:4], 'rule': 5, 'guidance': 'Mutate reclaim rule 5 of the best archived parent; preserve gpu0 owner placement and low replication, and reject the high-replication host0 pattern seen in weaker evidence.'})
    return str({'operator': 'construct', 'parents': [], 'evidence': ev[:4], 'rule': 0, 'guidance': 'Construct a low-replication local-GPU owner strategy with an explicit rule-5 reclaim fallback; avoid host0 owner replication from the weaker evidence.'})

def build_meta_prompt(context):
    evidence = context.get('evidence_ids', [])
    round_index = context.get('round', 0)
    ev = []
    if len(evidence) > 0:
        ev.append(evidence[0])
    if len(evidence) > 2:
        ev.append(evidence[2])
    if len(evidence) > 1:
        ev.append(evidence[1])
    if len(evidence) > 3:
        ev.append(evidence[3])
    if round_index > 0 and round_index % 3 == 0:
        return str({'mode': 'rethink', 'evidence': ev[:4], 'guidance': 'Rethink after repeated repair: compare the best local-GPU/low-replication pattern against the high-replication and host0 controls, and choose a different archived parent if the current repair lineage stops improving.'})
    return str({'mode': 'repair', 'evidence': ev[:4], 'guidance': 'Repair around child/01 and parent/01: preserve gpu0 ownership, low replication, and rule-5 reclaim; use child/00 and parent/00 as negative controls for host0 placement and excessive replicas.'})
