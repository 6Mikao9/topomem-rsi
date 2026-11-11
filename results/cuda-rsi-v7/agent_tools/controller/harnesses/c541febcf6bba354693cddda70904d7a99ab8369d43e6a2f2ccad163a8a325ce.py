def build_prompt(context):
    parents = context.get('parent_ids', [])
    evidence = context.get('evidence_ids', [])
    slot = context.get('slot', 0)
    ev = []
    if len(evidence) > 2:
        ev.append(evidence[2])
    if len(evidence) > 0:
        ev.append(evidence[0])
    if len(evidence) > 3:
        ev.append(evidence[3])
    if len(evidence) > 1:
        ev.append(evidence[1])
    if slot % 2 == 0:
        if len(parents) > 0:
            return str({'operator': 'mutate', 'parents': parents[:1], 'evidence': ev[:4], 'rule': 5, 'guidance': 'Mutate the top-ranked feasible archive at rule 5, preserving the low-replication local-GPU owner pattern seen in child/00; do not follow the high-replication host0 or infeasible parent/01 patterns.'})
        return str({'operator': 'construct', 'parents': [], 'evidence': ev[:4], 'rule': 0, 'guidance': 'Construct a low-replication local-GPU owner with an explicit rule-5 reclaim fallback; avoid high-replication host0 and infeasible producer-capacity patterns.'})
    if len(parents) > 1:
        return str({'operator': 'crossover', 'parents': parents[:2], 'evidence': ev[:4], 'rule': 0, 'guidance': 'Crossover the two ranked feasible parents while biasing evidence toward the best observed child/00; keep gpu0 ownership, low replication, and rule-5 reclaim, and treat parent/00 high replication as a negative control.'})
    if len(parents) > 0:
        return str({'operator': 'mutate', 'parents': parents[:1], 'evidence': ev[:4], 'rule': 5, 'guidance': 'Mutate the available feasible parent at rule 5; preserve low replication and local-GPU ownership.'})
    return str({'operator': 'construct', 'parents': [], 'evidence': ev[:4], 'rule': 0, 'guidance': 'Construct a low-replication local-GPU owner with rule-5 reclaim.'})

def build_meta_prompt(context):
    evidence = context.get('evidence_ids', [])
    round_index = context.get('round', 0)
    ev = []
    if len(evidence) > 2:
        ev.append(evidence[2])
    if len(evidence) > 0:
        ev.append(evidence[0])
    if len(evidence) > 3:
        ev.append(evidence[3])
    if len(evidence) > 1:
        ev.append(evidence[1])
    if round_index > 0 and round_index % 3 == 0:
        return str({'mode': 'repair', 'evidence': ev[:4], 'guidance': 'Repair around the best feasible child/00 rather than continuing the previous rethink lineage: keep gpu0 ownership, low replication, and rule-5 reclaim; use parent/00 high-replication evidence and parent/01 infeasibility as negative controls.'})
    if round_index > 0 and round_index % 2 == 0:
        return str({'mode': 'rethink', 'evidence': ev[:4], 'guidance': 'Rethink archive choice by comparing the low-replication child/00 pattern against the high-replication parent/00 control and excluding infeasible parent/01; switch parent lineage if repair stalls.'})
    return str({'mode': 'repair', 'evidence': ev[:4], 'guidance': 'Repair around the best feasible child/00: preserve gpu0 ownership, low replication, and rule-5 reclaim; treat high-replication host0 placement and infeasible capacity plans as negative controls.'})
