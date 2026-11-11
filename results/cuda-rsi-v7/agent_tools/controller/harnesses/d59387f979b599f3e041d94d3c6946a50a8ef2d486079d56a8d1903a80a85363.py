def build_prompt(context):
    parents = context.get('parent_ids', [])
    evidence = context.get('evidence_ids', [])
    slot = context.get('slot', 0)
    round_index = context.get('round', 0)
    ev = []
    if len(evidence) > 1:
        ev.append(evidence[1])
    if len(evidence) > 0:
        ev.append(evidence[0])
    if len(evidence) > 3:
        ev.append(evidence[3])
    if len(evidence) > 2:
        ev.append(evidence[2])
    chosen = []
    if slot % 2 == 0:
        if len(parents) >= 1:
            chosen.append(parents[0])
        return str({'operator': 'mutate', 'parents': chosen, 'evidence': ev[:4], 'rule': 4, 'guidance': 'Mutate the top archived parent around rule-4 eviction and lower replication; preserve gpu0 owner placement and rule-5 reclaim, using parent/00 and child/00 as lower-replicate controls and child/01/parent/01 as the rule-4/high-replicate contrast.'})
    return str({'operator': 'construct', 'parents': [], 'evidence': ev[:4], 'rule': 0, 'guidance': 'Construct a fresh low-replication gpu0-owner strategy with explicit rule-5 reclaim and bounded rule-4 eviction; avoid the higher-replicate child/01/parent/01 pattern and compare against parent/00/child/00 evidence.'})

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
    if round_index % 3 == 0:
        return str({'mode': 'revise', 'evidence': ev[:4], 'guidance': 'Revise the search after comparing the lower-replicate parent/00 and child/00 evidence against the higher-replicate parent/01 and child/01 cases; prioritize rule-4 eviction and replication reductions while retaining gpu0 ownership and rule-5 reclaim.'})
    return str({'mode': 'rethink', 'evidence': ev[:4], 'guidance': 'Rethink the repair lineage: the selected evidence separates lower-replicate runs from higher-replicate runs, so test a focused rule-4 eviction and replication mutation rather than another rule-5 crossover; keep gpu0 owner placement and reclaim fallback.'})
