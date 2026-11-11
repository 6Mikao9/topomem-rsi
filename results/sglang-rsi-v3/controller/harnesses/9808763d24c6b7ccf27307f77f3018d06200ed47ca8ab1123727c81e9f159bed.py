def build_prompt(context):
    slot = context.get('slot', 0)
    parents = list(context.get('parent_ids', []))
    evidence = sorted(context.get('evidence_ids', []))
    if len(evidence) > 4:
        evidence = evidence[:4]
    if slot % 2 == 0:
        ev = evidence[:3]
        if len(ev) == 0:
            ev = evidence
        return str({'operator': 'construct', 'parents': [], 'evidence': ev, 'rule': 0, 'guidance': 'Construct a tiered candidate with hot_capacity=6144 and backing_mode=tiered. Training shows 4096 leaves selected cold misses while 6144 covers the fixed maximum request, so place the earliest layer slabs on local HBM first and use host only after local HBM is exhausted. Keep a valid tiered family with at least one allocate/place rule.'})
    if len(parents) == 0:
        return str({'operator': 'construct', 'parents': [], 'evidence': evidence[:3], 'rule': 0, 'guidance': 'Construct the best resident 6144 program from evidence; require hot_capacity=6144 and backing_mode=resident with no allocate/place rules.'})
    return str({'operator': 'mutate', 'parents': parents[:1], 'evidence': evidence[:3], 'rule': 0, 'guidance': 'Mutate the best measured parent. If it is the resident 6144 program, keep the qualified hot_capacity and backing_mode rules and test only a single rule replacement that cannot add allocate/place rules. Otherwise replace the hot_capacity rule with 6144 and preserve earliest-layer local-HBM placement.'})

def build_meta_prompt(context):
    evidence = sorted(context.get('evidence_ids', []))[:4]
    return str({'mode': 'repair', 'evidence': evidence, 'guidance': 'Repair the search using sorted training evidence: prioritize the best-scoring child, compare it with the resident parent and the strongest tiered parent, and keep the resident 6144 proof as the incumbent unless a tiered 6144 variant with zero selected misses is measured better. Change one operator or parent rule at a time.'})