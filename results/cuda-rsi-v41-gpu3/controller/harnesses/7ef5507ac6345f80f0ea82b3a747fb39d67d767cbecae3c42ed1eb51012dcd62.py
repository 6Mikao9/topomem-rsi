def build_prompt(context):
    parents = context['parent_ids']
    evidence = context['evidence_ids']
    slot = context['slot']
    if len(evidence) >= 4:
        chosen = [evidence[-1], evidence[-2], evidence[0], evidence[1]]
    else:
        chosen = evidence[-4:]
    best = parents[:1]
    if slot == 0 or not parents:
        return str({'operator': 'construct', 'parents': [], 'evidence': chosen, 'rule': 0, 'guidance': 'Construct a fresh policy that mirrors the best archived parent emphasis on allocation rule 1 and reclamation rule 5; include a safe fallback and a specific transfer or replicate hypothesis.'})
    return str({'operator': 'mutate', 'parents': best, 'evidence': chosen, 'rule': 1, 'guidance': 'Mutate the best archived parent allocation rule 1 while preserving its rule 5 reclamation and rule 2 replicate behavior; test whether avoiding rule 0 host placement reduces transfer and eviction work.'})

def build_meta_prompt(context):
    evidence = context['evidence_ids']
    if len(evidence) >= 4:
        chosen = [evidence[-1], evidence[-2], evidence[0], evidence[1]]
    else:
        chosen = evidence[-4:]
    return str({'mode': 'rethink', 'evidence': chosen, 'guidance': 'Rethink the search because the child underperformed the best archived parent; shift mutation focus from rule 0 to the parent rule 1 allocation and rule 5 reclamation, and prefer a fresh construct with an explicit fallback before further revision.'})
