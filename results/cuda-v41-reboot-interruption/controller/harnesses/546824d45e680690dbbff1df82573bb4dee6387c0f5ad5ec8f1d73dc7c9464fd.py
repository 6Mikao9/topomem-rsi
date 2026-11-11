def build_prompt(context):
    parents = context.get("parent_ids", [])
    evidence = context.get("evidence_ids", [])
    slot = context.get("slot", 0)
    if slot % 2 == 0:
        chosen_evidence = evidence[:1]
        if parents:
            operator = "mutate"
            chosen_parents = parents[:1]
        else:
            operator = "construct"
            chosen_parents = []
        guidance = "Mutate the strongest archived parent and keep one evidence item; target reclaim/evict balance and avoid redundant retain/place rules."
    else:
        chosen_evidence = evidence[-1:] if evidence else []
        if len(parents) >= 2:
            operator = "crossover"
            chosen_parents = parents[:2]
        elif parents:
            operator = "mutate"
            chosen_parents = parents[:1]
        else:
            operator = "construct"
            chosen_parents = []
        guidance = "Crossover two ranked parents when available, otherwise mutate one; compare reclaim/evict/replicate counts and avoid redundant retain_producer_replica actions."
    return str({"operator": operator, "parents": chosen_parents, "evidence": chosen_evidence, "rule": 0, "guidance": guidance})

def build_meta_prompt(context):
    evidence = context.get("evidence_ids", [])
    return str({"mode": "rethink", "evidence": evidence[-1:] if evidence else [], "guidance": "Treat the native reference as a lower-overhead baseline: remove redundant retain_producer_replica and place actions, and search parent edits that reduce validated median time while preserving required place coverage."})
