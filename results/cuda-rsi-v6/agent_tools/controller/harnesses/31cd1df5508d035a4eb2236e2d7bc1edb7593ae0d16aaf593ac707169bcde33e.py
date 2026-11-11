def build_prompt(context):
    parents = context.get('parent_ids', [])
    evidence = context.get('evidence_ids', [])
    slot = context.get('slot', 0)
    rnd = context.get('round', 0)
    ev = evidence[:4]
    phase = (slot + rnd) % 6
    if phase == 0:
        return str({'operator': 'construct', 'parents': [], 'evidence': ev, 'rule': 0, 'guidance': 'Construct a capacity-aware KV policy for a small compute-GPU pool. Training feedback shows gpu0 at its managed cap, frequent eviction/reclamation, and repeated fetch misses on moderately hot pages. Keep high-attention pages resident, use host0 as backing, replicate sparingly, and order victims by coldness plus version utility. Include complete demand, eviction, and reclaim rules with a capacity-pressure fallback.'})
    if phase == 1 and parents:
        return str({'operator': 'mutate', 'parents': parents[:1], 'evidence': ev, 'rule': 0, 'guidance': 'Mutate the selected parent around one measurable bottleneck: near-full gpu0 managed capacity, high evict/reclaim volume, or residual fetch misses. Preserve rules that retain hot pages; change only the replication threshold, victim score, or fallback path. Validate that the change reduces misses without adding avoidable pinned-host traffic.'})
    if phase == 2 and len(parents) >= 2:
        return str({'operator': 'crossover', 'parents': parents[:2], 'evidence': ev, 'rule': 0, 'guidance': 'Combine two archived policies: take hot-page retention/replication behavior from one and safe eviction/idle-reclamation ordering from the other. Keep the result feasible under the tiny gpu0 budget, with host0 as cold backing and no unbounded relay staging. Prefer simple first-match rules and a fallback.'})
    if phase == 3:
        return str({'operator': 'construct', 'parents': [], 'evidence': ev, 'rule': 0, 'guidance': 'Explore a fresh topology-aware rule family. Because gpu0 capacity is the binding constraint and host round trips are costly, separate ownership placement from demand replication: keep the working set local, back cold versions on host0, and reclaim zero-reference blocks promptly. State a concrete transfer hypothesis and a fallback for pressure.'})
    if phase == 4 and parents:
        return str({'operator': 'mutate', 'parents': parents[:1], 'evidence': ev, 'rule': 0, 'guidance': 'Refine the parent decision boundary using observed page activity: distinguish repeatedly read pages from one-use pages, and prefer local residency for the former. Tune eviction and reclaim to avoid evicting blocks that are likely to be fetched again. Keep all required event/action rules and a safe fallback.'})
    if phase == 5 and parents:
        return str({'operator': 'mutate', 'parents': parents[:1], 'evidence': ev, 'rule': 0, 'guidance': 'Stress-test the parent under capacity pressure: simulate a full gpu0, then ensure each eviction/reclaim decision chooses a cold victim and each demand read has a legal route. If a rule risks thrashing hot pages, adjust victim ordering or replication policy, not the evaluator or fixed limits.'})
    return str({'operator': 'construct', 'parents': [], 'evidence': ev, 'rule': 0, 'guidance': 'Construct a complete policy for the small compute-GPU pool: keep hot KV local, use host0 for cold backing, replicate only high-value pages, and pair demand handling with deterministic eviction and reclamation. Use the training evidence to set conservative thresholds and include a fallback for full capacity.'})

def build_meta_prompt(context):
    evidence = context.get('evidence_ids', [])
    rnd = context.get('round', 0)
    ev = evidence[:4]
    mode = 'revise'
    if rnd % 5 == 1:
        mode = 'repair'
    elif rnd % 5 == 2:
        mode = 'rethink'
    elif rnd % 5 == 3:
        mode = 'revise'
    elif rnd % 5 == 4:
        mode = 'repair'
    guidance = 'Revise the generator, not the fixed evaluator. Preserve useful construct/mutate/crossover exploration but make it evidence-driven: alternate fresh construction with mutation of archived parents and occasional crossover; spend more slots on capacity-pressure failures. Use both training cases feedback (near-full gpu0, many evictions/reclaims, repeated fetch misses) to guide the next policy search toward hot-page retention, conservative replication, host backing, safe victim selection, and complete fallback rules. Repair missing/infeasible rules before tuning. Rethink only if repeated revisions do not reduce misses or if guidance overfits one case.'
    return str({'mode': mode, 'evidence': ev, 'guidance': guidance})
