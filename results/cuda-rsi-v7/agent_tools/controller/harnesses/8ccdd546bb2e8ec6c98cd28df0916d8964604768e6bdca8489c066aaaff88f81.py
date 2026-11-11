def build_prompt(context):
    parents = context.get('parent_ids', [])
    evidence = context.get('evidence_ids', [])
    slot = context.get('slot', 0)
    if slot == 0:
        if parents:
            return str({'operator': 'mutate', 'parents': parents[:1], 'evidence': evidence[:4], 'rule': 0, 'guidance': 'Training feedback shows rising fetch_misses and evictions while hot pages 6 and 11 had zero misses. Mutate the highest-ranked real parent and keep its reclamation skeleton, but target retention of hot pages and demand for pages with observed misses.'})
        return str({'operator': 'construct', 'parents': [], 'evidence': evidence[:4], 'rule': 0, 'guidance': 'No parent is available, so construct a fallback that keeps hot pages and uses demand only for pages with observed fetch misses.'})
    if len(parents) >= 2:
        return str({'operator': 'crossover', 'parents': parents[:2], 'evidence': evidence[:4], 'rule': 0, 'guidance': 'Use two real archived parents to combine reclamation safety with demand and replicate behavior aimed at pages that showed fetch misses.'})
    return str({'operator': 'construct', 'parents': [], 'evidence': evidence[:4], 'rule': 0, 'guidance': 'Use a construct fallback on the second slot to escape the fixed parity mutation; focus on hot-page retention and avoid evictions of pages with observed read pressure.'})

def build_meta_prompt(context):
    evidence = context.get('evidence_ids', [])
    return str({'mode': 'repair', 'evidence': evidence[:2], 'guidance': 'Repair the current search using the supplied training feedback: fetch_misses rose from 54 to 66 and evict from 71 to 83 while demand_replica also rose. Prioritize rules that retain hot pages 6 and 11 and reduce avoidable misses on pages 0, 2, 4, and 9 without changing fixed limits.'})
