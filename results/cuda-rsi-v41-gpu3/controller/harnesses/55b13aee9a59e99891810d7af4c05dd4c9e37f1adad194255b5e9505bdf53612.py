def build_prompt(context):
    parents = context.get('parent_ids', [])
    evidence = context.get('evidence_ids', [])
    slot = context.get('slot', 0)
    focused = []
    for item in evidence:
        if item == 'native':
            focused = ['native']
            break
    if not focused:
        focused = evidence[:1]
    if slot % 2 == 1 and parents:
        return str({'operator': 'mutate', 'parents': parents[:1], 'evidence': focused,
                    'rule': 0, 'guidance': 'Mutate the retained parent while prioritizing the native training feedback; preserve required allocate/place, eviction and reclaim coverage, and compare demand/replica handling against the parent.'})
    return str({'operator': 'construct', 'parents': [], 'evidence': focused, 'rule': 0,
                'guidance': 'Construct with native-focused evidence; keep required allocate/place and reclaim rules, and test a simpler demand/replica policy with explicit capacity fallback.'})

def build_meta_prompt(context):
    evidence = context.get('evidence_ids', [])
    focused = []
    for item in evidence:
        if item == 'native':
            focused = ['native']
            break
    if not focused:
        focused = evidence[:1]
    return str({'mode': 'repair', 'evidence': focused,
                'guidance': 'Repair the current search using the native training feedback: keep the construct/mutate structure, narrow evidence to native, and focus revisions on demand/replica and placement coverage rather than broad re-exploration.'})
