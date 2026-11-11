def build_prompt(context):
    parents = context.get('parent_ids', [])
    evidence = context.get('evidence_ids', [])
    slot = context.get('slot', 0)
    ordered = list(reversed(evidence[:4]))
    if slot % 2 == 0:
        return str({'operator': 'construct', 'parents': [], 'evidence': ordered, 'rule': 0, 'guidance': 'Construct a complete proof-first program from the measured startup evidence. Prefer a qualified resident backing choice only when the fixed prompt plus decode bound fits the native page-reserved hot pool; otherwise keep a valid tiered program with measured hot capacity and a safe fallback. Do not rely on cold allocation when resident proof holds.'})
    if parents:
        return str({'operator': 'mutate', 'parents': parents[-1:], 'evidence': ordered, 'rule': 0, 'guidance': 'Mutate the selected parent at rule 0 toward the measured hot-capacity or backing-mode decision. Remove cold-backup dependence only with a valid resident proof; otherwise retune one hot-capacity rule and preserve the selected family and all byte and route checks.'})
    return str({'operator': 'construct', 'parents': [], 'evidence': ordered, 'rule': 0, 'guidance': 'Construct a proof-first candidate: test a qualified resident startup if the fixed bound fits, or a tiered program with measured hot capacity and safe fallback.'})

def build_meta_prompt(context):
    evidence = context.get('evidence_ids', [])
    ordered = list(reversed(evidence[:4]))
    round_no = context.get('round', 0)
    if round_no == 0:
        mode = 'rethink'
    else:
        mode = 'revise'
    return str({'mode': mode, 'evidence': ordered, 'guidance': 'Self-revise by comparing the incumbent and native training evidence. Keep the best measured training program, then use construct to test a proof-qualified resident or hot-capacity strategy and use mutate only to repair one rule without breaking the selected program family.'})
