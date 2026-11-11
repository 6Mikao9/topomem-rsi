def build_prompt(context):
    parents = context.get('parent_ids', [])
    evidence = context.get('evidence_ids', [])
    slot = context.get('slot', 0)
    ordered = []
    for e in evidence:
        if e == 'native':
            ordered.append(e)
    for e in evidence:
        if e != 'native':
            ordered.append(e)
    if len(ordered) > 4:
        ordered = ordered[:4]
    if len(ordered) == 0:
        ordered = ['native']
    guidance = 'Training evidence: native 4096 all-host tiered is the best measured control (1075.58 ms); bootstrap incumbent is 1086.58 ms. Resident 6144 is explicitly eligible and is the predeclared manual control, with 290816 extra metadata bytes per GPU. Construct or repair a configure-only candidate: set backing_mode to resident when resident_hot_slot_margin is nonnegative and the case proves eligibility; set hot_capacity to 6144 when candidate_hot_slots 6144 has candidate_min_remaining_cold_bytes >= 0 and resident_hot_slot_margin >= 0. Do not add allocate/place rules in resident mode. Keep a tiered 4096 native-equivalent fallback only if resident proof fails. Lower score wins in the first eligible rule.'
    if slot % 2 == 0:
        return str({'operator': 'construct', 'parents': [], 'evidence': ordered, 'rule': 0, 'guidance': guidance})
    if parents:
        return str({'operator': 'mutate', 'parents': parents[:1], 'evidence': ordered, 'rule': 0, 'guidance': guidance})
    return str({'operator': 'construct', 'parents': [], 'evidence': ordered, 'rule': 0, 'guidance': guidance})

def build_meta_prompt(context):
    evidence = context.get('evidence_ids', [])
    ordered = []
    for e in evidence:
        if e == 'native':
            ordered.append(e)
    for e in evidence:
        if e != 'native':
            ordered.append(e)
    if len(ordered) > 4:
        ordered = ordered[:4]
    if len(ordered) == 0:
        ordered = ['native']
    return str({'mode': 'rethink', 'evidence': ordered, 'guidance': 'Self-revision: compare the current task decision against the measured controls. If the selected parent or rule does not target the resident 6144 configure path, switch to construct or mutate with native-first evidence and rule 0. Prefer a configure-only resident program with hot_capacity 6144 when the case proves resident eligibility and a nonnegative resident_hot_slot_margin; otherwise keep the native 4096 tiered control. Reject guidance that merely restates the current harness or embeds a fixed manager answer.'})