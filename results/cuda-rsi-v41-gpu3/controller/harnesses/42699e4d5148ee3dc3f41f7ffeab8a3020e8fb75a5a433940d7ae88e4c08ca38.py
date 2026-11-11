def build_prompt(context):
    parents = context["parent_ids"]
    evidence = context["evidence_ids"]
    slot = context["slot"]
    if parents and slot % 2 == 1:
        return str({"operator": "mutate", "parents": parents[:1], "evidence": evidence[:3],
                    "rule": 0, "guidance": "Use measured failures to revise the first allocation rule."})
    return str({"operator": "construct", "parents": [], "evidence": evidence[:3], "rule": 0,
                "guidance": "Try a topology-aware rule with a fallback and a specific transfer hypothesis."})

def build_meta_prompt(context):
    return str({"mode": "revise", "evidence": context["evidence_ids"][:4],
                "guidance": "Use the past training evidence to improve both future search and self-revision."})
