"""Conservative state-key lowering for stable pressure minima."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class PressureKey:
    index: int
    value: object


def _score(node, upper):
    op = node["op"]
    if op == "const":
        value = node["value"]
        return (lambda runtime, state: value), value, True
    if op == "feature":
        name = node["name"]
        if name == "last_use_step":
            return (lambda runtime, state: float(state.last_use)), upper[name], False
        if name == "page_accesses":
            return (lambda runtime, state: float(runtime.reads[state.spec.layer, state.spec.index])), upper[name], False
        return None
    if op not in ("add", "mul"):
        return None
    left, right = (_score(child, upper) for child in node["args"])
    if left is None or right is None or left[1] < 0 or right[1] < 0 or op == "mul" and not (left[2] or right[2]):
        return None
    # Scores retain their actual rounded float keys. Equal rounded values still tie by source order.
    if op == "add":
        value = lambda runtime, state: left[0](runtime, state) + right[0](runtime, state)
        bound = left[1] + right[1]
    else:
        value = lambda runtime, state: left[0](runtime, state) * right[0](runtime, state)
        bound = left[1] * right[1]
    return (value, bound, left[2] and right[2]) if math.isfinite(bound) else None


def lower_pressure_keys(source, compiled, route_features, dynamic_features, workload):
    upper = {"last_use_step": float(len(workload.steps)),
             "page_accesses": float(sum(len(request.selected) for step in workload.steps for request in step.attention))}
    if any(bound >= 2**53 for bound in upper.values()):
        return {}
    lowered = {}
    for group, rules in compiled.rules.items():
        if group[1] not in ("evict", "reclaim", "demote") or len(rules) != 1:
            continue
        names, when = compiled.predicates[group][0]
        if names & dynamic_features:
            continue
        index = rules[0][0]
        score = _score(source.rules[index].score, upper)
        if score is None:
            continue
        try:
            for (candidate_group, _), (static, _, _) in route_features.items():
                if candidate_group == group and when(static) <= 0:
                    raise ValueError("static route did not satisfy its predicate")
        except ValueError:
            # A possible runtime arithmetic failure belongs to the general evaluator.
            continue
        lowered[group] = PressureKey(index, score[0])
    return lowered
