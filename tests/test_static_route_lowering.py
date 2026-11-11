from itertools import product
from types import SimpleNamespace
import unittest

from integrations.topology_memory.torch_route_lowering import lower_static_routes
from rsi_memory.compiled_memory import CompiledProgram
from rsi_memory.memory_program import MemoryProgram
from test_torch_workload import topology


def const(value):
    return {"op": "const", "value": value}


def feature(name):
    return {"op": "feature", "name": name}


def rule(event, action, when, score):
    return {"event": event, "action": action, "when": when, "score": score}


def adapter(rules):
    program = CompiledProgram(MemoryProgram.from_dict({"schema": 1, "kind": "kv_memory_program", "rules": rules}))
    graph = topology(4096)
    pools = {node.id: SimpleNamespace(free=[0]) for node in graph.nodes}
    dynamic = {"page_accesses"}
    routes, fields = {}, {}
    for group in program.rules:
        for src, dst in product(pools, repeat=2):
            eligible = []
            for route in graph.routes(src, dst):
                observed = {"target_is_local": int(dst == "gpu0"), "target_is_host": int(dst == "host"),
                            "target_capacity_bytes": graph.node(dst).capacity_bytes,
                            "route_hops": len(route.links)}
                if program.possibly_eligible(group, observed, dynamic):
                    fields[group, route] = observed, (), 0
                    eligible.append(route)
            routes[group, src, dst] = tuple(eligible)
    lowered = lower_static_routes(program, routes, fields, dynamic, pools)

    def oracle(group, copies, targets, *, allow_local=False, blocked=None):
        candidates = []
        for src in sorted(copies):
            for dst in sorted(targets):
                if (src == dst and not allow_local) or (dst in copies and src != dst):
                    continue
                for route in routes[group, src, dst]:
                    full = {node for node in route.nodes[1:] if not pools[node].free}
                    if full:
                        if blocked is not None:
                            blocked.update(full)
                        continue
                    candidates.append((route, fields[group, route][0]))
        return program.choose(*group, candidates)

    return lowered, pools, oracle


class StaticRouteLoweringTests(unittest.TestCase):
    def test_first_rule_and_candidate_order_match_all_small_capacity_states(self):
        rules = [rule("allocate", "place", feature("target_is_host"), const(1)),
                 rule("allocate", "place", const(1), const(0)),
                 rule("access", "replicate", feature("target_is_local"), const(1)),
                 rule("access", "replicate", const(1), const(0))]
        lowered, pools, oracle = adapter(rules)
        nodes = list(pools)
        sets = [{node for bit, node in enumerate(nodes) if mask & (1 << bit)} for mask in range(1, 8)]
        for group, capacity, copies, targets in product(lowered, range(8), sets, sets):
            for bit, node in enumerate(nodes):
                pools[node].free = [0] if capacity & (1 << bit) else []
            allow_local = group[0] == "allocate"
            fast_blocked, slow_blocked = set(), set()
            fast = lowered[group].choose(copies, copies, targets, allow_local=allow_local, blocked=fast_blocked)
            slow = oracle(group, copies, targets, allow_local=allow_local, blocked=slow_blocked)
            self.assertEqual(fast, slow)
            if fast is None:
                self.assertEqual(fast_blocked, slow_blocked)

    def test_earlier_rule_or_blocked_route_defers_arithmetic_error(self):
        overflow = {"op": "mul", "args": [feature("target_capacity_bytes"), const(1e308)]}
        peer_error = {"op": "if_lt", "args": [feature("target_is_host"), const(1), overflow, const(0)]}
        rules = [rule("allocate", "place", feature("target_is_local"), const(0)),
                 rule("allocate", "place", const(1), peer_error),
                 rule("access", "replicate", feature("page_accesses"), const(0))]
        lowered, pools, oracle = adapter(rules)
        group, copies = ("allocate", "place"), {"gpu0"}
        self.assertNotIn(("access", "replicate"), lowered)
        self.assertEqual(lowered[group].choose(copies, copies, pools, allow_local=True),
                         oracle(group, copies, pools, allow_local=True))
        with self.assertRaisesRegex(ValueError, "finite real"):
            lowered[group].choose(copies, copies, pools)
        with self.assertRaisesRegex(ValueError, "finite real"):
            oracle(group, copies, pools)
        pools["gpu1"].free = []
        self.assertEqual(lowered[group].choose(copies, copies, pools), oracle(group, copies, pools))
