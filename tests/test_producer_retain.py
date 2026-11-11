"""Differential tests against the archived v11 allocation-replica path."""
from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from itertools import product
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from zipfile import ZipFile

from integrations.topology_memory import torch_runtime as NEW
from rsi_memory.compiled_memory import CompiledProgram
from rsi_memory.memory_program import MemoryProgram
from rsi_memory.topology import Link, Route


REPO = Path(__file__).resolve().parents[1]
SOURCE = REPO / "results/engineering-producer-v13/inputs/v11.source.zip"
MEMBER = "integrations/topology_memory/torch_runtime.py"
BASELINE_SHA256 = "ada57d5b110fd1090cde60732b254e8e03dab4859ef87d865e315182e278761c"


def load_baseline():
    with ZipFile(SOURCE) as package:
        raw = package.read(MEMBER)
    if sha256(raw).hexdigest() != BASELINE_SHA256:
        raise ValueError("archived v11 runtime differs from the qualified baseline")
    name = "integrations.topology_memory._retain_v11"
    spec = spec_from_file_location(name, str(SOURCE) + "!/" + MEMBER)
    module = module_from_spec(spec)
    sys.modules[name] = module
    try:
        exec(compile(raw, spec.origin, "exec"), module.__dict__)
    finally:
        del sys.modules[name]
    return module


OLD = load_baseline()
GROUP = "allocate", "replicate"
LOCAL = "gpu3"
HOST = "host0"
NODES = (LOCAL, HOST)


def const(value):
    return {"op": "const", "value": value}


def feature(name):
    return {"op": "feature", "name": name}


def rule(when, score, event="allocate", action="replicate"):
    return {"event": event, "action": action, "when": when, "score": score}


def compiled(*rules):
    place = rule(feature("target_is_host"), const(1), action="place")
    return CompiledProgram(MemoryProgram.from_dict({"schema": 1, "kind": "kv_memory_program", "rules": [*rules, place]}))


def runtime(module, rules, *, free=True, counters=False):
    engine = module._Runtime.__new__(module._Runtime)
    engine.program = compiled(*rules)
    engine.local = LOCAL
    engine.ordinal = 7
    engine.rule_routes = {}
    engine.rule_features = {}
    engine.static_routes = {}
    engine.timeline = []
    engine.transfers = []
    engine.pools = {node: SimpleNamespace(free=[0] if free else []) for node in NODES}
    engine._reserve = lambda state, route, mode, rule: (route.nodes, route.ids, mode, rule)
    engine._execute = engine.transfers.extend
    for source, target in product(NODES, repeat=2):
        link = Link(f"{source}-{target}", source, target, 10000, 1, (), "pinned_host")
        route = Route((source,), ()) if source == target else Route((source, target), (link,))
        for group, names in engine.program.features.items():
            static = {"target_is_local": int(target == LOCAL), "target_is_host": int(target == HOST)}
            dynamic = names & module._DYNAMIC_FEATURES
            engine.rule_routes[group, source, target] = ()
            if engine.program.possibly_eligible(group, static, module._DYNAMIC_FEATURES):
                readers = tuple((name, module._FEATURE_READERS[name]) for name in sorted(dynamic))
                engine.rule_routes[group, source, target] = (route,)
                engine.rule_features[group, route] = (static, readers, 128)
    if module is NEW:
        engine.producer_retain = NEW._lower_producer_retain(engine.program, LOCAL, engine.rule_routes, engine.rule_features)
    engine.work_counts = module.WorkCounts() if counters else None
    if counters:
        engine.work_counts.instrument(engine.program)
    return engine


def state(has_local=True, reads=3):
    copies = {HOST: 0, LOCAL: 0} if has_local else {HOST: 0}
    return SimpleNamespace(owner=HOST, copies=copies, reads=reads, spec=SimpleNamespace(id="block"))


LRU_RETAIN = (rule(feature("target_is_local"), const(1)),)
OVERFLOW = {"op": "mul", "args": [const(1e200), const(1e200)]}


class ProducerRetain(unittest.TestCase):
    def compare(self, rules, *, free=True, has_local=True, retain_source=True, event="allocate", expect_fast=True):
        old, new = (runtime(module, rules, free=free, counters=True) for module in (OLD, NEW))
        self.assertEqual(new.producer_retain is not None, expect_fast)
        outcomes = []
        for engine in (old, new):
            try:
                outcomes.append(engine._optional(event, state(has_local), retain_source=retain_source))
            except ValueError as error:
                outcomes.append((type(error), error.args))
        self.assertEqual(outcomes[0], outcomes[1])
        self.assertEqual(old.timeline, new.timeline)
        self.assertEqual(old.transfers, new.transfers)
        if expect_fast and has_local and retain_source and event == "allocate":
            self.assertEqual(new.work_counts.report()["allocate/replicate"], {"static_producer_retain_choices": 1})
        else:
            self.assertEqual(old.work_counts.report(), new.work_counts.report())
        return old.timeline, old.transfers, outcomes[0]

    def test_lru_retain_pressure_and_fallback(self):
        for free, has_local, retain_source in product((False, True), repeat=3):
            with self.subTest(free=free, has_local=has_local, retain_source=retain_source):
                self.compare(LRU_RETAIN, free=free, has_local=has_local, retain_source=retain_source)

    def test_first_matching_rule_before_score_and_later_error(self):
        rules = (rule(feature("target_is_host"), const(0), event="access"),
                 rule(const(0), OVERFLOW), rule(feature("target_is_local"), const(1)),
                 rule(feature("target_is_local"), OVERFLOW))
        timeline, _, success = self.compare(rules)
        self.assertTrue(success)
        self.assertEqual(timeline[0]["rule"], 2)

    def test_numeric_error_stays_deferred_to_candidate(self):
        rules = (rule(feature("target_is_local"), OVERFLOW), *LRU_RETAIN)
        _, _, error = self.compare(rules, expect_fast=False)
        self.assertIs(error[0], ValueError)
        _, _, result = self.compare(rules, expect_fast=False, retain_source=False)
        self.assertFalse(result)

    def test_dynamic_predicate_or_score_falls_back(self):
        self.compare((rule(feature("target_is_local"), feature("accesses")),), expect_fast=False)
        self.compare((rule(feature("accesses"), const(1)),), expect_fast=False)

    def test_competing_routes_and_other_actions_fall_back(self):
        self.compare((rule(const(1), const(1)),), expect_fast=False)
        self.compare((*LRU_RETAIN, rule(feature("target_is_local"), const(1), action="prefetch")),
                     expect_fast=False)
        # An access choice still runs through the ordinary path.
        self.compare((*LRU_RETAIN, rule(feature("target_is_host"), const(1), event="access")),
                     event="access")


if __name__ == "__main__":
    unittest.main()
