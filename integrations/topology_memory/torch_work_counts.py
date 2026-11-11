"""Optional work counters for verified CPU/GPU memory policy diagnostics."""

from collections import Counter, defaultdict


class WorkCounts:
    def __init__(self):
        self.groups = defaultdict(Counter)

    def add(self, group, item, value):
        self.groups[group][item] += value

    def instrument(self, program):
        choose = program.choose

        def count_choice(event, action, candidates):
            counters = self.groups[event, action]
            counters["choice_calls"] += 1
            counters["candidate_records"] += len(candidates)
            return choose(event, action, candidates)

        def counted(fn, counters, item):
            def run(features):
                counters[item] += 1
                return fn(features)
            return run

        for group, rules in program.rules.items():
            counters = self.groups[group]
            program.rules[group] = [(index, counted(when, counters, "predicate_evals"), counted(score, counters, "score_evals"))
                                    for index, when, score in rules]
        program.choose = count_choice

    def report(self):
        return {f"{event}/{action}": dict(values) for (event, action), values in sorted(self.groups.items()) if values}
