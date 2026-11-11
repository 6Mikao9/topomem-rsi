"""Precompute route-static priorities while retaining deferred numeric errors."""

from dataclasses import dataclass


_GROUPS = (("allocate", "place"), ("access", "replicate"))


@dataclass(frozen=True)
class _RouteOutcome:
    rule: int
    order: int
    route: object
    required: tuple
    score: float | None = None
    error_args: tuple | None = None

    def available(self, copies, sources, targets, allow_local, blocked):
        source, target = self.route.nodes[0], self.route.nodes[-1]
        if source not in sources or target not in targets:
            return False
        if source == target and not allow_local or source != target and target in copies:
            return False
        for node, pool in self.required:
            if not pool.free:
                if blocked is not None:
                    blocked.update(name for name, item in self.required if not item.free)
                return False
        return True


@dataclass(frozen=True)
class StaticRoutes:
    choices: tuple[_RouteOutcome, ...]
    errors: tuple[_RouteOutcome, ...]
    destinations: frozenset[str]

    def choose(self, copies, sources, targets, *, allow_local=False, blocked=None):
        if not allow_local and self.destinations.issubset(copies):
            return None
        winner = next((row for row in self.choices if row.available(copies, sources, targets, allow_local, blocked)), None)
        # An error in the winning rule still matters, even after another candidate qualified.
        for error in self.errors:
            if winner is not None and error.rule > winner.rule:
                break
            if error.available(copies, sources, targets, allow_local, blocked):
                raise ValueError(*error.error_args)
        return None if winner is None else (winner.rule, winner.route)


def lower_static_routes(program, rule_routes, route_features, dynamic_features, pools):
    lowered = {}
    for group in _GROUPS:
        if group not in program.rules or program.features[group] & dynamic_features:
            continue
        choices, errors = [], []
        routes = (route for source in sorted(pools) for target in sorted(pools) for route in rule_routes[group, source, target])
        for order, route in enumerate(routes):
            static, _, _ = route_features[group, route]
            required = tuple((node, pools[node]) for node in route.nodes[1:])
            # Later rules cannot be reached for this candidate once it succeeds or fails.
            for index, when, score in program.rules[group]:
                try:
                    if when(static) <= 0:
                        continue
                    value = score(static)
                except ValueError as exc:
                    errors.append(_RouteOutcome(index, order, route, required, error_args=exc.args))
                else:
                    choices.append(_RouteOutcome(index, order, route, required, score=value))
                break
        destinations = frozenset(row.route.nodes[-1] for row in (*choices, *errors))
        lowered[group] = StaticRoutes(tuple(sorted(choices, key=lambda row: (row.rule, row.score, row.order))),
                                      tuple(sorted(errors, key=lambda row: (row.rule, row.order))), destinations)
    return lowered
