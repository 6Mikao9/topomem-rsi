"""Compile bounded, validated scalar expressions to fixed Python bytecode."""

import ast
from collections import defaultdict
import math

from .memory_program import FEATURE_UNITS
from .scalar_program import _finite, _safe_div, validate_program


_BINARY = {"add": ast.Add, "sub": ast.Sub, "mul": ast.Mult}
_GLOBALS = {"__builtins__": {}, "finite": _finite, "divide": _safe_div,
            "log1p": math.log1p, "abs": abs, "min": min, "max": max}


def compile_scalar(program):
    """Construct bytecode from nodes only, with the interpreter's numeric semantics."""
    program = validate_program(program, FEATURE_UNITS)

    def call(name, *args):
        return ast.Call(func=ast.Name(id=name, ctx=ast.Load()), args=list(args), keywords=[])

    def emit(node):
        op = node["op"]
        if op == "const":
            return ast.Constant(value=node["value"])
        if op == "feature":
            return call("finite", ast.Subscript(value=ast.Name(id="f", ctx=ast.Load()),
                                                slice=ast.Constant(value=node["name"]), ctx=ast.Load()))
        args = [emit(child) for child in node["args"]]
        if op == "if_lt":
            return ast.IfExp(test=ast.Compare(left=args[0], ops=[ast.Lt()], comparators=[args[1]]),
                             body=args[2], orelse=args[3])
        if op == "log1p":
            return call("log1p", call("abs", args[0]))
        if op in _BINARY:
            result = ast.BinOp(left=args[0], op=_BINARY[op](), right=args[1])
        else:
            result = call("divide" if op == "div" else op, *args)
        return call("finite", result)

    arguments = ast.arguments(posonlyargs=[], args=[ast.arg(arg="f")], vararg=None,
                              kwonlyargs=[], kw_defaults=[], kwarg=None, defaults=[])
    expression = ast.Expression(body=ast.Lambda(args=arguments, body=emit(program)))
    return eval(compile(ast.fix_missing_locations(expression), "<validated-memory-ast>", "eval"), _GLOBALS)


class CompiledProgram:
    def __init__(self, program):
        self.rules = defaultdict(list)
        self.features = defaultdict(set)
        self.predicates = defaultdict(list)
        for index, rule in enumerate(program.rules):
            group = rule.event, rule.action
            when, score = compile_scalar(rule.when), compile_scalar(rule.score)
            observed = _features(rule.when)
            self.rules[group].append((index, when, score))
            self.features[group].update(observed | _features(rule.score))
            self.predicates[group].append((observed, when))

    def possibly_eligible(self, group, static, dynamic_features):
        for features, when in self.predicates[group]:
            if features & dynamic_features:
                return True
            try:
                if when(static) > 0:
                    return True
            except ValueError:
                # Keep normal runtime error semantics for a route that is actually used.
                return True
        return False

    def choose(self, event, action, candidates):
        if not isinstance(candidates, (list, tuple)):
            candidates = tuple(candidates)
        for index, when, score in self.rules.get((event, action), ()):
            best = None
            for candidate, features in candidates:
                if when(features) <= 0:
                    continue
                value = score(features)
                if best is None or value < best[0]:
                    best = value, candidate
            if best is not None:
                return index, best[1]
        return None


def _features(node):
    if node["op"] == "feature":
        return {node["name"]}
    return set().union(*(_features(child) for child in node.get("args", ())))
