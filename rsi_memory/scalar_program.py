"""Bounded scalar AST validation and evaluation for topology memory rules."""

import math
import operator


MAX_NODES = 64
MAX_DEPTH = 8
DIV_EPSILON = 1e-6
_BINARY = {"add": operator.add, "sub": operator.sub, "mul": operator.mul, "min": min, "max": max}
_ARITY = {"add": 2, "sub": 2, "mul": 2, "div": 2, "min": 2, "max": 2, "log1p": 1, "if_lt": 4}


class _FrozenAST(dict):
    """Immutable, JSON-compatible nodes compatible with dataclasses.asdict."""

    def _deny_mutation(self, *args, **kwargs):
        raise TypeError("validated AST nodes are immutable")

    __setitem__ = __delitem__ = __ior__ = clear = pop = popitem = setdefault = update = _deny_mutation


def _finite(value):
    if type(value) not in (float, int):
        raise ValueError("program values must be finite real numbers")
    try:
        value = float(value)
    except OverflowError as exc:
        raise ValueError("program values must be finite real numbers") from exc
    if not math.isfinite(value):
        raise ValueError("program values must be finite real numbers")
    return value


def validate_program(program, allowed_features):
    """Validate the complete AST and return an immutable JSON-compatible copy."""
    allowed_features = frozenset(allowed_features)
    count = 0

    def visit(node, depth):
        nonlocal count
        count += 1
        if depth > MAX_DEPTH or count > MAX_NODES:
            raise ValueError(f"program exceeds depth {MAX_DEPTH} or node count {MAX_NODES}")
        if type(node) not in (dict, _FrozenAST) or type(node.get("op")) is not str:
            raise ValueError("each program node must be an object with a string op")
        op = node["op"]
        if op == "const" and set(node) == {"op", "value"}:
            return _FrozenAST(op=op, value=_finite(node["value"]))
        if op == "feature" and set(node) == {"op", "name"}:
            name = node["name"]
            if type(name) is not str or name not in allowed_features:
                raise ValueError(f"unsupported program feature: {name!r}")
            return _FrozenAST(op=op, name=name)
        if op not in _ARITY or set(node) != {"op", "args"}:
            raise ValueError(f"unsupported op or unexpected node fields: {op!r}")
        args = node["args"]
        if type(args) not in (list, tuple) or len(args) != _ARITY[op]:
            raise ValueError(f"{op} requires {_ARITY[op]} operands")
        return _FrozenAST(op=op, args=tuple(visit(arg, depth + 1) for arg in args))

    return visit(program, 1)


def _safe_div(left, right):
    return left / math.copysign(max(abs(right), DIV_EPSILON), right)


def evaluate_scalar(node, features):
    op = node["op"]
    if op == "const":
        return node["value"]
    if op == "feature":
        return _finite(features[node["name"]])
    args = node["args"]
    left = evaluate_scalar(args[0], features)
    if op == "log1p":
        return math.log1p(abs(left))
    right = evaluate_scalar(args[1], features)
    if op == "if_lt":
        return evaluate_scalar(args[2] if left < right else args[3], features)
    if op == "div":
        return _finite(_safe_div(left, right))
    return _finite(_BINARY[op](left, right))
