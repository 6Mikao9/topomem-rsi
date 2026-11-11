"""Run versionable prompt builders with an AST allowlist and an isolated process.

The allowlist limits what Python syntax can do. The process boundary bounds wall
time and avoids inheriting secrets. It is not a general-purpose security sandbox.
"""

import ast
import json
import math
import os
import subprocess
import sys


MAX_SOURCE_CHARS = 24_000
MAX_PROMPT_CHARS = 32_000
MAX_OUTPUT_BYTES = 64 * 1024
MAX_INPUT_BYTES = 1024 * 1024

_FUNCTIONS = {"build_prompt", "build_meta_prompt"}
_BUILTINS = {
    "abs", "dict", "enumerate", "len", "list", "max", "min", "range",
    "reversed", "round", "sorted", "str", "sum", "zip",
}
_METHODS = {"append", "get", "items", "join", "keys", "values"}
_NODES = {
    ast.Assign, ast.AugAssign, ast.BinOp, ast.BoolOp, ast.Break, ast.Compare,
    ast.Continue, ast.Dict, ast.DictComp, ast.Expr, ast.For, ast.FormattedValue,
    ast.GeneratorExp, ast.If, ast.IfExp, ast.JoinedStr, ast.List, ast.ListComp,
    ast.Load, ast.Pass, ast.Return, ast.Set, ast.SetComp, ast.Slice, ast.Store,
    ast.Subscript, ast.Tuple, ast.UnaryOp, ast.comprehension,
    ast.Add, ast.Sub, ast.Mult, ast.Div, ast.FloorDiv, ast.Mod, ast.Pow,
    ast.And, ast.Or, ast.Not, ast.UAdd, ast.USub,
    ast.Eq, ast.NotEq, ast.Lt, ast.LtE, ast.Gt, ast.GtE, ast.In, ast.NotIn,
    ast.Is, ast.IsNot,
}


SEED_HARNESS_SOURCE = r'''
def build_prompt(context):
    return (
        "Propose one manager program for the stated sparse attention and device. "
        "Use the current manager archive and measured training feedback to choose "
        "a useful new experiment. Explain the expected effect in the requested response.\n"
        + str(context)[:30000]
    )

def build_meta_prompt(context):
    source = context.get("harness_source", "")
    evidence = {key: value for key, value in context.items() if key != "harness_source"}
    return (
        "Rewrite this prompt harness to improve how future experiments and "
        "harness revisions are proposed. Use the training evidence and preserve both "
        "entry points.\nCurrent harness source:\n" + source
        + "\nTraining evidence:\n" + str(evidence)[:max(0, 30000 - len(source))]
    )
'''


class _BodyValidator(ast.NodeVisitor):
    def generic_visit(self, node):
        if type(node) not in _NODES:
            raise ValueError(f"harness syntax is not allowed: {type(node).__name__}")
        super().generic_visit(node)

    def visit_Name(self, node):
        if "__" in node.id or node.id in _BUILTINS or node.id in _FUNCTIONS:
            raise ValueError(f"harness name is not allowed: {node.id}")

    def visit_Constant(self, node):
        if type(node.value) not in (str, int, float, bool, type(None)):
            raise ValueError("harness literal type is not allowed")

    def visit_Call(self, node):
        target = node.func
        if isinstance(target, ast.Name) and target.id in _BUILTINS:
            pass
        elif isinstance(target, ast.Attribute) and target.attr in _METHODS:
            self.visit(target.value)
        else:
            raise ValueError("harness calls must use an allowed builtin or data method")
        for arg in node.args:
            self.visit(arg)
        for keyword in node.keywords:
            if keyword.arg is None or "__" in keyword.arg:
                raise ValueError("harness call keyword is not allowed")
            self.visit(keyword.value)


def validate_harness(source):
    """Raise ValueError unless source defines only the two approved entry points."""
    if not isinstance(source, str) or not source or len(source) > MAX_SOURCE_CHARS:
        raise ValueError(f"harness source must have 1 to {MAX_SOURCE_CHARS} characters")
    try:
        tree = ast.parse(source)
    except (SyntaxError, ValueError, RecursionError):
        raise ValueError("invalid harness Python syntax") from None

    if len(tree.body) != 2 or any(type(node) is not ast.FunctionDef for node in tree.body):
        raise ValueError("harness must define only build_prompt and build_meta_prompt")
    if {node.name for node in tree.body} != _FUNCTIONS:
        raise ValueError("harness must define both build_prompt and build_meta_prompt")

    validator = _BodyValidator()
    for function in tree.body:
        args = function.args
        if (
            function.decorator_list or function.returns or function.type_comment
            or getattr(function, "type_params", ()) or args.posonlyargs or args.vararg
            or args.kwonlyargs or args.kwarg or args.defaults or len(args.args) != 1
            or args.args[0].arg != "context" or args.args[0].annotation
        ):
            raise ValueError(f"harness function signature must be {function.name}(context)")
        try:
            for statement in function.body:
                validator.visit(statement)
        except RecursionError:
            raise ValueError("harness syntax is nested too deeply") from None
    try:
        compile(tree, "<harness>", "exec")
    except (SyntaxError, ValueError, RecursionError):
        raise ValueError("invalid harness Python syntax") from None


_WORKER = '''\
import builtins
import json
import sys

try:
    import resource

    limit = 256 * 1024 * 1024
    soft, hard = resource.getrlimit(resource.RLIMIT_AS)
    if soft != resource.RLIM_INFINITY:
        limit = min(limit, soft)
    if hard != resource.RLIM_INFINITY:
        limit = min(limit, hard)
    resource.setrlimit(resource.RLIMIT_AS, (limit, hard))
except (ImportError, AttributeError, OSError, ValueError):
    pass

try:
    request = json.loads(sys.stdin.buffer.read())
    scope = {"__builtins__": {name: getattr(builtins, name) for name in sys.argv[1].split(",")}}
    exec(compile(request["source"], "<harness>", "exec"), scope)
    result = scope[request["function"]](request["context"])
    if type(result) is not str or len(result) > int(sys.argv[2]):
        response = {"error": "harness result must be a string of at most 32000 characters"}
    else:
        response = {"text": result}
    output = json.dumps(response, ensure_ascii=False).encode("utf-8")
    if len(output) > int(sys.argv[3]):
        output = b'{"error":"harness JSON output exceeds 64 KiB"}'
except Exception:
    output = b'{"error":"harness execution failed"}'

sys.stdout.buffer.write(output)
'''


def run_harness(source, function, context, timeout=5):
    """Return the requested user-message text from JSON-only input."""
    validate_harness(source)
    if not isinstance(function, str) or function not in _FUNCTIONS:
        raise ValueError("unknown harness function")
    if not isinstance(context, dict):
        raise ValueError("harness context must be a JSON object")
    if not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("harness timeout must be a finite positive number")
    try:
        request = json.dumps(
            {"source": source, "function": function, "context": context},
            ensure_ascii=True, allow_nan=False,
        ).encode("ascii")
    except (TypeError, ValueError, RecursionError):
        raise ValueError("harness context must contain only JSON data") from None
    if len(request) > MAX_INPUT_BYTES:
        raise ValueError("harness JSON input exceeds 1 MiB")

    env = {}
    if os.name == "nt" and "SystemRoot" in os.environ:
        env["SystemRoot"] = os.environ["SystemRoot"]
    try:
        process = subprocess.run(
            [sys.executable, "-I", "-S", "-B", "-c", _WORKER, ",".join(sorted(_BUILTINS)),
             str(MAX_PROMPT_CHARS), str(MAX_OUTPUT_BYTES)],
            input=request, capture_output=True, env=env, timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        raise TimeoutError("harness timed out") from None
    except OSError:
        raise ValueError("could not start the harness process") from None

    if process.returncode or len(process.stdout) > MAX_OUTPUT_BYTES:
        raise ValueError("harness process failed or exceeded the output limit")
    try:
        response = json.loads(process.stdout)
    except (ValueError, UnicodeError):
        raise ValueError("harness process returned invalid JSON") from None
    if "error" in response:
        raise ValueError(response["error"])
    return response["text"]
