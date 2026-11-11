"""Expose live CUDA policy features only when a validated expression reads them."""


class DynamicFeatureView:
    """Bind a route and a state for one synchronous policy decision.

    The runtime leaves that state unchanged while it ranks the candidates.
    Feature readers are pure and total over valid runtime state. Missing keys
    remain errors only if an executed AST branch reads them.
    """

    __slots__ = ("_static", "_readers", "_arguments")

    def __init__(self, static, readers, arguments):
        self._static = static
        self._readers = readers
        self._arguments = arguments

    def __getitem__(self, name):
        reader = self._readers.get(name)
        if reader is not None:
            return reader(*self._arguments)
        return self._static[name]
