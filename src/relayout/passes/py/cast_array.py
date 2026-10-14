# Copyright 2026 ETH Zurich and the relayout authors.
"""``cast``: an array of a Python function in another element type, the counterpart of
:mod:`relayout.passes.sdfg.cast_array`. A local is cast in place at its definition; an argument keeps its array
for the caller, and the function works on a local ``<name>_<element>`` converted in at the entry and out at the
exit (numpy casts on each assignment).
"""

import ast
from collections.abc import Mapping

from relayout.layout import IndexMap
from relayout.ops import Precision
from relayout.passes.py.common import (
    Array,
    LayoutError,
    allocation,
    arguments,
    definition,
    ensure_numpy,
    kernel,
    parse,
    statement,
    tuple_text,
)
from relayout.passes.py.placement import Laid
from relayout.passes.py.rewrite_accesses import move_array

__all__ = ["PRECISION_TYPES", "cast_array"]

#: The element type a ``cast`` asks for, as numpy has it.
PRECISION_TYPES = {
    Precision.FP64: "float64",
    Precision.FP32: "float32",
    Precision.FP16: "float16",
}


def cast_array(
    tree: ast.Module, name: str, precision: Precision, arrays: Mapping[str, Array], function: str | None = None
) -> str:
    """Array ``name`` of ``tree``'s function (shape and element type in ``arrays``) in the element type of
    ``precision``; returns the name the function now works on."""
    target = PRECISION_TYPES.get(precision)
    if target is None or name not in arrays:
        raise LayoutError(f"cast of {name!r} to {precision.value}: no such array, or no such element type in numpy")
    spec = arrays[name]
    if spec.element == target:
        return name
    func = kernel(tree, function)
    ensure_numpy(tree)
    dtype = f"np.{target}"
    if name not in arguments(func):
        body, index = definition(func, name)
        stmt = body[index]
        assert isinstance(stmt, ast.Assign)
        alloc = allocation(stmt.value)
        if alloc is not None and isinstance(stmt.value, ast.Call):
            stmt.value.keywords = [k for k in stmt.value.keywords if k.arg != "dtype"]
            positional = 2 if alloc.kind == "full" else 1
            stmt.value.args = stmt.value.args[:positional]
            stmt.value.keywords.append(ast.keyword("dtype", parse(dtype)))
        else:
            stmt.value = parse(f"({ast.unparse(stmt.value)}).astype({dtype})")
        return name
    cast = f"{name}_{target}"

    def allocate(_: str) -> list[ast.stmt]:
        return [statement(f"{cast} = np.empty({tuple_text(spec.shape)}, dtype={dtype})")]

    move_array(func, name, cast, Laid(IndexMap.identity(spec.shape)), spec.shape, allocate)
    return cast
