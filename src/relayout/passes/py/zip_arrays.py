# Copyright 2026 ETH Zurich and the relayout authors.
"""``zip``: arrays of one shape fused into one new local, stacked along a new dimension (one element type) or
as a structured array with one member per array (different element types), the counterpart of
:mod:`relayout.passes.sdfg.zip_arrays`. Arguments keep their arrays for the caller and are converted in and out
(:func:`~relayout.passes.py.rewrite_accesses.move_array`); a local is moved in at its definition.
"""

import ast
from collections.abc import Mapping, Sequence

from relayout.passes.py.common import Array, LayoutError, add_entry, ensure_numpy, kernel, statement, tuple_text
from relayout.passes.py.pack_mpi_buffers import MpiBuffer, pack_mpi_buffers
from relayout.passes.py.placement import Field, Member, Placement
from relayout.passes.py.rewrite_accesses import move_array

__all__ = ["zip_arrays", "zipped_array"]


def zip_arrays(
    tree: ast.Module,
    fields: Sequence[str],
    into: str,
    arrays: Mapping[str, Array],
    dim: int | None = None,
    function: str | None = None,
    pending: list[MpiBuffer] | None = None,
) -> str:
    """Fuse arrays ``fields`` of ``tree``'s function (shapes and element types in ``arrays``) into the new local
    ``into``: of one element type, stacked along a new dimension at ``dim`` (None: last; ``into[..., k]`` is
    ``fields[k]``), the inverse of :func:`~relayout.passes.py.unzip_array.unzip_array`; of different ones, a
    structured array with one member per field, named after it (:func:`zipped_array`). Returns ``into``."""
    func = kernel(tree, function)
    ensure_numpy(tree)
    zipped, dtype, placements = zipped_array(fields, arrays, dim)
    own: list[MpiBuffer] = [] if pending is None else pending
    for field, placement in zip(fields, placements, strict=True):
        move_array(func, field, into, placement, arrays[field].shape, None, own)
    add_entry(func, [statement(f"{into} = np.zeros({tuple_text(zipped.shape)}, dtype={dtype})")])
    if pending is None:
        pack_mpi_buffers(func, own)
    return into


def zipped_array(
    fields: Sequence[str], arrays: Mapping[str, Array], dim: int | None
) -> tuple[Array, str, list[Placement]]:
    """The array a zip of ``fields`` makes, its element type as a numpy expression, and where each field lives
    in it."""
    specs = [arrays[field] for field in fields]
    shapes = {spec.shape for spec in specs}
    if len(shapes) != 1:
        raise LayoutError(f"zip of {list(fields)}: the arrays must share one shape")
    (shape,) = shapes
    elements = [spec.element for spec in specs]
    placements: list[Placement]
    if len(set(elements)) == 1 and elements[0] is not None:
        stacked = list(shape)
        stacked.insert(len(stacked) if dim is None else dim, str(len(fields)))
        zipped = Array(tuple(stacked), elements[0])
        dtype = f"np.dtype({elements[0]!r})"
        placements = [Field(k, dim) for k in range(len(fields))]
    else:
        if dim is not None:
            raise LayoutError(
                f"zip of {list(fields)}: arrays of different element types make a struct, which has no dim"
            )
        if any(element is None for element in elements):
            raise LayoutError(f"zip of {list(fields)}: a struct is no member of another struct")
        zipped = Array(shape, None)
        dtype = (
            "[" + ", ".join(f"({field!r}, {element!r})" for field, element in zip(fields, elements, strict=True)) + "]"
        )
        placements = [Member(field) for field in fields]
    return zipped, dtype, placements
