# Copyright 2026 ETH Zurich and the relayout authors.
"""A request's operations carried out on a Python function: ``cast``, ``zip`` and ``unzip`` in program order, then
every array a digit operation touched laid out
(:func:`~relayout.passes.py.relayout_array.relayout_array`) and the MPI buffers packed once all are; the
counterpart of :mod:`relayout.passes.sdfg.apply_requests`.
"""

import ast
from collections.abc import Mapping

from relayout.ops import ArraySpec, Cast, DigitOperation, Requests, Unzip, Zip
from relayout.passes.py.cast_array import PRECISION_TYPES, cast_array
from relayout.passes.py.common import Array, kernel
from relayout.passes.py.pack_mpi_buffers import MpiBuffer, pack_mpi_buffers
from relayout.passes.py.relayout_array import relayout_array
from relayout.passes.py.unzip_array import unzip_array
from relayout.passes.py.zip_arrays import zip_arrays, zipped_array

__all__ = ["apply_requests"]


def apply_requests(
    tree: ast.Module, requests: Requests, arrays: Mapping[str, Array], function: str | None = None
) -> dict[str, str]:
    """Carry ``requests``' operations out on ``tree``'s function (single node; its arrays' logical shapes and
    element types in ``arrays``): ``cast`` (:func:`~relayout.passes.py.cast_array.cast_array`), ``zip`` and
    ``unzip`` in program order, so a cast before a zip makes a struct of the cast types; then every array a digit
    operation touched is laid out as the operations leave it. Returns the name each requested array is now worked
    on."""
    known = dict(arrays)
    state = requests.run([ArraySpec(name, len(a.shape), element=a.element) for name, a in known.items()])
    working: dict[str, str] = {}
    stacked: dict[str, int] = {}  # a zipped array's field dimension, which its unzip splits along by default
    for op in requests.operations:
        if isinstance(op, Cast):
            source = working.get(op.array, op.array)
            target = cast_array(tree, source, op.to, known, function)
            known[target] = Array(known[source].shape, PRECISION_TYPES[op.to])
            working[op.array] = target
        elif isinstance(op, Zip):
            fields = [working.pop(name, name) for name in op.arrays]
            zipped, _, _ = zipped_array(fields, known, op.dim)
            zip_arrays(tree, fields, op.into, known, op.dim, function)
            known[op.into] = zipped
            if zipped.element is not None:
                stacked[op.into] = len(zipped.shape) - 1 if op.dim is None else op.dim
        elif isinstance(op, Unzip):
            spec = known[op.array]
            dim = stacked.get(op.array, len(spec.shape) - 1) if op.dim is None else op.dim
            unzip_array(tree, op.array, op.into, dim, known, function)
            rest = tuple(extent for k, extent in enumerate(spec.shape) if k != dim) or ("1",)
            known.update({part: Array(rest, spec.element) for part in op.into})
    laid_out = {op.array for op in requests.operations if isinstance(op, DigitOperation)}
    pending: list[MpiBuffer] = []
    for name in sorted(laid_out):
        target = working.get(name, name)
        layout = state.layout(name, list(known[target].shape))
        working[name] = relayout_array(tree, target, layout, function, pending)
    pack_mpi_buffers(kernel(tree, function), pending)
    return working
