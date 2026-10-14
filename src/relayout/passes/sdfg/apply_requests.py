# Copyright 2026 ETH Zurich and the relayout authors.
"""A request's operations carried out on an SDFG: ``cast``, ``zip`` and ``unzip`` in program order, then
every array a digit operation touched laid out (:func:`~relayout.passes.sdfg.relayout_array.relayout_array`)
and the MPI buffers packed once all are.
"""

import dace
import numpy as np
from dace import data
from dace.sdfg import propagation

from relayout.ops import ArraySpec, Cast, DigitOperation, Requests, Unzip, Zip
from relayout.passes.sdfg.cast_array import cast_array
from relayout.passes.sdfg.pack_mpi_buffers import MpiBuffer, pack_mpi_buffers
from relayout.passes.sdfg.relayout_array import relayout_array
from relayout.passes.sdfg.unzip_array import unzip_array
from relayout.passes.sdfg.zip_arrays import zip_arrays
from relayout.passes.sdfg.common import LayoutError
from relayout.passes.single_node import unapplied

__all__ = ["apply_requests"]


def apply_requests(sdfg: dace.SDFG, requests: Requests) -> dict[str, str]:
    """Carry ``requests``' operations out on ``sdfg`` (single node): ``cast`` (:func:`cast_array`, FP-Arena),
    ``zip`` and ``unzip`` (:func:`zip_arrays`, :func:`unzip_array`) in program order, so a cast before a zip
    makes a struct of the cast types; then every array a digit operation touched is laid out as the
    operations leave it (:func:`relayout_array`). Returns the name each requested array is now worked on. A
    request the caller must carry out in part is refused (:func:`~relayout.passes.single_node.unapplied`)."""

    refusal = unapplied(requests)
    if refusal is not None:
        raise LayoutError(refusal)

    def arrays() -> dict[str, data.Array]:
        return {n: d for n, d in sdfg.arrays.items() if isinstance(d, data.Array) and not isinstance(d, data.View)}

    state = requests.run(
        [
            ArraySpec(name, len(desc.shape), element=np.dtype(desc.dtype.as_numpy_dtype()).name)
            for name, desc in arrays().items()
        ]
    )
    working: dict[str, str] = {}
    stacked: dict[str, int] = {}  # a zipped array's field dimension, which its unzip splits along by default
    for op in requests.operations:
        if isinstance(op, Cast):
            working[op.array] = cast_array(sdfg, working.get(op.array, op.array), op.to)
        elif isinstance(op, Zip):
            zip_arrays(sdfg, [working.pop(name, name) for name in op.arrays], op.into, op.dim)
            if not isinstance(sdfg.arrays[op.into].dtype, dace.struct):
                stacked[op.into] = len(sdfg.arrays[op.into].shape) - 1 if op.dim is None else op.dim
        elif isinstance(op, Unzip):
            last = len(sdfg.arrays[op.array].shape) - 1
            dim = stacked.get(op.array, last) if op.dim is None else op.dim
            unzip_array(sdfg, op.array, op.into, dim)
    laid_out = {op.array for op in requests.operations if isinstance(op, DigitOperation)}
    pending: list[MpiBuffer] = []
    for name in sorted(laid_out):
        target = working.get(name, name)
        shape = [str(extent) for extent in arrays()[target].shape]
        working[name] = relayout_array(sdfg, target, state.layout(name, shape), pending)
    pack_mpi_buffers(pending)
    propagation.propagate_memlets_sdfg(sdfg)
    return working
