# Copyright 2026 ETH Zurich and the relayout authors.
"""One array of a Python function laid out by an index map: one rewrite for every layout (permute, block, split,
pad, shuffle), the counterpart of :mod:`relayout.passes.sdfg.relayout_array`.

:func:`relayout_array` lays one array out by an :class:`~relayout.layout.IndexMap`: the function works on the
packed row-major buffer of the layout's extents, and every access follows the layout
(:mod:`relayout.passes.py.rewrite_accesses`). A local is laid out in place at its definition. An argument keeps
its array for the caller: the function works on a local ``laid_<name>``, converted in at the entry and out at
the exit by loops over the logical index space. Dead cells (a pad, a ragged block) hold zeros.
"""

import ast

from relayout.layout import IndexMap
from relayout.passes.py.common import arguments, ensure_numpy, kernel, laid_name, statement, tuple_text
from relayout.passes.py.pack_mpi_buffers import MpiBuffer, pack_mpi_buffers
from relayout.passes.py.placement import Laid
from relayout.passes.py.rewrite_accesses import move_array

__all__ = ["relayout_array"]


def relayout_array(
    tree: ast.Module,
    name: str,
    layout: IndexMap,
    function: str | None = None,
    pending: list[MpiBuffer] | None = None,
) -> str:
    """Lay array ``name`` of ``tree``'s function (the one named ``function``, or the only one) out by ``layout``
    (see the module docstring); returns the name the function now works on (``name`` for a local,
    ``laid_<name>`` for an argument). Its MPI buffers are packed at once
    (:func:`~relayout.passes.py.pack_mpi_buffers.pack_mpi_buffers`), or recorded in ``pending`` for the caller to
    pack once every array is laid out."""
    if layout.is_identity():
        return name
    func = kernel(tree, function)
    ensure_numpy(tree)
    laid = laid_name(name) if name in arguments(func) else name
    extents = tuple_text(layout.extents())

    def allocate(dtype: str) -> list[ast.stmt]:
        return [statement(f"{laid} = np.zeros({extents}, dtype={dtype})")]

    own: list[MpiBuffer] = [] if pending is None else pending
    move_array(func, name, laid, Laid(layout), layout.shape, allocate, own)
    if pending is None:
        pack_mpi_buffers(func, own)
    return laid
