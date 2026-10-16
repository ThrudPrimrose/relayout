# Copyright 2026 ETH Zurich and the relayout authors.
"""One array of an SDFG laid out by an index map: one rewrite for every layout (permute, block, split, pad,
shuffle).

:func:`relayout_array` lays one array out by an :class:`~relayout.layout.IndexMap`. Its descriptor
becomes the packed row-major buffer of the layout's extents, and every access follows the layout:

- a point access (a memlet of one element, an interstate subscript) reads the physical cell;
- a range access keeps its exact range on the axes a permutation or a pad moves, and covers a whole
  axis that a cut or a renumbering spreads it over (re-propagated from the point accesses inside);
- the arrays a nested SDFG receives are laid out the same way, with the nested descriptor's shape;
- a library node reading or writing it absorbs a permutation by its rule (:mod:`relayout.rules`) or
  refuses the layout by name.

A transient is laid out in place. An argument of the SDFG keeps its descriptor for the caller: the
program works on a transient ``laid_<name>``, converted in at the start (when it is read) and out at
the end (when it is written) by maps over the logical index space. With ``convert=False`` an argument
is laid out in place too: the caller passes the laid-out buffer (:mod:`relayout.convert`), so the
program's signature changes and no conversion runs inside it.
"""

import dace
from dace.sdfg import propagation

from relayout.layout import IndexMap
from relayout.passes.sdfg.common import array_desc, laid_name, laid_out_descriptor
from relayout.passes.sdfg.pack_mpi_buffers import MpiBuffer, pack_mpi_buffers
from relayout.passes.sdfg.placement import Laid
from relayout.passes.sdfg.rewrite_accesses import move_array

__all__ = ["relayout_array"]


def relayout_array(
    sdfg: dace.SDFG, name: str, layout: IndexMap, pending: list[MpiBuffer] | None = None, convert: bool = True
) -> str:
    """Lay ``sdfg``'s array ``name`` out by ``layout`` (see the module docstring); returns the name the
    program now works on (``name`` for a transient or, without ``convert``, an argument; ``laid_<name>``
    for a converted argument). Its MPI buffers are packed at once (:func:`pack_mpi_buffers`), or recorded
    in ``pending`` for the caller to pack once every array is laid out."""
    desc = array_desc(sdfg, name)
    if layout.is_identity():
        return name
    in_place = desc.transient or not convert
    laid = name if in_place else laid_name(name)
    laid_desc = laid_out_descriptor(desc, layout, transient=desc.transient)
    if in_place:
        sdfg.arrays[name] = laid_desc
    else:
        laid_desc.transient = True
        sdfg.add_datadesc(laid, laid_desc)
    own = [] if pending is None else pending
    move_array(sdfg, name, laid, Laid(layout), own, convert=convert)
    if pending is None:
        pack_mpi_buffers(own)
    propagation.propagate_memlets_sdfg(sdfg)
    return laid
