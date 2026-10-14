# Copyright 2026 ETH Zurich and the relayout authors.
"""The SDFG in the form the layout passes read best: tiled loops over blocked arrays untiled, canonicalized,
nested SDFGs receiving whole arrays, no views but those a library node reads, implicit copies explicit.
"""

import dace
from dace import data
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.transformation.passes.remove_views import RemoveViews

from relayout.passes.sdfg.common import LayoutError
from relayout.passes.sdfg.untile_blocked_loops import UntileBlockedLoops

__all__ = ["prepare"]


def prepare(sdfg: dace.SDFG, target: str = "cpu") -> dace.SDFG:
    """``sdfg`` in the form the rewrite reads best, in place: tiled loops and blocked arrays untiled,
    canonicalized (loops become maps), nested SDFGs receiving whole arrays, no views but those a
    library node reads, implicit copies explicit; a non-packed array is refused."""
    UntileBlockedLoops().apply_pass(sdfg, {})
    canonicalize(sdfg, target=target, validate=True)
    sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs)
    RemoveViews().apply_pass(sdfg, {})
    InsertExplicitCopies().apply_pass(sdfg, {})
    for _, name, desc in sdfg.arrays_recursive():
        if isinstance(desc, data.Array) and not isinstance(desc, (data.View, data.ContainerArray)):
            if not (desc.is_packed_c_strides() or desc.is_packed_fortran_strides()):
                raise LayoutError(f"{name!r} has strides {desc.strides} over {desc.shape}; lay out packed arrays only")
    sdfg.validate()
    return sdfg
