# Copyright 2026 ETH Zurich and the relayout authors.
"""``cast``: an array in another element type, by FP-Arena's floating-point type change (extra ``fparena``)."""

import dace

from relayout.ops import Precision
from relayout.passes.sdfg.common import LayoutError, array_desc

__all__ = ["PRECISION_TYPES", "cast_array"]


#: The element type a ``cast`` asks for, as dace has it.
PRECISION_TYPES = {
    Precision.FP64: dace.float64,
    Precision.FP32: dace.float32,
    Precision.FP16: dace.float16,
}


def cast_array(sdfg: dace.SDFG, name: str, precision: Precision) -> str:
    """``sdfg``'s array ``name`` in the element type of ``precision``, by FP-Arena's type change (extra
    ``fparena``): a transient in place, an argument through a transient converted in and out; returns the
    name the program now works on."""
    from fp_arena.transformations.change_fp_types import change_fptype  # the optional extra 'fparena'

    target = PRECISION_TYPES.get(precision)
    if target is None or name not in sdfg.arrays:
        raise LayoutError(f"cast of {name!r} to {precision.value}: no such array, or no such element type in dace")
    desc = array_desc(sdfg, name)
    if desc.dtype == target:
        return name
    argument = not desc.transient  # change_fptype makes the descriptor it renames transient
    change_fptype(sdfg, desc.dtype, target, cast_in_and_out_data=argument, arrays_to_replace={name})
    return f"{name}_{target.to_string()}" if argument else name
