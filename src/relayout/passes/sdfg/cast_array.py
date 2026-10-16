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


def cast_array(sdfg: dace.SDFG, name: str, precision: Precision, convert: bool = True) -> str:
    """``sdfg``'s array ``name`` in the element type of ``precision``, by FP-Arena's type change and propagation
    (extra ``fparena``). Every other floating-point argument keeps its type; the transients the array reaches take
    the type the propagation gives them. A transient is cast in place; an argument works through a transient
    converted in and out, or, without ``convert``, is passed in the new type by the caller. Returns the name the
    program now works on."""
    from fp_arena.transformations import change_and_propagate_fp_types  # the optional extra 'fparena'

    target = PRECISION_TYPES.get(precision)
    if target is None or name not in sdfg.arrays:
        raise LayoutError(f"cast of {name!r} to {precision.value}: no such array, or no such element type in dace")
    desc = array_desc(sdfg, name)
    if desc.dtype == target:
        return name
    pins = {
        other: other_desc.dtype
        for other, other_desc in sdfg.arrays.items()
        if other != name and not other_desc.transient and other_desc.dtype in PRECISION_TYPES.values()
    }
    pins[name] = target
    argument = not desc.transient  # FP-Arena makes this descriptor the transient working copy
    change_and_propagate_fp_types(sdfg, pins, preserve_interface=convert)
    if not argument or not convert:
        return name
    # FP-Arena's working copy, under the name the Python pass gives it (a zip's struct member is named after it).
    working = f"{name}_{target.to_string()}"
    sdfg.replace_dict({f"fp_casted_{working}": working})
    return working
