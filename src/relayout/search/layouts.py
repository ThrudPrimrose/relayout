# Copyright 2026 ETH Zurich and the relayout authors.
"""The layouts the search chooses among: an array's digit operations under a tag, and the candidate
families (permutations, blockings) it enumerates."""

import itertools
from collections.abc import Sequence
from dataclasses import dataclass

import dace

from relayout.frontends.schedule import canonicalize_schedule
from relayout.frontends.sdfg import relayout_array
from relayout.layout import IndexMap
from relayout.ops import (
    ArrayName,
    ArraySpec,
    Block,
    DigitName,
    DigitNames,
    DigitOperation,
    DigitPair,
    Expr,
    Permute,
    run,
)

__all__ = ["IDENTITY", "Layout", "blockings", "lay_out", "permutations"]

#: The tag of the layout the program has (no operation).
IDENTITY = "identity"

#: The most dimensions whose every permutation the search enumerates.
MAX_PERMUTED_DIMS = 3


@dataclass(frozen=True, slots=True)
class Layout:
    """One array's layout: the digit operations from its row-major identity, named by ``tag``."""

    tag: str
    operations: tuple[DigitOperation, ...] = ()

    @property
    def is_identity(self) -> bool:
        return not self.operations

    def index_map(self, array: str, shape: Sequence[str]) -> IndexMap:
        return run(self.operations, [ArraySpec(array, len(shape))], (), f"layout {self.tag!r}").layout(array, shape)


def permutations(array: str, ndim: int) -> list[Layout]:
    """The identity, then every other order of ``array``'s dimensions."""
    if ndim > MAX_PERMUTED_DIMS:
        raise ValueError(f"{array!r} has {ndim} dimensions; enumerate at most {MAX_PERMUTED_DIMS} (pass candidates)")
    digits = [f"i{d}" for d in range(ndim)]
    return [Layout(IDENTITY)] + [
        Layout(f"perm{''.join(str(digits.index(d)) for d in order)}", (Permute(ArrayName(array), DigitNames(order)),))
        for order in itertools.permutations(digits)
        if list(order) != digits
    ]


def blockings(array: str, ndim: int, factors: Sequence[int] = (8, 16, 32)) -> list[Layout]:
    """The identity, then ``array`` with one dimension (not the last) cut into blocks of each factor,
    each block stored contiguous (its offset the innermost digit)."""
    out = [Layout(IDENTITY)]
    for d in range(ndim - 1):
        for factor in factors:
            coarse, fine = f"b{d}", f"f{d}"
            order = [coarse if k == d else f"i{k}" for k in range(ndim)] + [fine]
            block = Block(ArrayName(array), DigitName(f"i{d}"), Expr(str(factor)), DigitPair((coarse, fine)))
            out.append(Layout(f"block{d}x{factor}", (block, Permute(ArrayName(array), DigitNames(tuple(order))))))
    return out


def lay_out(sdfg: dace.SDFG, array: str, layout: Layout) -> str:
    """``sdfg``'s ``array`` laid out by ``layout`` and its maps re-tiled to it; returns the name the program
    works on."""
    shape = [str(extent) for extent in sdfg.arrays[array].shape]
    name = relayout_array(sdfg, array, layout.index_map(array, shape))
    canonicalize_schedule(sdfg)
    return name
