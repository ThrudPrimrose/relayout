# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
r"""Untile a tiled loop nest whose body reads an array blocked to match the tile, unblocking the array.

:class:`UntileBlockedLoops` is dace's :class:`~dace.transformation.passes.canonicalize.untile_loops.UntileLoops`
that also admits a case-A body reading ``A[..., int_floor(i, K), ii]`` (last extent ``K``): before the
substitution it flattens that array with :class:`UnblockDimensions`, the inverse of blocking, which merges each
blocked (outer, inner) dimension pair back into one, ``outer*factor + inner``. It assumes a packed blocked
array and single-tile outer accesses; multi-tile outer accesses raise. :func:`relayout.passes.sdfg.prepare.prepare`
runs it so a hand-blocked program reaches the layout search flat.

An interstate assignment holds a whole expression, so its accesses are located structurally
(``pystr_to_symbolic`` lowers ``A[i, j]`` to :class:`~dace.symbolic.Subscript`), never by text: a greedy
``name\[(.*)\]`` over ``A[i] * B[i]`` runs to B's closing bracket. Sympy re-associates the rewritten
expression as it parses (``0.5 * (A + B)`` prints as ``0.5*A + 0.5*B``); that is accepted.
"""

from typing import Any

import dace
import sympy
from dace import SDFG, symbolic
from dace.sdfg import nodes
from dace.sdfg.narrowing import as_basic, as_range
from dace.sdfg.state import LoopRegion
from dace.transformation import pass_pipeline as ppl
from dace.transformation.passes.canonicalize.untile_loops import (
    UntileLoops,
    _audit_combined_access,
    _diff_is_zero,
    free_symbol_names,
    memlet_bound_texts,
    reads_only_through_sum,
)

from relayout.passes.sdfg.unblock_dimensions import UnblockDimensions

__all__ = ["UntileBlockedLoops"]


#: ``{array: (masks, factors)}`` for :class:`UnblockDimensions`.
BlockedArrays = dict[str, tuple[list[bool], list[int]]]


def is_unit_point(
    rng: tuple[symbolic.SymbolicType, symbolic.SymbolicType, symbolic.SymbolicType], target: symbolic.SymbolicType
) -> bool:
    """``True`` iff ``rng`` is the unit-stride single point ``target``."""
    lo, hi, stp = rng
    return _diff_is_zero(lo, hi) and _diff_is_zero(stp, 1) and _diff_is_zero(lo, target)


def match_block_memlet(
    sdfg: SDFG, memlet: dace.Memlet, outer_var: str, inner_var: str, K_expr: symbolic.SymbolicType, K_const: int
) -> tuple[list[bool], list[int]] | None:
    """``(masks, factors)`` unblocking ``memlet``'s array if it reads ``A[..., int_floor(i, K), ii]`` with the last
    extent ``K`` and no leading axis naming ``i`` or ``ii``; else ``None``."""
    arr = sdfg.arrays.get(memlet.data)
    ranges = as_range(memlet.subset).ranges
    rank = len(ranges)
    if arr is None or rank < 2 or len(arr.shape) != rank:
        return None
    ii_sym = symbolic.pystr_to_symbolic(inner_var)
    if not is_unit_point(ranges[rank - 1], ii_sym) or not _diff_is_zero(arr.shape[rank - 1], K_const):
        return None
    block_index = symbolic.pystr_to_symbolic(f"int_floor({outer_var}, {symbolic.symstr(K_expr)})")
    if not is_unit_point(ranges[rank - 2], block_index):
        return None
    for rng in ranges[: rank - 2]:
        for bound in rng:
            names = free_symbol_names(str(bound))
            if outer_var in names or inner_var in names:
                return None
    return [False] * (rank - 2) + [True], [1] * (rank - 2) + [K_const]


def blocked_arrays_of(
    inner: LoopRegion,
    outer_var: str,
    inner_var: str,
    case: str,
    K_expr: symbolic.SymbolicType,
    K_const: int | None,
    sdfg: SDFG,
    aligned: bool,
) -> BlockedArrays | None:
    """:func:`_audit_combined_access` that also admits a case-A read of an array blocked to match the tile, and
    (tiles starting at multiples of ``K``: ``aligned``) one unblocked already, ``K*int_floor(i, K) + ii``.

    :returns: ``None`` to refuse, else the arrays to unblock before the substitution.
    """
    if case == "B":
        return {} if _audit_combined_access(inner, outer_var, inner_var, case) else None
    blocked: BlockedArrays = {}
    for st in inner.states():
        for e in st.edges():
            if e.data is None or e.data.is_empty():
                continue
            # Naming both tile variables is not enough: ``2*i + ii`` collapses to ``k`` as well.
            texts = memlet_bound_texts(e.data)
            if aligned:
                texts = [fold_tile_origin(text, outer_var, K_expr) for text in texts]
            if all(reads_only_through_sum(text, outer_var, inner_var) for text in texts):
                continue
            # A copy reading the array (a scalar the frontend reads it into) keeps its other side as it is.
            if K_const is None or e.data.subset is None:
                return None
            spec = match_block_memlet(sdfg, e.data, outer_var, inner_var, K_expr, K_const)
            if spec is None or blocked.setdefault(e.data.data, spec) != spec:
                return None
    return blocked


def fold_tile_origin(text: str, outer_var: str, K_expr: symbolic.SymbolicType) -> str:
    """``text`` with ``int_floor(i, K)`` as ``i / K``: exact for a tile origin ``i`` that is a multiple of ``K``, so
    ``K*int_floor(i, K)`` reads as ``i`` and a lone block index stays fractional (and is refused)."""
    block_index = symbolic.pystr_to_symbolic(f"int_floor({outer_var}, {symbolic.symstr(K_expr)})")
    origin = symbolic.pystr_to_symbolic(outer_var)
    return symbolic.symstr(sympy.expand(as_basic(symbolic.pystr_to_symbolic(text)).subs(block_index, origin / K_expr)))


def unblock_is_safe(sdfg: SDFG, blocked_arrays: BlockedArrays) -> bool:
    """``True`` iff every access to every array in ``blocked_arrays`` is one ``UnblockDimensions`` can rewrite: none
    crosses a NestedSDFG boundary, reaches it through the other side of a copy, or has another rank."""
    for arr_name, spec in blocked_arrays.items():
        masks = spec[0]
        arr = sdfg.arrays.get(arr_name)
        expected = len(masks) + sum(1 for m in masks if m)
        if arr is None or len(arr.shape) != expected:
            return False
        for state in sdfg.states():
            for node in state.nodes():
                if isinstance(node, nodes.NestedSDFG) and any(
                    e.data is not None and e.data.data == arr_name for e in state.in_edges(node) + state.out_edges(node)
                ):
                    return False
            for edge in state.edges():
                if edge.data is None or edge.data.is_empty():
                    continue
                if edge.data.data != arr_name:
                    # a copy named after its other side would index the array through ``other_subset``
                    ends = (edge.src, edge.dst)
                    if edge.data.other_subset is not None and any(
                        isinstance(n, nodes.AccessNode) and n.data == arr_name for n in ends
                    ):
                        return False
                    continue
                if edge.data.subset is None or len(as_range(edge.data.subset).ranges) != expected:
                    return False
    return True


def unblock_before_untile(
    sdfg: SDFG, outer_start: symbolic.SymbolicType, K_expr: symbolic.SymbolicType, blocked_arrays: BlockedArrays
) -> bool:
    """Unblock ``blocked_arrays`` so the case-A substitution folds ``int_floor(i, K)*K + ii`` to ``k``.

    :returns: ``False``, with the SDFG untouched, when that fold would move elements.
    """
    if not blocked_arrays:
        return True
    # Every tile origin must be a multiple of K, else each element shifts by ``start mod K``.
    if symbolic.simplify(sympy.Mod(outer_start, K_expr)) != 0 or not unblock_is_safe(sdfg, blocked_arrays):
        return False
    UnblockDimensions(unblock_map=blocked_arrays).apply_pass(sdfg, {})
    return True


class UntileBlockedLoops(UntileLoops):
    """:class:`UntileLoops` that also untiles a case-A nest whose body reads an array blocked to match the tile
    (``A[..., int_floor(i, K), ii]`` with extent ``K``), unblocking that array in the same rewrite. Needs a
    concrete ``K``; refuses clamped nests. Takes the Map round trip only when forced: the layout preparation
    canonicalizes right after, and canonicalize takes the trip there."""

    def modifies(self) -> ppl.Modifies:
        # Descriptors: an unblocked array changes rank.
        return super().modifies() | ppl.Modifies.Descriptors

    def auto_roundtrip(self, sdfg: SDFG) -> bool:
        return False

    def audit_candidate(
        self,
        candidate: LoopRegion,
        outer_var: str,
        case: str,
        clamped: bool,
        K_expr: symbolic.SymbolicType,
        K_const: int | None,
        sdfg: SDFG,
        outer_start: symbolic.SymbolicType,
    ) -> BlockedArrays | None:
        """The arrays to unblock before collapsing ``candidate`` with its outer tile loop, or ``None`` to refuse."""
        # The unblock rewrite has no remainder-tile form; canonicalize collapses a clamped nest later.
        if clamped:
            return None
        aligned = symbolic.simplify(sympy.Mod(outer_start, K_expr)) == 0
        return blocked_arrays_of(candidate, outer_var, candidate.loop_variable, case, K_expr, K_const, sdfg, aligned)

    def prepare_untile(
        self, sdfg: SDFG, outer_start: symbolic.SymbolicType, K_expr: symbolic.SymbolicType, audit: dict[str, Any]
    ) -> bool:
        return unblock_before_untile(sdfg, outer_start, K_expr, audit)
