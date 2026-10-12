# Copyright 2026 ETH Zurich and the relayout authors.
"""How a library call absorbs a laid-out operand, independent of the frontend.

An operand ``X`` stored permuted is the buffer ``S = transpose(X, perm)``: axis ``k`` of ``S`` is axis
``perm[k]`` of ``X`` (:meth:`relayout.layout.IndexMap.permutation`). A call on ``X`` becomes the same
call on ``S`` with its axis arguments renamed: a transpose composes, an einsum operand reorders its
subscripts, a contraction or a reduction renames its axes, a 2-D BLAS operand flips its transpose
flag. Each rule is a pure function of the call's arguments; the Python and SDFG frontends apply them
to their own call forms. A layout that is not a permutation is not absorbed by any call: the frontend
converts the operand (:mod:`relayout.convert`) or refuses.
"""

from collections.abc import Sequence

__all__ = ["contracted_axes", "einsum_operand", "flips_transpose", "inverse", "transpose_axes"]


def inverse(perm: Sequence[int]) -> tuple[int, ...]:
    """The permutation undoing ``perm``: ``inverse(perm)[perm[k]] == k``."""
    out = [0] * len(perm)
    for k, axis in enumerate(perm):
        out[axis] = k
    return tuple(out)


def transpose_axes(axes: Sequence[int], perm: Sequence[int]) -> tuple[int, ...]:
    """``transpose(X, axes)`` as a transpose of ``S = transpose(X, perm)``."""
    back = inverse(perm)
    return tuple(back[axis] for axis in axes)


def einsum_operand(subscripts: str, perm: Sequence[int]) -> str:
    """An einsum operand's subscripts (one letter per axis of ``X``) for ``S``."""
    if len(subscripts) != len(perm):
        raise ValueError(f"einsum operand {subscripts!r} has {len(subscripts)} axes, the layout {len(perm)}")
    return "".join(subscripts[axis] for axis in perm)


def contracted_axes(axes: Sequence[int], perm: Sequence[int]) -> tuple[int, ...]:
    """Axes of ``X`` a contraction or a reduction names, as axes of ``S``."""
    back = inverse(perm)
    return tuple(back[axis] for axis in axes)


def flips_transpose(perm: Sequence[int]) -> bool:
    """Whether a 2-D BLAS operand stored as ``perm`` flips its transpose flag: ``(1, 0)`` does, the
    identity does not, and anything else is not a 2-D layout a flag expresses."""
    if tuple(perm) not in ((0, 1), (1, 0)):
        raise ValueError(f"a BLAS operand's layout must be a 2-D permutation, not {tuple(perm)}")
    return tuple(perm) == (1, 0)
