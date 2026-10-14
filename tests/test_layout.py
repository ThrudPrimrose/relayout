# Copyright 2026 ETH Zurich and the relayout authors.
"""The layout an array's operations leave (``IndexMap``): every logical cell lands on exactly one live
physical cell and comes back, and the cuts agree with numpy's reshape and transpose."""

import itertools

import numpy as np
import pytest

from tests.conftest import LAYOUT_CASES, LAYOUT_SHAPE
from relayout.layout import IndexMap
from relayout.ops import (
    ArraySpec,
    Block,
    Distribute,
    LinearShuffle,
    Pad,
    Permute,
    Scheme,
    Shuffle,
    Split,
    eval_index,
    inverse_sequence,
    run,
)

SWIZZLE = ((1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1))


def layout_of(ops: list, shape: tuple[int, ...], grid: tuple[int, ...] = ()) -> IndexMap:
    state = run(ops, [ArraySpec("A", len(shape))], grid, "operations")
    return state.layout("A", [str(extent) for extent in shape])


def placement(layout: IndexMap, shape: tuple[int, ...]) -> np.ndarray:
    """The laid-out buffer of ``np.arange`` over ``shape``: each cell holds the flat logical index it
    stores, -1 where it is dead; checks the round trip and the live conditions on the way."""
    names = [f"x{d}" for d in range(len(shape))]
    position = layout.physical(names)
    extents = tuple(eval_index(e, {}, "t") for e in layout.extents())
    out = np.full(extents, -1, dtype=np.int64)
    for index in itertools.product(*(range(extent) for extent in shape)):
        bound = dict(zip(names, index, strict=True))
        cell = tuple(eval_index(expr, bound, "t") for expr in position)
        assert out[cell] == -1, f"{index} and another index share cell {cell}"
        out[cell] = np.ravel_multi_index(index, shape)
        at = {f"p{k}": v for k, v in enumerate(cell)}
        back = [eval_index(expr, at, "t") for expr in layout.logical(list(at))]
        assert back == list(index)
        assert all(eval_live(cond, at) for cond in layout.live(list(at)))
    for cell in zip(*np.nonzero(out < 0), strict=True):
        at = {f"p{k}": int(v) for k, v in enumerate(cell)}
        assert not all(eval_live(cond, at) for cond in layout.live(list(at))), f"dead cell {cell} reads as live"
    return out


def eval_live(condition: str, names: dict[str, int]) -> bool:
    left, right = condition.rsplit(" < ", 1)
    return eval_index(left, names, "t") < eval_index(right, names, "t")


def test_permute_and_block_are_numpys_transpose_and_reshape() -> None:
    flat = np.arange(6 * 8).reshape(6, 8)
    transposed = placement(layout_of([Permute("A", ("i1", "i0"))], (6, 8)), (6, 8))
    np.testing.assert_array_equal(transposed, flat.T)
    blocked = placement(layout_of([Block("A", "i1", "4", ("o", "f")), Permute("A", ("o", "i0", "f"))], (6, 8)), (6, 8))
    np.testing.assert_array_equal(blocked, flat.reshape(6, 2, 4).transpose(1, 0, 2))


def test_a_ragged_block_and_a_pad_leave_dead_cells() -> None:
    ragged = placement(layout_of([Block("A", "i0", "4", ("o", "f"))], (10,)), (10,))
    assert ragged.shape == (3, 4)
    assert list(ragged.ravel()) == [*range(10), -1, -1]
    padded = placement(layout_of([Pad("A", "i1", "3")], (2, 5)), (2, 5))
    assert padded.shape == (2, 8)
    np.testing.assert_array_equal(padded[:, :5], np.arange(10).reshape(2, 5))
    assert (padded[:, 5:] == -1).all()


def test_shuffles_renumber_their_digits_and_come_back() -> None:
    xor = [Block("A", "i1", "4", ("o", "c")), Block("A", "i0", "4", ("r", "w"))]
    xor.append(LinearShuffle("A", ("w", "c"), (2, 2), SWIZZLE))
    swizzled = placement(layout_of(xor, (8, 8)), (8, 8))
    assert sorted(swizzled.ravel()) == list(range(64))
    reversal = [Block("A", "i0", "4", ("o", "w")), Shuffle("A", ("w",), ("3 - w",), ("3 - w",))]
    reversed_blocks = placement(layout_of(reversal, (8,)), (8,))
    np.testing.assert_array_equal(reversed_blocks, [[3, 2, 1, 0], [7, 6, 5, 4]])


def test_a_shuffle_that_is_not_a_bijection_is_refused() -> None:
    with pytest.raises(ValueError, match="f_inv must undo f"):
        layout_of([Block("A", "i0", "4", ("o", "w")), Shuffle("A", ("w",), ("3 - w",), ("w",))], (8,))
    with pytest.raises(ValueError, match="outside the digits' extents"):
        layout_of([Block("A", "i0", "4", ("o", "w")), Shuffle("A", ("w",), ("w + 1",), ("w - 1",))], (8,))


def test_a_distributed_digit_leaves_the_rank_local_layout() -> None:
    """Each rank holds its tile row-major: a cyclic or block-cyclic deal divides the coarser strides."""
    cyclic = [Distribute("A", 0, dim=0, scheme=Scheme.CYCLIC)]
    assert layout_of(cyclic, (3,), (4,)).is_identity()
    blocked = [Block("A", "i0", "2*P0", ("o", "c")), Block("A", "c", "2", ("r", "b")), Distribute("A", 0, digit="r")]
    local = layout_of(blocked, (6,), (4,))
    assert [(a.name, a.stride) for a in local.axes] == [("o", 2), ("b", 1)]
    placement(local, (6,))
    with pytest.raises(ValueError, match="split a dimension into bands"):
        layout_of([Split("A", "i0", "P0", ("r", "i0"))], (8,), (4,))


@pytest.mark.parametrize("case", LAYOUT_CASES)
def test_every_layout_case_is_a_bijection_its_inverses_undo(case: str) -> None:
    """Every logical cell lands on one live cell and comes back (:func:`placement`), every dead cell reads as
    dead, and the operations followed by their inverses leave the identity."""
    ops = LAYOUT_CASES[case]
    placed = placement(layout_of(ops, LAYOUT_SHAPE), LAYOUT_SHAPE)
    assert sorted(placed[placed >= 0]) == list(range(np.prod(LAYOUT_SHAPE)))
    assert layout_of([*ops, *inverse_sequence(ops, len(LAYOUT_SHAPE))], LAYOUT_SHAPE).is_identity()


if __name__ == "__main__":
    test_permute_and_block_are_numpys_transpose_and_reshape()
    test_a_ragged_block_and_a_pad_leave_dead_cells()
    test_shuffles_renumber_their_digits_and_come_back()
    test_a_shuffle_that_is_not_a_bijection_is_refused()
    test_a_distributed_digit_leaves_the_rank_local_layout()
    for name in LAYOUT_CASES:
        test_every_layout_case_is_a_bijection_its_inverses_undo(name)
