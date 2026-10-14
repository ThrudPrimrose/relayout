# Copyright 2026 ETH Zurich and the relayout authors.
"""untile_blocked_loops on Python: a tile nest reading an array blocked to the tile collapses, the array unblocked;
the counterpart of tests/test_sdfg_untile_blocked_loops.py."""

import ast

import numpy as np

from relayout.passes.py.common import Array
from relayout.passes.py.prepare import prepare
from relayout.passes.py.untile_blocked_loops import untile_blocked_loops
from tests.conftest import rewritten


def blocked_tiles(a: np.ndarray, b: np.ndarray, N: int) -> None:  # noqa: N803
    blk = np.empty((N // 4, 4), dtype=np.float64)
    for i in range(0, N, 4):
        for ii in range(4):
            blk[i // 4, ii] = b[i + ii] + 1.0
    for i in range(0, N, 4):
        for ii in range(4):
            a[i + ii] = blk[i // 4, ii] * 2.0


def loops(source: str) -> int:
    return sum(isinstance(node, ast.For) for node in ast.walk(ast.parse(source)))


def test_an_array_blocked_to_the_tile_is_unblocked_and_both_nests_untile() -> None:
    """``blk[i // 4, ii]`` is written by one tile nest and read by another: ``blk`` is flattened to ``blk[k]``
    (allocated ``(N,)``) and both nests collapse to one loop each."""
    count: list[int] = []
    function = rewritten(blocked_tiles, lambda tree: count.append(untile_blocked_loops(tree)))
    assert count == [2] and loops(function.source) == 2
    assert "blk = np.empty((N,)" in function.source
    assert "blk[_untile_k_0] = " in function.source and "blk[_untile_k_1] * 2.0" in function.source
    a, b = np.zeros(12), np.random.default_rng(0).random(12)
    function.function(a, b, N=12)
    np.testing.assert_allclose(a, 2.0 * (b + 1.0))


def tiles(a: np.ndarray, b: np.ndarray, N: int) -> None:  # noqa: N803
    for i in range(0, N, 4):
        for ii in range(i, min(i + 4, N)):
            a[ii] = b[ii] * 3.0
    for i in range(0, N, 4):
        for ii in range(min(4, N - i)):
            b[i + ii] = a[i + ii] + 1.0


def test_case_a_and_case_b_nests_untile_also_when_clamped() -> None:
    function = rewritten(tiles, lambda tree: prepare(tree))
    assert loops(function.source) == 2 and "range(0, N)" in function.source
    a, b = np.zeros(10), np.random.default_rng(1).random(10)
    want_a = 3.0 * b
    function.function(a, b, N=10)
    np.testing.assert_allclose(a, want_a)
    np.testing.assert_allclose(b, want_a + 1.0)


def blocked_argument(a: np.ndarray, blk: np.ndarray, N: int) -> None:  # noqa: N803
    for i in range(0, N, 4):
        for ii in range(4):
            a[i + ii] = blk[i // 4, ii]


def test_a_blocked_argument_unblocks_only_when_its_last_extent_is_the_tile() -> None:
    """An argument's last extent is known from its shape only: unknown or another extent leaves the nest."""
    unknown = rewritten(blocked_argument, lambda tree: untile_blocked_loops(tree))
    assert loops(unknown.source) == 2
    shaped = {"blk": Array(("N // 4", "4"), "float64")}
    flat = rewritten(blocked_argument, lambda tree: untile_blocked_loops(tree, shaped))
    assert loops(flat.source) == 1
    a, blk = np.zeros(8), np.random.default_rng(2).random(8)
    flat.function(a, blk, N=8)  # the caller passes blk unblocked
    np.testing.assert_allclose(a, blk)


if __name__ == "__main__":
    test_an_array_blocked_to_the_tile_is_unblocked_and_both_nests_untile()
    test_case_a_and_case_b_nests_untile_also_when_clamped()
    test_a_blocked_argument_unblocks_only_when_its_last_extent_is_the_tile()
