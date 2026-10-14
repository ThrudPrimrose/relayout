# Copyright 2026 ETH Zurich and the relayout authors.
"""The loops of a Python function after a layout; the counterpart of tests/test_sdfg_canonicalize_schedule.py: a
loop over a blocked array is tiled so its accesses lose their ``//`` and ``%``, a tiled nest over a flat array
is merged back, and an independent loop nest is ordered by stride; every result computes what the function
computed."""

import ast

import numpy as np

from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Block, Permute, run
from relayout.passes.py.canonicalize_schedule import canonicalize_schedule
from relayout.passes.py.relayout_array import relayout_array
from tests.conftest import rewritten


def layout(ops: list, shape: list[str]) -> IndexMap:
    return run(ops, [ArraySpec("A", len(shape))], (), "operations").layout("A", shape)


def scale(A: np.ndarray, B: np.ndarray, N: int, M: int) -> None:  # noqa: N803
    for i in range(N):
        for j in range(M):
            B[i, j] = A[i, j] * 2.0


def tiled(A: np.ndarray, B: np.ndarray, K: int) -> None:  # noqa: N803
    for jo in range(K):
        for ji in range(4):
            B[4 * jo + ji] = 2.0 * A[4 * jo + ji]


def interior(A: np.ndarray, B: np.ndarray, N: int) -> None:  # noqa: N803
    for i in range(1, N - 1):
        B[i] = A[i] * 2.0


def accumulate(A: np.ndarray, s: np.ndarray, N: int, M: int) -> None:  # noqa: N803
    for i in range(N):
        for j in range(M):
            s[0] = s[0] + A[i, j]


def nesting(source: str) -> list[list[str]]:
    """The variables of every perfect loop nest, outermost first."""
    out, inner = [], set()
    for node in ast.walk(ast.parse(source)):
        if not isinstance(node, ast.For) or id(node) in inner:
            continue
        chain = [node]
        while len(chain[-1].body) == 1 and isinstance(chain[-1].body[0], ast.For):
            chain.append(chain[-1].body[0])
            inner.add(id(chain[-1]))
        out.append([ast.unparse(loop.target) for loop in chain])
    return out


def test_a_blocked_array_tiles_its_loop() -> None:
    blocked = layout([Block("A", "i1", "4", ("o", "f"))], ["N", "M"])
    count: list[int] = []

    def transform(tree: ast.Module) -> None:
        for array in ("A", "B"):
            relayout_array(tree, array, blocked)
        count.append(canonicalize_schedule(tree))

    function = rewritten(scale, transform)
    assert count == [3]  # the function's loop and the two conversions
    assert "laid_A[i, j_o, j]" in function.source and "%" not in function.source
    a = np.random.default_rng(0).random((5, 10))
    b = np.zeros((5, 10))
    function.function(a, b, N=5, M=10)
    np.testing.assert_allclose(b, 2 * a)


def test_a_permuted_array_puts_its_unit_stride_loop_innermost() -> None:
    def transform(tree: ast.Module) -> None:
        for array in ("A", "B"):
            relayout_array(tree, array, layout([Permute("A", ("i1", "i0"))], ["N", "M"]))
        canonicalize_schedule(tree)

    function = rewritten(scale, transform)
    assert ["j", "i"] in nesting(function.source)
    a = np.random.default_rng(1).random((5, 10))
    b = np.zeros((5, 10))
    function.function(a, b, N=5, M=10)
    np.testing.assert_allclose(b, 2 * a)


def test_a_tiled_loop_merges_over_flat_arrays_and_keeps_its_tiles_over_blocked_ones() -> None:
    a = np.random.default_rng(2).random(12)
    flat = rewritten(tiled, canonicalize_schedule)
    assert nesting(flat.source) == [["ji"]] and "B[ji] = 2.0 * A[ji]" in flat.source

    def transform(tree: ast.Module) -> None:
        relayout_array(tree, "A", layout([Block("A", "i0", "4", ("o", "f"))], ["4*K"]))
        canonicalize_schedule(tree)

    blocked = rewritten(tiled, transform)
    assert "laid_A[jo, ji]" in blocked.source
    for function in (flat, blocked):
        b = np.zeros(12)
        function.function(a, b, K=3)
        np.testing.assert_allclose(b, 2 * a)


def test_a_loop_not_starting_at_zero_keeps_its_blocked_accesses() -> None:
    """Only a loop from 0 by 1 tiles into whole blocks; one from 1 is left as it is (and does not stall the pass)."""

    def transform(tree: ast.Module) -> None:
        relayout_array(tree, "A", layout([Block("A", "i0", "4", ("o", "f"))], ["N"]))
        canonicalize_schedule(tree)

    function = rewritten(interior, transform)
    assert "laid_A[i // 4, i % 4]" in function.source
    a = np.random.default_rng(3).random(12)
    b = np.zeros(12)
    function.function(a, b, N=12)
    np.testing.assert_allclose(b[1:-1], 2 * a[1:-1])


def test_a_nest_carrying_a_reduction_keeps_its_order() -> None:
    """Summing into one cell orders the floating-point additions: the nest is not reordered."""

    def transform(tree: ast.Module) -> None:
        relayout_array(tree, "A", layout([Permute("A", ("i1", "i0"))], ["N", "M"]))
        canonicalize_schedule(tree)

    function = rewritten(accumulate, transform)
    assert ["i", "j"] in nesting(function.source)
    a = np.random.default_rng(4).random((3, 5))
    s = np.zeros(1)
    function.function(a, s, N=3, M=5)
    np.testing.assert_allclose(s[0], a.sum())


if __name__ == "__main__":
    test_a_blocked_array_tiles_its_loop()
    test_a_permuted_array_puts_its_unit_stride_loop_innermost()
    test_a_tiled_loop_merges_over_flat_arrays_and_keeps_its_tiles_over_blocked_ones()
    test_a_loop_not_starting_at_zero_keeps_its_blocked_accesses()
    test_a_nest_carrying_a_reduction_keeps_its_order()
