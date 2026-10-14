# Copyright 2026 ETH Zurich and the relayout authors.
"""The numpy / mpi4py frontend: a program rewritten for laid-out arrays, run on the laid-out buffers,
gives what the original gives on the logical arrays."""

import ast
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from tests.conftest import LAYOUT_CASES, LAYOUT_SHAPE, mpi_or_skip
from relayout.convert import from_layout, to_layout
from relayout.frontends.python import LayoutError, Unzipped, Zipped, rewrite
from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Block, Pad, Permute, Shuffle, run


def layout(ops: list, shape: tuple[str, ...]) -> IndexMap:
    return run(ops, [ArraySpec("A", len(shape))], (), "operations").layout("A", list(shape))


def rewritten(function: Callable[..., None], layouts: dict[str, IndexMap], **regroup: object) -> Callable[..., None]:
    """``function`` compiled from its rewritten source."""
    tree = rewrite(ast.parse(textwrap.dedent(inspect.getsource(function))), layouts, **regroup)  # type: ignore[arg-type]
    namespace = dict(function.__globals__)  # the rewritten function sees its module, as the original does
    exec(compile(tree, "<rewritten>", "exec"), namespace)  # the test's own kernel
    rebuilt = namespace[function.__name__]
    assert callable(rebuilt)
    return rebuilt


def stencil(A: np.ndarray, B: np.ndarray, N: int, M: int) -> None:
    for i in range(N):
        for j in range(M):
            B[i, j] = A[i, j] * 2.0 + A[i, (j + 1) % M]


def test_element_accesses_follow_any_layout() -> None:
    sizes = {"N": 5, "M": 12}
    cut = layout([Block("A", "i1", "4", ("o", "f")), Shuffle("A", ("f",), ("3 - f",), ("3 - f",))], ("N", "M"))
    padded = layout([Pad("A", "i0", "2"), Permute("A", ("i1", "i0"))], ("N", "M"))
    a = np.random.default_rng(0).standard_normal((5, 12))
    expected = np.zeros((5, 12))
    stencil(a, expected, **sizes)
    stored_b = to_layout(np.zeros((5, 12)), padded, sizes)
    rewritten(stencil, {"A": cut, "B": padded})(to_layout(a, cut, sizes), stored_b, **sizes)
    np.testing.assert_array_equal(from_layout(stored_b, padded, sizes), expected)


def whole(A: np.ndarray, x: np.ndarray, y: np.ndarray, N: int, M: int) -> None:
    y[:] = A @ x + A.T @ x + A[1, :].sum() + A.shape[0]


def test_whole_array_uses_go_through_the_logical_view() -> None:
    sizes = {"N": 6, "M": 6}
    permuted = layout([Pad("A", "i1", "3"), Permute("A", ("i1", "i0"))], ("N", "M"))
    a = np.random.default_rng(1).standard_normal((6, 6))
    x = np.arange(6.0)
    expected = np.zeros(6)
    whole(a, x, expected, **sizes)
    got = np.zeros(6)
    rewritten(whole, {"A": permuted})(to_layout(a, permuted, sizes), x, got, **sizes)
    np.testing.assert_allclose(got, expected)


def test_a_cut_array_used_whole_is_refused_by_its_line() -> None:
    cut = layout([Block("A", "i1", "4", ("o", "f"))], ("N", "M"))
    with pytest.raises(LayoutError, match=r"line 2: 'A' is laid out by cuts"):
        rewritten(whole, {"A": cut})


def axpy3(x: np.ndarray, y: np.ndarray, out: np.ndarray, N: int) -> None:  # noqa: N803
    for i in range(N):
        out[i] = x[i] + 2.0 * y[i]


def test_zipped_and_unzipped_arrays_follow_their_new_arrays() -> None:
    """Zip x and y into p along a new last dimension (then SoA by a permutation) or a new first one; unzip
    out along its first dimension into one array per index."""
    rng = np.random.default_rng(4)
    x, y = rng.random(6), rng.random(6)
    expected = x + 2.0 * y
    soa = layout([Permute("A", ("i1", "i0"))], ("N", "2"))
    p = to_layout(np.stack([x, y], axis=-1), soa, {"N": 6})
    out = np.zeros(6)
    rewritten(axpy3, {"p": soa}, zipped={"x": Zipped("p", 0), "y": Zipped("p", 1)})(p, out, N=6)
    np.testing.assert_allclose(out, expected)
    out = np.zeros(6)
    rewritten(axpy3, {}, zipped={"x": Zipped("q", 0, dim=0), "y": Zipped("q", 1, dim=0)})(np.stack([x, y]), out, N=6)
    np.testing.assert_allclose(out, expected)

    def two_rows(out: np.ndarray, N: int) -> None:  # noqa: N803
        for i in range(N):
            out[0, i] = i
            out[1, i] = 2 * i

    out_a, out_b = np.zeros(4), np.zeros(4)
    rewritten(two_rows, {}, unzipped={"out": Unzipped(("out_a", "out_b"), 0)})(out_a, out_b, N=4)
    np.testing.assert_array_equal(out_b, 2 * np.arange(4.0))

    def picked(out: np.ndarray, which: np.ndarray, N: int) -> None:  # noqa: N803
        for i in range(N):
            out[which[i], i] = out[1 - which[i], i] + 1.0

    out_a, out_b = np.zeros(4), np.full(4, 5.0)
    which = np.array([0, 1, 0, 1])
    rewritten(picked, {}, unzipped={"out": Unzipped(("out_a", "out_b"), 0)})(out_a, out_b, which, N=4)
    np.testing.assert_array_equal(out_a, [6.0, 0.0, 6.0, 0.0])
    np.testing.assert_array_equal(out_b, [5.0, 1.0, 5.0, 1.0])


def exchange(comm: Any, A: np.ndarray, N: int, M: int) -> None:  # an mpi4py communicator; N, M: A's shape symbols
    comm.Sendrecv(A[0, :], dest=0, recvbuf=A[2, :], source=0)


#: The layout cases numpy views without a copy, which an mpi4py buffer of the Python frontend needs.
VIEWED_CASES = [case for case in LAYOUT_CASES if layout(LAYOUT_CASES[case], ("N", "M")).view() is not None]


@pytest.mark.mpi
@pytest.mark.parametrize("case", VIEWED_CASES)
def test_mpi_buffers_are_packed_and_unpacked(case: str) -> None:
    """A row sent from and received into an array in a viewed layout case moves the logical row."""
    MPI = mpi_or_skip()

    sizes = dict(zip(("N", "M"), LAYOUT_SHAPE, strict=True))
    laid = layout(LAYOUT_CASES[case], ("N", "M"))
    a = np.arange(float(np.prod(LAYOUT_SHAPE))).reshape(LAYOUT_SHAPE)
    expected = a.copy()
    expected[2, :] = a[0, :]
    stored = to_layout(a, laid, sizes)
    rewritten(exchange, {"A": laid})(MPI.COMM_SELF, stored, **sizes)
    np.testing.assert_array_equal(from_layout(stored, laid, sizes), expected)


if __name__ == "__main__":
    test_element_accesses_follow_any_layout()
    test_whole_array_uses_go_through_the_logical_view()
    test_a_cut_array_used_whole_is_refused_by_its_line()
    test_zipped_and_unzipped_arrays_follow_their_new_arrays()
    try:
        for name in VIEWED_CASES:
            test_mpi_buffers_are_packed_and_unpacked(name)
    except pytest.skip.Exception as skipped:
        print(f"skipped: {skipped}")
