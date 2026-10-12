# Copyright 2026 ETH Zurich and the relayout authors.
"""The numpy / mpi4py frontend: a program rewritten for laid-out arrays, run on the laid-out buffers,
gives what the original gives on the logical arrays."""

import ast
import importlib.util
import inspect
import textwrap
from collections.abc import Callable
from typing import Any

import numpy as np
import pytest

from relayout.convert import from_layout, to_layout
from relayout.frontends.python import LayoutError, rewrite
from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Block, Pad, Permute, Shuffle, run


def layout(ops: list, shape: tuple[str, ...]) -> IndexMap:
    return run(ops, [ArraySpec("A", len(shape))], (), "operations").layout("A", list(shape))


def rewritten(function: Callable[..., None], layouts: dict[str, IndexMap]) -> Callable[..., None]:
    """``function`` compiled from its rewritten source."""
    tree = rewrite(ast.parse(textwrap.dedent(inspect.getsource(function))), layouts)
    namespace: dict[str, object] = {"np": np}
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


def exchange(comm: Any, A: np.ndarray) -> None:  # an mpi4py communicator
    comm.Sendrecv(A[0, :], dest=0, recvbuf=A[2, :], source=0)


def mpi_or_skip() -> Any:  # the mpi4py.MPI module
    """mpi4py's ``MPI``, or a skip when the extra or an MPI library is missing."""
    if importlib.util.find_spec("mpi4py") is None:
        pytest.skip("no mpi4py (extra 'mpi')")
    try:
        from mpi4py import MPI  # optional extra, and it loads the MPI library
    except RuntimeError as exc:
        pytest.skip(f"mpi4py finds no MPI library: {str(exc).splitlines()[0]}")
    return MPI


def test_mpi_buffers_are_packed_and_unpacked() -> None:
    MPI = mpi_or_skip()

    sizes = {"N": 3, "M": 4}
    transposed = layout([Permute("A", ("i1", "i0"))], ("N", "M"))
    a = np.arange(12.0).reshape(3, 4)
    expected = a.copy()
    expected[2, :] = a[0, :]
    stored = to_layout(a, transposed, sizes)
    rewritten(exchange, {"A": transposed})(MPI.COMM_SELF, stored)
    np.testing.assert_array_equal(from_layout(stored, transposed, sizes), expected)


if __name__ == "__main__":
    test_element_accesses_follow_any_layout()
    test_whole_array_uses_go_through_the_logical_view()
    test_a_cut_array_used_whole_is_refused_by_its_line()
    try:
        test_mpi_buffers_are_packed_and_unpacked()
    except pytest.skip.Exception as skipped:
        print(f"skipped: {skipped}")
