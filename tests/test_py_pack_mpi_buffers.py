# Copyright 2026 ETH Zurich and the relayout authors.
"""mpi4py calls on laid-out buffers: the transfer moves a packed copy in the logical order, so the wire order is
the original function's; the counterpart of tests/test_sdfg_pack_mpi_buffers.py. A one-rank stand-in for the
communicator (raw byte copies of C-contiguous buffers, as mpi4py moves them; an ``Irecv`` lands at its ``Wait``)
checks every layout case anywhere; run ``mpirun -n 2 pytest -m mpi tests/test_py_pack_mpi_buffers.py`` for the
real exchange."""

import ast
from typing import Any

import numpy as np
import pytest

from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Block, Pad, Permute, Shuffle, run
from relayout.passes.py.common import LayoutError
from relayout.passes.py.pack_mpi_buffers import MpiBuffer, pack_mpi_buffers
from relayout.passes.py.common import kernel
from relayout.passes.py.relayout_array import relayout_array
from tests.conftest import LAYOUT_CASES, mpi_or_skip, rewritten


def laid(ops: list) -> IndexMap:
    return run(ops, [ArraySpec("A", 2)], (), "ops").layout("A", ["P", "N"])


class Request:
    def __init__(self, comm: "OneRank", buffer: np.ndarray) -> None:
        self.comm, self.buffer = comm, buffer

    def Wait(self) -> None:  # noqa: N802  # mpi4py's name
        self.comm.land(self.buffer)


class OneRank:
    """A communicator of one rank: every transfer copies the raw cells of C-contiguous buffers, as mpi4py does."""

    def __init__(self) -> None:
        self.mailbox: list[np.ndarray] = []

    @staticmethod
    def check(*buffers: np.ndarray) -> None:
        assert all(b.flags.c_contiguous for b in buffers), "mpi4py takes contiguous buffers only"

    def land(self, buffer: np.ndarray) -> None:
        self.check(buffer)
        buffer.reshape(-1)[...] = self.mailbox.pop(0)

    def Get_size(self) -> int:  # noqa: N802
        return 1

    def Get_rank(self) -> int:  # noqa: N802
        return 0

    def Alltoall(self, sendbuf: np.ndarray, recvbuf: np.ndarray) -> None:  # noqa: N802
        self.check(sendbuf, recvbuf)
        recvbuf.reshape(-1)[...] = sendbuf.reshape(-1)

    def Send(self, buf: np.ndarray, dest: int) -> None:  # noqa: N802
        self.check(buf)
        self.mailbox.append(buf.reshape(-1).copy())

    def Irecv(self, buf: np.ndarray, source: int) -> Request:  # noqa: N802
        return Request(self, buf)


def exchange(comm: Any, A: np.ndarray, B: np.ndarray, P: int, N: int) -> None:  # noqa: N803
    comm.Alltoall(A, B)


def posted(comm: Any, A: np.ndarray, B: np.ndarray, P: int, N: int) -> None:  # noqa: N803
    request = comm.Irecv(B[1, :], source=0)
    comm.Send(A[0, :], dest=0)
    request.Wait()


def unwaited(comm: Any, A: np.ndarray, B: np.ndarray, P: int, N: int) -> None:  # noqa: N803
    request = comm.Irecv(B[1, :], source=0)
    return request


TRANSPOSED = [Permute("A", ("i1", "i0"))]
PADDED = [Pad("A", "i1", "2"), Permute("A", ("i1", "i0"))]
SWIZZLED = [Block("A", "i1", "2", ("o", "f")), Shuffle("A", ("f",), ("1 - f",), ("1 - f",))]


def laid_out(function: Any, a_ops: list, b_ops: list | None, together: bool = True) -> Any:
    """``function`` with A (and B) laid out, the MPI buffers packed once both are (``together``) or each at once."""

    def transform(tree: ast.Module) -> None:
        pending: list[MpiBuffer] | None = [] if together else None
        relayout_array(tree, "A", laid(a_ops), pending=pending)
        if b_ops is not None:
            relayout_array(tree, "B", laid(b_ops), pending=pending)
        if pending is not None:
            pack_mpi_buffers(kernel(tree), pending)

    return rewritten(function, transform)


def run_exchange(function: Any, comm: Any, size: int, rank: int, n: int) -> None:
    """Row r of each rank's A lands on rank r."""
    a = np.arange(size * n, dtype=np.float64).reshape(size, n) + 100 * rank
    b = np.zeros((size, n))
    function(comm, a, b, P=size, N=n)
    expected = np.stack([np.arange(rank * n, (rank + 1) * n) + 100.0 * source for source in range(size)])
    np.testing.assert_array_equal(b, expected)


def test_a_transfer_of_a_laid_out_array_goes_through_a_packed_buffer() -> None:
    function = laid_out(exchange, TRANSPOSED, PADDED)
    assert "comm.Alltoall(relayout_packed_0, relayout_packed_1)" in function.source
    assert "relayout_packed_0 = np.ascontiguousarray(laid_A.transpose(1, 0)[0:P, 0:N])" in function.source
    assert "laid_B[:N, :].transpose(1, 0)[0:P, 0:N][...] = relayout_packed_1" in function.source
    assert "laid_A.transpose(1, 0)[0:P, 0:N][...]" not in function.source  # a send buffer is not unpacked
    assert "A[__r0, __r1] = laid_A" not in function.source  # nor converted out: the send writes nothing
    run_exchange(function.function, OneRank(), 1, 0, 3)


def test_one_layout_on_both_buffers_with_rows_outermost_moves_without_a_pack() -> None:
    """A shuffle inside each row on the send and the receive buffer: the wire carries the laid-out rows."""
    function = laid_out(exchange, SWIZZLED, SWIZZLED)
    assert "comm.Alltoall(laid_A, laid_B)" in function.source and "relayout_packed" not in function.source
    run_exchange(function.function, OneRank(), 1, 0, 4)
    packed_one_by_one = laid_out(exchange, SWIZZLED, SWIZZLED, together=False)
    assert "relayout_packed" in packed_one_by_one.source  # each array alone cannot see its partner's layout


def test_a_cut_layout_packs_through_loops_and_an_irecv_unpacks_after_its_wait() -> None:
    function = laid_out(posted, SWIZZLED, LAYOUT_CASES["block_ragged"])
    waited = function.source.index("request.Wait()")
    assert function.source.index("laid_B[1, ", waited) > waited  # the unpack comes after the wait
    a = np.random.default_rng(0).random((3, 6))
    b = np.zeros((3, 6))
    function.function(OneRank(), a, b, P=3, N=6)
    np.testing.assert_array_equal(b[1], a[0])
    with pytest.raises(LayoutError, match="no Wait for the request"):
        laid_out(unwaited, TRANSPOSED, TRANSPOSED)


@pytest.mark.parametrize("both", [True, False], ids=["both_buffers", "send_only"])
@pytest.mark.parametrize("case", LAYOUT_CASES)
def test_every_layout_case_moves_the_logical_rows_on_one_rank(case: str, both: bool) -> None:
    """On one rank the Alltoall copies the whole buffer: of several rows, so a buffer in another order shows."""
    function = laid_out(exchange, LAYOUT_CASES[case], LAYOUT_CASES[case] if both else None)
    a = np.random.default_rng(11).random((4, 6))
    b = np.zeros((4, 6))
    function.function(OneRank(), a, b, P=4, N=6)
    np.testing.assert_array_equal(b, a)


def check_exchange(case: str, both: bool) -> None:
    comm = mpi_or_skip().COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
    if size < 2:
        pytest.skip("needs mpirun -n >= 2")
    function = laid_out(exchange, LAYOUT_CASES[case], LAYOUT_CASES[case] if both else None)
    run_exchange(function.function, comm, size, rank, 6)


@pytest.mark.mpi
@pytest.mark.parametrize("both", [True, False], ids=["both_buffers", "send_only"])
@pytest.mark.parametrize("case", LAYOUT_CASES)
def test_every_layout_case_moves_the_logical_rows(case: str, both: bool) -> None:
    """The Alltoall's buffers in one layout case (both, so the pack may be left out, or the send buffer alone, so
    it is packed) move the logical rows between real ranks."""
    check_exchange(case, both)


if __name__ == "__main__":
    test_a_transfer_of_a_laid_out_array_goes_through_a_packed_buffer()
    test_one_layout_on_both_buffers_with_rows_outermost_moves_without_a_pack()
    test_a_cut_layout_packs_through_loops_and_an_irecv_unpacks_after_its_wait()
    for name in LAYOUT_CASES:
        test_every_layout_case_moves_the_logical_rows_on_one_rank(name, True)
        test_every_layout_case_moves_the_logical_rows_on_one_rank(name, False)
    try:
        for name in LAYOUT_CASES:
            test_every_layout_case_moves_the_logical_rows(name, True)
            test_every_layout_case_moves_the_logical_rows(name, False)
    except pytest.skip.Exception as skipped:
        print(f"skipped: {skipped}")
