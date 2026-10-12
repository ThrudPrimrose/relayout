# Copyright 2026 ETH Zurich and the relayout authors.
"""MPI library nodes on laid-out buffers: the transfer moves a packed copy in the logical order, so the
wire order is the original program's. Run under ``mpirun -n 2 pytest tests/test_frontend_sdfg_mpi.py``;
the structure check runs anywhere."""

import numpy as np
import pytest

import dace
from conftest import mpi_or_skip
from dace import nodes
from relayout.frontends.sdfg import relayout_array
from relayout.ops import ArraySpec, Pad, Permute, run

P, N = (dace.symbol(s) for s in ("P", "N"))


@dace.program
def exchange(A: dace.float64[P, N], B: dace.float64[P, N]):  # noqa: N803
    dace.comm.Alltoall(A, B)


def laid_out_exchange() -> dace.SDFG:
    sdfg = exchange.to_sdfg(simplify=True)
    transposed = run([Permute("A", ("i1", "i0"))], [ArraySpec("A", 2)], (), "ops").layout("A", ["P", "N"])
    padded = run([Pad("A", "i1", "2"), Permute("A", ("i1", "i0"))], [ArraySpec("A", 2)], (), "ops").layout(
        "A", ["P", "N"]
    )
    relayout_array(sdfg, "A", transposed)
    relayout_array(sdfg, "B", padded)
    return sdfg


def test_a_transfer_of_a_laid_out_array_goes_through_a_packed_buffer() -> None:
    sdfg = laid_out_exchange()
    sdfg.validate()
    state = next(s for s in sdfg.states() if any(isinstance(n, nodes.LibraryNode) for n in s.nodes()))
    transfer = next(n for n in state.nodes() if isinstance(n, nodes.LibraryNode))
    buffers = {e.data.data for e in state.all_edges(transfer)}
    assert all(name.startswith("packed_laid_") for name in buffers), buffers
    maps = sorted(n.label for n in state.nodes() if isinstance(n, nodes.MapEntry))
    assert maps == ["pack_packed_laid_A_map", "unpack_packed_laid_B_map"]


@pytest.mark.mpi
def test_the_exchange_moves_the_logical_rows() -> None:
    comm = mpi_or_skip().COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
    if size < 2:
        pytest.skip("needs mpirun -n >= 2")
    from dace.sdfg.utils import distributed_compile  # noqa: PLC0415  # MPI-only

    function = distributed_compile(laid_out_exchange(), comm)
    n = 3
    a = np.arange(size * n, dtype=np.float64).reshape(size, n) + 100 * rank
    b = np.zeros((size, n))
    function(A=a, B=b, P=size, N=n)
    expected = np.stack([np.arange(rank * n, (rank + 1) * n) + 100.0 * source for source in range(size)])
    np.testing.assert_array_equal(b, expected)


if __name__ == "__main__":
    test_a_transfer_of_a_laid_out_array_goes_through_a_packed_buffer()
    try:
        test_the_exchange_moves_the_logical_rows()
    except pytest.skip.Exception as skipped:
        print(f"skipped: {skipped}")
