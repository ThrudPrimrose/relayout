# Copyright 2026 ETH Zurich and the relayout authors.
"""MPI library nodes on laid-out buffers: the transfer moves a packed copy in the logical order, so the
wire order is the original program's. Run under ``mpirun -n 2 pytest tests/test_frontend_sdfg_mpi.py``;
the structure check runs anywhere."""

import numpy as np
import pytest

import dace
from tests.conftest import LAYOUT_CASES, mpi_or_skip, variant
from dace import nodes
from relayout.frontends.sdfg import pack_mpi_buffers, relayout_array
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from relayout.ops import ArraySpec, Block, Pad, Permute, Shuffle, run

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


def check_exchange(sdfg: dace.SDFG, n: int) -> None:
    """``sdfg`` (the exchange, laid out) run on every rank moves row r of each rank's A to rank r."""
    comm = mpi_or_skip().COMM_WORLD
    size, rank = comm.Get_size(), comm.Get_rank()
    if size < 2:
        pytest.skip("needs mpirun -n >= 2")
    from dace.sdfg.utils import distributed_compile  # noqa: PLC0415  # MPI-only

    function = distributed_compile(sdfg, comm)
    a = np.arange(size * n, dtype=np.float64).reshape(size, n) + 100 * rank
    b = np.zeros((size, n))
    function(A=a, B=b, P=size, N=n)
    expected = np.stack([np.arange(rank * n, (rank + 1) * n) + 100.0 * source for source in range(size)])
    np.testing.assert_array_equal(b, expected)


def laid(ops: list) -> object:
    return run(ops, [ArraySpec("A", 2)], (), "ops").layout("A", ["P", "N"])


def movers(sdfg: dace.SDFG) -> tuple[list[str], int, set[str]]:
    """The pack / unpack maps, the copy nodes, and the arrays the Alltoall's edges name."""
    state = next(s for s in sdfg.states() if any(isinstance(n, nodes.LibraryNode) for n in s.nodes()))
    transfer = next(n for n in state.nodes() if isinstance(n, nodes.LibraryNode) and not isinstance(n, CopyLibraryNode))
    maps = sorted(n.label for n in state.nodes() if isinstance(n, nodes.MapEntry))
    copies = sum(isinstance(n, CopyLibraryNode) for n in state.nodes())
    return maps, copies, {e.data.data for e in state.all_edges(transfer)}


def test_one_layout_on_both_buffers_with_rows_outermost_moves_without_a_pack() -> None:
    """A shuffle inside each row on the send and the receive buffer: the wire carries the laid-out rows."""
    sdfg = exchange.to_sdfg(simplify=True)
    swizzle = laid([Block("A", "i1", "2", ("o", "f")), Shuffle("A", ("f",), ("1 - f",), ("1 - f",))])
    relayout_array(sdfg, "A", swizzle, pending := [])
    relayout_array(sdfg, "B", swizzle, pending)
    pack_mpi_buffers(pending)
    sdfg.validate()
    assert movers(sdfg) == ([], 0, {"laid_A", "laid_B"})


def test_a_strided_box_in_logical_order_packs_through_a_copy_node() -> None:
    """A padded receive buffer (rows in order, with dead cells) packs by a copy node, a transposed one by a map."""
    sdfg = exchange.to_sdfg(simplify=True)
    relayout_array(sdfg, "B", laid([Pad("A", "i1", "2")]))
    sdfg.validate()
    maps, copies, _ = movers(sdfg)
    assert (maps, copies) == ([], 1)
    assert movers(laid_out_exchange())[1] == 0


@pytest.mark.mpi
def test_the_exchange_moves_the_logical_rows() -> None:
    check_exchange(laid_out_exchange(), 3)


@pytest.mark.mpi
@pytest.mark.parametrize("both", [True, False], ids=["both_buffers", "send_only"])
@pytest.mark.parametrize("case", LAYOUT_CASES)
def test_every_layout_case_moves_the_logical_rows(case: str, both: bool) -> None:
    """The Alltoall's buffers in one layout case (both, so the pack may be left out, or the send buffer
    alone, so it is packed) move the logical rows."""
    sdfg = exchange.to_sdfg(simplify=True)
    variant(sdfg, f"case_{case}_{both}")
    layout = laid(LAYOUT_CASES[case])
    pending: list = []
    relayout_array(sdfg, "A", layout, pending)
    if both:
        relayout_array(sdfg, "B", layout, pending)
    pack_mpi_buffers(pending)
    check_exchange(sdfg, 6)


if __name__ == "__main__":
    test_a_transfer_of_a_laid_out_array_goes_through_a_packed_buffer()
    test_one_layout_on_both_buffers_with_rows_outermost_moves_without_a_pack()
    test_a_strided_box_in_logical_order_packs_through_a_copy_node()
    try:
        test_the_exchange_moves_the_logical_rows()
        for name in LAYOUT_CASES:
            test_every_layout_case_moves_the_logical_rows(name, True)
            test_every_layout_case_moves_the_logical_rows(name, False)
    except pytest.skip.Exception as skipped:
        print(f"skipped: {skipped}")
