"""Distributed OMEN SSE (energy -> momentum Alltoall). Run under ``mpirun -n 2 pytest tests/test_omen_sse_mpi.py``."""

import shutil
from pathlib import Path

import numpy as np
import pytest

from relayout.kernels import omen_sse

pytestmark = pytest.mark.mpi
HERE = Path(omen_sse.__file__).parent
# Nkz and NE must split evenly over the ranks.
SIZE = dict(omen_sse.SIZES["S"], Nkz=4, NE=4)


def comm_or_skip():
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("needs mpirun -n >= 2")
    return comm


def build_on_root(comm, build):
    """Compile on rank 0 only, then let the others load the same artifact."""
    from dace.sdfg.utils import distributed_compile

    return distributed_compile(build() if comm.Get_rank() == 0 else None, comm)


def local_problem(comm):
    """Every rank builds the same global inputs and slices its own G energies and Sigma momenta."""
    rank, P = comm.Get_rank(), comm.Get_size()
    neigh_idx, dH, G, D, Sigma = omen_sse.initialize(**SIZE)
    omen_sse.reference(neigh_idx, dH, G, D, Sigma)
    Nkzl, NEl = SIZE["Nkz"] // P, SIZE["NE"] // P
    Gl = G[:, rank * NEl : (rank + 1) * NEl].copy()
    expected = Sigma[rank * Nkzl : (rank + 1) * Nkzl]
    sizes = dict(
        P=P,
        Nkzl=Nkzl,
        NEl=NEl,
        Nqz=SIZE["Nqz"],
        Nw=SIZE["Nw"],
        N3D=SIZE["N3D"],
        NA=SIZE["NA"],
        NB=SIZE["NB"],
        Norb=SIZE["Norb"],
    )
    return dict(neigh_idx=neigh_idx, dH=dH, Gl=Gl, D=D), expected, sizes


def test_python_frontend_mpi():
    from relayout.kernels.omen_sse.sse_mpi import sse_mpi

    comm = comm_or_skip()
    args, expected, sizes = local_problem(comm)
    func = build_on_root(comm, lambda: sse_mpi.to_sdfg(simplify=True))
    Sigmal = np.zeros_like(expected)
    func(**args, Sigmal=Sigmal, NE=SIZE["NE"], **sizes)
    np.testing.assert_allclose(Sigmal, expected, rtol=1e-12)


@pytest.mark.fortran
@pytest.mark.skipif(shutil.which("flang") is None, reason="no LLVM flang on PATH")
def test_fortran_frontend_mpi(tmp_path_factory):
    import dace_fortran

    comm = comm_or_skip()
    args, expected, sizes = local_problem(comm)
    out = tmp_path_factory.mktemp("sdfg")
    func = build_on_root(
        comm,
        lambda: dace_fortran.build_sdfg(
            (HERE / "sse_mpi.f90").read_text(), entry="sse_mpi_mod::sse_mpi", name="omen_sse_mpi_f", out_dir=str(out)
        ),
    )
    fargs = {n: np.asfortranarray(v) for n, v in args.items()}
    Sigmal = np.zeros_like(expected, order="F")
    func(**fargs, Sigmal=Sigmal, **sizes)
    np.testing.assert_allclose(Sigmal, expected, rtol=1e-12)
