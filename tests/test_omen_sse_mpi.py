"""Distributed OMEN SSE (energy -> momentum Alltoall). Run under ``mpirun -n 2 pytest tests/test_omen_sse_mpi.py``."""

import importlib.util
from pathlib import Path

import numpy as np
import pytest

from relayout.kernels import omen_sse

pytestmark = pytest.mark.mpi
HERE = Path(omen_sse.__file__).parent
# Nkz and NE must split evenly over the ranks.
SIZE = dict(omen_sse.SIZES["S"], Nkz=4, NE=4)


def flang_missing() -> bool:
    """The Fortran frontend needs the optional ``fortran`` extra and an LLVM flang."""
    if importlib.util.find_spec("dace_fortran") is None:
        return True
    from dace_fortran.llvm_toolchain import find_flang  # optional extra: imported only when installed

    return find_flang() is None


def comm_or_skip():
    MPI = pytest.importorskip("mpi4py.MPI")
    comm = MPI.COMM_WORLD
    if comm.Get_size() < 2:
        pytest.skip("needs mpirun -n >= 2")
    return comm


def build_on_root(comm, build):
    """Build and compile on rank 0 only, then let the others load the same artifact.

    Rank 0's build error is broadcast, so every rank raises it instead of the others waiting in the compile broadcast.
    """
    from dace.sdfg.utils import distributed_compile

    sdfg, error = None, None
    if comm.Get_rank() == 0:
        try:
            sdfg = build()
        except Exception as exc:  # re-raised on every rank below
            error = f"{type(exc).__name__}: {exc}"
    error = comm.bcast(error, root=0)
    if error is not None:
        raise RuntimeError(f"rank 0 failed to build the SDFG: {error}")
    return distributed_compile(sdfg, comm)


def local_problem(comm):
    """Every rank builds the same global inputs and slices its own G energies and Sigma momenta."""
    rank, P = comm.Get_rank(), comm.Get_size()
    neigh_idx, dH, G, D, Sigma = omen_sse.initialize(**SIZE)
    omen_sse.reference(neigh_idx, dH, G, D, Sigma)
    Nkzl, NEl = SIZE["Nkz"] // P, SIZE["NE"] // P
    Gl = G[:, rank * NEl : (rank + 1) * NEl].copy()
    expected = Sigma[rank * Nkzl : (rank + 1) * Nkzl]
    sizes = {
        "P": P,
        "Nkzl": Nkzl,
        "NEl": NEl,
        "Nqz": SIZE["Nqz"],
        "Nw": SIZE["Nw"],
        "N3D": SIZE["N3D"],
        "NA": SIZE["NA"],
        "NB": SIZE["NB"],
        "Norb": SIZE["Norb"],
    }
    return {"neigh_idx": neigh_idx, "dH": dH, "Gl": Gl, "D": D}, expected, sizes


def test_python_frontend_mpi():
    from relayout.kernels.omen_sse.sse_mpi import sse_mpi

    comm = comm_or_skip()
    args, expected, sizes = local_problem(comm)
    func = build_on_root(comm, lambda: sse_mpi.to_sdfg(simplify=True))
    Sigmal = np.zeros_like(expected)
    func(**args, Sigmal=Sigmal, NE=SIZE["NE"], **sizes)
    np.testing.assert_allclose(Sigmal, expected, rtol=1e-12)


@pytest.mark.fortran
@pytest.mark.skipif(flang_missing(), reason="no dace-fortran (extra 'fortran') or no LLVM flang")
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
    # Fortran names are case-insensitive: the SDFG declares them lower-case
    func(**{name.lower(): value for name, value in {**fargs, "Sigmal": Sigmal, **sizes}.items()})
    np.testing.assert_allclose(Sigmal, expected, rtol=1e-12)
