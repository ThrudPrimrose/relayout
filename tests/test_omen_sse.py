"""Single-process OMEN SSE: the Python and Fortran frontends against the numpy closed form."""

from pathlib import Path

import numpy as np
import pytest
from dace_fortran.llvm_toolchain import find_flang

from relayout.kernels import omen_sse
from relayout.kernels.omen_sse.sse import sse

HERE = Path(omen_sse.__file__).parent
SIZE = omen_sse.SIZES["S"]


def naive(neigh_idx, dH, G, D, Sigma):
    """The OMEN loop nest in plain numpy; pins the closed form in ``reference``."""
    Nkz, NE, _, _, _ = G.shape
    Nqz, Nw, NA, NB, N3D, _ = D.shape
    for k in range(Nkz):
        for E in range(NE):
            for q in range(Nqz):
                for w in range(min(Nw, E + 1)):
                    for i in range(N3D):
                        for j in range(N3D):
                            for a in range(NA):
                                for b in range(NB):
                                    dHG = G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]
                                    Sigma[k, E, a] += dHG @ (dH[a, b, j] * D[q, w, a, b, i, j])


def test_reference_matches_loop_nest():
    args = omen_sse.initialize(**SIZE)
    expected = args[-1].copy()
    naive(*args[:-1], expected)
    omen_sse.reference(*args)
    np.testing.assert_allclose(args[-1], expected, rtol=1e-12)


def test_python_frontend():
    neigh_idx, dH, G, D, Sigma = omen_sse.initialize(**SIZE)
    expected = Sigma.copy()
    omen_sse.reference(neigh_idx, dH, G, D, expected)
    sdfg = sse.to_sdfg(simplify=True)
    sdfg(neigh_idx=neigh_idx, dH=dH, G=G, D=D, Sigma=Sigma, **SIZE)
    np.testing.assert_allclose(Sigma, expected, rtol=1e-12)


@pytest.mark.fortran
@pytest.mark.skipif(find_flang() is None, reason="no LLVM flang found")
@pytest.mark.xfail(
    strict=True,
    reason="dace-fortran main cannot build MATMUL over a data-indexed section, into a section destination or inside "
    "an expression; fixed by spcl/dace-fortran#5",
)
def test_fortran_frontend(tmp_path):
    import dace_fortran

    sdfg = dace_fortran.build_sdfg(
        (HERE / "sse.f90").read_text(), entry="sse_mod::sse", name="omen_sse_f", out_dir=str(tmp_path / "sdfg")
    )
    neigh_idx, dH, G, D, Sigma = omen_sse.initialize(**SIZE)
    expected = Sigma.copy()
    omen_sse.reference(neigh_idx, dH, G, D, expected)
    # Column-major SDFG: hand it the same logical arrays in Fortran order.
    fargs = {
        n: np.asfortranarray(v) for n, v in {"neigh_idx": neigh_idx, "dH": dH, "G": G, "D": D, "Sigma": Sigma}.items()
    }
    # Fortran names are case-insensitive: the SDFG declares them lower-case
    sdfg(**{name.lower(): value for name, value in {**fargs, **SIZE}.items()})
    np.testing.assert_allclose(fargs["Sigma"], expected, rtol=1e-12)
