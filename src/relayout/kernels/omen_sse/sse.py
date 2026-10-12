"""SSE as written in OMEN: the eight-deep loop nest the SC19 sequence starts from."""

import dace

Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb = (
    dace.symbol(s, dtype=dace.int64, positive=True) for s in ("Nkz", "NE", "Nqz", "Nw", "N3D", "NA", "NB", "Norb")
)


@dace.program
def sse(
    neigh_idx: dace.int32[NA, NB],
    dH: dace.complex128[NA, NB, N3D, Norb, Norb],
    G: dace.complex128[Nkz, NE, NA, Norb, Norb],
    D: dace.complex128[Nqz, Nw, NA, NB, N3D, N3D],
    Sigma: dace.complex128[Nkz, NE, NA, Norb, Norb],
):
    for k in range(Nkz):
        for E in range(NE):
            for q in range(Nqz):
                for w in range(Nw):
                    for i in range(N3D):
                        for j in range(N3D):
                            for a in range(NA):
                                for b in range(NB):
                                    if E - w >= 0:
                                        dHG = G[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigma[k, E, a] = Sigma[k, E, a] + dHG @ dHD
