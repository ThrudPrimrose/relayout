"""Distributed SSE: G arrives energy-distributed (as the RGF phase produces it), SSE runs momentum-distributed.

Rank r owns energies [r*NEl, (r+1)*NEl) of G and momenta [r*Nkzl, (r+1)*Nkzl) of Sigma. One Alltoall moves
G between the two distributions. In the default k-outer layout the pack is contiguous (chunk d is
Gl[d*Nkzl:(d+1)*Nkzl]) and the unpack is the strided transpose.
"""

import dace

P, Nkzl, NEl, NE, Nqz, Nw, N3D, NA, NB, Norb = (
    dace.symbol(s, dtype=dace.int64, positive=True)
    for s in ("P", "Nkzl", "NEl", "NE", "Nqz", "Nw", "N3D", "NA", "NB", "Norb")
)


@dace.program
def sse_mpi(
    neigh_idx: dace.int32[NA, NB],
    dH: dace.complex128[NA, NB, N3D, Norb, Norb],
    Gl: dace.complex128[P * Nkzl, NEl, NA, Norb, Norb],
    D: dace.complex128[Nqz, Nw, NA, NB, N3D, N3D],
    Sigmal: dace.complex128[Nkzl, NE, NA, Norb, Norb],
):
    send = dace.define_local([P, Nkzl, NEl, NA, Norb, Norb], dace.complex128)
    recv = dace.define_local([P, Nkzl, NEl, NA, Norb, Norb], dace.complex128)
    Gk = dace.define_local([Nkzl, NE, NA, Norb, Norb], dace.complex128)
    for d, kl, el, a, x, y in dace.map[0:P, 0:Nkzl, 0:NEl, 0:NA, 0:Norb, 0:Norb]:
        send[d, kl, el, a, x, y] = Gl[d * Nkzl + kl, el, a, x, y]
    dace.comm.Alltoall(send, recv)
    for s, kl, el, a, x, y in dace.map[0:P, 0:Nkzl, 0:NEl, 0:NA, 0:Norb, 0:Norb]:
        Gk[kl, s * NEl + el, a, x, y] = recv[s, kl, el, a, x, y]
    for k in range(Nkzl):
        for E in range(NE):
            for q in range(Nqz):
                for w in range(Nw):
                    for i in range(N3D):
                        for j in range(N3D):
                            for a in range(NA):
                                for b in range(NB):
                                    if E - w >= 0:
                                        dHG = Gk[k, E - w, neigh_idx[a, b]] @ dH[a, b, i]
                                        dHD = dH[a, b, j] * D[q, w, a, b, i, j]
                                        Sigmal[k, E, a] = Sigmal[k, E, a] + dHG @ dHD
