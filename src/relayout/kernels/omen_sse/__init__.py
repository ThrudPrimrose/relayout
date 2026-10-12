"""OMEN scattering self-energy (SSE) kernel, the SC19 Gordon Bell example.

Ziogas et al., "A Data-Centric Approach to Extreme-Scale Ab initio Dissipative Quantum Transport
Simulations", SC'19, doi.org/10.1145/3295500.3357156. Sizes follow the hpcagent-bench
``scattering_self_energies`` manifest (default layout: every array k-outer, C order).
"""

import numpy as np

SIZES = {
    "S": dict(Nkz=2, NE=4, Nqz=2, Nw=2, N3D=2, NA=6, NB=2, Norb=3),
    "M": dict(Nkz=4, NE=16, Nqz=4, Nw=2, N3D=3, NA=128, NB=4, Norb=10),
}


def initialize(Nkz, NE, Nqz, Nw, N3D, NA, NB, Norb, seed=42):
    """Inputs in the default layout: neigh_idx, dH, G, D, and a zeroed Sigma."""
    rng = np.random.default_rng(seed)

    def cplx(*shape):
        return rng.random(shape) + 1j * rng.random(shape)

    neigh_idx = ((np.arange(NA)[:, None] + np.arange(NB)[None, :] - NB // 2) % NA).astype(np.int32)
    dH = cplx(NA, NB, N3D, Norb, Norb)
    G = cplx(Nkz, NE, NA, Norb, Norb)
    D = cplx(Nqz, Nw, NA, NB, N3D, N3D)
    Sigma = np.zeros((Nkz, NE, NA, Norb, Norb), np.complex128)
    return neigh_idx, dH, G, D, Sigma


def reference(neigh_idx, dH, G, D, Sigma):
    """Sigma += SSE(G, D); the closed form the SC19 sequence ends at.

    dHG drops q and j (it reads neither), and D is summed over q and contracted with dH over j
    before the multiplication, since the product is linear in dHD.
    """
    NE, Nw = G.shape[1], D.shape[1]
    dHG = np.einsum("keabxy,abiyz->keabixz", G[:, :, neigh_idx], dH)
    S = np.einsum("abjxy,wabij->wabixy", dH, D.sum(axis=0))
    for w in range(Nw):
        Sigma[:, w:] += np.einsum("keabixy,abiyz->keaxz", dHG[:, : NE - w], S[w])
