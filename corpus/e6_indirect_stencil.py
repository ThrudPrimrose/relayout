# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""e6 ICON indirect (connectivity-table) stencil -- the data-dependent-gather Permute witness.

E6_VelocityTendencies loopnest_1, the ICON semi-structured *indirect* stencil (paper Listing 2). k07
is the DIRECT structured sibling (a fixed +/-1 neighbour stencil); this one gathers its neighbours
through unstructured connectivity tables, so the layout question is different in kind.

For every edge ``je`` and level ``jk`` the tendency reads its two incident cell columns of ``w`` and
its two incident vertex columns of ``z_w_v`` through edge->cell / edge->vertex connectivity tables::

    w0 = w[cell_idx[je,0], jk]   ; w1 = w[cell_idx[je,1], jk]
    zw0 = z_w_v[vert_idx[je,0], jk] ; zw1 = z_w_v[vert_idx[je,1], jk]
    out[je,jk] = vn_ie[je,jk]*inv_dual[je]*(w0 - w1)
               + z_vt_ie[je,jk]*inv_primal[je]*tangent[je]*(zw0 - zw1)

It is pure elementwise per ``(je, jk)`` -- no reduction. The gathers ``w[cell_idx[je,c], jk]`` lower
through ``add_indirection_subgraph`` (an Indirection tasklet reading the whole field via a dynamic
memlet, with the connectivity entries fed in as index memlets); the index tables ``cell_idx`` /
``vert_idx`` are int64 inputs.

Two families of global layout decision are exposed, both transparent (the permute reaches into the
indirection's nested SDFG, so every candidate reproduces the oracle -- the sweep only picks the
physical layout):

  * the GATHERED fields ``w`` / ``z_w_v`` -- ``[N, NL]`` (horizontal-first, ICON (nproma, nlev)
    Fortran default) vs ``[NL, N]`` (vertical-first / level-contiguous). The indirection carries the
    permute because the whole gather lives in a nested SDFG whose memlets are permuted with the field.
  * the CONNECTIVITY tables ``cell_idx`` / ``vert_idx`` -- ``[NE, 2]`` (AoS: each edge's two neighbours
    adjacent) vs ``[2, NE]`` (SoA: neighbour-0 and neighbour-1 in separate contiguous streams). This
    is the classic connectivity-table storage decision for unstructured codes.


Source: G. Zaengl, D. Reinert, P. Ripodas, M. Baldauf, "The ICON modelling framework of DWD and
MPI-M," QJRMS 141, 2015 (dynamical core, edge/cell/vertex connectivity); M. Giorgetta et al., "The
ICON-A model for direct QBO simulations on GPUs," GMD 2022 ((nproma, nlev) vs level-contiguous order);
SC26 layout paper Listing 2 (E6_VelocityTendencies indirect stencil), connectivity AoS/SoA decision.
"""

import dace
import numpy


NC, NV, NE, NL = dace.symbol("NC"), dace.symbol("NV"), dace.symbol("NE"), dace.symbol("NL")


@dace.program
def indirect_stencil(
    w: dace.float64[NC, NL],
    z_w_v: dace.float64[NV, NL],
    cell_idx: dace.int64[NE, 2],
    vert_idx: dace.int64[NE, 2],
    vn_ie: dace.float64[NE, NL],
    z_vt_ie: dace.float64[NE, NL],
    inv_dual: dace.float64[NE],
    inv_primal: dace.float64[NE],
    tangent: dace.float64[NE],
    out: dace.float64[NE, NL],
):
    """One velocity-tendency edge stencil: gather two cell columns of ``w`` and two vertex columns of
    ``z_w_v`` through the connectivity tables, combine per (edge, level). Pure elementwise (no WCR)."""
    for je, jk in dace.map[0:NE, 0:NL] @ dace.ScheduleType.Sequential:
        out[je, jk] = vn_ie[je, jk] * inv_dual[je] * (w[cell_idx[je, 0], jk] - w[cell_idx[je, 1], jk]) + z_vt_ie[
            je, jk
        ] * inv_primal[je] * tangent[je] * (z_w_v[vert_idx[je, 0], jk] - z_w_v[vert_idx[je, 1], jk])


def oracle(w, z_w_v, cell_idx, vert_idx, vn_ie, z_vt_ie, inv_dual, inv_primal, tangent):
    """Pure-numpy vectorization of the indirect stencil: fancy-index the gathers, broadcast the
    per-edge scalars over the level axis, elementwise combine."""
    w0, w1 = w[cell_idx[:, 0], :], w[cell_idx[:, 1], :]
    zw0, zw1 = z_w_v[vert_idx[:, 0], :], z_w_v[vert_idx[:, 1], :]
    out = vn_ie * inv_dual[:, None] * (w0 - w1) + z_vt_ie * inv_primal[:, None] * tangent[:, None] * (zw0 - zw1)
    return {"out": out}


def make_inputs(nc, nv, ne, nl, seed=0):
    """Small synthetic ICON-shaped inputs: random fields plus random-but-valid int64 connectivity
    (each edge points at two in-range cells / vertices)."""
    rng = numpy.random.default_rng(seed)
    return {
        "w": rng.random((nc, nl)),
        "z_w_v": rng.random((nv, nl)),
        "cell_idx": rng.integers(0, nc, size=(ne, 2)).astype(numpy.int64),
        "vert_idx": rng.integers(0, nv, size=(ne, 2)).astype(numpy.int64),
        "vn_ie": rng.random((ne, nl)),
        "z_vt_ie": rng.random((ne, nl)),
        "inv_dual": rng.random(ne),
        "inv_primal": rng.random(ne),
        "tangent": rng.random(ne),
    }


def run_closure(inputs, nc, nv, ne, nl):
    """A ``run(sdfg) -> outputs`` closure for the sweep: binds copies of the inputs and fresh outputs each call."""

    def run(sdfg):
        args = {name: arr.copy() for name, arr in inputs.items()}
        out = numpy.zeros((ne, nl))
        sdfg(NC=nc, NV=nv, NE=ne, NL=nl, out=out, **args)
        return {"out": out}

    return run
