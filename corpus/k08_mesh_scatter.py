# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k08 mesh edge-residual scatter -- the Pad / Permute / Shuffle witness (OP2 airfoil ``res_calc``).

An unstructured-mesh edge residual: each edge carries a flux from its two incident cells, scattered
back onto those cells::

    f[e] = (x[c0[e]] - x[c1[e]]) * 0.5     (per-edge flux)
    y[c0[e]] += f[e] ;  y[c1[e]] -= f[e]   (edge -> cell scatter)

The scatter has write conflicts (two edges touch the same cell), so this port uses the RACE-FREE
padded-gather form (the OP2/coloring alternative): precompute, per cell, the table of incident edges
with their signs, valence-padded to a uniform width ``V`` (Pad -- a cell of degree ``< V`` gets its
unused slots filled with edge 0 and sign 0, which add nothing), then recast the scatter as a per-cell
gather-and-reduce::

    y[c] = sum_v  f[tab[c, v]] * tsg[c, v]

Two maps: the flux ``f`` is computed over edges into a single-writer transient (indirect gathers
``x[e0[e]]`` / ``x[e1[e]]``); then ``y`` is reduced over cells, the inner valence axis ``V`` folded
by a WCR map (``y`` pre-zeroed by ``run_closure``), gathering ``f`` through the flat incident-edge
table.

Layout decision -- the physical layout of the per-cell padded gather table ``tsg`` (the sign / weight
column of the ``(edge, sign)`` slot table):

    AoS  : ``[NC, V]``  -- a cell's V slots are contiguous   (identity)
    SoA  : ``[V, NC]``  -- slot ``v`` across all cells is contiguous (Permute)
    padded: ``[NC, V+p]`` -- the valence axis grown by trailing pad (Pad primitive)

Both are transparent (the Permute wrap-around transpose / the Pad's never-accessed trailing cells
preserve the result), so every candidate reproduces the numpy oracle; the sweep only picks the
physical layout.

Attribution:
  * M. Giles, G. Mudalige, et al., OP2 airfoil ``res_calc`` (OP-DSL/OP2-Common); Giles et al.,
    "Designing OP2 for GPU architectures," JPDC 2013.
  * A. Sulyok, G. Balogh, I. Reguly, G. Mudalige, "Locality optimized unstructured mesh algorithms
    on GPUs," JPDC 2019 (cell/edge reordering; the atomics-vs-gather trade).
  * SC26 layout-algebra paper (Pad = valence padding; Permute = AoS/SoA of the gather table).

A Shuffle candidate renumbers the per-cell field ``x`` by a closed-form cell bijection ``sigma``
(cyclic shift / reflection -- a mesh reorder). ``x`` is consumed through the runtime indirection
``x[e0[e]]``; the Shuffle renumbers the store (``x'[i] = x[sigma(i)]``) and composes the
inverse into that consumer (``x'[sigma^{-1}(e0[e])]``), so the reorder is transparent and ``x`` is
still bound as-is.

Deferred (honest scope): the incident-edge INDEX table is passed FLAT (``tabf[NC*V]``) rather than as
a 2-D ``[NC, V]`` array, because a 2-D array used as a frontend indirection INDEX mis-generates C++
after the nested-SDFG boundary widening the layout passes require (the index subscript lowers to the
uncompilable ``tab[std::make_tuple(c, v)]``). The Permute/Pad witnesses therefore act on the 2-D sign
table ``tsg``; the edge-index table stays a flat CSR-style gather index.
"""

import dace
import numpy


NC, NE, V = dace.symbol("NC"), dace.symbol("NE"), dace.symbol("V")


@dace.program
def mesh_edge_residual(
    x: dace.float64[NC],
    e0: dace.int64[NE],
    e1: dace.int64[NE],
    tabf: dace.int64[NC * V],
    tsg: dace.float64[NC, V],
    y: dace.float64[NC],
):
    """Per-edge flux into a single-writer transient, then a per-cell WCR gather-reduce over V."""
    f = dace.define_local([NE], dace.float64)
    for e in dace.map[0:NE] @ dace.ScheduleType.Sequential:
        f[e] = (x[e0[e]] - x[e1[e]]) * 0.5
    for c, v in dace.map[0:NC, 0:V] @ dace.ScheduleType.Sequential:
        y[c] += f[tabf[c * V + v]] * tsg[c, v]


def grid_edges(n):
    """An ``n x n`` structured quad mesh as an UNstructured edge list (cells = grid points; right +
    down neighbours), so cell/edge orderings are non-trivial. Returns ``(edges[NE, 2], NC)``."""
    idx = numpy.arange(n * n, dtype=numpy.int64).reshape(n, n)
    right = numpy.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], 1)
    down = numpy.stack([idx[:-1].ravel(), idx[1:].ravel()], 1)
    return numpy.concatenate([right, down]), n * n


def make_padded_gather(e0, e1, ncells):
    """Cell <- incident-edge table, valence-padded (the Pad primitive): turns the conflicting scatter
    into a per-cell gather (race-free by construction). Unused slots stay ``(edge 0, sign 0)`` -- they
    read a valid edge and contribute nothing. Returns ``(tab[NC, V], tsg[NC, V])``."""
    inc = numpy.concatenate([e0, e1])
    sgn = numpy.concatenate([numpy.ones(e0.size), -numpy.ones(e1.size)])
    eid = numpy.concatenate([numpy.arange(e0.size), numpy.arange(e1.size)])
    order = numpy.argsort(inc, kind="stable")
    inc, sgn, eid = inc[order], sgn[order], eid[order]
    deg = numpy.bincount(inc, minlength=ncells)
    v = int(deg.max())
    tab = numpy.zeros((ncells, v), dtype=numpy.int64)
    tsg = numpy.zeros((ncells, v))
    slot = numpy.arange(inc.size) - numpy.repeat(numpy.cumsum(deg) - deg, deg)
    tab[inc, slot] = eid
    tsg[inc, slot] = sgn
    return numpy.ascontiguousarray(tab), numpy.ascontiguousarray(tsg)


def oracle(x, e0, e1, tab, tsg):
    """The dense padded-gather residual (== the ``np.add.at`` scatter, addition commutes): the
    per-edge flux gathered per cell and signed-summed over the valence axis."""
    f = (x[e0] - x[e1]) * 0.5
    return {"y": (f[tab] * tsg).sum(axis=1)}


def make_inputs(n=8, seed=0):
    """Small ``n x n`` mesh (``NC = n*n`` cells). Returns the per-cell field ``x``, the edge endpoint
    tables ``e0`` / ``e1``, and the valence-padded gather table ``tab`` (edge ids) / ``tsg`` (signs)."""
    edges, ncells = grid_edges(n)
    rng = numpy.random.default_rng(seed)
    e0 = numpy.ascontiguousarray(edges[:, 0])
    e1 = numpy.ascontiguousarray(edges[:, 1])
    x = numpy.ascontiguousarray(rng.random(ncells))
    tab, tsg = make_padded_gather(e0, e1, ncells)
    return {"x": x, "e0": e0, "e1": e1, "tab": tab, "tsg": tsg}


def run_closure(inputs, n=8):
    """A ``run(sdfg) -> outputs`` closure for the sweep: flattens the incident-edge table for the indirect
    gather and binds a fresh zeroed ``y``."""
    nc = int(inputs["x"].shape[0])
    ne = int(inputs["e0"].shape[0])
    v = int(inputs["tab"].shape[1])
    tabf = numpy.ascontiguousarray(inputs["tab"].reshape(-1))

    def run(sdfg):
        y = numpy.zeros(nc)
        sdfg(
            x=inputs["x"].copy(),
            e0=inputs["e0"].copy(),
            e1=inputs["e1"].copy(),
            tabf=tabf.copy(),
            tsg=inputs["tsg"].copy(),
            y=y,
            NC=nc,
            NE=ne,
            V=v,
        )
        return {"y": y}

    return run
