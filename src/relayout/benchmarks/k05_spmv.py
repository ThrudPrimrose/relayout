# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k05 SpMV under formats and orderings -- the Pad + Block + Permute witness (ELL sparse gather).

Sparse matrix-vector product ``y[i] = sum_j val[j] * x[col[j]]`` (CSR). The port uses the ELL
(dense padded) form of the same graph: every row is padded to the maximum vertex degree ``W`` and
stored as a dense ``[NR, W]`` gather, where padded lanes point at the row's own index with value 0
(so they contribute ``0 * x[i] = 0`` and the ELL product equals the CSR product exactly)::

    y[i] = sum_k eval_[i, k] * x[ecol[i, k]]          (ELL, k over the W padded columns)

The padding to ``W = maxdeg`` is the **Pad** primitive; row tiling of the ELL matrix is **Block**;
the row-major ELL (``eval_[NR, W]``) vs column-major SELL slab (``eval_.T[W, NR]``) is **Permute**.
The gather ``x[ecol[i, k]]`` is a genuine data-dependent indirection (an ``add_indirection_subgraph``
lowering), which is what makes the sparse layout decision real rather than a dense reshape.

The kernel is written as gather-then-reduce (two maps over the same ``[0:NR, 0:W]`` space)::

    g[i, k] = x[ecol[i, k]]                 # gather: the only indirect map
    y[i]   += eval_[i, k] * g[i, k]         # reduce: a plain WCR map (mvt idiom, y pre-zeroed)

Separating the gather from the reduce keeps ``eval_`` out of the indirection's nested SDFG; ``ecol``
and ``x`` remain inside the indirection subgraph. The result is identical to the single-line SpMV;
the two maps only expose more of the ELL layout to the sweep.

Layout decisions the sweep explores: the storage orientation of the value/index matrices (row-major
ELL vs column-major SELL slab -- Permute), row tiling of the value matrix (Block), and padding the
ragged width up to a SIMD multiple (Pad). Every candidate is transparent, so all reproduce the
oracle; the sweep only picks the physical layout.

Source: M. Kreutzer, G. Hager, G. Wellein, H. Fehske, A. Bishop, "A Unified Sparse Matrix Data Format
for Efficient General SpMV on Modern Processors with Wide SIMD Units," SIAM J. Sci. Comput. 36(5),
2014 (SELL-C-sigma = Permute+Block+Pad+Permute); E. Cuthill, J. McKee, "Reducing the bandwidth of
sparse symmetric matrices," ACM 1969 (vertex ordering); SC26 layout-algebra paper SS IV (sparse
gather isolation: Pad width, Block rows, Permute ELL<->SELL).
"""

import dace
import numpy

from relayout.benchmarks import layouts_of
from relayout.ops import Pad
from relayout.search.layouts import Layout, blockings, permutations

NR = dace.symbol("NR")  # number of rows (matrix and output length)
NC = dace.symbol("NC")  # number of columns (length of the gathered vector x)
W = 4  # ELL width == max vertex degree of grid5 (an interior 5-point node has 4 neighbours)


@dace.program
def spmv_ell(ecol: dace.int64[NR, W], eval_: dace.float64[NR, W], x: dace.float64[NC], y: dace.float64[NR]):
    """ELL SpMV as gather-then-reduce: ``y[i] = sum_k eval_[i,k] * x[ecol[i,k]]`` (y pre-zeroed)."""
    g = numpy.empty((NR, W), dtype=numpy.float64)  # per-row gathered x values (transient, fully written)
    for i, k in dace.map[0:NR, 0:W]:
        g[i, k] = x[ecol[i, k]]  # the indirect gather (data-dependent read of x)
    for i, k in dace.map[0:NR, 0:W] @ dace.ScheduleType.Sequential:
        y[i] += eval_[i, k] * g[i, k]  # WCR reduction over the W columns


# graph -> ELL (numpy spec) ---------
def grid5(side):
    """2D ``side x side`` 5-point Laplacian as a symmetric edge list (upper triangle), ``NR = side^2``."""
    idx = numpy.arange(side * side).reshape(side, side)
    right = numpy.stack([idx[:, :-1].ravel(), idx[:, 1:].ravel()], axis=1)
    down = numpy.stack([idx[:-1, :].ravel(), idx[1:, :].ravel()], axis=1)
    return numpy.concatenate([right, down]), side * side


def to_csr(edges, n):
    """Symmetrize the edge list into CSR arrays ``(indptr, col, val, row-of-nnz)``."""
    src = numpy.concatenate([edges[:, 0], edges[:, 1]])
    dst = numpy.concatenate([edges[:, 1], edges[:, 0]])
    order = numpy.lexsort((dst, src))
    src, dst = src[order], dst[order]
    indptr = numpy.zeros(n + 1, dtype=numpy.int64)
    numpy.add.at(indptr, src + 1, 1)
    indptr = numpy.cumsum(indptr)
    val = numpy.ones(dst.shape[0], dtype=numpy.float64)
    return indptr, numpy.ascontiguousarray(dst), val, numpy.ascontiguousarray(src)


def make_ell(indptr, col, val, n):
    """Pad CSR to a dense ``(n, maxdeg)`` ELL: padded columns point at the row's own index, value 0."""
    deg = numpy.diff(indptr)
    w = int(deg.max())
    ecol = numpy.repeat(numpy.arange(n, dtype=numpy.int64)[:, None], w, axis=1)
    eval_ = numpy.zeros((n, w), dtype=numpy.float64)
    live = numpy.arange(w)[None, :] < deg[:, None]
    ecol[live] = col
    eval_[live] = val
    return numpy.ascontiguousarray(ecol), numpy.ascontiguousarray(eval_), w


def spmv_csr(indptr, col, val, row, x, n):
    """Reference CSR SpMV, used only to cross-check the ELL oracle in the sweep driver."""
    return numpy.bincount(row, weights=val * x[col], minlength=n)


def oracle(ecol, eval_, x):
    """The ELL SpMV ``y = (eval_ * x[ecol]).sum(axis=1)`` -- equal to ``spmv_csr`` on the same graph."""
    return {"y": (eval_ * x[ecol]).sum(axis=1)}


def make_inputs(side=8, seed=0):
    """Build the grid5 graph, convert to CSR, pad to ELL, and draw a random ``x`` (int64 index array)."""
    edges, n = grid5(side)
    indptr, col, val, _row = to_csr(edges, n)
    ecol, eval_, w = make_ell(indptr, col, val, n)
    if w != W:
        raise ValueError(f"k05_spmv: grid5(side={side}) has max degree {w}, expected W={W}")
    x = numpy.random.default_rng(seed).random(n)
    return {"ecol": ecol, "eval_": eval_, "x": x}


def candidates():
    """The global layout candidates for the ELL matrices:

      * ``eval_:perm10`` / ``ecol:perm10`` -- row-major ELL ``[NR, W]`` vs column-major SELL slab ``[W, NR]``
        (Permute of the value / index matrix).
      * ``eval_:block0x*`` -- row tiling of the value matrix (Block of the row axis ``NR``).
      * ``eval_:pad_w`` -- grow the ragged width ``W`` to ``2 * W`` lanes (Pad); the extra lanes are never
        read (the map iterates ``0:W``), so the pad is transparent.

    Every candidate is transparent, so all reproduce the oracle."""
    cands = layouts_of("eval_", permutations("eval_", 2))
    cands.update(layouts_of("ecol", permutations("ecol", 2)))
    cands.update(layouts_of("eval_", blockings("eval_", 2, (8, 16, 32))))
    cands.update(layouts_of("eval_", [Layout("pad_w", (Pad("eval_", "i1", str(W)),))]))
    return cands


def run_closure(inputs, side):
    """A ``run(sdfg) -> {"y": ...}`` closure for the sweep: binds the inputs and a fresh zeroed ``y``."""
    n = side * side

    def run(sdfg):
        y = numpy.zeros(n)
        sdfg(ecol=inputs["ecol"].copy(), eval_=inputs["eval_"].copy(), x=inputs["x"].copy(), y=y, NR=n, NC=n)
        return {"y": y}

    return run
