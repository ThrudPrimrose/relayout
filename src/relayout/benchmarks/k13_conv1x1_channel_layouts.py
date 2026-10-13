# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k13 1x1 convolution -- NCHW vs NHWC vs nChw16c channel layouts.

    out[n,k,h,w] = sum_c Wt[k,c] * X[n,c,h,w]

A 1x1 convolution is a per-pixel matmul over the channel axis, so the layout decision is the
ORIENTATION of the channel reduction in the activation tensor ``X``:

    NCHW    (N, C, H, W): channels strided by H*W per pixel (identity).
    NHWC    (N, H, W, C): channels contiguous per pixel -> pixel-major GEMM (a Permute of X).
    nChw16c (N, C/16, H, W, 16): 16-channel SIMD panels contiguous per pixel (a Block of X's C axis).

Every layout is transparent, so every dimension order of X reproduces the oracle; NCHW is the identity
and NHWC is the permutation (0,2,3,1). The nChw16c Block cuts X's C axis and moves the 16-lane panel to
the last axis, the physical ``[N, C/16, H, W, 16]`` layout. The reduction (not BLAS gemm) form keeps the
layout honest -- a BLAS gemm would pack its operands internally and hide the channel stride.

Source: Intel oneDNN "Understanding Memory Formats" (nChw16c); NVIDIA cuDNN (NHWC tensor-core convs);
Georganas et al., "Anatomy of High-Performance Deep Learning Convolutions on SIMD Architectures,"
SC'18 (blocked conv layouts).
"""

import itertools

import dace
import numpy

from relayout.benchmarks import layouts_of
from relayout.ops import Block, Permute
from relayout.search.layouts import IDENTITY, Layout

N, C, H, W, K = (dace.symbol(s) for s in ("N", "C", "H", "W", "K"))
CBLOCK = 16  # nChw16c SIMD-lane panel width (C must be divisible by it for the Block candidate)


@dace.program
def conv1x1(X: dace.float64[N, C, H, W], Wt: dace.float64[K, C], out: dace.float64[N, K, H, W]):
    for n, k, h, w, c in dace.map[0:N, 0:K, 0:H, 0:W, 0:C] @ dace.ScheduleType.Sequential:
        out[n, k, h, w] += Wt[k, c] * X[n, c, h, w]


def oracle(X, Wt):
    return {"out": numpy.einsum("kc,nchw->nkhw", Wt, X)}


def make_inputs(n, c, h, w, k, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"X": rng.random((n, c, h, w)), "Wt": rng.random((k, c))}


def candidates():
    """Permute family over X (NCHW and its 23 other dimension orders) plus the nChw16c Block of X's C axis."""
    orders = [order for order in itertools.permutations(range(4)) if order != (0, 1, 2, 3)]
    permuted = [Layout(f"perm{''.join(map(str, o))}", (Permute("X", tuple(f"i{d}" for d in o)),)) for o in orders]
    panels = (Block("X", "i1", str(CBLOCK), ("cb", "cl")), Permute("X", ("i0", "cb", "i2", "i3", "cl")))
    return layouts_of("X", [Layout(IDENTITY), *permuted, Layout(f"nChw{CBLOCK}c", panels)])


def run_closure(inputs, n, c, h, w, k):
    """A ``run(sdfg) -> outputs`` closure for the sweep: binds a fresh zeroed ``out`` each call."""

    def run(sdfg):
        out = numpy.zeros((n, k, h, w))
        sdfg(X=inputs["X"].copy(), Wt=inputs["Wt"].copy(), out=out, N=n, C=c, H=h, W=w, K=k)
        return {"out": out}

    return run
