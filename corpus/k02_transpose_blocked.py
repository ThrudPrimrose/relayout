# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k02 blocked transpose -- the Block/Permute witness (out-of-place 2D transpose).

    B[j, i] = A[i, j]        (out-of-place transpose)

The write scatters at stride N under row-major, so no single loop order serves both the contiguous
reads of ``A`` and the contiguous writes of ``B`` -- the classic transpose memory problem. The
layout decision is how ``A`` is stored:

    row-major : addr(i, j) = i*N + j             (baseline, strided writes)
    blocked   : A reshaped [N/T, T, N] (Block)   (T x T tiles stored contiguously)
    permuted  : A stored column-major (Permute)  (transparent dim swap)

The sweep exercises the Block family (a Block cuts the rows of ``A`` into contiguous
``[T, N]`` panels) and the Permute family (a transparent dimension swap of ``A``) -- both preserve the
result, so every candidate reproduces the oracle; the sweep only picks the physical layout.

Source: Springer, Su, Bientinesi, "HPTT," ARRAY@PLDI'17 (reference transposer); Chatterjee et al.,
"Nonlinear array layouts for hierarchical memory systems," ICS'99 (4D blocked layout); SC26 layout
paper SS IV-B2 (transpose microbenchmark, Block primitive).
"""

import dace
import numpy


N = dace.symbol("N")


@dace.program
def transpose(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        B[j, i] = A[i, j]


def oracle(A):
    return {"B": A.T.copy()}


def make_inputs(n, seed=0):
    return {"A": numpy.random.default_rng(seed).random((n, n))}


def run_closure(inputs, n):
    """A ``run(sdfg) -> outputs`` closure for the sweep: binds a fresh output each call."""

    def run(sdfg):
        B = numpy.zeros((n, n))
        sdfg(A=inputs["A"].copy(), B=B, N=n)
        return {"B": B}

    return run
