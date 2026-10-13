# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""E1 matrix add -- the SC26 Fig.4 running example (dense two-input elementwise add).

    C[i, j] = A[i, j] + B[i, j]        (dense elementwise add over an [N, N] domain)

This is the paper's running example (SC26 Fig.4, E1_MatrixAdd): the simplest dense kernel that still
carries a real layout decision. Both operands are read contiguously under row-major, so no layout is
forced -- which makes it the honest instrument for the *read* side of the algebra. The decision here
is on operand ``B``: how it is physically stored while ``A`` and ``C`` stay logical [N, N].

    row-major : addr(i, j) = i*N + j             (baseline, identity Permute)
    permuted  : B stored column-major (Permute)  (transparent dim swap of the read operand)
    blocked   : B in [T, N] row panels         (Block, panel height T)

The sweep exercises the Permute family (row- vs column-major storage of ``B``) and the Block family
(a Block cuts the rows of ``B`` into contiguous ``[T, N]`` panels). Every candidate is transparent
-- the permute wraps each read memlet subset, the block re-tiles the read -- so all reproduce the
oracle; the sweep only picks the physical layout of ``B``. Structurally this is k02 (blocked
transpose) minus the transposed write: a plain contiguous ``C`` output.

Source: SC26 layout-algebra paper Fig.4 (E1_MatrixAdd running example, Permute + Block primitives on
the read operand); Chatterjee et al., "Nonlinear array layouts for hierarchical memory systems,"
ICS'99 (blocked layout).
"""

import dace
import numpy

from relayout.benchmarks import layouts_of
from relayout.search.layouts import blockings, permutations

N = dace.symbol("N")


@dace.program
def matrix_add(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N] @ dace.ScheduleType.Sequential:
        C[i, j] = A[i, j] + B[i, j]


def oracle(A, B):
    return {"C": A + B}


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"A": rng.random((n, n)), "B": rng.random((n, n))}


def candidates():
    """The Permute family (row- vs column-major storage of ``B``) and the Block family (rows of ``B`` in
    contiguous ``[T, N]`` panels for T in {8, 16, 32})."""
    cands = layouts_of("B", permutations("B", 2))
    cands.update(layouts_of("B", blockings("B", 2, (8, 16, 32))))
    return cands


def run_closure(inputs, n):
    """A ``run(sdfg) -> outputs`` closure for the sweep: binds a fresh output each call."""

    def run(sdfg):
        C = numpy.zeros((n, n))
        sdfg(A=inputs["A"].copy(), B=inputs["B"].copy(), C=C, N=n)
        return {"C": C}

    return run
