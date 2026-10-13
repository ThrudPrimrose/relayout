# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k12 Dimension-Lifted Transposition (DLT) -- a 1D 3-point stencil under a blocked layout.

    y[i] = w0*x[i-1] + w1*x[i] + w2*x[i+1]

The DLT layout cuts the flat array ``x`` into ``[N/V, V]`` (Block) and transposes it to ``[V, N/V]``
(Permute), so the V lanes of a vector read V consecutive blocks and the three shifted streams
``x[i-1] / x[i] / x[i+1]`` stay aligned. The sweep exercises that layout over ``x`` for each V
(a boundary-crossing neighbor access: the offset indices ``x[i-1]``/``x[i+1]`` cross block borders).

Source: Henretty et al., CC'11 (data-layout transformation for SIMD stencils).
"""

import dace
import numpy

from relayout.benchmarks import layouts_of
from relayout.ops import Block, Permute
from relayout.search.layouts import IDENTITY, Layout

N = dace.symbol("N")
W0, W1, W2 = 0.25, 0.5, 0.25


@dace.program
def stencil(x: dace.float64[N], y: dace.float64[N]):
    for i in dace.map[1 : N - 1] @ dace.ScheduleType.Sequential:
        y[i] = W0 * x[i - 1] + W1 * x[i] + W2 * x[i + 1]


def oracle(x):
    y = numpy.zeros_like(x)  # boundaries y[0], y[-1] are never written -> 0
    y[1:-1] = W0 * x[:-2] + W1 * x[1:-1] + W2 * x[2:]
    return {"y": y}


def make_inputs(n, seed=0):
    return {"x": numpy.random.default_rng(seed).random(n)}


def candidates():
    """``x`` as it is and dimension-lifted ([V, N/V]) for V in {4, 8}."""
    lifted = [Layout(f"dlt{v}", (Block("x", "i0", str(v), ("o", "f")), Permute("x", ("f", "o")))) for v in (4, 8)]
    return layouts_of("x", [Layout(IDENTITY), *lifted])


def run_closure(inputs, n):
    """A ``run(sdfg) -> outputs`` closure for the sweep: binds a fresh output each call."""

    def run(sdfg):
        y = numpy.zeros(n)
        sdfg(x=inputs["x"].copy(), y=y, N=n)
        return {"y": y}

    return run
