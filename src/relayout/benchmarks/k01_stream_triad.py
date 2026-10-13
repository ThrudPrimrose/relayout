# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k01 STREAM triad -- the bandwidth / fragmentation instrument (McCalpin STREAM).

    a[i] = b[i] + Q * c[i]

The reference kernel is a memory-bandwidth calibration instrument -- it carries no layout primitive
of its own; it measures the fraction of a cache line that is useful under a strided read. Ported here
as a 1D elementwise triad whose read operands ``b``, ``c`` can be BLOCKED: a Block cuts
the contiguous axis into ``[N/T, T]`` tiles, the tile width ``T`` being exactly the cache-line
fraction the STREAM strided variant sweeps. Every block factor is transparent (the tiled read
reproduces the contiguous one), so all candidates match the oracle -- the sweep only picks the
physical tiling.

Source: J. D. McCalpin, "Memory Bandwidth and Machine Balance in Current High Performance
Computers," IEEE TCCA Newsletter, 1995 (STREAM Triad); SC26 layout paper (fragmentation parameter).
"""

import dace
import numpy

from relayout.benchmarks import layouts_of
from relayout.ops import Block
from relayout.search.layouts import IDENTITY, Layout

N = dace.symbol("N")
Q = 1.5


@dace.program
def triad(a: dace.float64[N], b: dace.float64[N], c: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        a[i] = b[i] + Q * c[i]


def oracle(b, c):
    return {"a": b + Q * c}


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"b": rng.random(n), "c": rng.random(n)}


def candidates():
    """Block family over each read operand (unblocked + ``[N/T, T]`` tiles for T in {8, 16})."""
    cands = {}
    for arr in ("b", "c"):
        blocks = [Layout(f"block{t}", (Block(arr, "i0", str(t), ("o", "f")),)) for t in (8, 16)]
        cands.update(layouts_of(arr, [Layout(IDENTITY), *blocks]))
    return cands


def run_closure(inputs, n):
    """A ``run(sdfg) -> outputs`` closure for the sweep: binds a fresh output each call."""

    def run(sdfg):
        a = numpy.zeros(n)
        sdfg(a=a, b=inputs["b"].copy(), c=inputs["c"].copy(), N=n)
        return {"a": a}

    return run
