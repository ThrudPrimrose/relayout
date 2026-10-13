# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k03 complex conjugate -- the Zip/Block layout witness (AoS vs SoA vs AoSoA).

Conjugate-and-scale a complex array of N numbers modeled as two real float64 fields (re, im):

    re[i] =  SCALE * re[i]
    im[i] = -SCALE * im[i]      (negate imag = conjugate; SCALE = uniform scale on both fields)

The two fields can be laid out three ways, and that layout is the decision the sweep explores:

  * SoA (unzipped): re and im are two separate [N] arrays -- each field its own contiguous stream.
  * AoS (zipped)   : one [N, 2] array (re, im interleaved) -- a Zip fuses the two fields.
  * AoSoA (V)      : one [N/V, 2, V] array -- the Zip, then a Block(V) of the element axis;
                     V-contiguous runs per field per tile.

Conjugation touches the imaginary field, so AoS pulls the untouched real value into every cache line
while SoA multiplies prefetcher streams -- the classic AoS/SoA/AoSoA layout tradeoff. Every candidate
is transparent (the fields carry the same values; only the addressing changes), so all reproduce the
oracle -- the sweep picks the layout, the Zip/Block algebra guarantees correctness.

Source: SC26 layout paper SS IV-B1 (conjugation microbenchmark); R. Strzodka, "Abstraction for AoS
and SoA Layout in C++," GPU Computing Gems Jade, 2012; Sung, Liu & Hwu, "DL: a data layout
transformation system for heterogeneous computing," InPar 2012 (ASTA = AoSoA). Primitives: Zip, Block.
"""

import dace
import numpy

from relayout.benchmarks import zip_fields

N = dace.symbol("N")
SCALE = 2.0


@dace.program
def conjugate(re: dace.float64[N], im: dace.float64[N]):
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        re[i] = SCALE * re[i]
        im[i] = -SCALE * im[i]


def oracle(re, im):
    return {"re": SCALE * re, "im": -SCALE * im}


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {"re": rng.random(n), "im": rng.random(n)}


def candidates():
    """AoS / SoA / AoSoA layouts of the (re, im) complex array: Zip, and Zip then Block (AoSoA)."""
    return {
        "unzipped_soa": lambda sdfg: None,
        "zipped_aos": lambda sdfg: zip_fields(sdfg, ["re", "im"], "cplx"),
        "aosoa_4": lambda sdfg: zip_fields(sdfg, ["re", "im"], "cplx", 4),
        "aosoa_8": lambda sdfg: zip_fields(sdfg, ["re", "im"], "cplx", 8),
    }


def run_closure(inputs, n):
    """A ``run(sdfg) -> outputs`` closure for the sweep: the fields are updated in place, so each call
    binds fresh copies."""

    def run(sdfg):
        re, im = inputs["re"].copy(), inputs["im"].copy()
        sdfg(re=re, im=im, N=n)
        return {"re": re, "im": im}

    return run
