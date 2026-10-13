# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""k09 particle field affinity -- the Zip/Block AoSoA witness (SoA <-> AoS <-> AoSoA).

A per-particle drift+kick+decay update over F=8 fields (x,y,z, vx,vy,vz, q, m). The state starts
fully unzipped (SoA: one contiguous ``[N]`` array per field). The global layout decision is how the
fields are packed together:

    SoA      : F separate ``[N]`` arrays                         (identity -- keep unzipped)
    AoS      : one ``[N, F]`` array, fields are strided columns  (Zip all fields)
    AoSoA-V  : one ``[N/V, F, V]`` array                         (Block particle axis + Zip)

Every field is touched at the same particle index ``i`` per iteration, so the packing is transparent:
Zip rewrites each ``field[i]`` memlet to ``P[i, f]`` (AoS) or ``P[i//V, f, i%V]`` (AoSoA), the
tasklets are untouched, and every candidate must reproduce the SoA oracle. The sweep picks the layout;
the Zip/Block algebra guarantees correctness.

Source: Slattery et al., Cabana (JOSS'22, AoSoA vector length V); Zhong et al., array regrouping via
reference affinity (PLDI'04); Chilimbi et al., cache-conscious structure definition (PLDI'99);
Sung et al., in-place AoS->AoSoA (InPar'12).
"""

import dace
import numpy

from relayout.benchmarks import zip_fields
from relayout.frontends.sdfg import prepare

N = dace.symbol("N")

FIELDS = ["x", "y", "z", "vx", "vy", "vz", "q", "m"]  # F = 8
DT = 1e-3


@dace.program
def particle_step(
    x: dace.float64[N],
    y: dace.float64[N],
    z: dace.float64[N],
    vx: dace.float64[N],
    vy: dace.float64[N],
    vz: dace.float64[N],
    q: dace.float64[N],
    m: dace.float64[N],
):
    """One drift+kick+decay step per particle, in place on every field (all reads use old values)."""
    for i in dace.map[0:N] @ dace.ScheduleType.Sequential:
        x[i] = x[i] + DT * vx[i]  # drift: positions from (old) velocities
        y[i] = y[i] + DT * vy[i]
        z[i] = z[i] + DT * vz[i]
        vx[i] = vx[i] + DT * 0.1 * q[i]  # kick: velocities from (old) charge
        vy[i] = vy[i] + DT * 0.2 * q[i]
        vz[i] = vz[i] + DT * 0.3 * q[i]
        m[i] = m[i] + DT * q[i]  # mass accretion from (old) charge
        q[i] = q[i] * 0.999  # tag: charge decay (written last)


def oracle(x, y, z, vx, vy, vz, q, m):
    """Pure-numpy reference in particle order; every output is a function of the ORIGINAL inputs."""
    return {
        "x": x + DT * vx,
        "y": y + DT * vy,
        "z": z + DT * vz,
        "vx": vx + DT * 0.1 * q,
        "vy": vy + DT * 0.2 * q,
        "vz": vz + DT * 0.3 * q,
        "m": m + DT * q,
        "q": q * 0.999,
    }


def make_inputs(n, seed=0):
    rng = numpy.random.default_rng(seed)
    return {f: rng.random(n) for f in FIELDS}


def candidates():
    """The global layout candidates for the particle state:

    * ``soa``     -- identity: keep the F fields fully unzipped ([N] each).
    * ``aos``     -- Zip all fields into one ``[N, F]`` array ``P`` (field-minor AoS).
    * ``aosoa_V`` -- the Zip, then a Block of the particle axis by V -> ``P`` is ``[N/V, F, V]``.
    """
    return {
        "soa": lambda sdfg: None,
        "aos": lambda sdfg: zip_fields(sdfg, FIELDS, "P"),
        "aosoa_4": lambda sdfg: zip_fields(prepare(sdfg), FIELDS, "P", 4),
        "aosoa_8": lambda sdfg: zip_fields(prepare(sdfg), FIELDS, "P", 8),
    }


def run_closure(inputs, n):
    """A ``run(sdfg) -> outputs`` closure for the sweep: the fields are updated in place, so each call
    binds fresh copies."""

    def run(sdfg):
        fields = {f: inputs[f].copy() for f in FIELDS}
        sdfg(N=n, **fields)
        return fields

    return run
