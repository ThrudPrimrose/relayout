# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""UntileBlockedLoops: a tile nest reading an array blocked to the tile collapses, the array unblocked."""

import dace
import numpy as np

from relayout.passes.sdfg.untile_blocked_loops import UntileBlockedLoops

N = dace.symbol("N")


def loops(sdfg: dace.SDFG) -> list[dace.sdfg.state.LoopRegion]:
    return [r for r in sdfg.all_control_flow_regions() if isinstance(r, dace.sdfg.state.LoopRegion) and r.loop_variable]


@dace.program
def blocked_tiles(a: dace.float64[N], b: dace.float64[N]):
    blk = np.empty([N // 4, 4], dtype=np.float64)
    for i in range(0, N, 4):
        for ii in range(4):
            blk[i // 4, ii] = b[i + ii] + 1.0
    for i in range(0, N, 4):
        for ii in range(4):
            a[i + ii] = blk[i // 4, ii] * 2.0


def test_an_array_blocked_to_the_tile_is_unblocked_and_both_nests_untile():
    """``blk[i // 4, ii]`` is written by one tile nest and read (through the scalar the frontend copies it into)
    by another: ``UntileBlockedLoops`` flattens ``blk`` to ``blk[i + ii]`` and collapses both nests, the second
    reading ``4*int_floor(i, 4) + ii`` once ``blk`` is flat, which is ``i + ii`` for tiles starting at 0."""
    sdfg = blocked_tiles.to_sdfg(simplify=True)
    assert len(loops(sdfg)) == 4
    assert UntileBlockedLoops().apply_pass(sdfg, {}) == 2
    sdfg.validate()
    assert len(sdfg.arrays["blk"].shape) == 1 and len(loops(sdfg)) == 2
    a, b = np.zeros(12), np.random.default_rng(0).random(12)
    sdfg(a=a, b=b, N=12)
    assert np.allclose(a, 2.0 * (b + 1.0))


if __name__ == "__main__":
    test_an_array_blocked_to_the_tile_is_unblocked_and_both_nests_untile()
