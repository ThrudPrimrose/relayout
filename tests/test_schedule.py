# Copyright 2026 ETH Zurich and the relayout authors.
"""The schedule after a layout: a map over a blocked array is tiled so its accesses lose their ``//`` and
``%``, a tiled map over a flat array is merged back, and a map nest is ordered by stride; every result
computes what the program computed."""

import numpy as np

import dace
from tests.conftest import variant
from dace import nodes
from relayout.frontends.schedule import canonicalize_schedule
from relayout.frontends.sdfg import prepare, relayout_array
from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Block, Permute, run

N, M, K = (dace.symbol(s) for s in "NMK")


def layout(ops: list, shape: list[str]) -> IndexMap:
    return run(ops, [ArraySpec("A", len(shape))], (), "operations").layout("A", shape)


@dace.program
def scale(A: dace.float64[N, M], B: dace.float64[N, M]):  # noqa: N803
    for i, j in dace.map[0:N, 0:M]:
        B[i, j] = A[i, j] * 2.0


@dace.program
def tiled(A: dace.float64[4 * K], B: dace.float64[4 * K]):  # noqa: N803
    for jo in dace.map[0:K]:
        for ji in dace.map[0:4]:
            with dace.tasklet:
                a << A[4 * jo + ji]
                b >> B[4 * jo + ji]
                b = 2.0 * a


@dace.program
def interior(A: dace.float64[N], B: dace.float64[N]):  # noqa: N803
    for i in dace.map[1 : N - 1]:
        B[i] = A[i] * 2.0


def point_accesses(sdfg: dace.SDFG, array: str) -> list[str]:
    return [
        str(e.data.subset)
        for state in sdfg.states()
        for e in state.edges()
        if e.data.data == array and (isinstance(e.dst, nodes.Tasklet) or isinstance(e.src, nodes.Tasklet))
    ]


def nesting(sdfg: dace.SDFG) -> list[list[str]]:
    """The parameters of every map, outermost first, in the state holding the program's maps."""
    out = []
    for state in sdfg.states():
        scopes = state.scope_dict()
        for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scopes[n] is None]:
            chain, inner = [list(entry.map.params)], entry
            while True:
                nested = [n for n in state.nodes() if isinstance(n, nodes.MapEntry) and scopes[n] is inner]
                if len(nested) != 1:
                    break
                inner = nested[0]
                chain.append(list(inner.map.params))
            out.append([p for level in chain for p in level])
    return out


def test_a_blocked_array_tiles_its_map() -> None:
    sdfg = prepare(scale.to_sdfg(simplify=True))
    variant(sdfg, "blocked")
    for array in ("A", "B"):
        relayout_array(sdfg, array, layout([Block("A", "i1", "4", ("o", "f"))], ["N", "M"]))
    assert canonicalize_schedule(sdfg) == 3  # the program's map and the two conversions
    sdfg.validate()
    accesses = point_accesses(sdfg, "laid_A")
    assert "i, j_o, j" in accesses
    assert not any("Mod" in a or "floor" in a for a in accesses)
    a = np.random.default_rng(0).random((5, 10))
    b = np.zeros((5, 10))
    sdfg(A=a, B=b, N=5, M=10)
    np.testing.assert_allclose(b, 2 * a)


def test_a_permuted_array_puts_its_unit_stride_parameter_innermost() -> None:
    sdfg = prepare(scale.to_sdfg(simplify=True))
    variant(sdfg, "permuted")
    for array in ("A", "B"):
        relayout_array(sdfg, array, layout([Permute("A", ("i1", "i0"))], ["N", "M"]))
    canonicalize_schedule(sdfg)
    assert ["j", "i"] in nesting(sdfg)
    a = np.random.default_rng(1).random((5, 10))
    b = np.zeros((5, 10))
    sdfg(A=a, B=b, N=5, M=10)
    np.testing.assert_allclose(b, 2 * a)


def test_a_tiled_map_merges_over_flat_arrays_and_keeps_its_tiles_over_blocked_ones() -> None:
    a = np.random.default_rng(2).random(12)
    flat = tiled.to_sdfg(simplify=True)
    variant(flat, "flat")
    canonicalize_schedule(flat)
    assert point_accesses(flat, "A") == ["ji"]
    blocked = tiled.to_sdfg(simplify=True)
    variant(blocked, "blocked")
    relayout_array(blocked, "A", layout([Block("A", "i0", "4", ("o", "f"))], ["4*K"]))
    canonicalize_schedule(blocked)
    assert "jo, ji" in point_accesses(blocked, "laid_A")
    for sdfg in (flat, blocked):
        b = np.zeros(12)
        sdfg(A=a, B=b, K=3)
        np.testing.assert_allclose(b, 2 * a)


def test_a_map_not_starting_at_zero_keeps_its_blocked_accesses() -> None:
    """Only a map from 0 by 1 tiles into whole blocks; one from 1 is left as it is (and does not stall the pass)."""
    sdfg = prepare(interior.to_sdfg(simplify=True))
    variant(sdfg, "interior")
    relayout_array(sdfg, "A", layout([Block("A", "i0", "4", ("o", "f"))], ["N"]))
    canonicalize_schedule(sdfg)
    a = np.random.default_rng(3).random(12)
    b = np.zeros(12)
    sdfg(A=a, B=b, N=12)
    np.testing.assert_allclose(b[1:-1], 2 * a[1:-1])


if __name__ == "__main__":
    test_a_blocked_array_tiles_its_map()
    test_a_permuted_array_puts_its_unit_stride_parameter_innermost()
    test_a_tiled_map_merges_over_flat_arrays_and_keeps_its_tiles_over_blocked_ones()
    test_a_map_not_starting_at_zero_keeps_its_blocked_accesses()
