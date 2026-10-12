# Copyright 2026 ETH Zurich and the relayout authors.
"""Zip on SDFGs (ported from dace@extended's ZipArrays tests): arrays of one element type become a field
dimension of one array, which a layout then places (AoS as zipped, SoA permuted), and reductions into a
zipped field keep accumulating."""

import numpy as np
import pytest

import dace
from conftest import variant
from relayout.frontends.sdfg import LayoutError, apply_requests, relayout_array, unzip_array, zip_arrays
from relayout.ops import ArraySpec, Permute, run
from relayout.requests import parse_requests

N = dace.symbol("N")
KRED = 5


@dace.program
def madd(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):  # noqa: N803
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = 0.5 * (A[i, j] + B[i, j])


@dace.program
def dual_reduce(x: dace.float64[N], acc0: dace.float64[N], acc1: dace.float64[N]):
    for i, j in dace.map[0:N, 0:KRED]:
        acc0[i] += x[i]
        acc1[i] += x[i] * 2.0


def test_zipped_fields_are_one_array_in_either_order() -> None:
    a, b = np.random.default_rng(0).random((2, 6, 6))
    for order in ("aos", "soa"):
        sdfg = madd.to_sdfg(simplify=True)
        variant(sdfg, order)
        zip_arrays(sdfg, ["A", "B"], "Z")
        assert [str(s) for s in sdfg.arrays["Z"].shape] == ["N", "N", "2"]
        if order == "soa":
            soa = run([Permute("A", ("i2", "i0", "i1"))], [ArraySpec("A", 3)], (), "ops").layout("A", ["N", "N", "2"])
            relayout_array(sdfg, "Z", soa)
            assert [str(s) for s in sdfg.arrays["Z"].shape] == ["2", "N", "N"]
        sdfg.validate()
        c = np.zeros((6, 6))
        sdfg(A=a.copy(), B=b.copy(), C=c, N=6)
        np.testing.assert_allclose(c, 0.5 * (a + b))


def test_a_reduction_into_zipped_fields_keeps_accumulating() -> None:
    sdfg = dual_reduce.to_sdfg(simplify=True)
    variant(sdfg, "zipped")
    zip_arrays(sdfg, ["acc0", "acc1"], "acc")
    x = np.random.default_rng(1).random(7)
    acc0, acc1 = np.ones(7), np.ones(7)
    sdfg(x=x, acc0=acc0, acc1=acc1, N=7)
    np.testing.assert_allclose(acc0, 1 + KRED * x)
    np.testing.assert_allclose(acc1, 1 + 2 * KRED * x)


NCLV = 3


@dace.program
def species_sum(q: dace.float64[NCLV, N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = q[0, i] + 2.0 * q[1, i] + 3.0 * q[2, i]


@dace.program
def species_loop(q: dace.float64[NCLV, N], out: dace.float64[N]):
    for s in range(NCLV):
        for i in dace.map[0:N]:
            out[i] = out[i] + (s + 1.0) * q[s, i]


K = dace.symbol("K")


@dace.program
def species_pick(q: dace.float64[NCLV, N], out: dace.float64[N]):
    for i in dace.map[0:N]:
        out[i] = q[K, i]


def test_unzip_splits_an_array_along_a_small_dimension() -> None:
    """SplitArray: one array per species, also when a loop walks the species (it is unrolled)."""
    q = np.random.default_rng(2).random((NCLV, 5))
    for program in (species_sum, species_loop):
        sdfg = program.to_sdfg(simplify=True)
        variant(sdfg, "split")
        unzip_array(sdfg, "q", ["q_rain", "q_ice", "q_snow"], 0)
        assert {"q_rain", "q_ice", "q_snow"} <= set(sdfg.arrays)
        sdfg.validate()
        out = np.zeros(5)
        sdfg(q=q.copy(), out=out, N=5)
        np.testing.assert_allclose(out, q[0] + 2 * q[1] + 3 * q[2])


def test_a_request_zips_then_lays_out() -> None:
    """The JSON an agent sends: zip A and B (AoS), then permute the field outermost (SoA)."""
    request = parse_requests(
        {
            "operations": [
                {"action": "zip", "arrays": ["A", "B"], "into": "Z"},
                {"action": "permute", "array": "Z", "order": ["i2", "i0", "i1"]},
            ]
        }
    )
    sdfg = madd.to_sdfg(simplify=True)
    variant(sdfg, "request")
    assert apply_requests(sdfg, request) == {"Z": "Z"}
    a, b = np.random.default_rng(3).random((2, 4, 4))
    c = np.zeros((4, 4))
    sdfg(A=a.copy(), B=b.copy(), C=c, N=4)
    np.testing.assert_allclose(c, 0.5 * (a + b))


def test_a_data_dependent_split_index_is_refused() -> None:
    sdfg = species_pick.to_sdfg(simplify=True)
    with pytest.raises(LayoutError, match="not a constant"):
        unzip_array(sdfg, "q", ["a", "b", "c"], 0)


if __name__ == "__main__":
    test_zipped_fields_are_one_array_in_either_order()
    test_a_reduction_into_zipped_fields_keeps_accumulating()
    test_unzip_splits_an_array_along_a_small_dimension()
    test_a_request_zips_then_lays_out()
    test_a_data_dependent_split_index_is_refused()
