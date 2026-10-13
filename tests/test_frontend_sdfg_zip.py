# Copyright 2026 ETH Zurich and the relayout authors.
"""Zip and unzip on SDFGs, each the other's inverse: arrays of one element type are stacked along a new
dimension of one array (last: AoS, first: SoA, or a layout places it), reductions into a zipped field keep
accumulating, and an array of a small constant extent unzips into one array per index."""

import numpy as np
import pytest

import dace
from conftest import variant
from relayout.frontends.sdfg import apply_requests, relayout_array, unzip_array, zip_arrays
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
    for order in ("aos", "soa", "stacked_first"):
        sdfg = madd.to_sdfg(simplify=True)
        variant(sdfg, order)
        zip_arrays(sdfg, ["A", "B"], "Z", 0 if order == "stacked_first" else None)
        expected = ["2", "N", "N"] if order == "stacked_first" else ["N", "N", "2"]
        assert [str(s) for s in sdfg.arrays["Z"].shape] == expected
        if order == "soa":
            soa = run([Permute("A", ("i2", "i0", "i1"))], [ArraySpec("A", 3)], (), "ops").layout("A", ["N", "N", "2"])
            relayout_array(sdfg, "Z", soa)
            assert [str(s) for s in sdfg.arrays["Z"].shape] == ["2", "N", "N"]
        sdfg.validate()
        c = np.zeros((6, 6))
        sdfg(A=a.copy(), B=b.copy(), C=c, N=6)
        np.testing.assert_allclose(c, 0.5 * (a + b))


@dace.program
def mixed(A: dace.float32[N, N], B: dace.float64[N, N], C: dace.float64[N, N]):  # noqa: N803
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = A[i, j] + B[i, j]
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = 2.0 * C[i, j]


def test_arrays_of_different_element_types_zip_into_a_struct() -> None:
    """A (fp32) and B (fp64) become one array of structs; B is read in one state and written in the next."""
    sdfg = mixed.to_sdfg(simplify=True)
    variant(sdfg, "struct")
    zip_arrays(sdfg, ["A", "B"], "Z")
    sdfg.validate()
    assert isinstance(sdfg.arrays["Z"].dtype, dace.struct) and [str(s) for s in sdfg.arrays["Z"].shape] == ["N", "N"]
    rng = np.random.default_rng(9)
    a, b = rng.random((4, 4)).astype(np.float32), rng.random((4, 4))
    want_c = a.astype(np.float64) + b
    c = np.zeros((4, 4))
    got_b = b.copy()
    sdfg(A=a.copy(), B=got_b, C=c, N=4)
    np.testing.assert_allclose(c, want_c)
    np.testing.assert_allclose(got_b, 2.0 * want_c)


def test_a_cast_before_a_zip_makes_a_struct_of_the_cast_types() -> None:
    pytest.importorskip("fp_arena")
    request = parse_requests(
        {
            "operations": [
                {"action": "cast", "array": "A", "to": "fp32"},
                {"action": "zip", "arrays": ["A", "B"], "into": "Z"},
            ]
        }
    )
    sdfg = madd.to_sdfg(simplify=True)
    variant(sdfg, "cast_zip")
    apply_requests(sdfg, request)
    members = dict(sdfg.arrays["Z"].dtype.fields)
    assert list(members.values()) == [dace.float32, dace.float64]
    a, b = np.random.default_rng(10).random((2, 4, 4))
    c = np.zeros((4, 4))
    sdfg(A=a.copy(), B=b.copy(), C=c, N=4)
    np.testing.assert_allclose(c, 0.5 * (a.astype(np.float32) + b), rtol=1e-6)


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
    """One array per species, also when a loop walks the species (it is unrolled)."""
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


def test_a_request_zips_along_a_dimension_and_unzips_back() -> None:
    """zip with dim stacks the fields first; the unzip that names no dim splits along that same dimension."""
    request = parse_requests(
        {
            "operations": [
                {"action": "zip", "arrays": ["A", "B"], "into": "Z", "dim": 0},
                {"action": "unzip", "array": "Z", "into": ["A2", "B2"]},
            ]
        }
    )
    sdfg = madd.to_sdfg(simplify=True)
    variant(sdfg, "zip_unzip")
    apply_requests(sdfg, request)
    assert {"A2", "B2"} <= set(sdfg.arrays) and "Z" not in sdfg.arrays
    a, b = np.random.default_rng(5).random((2, 4, 4))
    c = np.zeros((4, 4))
    sdfg(A=a.copy(), B=b.copy(), C=c, N=4)
    np.testing.assert_allclose(c, 0.5 * (a + b))


def test_a_data_dependent_index_selects_among_the_parts() -> None:
    """``q[K, i]`` with K known only at run time: the tasklet reads every part and picks the one K names."""
    sdfg = species_pick.to_sdfg(simplify=True)
    variant(sdfg, "pick")
    unzip_array(sdfg, "q", ["a", "b", "c"], 0)
    sdfg.validate()
    q = np.random.default_rng(6).random((NCLV, 5))
    for k in range(NCLV):
        out = np.zeros(5)
        sdfg(q=q.copy(), out=out, N=5, K=k)
        np.testing.assert_allclose(out, q[k])


@dace.program
def species_put(q: dace.float64[NCLV, N], x: dace.float64[N]):
    for i in dace.map[0:N]:
        q[K, i] = x[i] * 2.0


def test_a_data_dependent_write_stores_into_the_part_it_names() -> None:
    sdfg = species_put.to_sdfg(simplify=True)
    variant(sdfg, "put")
    unzip_array(sdfg, "q", ["a", "b", "c"], 0)
    sdfg.validate()
    x = np.random.default_rng(7).random(5)
    q = np.zeros((NCLV, 5))
    sdfg(q=q, x=x, N=5, K=1)
    np.testing.assert_allclose(q, np.stack([np.zeros(5), 2 * x, np.zeros(5)]))


if __name__ == "__main__":
    test_zipped_fields_are_one_array_in_either_order()
    test_arrays_of_different_element_types_zip_into_a_struct()
    test_a_cast_before_a_zip_makes_a_struct_of_the_cast_types()
    test_a_reduction_into_zipped_fields_keeps_accumulating()
    test_unzip_splits_an_array_along_a_small_dimension()
    test_a_request_zips_then_lays_out()
    test_a_request_zips_along_a_dimension_and_unzips_back()
    test_a_data_dependent_index_selects_among_the_parts()
    test_a_data_dependent_write_stores_into_the_part_it_names()
