# Copyright 2026 ETH Zurich and the relayout authors.
"""Zip and unzip on Python functions, each the other's inverse; the counterpart of tests/test_sdfg_zip_unzip.py:
arrays of one element type are stacked along a new dimension of one array (last: AoS, first: SoA, or a layout
places it), arrays of different ones zip into a structured array, reductions into a zipped field keep
accumulating, and an array of a small constant extent unzips into one array per index."""

import ast

import numpy as np

from relayout.ops import ArraySpec, Permute, run
from relayout.passes.py.apply_requests import apply_requests
from relayout.passes.py.common import Array
from relayout.passes.py.relayout_array import relayout_array
from relayout.passes.py.unzip_array import unzip_array
from relayout.passes.py.zip_arrays import zip_arrays
from relayout.requests import parse_requests
from tests.conftest import rewritten

SQUARE = ("N", "N")
KRED = 5
NCLV = 3


def madd(A: np.ndarray, B: np.ndarray, C: np.ndarray, N: int) -> None:  # noqa: N803
    for i in range(N):
        for j in range(N):
            C[i, j] = 0.5 * (A[i, j] + B[i, j])


def float64s(*names: str, shape: tuple[str, ...] = SQUARE) -> dict[str, Array]:
    return {name: Array(shape, "float64") for name in names}


def test_zipped_fields_are_one_array_in_either_order() -> None:
    a, b = np.random.default_rng(0).random((2, 6, 6))
    for order in ("aos", "soa", "stacked_first"):

        def transform(tree: ast.Module, order: str = order) -> None:
            zip_arrays(tree, ["A", "B"], "Z", float64s("A", "B"), 0 if order == "stacked_first" else None)
            if order == "soa":
                soa = run([Permute("A", ("i2", "i0", "i1"))], [ArraySpec("A", 3)], (), "ops")
                relayout_array(tree, "Z", soa.layout("A", ["N", "N", "2"]))

        function = rewritten(madd, transform)
        expected = {"aos": "(N, N, 2)", "soa": "(2, N, N)", "stacked_first": "(2, N, N)"}[order]
        assert f"Z = np.zeros({expected}" in function.source
        c = np.zeros((6, 6))
        function.function(a.copy(), b.copy(), c, N=6)
        np.testing.assert_allclose(c, 0.5 * (a + b))


def mixed(A: np.ndarray, B: np.ndarray, C: np.ndarray, N: int) -> None:  # noqa: N803
    for i in range(N):
        for j in range(N):
            C[i, j] = A[i, j] + B[i, j]
    for i in range(N):
        for j in range(N):
            B[i, j] = 2.0 * C[i, j]


def test_arrays_of_different_element_types_zip_into_a_struct() -> None:
    """A (fp32) and B (fp64) become one structured array; B is read in one loop and written in the next."""
    arrays = {"A": Array(SQUARE, "float32"), "B": Array(SQUARE, "float64")}
    function = rewritten(mixed, lambda tree: zip_arrays(tree, ["A", "B"], "Z", arrays))
    assert "dtype=[('A', 'float32'), ('B', 'float64')]" in function.source
    rng = np.random.default_rng(9)
    a, b = rng.random((4, 4)).astype(np.float32), rng.random((4, 4))
    want_c = a.astype(np.float64) + b
    c = np.zeros((4, 4))
    got_b = b.copy()
    function.function(a.copy(), got_b, c, N=4)
    np.testing.assert_allclose(c, want_c)
    np.testing.assert_allclose(got_b, 2.0 * want_c)


def test_a_cast_before_a_zip_makes_a_struct_of_the_cast_types() -> None:
    request = parse_requests(
        {
            "operations": [
                {"action": "cast", "array": "A", "to": "fp32"},
                {"action": "zip", "arrays": ["A", "B"], "into": "Z"},
            ]
        }
    )
    function = rewritten(madd, lambda tree: apply_requests(tree, request, float64s("A", "B", "C")))
    assert "dtype=[('A_float32', 'float32'), ('B', 'float64')]" in function.source
    a, b = np.random.default_rng(10).random((2, 4, 4))
    c = np.zeros((4, 4))
    function.function(a.copy(), b.copy(), c, N=4)
    np.testing.assert_allclose(c, 0.5 * (a.astype(np.float32) + b), rtol=1e-6)


def dual_reduce(x: np.ndarray, acc0: np.ndarray, acc1: np.ndarray, N: int) -> None:  # noqa: N803
    for i in range(N):
        for _ in range(KRED):
            acc0[i] += x[i]
            acc1[i] += x[i] * 2.0


def test_a_reduction_into_zipped_fields_keeps_accumulating() -> None:
    function = rewritten(
        dual_reduce, lambda tree: zip_arrays(tree, ["acc0", "acc1"], "acc", float64s("acc0", "acc1", shape=("N",)))
    )
    x = np.random.default_rng(1).random(7)
    acc0, acc1 = np.ones(7), np.ones(7)
    function.function(x, acc0, acc1, N=7)
    np.testing.assert_allclose(acc0, 1 + KRED * x)
    np.testing.assert_allclose(acc1, 1 + 2 * KRED * x)


def species_sum(q: np.ndarray, out: np.ndarray, N: int) -> None:  # noqa: N803
    for i in range(N):
        out[i] = q[0, i] + 2.0 * q[1, i] + 3.0 * q[2, i]


def species_loop(q: np.ndarray, out: np.ndarray, N: int) -> None:  # noqa: N803
    for s in range(NCLV):
        for i in range(N):
            out[i] = out[i] + (s + 1.0) * q[s, i]


SPECIES = {"q": Array((str(NCLV), "N"), "float64")}


def test_unzip_splits_an_array_along_a_small_dimension() -> None:
    """One array per species, also when a loop walks the species (its index picks the part at run time)."""
    q = np.random.default_rng(2).random((NCLV, 5))
    for program in (species_sum, species_loop):
        function = rewritten(program, lambda tree: unzip_array(tree, "q", ["q_rain", "q_ice", "q_snow"], 0, SPECIES))
        assert "q_rain = np.zeros((N,)" in function.source
        out = np.zeros(5)
        function.function(q.copy(), out, N=5)
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
    working: dict[str, str] = {}
    function = rewritten(madd, lambda tree: working.update(apply_requests(tree, request, float64s("A", "B", "C"))))
    assert working == {"Z": "Z"}
    a, b = np.random.default_rng(3).random((2, 4, 4))
    c = np.zeros((4, 4))
    function.function(a.copy(), b.copy(), c, N=4)
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
    function = rewritten(madd, lambda tree: apply_requests(tree, request, float64s("A", "B", "C")))
    assert "A2 = np.zeros" in function.source and "Z" not in function.source
    a, b = np.random.default_rng(5).random((2, 4, 4))
    c = np.zeros((4, 4))
    function.function(a.copy(), b.copy(), c, N=4)
    np.testing.assert_allclose(c, 0.5 * (a + b))


def species_pick(q: np.ndarray, out: np.ndarray, N: int, K: int) -> None:  # noqa: N803
    for i in range(N):
        out[i] = q[K, i]


def test_a_data_dependent_index_selects_among_the_parts() -> None:
    """``q[K, i]`` with K known only at run time reads the part K names."""
    function = rewritten(species_pick, lambda tree: unzip_array(tree, "q", ["a", "b", "c"], 0, SPECIES))
    q = np.random.default_rng(6).random((NCLV, 5))
    for k in range(NCLV):
        out = np.zeros(5)
        function.function(q.copy(), out, N=5, K=k)
        np.testing.assert_allclose(out, q[k])


def species_put(q: np.ndarray, x: np.ndarray, N: int, K: int) -> None:  # noqa: N803
    for i in range(N):
        q[K, i] = x[i] * 2.0


def test_a_data_dependent_write_stores_into_the_part_it_names() -> None:
    function = rewritten(species_put, lambda tree: unzip_array(tree, "q", ["a", "b", "c"], 0, SPECIES))
    x = np.random.default_rng(7).random(5)
    q = np.zeros((NCLV, 5))
    function.function(q, x, N=5, K=1)
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
