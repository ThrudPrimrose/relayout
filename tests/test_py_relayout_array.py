# Copyright 2026 ETH Zurich and the relayout authors.
"""The Python passes: a function whose arrays are laid out (arguments converted in and out, locals in place)
computes what it computed before; the counterpart of tests/test_sdfg_relayout_array.py."""

import ast

import numpy as np
import pytest

from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Block, Pad, Permute, Precision, Shuffle, run
from relayout.passes.py.apply_requests import apply_requests
from relayout.passes.py.cast_array import cast_array
from relayout.passes.py.common import Array, LayoutError
from relayout.passes.py.relayout_array import relayout_array
from relayout.requests import parse_requests
from tests.conftest import LAYOUT_CASES, rewritten

SHAPE = ("N", "M")
ARRAYS = {"A": Array(SHAPE, "float64"), "B": Array(SHAPE, "float64")}


def layout(ops: list, shape: tuple[str, ...] = SHAPE) -> IndexMap:
    return run(ops, [ArraySpec("A", len(shape))], (), "operations").layout("A", list(shape))


def stencil(A: np.ndarray, B: np.ndarray, N: int, M: int) -> None:  # noqa: N803
    tmp = np.zeros((N, M))
    for i in range(N):
        for j in range(M):
            tmp[i, j] = A[i, j] * 2.0
    for i in range(N):
        for j in range(M):
            B[i, j] = tmp[i, j] + A[i, (j + 1) % M]


def reference(a: np.ndarray) -> np.ndarray:
    return a * 2.0 + np.roll(a, -1, axis=1)


def run_stencil(function: object, seed: int, n: int = 6, m: int = 12) -> None:
    a = np.random.default_rng(seed).standard_normal((n, m))
    b = np.zeros((n, m))
    function(a, b, N=n, M=m)  # type: ignore[operator]
    np.testing.assert_allclose(b, reference(a))


@pytest.mark.parametrize(
    ("a_ops", "b_ops", "tmp_ops"),
    [
        ([Permute("A", ("i1", "i0"))], [Pad("A", "i0", "2")], [Permute("A", ("i1", "i0"))]),
        (
            [Block("A", "i1", "4", ("o", "f")), Shuffle("A", ("f",), ("3 - f",), ("3 - f",))],
            [Pad("A", "i1", "1"), Permute("A", ("i1", "i0"))],
            [Block("A", "i0", "2", ("o", "f")), Permute("A", ("f", "i1", "o"))],
        ),
    ],
)
def test_laid_out_arguments_and_locals_compute_the_same(a_ops: list, b_ops: list, tmp_ops: list) -> None:
    def transform(tree: ast.Module) -> None:
        assert relayout_array(tree, "A", layout(a_ops)) == "laid_A"
        assert relayout_array(tree, "B", layout(b_ops)) == "laid_B"
        assert relayout_array(tree, "tmp", layout(tmp_ops)) == "tmp"

    run_stencil(rewritten(stencil, transform).function, 0)


@pytest.mark.parametrize("case", LAYOUT_CASES)
def test_every_layout_case_computes_the_same_on_arguments_and_locals(case: str) -> None:
    def transform(tree: ast.Module) -> None:
        for name in ("A", "B", "tmp"):
            relayout_array(tree, name, layout(LAYOUT_CASES[case]))

    run_stencil(rewritten(stencil, transform).function, 1)


def whole(A: np.ndarray, B: np.ndarray, N: int, M: int) -> None:  # noqa: N803
    B[:] = A * 2.0 + np.roll(A, -1, axis=1)


def test_whole_array_uses_go_through_the_logical_view() -> None:
    """A permuted or padded array is used whole through numpy's view of its buffer."""
    for ops in ([Permute("A", ("i1", "i0"))], [Pad("A", "i1", "3")]):
        function = rewritten(whole, lambda tree, ops=ops: [relayout_array(tree, n, layout(ops)) for n in "AB"])
        run_stencil(function.function, 2)


def test_a_cut_array_used_whole_is_refused_by_its_line() -> None:
    with pytest.raises(LayoutError, match="line 2: 'A' is laid out by cuts"):
        rewritten(whole, lambda tree: relayout_array(tree, "A", layout([Block("A", "i1", "2", ("o", "f"))])))


def products(A: np.ndarray, B: np.ndarray, C: np.ndarray, D: np.ndarray, N: int, K: int, M: int) -> None:  # noqa: N803
    C[:] = A @ B
    D[:] = np.einsum("ij,jk->ik", A, B)


def test_library_calls_take_a_permuted_operand_through_its_view() -> None:
    """``@`` and ``np.einsum`` (a library node on an SDFG) read a transposed operand through its view."""
    transposed = [Permute("A", ("i1", "i0"))]

    def transform(tree: ast.Module) -> None:
        relayout_array(tree, "A", layout(transposed, ("N", "K")))
        relayout_array(tree, "B", layout(transposed, ("K", "M")))
        relayout_array(tree, "D", layout(transposed, ("N", "M")))

    rng = np.random.default_rng(3)
    a, b = rng.standard_normal((4, 5)), rng.standard_normal((5, 3))
    c, d = np.zeros((4, 3)), np.zeros((4, 3))
    rewritten(products, transform).function(a, b, c, d, N=4, K=5, M=3)
    np.testing.assert_allclose(c, a @ b)
    np.testing.assert_allclose(d, a @ b)


def test_a_json_request_lays_out_the_function() -> None:
    """The request an agent sends is the function's layout: one JSON, the same operations as on an SDFG."""
    request = parse_requests(
        {
            "operations": [
                {"action": "block", "array": "A", "digit": "i1", "factor": 4, "into": ["o", "f"]},
                {"action": "permute", "array": "A", "order": ["o", "i0", "f"]},
                {"action": "pad", "array": "B", "digit": "i1", "amount": 3},
            ]
        }
    )
    working: dict[str, str] = {}
    function = rewritten(stencil, lambda tree: working.update(apply_requests(tree, request, ARRAYS)))
    assert working == {"A": "laid_A", "B": "laid_B"}
    run_stencil(function.function, 5, 3, 8)


@pytest.mark.parametrize(
    "request_json",
    [
        {"operations": [{"action": "replicate", "array": "A"}]},
        {"operations": [{"action": "place", "array": "A", "on": "device"}]},
        {
            "startup": {"grid": ["P"]},
            "operations": [{"action": "distribute", "array": "A", "grid_dim": 0, "dim": 0, "scheme": "block"}],
        },
        {"startup": {"load": ["blas"]}, "operations": []},
        {"operations": [{"action": "format", "array": "A", "to": "csr"}]},
    ],
    ids=["replicate", "place", "distribute", "load", "format"],
)
def test_what_the_caller_carries_out_is_refused_by_name(request_json: dict) -> None:
    """A rewrite of one node's function refuses what only the caller of the function can carry out."""
    with pytest.raises(LayoutError, match="belong to the caller"):
        rewritten(stencil, lambda tree: apply_requests(tree, parse_requests(request_json), ARRAYS))


def test_a_cast_changes_the_element_type_before_the_layout() -> None:
    """``cast`` A to fp32 (the interface stays fp64) and permute the fp32 copy."""
    request = parse_requests(
        {
            "operations": [
                {"action": "cast", "array": "A", "to": "fp32"},
                {"action": "permute", "array": "A", "order": ["i1", "i0"]},
            ]
        }
    )
    working: dict[str, str] = {}
    function = rewritten(stencil, lambda tree: working.update(apply_requests(tree, request, ARRAYS)))
    assert working == {"A": "A_float32"}
    assert "A_float32 = np.zeros((M, N), dtype=np.float32)" in function.source
    a = np.random.default_rng(6).standard_normal((3, 8))
    b = np.zeros((3, 8))
    function.function(a, b, N=3, M=8)
    np.testing.assert_allclose(b, reference(a.astype(np.float32).astype(np.float64)), rtol=1e-6)


def test_a_local_is_cast_in_place_and_a_precision_numpy_lacks_is_refused() -> None:
    function = rewritten(
        stencil, lambda tree: cast_array(tree, "tmp", Precision.FP32, {"tmp": Array(SHAPE, "float64")})
    )
    assert "tmp = np.zeros((N, M), dtype=np.float32)" in function.source
    a = np.random.default_rng(7).standard_normal((3, 8))
    b = np.zeros((3, 8))
    function.function(a, b, N=3, M=8)
    np.testing.assert_allclose(b, reference(a), rtol=1e-6)
    with pytest.raises(LayoutError, match="no such element type"):
        rewritten(stencil, lambda tree: cast_array(tree, "A", Precision.BF16, ARRAYS))


def half(A: np.ndarray, B: np.ndarray, N: int, M: int) -> None:  # noqa: N803
    for i in range(N // 2):
        for j in range(M):
            B[i, j] = A[i, j] + 1.0


def test_an_argument_written_in_part_keeps_the_cells_the_function_leaves() -> None:
    function = rewritten(half, lambda tree: relayout_array(tree, "B", layout([Block("A", "i1", "4", ("o", "f"))])))
    a = np.random.default_rng(8).standard_normal((4, 6))
    b = np.full((4, 6), 7.0)
    function.function(a, b, N=4, M=6)
    np.testing.assert_allclose(b[:2], a[:2] + 1.0)
    np.testing.assert_allclose(b[2:], 7.0)


def defined(A: np.ndarray, B: np.ndarray, N: int, M: int) -> None:  # noqa: N803
    tmp = A * 2.0
    for i in range(N):
        for j in range(M):
            B[i, j] = tmp[i, j] + A[i, (j + 1) % M]
    B[-1, -1] = tmp[-1, -1] + A[-1, 0]


def test_a_local_defined_by_an_expression_is_converted_in_and_negative_indices_count_from_the_end() -> None:
    def transform(tree: ast.Module) -> None:
        relayout_array(tree, "tmp", layout(LAYOUT_CASES["linear_shuffle"]))
        relayout_array(tree, "A", layout(LAYOUT_CASES["block_ragged"]))

    with pytest.raises(LayoutError, match="cuts or a shuffle"):
        rewritten(defined, transform)  # A is used whole in tmp's definition
    run_stencil(
        rewritten(defined, lambda tree: relayout_array(tree, "tmp", layout(LAYOUT_CASES["composed"]))).function, 9
    )


if __name__ == "__main__":
    for refused in (
        {"operations": [{"action": "replicate", "array": "A"}]},
        {"operations": [{"action": "place", "array": "A", "on": "device"}]},
        {"startup": {"load": ["blas"]}, "operations": []},
        {"operations": [{"action": "format", "array": "A", "to": "csr"}]},
    ):
        test_what_the_caller_carries_out_is_refused_by_name(refused)
    test_laid_out_arguments_and_locals_compute_the_same(
        [Permute("A", ("i1", "i0"))], [Pad("A", "i0", "2")], [Permute("A", ("i1", "i0"))]
    )
    test_laid_out_arguments_and_locals_compute_the_same(
        [Block("A", "i1", "4", ("o", "f")), Shuffle("A", ("f",), ("3 - f",), ("3 - f",))],
        [Pad("A", "i1", "1"), Permute("A", ("i1", "i0"))],
        [Block("A", "i0", "2", ("o", "f")), Permute("A", ("f", "i1", "o"))],
    )
    for name in LAYOUT_CASES:
        test_every_layout_case_computes_the_same_on_arguments_and_locals(name)
    test_whole_array_uses_go_through_the_logical_view()
    test_a_cut_array_used_whole_is_refused_by_its_line()
    test_library_calls_take_a_permuted_operand_through_its_view()
    test_a_json_request_lays_out_the_function()
    test_a_cast_changes_the_element_type_before_the_layout()
    test_a_local_is_cast_in_place_and_a_precision_numpy_lacks_is_refused()
    test_an_argument_written_in_part_keeps_the_cells_the_function_leaves()
    test_a_local_defined_by_an_expression_is_converted_in_and_negative_indices_count_from_the_end()
