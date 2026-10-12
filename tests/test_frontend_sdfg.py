# Copyright 2026 ETH Zurich and the relayout authors.
"""The SDFG frontend: a program whose arrays are laid out (arguments converted in and out, transients
in place) computes what it computed before."""

import numpy as np
import pytest

import dace
from relayout.frontends.sdfg import LayoutError, apply_requests, relayout_array
from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Block, Pad, Permute, Shuffle, run
from relayout.requests import parse_requests

N, M = (dace.symbol(s) for s in "NM")


@dace.program
def stencil(A: dace.float64[N, M], B: dace.float64[N, M]):  # noqa: N803
    tmp = dace.define_local([N, M], dace.float64)
    for i, j in dace.map[0:N, 0:M]:
        tmp[i, j] = A[i, j] * 2.0
    for i, j in dace.map[0:N, 0:M]:
        B[i, j] = tmp[i, j] + A[i, (j + 1) % M]


def layout(ops: list, shape: list[str]) -> IndexMap:
    return run(ops, [ArraySpec("A", len(shape))], (), "operations").layout("A", shape)


def reference(a: np.ndarray) -> np.ndarray:
    return a * 2.0 + np.roll(a, -1, axis=1)


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
def test_laid_out_arguments_and_transients_compute_the_same(a_ops: list, b_ops: list, tmp_ops: list) -> None:
    sdfg = stencil.to_sdfg(simplify=True)
    shape = ["N", "M"]
    relayout_array(sdfg, "A", layout(a_ops, shape))
    relayout_array(sdfg, "B", layout(b_ops, shape))
    relayout_array(sdfg, "tmp", layout(tmp_ops, shape))
    sdfg.validate()
    a = np.random.default_rng(0).standard_normal((6, 12))
    b = np.zeros((6, 12))
    sdfg(A=a, B=b, N=6, M=12)
    np.testing.assert_allclose(b, reference(a))


K = dace.symbol("K")


@dace.program
def matmul(A: dace.float64[N, K], B: dace.float64[K, M], C: dace.float64[N, M]):  # noqa: N803
    C[:] = A @ B


@dace.program
def contraction(A: dace.float64[N, K, M], B: dace.float64[K, M], C: dace.float64[N]):  # noqa: N803
    C[:] = np.tensordot(A, B, axes=([1, 2], [0, 1]))


@dace.program
def einsum(A: dace.float64[N, K], B: dace.float64[K, M], C: dace.float64[N, M]):  # noqa: N803
    C[:] = np.einsum("ij,jk->ik", A, B)


@dace.program
def transpose(A: dace.float64[N, K, M], C: dace.float64[M, N, K]):  # noqa: N803
    C[:] = np.transpose(A, (2, 0, 1))


def symbol_values(sdfg: dace.SDFG, shapes: dict[str, tuple[int, ...]]) -> dict[str, int]:
    """The size symbols an argument's declared shape names, bound by the array passed for it."""
    values = {}
    for name, shape in shapes.items():
        for declared, extent in zip(sdfg.arrays[name].shape, shape, strict=True):
            if isinstance(declared, dace.symbol):
                values[str(declared)] = extent
    return values


TRANSPOSED = [Permute("A", ("i1", "i0"))]
ROTATED = [Permute("A", ("i2", "i0", "i1"))]


@pytest.mark.parametrize(
    ("program", "shapes", "layouts", "reference"),
    [
        (matmul, {"A": (4, 5), "B": (5, 3), "C": (4, 3)}, {"A": TRANSPOSED, "B": TRANSPOSED}, lambda a, b: a @ b),
        (einsum, {"A": (4, 5), "B": (5, 3), "C": (4, 3)}, {"A": TRANSPOSED, "C": TRANSPOSED}, lambda a, b: a @ b),
        (
            contraction,
            {"A": (4, 5, 3), "B": (5, 3), "C": (4,)},
            {"A": ROTATED},
            lambda a, b: np.tensordot(a, b, axes=([1, 2], [0, 1])),
        ),
        (
            transpose,
            {"A": (4, 5, 3), "C": (3, 4, 5)},
            {"A": ROTATED, "C": ROTATED},
            lambda a: np.transpose(a, (2, 0, 1)),
        ),
    ],
)
def test_library_nodes_absorb_a_permutation(program: object, shapes: dict, layouts: dict, reference: object) -> None:
    sdfg = program.to_sdfg(simplify=True)  # type: ignore[attr-defined]
    for name, ops in layouts.items():
        names = [str(s) for s in sdfg.arrays[name].shape]
        relayout_array(sdfg, name, layout(ops, names))
    rng = np.random.default_rng(3)
    args = {name: rng.standard_normal(shape) for name, shape in shapes.items()}
    inputs = [args[name] for name in shapes if name != "C"]
    expected = reference(*inputs)  # type: ignore[operator]
    sdfg(**args, **symbol_values(sdfg, shapes))
    np.testing.assert_allclose(args["C"], expected, rtol=1e-12)


def test_a_json_request_lays_out_the_program() -> None:
    """The request an agent sends is the SDFG's layout: one JSON, the same operations."""
    request = parse_requests(
        {
            "operations": [
                {"action": "block", "array": "A", "digit": "i1", "factor": 4, "into": ["o", "f"]},
                {"action": "permute", "array": "A", "order": ["o", "i0", "f"]},
                {"action": "pad", "array": "B", "digit": "i1", "amount": 3},
            ]
        }
    )
    sdfg = stencil.to_sdfg(simplify=True)
    assert apply_requests(sdfg, request) == {"A": "laid_A", "B": "laid_B"}
    a = np.random.default_rng(5).standard_normal((3, 8))
    b = np.zeros((3, 8))
    sdfg(A=a, B=b, N=3, M=8)
    np.testing.assert_allclose(b, reference(a))
    with pytest.raises(LayoutError, match="FP-Arena"):
        apply_requests(
            stencil.to_sdfg(), parse_requests({"operations": [{"action": "cast", "array": "A", "to": "fp32"}]})
        )


def test_a_library_node_without_a_rule_is_refused() -> None:
    @dace.program
    def product(A: dace.float64[N, N], x: dace.float64[N], y: dace.float64[N]):  # noqa: N803
        y[:] = A @ x

    sdfg = product.to_sdfg(simplify=True)
    with pytest.raises(LayoutError, match="has no rule"):
        relayout_array(sdfg, "A", layout([Block("A", "i0", "2", ("o", "f"))], ["N", "N"]))


if __name__ == "__main__":
    test_laid_out_arguments_and_transients_compute_the_same(
        [Permute("A", ("i1", "i0"))], [Pad("A", "i0", "2")], [Permute("A", ("i1", "i0"))]
    )
    test_a_json_request_lays_out_the_program()
    test_a_library_node_without_a_rule_is_refused()
