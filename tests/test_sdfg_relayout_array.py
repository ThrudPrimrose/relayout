# Copyright 2026 ETH Zurich and the relayout authors.
"""The SDFG passes: a program whose arrays are laid out (arguments converted in and out, transients
in place) computes what it computed before."""

import copy

import dace
import numpy as np
import pytest
from tests.conftest import LAYOUT_CASES, variant
from dace import nodes
from dace.libraries.blas.nodes.symm import Symm
from dace.libraries.blas.nodes.syrk import Syrk

from relayout.passes.sdfg.common import LayoutError

from relayout.passes.sdfg.apply_requests import apply_requests

from relayout.passes.sdfg.relayout_array import relayout_array
from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Block, Pad, Permute, Shuffle, run
from relayout.requests import parse_requests

N, M = (dace.symbol(s) for s in "NM")


@dace.program
def stencil(A: dace.float64[N, M], B: dace.float64[N, M]):
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
    variant(sdfg, f"{len(a_ops)}_{len(b_ops)}_{len(tmp_ops)}")
    shape = ["N", "M"]
    relayout_array(sdfg, "A", layout(a_ops, shape))
    relayout_array(sdfg, "B", layout(b_ops, shape))
    relayout_array(sdfg, "tmp", layout(tmp_ops, shape))
    sdfg.validate()
    a = np.random.default_rng(0).standard_normal((6, 12))
    b = np.zeros((6, 12))
    sdfg(A=a, B=b, N=6, M=12)
    np.testing.assert_allclose(b, reference(a))


@pytest.mark.parametrize("case", LAYOUT_CASES)
def test_every_layout_case_computes_the_same_on_arguments_and_transients(case: str) -> None:
    """End to end: one layout case on the input, the output and the transient of a compiled program."""
    sdfg = stencil.to_sdfg(simplify=True)
    variant(sdfg, f"case_{case}")
    for name in ("A", "B", "tmp"):
        relayout_array(sdfg, name, layout(LAYOUT_CASES[case], ["N", "M"]))
    sdfg.validate()
    a = np.random.default_rng(1).standard_normal((6, 12))
    b = np.zeros((6, 12))
    sdfg(A=a, B=b, N=6, M=12)
    np.testing.assert_allclose(b, reference(a))


K = dace.symbol("K")


@dace.program
def matmul(A: dace.float64[N, K], B: dace.float64[K, M], C: dace.float64[N, M]):
    C[:] = A @ B


@dace.program
def contraction(A: dace.float64[N, K, M], B: dace.float64[K, M], C: dace.float64[N]):
    C[:] = np.tensordot(A, B, axes=([1, 2], [0, 1]))


@dace.program
def einsum(A: dace.float64[N, K], B: dace.float64[K, M], C: dace.float64[N, M]):
    C[:] = np.einsum("ij,jk->ik", A, B)


@dace.program
def transpose(A: dace.float64[N, K, M], C: dace.float64[M, N, K]):
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
    variant(sdfg, "laid")
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
    """A rewrite of one node's program refuses what only the caller of the program can carry out."""
    sdfg = stencil.to_sdfg(simplify=True)
    with pytest.raises(LayoutError, match="belong to the caller"):
        apply_requests(sdfg, parse_requests(request_json))


def test_a_cast_changes_the_element_type_before_the_layout() -> None:
    """``cast`` A to fp32 (FP-Arena, the interface stays fp64) and permute the fp32 copy."""
    pytest.importorskip("fp_arena")
    request = parse_requests(
        {
            "operations": [
                {"action": "cast", "array": "A", "to": "fp32"},
                {"action": "permute", "array": "A", "order": ["i1", "i0"]},
            ]
        }
    )
    sdfg = stencil.to_sdfg(simplify=True)
    variant(sdfg, "cast")
    working = apply_requests(sdfg, request)
    assert working == {"A": "A_float32"} and sdfg.arrays["A_float32"].dtype == dace.float32
    assert [str(s) for s in sdfg.arrays["A_float32"].shape] == ["M", "N"]
    a = np.random.default_rng(6).standard_normal((3, 8))
    b = np.zeros((3, 8))
    sdfg(A=a, B=b, N=3, M=8)
    np.testing.assert_allclose(b, reference(a.astype(np.float32).astype(np.float64)), rtol=1e-6)


def library_sdfg(name: str, node: nodes.LibraryNode, operands: dict[str, tuple[str, tuple[int, ...]]]) -> dace.SDFG:
    """One library node wired to arrays: ``{connector: (array, shape)}``, an output connector last."""
    sdfg = dace.SDFG(name)
    state = sdfg.add_state()
    state.add_node(node)
    for connector, (array, shape) in operands.items():
        if array not in sdfg.arrays:
            sdfg.add_array(array, shape, dace.float64)
        memlet = dace.Memlet.from_array(array, sdfg.arrays[array])
        if connector in node.in_connectors:
            state.add_edge(state.add_read(array), None, node, connector, memlet)
        if connector in node.out_connectors:
            state.add_edge(node, connector, state.add_write(array), None, copy.deepcopy(memlet))
    return sdfg


def transposed(array: str, shape: tuple[int, ...]) -> IndexMap:
    return run([Permute("A", ("i1", "i0"))], [ArraySpec("A", 2)], (), "ops").layout("A", [str(n) for n in shape])


def test_syrk_symm_and_reduce_absorb_a_transposed_operand() -> None:
    """Syrk: A and C transposed (trans and uplo flip); Symm: A transposed (uplo flips); Reduce over a
    transposed input (its axes follow)."""
    rng = np.random.default_rng(8)
    a = rng.standard_normal((4, 3))
    syrk = library_sdfg("laid_syrk", Syrk("syrk", uplo="L", trans="N"), {"_a": ("A", (4, 3)), "_c": ("C", (4, 4))})
    relayout_array(syrk, "A", transposed("A", (4, 3)))
    relayout_array(syrk, "C", transposed("C", (4, 4)))
    syrk.expand_library_nodes()
    c = np.zeros((4, 4))
    syrk(A=a.copy(), C=c)
    np.testing.assert_allclose(np.tril(c), np.tril(a @ a.T))

    s = rng.standard_normal((4, 4))
    s = s + s.T
    b = rng.standard_normal((4, 2))
    symm = library_sdfg(
        "laid_symm", Symm("symm", side="L", uplo="L"), {"_a": ("A", (4, 4)), "_b": ("B", (4, 2)), "_c": ("C", (4, 2))}
    )
    relayout_array(symm, "A", transposed("A", (4, 4)))
    symm.expand_library_nodes()
    c = np.zeros((4, 2))
    symm(A=np.tril(s) + np.triu(rng.standard_normal((4, 4)), 1), B=b, C=c)
    np.testing.assert_allclose(c, s @ b)

    sdfg = dace.SDFG("laid_reduce")
    sdfg.add_array("A", (4, 3), dace.float64)
    sdfg.add_array("R", (3,), dace.float64)
    state = sdfg.add_state()
    reduce = state.add_reduce("lambda x, y: x + y", axes=[0], identity=0)
    state.add_edge(state.add_read("A"), None, reduce, None, dace.Memlet("A[0:4, 0:3]"))
    state.add_edge(reduce, None, state.add_write("R"), None, dace.Memlet("R[0:3]"))
    relayout_array(sdfg, "A", transposed("A", (4, 3)))
    assert reduce.axes == [1]
    r = np.zeros(3)
    sdfg(A=a.copy(), R=r)
    np.testing.assert_allclose(r, a.sum(axis=0))


def test_a_library_node_without_a_rule_is_refused() -> None:
    @dace.program
    def product(A: dace.float64[N, N], x: dace.float64[N], y: dace.float64[N]):
        y[:] = A @ x

    sdfg = product.to_sdfg(simplify=True)
    with pytest.raises(LayoutError, match="has no rule"):
        relayout_array(sdfg, "A", layout([Block("A", "i0", "2", ("o", "f"))], ["N", "N"]))


@dace.program
def half(A: dace.float64[N, M], B: dace.float64[N, M]):
    for i, j in dace.map[0 : N // 2, 0:M]:
        B[i, j] = A[i, j] + 1.0


def test_an_argument_written_in_part_keeps_the_cells_the_program_leaves() -> None:
    sdfg = half.to_sdfg(simplify=True)
    variant(sdfg, "partial")
    relayout_array(sdfg, "B", layout([Permute("A", ("i1", "i0"))], ["N", "M"]))
    a = np.random.default_rng(3).random((6, 4))
    b = np.random.default_rng(4).random((6, 4))
    want = b.copy()
    want[:3] = a[:3] + 1.0
    sdfg(A=a, B=b, N=6, M=4)
    np.testing.assert_array_equal(b, want)


if __name__ == "__main__":
    for refused in (
        {"operations": [{"action": "replicate", "array": "A"}]},
        {"operations": [{"action": "place", "array": "A", "on": "device"}]},
        {"startup": {"load": ["blas"]}, "operations": []},
        {"operations": [{"action": "format", "array": "A", "to": "csr"}]},
    ):
        test_what_the_caller_carries_out_is_refused_by_name(refused)
    test_syrk_symm_and_reduce_absorb_a_transposed_operand()
    test_a_cast_changes_the_element_type_before_the_layout()
    test_an_argument_written_in_part_keeps_the_cells_the_program_leaves()
    test_laid_out_arguments_and_transients_compute_the_same(
        [Permute("A", ("i1", "i0"))], [Pad("A", "i0", "2")], [Permute("A", ("i1", "i0"))]
    )
    for name in LAYOUT_CASES:
        test_every_layout_case_computes_the_same_on_arguments_and_transients(name)
    test_a_json_request_lays_out_the_program()
    test_a_library_node_without_a_rule_is_refused()
