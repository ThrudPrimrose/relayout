# Copyright 2026 ETH Zurich and the relayout authors.
"""How a library node takes an operand stored permuted (:mod:`relayout.rules` on dace's library nodes):
it flips a flag or renames its axes, or the layout is refused by name.
"""

from collections.abc import Callable
from typing import Any

import dace
from dace import nodes
from dace.libraries.blas.nodes.einsum import Einsum
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.matmul import MatMul
from dace.libraries.blas.nodes.symm import Symm
from dace.libraries.blas.nodes.syrk import Syrk
from dace.libraries.linalg import TensorDot, TensorTranspose
from dace.libraries.mpi.nodes.node import MPINode
from dace.libraries.standard.nodes.reduce import Reduce
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState

from relayout.passes.sdfg.common import LayoutError, as_range
from relayout.passes.sdfg.placement import Placement
from relayout.rules import einsum_operand, flips_transpose, transpose_axes

__all__ = ["LIBRARY_RULES", "library_rule"]


def blas_rule(state: SDFGState, node: Gemm | MatMul, connector: str, perm: tuple[int, ...], output: bool) -> None:
    """A 2-D BLAS operand stored transposed toggles its transpose flag (``transA`` / ``transB``); an
    output stored transposed swaps the operands, ``C^T = op(B)^T op(A)^T`` (its ``_cin``, the same
    array, follows)."""
    if not flips_transpose(perm) or connector == "_cin":
        return
    if connector == "_a":
        node.transA = not node.transA
    elif connector == "_b":
        node.transB = not node.transB
    else:
        for edge in state.in_edges(node):
            if edge.dst_conn in ("_a", "_b"):
                edge.dst_conn = "_b" if edge.dst_conn == "_a" else "_a"
        node.transA, node.transB = not node.transB, not node.transA


def einsum_rule(state: SDFGState, node: Einsum, connector: str, perm: tuple[int, ...], output: bool) -> None:
    """An einsum operand (or its output) stored permuted names its subscripts in stored order."""
    inputs, _, output = node.einsum_str.partition("->")
    groups = [group.strip() for group in inputs.split(",")]
    if connector.startswith("_ein"):
        k = sorted(c for c in node.in_connectors if c.startswith("_ein")).index(connector)
        groups[k] = einsum_operand(groups[k], perm)
    else:
        output = einsum_operand(output.strip(), perm)
    node.einsum_str = f"{','.join(groups)}->{output.strip()}"


def tensor_transpose_rule(
    state: SDFGState, node: TensorTranspose, connector: str, perm: tuple[int, ...], output: bool
) -> None:
    """``out = transpose(inp, axes)`` with ``inp`` (or ``out``) stored permuted."""
    axes = list(node.axes)
    node.axes = list(transpose_axes(axes, perm)) if connector == "_inp_tensor" else [axes[k] for k in perm]


def operand_rank(state: SDFGState, node: nodes.LibraryNode, connector: str) -> int:
    edge = next(e for e in state.in_edges(node) if e.dst_conn == connector)
    return len(as_range(edge.data.subset))


def tensor_dot_rule(state: SDFGState, node: TensorDot, connector: str, perm: tuple[int, ...], output: bool) -> None:
    """A contraction with an operand (or its output) stored permuted: one letter per operand axis,
    shared by each contracted pair; the stored order renames the contracted axes, and ``permutation``
    keeps the output's axes where they were."""
    letters = iter("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ")
    left = [next(letters) for _ in range(operand_rank(state, node, "_left_tensor"))]
    right: list[str] = [""] * operand_rank(state, node, "_right_tensor")
    for a, b in zip(node.left_axes, node.right_axes, strict=True):
        right[b] = left[a]
    right = [label or next(letters) for label in right]
    pairs = [left[a] for a in node.left_axes]

    def standard(lhs: list[str], rhs: list[str]) -> list[str]:
        return [c for c in lhs if c not in pairs] + [c for c in rhs if c not in pairs]

    old = standard(left, right)
    out = [old[k] for k in (node.permutation or range(len(old)))]
    if connector == "_out_tensor":
        out = [out[k] for k in perm]
    elif connector == "_left_tensor":
        left = list(einsum_operand("".join(left), perm))
    else:
        right = list(einsum_operand("".join(right), perm))
    node.left_axes = [left.index(c) for c in pairs]
    node.right_axes = [right.index(c) for c in pairs]
    new = standard(left, right)
    permutation = [new.index(c) for c in out]
    node.permutation = None if permutation == list(range(len(permutation))) else permutation


def syrk_rule(state: SDFGState, node: Syrk, connector: str, perm: tuple[int, ...], output: bool) -> None:
    """``C = alpha op(A) op(A)^T + beta C``: ``A`` stored transposed toggles ``trans``; ``C`` stored transposed
    keeps the same symmetric update in the other triangle (``uplo``), flipped once, on its output edge."""
    if not flips_transpose(perm):
        return
    if connector == "_a":
        node.trans = "N" if node.trans == "T" else "T"
    elif connector == "_c" and output:
        node.uplo = "U" if node.uplo == "L" else "L"


def symm_rule(state: SDFGState, node: Symm, connector: str, perm: tuple[int, ...], output: bool) -> None:
    """``C = alpha A B + beta C`` with ``A`` symmetric: ``A`` stored transposed is the same matrix with its
    other triangle referenced (``uplo``); ``B`` or ``C`` stored transposed has no flag and is refused."""
    if not flips_transpose(perm):
        return
    if connector != "_a":
        raise LayoutError(f"{state.label}: Symm {node.label!r} takes {connector!r} only as it is; lay out A")
    node.uplo = "U" if node.uplo == "L" else "L"


def reduce_rule(state: SDFGState, node: Reduce, connector: str, perm: tuple[int, ...], output: bool) -> None:
    """A reduction of an input stored permuted reduces the stored positions of its axes, as long as the
    axes it keeps stay in their order (the output's); an output stored permuted is refused."""
    kept_order = [axis for axis in perm if node.axes is not None and axis not in node.axes]
    if output or node.axes is None or kept_order != sorted(kept_order):
        raise LayoutError(f"{state.label}: reduce {node.label!r} cannot keep its output order under this layout")
    node.axes = sorted(perm.index(axis) for axis in node.axes)


#: Library node type -> how it takes an operand (by connector) stored permuted.
LIBRARY_RULES: dict[type, Callable[[SDFGState, Any, str, tuple[int, ...], bool], None]] = {
    Gemm: blas_rule,
    MatMul: blas_rule,
    Einsum: einsum_rule,
    TensorTranspose: tensor_transpose_rule,
    TensorDot: tensor_dot_rule,
    Syrk: syrk_rule,
    Symm: symm_rule,
    Reduce: reduce_rule,
}


def library_rule(state: SDFGState, edge: MultiConnectorEdge[dace.Memlet], placement: Placement) -> None:
    """A library node at ``edge`` takes the placement by its rule, or the placement is refused."""
    for node, connector in ((edge.src, edge.src_conn), (edge.dst, edge.dst_conn)):
        if not isinstance(node, nodes.LibraryNode) or isinstance(node, MPINode):
            continue
        perm = placement.permutation()
        rule = LIBRARY_RULES.get(type(node))
        if rule is None or perm is None:
            raise LayoutError(
                f"{state.label}: library node {node.label!r} ({type(node).__name__}) has no rule for this layout"
            )
        if perm != tuple(range(len(perm))):
            rule(state, node, connector, perm, node is edge.src)
