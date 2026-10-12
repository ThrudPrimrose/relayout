# Copyright 2026 ETH Zurich and the relayout authors.
"""An SDFG rewritten to run on laid-out arrays: one rewrite for every layout.

:func:`relayout_array` lays one array out by an :class:`~relayout.layout.IndexMap`. Its descriptor
becomes the packed row-major buffer of the layout's extents, and every access follows the layout:

- a point access (a memlet of one element, an interstate subscript) reads the physical cell;
- a range access keeps its exact range on the axes a permutation or a pad moves, and covers a whole
  axis that a cut or a renumbering spreads it over (re-propagated from the point accesses inside);
- the arrays a nested SDFG receives are laid out the same way, with the nested descriptor's shape;
- a library node reading or writing it absorbs a permutation by its rule (:mod:`relayout.rules`) or
  refuses the layout by name.

A transient is laid out in place. An argument of the SDFG keeps its descriptor for the caller: the
program works on a transient ``laid_<name>``, converted in at the start (when it is read) and out at
the end (when it is written) by maps over the logical index space.
"""

import ast
import copy
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import dace
import sympy
from dace import data, nodes, subsets, symbolic
from dace.libraries.blas.nodes.einsum import Einsum
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.matmul import MatMul
from dace.libraries.linalg import TensorDot, TensorTranspose
from dace.libraries.mpi.nodes import Irecv, Wait, Waitall
from dace.libraries.mpi.nodes.node import MPINode
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.properties import CodeBlock
from dace.transformation.interstate.expand_nested_sdfg_inputs import ExpandNestedSDFGInputs
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.canonicalize.untile_loops import UntileLoops
from dace.transformation.passes.insert_explicit_copies import InsertExplicitCopies
from dace.transformation.passes.remove_views import RemoveViews
from dace.sdfg import propagation
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.dataflow.map_unroll import MapUnroll
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.passes.analysis import loop_analysis

from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Cast, DigitOperation, Requests, Unzip, Zip
from relayout.rules import einsum_operand, flips_transpose, transpose_axes

__all__ = [
    "Field",
    "Laid",
    "LayoutError",
    "Placement",
    "apply_requests",
    "laid_name",
    "physical_subset",
    "prepare",
    "relayout_array",
    "rewrite_subscripts",
    "unzip_array",
    "zip_arrays",
]

#: The prefix of the transient an argument is laid out into.
LAID = "laid_"


class LayoutError(ValueError):
    """A use of an array the layout cannot be carried through."""


def laid_name(name: str) -> str:
    return f"{LAID}{name}"


#: A dace symbolic value: a sympy expression or a plain int.
type Symbolic = sympy.Expr | int


def prepare(sdfg: dace.SDFG, target: str = "cpu") -> dace.SDFG:
    """``sdfg`` in the form the rewrite reads best, in place: tiled loops and blocked arrays untiled,
    canonicalized (loops become maps), nested SDFGs receiving whole arrays, no views but those a
    library node reads, implicit copies explicit; a non-packed array is refused."""
    UntileLoops(unblock_arrays=True).apply_pass(sdfg, {})
    canonicalize(sdfg, target=target, validate=True)
    sdfg.apply_transformations_repeated(ExpandNestedSDFGInputs)
    RemoveViews().apply_pass(sdfg, {})
    InsertExplicitCopies().apply_pass(sdfg, {})
    for _, name, desc in sdfg.arrays_recursive():
        if isinstance(desc, data.Array) and not isinstance(desc, (data.View, data.ContainerArray)):
            if not (desc.is_packed_c_strides() or desc.is_packed_fortran_strides()):
                raise LayoutError(f"{name!r} has strides {desc.strides} over {desc.shape}; lay out packed arrays only")
    sdfg.validate()
    return sdfg


def expr(text: str) -> sympy.Expr:
    value = symbolic.pystr_to_symbolic(text)
    if not isinstance(value, sympy.Expr):
        raise LayoutError(f"{text!r} is not an arithmetic expression")
    return value


def as_range(subset: subsets.Subset | None) -> subsets.Range:
    if not isinstance(subset, subsets.Range):
        raise LayoutError(f"{subset} is not a range; only ranges are laid out")
    return subset


def feeding_dims(layout: IndexMap, axis: int) -> set[int]:
    """The logical dimensions physical axis ``axis``'s value depends on: its own, and through a
    renumbering every dimension of the axes renumbered with it."""
    name = layout.axes[axis].name
    group = {name}.union(*(set(r.axes) for r in layout.renumbers if name in r.axes))
    return {a.dim for a in layout.axes if a.name in group}


def physical_subset(subset: subsets.Subset | None, layout: IndexMap) -> subsets.Range:
    """``subset`` (one range per logical dimension) as a range of the laid-out buffer: exact for a point
    and for any range of a permuted or padded array, else the whole extent of every axis a ranged
    dimension spreads over."""
    ranges = list(as_range(subset).ranges)
    if len(ranges) != len(layout.shape):
        raise LayoutError(f"a subset of rank {len(ranges)} on a rank-{len(layout.shape)} array: {subset}")
    if layout.view() is not None:
        return subsets.Range([ranges[axis.dim] for axis in layout.axes])
    point = [symbolic.equal_valued(begin, end) for begin, end, _ in ranges]
    position = layout.physical([str(begin) for begin, _, _ in ranges])
    out = []
    for k, (value, extent) in enumerate(zip(position, layout.extents(), strict=True)):
        if all(point[dim] for dim in feeding_dims(layout, k)):
            out.append((expr(value), expr(value), 1))
        else:
            out.append((0, expr(extent) - 1, 1))
    return subsets.Range(out)


class Placement(Protocol):
    """Where each cell of an array's logical index space lives in a target buffer: a layout of its own
    (:class:`Laid`) or one index of a zipped array's field dimension (:class:`Field`)."""

    def point(self, index: Sequence[str]) -> list[str]:
        """The target position of the logical index ``index``."""
        ...

    def box(self, subset: subsets.Subset | None) -> subsets.Range:
        """The target range covering the logical box ``subset``."""
        ...

    def view_strides(self, varying: Sequence[int], target: data.Array) -> list[object] | None:
        """The strides a view of a box (its ``varying`` dimensions, in order) takes in ``target``, or None
        when the box is no strided slice of it."""
        ...

    def in_order(self, subset: subsets.Subset | None, target: data.Array) -> bool:
        """Whether the target cells of the box are one contiguous run in the box's own order."""
        ...

    def permutation(self) -> tuple[int, ...] | None:
        """The axis permutation the target stores the array in (a library rule's input), else None."""
        ...

    def nested(self, shape: Sequence[str]) -> "Placement":
        """The same placement for a nested SDFG's array of logical ``shape``."""
        ...


@dataclass(frozen=True, slots=True)
class Laid:
    """An array in a buffer of its own laid out by ``layout``."""

    layout: IndexMap

    def point(self, index: Sequence[str]) -> list[str]:
        return list(self.layout.physical(index))

    def box(self, subset: subsets.Subset | None) -> subsets.Range:
        return physical_subset(subset, self.layout)

    def view_strides(self, varying: Sequence[int], target: data.Array) -> list[object] | None:
        if self.layout.view() is None:
            return None
        axis_of = {axis.dim: k for k, axis in enumerate(self.layout.axes)}
        return [target.strides[axis_of[d]] for d in varying]

    def in_order(self, subset: subsets.Subset | None, target: data.Array) -> bool:
        return in_order(subset, self.layout)

    def permutation(self) -> tuple[int, ...] | None:
        return self.layout.permutation()

    def nested(self, shape: Sequence[str]) -> "Laid":
        return Laid(self.layout.with_shape(shape))


@dataclass(frozen=True, slots=True)
class Field:
    """An array stored at ``index`` of a larger array's dimension ``dim`` (None = after the last): a field
    of a zipped array, or one part of an array split along a dimension."""

    index: int
    dim: int | None = None

    def insert[T](self, values: Sequence[T], value: T) -> list[T]:
        at = len(values) if self.dim is None else self.dim
        return [*values[:at], value, *values[at:]]

    def point(self, index: Sequence[str]) -> list[str]:
        return self.insert(index, str(self.index))

    def box(self, subset: subsets.Subset | None) -> subsets.Range:
        return subsets.Range(self.insert(list(as_range(subset).ranges), (self.index, self.index, 1)))

    def view_strides(self, varying: Sequence[int], target: data.Array) -> list[object] | None:
        dims = self.insert(list(range(len(target.shape) - 1)), -1)
        return [target.strides[dims.index(d)] for d in varying]

    def in_order(self, subset: subsets.Subset | None, target: data.Array) -> bool:
        return all(symbolic.equal_valued(b, e) for b, e, _ in as_range(subset).ranges) or symbolic.equal_valued(
            target.shape[-1], 1
        )

    def permutation(self) -> tuple[int, ...] | None:
        return None

    def nested(self, shape: Sequence[str]) -> "Field":
        return self


def rewrite_subscripts(text: str, name: str, new_name: str, indices: Callable[[list[str]], Sequence[str]]) -> str:
    """``text`` (an interstate expression) with every ``name[...]`` subscript renamed ``new_name`` and its
    indices replaced by ``indices`` of them, located structurally (a text match would run past the
    access's bracket and swallow the rest of the expression)."""

    class Replace(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            self.generic_visit(node)
            if not (isinstance(node.value, ast.Name) and node.value.id == name):
                return node
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            new = [ast.parse(e, mode="eval").body for e in indices([ast.unparse(e) for e in elements])]
            return ast.Subscript(ast.Name(new_name, ast.Load()), ast.Tuple(new, ast.Load()), node.ctx)

    if name not in text:
        return text
    return ast.unparse(Replace().visit(ast.parse(text, mode="eval")))


def nested_uses(sdfg: dace.SDFG, name: str) -> Iterator[tuple[dace.SDFG, str]]:
    """Each nested SDFG ``name`` flows into, with its inner connector, once (a read-write array's two
    connectors are one inner array)."""
    seen: set[tuple[int, str]] = set()
    for state in sdfg.states():
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            edges = [(e, e.dst_conn) for e in state.in_edges(node)] + [(e, e.src_conn) for e in state.out_edges(node)]
            for edge, connector in edges:
                if edge.data.data == name and connector is not None and (id(node.sdfg), connector) not in seen:
                    seen.add((id(node.sdfg), connector))
                    yield node.sdfg, connector


def laid_out_descriptor(desc: data.Array, layout: IndexMap, transient: bool) -> data.Array:
    """``desc`` as the packed row-major buffer of ``layout``."""
    shape = [expr(extent) for extent in layout.extents()]
    out = copy.deepcopy(desc)
    out.transient = transient
    out.set_shape(shape, strides=packed_strides(shape), total_size=product(shape))
    return out


def packed_strides(shape: Sequence[Symbolic]) -> list[Symbolic]:
    strides: list[Symbolic] = [sympy.Integer(1)] * len(shape)
    for k in range(len(shape) - 2, -1, -1):
        strides[k] = sympy.Integer(1) * strides[k + 1] * shape[k + 1]
    return strides


def product(values: Sequence[Symbolic]) -> Symbolic:
    out: Symbolic = sympy.Integer(1)
    for value in values:
        out = out * value
    return out


def rewrite_accesses(sdfg: dace.SDFG, name: str, target: str, placement: Placement) -> None:
    """Every access to ``name`` in ``sdfg`` moved to ``target`` (already described in ``sdfg``) through
    ``placement``. A nested SDFG ``name`` flows into receives the whole target: its array keeps its name,
    takes the target's descriptor and is rewritten the same way."""
    target_desc = sdfg.arrays[target]
    if not isinstance(target_desc, data.Array):
        raise LayoutError(f"{target!r} is no array")
    for inner_sdfg, inner in list(nested_uses(sdfg, name)):
        inner_desc = inner_sdfg.arrays[inner]
        if len(inner_desc.shape) != len(sdfg.arrays[name].shape):
            raise LayoutError(
                f"nested SDFG {inner_sdfg.label!r} receives {name!r} as {inner!r} of rank {len(inner_desc.shape)}, "
                f"a slice of the rank-{len(sdfg.arrays[name].shape)} array; pass it whole (prepare widens nested inputs)"
            )
        inner_target = copy.deepcopy(target_desc)
        inner_target.transient = False
        inner_sdfg.arrays[inner] = inner_target
        rewrite_accesses_in(inner_sdfg, inner, inner, placement.nested([str(s) for s in inner_desc.shape]))
    rewrite_accesses_in(sdfg, name, target, placement)


def rewrite_accesses_in(sdfg: dace.SDFG, name: str, target: str, placement: Placement) -> None:
    """:func:`rewrite_accesses` within ``sdfg`` itself (its nested SDFGs done)."""
    target_desc = sdfg.arrays[target]
    for state in sdfg.states():
        for copy_node in [n for n in state.nodes() if isinstance(n, CopyLibraryNode)]:
            if any(e.data.data == name for e in state.all_edges(copy_node)):
                expand_copy(state, copy_node)
        for node in state.data_nodes():
            if node.data == name:
                node.data = target
        for edge in state.edges():
            memlet = edge.data
            if memlet.is_empty():
                continue
            if memlet.data == name and mpi_buffer(edge) and not placement.in_order(memlet.subset, target_desc):
                packed_transfer(sdfg, state, edge, target, placement)
            elif memlet.data == name:
                view = next((n for n in (edge.src, edge.dst) if is_view(sdfg, n)), None)
                if view is not None:
                    restride_view(sdfg, view.data, memlet.subset, placement, target_desc)
                memlet.data = target
                nested = any(isinstance(n, nodes.NestedSDFG) for n in (edge.src, edge.dst))
                memlet.subset = subsets.Range.from_array(target_desc) if nested else placement.box(memlet.subset)
                library_rule(state, edge, placement)
            elif memlet.other_subset is not None and target in (access_name(edge.src), access_name(edge.dst)):
                memlet.other_subset = placement.box(memlet.other_subset)
    for edge in sdfg.all_interstate_edges():
        assignments = edge.data.assignments
        edge.data.assignments = {
            k: rewrite_subscripts(v, name, target, placement.point) for k, v in assignments.items()
        }
        condition = edge.data.condition.as_string
        if name in condition:
            edge.data.condition = CodeBlock(rewrite_subscripts(condition, name, target, placement.point))


def is_view(sdfg: dace.SDFG, node: nodes.Node) -> bool:
    return isinstance(node, nodes.AccessNode) and isinstance(sdfg.arrays[node.data], data.View)


def restride_view(
    sdfg: dace.SDFG, view: str, subset: subsets.Subset | None, placement: Placement, target: data.Array
) -> None:
    """A view of the box ``subset`` reads the target with the target's strides: each view dimension is a
    varying dimension of the box, in order. A cut or a shuffle has no such strides."""
    ranges = as_range(subset).ranges
    varying = [d for d, (begin, end, _) in enumerate(ranges) if not symbolic.equal_valued(begin, end)]
    desc = sdfg.arrays[view]
    if len(desc.shape) != len(varying):
        raise LayoutError(
            f"{view!r} reshapes its slice ({len(varying)} varying dimensions, view rank {len(desc.shape)})"
        )
    strides = placement.view_strides(varying, target)
    if strides is None:
        raise LayoutError(f"{view!r} views a slice of an array laid out by cuts or a shuffle; copy the slice instead")
    desc.strides = tuple(strides)


def access_name(node: nodes.Node) -> str | None:
    return node.data if isinstance(node, nodes.AccessNode) else None


def blas_rule(state: SDFGState, node: Gemm | MatMul, connector: str, perm: tuple[int, ...]) -> None:
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


def einsum_rule(state: SDFGState, node: Einsum, connector: str, perm: tuple[int, ...]) -> None:
    """An einsum operand (or its output) stored permuted names its subscripts in stored order."""
    inputs, _, output = node.einsum_str.partition("->")
    groups = [group.strip() for group in inputs.split(",")]
    if connector.startswith("_ein"):
        k = sorted(c for c in node.in_connectors if c.startswith("_ein")).index(connector)
        groups[k] = einsum_operand(groups[k], perm)
    else:
        output = einsum_operand(output.strip(), perm)
    node.einsum_str = f"{','.join(groups)}->{output.strip()}"


def tensor_transpose_rule(state: SDFGState, node: TensorTranspose, connector: str, perm: tuple[int, ...]) -> None:
    """``out = transpose(inp, axes)`` with ``inp`` (or ``out``) stored permuted."""
    axes = list(node.axes)
    node.axes = list(transpose_axes(axes, perm)) if connector == "_inp_tensor" else [axes[k] for k in perm]


def operand_rank(state: SDFGState, node: nodes.LibraryNode, connector: str) -> int:
    edge = next(e for e in state.in_edges(node) if e.dst_conn == connector)
    return len(as_range(edge.data.subset))


def tensor_dot_rule(state: SDFGState, node: TensorDot, connector: str, perm: tuple[int, ...]) -> None:
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


#: Library node type -> how it takes an operand (by connector) stored permuted.
LIBRARY_RULES: dict[type, Callable[[SDFGState, Any, str, tuple[int, ...]], None]] = {
    Gemm: blas_rule,
    MatMul: blas_rule,
    Einsum: einsum_rule,
    TensorTranspose: tensor_transpose_rule,
    TensorDot: tensor_dot_rule,
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
            rule(state, node, connector, perm)


def expand_copy(state: SDFGState, node: CopyLibraryNode) -> None:
    """A copy as a map of point copies over its box, so each side follows its own layout; the box's
    varying dimensions pair up in order, and a reduction (``wcr``) on the target stays."""
    source = next(e for e in state.in_edges(node) if e.dst_conn == CopyLibraryNode.INPUT_CONNECTOR_NAME)
    target = next(e for e in state.out_edges(node) if e.src_conn == CopyLibraryNode.OUTPUT_CONNECTOR_NAME)
    boxes = [as_range(e.data.subset) for e in (source, target)]
    varying = [[d for d, (b, e, _) in enumerate(box.ranges) if not symbolic.equal_valued(b, e)] for box in boxes]
    sizes = [[box.size()[d] for d in dims] for box, dims in zip(boxes, varying, strict=True)]
    if len(sizes[0]) != len(sizes[1]) or not all(symbolic.equal_valued(a, b) for a, b in zip(*sizes, strict=True)):
        raise LayoutError(
            f"{state.label}: copy {node.label!r} reshapes {boxes[0]} into {boxes[1]}; lay out a plain copy"
        )
    params = [f"__c{k}" for k in range(len(sizes[0]))]

    def point(box: subsets.Range, dims: list[int]) -> str:
        index = [str(begin) for begin, _, _ in box.ranges]
        for param, dim in zip(params, dims, strict=True):
            index[dim] = f"{index[dim]} + {param}"
        return ", ".join(index)

    src, dst = source.data.data, target.data.data
    if not (src and dst):
        raise LayoutError(f"{state.label}: copy {node.label!r} names no data on one side")
    inside = dace.Memlet(f"{src}[{point(boxes[0], varying[0])}]")
    outside = dace.Memlet(f"{dst}[{point(boxes[1], varying[1])}]", wcr=target.data.wcr)
    tasklet = state.add_tasklet(f"{node.label}_point", {"__inp"}, {"__out"}, "__out = __inp")
    if params:
        entry, exit_ = state.add_map(
            f"{node.label}_points", {p: f"0:{size}" for p, size in zip(params, sizes[0], strict=True)}
        )
        state.add_memlet_path(source.src, entry, tasklet, src_conn=source.src_conn, dst_conn="__inp", memlet=inside)
        state.add_memlet_path(tasklet, exit_, target.dst, src_conn="__out", dst_conn=target.dst_conn, memlet=outside)
    else:
        state.add_edge(source.src, source.src_conn, tasklet, "__inp", inside)
        state.add_edge(tasklet, "__out", target.dst, target.dst_conn, outside)
    state.remove_node(node)


def mpi_buffer(edge: MultiConnectorEdge[dace.Memlet]) -> bool:
    """Whether ``edge`` is an MPI library node's buffer (``_buffer``, ``_inbuffer``, ``_out_buffer``, ...)."""
    for node, connector in ((edge.dst, edge.dst_conn), (edge.src, edge.src_conn)):
        if isinstance(node, MPINode):
            return connector is not None and connector.endswith("buffer")
    return False


def in_order(subset: subsets.Subset | None, layout: IndexMap) -> bool:
    """Whether the laid-out cells of the logical box ``subset`` are one contiguous run in the box's own
    (row-major) order, so a transfer can use them as they are: a permuted or padded layout whose
    varying axes keep their order and are whole after the first; a cut never (unless a point)."""
    ranges = list(as_range(subset).ranges)
    view = layout.view()
    if view is None:
        return all(symbolic.equal_valued(begin, end) for begin, end, _ in ranges)
    extents = [expr(extent) for extent in layout.extents()]
    physical = [ranges[axis.dim] for axis in layout.axes]
    varying = [k for k, (begin, end, _) in enumerate(physical) if not symbolic.equal_valued(begin, end)]
    if not varying:
        return True
    dims = [layout.axes[k].dim for k in varying]
    whole = all(
        symbolic.equal_valued(physical[k][0], 0) and symbolic.equal_valued(physical[k][1], extents[k] - 1)
        for k in range(varying[0] + 1, len(physical))
    )
    return dims == sorted(dims) and whole


def wait_state(sdfg: dace.SDFG, state: SDFGState, node: nodes.LibraryNode) -> SDFGState:
    """The state holding the ``Wait`` of a non-blocking receive's request."""
    request = next((e.data.data for e in state.out_edges(node) if e.src_conn == "_request"), None)
    for candidate in sdfg.states():
        for wait in candidate.nodes():
            if isinstance(wait, (Wait, Waitall)) and any(
                e.data.data == request for e in candidate.in_edges(wait) if e.dst_conn == "_request"
            ):
                return candidate
    raise LayoutError(f"{state.label}: no Wait for the request of {node.label!r}; its unpack has no place")


def packed_transfer(
    sdfg: dace.SDFG, state: SDFGState, edge: MultiConnectorEdge[dace.Memlet], laid: str, placement: Placement
) -> None:
    """An MPI buffer of a laid-out array moved through a contiguous transient in logical order: packed
    from the laid-out cells before a send, unpacked into them after a receive (at the wait of an
    ``Irecv``). The wire order is the logical one, as the program had it before the layout."""
    box = as_range(edge.data.subset)
    sizes, first = box.size(), box.min_element()
    desc = sdfg.arrays[laid]
    packed, _ = sdfg.add_transient(
        f"packed_{laid}",
        sizes,
        desc.dtype,
        desc.storage,
        lifetime=dace.dtypes.AllocationLifetime.SDFG,
        find_new_name=True,
    )
    params = [f"__j{k}" for k in range(len(sizes))]
    ranges = {p: f"0:{size}" for p, size in zip(params, sizes, strict=True)}
    position = placement.point([f"{low} + {p}" for low, p in zip(first, params, strict=True)])
    laid_memlet = dace.Memlet(f"{laid}[{', '.join(position)}]")
    packed_memlet = dace.Memlet(f"{packed}[{', '.join(params)}]")
    whole = dace.Memlet.from_array(packed, sdfg.arrays[packed])
    sending = isinstance(edge.dst, nodes.LibraryNode)
    state.remove_edge(edge)
    if sending:
        source = edge.src
        if isinstance(source, nodes.AccessNode):
            source.data = laid
        packed_node = state.add_access(packed)
        state.add_mapped_tasklet(
            f"pack_{packed}",
            ranges,
            {"__inp": laid_memlet},
            "__out = __inp",
            {"__out": packed_memlet},
            input_nodes={laid: source},
            output_nodes={packed: packed_node},
            external_edges=True,
        )
        state.add_edge(packed_node, None, edge.dst, edge.dst_conn, whole)
        return
    packed_node = state.add_access(packed)
    state.add_edge(edge.src, edge.src_conn, packed_node, None, whole)
    if isinstance(edge.src, Irecv):
        if state.degree(edge.dst) == 0:
            state.remove_node(edge.dst)
        after = wait_state(sdfg, state, edge.src)
        target = after.parent_graph.add_state_after(after, f"unpack_{packed}")
        target.add_mapped_tasklet(
            f"unpack_{packed}",
            ranges,
            {"__inp": packed_memlet},
            "__out = __inp",
            {"__out": laid_memlet},
            external_edges=True,
        )
        return
    destination = edge.dst
    if isinstance(destination, nodes.AccessNode):
        destination.data = laid
    state.add_mapped_tasklet(
        f"unpack_{packed}",
        ranges,
        {"__inp": packed_memlet},
        "__out = __inp",
        {"__out": laid_memlet},
        input_nodes={packed: packed_node},
        output_nodes={laid: destination},
        external_edges=True,
    )


def conversion(
    state: SDFGState, logical: str, laid: str, shape: Sequence[str], placement: Placement, inward: bool
) -> None:
    """A map over the logical index space of ``shape`` moving ``logical`` into ``laid`` (or back)."""
    params = [f"__r{d}" for d in range(len(shape))]
    ranges = {p: f"0:{extent}" for p, extent in zip(params, shape, strict=True)}
    position = ", ".join(placement.point(params))
    logical_at, laid_at = ", ".join(params), position
    source, target = (
        (f"{logical}[{logical_at}]", f"{laid}[{laid_at}]")
        if inward
        else (f"{laid}[{laid_at}]", f"{logical}[{logical_at}]")
    )
    state.add_mapped_tasklet(
        f"relayout_{'in' if inward else 'out'}_{logical}",
        ranges,
        {"__inp": dace.Memlet(source)},
        "__out = __inp",
        {"__out": dace.Memlet(target)},
        external_edges=True,
    )


def is_read(sdfg: dace.SDFG, name: str) -> bool:
    """Whether ``sdfg`` reads ``name``: an access node it leaves, or a reduction (``wcr``) into it, which
    reads what it accumulates onto (the reduction's memlet sits on the edge inside the scope)."""
    for state in sdfg.states():
        if any(n.data == name and state.out_degree(n) > 0 for n in state.data_nodes()):
            return True
        if any(e.data.data == name and e.data.wcr is not None for e in state.edges()):
            return True
    return False


def is_written(sdfg: dace.SDFG, name: str) -> bool:
    return any(n.data == name and s.in_degree(n) > 0 for s in sdfg.states() for n in s.data_nodes())


def move_array(sdfg: dace.SDFG, name: str, target: str, placement: Placement) -> None:
    """Move ``name``'s accesses to ``target`` (described in ``sdfg``) through ``placement``. An argument
    keeps its descriptor for the caller and is converted into the target at the start (when read) and
    back at the end (when written); a transient is removed."""
    desc = sdfg.arrays[name]
    shape = [str(extent) for extent in desc.shape]
    read, written = is_read(sdfg, name), is_written(sdfg, name)
    rewrite_accesses(sdfg, name, target, placement)
    if desc.transient:
        if name != target:
            sdfg.remove_data(name, validate=False)
        return
    if read:
        start = sdfg.add_state_before(sdfg.start_block, f"relayout_in_{name}")
        conversion(start, name, target, shape, placement, inward=True)
    if written:
        for sink in [b for b in sdfg.nodes() if sdfg.out_degree(b) == 0]:
            conversion(sdfg.add_state_after(sink, f"relayout_out_{name}"), name, target, shape, placement, inward=False)


def array_desc(sdfg: dace.SDFG, name: str) -> data.Array:
    desc = sdfg.arrays[name]
    if not isinstance(desc, data.Array) or isinstance(desc, data.View):
        raise LayoutError(f"{name!r} is a {type(desc).__name__}; only arrays are laid out")
    return desc


def relayout_array(sdfg: dace.SDFG, name: str, layout: IndexMap) -> str:
    """Lay ``sdfg``'s array ``name`` out by ``layout`` (see the module docstring); returns the name the
    program now works on (``name`` for a transient, ``laid_<name>`` for an argument)."""
    desc = array_desc(sdfg, name)
    if layout.is_identity():
        return name
    laid = name if desc.transient else laid_name(name)
    laid_desc = laid_out_descriptor(desc, layout, transient=True)
    if desc.transient:
        sdfg.arrays[name] = laid_desc
    else:
        sdfg.add_datadesc(laid, laid_desc)
    move_array(sdfg, name, laid, Laid(layout))
    propagation.propagate_memlets_sdfg(sdfg)
    return laid


def zip_arrays(sdfg: dace.SDFG, fields: Sequence[str], into: str) -> str:
    """Fuse ``sdfg``'s arrays ``fields`` (one shape, one element type) into the new transient ``into`` with
    a last dimension of the fields: ``into[..., k]`` is ``fields[k]``. Arguments keep their descriptors
    and are converted in and out (:func:`move_array`)."""
    descs = [array_desc(sdfg, field) for field in fields]
    shapes = {tuple(str(extent) for extent in desc.shape) for desc in descs}
    types = {desc.dtype for desc in descs}
    if len(shapes) != 1 or len(types) != 1:
        raise LayoutError(f"zip of {list(fields)}: the arrays must share one shape and one element type")
    shape = [*descs[0].shape, len(fields)]
    zipped = copy.deepcopy(descs[0])
    zipped.transient = True
    zipped.set_shape(shape, strides=packed_strides(shape), total_size=product(shape))
    sdfg.add_datadesc(into, zipped, find_new_name=False)
    for k, field in enumerate(fields):
        move_array(sdfg, field, into, Field(k))
    propagation.propagate_memlets_sdfg(sdfg)
    return into


def constant(value: object) -> int | None:
    """``value`` as an int when it is a constant, else None."""
    try:
        return int(str(value))
    except ValueError:
        return None


def split_indices(sdfg: dace.SDFG, name: str, dim: int) -> list[object]:
    """The index each point access of ``name`` (a memlet inside every scope; the scopes' propagated ranges
    follow from them) takes on dimension ``dim``, unless it is a constant."""
    found = []
    for state in sdfg.states():
        for edge in state.edges():
            leaf = not isinstance(edge.dst, nodes.EntryNode) and not isinstance(edge.src, nodes.ExitNode)
            if edge.data.data == name and not edge.data.is_empty() and leaf:
                begin, end, _ = as_range(edge.data.subset).ranges[dim]
                if not symbolic.equal_valued(begin, end) or constant(begin) is None:
                    found.append(begin if symbolic.equal_valued(begin, end) else (begin, end))
    return found


def trip_count(loop: LoopRegion) -> int | None:
    """A loop's constant number of iterations, else None."""
    start, end, step = (
        loop_analysis.get_init_assignment(loop),
        loop_analysis.get_loop_end(loop),
        loop_analysis.get_loop_stride(loop),
    )
    if start is None or end is None or step is None:
        return None
    return constant(symbolic.int_floor(end + 1 - start, step))


def unroll_over(sdfg: dace.SDFG, extent: int) -> None:
    """Unroll every map and loop of exactly ``extent`` iterations (the dimension an unzip splits), one at
    a time: an unroll replaces the nodes the search walked."""
    while True:
        for node, graph in sdfg.all_nodes_recursive():
            if isinstance(node, nodes.MapEntry) and all(constant(r) == extent for r in node.map.range.size()):
                assert isinstance(graph, SDFGState)
                MapUnroll().apply_to(sdfg=graph.sdfg, options={}, map_entry=node)
                break
            if isinstance(node, LoopRegion) and trip_count(node) == extent:
                LoopUnroll().apply_to(sdfg=node.sdfg, options={"inline_iterations": True}, loop=node)
                break
        else:
            return


def unzip_array(sdfg: dace.SDFG, name: str, into: Sequence[str], dim: int) -> list[str]:
    """Split ``sdfg``'s array ``name`` along dimension ``dim`` (constant extent ``len(into)``) into one array
    per index, ``into[c]`` holding ``name[..., c, ...]``: the maps and loops over the dimension are
    unrolled so every index on it is a constant. An argument keeps its descriptor, converted into the
    parts and back. A data-dependent index (or a nested SDFG receiving the array) is refused."""
    desc = array_desc(sdfg, name)
    if constant(desc.shape[dim]) != len(into):
        raise LayoutError(
            f"{name!r} has extent {desc.shape[dim]} on dimension {dim}; name one array per index ({len(into)} given)"
        )
    if any(True for _ in nested_uses(sdfg, name)):
        raise LayoutError(f"{name!r} flows into a nested SDFG; inline it before splitting the array")
    unroll_over(sdfg, len(into))
    dynamic = split_indices(sdfg, name, dim)
    if dynamic:
        raise LayoutError(f"{name!r} is indexed on dimension {dim} by {dynamic[0]}, not a constant; branch on it first")
    rest = [extent for k, extent in enumerate(desc.shape) if k != dim] or [1]
    for part in into:
        piece = copy.deepcopy(desc)
        piece.transient = True
        piece.set_shape(rest, strides=packed_strides(rest), total_size=product(rest))
        sdfg.add_datadesc(part, piece, find_new_name=False)

    def part_of(subset: subsets.Subset | None) -> tuple[str, subsets.Range]:
        ranges = list(as_range(subset).ranges)
        index = constant(ranges.pop(dim)[0])
        return into[index or 0], subsets.Range(ranges or [(0, 0, 1)])

    for state in sdfg.states():
        for node in [n for n in state.data_nodes() if n.data == name]:
            split_access(state, node, part_of)
    for edge in sdfg.all_interstate_edges():
        assignments = edge.data.assignments
        edge.data.assignments = {k: split_subscripts(v, name, into, dim) for k, v in assignments.items()}
    if not desc.transient:
        shape = [str(extent) for k, extent in enumerate(desc.shape) if k != dim]
        if is_read_in_parts(sdfg, into):
            start = sdfg.add_state_before(sdfg.start_block, f"unzip_in_{name}")
            for c, part in enumerate(into):
                conversion(start, part, name, shape, Field(c, dim), inward=False)
        sinks = [b for b in sdfg.nodes() if sdfg.out_degree(b) == 0]
        if any(is_written(sdfg, part) for part in into):
            end = sdfg.add_state_after(sinks[0], f"unzip_out_{name}")
            for c, part in enumerate(into):
                conversion(end, part, name, shape, Field(c, dim), inward=True)
    else:
        sdfg.remove_data(name, validate=False)
    propagation.propagate_memlets_sdfg(sdfg)
    return list(into)


def split_access(
    state: SDFGState, node: nodes.AccessNode, part_of: Callable[[subsets.Subset | None], tuple[str, subsets.Range]]
) -> None:
    """Each memlet path through access node ``node`` rebuilt from a new access node of the part its leaf
    (the point access inside the scopes) names, through the same scopes; ``node`` is then removed."""
    parts: dict[str, nodes.AccessNode] = {}
    for edge in list(state.out_edges(node)) + list(state.in_edges(node)):
        if edge not in state.edges():
            continue
        outgoing = edge.src is node
        tree = state.memlet_tree(edge)
        for leaf in list(tree.leaves()):
            if leaf.data.data != node.data:
                part, _ = part_of(leaf.data.other_subset)
                leaf.data.other_subset = part_of(leaf.data.other_subset)[1]
                memlet = leaf.data
            else:
                part, subset = part_of(leaf.data.subset)
                memlet = dace.Memlet(data=part, subset=subset, wcr=leaf.data.wcr, dynamic=leaf.data.dynamic)
            path = state.memlet_path(leaf)
            access = parts.setdefault(part, state.add_access(part))
            if outgoing:
                scopes = [e.dst for e in path[:-1]]
                state.add_memlet_path(access, *scopes, leaf.dst, dst_conn=leaf.dst_conn, memlet=memlet)
            else:
                scopes = [e.src for e in path[1:]]
                state.add_memlet_path(leaf.src, *scopes, access, src_conn=leaf.src_conn, memlet=memlet)
        for old in list(tree):
            state.remove_edge_and_connectors(old)
    state.remove_node(node)


def is_read_in_parts(sdfg: dace.SDFG, parts: Sequence[str]) -> bool:
    return any(is_read(sdfg, part) for part in parts)


def split_subscripts(text: str, name: str, into: Sequence[str], dim: int) -> str:
    """``text`` with every ``name[..., c, ...]`` subscript (a constant ``c`` on ``dim``) as ``into[c][...]``."""

    class Replace(ast.NodeTransformer):
        def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
            self.generic_visit(node)
            if not (isinstance(node.value, ast.Name) and node.value.id == name):
                return node
            elements = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
            index = elements.pop(dim)
            if not (isinstance(index, ast.Constant) and isinstance(index.value, int)):
                raise LayoutError(f"{name!r} is indexed by {ast.unparse(index)} in {text!r}; a split needs a constant")
            rest = elements or [ast.Constant(0)]
            return ast.Subscript(ast.Name(into[index.value], ast.Load()), ast.Tuple(rest, ast.Load()), node.ctx)

    if name not in text:
        return text
    return ast.unparse(Replace().visit(ast.parse(text, mode="eval")))


def apply_requests(sdfg: dace.SDFG, requests: Requests) -> dict[str, str]:
    """Carry ``requests``' operations out on ``sdfg`` (single node): ``zip`` / ``unzip`` regroup arrays in
    program order (:func:`zip_arrays`, :func:`unzip_array`), then every array a digit operation touched is
    laid out as the operations leave it (:func:`relayout_array`); returns each laid-out array's working
    name. A precision is an element type, FP-Arena's transformation and not a layout: a ``cast`` is
    refused here, as is a ``zip`` of different element types (a struct, not yet rewritten)."""
    casts = [op.array for op in requests.operations if isinstance(op, Cast)]
    if casts:
        raise LayoutError(f"cast of {casts}: an SDFG changes element types through FP-Arena, not relayout")

    def arrays() -> dict[str, data.Array]:
        return {n: d for n, d in sdfg.arrays.items() if isinstance(d, data.Array) and not isinstance(d, data.View)}

    state = requests.run([ArraySpec(name, len(desc.shape)) for name, desc in arrays().items()])
    for op in requests.operations:
        if isinstance(op, Zip):
            zip_arrays(sdfg, op.arrays, op.into)
        elif isinstance(op, Unzip):
            dim = len(sdfg.arrays[op.array].shape) - 1 if op.dim is None else op.dim
            unzip_array(sdfg, op.array, op.into, dim)
    laid_out = {op.array for op in requests.operations if isinstance(op, DigitOperation)}
    current = arrays()
    return {
        name: relayout_array(sdfg, name, state.layout(name, [str(extent) for extent in current[name].shape]))
        for name in sorted(laid_out)
    }
