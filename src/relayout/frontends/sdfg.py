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
from typing import Any

import dace
import sympy
from dace import data, nodes, subsets, symbolic
from dace.libraries.blas.nodes.einsum import Einsum
from dace.libraries.blas.nodes.gemm import Gemm
from dace.libraries.blas.nodes.matmul import MatMul
from dace.libraries.linalg import TensorDot, TensorTranspose
from dace.libraries.mpi.nodes import Irecv, Wait, Waitall
from dace.libraries.mpi.nodes.node import MPINode
from dace.properties import CodeBlock
from dace.sdfg import propagation
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState

from relayout.layout import IndexMap
from relayout.rules import einsum_operand, flips_transpose, transpose_axes

__all__ = ["LayoutError", "laid_name", "physical_subset", "relayout_array", "rewrite_subscripts"]

#: The prefix of the transient an argument is laid out into.
LAID = "laid_"


class LayoutError(ValueError):
    """A use of an array the layout cannot be carried through."""


def laid_name(name: str) -> str:
    return f"{LAID}{name}"


#: A dace symbolic value: a sympy expression or a plain int.
type Symbolic = sympy.Expr | int


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


def rewrite_accesses(sdfg: dace.SDFG, name: str, new_name: str, layout: IndexMap) -> None:
    """Every access to ``name`` in ``sdfg`` addressed through ``layout`` and renamed ``new_name``; the
    nested SDFGs it flows into are laid out in turn."""
    for inner_sdfg, inner in list(nested_uses(sdfg, name)):
        inner_desc = inner_sdfg.arrays[inner]
        inner_layout = layout.with_shape([str(s) for s in inner_desc.shape])
        rewrite_accesses(inner_sdfg, inner, inner, inner_layout)
        inner_sdfg.arrays[inner] = laid_out_descriptor(inner_desc, inner_layout, inner_desc.transient)
    for state in sdfg.states():
        for node in state.data_nodes():
            if node.data == name:
                node.data = new_name
        for edge in state.edges():
            memlet = edge.data
            if memlet.is_empty():
                continue
            if memlet.data == name and mpi_buffer(edge) and not in_order(memlet.subset, layout):
                packed_transfer(sdfg, state, edge, new_name, layout)
            elif memlet.data == name:
                memlet.data = new_name
                memlet.subset = physical_subset(memlet.subset, layout)
                library_rule(state, edge, layout)
            elif memlet.other_subset is not None and new_name in (access_name(edge.src), access_name(edge.dst)):
                memlet.other_subset = physical_subset(memlet.other_subset, layout)
    for edge in sdfg.all_interstate_edges():
        assignments = edge.data.assignments
        edge.data.assignments = {
            k: rewrite_subscripts(v, name, new_name, layout.physical) for k, v in assignments.items()
        }
        condition = edge.data.condition.as_string
        if name in condition:
            edge.data.condition = CodeBlock(rewrite_subscripts(condition, name, new_name, layout.physical))


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


def library_rule(state: SDFGState, edge: MultiConnectorEdge[dace.Memlet], layout: IndexMap) -> None:
    """A library node at ``edge`` takes the layout by its rule, or the layout is refused."""
    for node, connector in ((edge.src, edge.src_conn), (edge.dst, edge.dst_conn)):
        if not isinstance(node, nodes.LibraryNode):
            continue
        perm = layout.permutation()
        rule = LIBRARY_RULES.get(type(node))
        if rule is None or perm is None:
            raise LayoutError(
                f"{state.label}: library node {node.label!r} ({type(node).__name__}) has no rule for this layout"
            )
        if perm != tuple(range(len(perm))):
            rule(state, node, connector, perm)


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
    sdfg: dace.SDFG, state: SDFGState, edge: MultiConnectorEdge[dace.Memlet], laid: str, layout: IndexMap
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
    position = layout.physical([f"{low} + {p}" for low, p in zip(first, params, strict=True)])
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


def conversion(state: SDFGState, logical: str, laid: str, layout: IndexMap, inward: bool) -> None:
    """A map over the logical index space moving ``logical`` into the laid-out ``laid`` (or back)."""
    params = [f"__r{d}" for d in range(len(layout.shape))]
    ranges = {p: f"0:{extent}" for p, extent in zip(params, layout.shape, strict=True)}
    position = ", ".join(layout.physical(params))
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
    return any(n.data == name and s.out_degree(n) > 0 for s in sdfg.states() for n in s.data_nodes())


def is_written(sdfg: dace.SDFG, name: str) -> bool:
    return any(n.data == name and s.in_degree(n) > 0 for s in sdfg.states() for n in s.data_nodes())


def relayout_array(sdfg: dace.SDFG, name: str, layout: IndexMap) -> str:
    """Lay ``sdfg``'s array ``name`` out by ``layout`` (see the module docstring); returns the name the
    program now works on (``name`` for a transient, ``laid_<name>`` for an argument)."""
    desc = sdfg.arrays[name]
    if not isinstance(desc, data.Array) or isinstance(desc, data.View):
        raise LayoutError(f"{name!r} is a {type(desc).__name__}; only arrays are laid out")
    if layout.is_identity():
        return name
    if desc.transient:
        rewrite_accesses(sdfg, name, name, layout)
        sdfg.arrays[name] = laid_out_descriptor(desc, layout, transient=True)
        propagation.propagate_memlets_sdfg(sdfg)
        return name
    laid = laid_name(name)
    read, written = is_read(sdfg, name), is_written(sdfg, name)
    sdfg.add_datadesc(laid, laid_out_descriptor(desc, layout, transient=True))
    rewrite_accesses(sdfg, name, laid, layout)
    if read:
        conversion(sdfg.add_state_before(sdfg.start_block, f"relayout_in_{name}"), name, laid, layout, inward=True)
    if written:
        for sink in [b for b in sdfg.nodes() if sdfg.out_degree(b) == 0]:
            conversion(sdfg.add_state_after(sink, f"relayout_out_{name}"), name, laid, layout, inward=False)
    propagation.propagate_memlets_sdfg(sdfg)
    return laid
