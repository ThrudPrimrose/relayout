# Copyright 2026 ETH Zurich and the relayout authors.
"""Every access to an array moved to a target buffer through a :class:`~relayout.passes.sdfg.placement.Placement`:
memlets, interstate subscripts, views, copies, nested SDFGs and library nodes. :func:`move_array` adds the
conversion maps an argument needs, as it keeps its descriptor for the caller.
"""

import ast
import copy
from collections.abc import Callable, Iterator, Sequence

import dace
from dace import data, nodes, subsets, symbolic
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.properties import CodeBlock
from dace.sdfg.state import SDFGState

from relayout.layout import IndexMap
from relayout.passes.sdfg.common import (
    CONVERSION,
    LayoutError,
    as_range,
    fully_written,
    is_read,
    is_written,
)
from relayout.passes.sdfg.library_rules import library_rule
from relayout.passes.sdfg.pack_mpi_buffers import MpiBuffer, mpi_buffer, packed_transfer
from relayout.passes.sdfg.placement import (
    Laid,
    Placement,
)

__all__ = ["conversion", "move_array", "nested_uses", "rewrite_accesses", "rewrite_subscripts", "transfer"]


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


def nested_uses(
    sdfg: dace.SDFG, name: str, states: Sequence[SDFGState] | None = None
) -> Iterator[tuple[dace.SDFG, str]]:
    """Each nested SDFG ``name`` flows into (in ``states``, default all), with its inner connector, once (a
    read-write array's two connectors are one inner array)."""
    seen: set[tuple[int, str]] = set()
    for state in sdfg.states() if states is None else states:
        for node in state.nodes():
            if not isinstance(node, nodes.NestedSDFG):
                continue
            edges = [(e, e.dst_conn) for e in state.in_edges(node)] + [(e, e.src_conn) for e in state.out_edges(node)]
            for edge, connector in edges:
                if edge.data.data == name and connector is not None and (id(node.sdfg), connector) not in seen:
                    seen.add((id(node.sdfg), connector))
                    yield node.sdfg, connector


def rewrite_accesses(
    sdfg: dace.SDFG,
    name: str,
    target: str,
    placement: Placement,
    states: Sequence[SDFGState] | None = None,
    pending: list[MpiBuffer] | None = None,
) -> None:
    """Every access to ``name`` in ``sdfg`` moved to ``target`` (already described in ``sdfg``) through
    ``placement``; with ``states``, only the accesses in those states (interstate edges untouched). A
    nested SDFG ``name`` flows into receives the whole target: its array keeps its name, takes the target's
    descriptor and is rewritten the same way. With ``pending``, an MPI buffer whose cells are no contiguous run
    in logical order is rewritten like any access and recorded there for :func:`pack_mpi_buffers`; without
    it, it is packed at once."""
    target_desc = sdfg.arrays[target]
    if not isinstance(target_desc, data.Array):
        raise LayoutError(f"{target!r} is no array")
    for inner_sdfg, inner in list(nested_uses(sdfg, name, states)):
        inner_desc = inner_sdfg.arrays[inner]
        if len(inner_desc.shape) != len(sdfg.arrays[name].shape):
            raise LayoutError(
                f"nested SDFG {inner_sdfg.label!r} receives {name!r} as {inner!r} of rank {len(inner_desc.shape)}, "
                f"a slice of the rank-{len(sdfg.arrays[name].shape)} array; pass it whole (prepare widens nested inputs)"
            )
        inner_target = copy.deepcopy(target_desc)
        inner_target.transient = False
        inner_sdfg.arrays[inner] = inner_target
        nested_placement = placement.nested([str(s) for s in inner_desc.shape])
        rewrite_accesses_in(inner_sdfg, inner, inner, nested_placement, pending=pending)
    rewrite_accesses_in(sdfg, name, target, placement, states, pending)


def rewrite_accesses_in(
    sdfg: dace.SDFG,
    name: str,
    target: str,
    placement: Placement,
    states: Sequence[SDFGState] | None = None,
    pending: list[MpiBuffer] | None = None,
) -> None:
    """:func:`rewrite_accesses` within ``sdfg`` itself (its nested SDFGs done)."""
    target_desc = sdfg.arrays[target]
    for state in sdfg.states() if states is None else states:
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
            packs = memlet.data == name and mpi_buffer(edge) and not placement.in_order(memlet.subset, target_desc)
            if packs and pending is None:
                packed_transfer(sdfg, state, edge, target, placement, memlet.subset)
            elif memlet.data == name:
                if packs and pending is not None:
                    pending.append(MpiBuffer(sdfg, state, edge, target, placement, copy.deepcopy(memlet.subset)))
                view = next((n for n in (edge.src, edge.dst) if is_view(sdfg, n)), None)
                if view is not None:
                    restride_view(sdfg, view.data, memlet.subset, placement, target_desc)
                memlet.data = target
                nested = any(isinstance(n, nodes.NestedSDFG) for n in (edge.src, edge.dst))
                memlet.subset = subsets.Range.from_array(target_desc) if nested else placement.box(memlet.subset)
                library_rule(state, edge, placement)
            elif memlet.other_subset is not None and target in (access_name(edge.src), access_name(edge.dst)):
                memlet.other_subset = placement.box(memlet.other_subset)
    if states is not None:
        return
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


def transfer(
    state: SDFGState,
    label: str,
    source: str,
    source_at: Placement,
    target: str,
    target_at: Placement,
    shape: Sequence[str],
) -> None:
    """A map over the logical index space of ``shape`` copying every cell of ``source`` (placed by
    ``source_at``) to the same logical cell of ``target`` (placed by ``target_at``)."""
    params = [f"__r{d}" for d in range(len(shape))]
    ranges = {p: f"0:{extent}" for p, extent in zip(params, shape, strict=True)}
    state.add_mapped_tasklet(
        label,
        ranges,
        {"__inp": dace.Memlet(f"{source}[{', '.join(source_at.point(params))}]")},
        "__out = __inp",
        {"__out": dace.Memlet(f"{target}[{', '.join(target_at.point(params))}]")},
        external_edges=True,
    )


def conversion(
    state: SDFGState, logical: str, laid: str, shape: Sequence[str], placement: Placement, inward: bool
) -> None:
    """A map over the logical index space of ``shape`` moving ``logical`` into ``laid`` (or back)."""
    logical_at = Laid(IndexMap.identity(shape))
    if inward:
        transfer(state, f"{CONVERSION}in_{logical}", logical, logical_at, laid, placement, shape)
    else:
        transfer(state, f"{CONVERSION}out_{logical}", laid, placement, logical, logical_at, shape)


def move_array(
    sdfg: dace.SDFG, name: str, target: str, placement: Placement, pending: list[MpiBuffer] | None = None
) -> None:
    """Move ``name``'s accesses to ``target`` (described in ``sdfg``) through ``placement``. An argument
    keeps its descriptor for the caller and is converted into the target at the start (when read, or
    written only in part: the conversion back must not overwrite the cells the program leaves) and back
    at the end (when written); a transient is removed."""
    desc = sdfg.arrays[name]
    shape = [str(extent) for extent in desc.shape]
    read, written = is_read(sdfg, name), is_written(sdfg, name)
    keeps_cells = written and not fully_written(sdfg, name)
    rewrite_accesses(sdfg, name, target, placement, pending=pending)
    if desc.transient:
        if name != target:
            sdfg.remove_data(name, validate=False)
        return
    if read or keeps_cells:
        start = sdfg.add_state_before(sdfg.start_block, f"relayout_in_{name}")
        conversion(start, name, target, shape, placement, inward=True)
    if written:
        for sink in [b for b in sdfg.nodes() if sdfg.out_degree(b) == 0]:
            conversion(sdfg.add_state_after(sink, f"relayout_out_{name}"), name, target, shape, placement, inward=False)
