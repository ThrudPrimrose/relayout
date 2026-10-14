# Copyright 2026 ETH Zurich and the relayout authors.
"""MPI buffers of laid-out arrays: :func:`pack_mpi_buffers` moves each buffer whose cells are no contiguous
run in logical order through a packed transient in logical order (pack before a send, unpack after a
receive), or keeps it in place where both ends provably share one layout.
"""

import copy
from collections.abc import Sequence
from dataclasses import dataclass

import dace
from dace import nodes, subsets, symbolic
from dace.libraries.mpi.nodes import Irecv, Wait, Waitall
from dace.libraries.mpi.nodes.node import MPINode
from dace.libraries.standard.nodes.copy import CopyLibraryNode
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState

from relayout.layout import IndexMap
from relayout.passes.sdfg.common import LayoutError, as_range, expr
from relayout.passes.sdfg.placement import Laid, Placement, physical_subset, trailing_axes

__all__ = ["MpiBuffer", "mpi_buffer", "pack_mpi_buffers", "packed_transfer"]


def mpi_buffer(edge: MultiConnectorEdge[dace.Memlet]) -> bool:
    """Whether ``edge`` is an MPI library node's buffer (``_buffer``, ``_inbuffer``, ``_out_buffer``, ...)."""
    for node, connector in ((edge.dst, edge.dst_conn), (edge.src, edge.src_conn)):
        if isinstance(node, MPINode):
            return connector is not None and connector.endswith("buffer")
    return False


@dataclass(frozen=True, slots=True)
class MpiBuffer:
    """An MPI node's buffer edge on a laid-out array whose cells are no contiguous run in logical order,
    rewritten to the laid-out buffer and waiting for :func:`pack_mpi_buffers`: the logical box it moves."""

    sdfg: dace.SDFG
    state: SDFGState
    edge: MultiConnectorEdge[dace.Memlet]
    laid: str
    placement: Placement
    logical: subsets.Subset

    def node(self) -> nodes.LibraryNode:
        return self.edge.dst if isinstance(self.edge.dst, MPINode) else self.edge.src


def run_subset(logical: subsets.Subset, layout: IndexMap) -> subsets.Range:
    """The laid-out buffer's range holding the contiguous run of the logical box ``logical``
    (:func:`trailing_axes`): its fixed axes at their positions, the spanned ones whole with their dead cells."""
    spanned = set(trailing_axes(logical, layout) or ())
    fixed = physical_subset(logical, layout).ranges
    return subsets.Range(
        [(0, expr(extent) - 1, 1) if k in spanned else fixed[k] for k, extent in enumerate(layout.extents())]
    )


def rows_outermost(layout: IndexMap) -> bool:
    """Whether the laid-out buffer keeps logical dimension 0 as its outermost, uncut axis: a collective that
    deals the buffer out in equal rank chunks then deals out whole logical rows."""
    first = layout.axes[0]
    renumbered = {name for renumber in layout.renumbers for name in renumber.axes}
    alone = sum(axis.dim == 0 for axis in layout.axes) == 1
    return first.dim == 0 and first.stride == 1 and not first.pad and first.name not in renumbered and alone


def pack_mpi_buffers(buffers: Sequence[MpiBuffer]) -> None:
    """Pack each recorded MPI buffer (:func:`packed_transfer`), except where the MPI node both sends and
    receives, every buffer of it is one array layout that keeps whole logical rows outermost
    (:func:`rows_outermost`), and each moves one contiguous run of its buffer (:func:`trailing_axes`): every
    rank runs the program, so the wire carries one layout's cells end to end, rank chunk by rank chunk. A
    send or a receive alone cannot see its partner's layout and is always packed."""
    by_node: dict[int, list[MpiBuffer]] = {}
    for buffer in buffers:
        by_node.setdefault(id(buffer.node()), []).append(buffer)
    for group in by_node.values():
        node, state = group[0].node(), group[0].state
        edges = [e for e in state.all_edges(node) if mpi_buffer(e)]
        layouts = {b.placement.layout if isinstance(b.placement, Laid) else None for b in group}
        both_ways = any(e.dst is node for e in edges) and any(e.src is node for e in edges)
        (layout,) = layouts if len(layouts) == 1 else (None,)
        shared = (
            both_ways
            and len(group) == len(edges)
            and layout is not None
            and rows_outermost(layout)
            and all(trailing_axes(b.logical, layout) is not None for b in group)
        )
        if shared:
            for b in group:  # the whole run of the buffer, dead cells included: both ends hold the same cells
                b.edge.data = dace.Memlet(data=b.laid, subset=run_subset(b.logical, layout))
            continue
        for b in group:
            packed_transfer(b.sdfg, b.state, b.edge, b.laid, b.placement, b.logical)


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
    sdfg: dace.SDFG,
    state: SDFGState,
    edge: MultiConnectorEdge[dace.Memlet],
    laid: str,
    placement: Placement,
    logical: subsets.Subset,
) -> None:
    """An MPI buffer of a laid-out array (the logical box ``logical``) moved through a contiguous transient in
    logical order: packed from the laid-out cells before a send, unpacked into them after a receive (at the
    wait of an ``Irecv``). The wire order is the logical one, as the program had it before the layout. Where
    the box's cells are a strided box of the buffer in logical order (:func:`copyable`) a copy node moves
    them, else a map over the box."""
    box = as_range(logical)
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
    by_copy = copyable(logical, placement)
    laid_memlet = dace.Memlet(data=laid, subset=placement.box(logical) if by_copy else ", ".join(position))
    packed_memlet = dace.Memlet(f"{packed}[{', '.join(params)}]")
    whole = dace.Memlet.from_array(packed, sdfg.arrays[packed])

    def move(target: SDFGState, label: str, inward: bool, laid_node: nodes.AccessNode, packed_node: nodes.AccessNode):
        """The cells from the laid-out buffer into the packed one (``inward``), or back."""
        source, destination = (laid_node, packed_node) if inward else (packed_node, laid_node)
        if by_copy:
            node = CopyLibraryNode(label)
            source_memlet, destination_memlet = (laid_memlet, whole) if inward else (whole, laid_memlet)
            target.add_edge(source, None, node, CopyLibraryNode.INPUT_CONNECTOR_NAME, copy.deepcopy(source_memlet))
            target.add_edge(
                node, CopyLibraryNode.OUTPUT_CONNECTOR_NAME, destination, None, copy.deepcopy(destination_memlet)
            )
            return
        read, write = (laid_memlet, packed_memlet) if inward else (packed_memlet, laid_memlet)
        target.add_mapped_tasklet(
            label,
            ranges,
            {"__inp": copy.deepcopy(read)},
            "__out = __inp",
            {"__out": copy.deepcopy(write)},
            input_nodes={source.data: source},
            output_nodes={destination.data: destination},
            external_edges=True,
        )

    sending = isinstance(edge.dst, nodes.LibraryNode)
    state.remove_edge(edge)
    if sending:
        source = edge.src
        if isinstance(source, nodes.AccessNode):
            source.data = laid
        packed_node = state.add_access(packed)
        move(state, f"pack_{packed}", True, source, packed_node)
        state.add_edge(packed_node, None, edge.dst, edge.dst_conn, whole)
        return
    packed_node = state.add_access(packed)
    state.add_edge(edge.src, edge.src_conn, packed_node, None, whole)
    if isinstance(edge.src, Irecv):
        if state.degree(edge.dst) == 0:
            state.remove_node(edge.dst)
        after = wait_state(sdfg, state, edge.src)
        target = after.parent_graph.add_state_after(after, f"unpack_{packed}")
        move(target, f"unpack_{packed}", False, target.add_access(laid), target.add_access(packed))
        return
    destination = edge.dst
    if isinstance(destination, nodes.AccessNode):
        destination.data = laid
    move(state, f"unpack_{packed}", False, destination, packed_node)


def copyable(logical: subsets.Subset, placement: Placement) -> bool:
    """Whether the laid-out cells of the logical box ``logical`` are a strided box of the buffer in logical
    order (a permuted or padded layout whose varying axes keep their order), so a copy node packs them."""
    if not isinstance(placement, Laid) or placement.layout.view() is None:
        return False
    ranges = as_range(logical).ranges
    dims = [axis.dim for axis in placement.layout.axes if not symbolic.equal_valued(*ranges[axis.dim][:2])]
    return dims == sorted(dims)
