# Copyright 2026 ETH Zurich and the relayout authors.
"""What every SDFG layout pass shares: the refusal, the names of laid-out arrays and conversion maps, packed
descriptors, whether a program reads or (fully) writes an array, and memlet-path surgery around tasklets.
"""

import copy
from collections.abc import Callable, Sequence

import dace
import sympy
from dace import data, nodes, subsets, symbolic
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import SDFGState

from relayout.layout import IndexMap

__all__ = [
    "CONVERSION",
    "LAID",
    "LayoutError",
    "Symbolic",
    "always_runs",
    "array_desc",
    "as_range",
    "constant",
    "copy_through_tasklets",
    "covers",
    "expr",
    "fully_written",
    "is_read",
    "is_written",
    "laid_name",
    "laid_out_descriptor",
    "packed_strides",
    "product",
    "remove_path_edge",
]


#: The prefix of the transient an argument is laid out into.
LAID = "laid_"

#: The label prefix of every map converting between layouts (an argument into its layout and back, or one
#: layout into another), which no program's own map has (dace labels those after the program's module).
CONVERSION = "relayout_conversion_"


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


def covers_dimension(begin: object, end: object, step: object, extent: object) -> bool:
    return (
        symbolic.equal_valued(begin, 0)
        and symbolic.equal_valued(step, 1)
        and symbolic.equal_valued(expr(str(end)), expr(str(extent)) - 1)
    )


def covers(state: SDFGState, name: str) -> bool:
    """Whether ``state`` provably writes every element of ``name``: through one unconditional point write
    whose every dimension is whole or a distinct parameter of a surrounding map that runs over the whole
    dimension, or through one whole-array memlet. False whenever unprovable."""
    extents = array_desc(state.sdfg, name).shape
    for sink in [n for n in state.data_nodes() if n.data == name and state.in_degree(n) > 0]:
        for edge in state.in_edges(sink):
            for leaf in state.memlet_tree(edge).leaves():
                memlet = leaf.data
                if memlet.data != name or memlet.wcr is not None or memlet.dynamic:
                    continue
                ranges = as_range(memlet.subset).ranges
                if len(ranges) != len(extents):
                    continue
                params: dict[str, tuple[object, object, object]] = {}
                for scope in state.memlet_path(leaf):
                    if isinstance(scope.src, nodes.MapExit):
                        params.update(zip(scope.src.map.params, scope.src.map.range.ranges, strict=True))
                used: set[str] = set()
                for (begin, end, step), extent in zip(ranges, extents, strict=True):
                    whole = covers_dimension(begin, end, step, extent)
                    param = str(begin)
                    if not whole and (
                        not symbolic.equal_valued(begin, end)
                        or param not in params
                        or param in used
                        or not covers_dimension(*params[param], extent)
                    ):
                        break
                    if not whole:
                        used.add(param)
                else:
                    return True
    return False


def always_runs(sdfg: dace.SDFG, state: SDFGState) -> bool:
    """Whether every run of ``sdfg`` passes through ``state``: a top-level state of an unconditional line."""
    return state.parent_graph is sdfg and all(
        sdfg.out_degree(block) <= 1 and all(e.data.is_unconditional() for e in sdfg.out_edges(block))
        for block in sdfg.nodes()
    )


def fully_written(sdfg: dace.SDFG, name: str) -> bool:
    """Whether a run of ``sdfg`` provably writes every element of ``name`` (:func:`covers`)."""
    return any(always_runs(sdfg, state) and covers(state, name) for state in sdfg.states())


def array_desc(sdfg: dace.SDFG, name: str) -> data.Array:
    desc = sdfg.arrays[name]
    if not isinstance(desc, data.Array) or isinstance(desc, data.View):
        raise LayoutError(f"{name!r} is a {type(desc).__name__}; only arrays are laid out")
    return desc


def constant(value: object) -> int | None:
    """``value`` as an int when it is a constant, else None."""
    try:
        return int(str(value))
    except ValueError:
        return None


def copy_through_tasklets(state: SDFGState, name: str, needs: Callable[[dace.Memlet], bool]) -> None:
    """Each copy between ``name`` and another container (its memlet named after either end) whose memlet
    ``needs`` it goes through a tasklet ``__out = __inp``, which a rewrite of tasklet accesses
    (:func:`select_part`, :func:`member_access`) can then rewrite."""
    for edge in list(state.edges()):
        if edge not in state.edges() or edge.data.is_empty():
            continue
        if not isinstance(edge.src, nodes.AccessNode | nodes.EntryNode):
            continue
        if not isinstance(edge.dst, nodes.AccessNode | nodes.ExitNode):
            continue
        path = state.memlet_path(edge)
        source, target = path[0].src, path[-1].dst
        if not isinstance(source, nodes.AccessNode) or not isinstance(target, nodes.AccessNode):
            continue
        if name not in (source.data, target.data) or source.data == target.data:
            continue
        reads = source.data == name
        other = target if reads else source
        memlet = edge.data
        own_subset, other_subset = (
            (memlet.subset, memlet.other_subset) if memlet.data == name else (memlet.other_subset, memlet.subset)
        )
        own_subset = own_subset or subsets.Range.from_array(state.sdfg.arrays[name])
        other_subset = other_subset or subsets.Range.from_array(state.sdfg.arrays[other.data])
        own = dace.Memlet(data=name, subset=copy.deepcopy(own_subset), wcr=memlet.wcr)
        if not needs(own):
            continue
        theirs = dace.Memlet(data=other.data, subset=copy.deepcopy(other_subset))
        tasklet = state.add_tasklet(f"copy_{name}", {"__inp"}, {"__out"}, "__out = __inp")
        state.remove_edge(edge)
        inner, outer = (own, theirs) if reads else (theirs, own)
        state.add_edge(edge.src, edge.src_conn, tasklet, "__inp", inner)
        state.add_edge(tasklet, "__out", edge.dst, edge.dst_conn, outer)


def remove_path_edge(state: SDFGState, edge: MultiConnectorEdge[dace.Memlet]) -> None:
    """Remove ``edge`` of a replaced memlet path, and the scope connectors it leaves unused; a code node's
    connector stays, for the new path that now uses it."""
    state.remove_edge(edge)
    if isinstance(edge.src, nodes.EntryNode | nodes.ExitNode) and edge.src_conn is not None:
        if not any(e.src_conn == edge.src_conn for e in state.out_edges(edge.src)):
            edge.src.remove_out_connector(edge.src_conn)
    if isinstance(edge.dst, nodes.EntryNode | nodes.ExitNode) and edge.dst_conn is not None:
        if not any(e.dst_conn == edge.dst_conn for e in state.in_edges(edge.dst)):
            edge.dst.remove_in_connector(edge.dst_conn)
