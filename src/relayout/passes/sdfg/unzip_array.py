# Copyright 2026 ETH Zurich and the relayout authors.
"""``unzip``: an array split along a dimension of constant extent into one array per index; the maps and
loops over that dimension are unrolled, and a data-dependent index selects its part in a tasklet.
"""

import ast
import copy
from collections.abc import Callable, Sequence

import dace
from dace import nodes, subsets, symbolic
from dace.properties import CodeBlock
from dace.sdfg import propagation
from dace.sdfg.graph import MultiConnectorEdge
from dace.sdfg.state import LoopRegion, SDFGState
from dace.transformation.dataflow.map_unroll import MapUnroll
from dace.transformation.interstate.loop_unroll import LoopUnroll
from dace.transformation.passes.analysis import loop_analysis

from relayout.passes.sdfg.common import (
    LayoutError,
    array_desc,
    as_range,
    constant,
    copy_through_tasklets,
    fully_written,
    is_read,
    is_written,
    packed_strides,
    product,
    remove_path_edge,
)
from relayout.passes.sdfg.placement import (
    Field,
)
from relayout.passes.sdfg.rewrite_accesses import (
    conversion,
    nested_uses,
)

__all__ = ["unzip_array"]


def split_indices(sdfg: dace.SDFG, name: str, dim: int) -> list[object]:
    """The ranges point accesses of ``name`` (a memlet inside every scope; the scopes' propagated ranges
    follow from them) take on dimension ``dim``: a range spans several parts, which no access can."""
    found = []
    for state in sdfg.states():
        for edge in state.edges():
            leaf = not isinstance(edge.dst, nodes.EntryNode) and not isinstance(edge.src, nodes.ExitNode)
            if edge.data.data == name and not edge.data.is_empty() and leaf:
                begin, end, _ = as_range(edge.data.subset).ranges[dim]
                if not symbolic.equal_valued(begin, end):
                    found.append((begin, end))
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
    unrolled so every index on it is a constant, and a Python tasklet indexing it by a data-dependent value
    selects among the parts (:func:`select_part`). An argument keeps its descriptor, converted into the
    parts and back. A range along the dimension (or a nested SDFG receiving the array) is refused."""
    desc = array_desc(sdfg, name)
    if isinstance(desc.dtype, dace.struct):
        raise LayoutError(f"unzip of the struct array {name!r} into its members is not rewritten on an SDFG yet")
    if constant(desc.shape[dim]) != len(into):
        raise LayoutError(
            f"{name!r} has extent {desc.shape[dim]} on dimension {dim}; name one array per index ({len(into)} given)"
        )
    if any(True for _ in nested_uses(sdfg, name)):
        raise LayoutError(f"{name!r} flows into a nested SDFG; inline it before splitting the array")
    unroll_over(sdfg, len(into))
    ranged = split_indices(sdfg, name, dim)
    if ranged:
        raise LayoutError(f"{name!r} is indexed on dimension {dim} by the range {ranged[0]}; split a point access")
    rest = [extent for k, extent in enumerate(desc.shape) if k != dim] or [1]
    for part in into:
        piece = copy.deepcopy(desc)
        piece.transient = True
        piece.set_shape(rest, strides=packed_strides(rest), total_size=product(rest))
        sdfg.add_datadesc(part, piece, find_new_name=False)

    for state in sdfg.states():
        copy_through_tasklets(state, name, lambda memlet: constant(as_range(memlet.subset).ranges[dim][0]) is None)
        for node in [n for n in state.data_nodes() if n.data == name]:
            split_access(state, node, into, dim)
    for edge in sdfg.all_interstate_edges():
        assignments = edge.data.assignments
        edge.data.assignments = {k: split_subscripts(v, name, into, dim) for k, v in assignments.items()}
    if not desc.transient:
        shape = [str(extent) for k, extent in enumerate(desc.shape) if k != dim]
        written = any(is_written(sdfg, part) for part in into)
        if is_read_in_parts(sdfg, into) or (written and not all(fully_written(sdfg, part) for part in into)):
            start = sdfg.add_state_before(sdfg.start_block, f"unzip_in_{name}")
            for c, part in enumerate(into):
                conversion(start, part, name, shape, Field(c, dim), inward=False)
        sinks = [b for b in sdfg.nodes() if sdfg.out_degree(b) == 0]
        if written:
            end = sdfg.add_state_after(sinks[0], f"unzip_out_{name}")
            for c, part in enumerate(into):
                conversion(end, part, name, shape, Field(c, dim), inward=True)
    else:
        sdfg.remove_data(name, validate=False)
    propagation.propagate_memlets_sdfg(sdfg)
    return list(into)


def split_access(state: SDFGState, node: nodes.AccessNode, into: Sequence[str], dim: int) -> None:
    """Each memlet path through access node ``node`` rebuilt from a new access node of the part its leaf
    (the point access inside the scopes) names on ``dim``, through the same scopes; a leaf whose index there
    is data-dependent goes to every part (:func:`select_part`). ``node`` is then removed."""
    accesses: dict[str, nodes.AccessNode] = {}

    def rebuild(leaf: MultiConnectorEdge[dace.Memlet], part: str, memlet: dace.Memlet, conn: str | None) -> None:
        """The path of ``leaf`` from (or to) a new access node of ``part`` carrying ``memlet``, at connector
        ``conn`` of its code node (None: the connector it had)."""
        path = state.memlet_path(leaf)
        access = accesses.setdefault(part, state.add_access(part))
        if path[0].src is node:
            scopes = [e.dst for e in path[:-1]]
            state.add_memlet_path(access, *scopes, leaf.dst, dst_conn=conn or leaf.dst_conn, memlet=memlet)
        else:
            scopes = [e.src for e in path[1:]]
            state.add_memlet_path(leaf.src, *scopes, access, src_conn=conn or leaf.src_conn, memlet=memlet)

    for edge in list(state.out_edges(node)) + list(state.in_edges(node)):
        if edge not in state.edges():
            continue
        tree = state.memlet_tree(edge)
        for leaf in list(tree.leaves()):
            named = leaf.data.data == node.data
            ranges = list(as_range(leaf.data.subset if named else leaf.data.other_subset).ranges)
            index = ranges.pop(dim)[0]
            rest = subsets.Range(ranges or [(0, 0, 1)])
            position = constant(index)
            if position is None:
                select_part(leaf, into, rest, index, rebuild)
                continue
            if named:
                memlet = dace.Memlet(data=into[position], subset=rest, wcr=leaf.data.wcr, dynamic=leaf.data.dynamic)
            else:
                leaf.data.other_subset = rest
                memlet = leaf.data
            rebuild(leaf, into[position], memlet, None)
        for old in list(tree):
            remove_path_edge(state, old)
    state.remove_node(node)


def select_part(
    leaf: MultiConnectorEdge[dace.Memlet],
    into: Sequence[str],
    rest: subsets.Range,
    index: object,
    rebuild: Callable[[MultiConnectorEdge[dace.Memlet], str, dace.Memlet, str | None], None],
) -> None:
    """A point access of a Python tasklet through the data-dependent ``index``: a read takes every part on a
    connector of its own and picks the one ``index`` names, a write stores into that one through dynamic
    memlets to every part."""
    reads = isinstance(leaf.dst, nodes.Tasklet)
    tasklet = leaf.dst if reads else leaf.src
    conn = leaf.dst_conn if reads else leaf.src_conn
    if not isinstance(tasklet, nodes.Tasklet) or tasklet.language is not dace.Language.Python or conn is None:
        raise LayoutError(f"{leaf.data} is indexed by {index}, not a constant, outside a Python tasklet")
    which = symbolic.symstr(index)
    parts = [f"{conn}__{k}" for k in range(len(into))]
    code = tasklet.code.as_string
    if reads:
        tasklet.remove_in_connector(conn)
        choice = " else ".join(f"{p} if ({which}) == {k}" for k, p in enumerate(parts[:-1])) + f" else {parts[-1]}"
        tasklet.code = CodeBlock(f"{conn} = {choice}\n{code}")
    else:
        tasklet.remove_out_connector(conn)
        stores = "\n".join(
            f"{'if' if k == 0 else 'elif'} ({which}) == {k}:\n    {p} = {conn}" for k, p in enumerate(parts)
        )
        tasklet.code = CodeBlock(f"{code}\n{stores}")
    for k, part in enumerate(into):
        if reads:
            tasklet.add_in_connector(parts[k])
        else:
            tasklet.add_out_connector(parts[k])
        memlet = dace.Memlet(
            data=part, subset=copy.deepcopy(rest), wcr=leaf.data.wcr, dynamic=not reads or leaf.data.dynamic
        )
        rebuild(leaf, part, memlet, parts[k])


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
