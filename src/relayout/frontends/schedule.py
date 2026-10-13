# Copyright 2026 ETH Zurich and the relayout authors.
"""The schedule after a layout: maps re-tiled to the blocks their arrays are cut into, untiled where a
layout merged the blocks back, and reordered so the innermost parameter walks the unit-stride axis.

A layout rewrites accesses, not loops: a map over ``j`` reading an array blocked by 4 reads
``A[j // 4, j % 4]``. :func:`canonicalize_schedule` tiles such a map by 4 (``j = 4*j_o + j``, the inner
``j`` in ``[0, 4)``) so the access is ``A[j_o, j]``; a tiled map whose accesses read only
``4*j_o + j`` (an unblocked array) merges its two parameters back into one; then dace's
``MinimizeStridePermutation`` orders every perfect map nest by stride. Padding moves no index, so it
needs nothing here.
"""

import sympy

import dace
from dace import nodes, symbolic
from dace.sdfg.narrowing import as_expr
from dace.sdfg.state import SDFGState
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.strip_mining import StripMining
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation

__all__ = ["block_widths", "canonicalize_schedule", "tile_by_block"]


def leaf_edges(state: SDFGState, entry: nodes.MapEntry) -> list:
    """The point accesses inside ``entry``'s scope: memlets into or out of a node that is no scope node."""
    scope = state.scope_subgraph(entry)
    return [
        e
        for e in scope.edges()
        if not e.data.is_empty() and not isinstance(e.dst, nodes.EntryNode) and not isinstance(e.src, nodes.ExitNode)
    ]


def block_widths(state: SDFGState, entry: nodes.MapEntry) -> dict[str, int]:
    """Each parameter of ``entry`` some access reads as ``p // b`` or ``p % b``, with one constant ``b``
    across them (a plain use of ``p`` becomes ``b*p_o + p`` when tiled and is no obstacle)."""
    params = {p: symbolic.symbol(p) for p in entry.map.params}
    widths: dict[str, set[int]] = {p: set() for p in params}
    for edge in leaf_edges(state, entry):
        for begin, _, _ in edge.data.subset.ranges:
            for atom in sympy.sympify(begin).atoms(sympy.Function, sympy.Mod):
                if type(atom).__name__ in ("__int_floor", "int_floor", "Mod") and len(atom.args) == 2:
                    base, width = atom.args
                    for p, symbol in params.items():
                        if base == symbol and width.is_Integer:
                            widths[p].add(int(width))
    inside = enclosing_params(state, entry)
    return {p: next(iter(ws)) for p, ws in widths.items() if len(ws) == 1 and f"{p}_o" not in inside}


def enclosing_params(state: SDFGState, entry: nodes.MapEntry) -> set[str]:
    """The parameters of the maps around ``entry`` (a tiled parameter's block index ``p_o`` among them)."""
    scopes = state.scope_dict()
    out: set[str] = set()
    outer = scopes[entry]
    while isinstance(outer, nodes.MapEntry):
        out.update(outer.map.params)
        outer = scopes[outer]
    return out


def simplify_blocks(expression: sympy.Basic, outer: sympy.Symbol, inner: sympy.Symbol, width: int) -> sympy.Basic:
    """``expression`` with ``(width*outer + inner) // width`` as ``outer`` and ``(width*outer + inner) %
    width`` as ``inner``, knowing ``0 <= inner < width`` (sympy already reduces the latter to
    ``inner % width``)."""
    tiled = width * outer + inner
    replacements = {}
    for atom in expression.atoms(sympy.Function, sympy.Mod):
        if len(atom.args) != 2 or atom.args[1] != width:
            continue
        if isinstance(atom, sympy.Mod) and sympy.simplify(atom.args[0] - inner) == 0:
            replacements[atom] = inner
        elif type(atom).__name__ in ("__int_floor", "int_floor") and sympy.simplify(atom.args[0] - tiled) == 0:
            replacements[atom] = outer
    return expression.xreplace(replacements)


def tile_by_block(sdfg: dace.SDFG, state: SDFGState, entry: nodes.MapEntry, param: str, width: int) -> None:
    """Tile ``entry``'s ``param`` by ``width``: an outer map over ``param_o`` (the block index) around an
    inner one over ``param`` (the position in the block), so ``param // width`` is ``param_o`` and
    ``param % width`` is ``param``."""
    begin, _, step = entry.map.range[entry.map.params.index(param)]
    if begin != 0 or step != 1:
        return
    StripMining.apply_to(
        sdfg,
        options={"dim_idx": entry.map.params.index(param), "tile_size": width, "skew": True, "new_dim_prefix": "tile"},
        map_entry=entry,
    )
    tile = f"tile_{param}"
    outer_entry = next(n for n in state.nodes() if isinstance(n, nodes.MapEntry) and tile in n.map.params)
    k = outer_entry.map.params.index(tile)
    _, end, _ = outer_entry.map.range[k]
    block = f"{param}_o"
    outer_entry.map.params[k] = block
    ranges = list(outer_entry.map.range.ranges)
    ranges[k] = (0, symbolic.int_ceil(as_expr(end) + 1, width) - 1, 1)
    outer_entry.map.range = dace.subsets.Range(ranges)
    state.replace(tile, f"{width}*{block}")
    outer, inner = symbolic.symbol(block), symbolic.symbol(param)
    for edge in leaf_edges(state, outer_entry):
        edge.data.subset = dace.subsets.Range(
            [tuple(simplify_blocks(sympy.sympify(v), outer, inner, width) for v in r) for r in edge.data.subset.ranges]
        )


def tile_pairs(state: SDFGState, entry: nodes.MapEntry) -> list[tuple[sympy.Symbol, sympy.Symbol, int]]:
    """``(outer, inner, width)`` for each parameter of ``entry`` that iterates ``0:width`` (a constant
    block) and each parameter of an enclosing map it may tile."""
    out = []
    for param, (begin, end, step) in zip(entry.map.params, entry.map.range, strict=True):
        width = sympy.sympify(as_expr(end)) + 1
        if begin == 0 and step == 1 and width.is_Integer:
            inner = symbolic.symbol(param)
            out += [(symbolic.symbol(o), inner, int(width)) for o in enclosing_params(state, entry)]
    return out


def simplify_tiles(sdfg: dace.SDFG) -> None:
    """Every point access through a tile (``width*outer + inner``, the inner in ``0:width``) read as its
    block index and offset: ``(width*outer + inner) // width`` is ``outer``, ``% width`` is ``inner``."""
    for state in sdfg.states():
        for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
            for outer, inner, width in tile_pairs(state, entry):
                for edge in leaf_edges(state, entry):
                    edge.data.subset = dace.subsets.Range(
                        [
                            tuple(simplify_blocks(sympy.sympify(v), outer, inner, width) for v in r)
                            for r in edge.data.subset.ranges
                        ]
                    )


def untile(sdfg: dace.SDFG) -> int:
    """Merge a tile back into one parameter: a map over ``inner`` in ``0:width`` directly inside a map
    over ``outer`` in ``0:count``, both used only as ``width*outer + inner`` (an unblocked array), become
    one map over ``0:width*count``. Returns the number merged."""
    merged = 0
    for state in sdfg.states():
        scopes = state.scope_dict()
        for inner_entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
            outer_entry = scopes[inner_entry]
            if (
                not isinstance(outer_entry, nodes.MapEntry)
                or len(inner_entry.map.params) != 1
                or len(outer_entry.map.params) != 1
            ):
                continue
            (outer_name,), (inner_name,) = outer_entry.map.params, inner_entry.map.params
            (o_begin, o_end, o_step), (i_begin, i_end, i_step) = outer_entry.map.range[0], inner_entry.map.range[0]
            width = sympy.sympify(as_expr(i_end)) + 1
            if not (o_begin == 0 and i_begin == 0 and o_step == 1 and i_step == 1 and width.is_Integer):
                continue
            outer, inner = symbolic.symbol(outer_name), symbolic.symbol(inner_name)
            flat = sympy.Symbol(inner_name, integer=True, nonnegative=True)
            rewritten, ok = [], True
            for edge in leaf_edges(state, inner_entry):
                ranges = [
                    tuple(sympy.sympify(v).subs(int(width) * outer + inner, flat) for v in r)
                    for r in edge.data.subset.ranges
                ]
                if any(sym in (outer, inner) for r in ranges for v in r for sym in sympy.sympify(v).free_symbols):
                    ok = False
                    break
                rewritten.append((edge, ranges))
            uses = [n for n in state.scope_subgraph(inner_entry).nodes() if isinstance(n, nodes.CodeNode)]
            if not ok or any({outer_name, inner_name} & n.free_symbols for n in uses):
                continue
            for edge, ranges in rewritten:
                edge.data.subset = dace.subsets.Range(ranges)
            count = sympy.sympify(as_expr(o_end)) + 1
            MapCollapse.apply_to(state.sdfg, outer_map_entry=outer_entry, inner_map_entry=inner_entry)
            collapsed = next(
                n.map
                for n in state.nodes()
                if isinstance(n, nodes.MapEntry) and n.map.params == [outer_name, inner_name]
            )
            collapsed.params = [inner_name]
            collapsed.range = dace.subsets.Range([(0, int(width) * count - 1, 1)])
            merged += 1
            break
    return merged


def canonicalize_schedule(sdfg: dace.SDFG) -> int:
    """Tile every map to the blocks its accesses read, then order every perfect map nest by stride;
    returns the number of maps tiled."""
    while untile(sdfg):
        pass
    simplify_tiles(sdfg)
    tiled = 0
    while True:
        target = None
        for state in sdfg.states():
            for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
                widths = block_widths(state, entry)
                if widths:
                    param, width = next(iter(widths.items()))
                    target = (state, entry, param, width)
                    break
            if target:
                break
        if target is None:
            break
        state, entry, param, width = target
        tile_by_block(state.sdfg, state, entry, param, width)
        tiled += 1
    MinimizeStridePermutation().apply_pass(sdfg, {})
    return tiled
