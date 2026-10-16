# Copyright 2026 ETH Zurich and the relayout authors.
"""The schedule after a layout: maps re-tiled to the blocks their arrays are cut into, untiled where a
layout merged the blocks back, and reordered so the innermost parameter walks the unit-stride axis.

A layout rewrites accesses, not loops: a map over ``j`` reading an array blocked by 4 reads
``A[j // 4, j % 4]``. :func:`canonicalize_schedule` tiles such a map by 4 (``j = 4*j_o + j``, the inner
``j`` in ``[0, 4)``) so the access is ``A[j_o, j]``; a tiled map whose accesses read only
``4*j_o + j`` (an unblocked array) merges its two parameters back into one; then dace's
``MinimizeStridePermutation`` orders every perfect map nest by stride. Padding moves no index, so it
needs nothing here.

Before the stride ordering, a map that accesses a laid-out array (``laid_out``) gets the parameter that walks that
array's new unit-stride (last) axis as its innermost parameter: the layout was chosen for that axis, so where
the stride ordering sees a tie (another array unit-stride along another parameter) the layout decides.
"""

import dace
import sympy
from dace import nodes, symbolic
from dace.sdfg.narrowing import as_expr
from dace.sdfg.state import SDFGState
from dace.transformation.dataflow.map_collapse import MapCollapse
from dace.transformation.dataflow.map_dim_shuffle import MapDimShuffle
from dace.transformation.dataflow.strip_mining import StripMining
from dace.transformation.passes.minimize_stride_permutation import MinimizeStridePermutation

__all__ = ["block_widths", "canonicalize_schedule", "laid_out_unit_stride_innermost", "tile_by_block"]


def leaf_edges(state: SDFGState, entry: nodes.MapEntry) -> list:
    """The point accesses inside ``entry``'s scope: memlets into or out of a node that is no scope node."""
    scope = state.scope_subgraph(entry)
    return [
        e
        for e in scope.edges()
        if not e.data.is_empty() and not isinstance(e.dst, nodes.EntryNode) and not isinstance(e.src, nodes.ExitNode)
    ]


def block_widths(state: SDFGState, entry: nodes.MapEntry) -> dict[str, int]:
    """Each parameter of ``entry`` that :func:`tile_by_block` can tile (from 0 by 1) and some access reads
    as ``p // b`` or ``p % b``, with one constant ``b`` across them (a plain use of ``p`` becomes ``b*p_o + p``
    when tiled and is no obstacle)."""
    params = {
        p: symbolic.symbol(p)
        for p, (begin, _, step) in zip(entry.map.params, entry.map.range, strict=True)
        if begin == 0 and step == 1
    }
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


def multiple_of(expression: sympy.Basic, width: int) -> bool:
    """Whether ``expression``, a polynomial in integer map parameters, is a multiple of ``width``."""
    expression = sympy.expand(expression)
    if expression == 0:
        return True
    symbols = sorted(expression.free_symbols, key=str)
    if not symbols:
        return expression.is_Integer and int(expression) % width == 0
    try:
        poly = sympy.Poly(expression, *symbols)
    except sympy.PolynomialError:
        return False
    return all(c.is_Integer and int(c) % width == 0 for c in poly.coeffs())


def simplify_blocks(expression: sympy.Basic, outer: sympy.Symbol, inner: sympy.Symbol, width: int) -> sympy.Basic:
    """``expression`` with ``(width*outer + inner) // width`` as ``outer`` and ``(width*outer + inner) %
    width`` (any multiple of ``width`` plus ``inner``, modulo ``width``) as ``inner``, knowing ``0 <= inner <
    width``. The parameters are integers, whether or not their symbols say so, and a symbol is matched by name
    (dace's symbols carry their declaration's assumptions, so one made here need not equal the memlet's)."""
    by_name = {str(symbol): symbol for symbol in expression.free_symbols}
    outer, inner = by_name.get(str(outer), outer), by_name.get(str(inner), inner)
    tiled = width * outer + inner
    replacements = {}
    for atom in expression.atoms(sympy.Function, sympy.Mod):
        if len(atom.args) != 2 or atom.args[1] != width:
            continue
        if isinstance(atom, sympy.Mod) and multiple_of(atom.args[0] - inner, width):
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
        raise ValueError(f"{entry.map.label}: {param!r} runs from {begin} by {step}; a tile starts at 0 by 1")
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
            flat = sympy.Symbol(inner_name, integer=True, nonnegative=True)
            rewritten, ok = [], True
            for edge in leaf_edges(state, inner_entry):
                ranges = []
                for r in edge.data.subset.ranges:
                    # Matched by name: the memlet's symbols carry their declaration's assumptions.
                    values = [sympy.sympify(v) for v in r]
                    by_name = {str(sym): sym for v in values for sym in v.free_symbols}
                    outer = by_name.get(outer_name, symbolic.symbol(outer_name))
                    inner = by_name.get(inner_name, symbolic.symbol(inner_name))
                    # A blocked access reads the tile's block and offset (``// width``, ``% width``) and keeps it.
                    blocks = [simplify_blocks(v, outer, inner, int(width)) for v in values]
                    ranges.append(tuple(v.subs(int(width) * outer + inner, flat) for v in blocks))
                names = {outer_name, inner_name}
                if any(sym is not flat and str(sym) in names for r in ranges for v in r for sym in v.free_symbols):
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


def unit_stride_params(state: SDFGState, entry: nodes.MapEntry, laid_out: set[str]) -> set[str]:
    """The parameters of ``entry`` that index the last (unit-stride) axis of a ``laid_out`` array with coefficient
    one, in the map's own accesses and in those of the nested SDFGs inside it."""
    params = set(entry.map.params)
    found: set[str] = set()

    def visit(begin: object, rename: dict[str, sympy.Basic]) -> None:
        # By name: a symbol carries its declaration's assumptions, so one made here need not equal the memlet's.
        expression = sympy.sympify(begin)
        expression = sympy.expand(
            expression.xreplace({sym: rename[str(sym)] for sym in expression.free_symbols if str(sym) in rename})
        )
        by_name = {str(sym): sym for sym in expression.free_symbols}
        for name in params & set(by_name):
            if expression.coeff(by_name[name]) == 1:
                found.add(name)

    def visit_nested(sdfg: dace.SDFG, arrays: set[str], rename: dict[str, sympy.Basic]) -> None:
        for edge, _ in sdfg.all_edges_recursive():
            memlet = getattr(edge, "data", None)
            if isinstance(memlet, dace.Memlet) and memlet.data in arrays and memlet.subset is not None:
                visit(memlet.subset.ranges[-1][0], rename)

    for edge in leaf_edges(state, entry):
        if edge.data.data in laid_out:
            visit(edge.data.subset.ranges[-1][0], {})
    for node in state.scope_subgraph(entry).nodes():
        if isinstance(node, nodes.NestedSDFG):
            inner = {e.dst_conn for e in state.in_edges(node) if e.data.data in laid_out}
            inner |= {e.src_conn for e in state.out_edges(node) if e.data.data in laid_out}
            rename = {k: sympy.sympify(v) for k, v in node.symbol_mapping.items()}
            visit_nested(node.sdfg, inner - {None}, rename)
    return found


def laid_out_unit_stride_innermost(sdfg: dace.SDFG, laid_out: set[str]) -> int:
    """Make the parameter that walks a laid-out array's unit-stride axis the innermost one of every map accessing
    it, when exactly one parameter of the map does; returns the number of maps reordered."""
    reordered = 0
    for state in sdfg.states():
        for entry in [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]:
            found = unit_stride_params(state, entry, laid_out)
            if len(found) != 1:
                continue
            (param,) = found
            if entry.map.params[-1] == param:
                continue
            order = [p for p in entry.map.params if p != param] + [param]
            MapDimShuffle.apply_to(sdfg, map_entry=entry, options={"parameters": order}, verify=False)
            reordered += 1
    return reordered


def canonicalize_schedule(sdfg: dace.SDFG, laid_out: set[str] | None = None) -> int:
    """Tile every map to the blocks its accesses read, put the unit-stride parameter of the ``laid_out`` arrays
    innermost, then order every perfect map nest by stride; returns the number of maps tiled."""
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
    if laid_out:
        laid_out_unit_stride_innermost(sdfg, laid_out)
    MinimizeStridePermutation().apply_pass(sdfg, {})
    return tiled
