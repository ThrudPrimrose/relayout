# Copyright 2026 ETH Zurich and the relayout authors.
"""The loops of a Python function after a layout: re-tiled to the blocks their arrays are cut into, untiled where
a layout merged the blocks back, and reordered so the innermost loop walks the unit-stride axis; the counterpart
of :mod:`relayout.passes.sdfg.canonicalize_schedule`.

A layout rewrites accesses, not loops: a loop over ``j`` reading an array blocked by 4 reads
``A[j // 4, j % 4]``. :func:`canonicalize_schedule` tiles such a loop by 4 (``for j_o``, then ``for j`` in
``range(min(4, N - 4*j_o))``, ``j`` read as ``4*j_o + j``) so the access is ``A[j_o, j]``; a tiled nest whose
accesses read only ``4*j_o + j`` (an unblocked array) merges back into one loop; then every perfect loop nest is
ordered by stride, innermost the loop that indexes the last axis. Strip-mining keeps the order of the
iterations; a reordering needs a nest whose iterations are independent (:func:`independent`), and keeps each
loop inside the loops its bounds read.
"""

import ast
import copy
import math
from collections.abc import Sequence

from relayout.passes.py.common import kernel, names_in, parse
from relayout.passes.py.rewrite_accesses import scalar_indices

__all__ = ["canonicalize_schedule"]

#: A linear form: name -> integer coefficient, the constant under "".
type Linear = dict[str, int]


def linear(node: ast.AST) -> Linear | None:
    """``node`` as a linear form over names with integer coefficients, else None."""
    if isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool):
        return {"": node.value}
    if isinstance(node, ast.Name):
        return {node.id: 1}
    if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
        inner = linear(node.operand)
        return None if inner is None else {k: -v for k, v in inner.items()}
    if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub)):
        left, right = linear(node.left), linear(node.right)
        if left is None or right is None:
            return None
        sign = 1 if isinstance(node.op, ast.Add) else -1
        out = dict(left)
        for k, v in right.items():
            out[k] = out.get(k, 0) + sign * v
        return {k: v for k, v in out.items() if v != 0}
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Mult):
        left, right = linear(node.left), linear(node.right)
        if left is None or right is None:
            return None
        for scale, form in ((left, right), (right, left)):
            if set(scale) <= {""}:
                factor = scale.get("", 0)
                return {k: v * factor for k, v in form.items() if v * factor != 0}
    return None


def range_bounds(loop: ast.For) -> tuple[ast.expr, ast.expr] | None:
    """``(start, stop)`` of a unit-step ``for <name> in range(...)`` without ``else``, else None."""
    call = loop.iter
    if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "range"):
        return None
    if call.keywords or loop.orelse or not isinstance(loop.target, ast.Name):
        return None
    args = list(call.args)
    if len(args) == 1:
        return ast.Constant(0), args[0]
    if len(args) == 2 or (len(args) == 3 and ast.unparse(args[2]) == "1"):
        return args[0], args[1]
    return None


def width_of(node: ast.expr) -> int | None:
    return node.value if isinstance(node, ast.Constant) and isinstance(node.value, int) and node.value > 0 else None


def loops_with_enclosing(func: ast.FunctionDef) -> list[tuple[ast.For, set[str]]]:
    """Every ``for`` loop of ``func`` with the names its enclosing loops bind."""
    out: list[tuple[ast.For, set[str]]] = []

    def walk(body: Sequence[ast.stmt], enclosing: set[str]) -> None:
        for stmt in body:
            inner = set(enclosing)
            if isinstance(stmt, ast.For):
                out.append((stmt, enclosing))
                inner |= names_in(stmt.target)
            for _, value in ast.iter_fields(stmt):
                if isinstance(value, list) and value and all(isinstance(v, ast.stmt) for v in value):
                    walk(value, inner)

    walk(func.body, set())
    return out


class Blocks(ast.NodeTransformer):
    """``(width*outer + inner) // width`` read as ``outer`` and ``(width*outer + inner) % width`` as ``inner``,
    knowing ``0 <= inner < width``."""

    def __init__(self, outer: str, inner: str, width: int) -> None:
        self.tiled = {outer: width, inner: 1}
        self.outer, self.inner, self.width = outer, inner, width

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        self.generic_visit(node)
        if isinstance(node.op, (ast.FloorDiv, ast.Mod)) and width_of(node.right) == self.width:
            if linear(node.left) == self.tiled:
                return ast.Name(self.outer if isinstance(node.op, ast.FloorDiv) else self.inner, ast.Load())
        return node


def simplify_tiles(func: ast.FunctionDef) -> None:
    """Every access through a tile (``width*outer + inner``, the inner loop over ``range(width)`` or
    ``range(min(width, ...))``) read as its block index and offset."""
    for loop, enclosing in loops_with_enclosing(func):
        bounds = range_bounds(loop)
        if bounds is None or ast.unparse(bounds[0]) != "0":
            continue
        stop = bounds[1]
        width = width_of(stop)
        if width is None and isinstance(stop, ast.Call) and ast.unparse(stop.func) == "min" and stop.args:
            width = width_of(stop.args[0])
        if width is None:
            continue
        assert isinstance(loop.target, ast.Name)
        for outer in sorted(enclosing):
            loop.body = [Blocks(outer, loop.target.id, width).visit(s) for s in loop.body]


def block_widths(loop: ast.For, enclosing: set[str]) -> int | None:
    """The one constant ``b`` the body of ``loop`` (from 0 by 1) reads its variable through as ``p // b`` or
    ``p % b``, when ``p_o`` is free, else None."""
    bounds = range_bounds(loop)
    if bounds is None or ast.unparse(bounds[0]) != "0" or not isinstance(loop.target, ast.Name):
        return None
    p = loop.target.id
    widths = {
        width_of(node.right)
        for stmt in loop.body
        for node in ast.walk(stmt)
        if isinstance(node, ast.BinOp)
        and isinstance(node.op, (ast.FloorDiv, ast.Mod))
        and isinstance(node.left, ast.Name)
        and node.left.id == p
    }
    if len(widths) != 1 or None in widths or f"{p}_o" in enclosing:
        return None
    return widths.pop()


class Replace(ast.NodeTransformer):
    def __init__(self, name: str, value: str) -> None:
        self.name, self.value = name, value

    def visit_Name(self, node: ast.Name) -> ast.AST:
        return parse(self.value) if node.id == self.name and isinstance(node.ctx, ast.Load) else node


def tile_by_block(loop: ast.For, width: int) -> None:
    """Tile ``loop`` (``for p in range(hi)``) by ``width``: an outer loop over the block index ``p_o`` around an
    inner one over the position in the block, ``p`` read as ``width*p_o + p``, so ``p // width`` is ``p_o`` and
    ``p % width`` is ``p``."""
    bounds = range_bounds(loop)
    assert bounds is not None and isinstance(loop.target, ast.Name)
    p, hi = loop.target.id, ast.unparse(bounds[1])
    block = f"{p}_o"
    tile = ast.parse(f"for {p} in range(min({width}, {hi} - {width} * {block})):\n    pass").body[0]
    assert isinstance(tile, ast.For)
    tile.body = [Replace(p, f"({width} * {block} + {p})").visit(s) for s in loop.body]
    tile.body = [Blocks(block, p, width).visit(s) for s in tile.body]
    loop.target = ast.Name(block, ast.Store())
    loop.iter = parse(f"range(({hi} + {width} - 1) // {width})")
    loop.body = [tile]


def untile(func: ast.FunctionDef) -> int:
    """Merge a tile back into one loop: ``for p in range(width)`` directly inside ``for o in range(count)``,
    both read only as ``width*o + p`` (an unblocked array), become ``for p in range(width*count)``."""
    merged = 0
    for outer, _ in loops_with_enclosing(func):
        outer_bounds = range_bounds(outer)
        if outer_bounds is None or ast.unparse(outer_bounds[0]) != "0" or len(outer.body) != 1:
            continue
        inner = outer.body[0]
        if not isinstance(inner, ast.For):
            continue
        inner_bounds = range_bounds(inner)
        if inner_bounds is None or ast.unparse(inner_bounds[0]) != "0":
            continue
        width = width_of(inner_bounds[1])
        assert isinstance(outer.target, ast.Name) and isinstance(inner.target, ast.Name)
        o, p = outer.target.id, inner.target.id
        if width is None or o in names_in(inner_bounds[1]):
            continue

        flat = "__relayout_flat"

        class Flatten(ast.NodeTransformer):
            def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
                if linear(node) == {o: width, p: 1}:
                    return ast.Name(flat, ast.Load())
                return self.generic_visit(node)

        # the block index and offset of the tile read through ``//`` and ``%`` first: an access of a blocked
        # array keeps the tile variables, which is no unblocked access
        blocks = Blocks(o, p, width)
        simplified = [Flatten().visit(blocks.visit(copy.deepcopy(s))) for s in inner.body]
        body = [Replace(flat, p).visit(copy.deepcopy(s)) for s in simplified]
        left = set().union(*(names_in(s) for s in simplified))
        if {o, p} & left:  # a use of either variable outside the tiled sum is no unblocked access
            continue
        outer.target = ast.Name(p, ast.Store())
        outer.iter = parse(f"range({width} * ({ast.unparse(outer_bounds[1])}))")
        outer.body = body
        merged += 1
        break
    return merged


ALLOWED_CALLS = frozenset({"min", "max", "abs", "float", "int"})


def pure_call(node: ast.Call) -> bool:
    """A call the reordering may keep: a builtin of :data:`ALLOWED_CALLS` or a function of ``np`` / ``math``."""
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in ALLOWED_CALLS
    return isinstance(func, ast.Attribute) and isinstance(func.value, ast.Name) and func.value.id in ("np", "math")


def owns_iteration(dims: Sequence[ast.expr], params: Sequence[str]) -> bool:
    """Whether an index names every loop variable of ``params`` alone, with a unit coefficient, in some
    dimension: then each iteration of the nest owns the cell."""
    alone = set()
    for dim in dims:
        form = linear(dim)
        names = set() if form is None else set(form) - {""}
        if form is not None and len(names) == 1 and form[next(iter(names))] == 1:
            alone |= names
    return set(params) <= alone


def independent(func: ast.FunctionDef, nest: Sequence[ast.For], body: Sequence[ast.stmt]) -> bool:
    """Whether the iterations of ``nest`` are independent, so any order of its loops computes the same: the body
    is assignments only; every array it writes is accessed in it at one index, which names every loop variable
    alone (each iteration owns its cells); a scalar it assigns is assigned before it is read and used nowhere
    else; it calls only pure functions."""
    params = [loop.target.id for loop in nest if isinstance(loop.target, ast.Name)]
    if not all(isinstance(s, (ast.Assign, ast.AugAssign)) for s in body):
        return False
    if any(isinstance(n, ast.Call) and not pure_call(n) for s in body for n in ast.walk(s)):
        return False
    subscripts = [
        n for s in body for n in ast.walk(s) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)
    ]
    written = {n.value.id for n in subscripts if isinstance(n.ctx, ast.Store)}  # type: ignore[union-attr]
    for name in written:
        indices = {ast.unparse(n.slice) for n in subscripts if n.value.id == name}  # type: ignore[union-attr]
        if len(indices) != 1:
            return False
        index = parse(indices.pop())
        if not owns_iteration(index.elts if isinstance(index, ast.Tuple) else [index], params):
            return False
    targets = [t for s in body for t in (s.targets if isinstance(s, ast.Assign) else [s.target])]
    if not all(isinstance(t, (ast.Name, ast.Subscript)) for t in targets):
        return False
    scalars = {t.id for t in targets if isinstance(t, ast.Name)}
    if any(isinstance(s, ast.AugAssign) and isinstance(s.target, ast.Name) for s in body) or scalars & set(params):
        return False
    assigned: set[str] = set()
    for stmt in body:
        reads = {n.id for n in ast.walk(stmt) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
        if reads & scalars - assigned:
            return False
        assigned |= {t.id for t in (stmt.targets if isinstance(stmt, ast.Assign) else []) if isinstance(t, ast.Name)}
    nest_nodes = {id(n) for n in ast.walk(nest[0])}
    outside = {n.id for n in ast.walk(func) if isinstance(n, ast.Name) and id(n) not in nest_nodes}
    return not scalars & outside


def stride_order(nest: Sequence[ast.For], body: Sequence[ast.stmt]) -> list[int]:
    """The loops of ``nest``, outermost first, ordered by stride: innermost the loop that indexes the last axis of
    an array alone with a unit coefficient; ties by the offsets it is read with, then its position. A loop stays
    inside every loop its bounds read."""
    params = [loop.target.id for loop in nest if isinstance(loop.target, ast.Name)]
    score: dict[str, float] = {p: math.inf for p in params}
    offset: dict[str, int] = {p: 0 for p in params}
    for stmt in body:
        for node in ast.walk(stmt):
            if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
                continue
            indices = scalar_indices(node)
            if indices is None:
                continue
            for d, index in enumerate(indices):
                form = linear(index)
                if form is None:
                    continue
                names = set(form) - {""}
                if len(names) == 1 and (p := next(iter(names))) in score and form[p] == 1:
                    score[p] = min(score[p], len(indices) - 1 - d)
                    offset[p] += abs(form.get("", 0))
    reads = [names_in(loop.iter) & set(params) for loop in nest]
    order: list[int] = []
    remaining = list(range(len(nest)))
    while remaining:
        ready = [k for k in remaining if all(params.index(r) in order for r in reads[k])]
        best = max(ready, key=lambda k: (score[params[k]], offset[params[k]], -k))
        order.append(best)
        remaining.remove(best)
    return order


def perfect_nests(func: ast.FunctionDef) -> list[list[ast.For]]:
    """Every maximal perfect nest of unit-step ``range`` loops of ``func`` (two loops or more)."""
    inner_of = set()
    nests = []
    for loop, _ in loops_with_enclosing(func):
        if id(loop) in inner_of or range_bounds(loop) is None:
            continue
        nest = [loop]
        while len(nest[-1].body) == 1 and isinstance(nest[-1].body[0], ast.For) and range_bounds(nest[-1].body[0]):
            nest.append(nest[-1].body[0])
            inner_of.add(id(nest[-1]))
        if len(nest) > 1:
            nests.append(nest)
    return nests


def permute_by_stride(func: ast.FunctionDef) -> int:
    """Reorder every perfect nest whose iterations are independent by :func:`stride_order`; returns the number
    reordered."""
    permuted = 0
    for nest in perfect_nests(func):
        body = nest[-1].body
        if not independent(func, nest, body):
            continue
        order = stride_order(nest, body)
        if order == list(range(len(nest))):
            continue
        headers = [(loop.target, loop.iter) for loop in nest]
        for loop, k in zip(nest, order, strict=True):
            loop.target, loop.iter = headers[k]
        permuted += 1
    return permuted


def canonicalize_schedule(tree: ast.Module, function: str | None = None) -> int:
    """Tile every loop of ``tree``'s function to the blocks its accesses read, then order every perfect loop nest
    by stride; returns the number of loops tiled."""
    func = kernel(tree, function)
    while untile(func):
        pass
    simplify_tiles(func)
    tiled = 0
    while True:
        target = next(
            (
                (loop, width)
                for loop, enclosing in loops_with_enclosing(func)
                if (width := block_widths(loop, enclosing))
            ),
            None,
        )
        if target is None:
            break
        tile_by_block(*target)
        tiled += 1
    permute_by_stride(func)
    return tiled
