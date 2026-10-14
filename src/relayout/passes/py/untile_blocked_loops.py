# Copyright 2026 ETH Zurich and the relayout authors.
"""Collapse a hand-tiled loop nest of a Python function back to one unit-stride loop, unblocking an array blocked
to match the tile; the counterpart of :mod:`relayout.passes.sdfg.untile_blocked_loops` (dace's ``UntileLoops``
with the blocked-array case). The outer loop is ``for i in range(lo, hi, K)`` holding exactly one inner loop:

- case A, ``for ii in range(K)``: the body indexes only through ``i + ii`` (or, for tiles starting at multiples
  of ``K``, ``A[..., i // K, ii]`` of an array whose last extent is ``K``, which is unblocked first); the nest
  becomes ``for k in range(lo, end)`` with ``i + ii`` read as ``k``;
- case B, ``for ii in range(i, i + K)``: the body indexes only through ``ii``; the nest becomes
  ``for ii in range(lo, end)``.

``end`` is the end of the last tile, ``hi`` itself for a clamped nest (``range(min(K, hi - i))``,
``range(i, min(i + K, hi))``), which never unblocks an array. Anything else is left as it is.
"""

import ast
import copy
from collections.abc import Mapping

from relayout.passes.py.common import Array, arguments, kernel, names_in, parse
from relayout.passes.py.unblock_dimensions import unblock_dimensions, unblock_is_safe

__all__ = ["untile_blocked_loops"]


def range_args(node: ast.expr) -> list[ast.expr] | None:
    """The arguments of ``range(...)`` with keywords absent, else None."""
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "range" and not node.keywords:
        return list(node.args)
    return None


def same(a: ast.AST, b: ast.AST) -> bool:
    return ast.unparse(a) == ast.unparse(b)


def is_name(node: ast.AST, name: str) -> bool:
    return isinstance(node, ast.Name) and node.id == name


def is_sum(node: ast.AST, a: str, b: str) -> bool:
    """``node`` is ``a + b`` or ``b + a``."""
    return (
        isinstance(node, ast.BinOp)
        and isinstance(node.op, ast.Add)
        and {ast.unparse(node.left), ast.unparse(node.right)} == {a, b}
        and all(isinstance(side, ast.Name) for side in (node.left, node.right))
    )


def is_min(node: ast.AST, first: str, second: str) -> bool:
    """``node`` is ``min(first, second)`` in either order."""
    return (
        isinstance(node, ast.Call)
        and is_name(node.func, "min")
        and len(node.args) == 2
        and not node.keywords
        and {ast.unparse(node.args[0]), ast.unparse(node.args[1])} == {first, second}
    )


def is_tile_origin(node: ast.AST, i: str, ii: str, K: str) -> bool:
    """``node`` is ``(i // K) * K + ii`` in either order of the sum: what an unblocked ``A[i // K, ii]`` reads."""
    if not (isinstance(node, ast.BinOp) and isinstance(node.op, ast.Add)):
        return False
    for origin, offset in ((node.left, node.right), (node.right, node.left)):
        if (
            is_name(offset, ii)
            and isinstance(origin, ast.BinOp)
            and isinstance(origin.op, ast.Mult)
            and ast.unparse(origin.left) == f"{i} // {K}"
            and ast.unparse(origin.right) == K
        ):
            return True
    return False


class Substitute(ast.NodeTransformer):
    """``i + ii`` (and, ``aligned``, ``(i // K) * K + ii``) read as ``k``."""

    def __init__(self, i: str, ii: str, K: str, k: str, aligned: bool) -> None:
        self.i, self.ii, self.K, self.k, self.aligned = i, ii, K, k, aligned

    def visit_BinOp(self, node: ast.BinOp) -> ast.AST:
        if is_sum(node, self.i, self.ii) or (self.aligned and is_tile_origin(node, self.i, self.ii, self.K)):
            return ast.Name(self.k, ast.Load())
        return self.generic_visit(node)


def blocked_reads(body: list[ast.stmt], i: str, ii: str, K: str) -> dict[str, int]:
    """Each array the body reads as ``A[..., i // K, ii]``, with its rank."""
    out: dict[str, int] = {}
    for stmt in body:
        for node in ast.walk(stmt):
            if not (isinstance(node, ast.Subscript) and isinstance(node.value, ast.Name)):
                continue
            elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
            if len(elements) >= 2 and ast.unparse(elements[-2]) == f"{i} // {K}" and is_name(elements[-1], ii):
                out[node.value.id] = len(elements)
    return out


def tile_end(lo: str, hi: str, K: str) -> str:
    """The end of the last tile of ``range(lo, hi, K)``."""
    if lo == "0":
        return f"({hi} + {K} - 1) // {K} * {K}"
    return f"{lo} + ({hi} - {lo} + {K} - 1) // {K} * {K}"


def fresh(func: ast.FunctionDef) -> str:
    taken = names_in(func)
    k = 0
    while f"_untile_k_{k}" in taken:
        k += 1
    return f"_untile_k_{k}"


def untile_one(func: ast.FunctionDef, outer: ast.For, arrays: Mapping[str, Array]) -> ast.FunctionDef | None:
    """``func`` with ``outer`` (a tile loop) collapsed, as a new function, or None when the nest does not match."""
    bounds = range_args(outer.iter)
    if not (bounds and len(bounds) == 3 and isinstance(outer.target, ast.Name) and not outer.orelse):
        return None
    lo, hi, K = (ast.unparse(b) for b in bounds)
    i = outer.target.id
    if not (len(outer.body) == 1 and isinstance(outer.body[0], ast.For)):
        return None
    inner = outer.body[0]
    inner_bounds = range_args(inner.iter)
    if not (inner_bounds and isinstance(inner.target, ast.Name) and not inner.orelse):
        return None
    ii = inner.target.id
    if len(inner_bounds) == 1:
        inner_bounds = [ast.Constant(0), inner_bounds[0]]
    if len(inner_bounds) != 2:
        return None
    start, stop = inner_bounds
    stores = {n.id for s in inner.body for n in ast.walk(s) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store)}
    if {i, ii} & stores:
        return None
    if ast.unparse(start) == "0" and (same(stop, parse(K)) or is_min(stop, K, f"{hi} - {i}")):
        case, clamped = "A", not same(stop, parse(K))
    elif is_name(start, i) and (is_sum(stop, i, K) or is_min(stop, f"{i} + {K}", hi)):
        case, clamped = "B", not is_sum(stop, i, K)
    else:
        return None
    end = hi if clamped else tile_end(lo, hi, K)
    out = copy.deepcopy(func)
    position = next(k for k, n in enumerate(ast.walk(func)) if n is outer)
    nest = next(n for k, n in enumerate(ast.walk(out)) if k == position)
    assert isinstance(nest, ast.For)
    nest_inner = nest.body[0]
    assert isinstance(nest_inner, ast.For)
    if case == "B":
        if i in {n for s in nest_inner.body for n in names_in(s)}:
            return None
        nest.target = ast.Name(ii, ast.Store())
        nest.iter = parse(f"range({lo}, {end})")
        nest.body = nest_inner.body
        return out
    aligned = lo == "0" or (lo.isdigit() and K.isdigit() and int(lo) % int(K) == 0)
    blocked = {} if clamped or not aligned or not K.isdigit() else blocked_reads(nest_inner.body, i, ii, K)
    for name, rank in blocked.items():
        last = arrays[name].shape[-1] if name in arrays else None
        if name in arguments(func) and last != K:
            return None  # an argument's last extent must be the tile, known from its shape
    if blocked:
        spec = {name: ([False] * (rank - 2) + [True], [1] * (rank - 2) + [int(K)]) for name, rank in blocked.items()}
        if not unblock_is_safe(out, spec):
            return None
        holder = ast.Module([out], [])
        unblock_dimensions(holder, spec)
    k = fresh(out)
    body = [Substitute(i, ii, K, k, aligned).visit(s) for s in nest_inner.body]
    if {i, ii} & {n for s in body for n in names_in(s)}:
        return None
    nest.target = ast.Name(k, ast.Store())
    nest.iter = parse(f"range({lo}, {end})")
    nest.body = body
    return out


def untile_blocked_loops(
    tree: ast.Module, arrays: Mapping[str, Array] | None = None, function: str | None = None
) -> int:
    """Collapse every hand-tiled nest of ``tree``'s function (see the module docstring), unblocking the arrays a
    nest reads blocked (an argument's shape from ``arrays``); returns the number collapsed."""
    func = kernel(tree, function)
    known = arrays or {}
    collapsed = 0
    while True:
        for loop in [n for n in ast.walk(func) if isinstance(n, ast.For)]:
            new = untile_one(func, loop, known)
            if new is not None:
                func.body = new.body
                collapsed += 1
                break
        else:
            return collapsed
