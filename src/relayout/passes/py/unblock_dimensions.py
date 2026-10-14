# Copyright 2026 ETH Zurich and the relayout authors.
"""Unblock arrays of a Python function: each blocked (outer, inner) dimension pair merged back into one,
``outer*factor + inner``, the inverse of blocking; the counterpart of :mod:`relayout.passes.sdfg.unblock_dimensions`.

An array is described as the Block transform made it: ``masks`` and ``factors`` over its original (unblocked)
rank, the inner dimension of each masked one appended after the outer ones. Every use must be an element access
of the blocked rank (:func:`unblock_is_safe`); a local's allocation takes the unblocked shape, an argument is
passed unblocked by the caller.
"""

import ast
from collections.abc import Mapping, Sequence

from relayout.passes.py.common import LayoutError, allocation, arguments, definition, kernel, parse, where
from relayout.passes.py.rewrite_accesses import scalar_indices

__all__ = ["BlockedArrays", "unblock_dimensions", "unblock_is_safe"]

#: ``{array: (masks, factors)}``: the same mapping the Block transform used.
type BlockedArrays = Mapping[str, tuple[Sequence[bool], Sequence[int]]]


def inner_slot(masks: Sequence[bool], d: int) -> int:
    """The appended inner dimension of masked dimension ``d``."""
    return len(masks) + sum(1 for m in masks[:d] if m)


def blocked_rank(masks: Sequence[bool]) -> int:
    return len(masks) + sum(1 for m in masks if m)


def recover_extent(blocked: ast.expr, factor: int) -> str:
    """The original extent of a blocked dimension: ``E`` of ``E // factor`` or ``(E + factor - 1) // factor``,
    else ``blocked * factor``."""
    if isinstance(blocked, ast.BinOp) and isinstance(blocked.op, ast.FloorDiv):
        right = blocked.right
        if isinstance(right, ast.Constant) and right.value == factor:
            left = blocked.left
            ceiling = (
                isinstance(left, ast.BinOp)
                and isinstance(left.op, ast.Add)
                and isinstance(left.right, ast.Constant)
                and left.right.value == factor - 1
            )
            return ast.unparse(left.left if ceiling and isinstance(left, ast.BinOp) else left)
    return f"({ast.unparse(blocked)}) * {factor}"


def uses(func: ast.FunctionDef, name: str) -> list[ast.AST]:
    """Every use of ``name`` in ``func``, a subscript for an element access, else the name itself."""
    subscripts = [n for n in ast.walk(func) if isinstance(n, ast.Subscript) and isinstance(n.value, ast.Name)]
    accessed = {id(n.value) for n in subscripts if n.value.id == name}  # type: ignore[union-attr]
    out: list[ast.AST] = [n for n in subscripts if n.value.id == name]  # type: ignore[union-attr]
    out += [n for n in ast.walk(func) if isinstance(n, ast.Name) and n.id == name and id(n) not in accessed]
    return out


def unblock_is_safe(func: ast.FunctionDef, blocked: BlockedArrays) -> bool:
    """Whether every use of every array in ``blocked`` is one :func:`unblock_dimensions` rewrites: an element
    access of the blocked rank, or the definition of a local by an allocation of the blocked rank."""
    for name, (masks, _) in blocked.items():
        expected = blocked_rank(masks)
        local = name not in arguments(func)
        definition_target = None
        if local:
            try:
                body, index = definition(func, name)
                stmt = body[index]
            except LayoutError:
                return False
            alloc = allocation(stmt.value) if isinstance(stmt, ast.Assign) else None
            shape = stmt.value.args[0] if alloc is not None and isinstance(stmt.value, ast.Call) else None
            if not (isinstance(shape, ast.Tuple) and len(shape.elts) == expected):
                return False
            definition_target = stmt.targets[0] if isinstance(stmt, ast.Assign) else None
        for use in uses(func, name):
            if use is definition_target:
                continue
            if not isinstance(use, ast.Subscript):
                return False
            indices = scalar_indices(use)
            if indices is None or len(indices) != expected:
                return False
    return True


def unblock_dimensions(tree: ast.Module, blocked: BlockedArrays, function: str | None = None) -> int:
    """Unblock every array of ``blocked`` in ``tree``'s function (see the module docstring); returns the number
    of accesses rewritten."""
    func = kernel(tree, function)
    if not unblock_is_safe(func, blocked):
        raise LayoutError(f"unblock of {sorted(blocked)}: a use is no element access of the blocked rank")
    rewritten = 0
    for name, (masks, factors) in blocked.items():
        if name not in arguments(func):
            body, index = definition(func, name)
            stmt = body[index]
            assert isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call)
            shape = stmt.value.args[0]
            assert isinstance(shape, ast.Tuple)
            extents = [
                recover_extent(shape.elts[d], factors[d]) if m else ast.unparse(shape.elts[d])
                for d, m in enumerate(masks)
            ]
            stmt.value.args[0] = parse(f"({', '.join(extents)},)")
        for use in uses(func, name):
            if not isinstance(use, ast.Subscript):
                continue
            indices = scalar_indices(use)
            if indices is None:
                raise LayoutError(f"{where(use)}: {name!r} is no element access")
            merged = [
                ast.BinOp(
                    ast.BinOp(indices[d], ast.Mult(), ast.Constant(factors[d])),
                    ast.Add(),
                    indices[inner_slot(masks, d)],
                )
                if m
                else indices[d]
                for d, m in enumerate(masks)
            ]
            use.slice = merged[0] if len(merged) == 1 else ast.Tuple(merged, ast.Load())
            rewritten += 1
    return rewritten
