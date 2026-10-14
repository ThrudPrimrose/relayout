# Copyright 2026 ETH Zurich and the relayout authors.
"""``unzip``: an array split along a dimension of constant extent into one array per index, the counterpart of
:mod:`relayout.passes.sdfg.unzip_array`. An access ``W[i, c]`` with a constant ``c`` on the dimension reads part
``c`` (``W_c[i]``); a data-dependent index picks its part at run time (``(W_0, W_1)[c][i]``), read or written. A
use of the whole array or of a range along the dimension is refused. An argument keeps its array for the
caller, converted into the parts at the entry and back at the exit; a local's definition becomes its parts'.
"""

import ast
from collections.abc import Mapping, Sequence

from relayout.passes.py.common import (
    Array,
    LayoutError,
    add_entry,
    add_exit,
    allocation,
    arguments,
    definition,
    element_loops,
    ensure_numpy,
    fully_written,
    is_read,
    is_written,
    kernel,
    locate,
    statement,
    tuple_text,
    where,
)
from relayout.passes.py.placement import Field

__all__ = ["unzip_array"]


def part_moves(name: str, parts: Sequence[str], dim: int, shape: Sequence[str], inward: bool) -> list[ast.stmt]:
    """Loops copying ``name`` into its parts (``inward``: the parts back into ``name``); a part of a rank-1
    array holds its one cell at index 0."""
    rest = [extent for k, extent in enumerate(shape) if k != dim]
    params = [f"__r{d}" for d in range(len(rest))]
    out: list[ast.stmt] = []
    for c, part in enumerate(parts):
        whole = Field(c, dim).access(name, params)
        piece = f"{part}[{', '.join(params) or '0'}]"
        copy = f"{whole} = {piece}" if inward else f"{piece} = {whole}"
        out += element_loops(rest, params, [statement(copy)])
    return out


class Split(ast.NodeTransformer):
    """The accesses of ``name`` moved to the part their index on ``dim`` names."""

    def __init__(self, name: str, parts: Sequence[str], dim: int, rank: int) -> None:
        self.name = name
        self.parts = tuple(parts)
        self.dim = dim
        self.rank = rank

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if not (isinstance(node.value, ast.Name) and node.value.id == self.name):
            return self.generic_visit(node)
        elements = list(node.slice.elts) if isinstance(node.slice, ast.Tuple) else [node.slice]
        elements = [self.visit(e) for e in elements]
        if len(elements) != self.rank or isinstance(elements[self.dim], ast.Slice):
            raise LayoutError(f"{where(node)}: {self.name!r} is unzipped along dimension {self.dim}; index it there")
        index = elements.pop(self.dim)
        rest = ast.Tuple(elements or [ast.Constant(0)], ast.Load())
        if isinstance(index, ast.Constant) and isinstance(index.value, int) and 0 <= index.value < len(self.parts):
            return ast.Subscript(ast.Name(self.parts[index.value], ast.Load()), rest, node.ctx)
        if isinstance(index, ast.Constant):
            raise LayoutError(f"{where(node)}: {self.name!r} has {len(self.parts)} parts; index {index.value} is none")
        parts = ast.Tuple([ast.Name(part, ast.Load()) for part in self.parts], ast.Load())
        return ast.Subscript(ast.Subscript(parts, index, ast.Load()), rest, node.ctx)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id == self.name:
            raise LayoutError(f"{where(node)}: {node.id!r} is unzipped into {list(self.parts)}; use it by element")
        return node


def unzip_array(
    tree: ast.Module,
    name: str,
    into: Sequence[str],
    dim: int,
    arrays: Mapping[str, Array],
    function: str | None = None,
) -> list[str]:
    """Split array ``name`` of ``tree``'s function (shape and element type in ``arrays``) along dimension ``dim``
    (constant extent ``len(into)``) into one array per index, ``into[c]`` holding ``name[..., c, ...]`` (see the
    module docstring)."""
    func = kernel(tree, function)
    ensure_numpy(tree)
    spec = arrays[name]
    if spec.element is None:
        raise LayoutError(f"unzip of the struct array {name!r} into its members is not rewritten yet")
    if spec.shape[dim] != str(len(into)):
        raise LayoutError(
            f"{name!r} has extent {spec.shape[dim]} on dimension {dim}; name one array per index ({len(into)} given)"
        )
    rest = [extent for k, extent in enumerate(spec.shape) if k != dim] or ["1"]
    allocate = [statement(f"{part} = np.zeros({tuple_text(rest)}, dtype=np.dtype({spec.element!r}))") for part in into]
    split = Split(name, into, dim, len(spec.shape))
    if name in arguments(func):
        read, written = is_read(func, name), is_written(func, name)
        keeps_cells = written and not fully_written(func, name, spec.shape)
        func.body = [split.visit(stmt) for stmt in func.body]
        entry = list(allocate)
        if read or keeps_cells:
            entry += part_moves(name, into, dim, spec.shape, inward=False)
        if written:  # the exit wraps the body only: the entry runs before it
            add_exit(func, part_moves(name, into, dim, spec.shape, inward=True))
        add_entry(func, entry)
        return list(into)
    body, index = definition(func, name)
    stmt = body[index]
    assert isinstance(stmt, ast.Assign)
    placeholder = ast.Pass()
    body[index] = placeholder
    func.body = [split.visit(s) for s in func.body]
    alloc = allocation(stmt.value)
    replacement: list[ast.stmt] = list(allocate)
    if alloc is None or alloc.kind not in ("zeros", "empty"):
        value = f"relayout_value_{name}"
        stmt.targets = [ast.Name(value, ast.Store())]
        replacement = [stmt, *allocate, *part_moves(value, into, dim, spec.shape, inward=False)]
    body, at = locate(func, placeholder)
    body[at : at + 1] = replacement
    return list(into)
