# Copyright 2026 ETH Zurich and the relayout authors.
"""``zip`` / ``unzip`` on a Python program: a zipped array's accesses address its field of the zipped array,
an unzipped array's address the part their index names, and the function takes the new arrays.
"""

import ast
from collections.abc import Mapping
from dataclasses import dataclass

from relayout.passes.py.common import LayoutError, parse, where

__all__ = ["Regroup", "Unzipped", "Zipped"]


@dataclass(frozen=True, slots=True)
class Zipped:
    """An array zipped into ``into`` at ``index`` of its field dimension ``dim`` (None: last)."""

    into: str
    index: int
    dim: int | None = None

    def view(self) -> str:
        """The field as a view of the zipped array."""
        if self.dim is None:
            return f"{self.into}[..., {self.index}]"
        return f"{self.into}[{', '.join([':'] * self.dim + [str(self.index)])}]"


@dataclass(frozen=True, slots=True)
class Unzipped:
    """An array unzipped along ``dim`` into ``parts``, one array per index."""

    parts: tuple[str, ...]
    dim: int


class Regroup(ast.NodeTransformer):
    """``zip`` / ``unzip`` on the program: a zipped array's accesses address its field of the zipped
    array (``A[i, j]`` -> ``Z[i, j, k]``, a whole use the view ``Z[..., k]``), an unzipped array's address
    the part their constant index names (``W[i, c]`` -> ``W_c[i]``), and the function takes the new arrays."""

    def __init__(self, zipped: Mapping[str, Zipped], unzipped: Mapping[str, Unzipped]) -> None:
        self.zipped = zipped
        self.unzipped = unzipped

    def visit_FunctionDef(self, node: ast.FunctionDef) -> ast.AST:
        params: list[ast.arg] = []
        for arg in node.args.args:
            if arg.arg in self.zipped:
                into = self.zipped[arg.arg].into
                if all(p.arg != into for p in params):
                    params.append(ast.arg(into))
            elif arg.arg in self.unzipped:
                params += [ast.arg(part) for part in self.unzipped[arg.arg].parts]
            else:
                params.append(arg)
        node.args.args = params
        return self.generic_visit(node)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        name = node.value.id if isinstance(node.value, ast.Name) else ""
        elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
        elements = [self.visit(e) for e in elements]
        if name in self.zipped:
            field = self.zipped[name]
            elements.insert(len(elements) if field.dim is None else field.dim, ast.Constant(field.index))
            index = ast.Tuple(elements, ast.Load())
            return ast.Subscript(ast.Name(field.into, ast.Load()), index, node.ctx)
        if name in self.unzipped:
            unzipped = self.unzipped[name]
            if unzipped.dim >= len(elements) or isinstance(elements[unzipped.dim], ast.Slice):
                raise LayoutError(f"{where(node)}: {name!r} is unzipped along dimension {unzipped.dim}; index it there")
            index = elements.pop(unzipped.dim)
            rest = ast.Tuple(elements or [ast.Constant(0)], ast.Load())
            if isinstance(index, ast.Constant) and isinstance(index.value, int):
                return ast.Subscript(ast.Name(unzipped.parts[index.value], ast.Load()), rest, node.ctx)
            # a data-dependent index picks its part at run time: ``(W_0, W_1)[c][i]``, read or written
            parts = ast.Tuple([ast.Name(part, ast.Load()) for part in unzipped.parts], ast.Load())
            return ast.Subscript(ast.Subscript(parts, index, ast.Load()), rest, node.ctx)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id in self.zipped and node.attr == "shape":
            field = self.zipped[node.value.id]
            dim = -1 if field.dim is None else field.dim
            return parse(f"{field.into}.shape[:{dim}] + {field.into}.shape[{dim}:][1:]")
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if node.id in self.zipped and isinstance(node.ctx, ast.Load):
            return parse(self.zipped[node.id].view())
        if node.id in self.unzipped:
            raise LayoutError(
                f"{where(node)}: {node.id!r} is unzipped into {list(self.unzipped[node.id].parts)}; use it by element"
            )
        return node
