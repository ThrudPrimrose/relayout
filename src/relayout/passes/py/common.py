# Copyright 2026 ETH Zurich and the relayout authors.
"""What the Python passes share: the refusal and AST helpers."""

import ast

__all__ = ["LayoutError", "parse", "where"]


class LayoutError(ValueError):
    """A use of a laid-out array the rewrite cannot express."""


def parse(text: str) -> ast.expr:
    return ast.parse(text, mode="eval").body


def where(node: ast.AST) -> str:
    return f"line {node.lineno}" if isinstance(node, (ast.expr, ast.stmt)) else "the program"
