# Copyright 2026 ETH Zurich and the relayout authors.
"""A numpy / mpi4py program rewritten to run on laid-out arrays: :class:`~relayout.passes.py.regroup.Regroup`
then :class:`~relayout.passes.py.rewrite_accesses.Rewriter`.
"""

import ast
from collections.abc import Mapping

from relayout.layout import IndexMap
from relayout.passes.py.regroup import Regroup, Unzipped, Zipped
from relayout.passes.py.rewrite_accesses import Rewriter

__all__ = ["rewrite"]


def imports_numpy(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Import) and any(alias.name == "numpy" and alias.asname == "np" for alias in node.names)
        for node in ast.walk(tree)
    )


def rewrite(
    tree: ast.Module,
    layouts: Mapping[str, IndexMap],
    zipped: Mapping[str, Zipped] | None = None,
    unzipped: Mapping[str, Unzipped] | None = None,
) -> ast.Module:
    """``tree`` running on the laid-out buffers of ``layouts`` (see the module docstring), after the
    arrays of ``zipped`` / ``unzipped`` are regrouped (:class:`Regroup`; a layout then names the new
    arrays); the input tree is rewritten in place and returned, locations fixed."""
    if zipped or unzipped:
        tree = Regroup(zipped or {}, unzipped or {}).visit(tree)
    rewriter = Rewriter(layouts)
    tree.body = rewriter.statements(tree.body)
    if rewriter.uses_numpy and not imports_numpy(tree):
        tree.body.insert(0, ast.parse("import numpy as np").body[0])
    return ast.fix_missing_locations(tree)
