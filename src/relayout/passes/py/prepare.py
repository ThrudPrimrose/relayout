# Copyright 2026 ETH Zurich and the relayout authors.
"""A Python function in the form the layout passes read best: hand-tiled loops over blocked arrays collapsed and
the arrays unblocked (:func:`~relayout.passes.py.untile_blocked_loops.untile_blocked_loops`); the counterpart of
:mod:`relayout.passes.sdfg.prepare`.
"""

import ast
from collections.abc import Mapping

from relayout.passes.py.common import Array
from relayout.passes.py.untile_blocked_loops import untile_blocked_loops

__all__ = ["prepare"]


def prepare(tree: ast.Module, arrays: Mapping[str, Array] | None = None, function: str | None = None) -> ast.Module:
    """``tree`` in the form the rewrite reads best, in place: tiled loops and blocked arrays untiled."""
    untile_blocked_loops(tree, arrays, function)
    return tree
