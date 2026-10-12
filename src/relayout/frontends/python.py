# Copyright 2026 ETH Zurich and the relayout authors.
"""A numpy / mpi4py program rewritten to run on laid-out arrays.

:func:`rewrite` takes the program's AST and each laid-out array's :class:`~relayout.layout.IndexMap`
and returns the program that computes the same results on the laid-out buffers; the symbols of each
layout's shape must be in scope wherever its array is used:

- an element access ``A[i, j]`` (one scalar index per dimension) reads and writes the physical cell
  ``A[*layout.physical((i, j))]``, for every layout;
- any other use of ``A`` (a slice, a whole-array operation, a call) uses the logical view of the
  buffer, a slice of the pad and a transpose, which numpy reads and writes through; a layout that
  cuts or renumbers a dimension has no such view, and the use is refused by its line;
- ``A.shape`` is the logical shape;
- an mpi4py transfer of a laid-out array sends a contiguous copy of the view and receives into a
  contiguous buffer copied back after the call (pack / unpack), since mpi4py takes contiguous
  buffers only; a non-blocking receive into one is refused, as its unpack would belong at the wait.

The access rewrite is what a lowering to C needs: hb lowers numpy to loops over element accesses
before emitting C, so the rewrite there runs on element accesses only.
"""

import ast
from collections.abc import Mapping

from relayout.layout import IndexMap

__all__ = ["MPI_BUFFERS", "LayoutError", "rewrite"]


class LayoutError(ValueError):
    """A use of a laid-out array the rewrite cannot express."""


#: mpi4py communicator method -> (positions of its send buffers, positions of its receive buffers),
#: the receive buffers also by keyword (``recvbuf``) and the send ones too (``sendbuf``, ``buf``).
MPI_BUFFERS: dict[str, tuple[tuple[int, ...], tuple[int, ...]]] = {
    "Send": ((0,), ()),
    "Ssend": ((0,), ()),
    "Rsend": ((0,), ()),
    "Bsend": ((0,), ()),
    "Isend": ((0,), ()),
    "Recv": ((), (0,)),
    "Irecv": ((), (0,)),
    "Bcast": ((0,), (0,)),
    "Sendrecv": ((0,), (3,)),
    "Allreduce": ((0,), (1,)),
    "Reduce": ((0,), (1,)),
    "Alltoall": ((0,), (1,)),
    "Allgather": ((0,), (1,)),
    "Gather": ((0,), (1,)),
    "Scatter": ((0,), (1,)),
    "Scan": ((0,), (1,)),
    "Exscan": ((0,), (1,)),
}
SEND_KEYWORDS, RECV_KEYWORDS = frozenset({"sendbuf", "buf"}), frozenset({"recvbuf"})


def parse(text: str) -> ast.expr:
    return ast.parse(text, mode="eval").body


def logical_view(name: str, layout: IndexMap, where: str) -> ast.expr:
    """``name``'s laid-out buffer read as the logical array, without a copy."""
    view = layout.view()
    if view is None:
        raise LayoutError(
            f"{where}: {name!r} is laid out by cuts or a shuffle, which numpy cannot view as the logical "
            f"array; index it one element at a time ({name}[i, j, ...])"
        )
    live, order = view
    text = name
    if any(extent is not None for extent in live):
        text += "[" + ", ".join(":" if extent is None else f":{extent}" for extent in live) + "]"
    if order != tuple(range(len(order))):
        text += f".transpose({', '.join(map(str, order))})"
    return parse(text)


def scalar_indices(node: ast.Subscript) -> list[ast.expr] | None:
    """The index expressions of an element access, or None when an index is a slice or an ellipsis."""
    elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    if any(isinstance(e, ast.Slice) or (isinstance(e, ast.Constant) and e.value in (Ellipsis, None)) for e in elements):
        return None
    return elements


def where(node: ast.AST) -> str:
    return f"line {node.lineno}" if isinstance(node, (ast.expr, ast.stmt)) else "the program"


class Rewriter(ast.NodeTransformer):
    """The rewrite of :func:`rewrite`, one statement at a time: a transfer adds the statements packing
    its buffers before it and unpacking them after it."""

    def __init__(self, layouts: Mapping[str, IndexMap]) -> None:
        self.layouts = layouts
        self.before: list[ast.stmt] = []
        self.after: list[ast.stmt] = []
        self.packs = 0
        self.uses_numpy = False

    def statements(self, body: list[ast.stmt]) -> list[ast.stmt]:
        outer = self.before, self.after
        out: list[ast.stmt] = []
        for statement in body:
            self.before, self.after = [], []
            rewritten = self.visit(statement)
            out += [*self.before, rewritten, *self.after]
        self.before, self.after = outer
        return out

    def generic_visit(self, node: ast.AST) -> ast.AST:
        for field, value in ast.iter_fields(node):
            if isinstance(value, list) and value and all(isinstance(v, ast.stmt) for v in value):
                value[:] = self.statements(value)
            elif isinstance(value, list):
                value[:] = [self.visit(v) if isinstance(v, ast.AST) else v for v in value]
            elif isinstance(value, ast.AST):
                setattr(node, field, self.visit(value))  # an AST field by name, as NodeTransformer does
        return node

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id in self.layouts:
            name = node.value.id
            layout = self.layouts[name]
            indices = scalar_indices(node)
            if indices is not None and len(indices) == len(layout.shape):
                index = [ast.unparse(self.visit(element)) for element in indices]
                position = [parse(expr) for expr in layout.physical(index)]
                return ast.Subscript(ast.Name(name, ast.Load()), ast.Tuple(position, ast.Load()), node.ctx)
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id in self.layouts and node.attr == "shape":
            return ast.Tuple([parse(extent) for extent in self.layouts[node.value.id].shape], ast.Load())
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id not in self.layouts:
            return node
        if not isinstance(node.ctx, ast.Load):
            raise LayoutError(f"{where(node)}: {node.id!r} is laid out; the program may not rebind it")
        return logical_view(node.id, self.layouts[node.id], where(node))

    def visit_Call(self, node: ast.Call) -> ast.AST:
        method = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if method not in MPI_BUFFERS:
            return self.generic_visit(node)
        sends, receives = MPI_BUFFERS[method]
        node.func = self.visit(node.func)
        node.args = [self.buffer(a, k in sends, k in receives, method) for k, a in enumerate(node.args)]
        for keyword in node.keywords:
            send, receive = keyword.arg in SEND_KEYWORDS, keyword.arg in RECV_KEYWORDS
            keyword.value = self.buffer(keyword.value, send, receive, method)
        return node

    def buffer(self, arg: ast.expr, send: bool, receive: bool, method: str) -> ast.expr:
        """``arg`` as an mpi4py buffer: a contiguous copy to send, a contiguous buffer to receive into."""
        laid_out = any(isinstance(n, ast.Name) and n.id in self.layouts for n in ast.walk(arg))
        view = self.visit(arg)
        if not laid_out or not (send or receive):
            return view
        if method == "Irecv":
            raise LayoutError(f"{where(arg)}: Irecv into a laid-out array; receive with Recv (or unpack at the wait)")
        self.uses_numpy = True
        if not receive:
            return parse(f"np.ascontiguousarray({ast.unparse(view)})")
        self.packs += 1
        packed = f"relayout_packed_{self.packs}"
        prepare = f"np.ascontiguousarray({ast.unparse(view)})" if send else f"np.empty_like({ast.unparse(view)})"
        self.before.append(ast.parse(f"{packed} = {prepare}").body[0])
        self.after.append(ast.parse(f"{ast.unparse(view)}[...] = {packed}").body[0])
        return ast.Name(packed, ast.Load())


def imports_numpy(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.Import) and any(alias.name == "numpy" and alias.asname == "np" for alias in node.names)
        for node in ast.walk(tree)
    )


def rewrite(tree: ast.Module, layouts: Mapping[str, IndexMap]) -> ast.Module:
    """``tree`` running on the laid-out buffers of ``layouts`` (see the module docstring); the input tree
    is rewritten in place and returned, locations fixed."""
    rewriter = Rewriter(layouts)
    tree.body = rewriter.statements(tree.body)
    if rewriter.uses_numpy and not imports_numpy(tree):
        tree.body.insert(0, ast.parse("import numpy as np").body[0])
    return ast.fix_missing_locations(tree)
