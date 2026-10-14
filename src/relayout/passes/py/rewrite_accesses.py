# Copyright 2026 ETH Zurich and the relayout authors.
"""Every access to an array of a Python function moved to a target buffer through a
:class:`~relayout.passes.py.placement.Placement`, the counterpart of :mod:`relayout.passes.sdfg.rewrite_accesses`:

- an element access ``A[i, j]`` (one scalar index per dimension) reads and writes the target's cell
  (:meth:`~relayout.passes.py.placement.Placement.access`), for every placement;
- any other use of ``A`` (a slice, a whole-array operation, a call) uses the logical view of the target
  (:meth:`~relayout.passes.py.placement.Placement.view`), which numpy reads and writes through; a layout that
  cuts or renumbers a dimension has no such view, and the use is refused by its line;
- ``A.shape`` is the logical shape and ``A.dtype`` the element type;
- an mpi4py buffer of ``A`` is recorded for :func:`~relayout.passes.py.pack_mpi_buffers.pack_mpi_buffers`.

:func:`move_array` adds what an argument needs, as it keeps its array for the caller: the target converted in
at the entry (when the function reads the array, or writes it without provably writing every cell) and
out at the exit (when it may write it), by loops over the logical index space. A local's definition becomes
its conversion into the target. The symbols of the logical shape must be in scope in the function.
"""

import ast
from collections.abc import Callable, Sequence

from relayout.layout import IndexMap
from relayout.passes.py.common import (
    LayoutError,
    add_entry,
    add_exit,
    allocation,
    arguments,
    definition,
    element_loops,
    fully_written,
    is_read,
    is_written,
    locate,
    parse,
    statement,
    where,
)
from relayout.passes.py.pack_mpi_buffers import (
    MPI_BUFFERS,
    RECV_KEYWORDS,
    SEND_KEYWORDS,
    Box,
    MpiBuffer,
    pack_mpi_buffers,
)
from relayout.passes.py.placement import Laid, Placement

__all__ = ["Rewriter", "conversion", "move_array", "rewrite_accesses", "transfer"]


def scalar_indices(node: ast.Subscript) -> list[ast.expr] | None:
    """The index expressions of an element access, or None when an index is a slice, an ellipsis or a new axis."""
    elements = node.slice.elts if isinstance(node.slice, ast.Tuple) else [node.slice]
    if any(isinstance(e, ast.Slice) or (isinstance(e, ast.Constant) and e.value in (Ellipsis, None)) for e in elements):
        return None
    return list(elements)


def root(node: ast.expr) -> str | None:
    """The array an element, a slice or an attribute of ``node`` belongs to."""
    while isinstance(node, (ast.Subscript, ast.Attribute)):
        node = node.value
    return node.id if isinstance(node, ast.Name) else None


class Rewriter(ast.NodeTransformer):
    """The accesses of ``name`` moved to ``target`` through ``placement``, one statement at a time (an MPI
    buffer records the statement its call is in)."""

    def __init__(
        self, name: str, target: str, placement: Placement, shape: Sequence[str], buffers: list[MpiBuffer]
    ) -> None:
        self.name = name
        self.target = target
        self.placement = placement
        self.shape = tuple(shape)
        self.buffers = buffers
        self.current: ast.stmt | None = None

    def statements(self, body: list[ast.stmt]) -> list[ast.stmt]:
        outer = self.current
        out: list[ast.stmt] = []
        for stmt in body:
            self.current = stmt
            out.append(self.visit(stmt))
        self.current = outer
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

    def logical_view(self, node: ast.AST) -> ast.expr:
        view = self.placement.view(self.target)
        if view is None:
            raise LayoutError(
                f"{where(node)}: {self.name!r} is laid out by cuts or a shuffle, which numpy cannot view as the "
                f"logical array; index it one element at a time ({self.name}[i, j, ...])"
            )
        return parse(view)

    def index_text(self, element: ast.expr, dim: int) -> str:
        """An index into dimension ``dim`` as source; a negative constant counts from the end, as in numpy."""
        value = self.visit(element)
        if (
            isinstance(value, ast.UnaryOp)
            and isinstance(value.op, ast.USub)
            and isinstance(value.operand, ast.Constant)
        ):
            return f"({self.shape[dim]}) - {ast.unparse(value.operand)}"
        return ast.unparse(value)

    def visit_Subscript(self, node: ast.Subscript) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == self.name:
            indices = scalar_indices(node)
            if indices is not None and len(indices) == len(self.shape):
                index = [self.index_text(element, dim) for dim, element in enumerate(indices)]
                access = parse(self.placement.access(self.target, index))
                assert isinstance(access, ast.Subscript)
                access.ctx = node.ctx
                return access
        return self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        if isinstance(node.value, ast.Name) and node.value.id == self.name:
            if node.attr == "shape":
                return ast.Tuple([parse(extent) for extent in self.shape], ast.Load())
            if node.attr == "dtype":
                return parse(self.placement.dtype(self.target))
        return self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> ast.expr:
        if node.id != self.name:
            return node
        if not isinstance(node.ctx, ast.Load):
            raise LayoutError(f"{where(node)}: {node.id!r} is laid out; the program may not rebind it")
        return self.logical_view(node)

    def visit_Call(self, node: ast.Call) -> ast.AST:
        method = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if method not in MPI_BUFFERS:
            return self.generic_visit(node)
        sends, receives = MPI_BUFFERS[method]
        node.func = self.visit(node.func)
        for k, arg in enumerate(node.args):
            node.args[k] = self.buffer(node, k, arg, k in sends, k in receives, method)
        for keyword in node.keywords:
            send, receive = keyword.arg in SEND_KEYWORDS, keyword.arg in RECV_KEYWORDS
            assert keyword.arg is not None or not (send or receive)
            keyword.value = self.buffer(node, keyword.arg or "", keyword.value, send, receive, method)
        return node

    def buffer(
        self, call: ast.Call, slot: int | str, arg: ast.expr, send: bool, receive: bool, method: str
    ) -> ast.expr:
        """``arg`` of an mpi4py call: recorded when it is a buffer of ``name`` (the array, or a box of points and
        plain slices of it), and left for :func:`pack_mpi_buffers`; else rewritten like any expression."""
        if not (send or receive) or root(arg) != self.name or isinstance(arg, ast.Attribute):
            return self.visit(arg)
        assert self.current is not None
        request = None
        if method == "Irecv":
            stmt = self.current
            if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1 and isinstance(stmt.targets[0], ast.Name)):
                raise LayoutError(f"{where(arg)}: bind the request of an Irecv into {self.name!r} to one name")
            request = stmt.targets[0].id
        box = None if isinstance(arg, ast.Name) else self.box(arg)
        self.buffers.append(
            MpiBuffer(call, slot, self.current, self.target, self.placement, self.shape, box, send, receive, request)
        )
        return ast.Constant(None)  # the buffer pack_mpi_buffers puts in its place

    def box(self, arg: ast.expr) -> Box:
        """The logical box a subscript of the array names: per dimension a point, or ``(begin, end)``."""
        if not (isinstance(arg, ast.Subscript) and isinstance(arg.value, ast.Name)):
            raise LayoutError(f"{where(arg)}: an MPI buffer of {self.name!r} is the array or one subscript of it")
        elements = arg.slice.elts if isinstance(arg.slice, ast.Tuple) else [arg.slice]
        if len(elements) > len(self.shape):
            raise LayoutError(f"{where(arg)}: {len(elements)} indices into the rank-{len(self.shape)} {self.name!r}")
        out: list[str | tuple[str, str]] = []
        for dim, extent in enumerate(self.shape):
            element = elements[dim] if dim < len(elements) else ast.Slice()
            if isinstance(element, ast.Slice):
                if element.step is not None:
                    raise LayoutError(f"{where(arg)}: an MPI buffer of {self.name!r} through a step")
                begin = "0" if element.lower is None else self.index_text(element.lower, dim)
                end = extent if element.upper is None else self.index_text(element.upper, dim)
                out.append((begin, end))
            elif isinstance(element, ast.Constant) and element.value in (Ellipsis, None):
                raise LayoutError(f"{where(arg)}: an MPI buffer of {self.name!r} through an ellipsis or a new axis")
            else:
                out.append(self.index_text(element, dim))
        return tuple(out)


def rewrite_accesses(
    func: ast.FunctionDef,
    name: str,
    target: str,
    placement: Placement,
    shape: Sequence[str],
    pending: list[MpiBuffer] | None = None,
) -> None:
    """Every access to ``name`` in ``func`` moved to ``target`` through ``placement``. With ``pending``, an MPI
    buffer is recorded there for :func:`pack_mpi_buffers`; without it, it is packed at once."""
    own: list[MpiBuffer] = []
    func.body = Rewriter(name, target, placement, shape, own).statements(func.body)
    if pending is None:
        pack_mpi_buffers(func, own)
    else:
        pending.extend(own)


def transfer(
    source: str, source_at: Placement, target: str, target_at: Placement, shape: Sequence[str]
) -> list[ast.stmt]:
    """Loops over the logical index space of ``shape`` copying every cell of ``source`` (placed by
    ``source_at``) to the same logical cell of ``target`` (placed by ``target_at``)."""
    params = [f"__r{d}" for d in range(len(shape))]
    body = statement(f"{target_at.access(target, params)} = {source_at.access(source, params)}")
    return element_loops(shape, params, [body])


def conversion(logical: str, laid: str, shape: Sequence[str], placement: Placement, inward: bool) -> list[ast.stmt]:
    """Loops moving ``logical`` (a row-major array of ``shape``) into ``laid`` (placed by ``placement``), or back."""
    logical_at = Laid(IndexMap.identity(shape))
    if inward:
        return transfer(logical, logical_at, laid, placement, shape)
    return transfer(laid, placement, logical, logical_at, shape)


#: The allocation of a local's target, given the expression of its element type.
Allocate = Callable[[str], Sequence[ast.stmt]]


def move_array(
    func: ast.FunctionDef,
    name: str,
    target: str,
    placement: Placement,
    shape: Sequence[str],
    allocate: Allocate | None = None,
    pending: list[MpiBuffer] | None = None,
) -> None:
    """Move ``name``'s accesses to ``target`` through ``placement``. An argument keeps its array for the caller:
    ``allocate`` (the target's allocation, if the target is new) and the conversion in run at the entry, the
    conversion out at the exit. A local's one definition becomes ``allocate`` and its conversion into the target;
    a ``np.zeros`` or ``np.empty`` definition needs no conversion (the target is allocated with zeros)."""
    if name in arguments(func):
        read, written = is_read(func, name), is_written(func, name)
        keeps_cells = written and not fully_written(func, name, shape)
        rewrite_accesses(func, name, target, placement, shape, pending)
        entry = list(allocate(f"{name}.dtype")) if allocate is not None else []
        if read or keeps_cells:  # a write in part must not lose the cells the program leaves
            entry += conversion(name, target, shape, placement, inward=True)
        if written:  # the exit wraps the body only: the entry runs before it
            add_exit(func, conversion(name, target, shape, placement, inward=False))
        add_entry(func, entry)
        return
    body, index = definition(func, name)
    stmt = body[index]
    assert isinstance(stmt, ast.Assign)
    placeholder = ast.Pass()
    body[index] = placeholder  # the definition is rebuilt below; the rewrite must not see it
    rewrite_accesses(func, name, target, placement, shape, pending)
    alloc = allocation(stmt.value)
    if alloc is not None and alloc.kind in ("zeros", "empty"):
        dtype = "np.float64" if alloc.dtype is None else ast.unparse(alloc.dtype)
        replacement = list(allocate(dtype)) if allocate is not None else []
    else:
        value = f"relayout_value_{name}"
        stmt.targets = [ast.Name(value, ast.Store())]
        made = list(allocate(f"{value}.dtype")) if allocate is not None else []
        replacement = [stmt, *made, *conversion(value, target, shape, placement, inward=True)]
    body, at = locate(func, placeholder)
    body[at : at + 1] = replacement
