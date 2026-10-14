# Copyright 2026 ETH Zurich and the relayout authors.
"""What every Python layout pass shares: the refusal, the program's function and its arrays, where an array
is defined, read and written, and the statements a pass adds at the function's entry and exit.

A pass rewrites one function of a parsed module (:func:`kernel`) in place. As on an SDFG, an argument keeps
its contract with the caller: the function works on a local of its own, converted in at the entry and out at
the exit (:func:`add_exit` wraps the body so every return and raise passes the conversion out). A local array
is rewritten in place at its one definition (:func:`definition`).
"""

import ast
from collections.abc import Iterator, Sequence
from dataclasses import dataclass

__all__ = [
    "MPI_BUFFERS",
    "RECV_KEYWORDS",
    "SEND_KEYWORDS",
    "LAID",
    "Allocation",
    "Array",
    "LayoutError",
    "add_entry",
    "add_exit",
    "allocation",
    "arguments",
    "definition",
    "element_loops",
    "fully_written",
    "ensure_numpy",
    "is_read",
    "is_written",
    "kernel",
    "locate",
    "laid_name",
    "names_in",
    "parse",
    "statement",
    "statements",
    "tuple_text",
    "where",
]

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

#: The prefix of the local an argument is laid out into.
LAID = "laid_"


class LayoutError(ValueError):
    """A use of an array the layout cannot be carried through."""


@dataclass(frozen=True, slots=True)
class Array:
    """A kernel array as the Python passes need it: its logical (row-major) ``shape``, one expression per
    dimension over names in scope in the function, and its element type's numpy name (None for a struct a
    ``zip`` made)."""

    shape: tuple[str, ...]
    element: str | None


def laid_name(name: str) -> str:
    return f"{LAID}{name}"


def parse(text: str) -> ast.expr:
    return ast.parse(text, mode="eval").body


def statement(text: str) -> ast.stmt:
    (out,) = ast.parse(text).body
    return out


def statements(text: str) -> list[ast.stmt]:
    return ast.parse(text).body


def where(node: ast.AST) -> str:
    return (
        f"line {node.lineno}" if isinstance(node, (ast.expr, ast.stmt)) and hasattr(node, "lineno") else "the program"
    )


def names_in(node: ast.AST) -> set[str]:
    return {n.id for n in ast.walk(node) if isinstance(n, ast.Name)}


def ensure_numpy(tree: ast.Module) -> None:
    """``import numpy as np`` at the top of ``tree`` unless it is there: the passes' statements call ``np``."""
    imported = any(
        isinstance(node, ast.Import) and any(alias.name == "numpy" and alias.asname == "np" for alias in node.names)
        for node in tree.body
    )
    if not imported:
        tree.body.insert(0, statement("import numpy as np"))


def tuple_text(values: Sequence[str]) -> str:
    """``values`` as the source of a tuple."""
    return f"({', '.join(values)}{',' if len(values) == 1 else ''})"


def kernel(tree: ast.Module, function: str | None = None) -> ast.FunctionDef:
    """The function a pass rewrites: the one named ``function``, or the module's only one."""
    found = [node for node in tree.body if isinstance(node, ast.FunctionDef)]
    if function is not None:
        found = [node for node in found if node.name == function]
    if len(found) != 1:
        named = "a function" if function is None else f"function {function!r}"
        raise LayoutError(f"the module defines {len(found)} matches for {named}; name the one to rewrite")
    return found[0]


def arguments(func: ast.FunctionDef) -> list[str]:
    args = func.args
    return [a.arg for a in (*args.posonlyargs, *args.args, *args.kwonlyargs)]


def body_start(func: ast.FunctionDef) -> int:
    """The index of the first statement after the docstring."""
    first = func.body[0] if func.body else None
    docstring = (
        isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str)
    )
    return 1 if docstring else 0


def add_entry(func: ast.FunctionDef, added: Sequence[ast.stmt], at: int | None = None) -> None:
    """``added`` run at the function's entry (after its docstring), or before top-level statement ``at``."""
    index = body_start(func) if at is None else at
    func.body[index:index] = list(added)


def add_exit(func: ast.FunctionDef, added: Sequence[ast.stmt]) -> None:
    """``added`` run whenever the function leaves: its body in a ``try`` with ``added`` as the ``finally``."""
    if not added:
        return
    start = body_start(func)
    head, body = func.body[:start], func.body[start:] or [ast.Pass()]
    func.body = [*head, ast.Try(body=body, handlers=[], orelse=[], finalbody=list(added))]


def statement_lists(node: ast.AST) -> list[list[ast.stmt]]:
    """Every statement list in ``node``."""
    out = []
    for walked in ast.walk(node):
        for _, value in ast.iter_fields(walked):
            if isinstance(value, list) and value and all(isinstance(v, ast.stmt) for v in value):
                out.append(value)
    return out


def locate(func: ast.FunctionDef, stmt: ast.stmt) -> tuple[list[ast.stmt], int]:
    """The statement list holding ``stmt`` and its index there."""
    for body in statement_lists(func):
        for k, candidate in enumerate(body):
            if candidate is stmt:
                return body, k
    raise LayoutError(f"{where(stmt)}: the statement is no longer in the function")


def element_loops(shape: Sequence[str], params: Sequence[str], body: Sequence[ast.stmt]) -> list[ast.stmt]:
    """``body`` inside one loop per dimension of ``shape`` (``params[d]`` over ``range(shape[d])``)."""
    out = list(body)
    for param, extent in reversed(list(zip(params, shape, strict=True))):
        loop = statement(f"for {param} in range({extent}):\n    pass")
        assert isinstance(loop, ast.For)
        loop.body = out
        out = [loop]
    return out


@dataclass(frozen=True, slots=True)
class Allocation:
    """An array allocation ``np.<kind>(shape, ...)``: ``fill`` is the value of a ``full`` (None otherwise) and
    ``dtype`` the element type expression (None = numpy's default)."""

    kind: str
    fill: ast.expr | None
    dtype: ast.expr | None


#: The numpy allocations whose cells a pass can allocate in a layout of its own.
ALLOCATIONS = ("zeros", "empty", "ones", "full")


def allocation(value: ast.expr) -> Allocation | None:
    """``value`` as a numpy allocation (``np.zeros(shape, dtype)``, ``np.full(shape, fill, dtype)``, ...), else
    None."""
    if not (isinstance(value, ast.Call) and isinstance(value.func, ast.Attribute)):
        return None
    module = value.func.value
    if not (isinstance(module, ast.Name) and module.id in ("np", "numpy") and value.func.attr in ALLOCATIONS):
        return None
    kind = value.func.attr
    positional = list(value.args)
    keywords = {k.arg: k.value for k in value.keywords}
    if None in keywords or set(keywords) - {"shape", "fill_value", "dtype"}:
        return None
    fill = keywords.get("fill_value", positional[1] if kind == "full" and len(positional) > 1 else None)
    dtype_at = 2 if kind == "full" else 1
    dtype = keywords.get("dtype", positional[dtype_at] if len(positional) > dtype_at else None)
    if kind == "full" and fill is None:
        return None
    return Allocation(kind, fill, dtype)


def assignments_to(func: ast.FunctionDef, name: str) -> Iterator[ast.AST]:
    """Every node of ``func`` that binds ``name``."""
    for node in ast.walk(func):
        if isinstance(node, ast.Name) and node.id == name and isinstance(node.ctx, (ast.Store, ast.Del)):
            yield node
        elif isinstance(node, ast.arg) and node.arg == name and node is not func:
            yield node


def definition(func: ast.FunctionDef, name: str) -> tuple[list[ast.stmt], int]:
    """The statement list and index of ``name = <value>``, the only binding of local ``name``: a statement of the
    function's body, or of the body of a ``try`` there (the exit conversions of an argument wrap the body in one)."""
    bindings = list(assignments_to(func, name))
    body = func.body
    while True:
        for index, stmt in enumerate(body):
            if (
                isinstance(stmt, ast.Assign)
                and len(stmt.targets) == 1
                and isinstance(stmt.targets[0], ast.Name)
                and stmt.targets[0].id == name
            ):
                if len(bindings) != 1:
                    raise LayoutError(
                        f"{where(stmt)}: {name!r} is bound {len(bindings)} times; lay out an array bound once"
                    )
                return body, index
        tries = [stmt for stmt in body if isinstance(stmt, ast.Try) and not stmt.handlers]
        if len(tries) != 1:
            raise LayoutError(f"{name!r} is neither an argument of {func.name!r} nor defined once at its top level")
        body = tries[0].body


class Uses(ast.NodeVisitor):
    """Whether a function reads and whether it may write an array: a write is a store through a subscript,
    an augmented assignment, a method call on it, any call it is passed to (which may write through it) but
    an mpi4py call it is sent by."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.read = False
        self.written = False
        self.store_targets: set[int] = set()

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self.mark_store(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.mark_store(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        if self.root(node.target) == self.name:
            self.read = self.written = True
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call) -> None:
        method = node.func.attr if isinstance(node.func, ast.Attribute) else None
        if method in MPI_BUFFERS:  # an mpi4py call writes its receive buffers only
            receives = MPI_BUFFERS[method][1]
            operands = [a for k, a in enumerate(node.args) if k in receives]
            operands += [k.value for k in node.keywords if k.arg in RECV_KEYWORDS]
        else:
            operands = [*node.args, *(k.value for k in node.keywords)]
            if isinstance(node.func, ast.Attribute):
                operands.append(node.func.value)
        if any(self.root(operand) == self.name for operand in operands):
            self.written = True
        self.generic_visit(node)

    def visit_Name(self, node: ast.Name) -> None:
        if node.id == self.name and isinstance(node.ctx, ast.Load) and id(node) not in self.store_targets:
            self.read = True

    def mark_store(self, target: ast.expr) -> None:
        if isinstance(target, ast.Subscript) and self.root(target) == self.name:
            self.written = True
            self.store_targets.add(id(target.value))
        if isinstance(target, (ast.Tuple, ast.List)):
            for element in target.elts:
                self.mark_store(element)

    @staticmethod
    def root(node: ast.expr) -> str | None:
        """The array an element, a slice or an attribute of ``node`` belongs to."""
        while isinstance(node, (ast.Subscript, ast.Attribute)):
            node = node.value
        return node.id if isinstance(node, ast.Name) else None


def is_read(func: ast.FunctionDef, name: str) -> bool:
    uses = Uses(name)
    uses.visit(func)
    return uses.read


def is_written(func: ast.FunctionDef, name: str) -> bool:
    uses = Uses(name)
    uses.visit(func)
    return uses.written


def top_level(func: ast.FunctionDef) -> list[list[ast.stmt]]:
    """The statement lists that run whenever the function runs to its end: its body, and the body of each ``try``
    without handlers there (the exit conversions of an argument wrap the body in one)."""
    out, body = [], func.body
    while True:
        out.append(body)
        tries = [stmt for stmt in body if isinstance(stmt, ast.Try) and not stmt.handlers]
        if len(tries) != 1:
            return out
        body = tries[0].body


def covering_store(nest: ast.For, name: str, shape: Sequence[str]) -> bool:
    """Whether ``nest`` (loops only, down to its statements) stores every element of ``name``: an assignment whose
    every index is a distinct variable of an enclosing loop over ``range(extent)`` of its dimension."""
    ranges: dict[str, str] = {}
    body: list[ast.stmt] = [nest]
    while len(body) >= 1 and all(isinstance(s, ast.For) for s in body[:1]) and isinstance(body[0], ast.For):
        loop = body[0]
        call = loop.iter
        whole = (
            isinstance(call, ast.Call)
            and isinstance(call.func, ast.Name)
            and call.func.id == "range"
            and len(call.args) == 1
            and not call.keywords
            and isinstance(loop.target, ast.Name)
            and not loop.orelse
        )
        if not whole or any(isinstance(n, (ast.Break, ast.Continue, ast.Return, ast.Raise)) for n in ast.walk(loop)):
            return False
        ranges[loop.target.id] = ast.unparse(call.args[0])  # type: ignore[union-attr]
        body = loop.body
    for stmt in body:
        if not (isinstance(stmt, ast.Assign) and len(stmt.targets) == 1):
            continue
        target = stmt.targets[0]
        if not (isinstance(target, ast.Subscript) and isinstance(target.value, ast.Name) and target.value.id == name):
            continue
        elements = target.slice.elts if isinstance(target.slice, ast.Tuple) else [target.slice]
        names = [e.id if isinstance(e, ast.Name) else None for e in elements]
        if (
            len(names) == len(shape)
            and None not in names
            and len(set(names)) == len(names)
            and all(ranges.get(n) == extent for n, extent in zip(names, shape, strict=True))  # type: ignore[arg-type]
        ):
            return True
    return False


def fully_written(func: ast.FunctionDef, name: str, shape: Sequence[str]) -> bool:
    """Whether a run of ``func`` provably writes every element of ``name``: a loop nest that runs whenever the
    function does (no return before it) and stores every element (:func:`covering_store`). False whenever
    unprovable."""
    for body in top_level(func):
        for stmt in body:
            if isinstance(stmt, ast.For) and covering_store(stmt, name, shape):
                return True
            if any(isinstance(n, (ast.Return, ast.Raise)) for n in ast.walk(stmt)):
                return False
    return False
