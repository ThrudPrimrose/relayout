# Copyright 2026 ETH Zurich and the relayout authors.
"""mpi4py buffers of laid-out arrays, the counterpart of :mod:`relayout.passes.sdfg.pack_mpi_buffers`:
:func:`pack_mpi_buffers` moves each buffer through a contiguous array in logical order, packed before the call
when it sends and unpacked after it when it receives (after the ``Wait`` of an ``Irecv``), so the wire order is
the logical one, as the program had it before the layout. Where numpy views the buffer's cells as the logical
box the view is packed (``np.ascontiguousarray``, a copy only when they are not one run), else loops move
them. Where one call both sends and receives, every buffer of it is a whole array of one layout and that layout
keeps whole logical rows outermost, the laid-out buffers go as they are: every rank runs the program, so the
wire carries one layout's cells end to end, rank chunk by rank chunk.
"""

import ast
from collections.abc import Sequence
from dataclasses import dataclass

from relayout.layout import rows_outermost
from relayout.passes.py.common import (
    MPI_BUFFERS,
    RECV_KEYWORDS,
    SEND_KEYWORDS,
    LayoutError,
    element_loops,
    locate,
    names_in,
    parse,
    statement,
    statements,
    where,
)
from relayout.passes.py.placement import Placement

__all__ = ["MPI_BUFFERS", "RECV_KEYWORDS", "SEND_KEYWORDS", "Box", "MpiBuffer", "pack_mpi_buffers"]

#: A logical box: per dimension a point (an index expression) or a range ``(begin, end)`` (end exclusive).
type Box = tuple[str | tuple[str, str], ...]


@dataclass(frozen=True, slots=True)
class MpiBuffer:
    """An mpi4py call's buffer of a laid-out array, waiting for :func:`pack_mpi_buffers`: the call and its
    argument ``slot`` (a position or a keyword), the statement the call is in, the buffer ``target`` holding the
    array through ``placement`` (logical ``shape``), the logical ``box`` it moves (None = the whole array),
    whether it sends and receives, and the request name of an ``Irecv``."""

    call: ast.Call
    slot: int | str
    stmt: ast.stmt
    target: str
    placement: Placement
    shape: tuple[str, ...]
    box: Box | None
    send: bool
    receive: bool
    request: str | None

    def put(self, value: ast.expr) -> None:
        if isinstance(self.slot, int):
            self.call.args[self.slot] = value
            return
        keyword = next(k for k in self.call.keywords if k.arg == self.slot)
        keyword.value = value

    def whole(self) -> tuple[tuple[str, str], ...]:
        return tuple(("0", extent) for extent in self.shape)

    def ranges(self) -> Box:
        return self.whole() if self.box is None else self.box


def wait_index(body: list[ast.stmt], after: int, request: str) -> int:
    """The index of the statement of ``body`` past ``after`` that waits on ``request`` (``request.Wait()``, or a
    ``Waitall`` naming it)."""
    for k in range(after + 1, len(body)):
        for node in ast.walk(body[k]):
            if not (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)):
                continue
            own = node.func.attr in ("Wait", "wait") and isinstance(node.func.value, ast.Name)
            if own and node.func.value.id == request:
                return k
            if node.func.attr in ("Waitall", "waitall") and any(request in names_in(a) for a in node.args):
                return k
    raise LayoutError(f"{where(body[after])}: no Wait for the request {request!r}; its unpack has no place")


def fresh(func: ast.FunctionDef, base: str) -> str:
    taken = names_in(func)
    k = 0
    while f"{base}{k}" in taken:
        k += 1
    return f"{base}{k}"


def box_subscript(box: Box) -> str:
    return ", ".join(element if isinstance(element, str) else f"{element[0]}:{element[1]}" for element in box)


def packed_moves(buffer: MpiBuffer, packed: str) -> tuple[list[ast.stmt], list[ast.stmt], str]:
    """The statements filling ``packed`` before the call, emptying it after, and its allocation."""
    view = buffer.placement.view(buffer.target)
    if view is not None:
        cells = f"{view}[{box_subscript(buffer.ranges())}]"
        allocate = f"np.ascontiguousarray({cells})" if buffer.send else f"np.empty_like({cells}, order='C')"
        unpack = statements(f"{cells}[...] = {packed}") if buffer.receive else []
        return [], unpack, allocate
    ranges = buffer.ranges()
    spans = [element for element in ranges if not isinstance(element, str)]
    params = [f"__j{k}" for k in range(len(spans))]
    index, k = [], 0
    for element in ranges:
        if isinstance(element, str):
            index.append(element)
        else:
            index.append(f"{element[0]} + {params[k]}")
            k += 1
    sizes = [f"({end}) - ({begin})" for begin, end in spans]
    at = f"{packed}[{', '.join(params) if params else '()'}]"
    cell = buffer.placement.access(buffer.target, index)
    pack = element_loops(sizes, params, [statement(f"{at} = {cell}")]) if buffer.send else []
    unpack = element_loops(sizes, params, [statement(f"{cell} = {at}")]) if buffer.receive else []
    shape = ", ".join(sizes) + ("," if len(sizes) == 1 else "")
    return pack, unpack, f"np.empty(({shape}), dtype={buffer.placement.dtype(buffer.target)})"


def packed_transfer(func: ast.FunctionDef, buffer: MpiBuffer) -> None:
    """``buffer`` moved through a contiguous array in logical order: packed before the call, unpacked after it
    (after the ``Wait`` of an ``Irecv``)."""
    body, k = locate(func, buffer.stmt)
    packed = fresh(func, "relayout_packed_")
    pack, unpack, allocate = packed_moves(buffer, packed)
    buffer.put(ast.Name(packed, ast.Load()))
    before = [statement(f"{packed} = {allocate}"), *pack]
    after_at = k if buffer.request is None else wait_index(body, k, buffer.request)
    body[after_at + 1 : after_at + 1] = unpack
    body[k:k] = before


def shared(group: Sequence[MpiBuffer]) -> bool:
    """Whether the buffers of one call go as they are: it sends and receives, every buffer of it is laid out
    here as a whole array, all in one layout that keeps whole logical rows outermost."""
    call = group[0].call
    method = call.func.attr if isinstance(call.func, ast.Attribute) else ""
    sends, receives = MPI_BUFFERS[method]
    slots = {k for k in (*sends, *receives) if k < len(call.args)}
    slots |= {k.arg for k in call.keywords if k.arg in SEND_KEYWORDS | RECV_KEYWORDS}
    layouts = {b.placement.layout() for b in group}
    (layout,) = layouts if len(layouts) == 1 else (None,)
    return (
        any(b.send for b in group)
        and any(b.receive for b in group)
        and {b.slot for b in group} == slots
        and all(b.box is None for b in group)
        and layout is not None
        and rows_outermost(layout)
    )


def pack_mpi_buffers(func: ast.FunctionDef, buffers: Sequence[MpiBuffer]) -> None:
    """Put every recorded buffer in its call: the laid-out buffers as they are where :func:`shared`, else through
    :func:`packed_transfer`."""
    by_call: dict[int, list[MpiBuffer]] = {}
    for buffer in buffers:
        by_call.setdefault(id(buffer.call), []).append(buffer)
    for group in by_call.values():
        if shared(group):
            for b in group:
                b.put(parse(b.target))
            continue
        for b in group:
            packed_transfer(func, b)
