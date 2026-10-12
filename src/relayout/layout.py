# Copyright 2026 ETH Zurich and the relayout authors.
"""One array's layout as index arithmetic, the model every frontend and kernel generator reads.

An :class:`IndexMap` places the cells of an array of logical (row-major) ``shape`` in a laid-out buffer
of :meth:`IndexMap.extents`: each physical axis is one mixed-radix digit ``(i[dim] // stride) % live``
of the logical index, stored in ``live + pad`` cells, and a :class:`Renumber` maps the values of some
axes through a bijection (a shuffle, an XOR swizzle). The maps both ways are index expressions
(:meth:`IndexMap.physical`, :meth:`IndexMap.logical`) in the grammar of
:func:`relayout.ops.check_index_expr`: Python syntax over non-negative integers, so ``//`` and ``%`` are
also C's ``/`` and ``%``. The AST rewrite substitutes them into subscripts, the SDFG rewrite into
memlets, and :mod:`relayout.convert` into the conversion kernels.
"""

import ast
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

from relayout.expr import eval_index

__all__ = [
    "Axis",
    "IndexMap",
    "Renumber",
    "add",
    "ceil_div",
    "div",
    "linear",
    "mod",
    "mul",
    "substitute",
]


def is_int(text: str) -> bool:
    return text.lstrip("-").isdigit()


def wrap(text: str) -> str:
    """``text`` as an operand of any operator: parenthesized unless it is a name, a non-negative literal
    or a call."""
    if text.isidentifier() or text.isdigit():
        return text
    return text if isinstance(ast.parse(text, mode="eval").body, ast.Call) else f"({text})"


def term(text: str) -> str:
    """``text`` as a term of a sum: parenthesized when its outermost operator binds looser than ``+``."""
    node = ast.parse(text, mode="eval").body
    if isinstance(node, ast.BinOp) and not isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.FloorDiv, ast.Mod)):
        return f"({text})"
    return text


def add(*terms: str) -> str:
    kept = [term(t) for t in terms if t != "0"]
    return " + ".join(kept) if kept else "0"


def mul(a: str, b: str) -> str:
    if "0" in (a, b):
        return "0"
    if a == "1":
        return b
    if b == "1":
        return a
    return f"{wrap(a)} * {wrap(b)}"


def div(a: str, b: str) -> str:
    if b == "1" or a == "0":
        return a
    if is_int(a) and is_int(b):
        return str(int(a) // int(b))
    return f"{wrap(a)} // {wrap(b)}"


def mod(a: str, b: str) -> str:
    if b == "1":
        return "0"
    return f"{wrap(a)} % {wrap(b)}"


def ceil_div(a: str, b: str) -> str:
    if b == "1":
        return a
    if is_int(a) and is_int(b):
        return str(-(-int(a) // int(b)))
    return div(add(a, str(int(b) - 1)) if is_int(b) else f"{a} + {b} - 1", b)


def linear(index: Sequence[str], extents: Sequence[str]) -> str:
    """The row-major offset of ``index`` in a buffer of ``extents``."""
    offset = "0"
    for value, extent in zip(index, extents, strict=True):
        offset = add(mul(offset, extent), value)
    return offset


def substitute(text: str, values: Mapping[str, str]) -> str:
    """``text`` with each name of ``values`` replaced by its (parenthesized) expression."""

    class Replace(ast.NodeTransformer):
        def visit_Name(self, node: ast.Name) -> ast.expr:
            if node.id not in values:
                return node
            return ast.parse(wrap(values[node.id]), mode="eval").body

    return ast.unparse(Replace().visit(ast.parse(text, mode="eval")))


@dataclass(frozen=True, slots=True)
class Axis:
    """One physical axis: the digit ``(i[dim] // stride) % live`` of the logical index ``i``, stored in
    ``live + pad`` cells. ``live`` None = the dimension's outermost digit, ``i[dim] // stride``, whose
    live extent follows from the shape."""

    name: str
    dim: int
    stride: int
    live: int | None = None
    pad: int = 0


@dataclass(frozen=True, slots=True)
class Renumber:
    """A bijection over the values of ``axes``: ``f`` (one expression per axis, over the axis names)
    gives the stored values, ``f_inv`` the live ones back."""

    axes: tuple[str, ...]
    f: tuple[str, ...]
    f_inv: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class IndexMap:
    """The layout of an array of logical ``shape`` (expressions): its physical ``axes``, outermost first,
    and the :class:`Renumber` s applied to their values in order."""

    shape: tuple[str, ...]
    axes: tuple[Axis, ...]
    renumbers: tuple[Renumber, ...] = ()

    @classmethod
    def identity(cls, shape: Sequence[str]) -> "IndexMap":
        return cls(tuple(shape), tuple(Axis(f"i{d}", d, 1) for d in range(len(shape))))

    def live_extent(self, axis: Axis) -> str:
        return ceil_div(self.shape[axis.dim], str(axis.stride)) if axis.live is None else str(axis.live)

    def extents(self) -> tuple[str, ...]:
        """The laid-out buffer's shape, one extent per axis (live cells and pad)."""
        return tuple(add(self.live_extent(axis), str(axis.pad)) for axis in self.axes)

    def physical(self, index: Sequence[str]) -> tuple[str, ...]:
        """The physical position of the logical index ``index``, one expression per axis."""
        values = {}
        for axis in self.axes:
            digit = div(index[axis.dim], str(axis.stride))
            values[axis.name] = digit if axis.live is None else mod(digit, str(axis.live))
        for renumber in self.renumbers:
            stored = [substitute(expr, values) for expr in renumber.f]
            values.update(zip(renumber.axes, stored, strict=True))
        return tuple(values[axis.name] for axis in self.axes)

    def logical(self, position: Sequence[str]) -> tuple[str, ...]:
        """The logical index stored at the physical ``position``, one expression per dimension."""
        values = {axis.name: value for axis, value in zip(self.axes, position, strict=True)}
        for renumber in reversed(self.renumbers):
            live = [substitute(expr, values) for expr in renumber.f_inv]
            values.update(zip(renumber.axes, live, strict=True))
        index = []
        for dim in range(len(self.shape)):
            terms = [mul(values[axis.name], str(axis.stride)) for axis in self.axes if axis.dim == dim]
            index.append(add(*terms))
        return tuple(index)

    def live(self, position: Sequence[str]) -> tuple[str, ...]:
        """The conditions under which the physical ``position`` holds a cell of the array: every padded
        axis below its live extent and every blocked dimension below its shape (empty = always)."""
        values = {axis.name: value for axis, value in zip(self.axes, position, strict=True)}
        for renumber in reversed(self.renumbers):
            values.update(zip(renumber.axes, (substitute(e, values) for e in renumber.f_inv), strict=True))
        conditions = [f"{wrap(values[axis.name])} < {wrap(self.live_extent(axis))}" for axis in self.axes if axis.pad]
        logical = self.logical(position)
        for dim, extent in enumerate(self.shape):
            own = [axis for axis in self.axes if axis.dim == dim]
            if len(own) > 1 and not divides(own, extent):
                conditions.append(f"{wrap(logical[dim])} < {wrap(extent)}")
        return tuple(conditions)

    def permutation(self) -> tuple[int, ...] | None:
        """The axis permutation this layout is (``axes[k]`` holds dimension ``perm[k]``), or None when it
        cuts, pads or renumbers."""
        if self.renumbers or len(self.axes) != len(self.shape):
            return None
        if any(axis.stride != 1 or axis.live is not None or axis.pad for axis in self.axes):
            return None
        return tuple(axis.dim for axis in self.axes)

    def is_identity(self) -> bool:
        return self.permutation() == tuple(range(len(self.shape)))

    def size(self, names: Mapping[str, int]) -> int:
        """The number of cells of the laid-out buffer with the shape's symbols bound to ``names``."""
        return math.prod(eval_index(extent, names, "extents") for extent in self.extents())

    def with_shape(self, shape: Sequence[str]) -> "IndexMap":
        return replace(self, shape=tuple(shape))


def divides(axes: Sequence[Axis], extent: str) -> bool:
    """Whether a dimension of ``extent`` cut into ``axes`` fills them exactly (statically known)."""
    if not is_int(extent):
        return False
    block = math.prod(axis.live for axis in axes if axis.live is not None)
    return int(extent) % block == 0
