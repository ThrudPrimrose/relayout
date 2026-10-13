# Copyright 2026 ETH Zurich and the relayout authors.
"""Relayout requests: how a caller asks for the data of a kernel's arrays to be changed.

A request has two parts. ``startup`` settles the run before any array is touched: the libraries to
``load`` and the processor ``grid``, its extents as expressions over the rank count ``P``. ``operations``
is a program run top to bottom over the kernel's arrays, each operation seeing what the previous ones
left: ``cast`` changes an array's precision, ``format`` a sparse array's storage, ``zip`` / ``unzip``
regroup arrays without changing their precisions, the digit operations (``block``, ``split``,
``permute``, ...) reorder an array's storage, ``distribute`` / ``replicate`` hand it to the ranks and
``place`` puts it in host or device memory.

An array starts as one digit per dimension, named ``i0, i1, ...``, over its row-major (numpy) index
space; a Fortran kernel sees the same buffer with reversed extents, a codegen step after the
operations. Every operation is one class registered under its :class:`Action` with typed fields, so
its checks and its effect live in one place, and :mod:`relayout.requests` maps it one to one to JSON.
:func:`lower` turns a run program into one :class:`AxisSplit` per distributed dimension; what it does
not cover yet is refused by name, never approximated. Design: docs/layout.md.
"""

import ast
import dataclasses
import enum
import itertools
import math
import types
import typing
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import Any, ClassVar, NamedTuple, NewType, NoReturn

from relayout.convert import ELEMENTS
from relayout.expr import check_index_expr, eval_index, is_root, node_names
from relayout.layout import Axis, IndexMap, Renumber, substitute
from relayout.sparse import SparseFormat

__all__ = [
    "DIM_DIGIT",
    "FIELD_CHECKS",
    "GRID_SYMBOL",
    "OPERATIONS",
    "RANKS_SYMBOL",
    "Action",
    "ArrayName",
    "ArrayNames",
    "ArraySpec",
    "ArrayState",
    "AxisSplit",
    "BitMatrix",
    "Bits",
    "Block",
    "Cast",
    "Digit",
    "DigitName",
    "DigitNames",
    "DigitOperation",
    "DigitPair",
    "DigitTuple",
    "Dim",
    "Distribute",
    "Expr",
    "Format",
    "FormatLabel",
    "GridDim",
    "IndexExprs",
    "LinearShuffle",
    "Location",
    "Operation",
    "Pad",
    "Permute",
    "Place",
    "Precision",
    "Registry",
    "RegistryError",
    "Replicate",
    "Requests",
    "Scheme",
    "Shuffle",
    "Split",
    "Startup",
    "State",
    "Unblock",
    "Unpad",
    "Unsplit",
    "Unzip",
    "Width",
    "Zip",
    "check_index_expr",
    "eval_index",
    "gf2_inverse",
    "grid_extents",
    "grid_names",
    "inverse_sequence",
    "is_reserved",
    "lower",
    "normal_split",
    "operation",
    "operations_for_axes",
    "optional_inner",
    "run",
]

#: Grid dimension ``g``'s extent in a factor/amount expression: ``P0``, ``P1``, ...
GRID_SYMBOL = "P{}"
#: The rank count in a grid extent expression.
RANKS_SYMBOL = "P"
#: The name of dimension ``d``'s digit before any operation.
DIM_DIGIT = "i{}"


class Action(enum.Enum):
    """What an operation does; its JSON ``action``."""

    CAST = "cast"
    FORMAT = "format"
    ZIP = "zip"
    UNZIP = "unzip"
    BLOCK = "block"
    UNBLOCK = "unblock"
    SPLIT = "split"
    UNSPLIT = "unsplit"
    PERMUTE = "permute"
    PAD = "pad"
    UNPAD = "unpad"
    SHUFFLE = "shuffle"
    LINEAR_SHUFFLE = "linear_shuffle"
    DISTRIBUTE = "distribute"
    REPLICATE = "replicate"
    PLACE = "place"


class Scheme(enum.Enum):
    """How a distributed dimension is dealt to its grid dimension."""

    BLOCK = "block"
    BLOCK_CYCLIC = "block_cyclic"
    CYCLIC = "cyclic"


class Location(enum.Enum):
    """Where an array lives during the call."""

    HOST = "host"
    DEVICE = "device"


class Precision(enum.Enum):
    """The element types a ``cast`` may ask for."""

    FP64 = "fp64"
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"

    @property
    def element(self) -> str:
        """The element type's numpy name (:data:`relayout.convert.ELEMENTS`)."""
        return PRECISION_ELEMENTS[self]


PRECISION_ELEMENTS = {
    Precision.FP64: "float64",
    Precision.FP32: "float32",
    Precision.FP16: "float16",
    Precision.BF16: "bfloat16",
    Precision.FP8_E4M3: "float8_e4m3fn",
    Precision.FP8_E5M2: "float8_e5m2",
}


def grid_names(grid: Sequence[int]) -> dict[str, int]:
    """``{"P0": grid[0], ...}``: the names a factor expression reads."""
    return {GRID_SYMBOL.format(g): int(p) for g, p in enumerate(grid)}


# ---- the processor grid ------------------------------------------------------------------------------


def grid_extents(grid: Sequence[str], ranks: int, where: str) -> tuple[int, ...] | None:
    """The grid ``grid`` spans at ``ranks`` ranks, or None when it does not span exactly that many (a
    square grid at an odd rank count); an extent below 1 is a request fault."""
    extents = tuple(eval_index(extent, {RANKS_SYMBOL: ranks}, f"{where}[{g}]") for g, extent in enumerate(grid))
    if any(extent < 1 for extent in extents):
        raise ValueError(f"{where} {list(grid)} has an extent below 1 at {RANKS_SYMBOL} = {ranks}: {list(extents)}")
    return extents if math.prod(extents) == ranks else None


# ---- field types (each has a JSON codec in relayout.requests) -----------------------------------------------

#: An array name.
ArrayName = NewType("ArrayName", str)
#: Two or more distinct array names.
ArrayNames = NewType("ArrayNames", tuple[str, ...])
#: A digit's name: an identifier other than the grid and rank symbols.
DigitName = NewType("DigitName", str)
#: Distinct digit names (a permutation, the digits a shuffle renumbers).
DigitNames = NewType("DigitNames", tuple[str, ...])
#: The two digits a cut leaves, coarse then fine.
DigitPair = NewType("DigitPair", tuple[str, str])
#: An array dimension.
Dim = NewType("Dim", int)
#: A processor-grid dimension.
GridDim = NewType("GridDim", int)
#: A block width.
Width = NewType("Width", int)
#: An integer expression over ``P0, P1, ...`` (a factor, an amount, a part count).
Expr = NewType("Expr", str)
#: One index expression per shuffled digit, over the shuffled digits' names and ``P0, P1, ...``.
IndexExprs = NewType("IndexExprs", tuple[str, ...])
#: The bit width of each shuffled digit (its extent is ``2**bits``).
Bits = NewType("Bits", tuple[int, ...])
#: A square 0/1 matrix over GF(2): row ``r`` is output bit ``r`` as the XOR of the input bits it marks.
BitMatrix = NewType("BitMatrix", tuple[tuple[int, ...], ...])
#: A sparse format label (``csr``, ``bsr:4``, ...); the caller checks it against the formats it offers.
FormatLabel = NewType("FormatLabel", str)


def is_reserved(name: str) -> bool:
    """Whether ``name`` is the rank symbol or a grid symbol ``P<g>``."""
    return name == RANKS_SYMBOL or (name[:1] == RANKS_SYMBOL and name[1:].isdigit())


def check_name(name: str, where: str) -> None:
    if not isinstance(name, str) or not name:
        raise ValueError(f"{where} must be a name; got {name!r}")


def check_digit_name(name: str, where: str) -> None:
    if not isinstance(name, str) or not name.isidentifier() or is_reserved(name):
        raise ValueError(f"{where} must be a digit name (an identifier other than P, P0, P1, ...); got {name!r}")


def check_distinct(names: Sequence[str], where: str, check: Callable[[str, str], None]) -> None:
    for k, name in enumerate(names):
        check(name, f"{where}[{k}]")
    if len(set(names)) != len(names):
        raise ValueError(f"{where} names one twice: {list(names)}")


def check_array_names(names: Sequence[str], where: str) -> None:
    if len(names) < len(("one", "two")):
        raise ValueError(f"{where} must be a list of two or more names; got {list(names)}")
    check_distinct(names, where, check_name)


def check_digit_names(names: Sequence[str], where: str) -> None:
    if not names:
        raise ValueError(f"{where} must be a non-empty list of digit names")
    check_distinct(names, where, check_digit_name)


def check_digit_pair(names: Sequence[str], where: str) -> None:
    if len(names) != len(("coarse", "fine")):
        raise ValueError(f"{where} must name the two digits the cut leaves, coarse then fine; got {list(names)}")
    check_distinct(names, where, check_digit_name)


def check_format_label(label: str, where: str) -> None:
    try:
        SparseFormat.parse(label)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None


def check_index_exprs(texts: Sequence[str], where: str) -> None:
    for k, text in enumerate(texts):
        check_index_expr(text, f"{where}[{k}]")


def optional_inner(hint: object) -> tuple[object, bool]:
    """``hint`` without ``| None`` and whether it had it; ``NewType | None`` is a ``typing.Union`` before
    Python 3.14 and a ``types.UnionType`` from it."""
    if typing.get_origin(hint) in (typing.Union, types.UnionType) and type(None) in typing.get_args(hint):
        (inner,) = (arg for arg in typing.get_args(hint) if arg is not type(None))
        return inner, True
    return hint, False


#: Field type -> its check, run on every operation at construction (from Python or from JSON).
FIELD_CHECKS: dict[object, Callable[[Any, str], object]] = {
    ArrayName: check_name,
    ArrayNames: check_array_names,
    DigitName: check_digit_name,
    DigitNames: check_digit_names,
    DigitPair: check_digit_pair,
    Expr: check_index_expr,
    IndexExprs: check_index_exprs,
    FormatLabel: check_format_label,
}


# ---- the operation registry ------------------------------------------------------------------------


class RegistryError(ValueError):
    """An operation class breaks the registry's contract, or its action is taken."""


class Registry[K, T]:
    """Key -> class, in registration order."""

    __slots__ = ("entries", "name")

    def __init__(self, name: str) -> None:
        self.name = name
        self.entries: dict[K, T] = {}

    def add(self, key: K, entry: T) -> None:
        if key in self.entries:
            raise RegistryError(f"{self.name} {key!r} is registered twice")
        self.entries[key] = entry


class Operation:
    """One operation of a request's program. A subclass is a frozen slotted dataclass registered with
    :func:`operation`: it names its :class:`Action`, checks itself against the :class:`State` and applies
    its effect there. ``lowers`` False = parsed and checked, refused by :func:`lower` as not implemented
    yet; ``sparse`` True = it applies to a sparse array too."""

    __slots__ = ()
    action: ClassVar[Action]
    lowers: ClassVar[bool] = True
    sparse: ClassVar[bool] = False

    def __post_init__(self) -> None:
        values = dataclasses.asdict(self)  # type: ignore[call-overload]  # every operation is a dataclass
        for spec in dataclasses.fields(self):  # type: ignore[arg-type]
            value = values[spec.name]
            hint, optional = optional_inner(spec.type)
            if optional and value is None:
                continue
            check = FIELD_CHECKS.get(hint)
            if check is not None:
                check(value, spec.name)
        self.validate()

    def validate(self) -> None:
        """Checks across fields, at construction (a request fault); each field's own check is
        :data:`FIELD_CHECKS`'."""

    def apply(self, state: "State") -> None:
        raise NotImplementedError

    def inverse(self, before: tuple[str, ...]) -> "Operation":
        """The operation undoing this one, given the array's digit names before it (digit operations)."""
        raise ValueError(f"{self.action.value} has no inverse operation")


#: Action -> its operation class.
OPERATIONS: Registry[Action, type[Operation]] = Registry("operation")


def operation[C: type[Operation]](cls: C) -> C:
    """Register an operation class under its action; it must define its own ``apply``."""
    if "apply" not in vars(cls):
        raise RegistryError(f"operation {cls.action.value!r} must define apply")
    OPERATIONS.add(cls.action, cls)
    return cls


@dataclass(frozen=True, slots=True)
class Digit:
    """One mixed-radix digit of dimension ``dim``: ``(i[dim] // stride) % extent``. ``extent`` None = not
    known without the array's extents (a dimension's remaining digit). ``band`` marks the coarse digit of
    a ``split``: the band index of ``extent`` load-balanced contiguous bands, not a radix digit."""

    name: str
    dim: int
    stride: int
    extent: int | None
    band: bool = False

    def coarseness(self) -> tuple[int, int, int]:
        """Sorts a dimension's digits coarse to fine (row-major): a band first, then by stride."""
        return (self.dim, 0 if self.band else 1, -self.stride)

    def shape(self) -> tuple[int, int, int | None, bool]:
        """The digit without its name."""
        return (self.dim, self.stride, self.extent, self.band)


class AxisSplit(NamedTuple):
    """How one dimension is split over the grid: ``scheme`` along ``grid_dim``."""

    grid_dim: int
    scheme: Scheme
    block_size: int


class DigitTuple:
    """One array's digit tuple while the operations apply. ``grid`` None = names and shapes
    only (an expression is not evaluated, its digit's extent stays unknown)."""

    __slots__ = ("digits", "distributed", "grid", "initial", "names", "ndim", "pads", "renumbers", "where")

    def __init__(self, ndim: int, grid: Sequence[int] | None, where: str, extents: Sequence[int | None] = ()) -> None:
        self.ndim = ndim
        self.grid = None if grid is None else tuple(grid)
        self.names = {} if grid is None else grid_names(grid)
        known = list(extents) or [None] * ndim
        self.digits = [Digit(DIM_DIGIT.format(d), d, 1, known[d]) for d in range(ndim)]
        #: The digits as the array starts, before any operation.
        self.initial = [digit.shape() for digit in self.digits]
        #: (the distributed digit, its grid dim), in distribute order.
        self.distributed: list[tuple[Digit, int]] = []
        #: Digit name -> its dead cells.
        self.pads: dict[str, int] = {}
        #: The bijections over digit values, in order; their digits are no longer cut or merged.
        self.renumbers: list[Renumber] = []
        #: The op being applied, for refusals.
        self.where = where

    def eval(self, expr: str) -> int | None:
        if self.grid is None:
            return None
        value = eval_index(expr, self.names, self.where)
        if value < 1:
            raise ValueError(f"{self.where}: {expr!r} = {value}; a factor or part count must be >= 1")
        return value

    def order(self) -> tuple[str, ...]:
        return tuple(digit.name for digit in self.digits)

    def whole(self) -> bool:
        """Whether every dimension is still its one uncut digit, in order (names aside)."""
        return [digit.shape() for digit in self.digits] == self.initial

    def find(self, name: str) -> int:
        for position, digit in enumerate(self.digits):
            if digit.name == name:
                return position
        raise ValueError(f"{self.where}: no digit {name!r}; the digits are {list(self.order())}")

    def cut(self, name: str) -> Digit:
        """Digit ``name`` for a cut or a merge: neither padded nor renumbered."""
        if name in self.pads or any(name in renumber.axes for renumber in self.renumbers):
            raise ValueError(f"{self.where}: {name!r} is padded or shuffled; cut or merge digits before that")
        return self.digits[self.find(name)]

    def pad(self, name: str, amount: int | None) -> None:
        """Grow digit ``name`` by ``amount`` dead cells (negative: drop them again)."""
        self.find(name)
        if amount is None:
            return
        cells = self.pads.get(name, 0) + amount
        if cells < 0:
            raise ValueError(f"{self.where}: {name!r} has {self.pads.get(name, 0)} dead cell(s), not {-amount}")
        self.pads[name] = cells
        if cells == 0:
            del self.pads[name]

    def renumber(self, names: Sequence[str], f: Sequence[str], f_inv: Sequence[str]) -> None:
        """Map the values of digits ``names`` through ``f`` (back by ``f_inv``), with ``P0, P1, ...`` bound."""
        for name in names:
            self.find(name)
        grid = {name: str(value) for name, value in self.names.items()}
        f, f_inv = (tuple(substitute(expr, grid) for expr in exprs) for exprs in (f, f_inv))
        last = self.renumbers[-1] if self.renumbers else None
        if last is not None and last.axes == tuple(names) and (last.f, last.f_inv) == (f_inv, f):
            self.renumbers.pop()  # the inverse of the last renumbering: the digits are as they were before it
            return
        for expr in (*f, *f_inv):
            if any(is_root(node) and node_names(node) for node in ast.walk(ast.parse(expr, mode="eval"))):
                raise ValueError(f"{self.where}: {expr!r}: root() takes grid expressions only, not digit values")
        extents = [self.digits[self.find(name)].extent for name in names]
        checked = all(extent is not None for extent in extents) and math.prod(extents) <= BIJECTION_CHECK_LIMIT  # type: ignore[arg-type]
        if checked:
            check_bijection(names, extents, f, f_inv, self.where)  # type: ignore[arg-type]
        self.renumbers.append(Renumber(tuple(names), f, f_inv, checked))

    def replace(self, position: int, new: Sequence[Digit]) -> None:
        """Put ``new`` where the digit at ``position`` was; their names must not clash with the others'."""
        others = {digit.name for k, digit in enumerate(self.digits) if k != position}
        clash = sorted(others & {digit.name for digit in new})
        if clash:
            raise ValueError(f"{self.where}: digit name(s) {clash} are taken; the digits are {list(self.order())}")
        self.digits[position : position + 1] = list(new)

    def block(self, name: str, factor: int | None, into: tuple[str, str]) -> None:
        """Cut digit ``name`` ``(d, s, e)`` into coarse ``(d, s*b, ceil(e/b))`` and fine ``(d, s, b)``, in place."""
        old = self.cut(name)
        position = self.find(name)
        coarser = any(
            d.dim == old.dim and (d.band or d.stride > old.stride)
            for d in (*self.digits, *(digit for digit, _ in self.distributed))
        )
        if coarser and old.extent is not None and factor is not None and old.extent % factor:
            raise ValueError(
                f"{self.where}: {name!r} has extent {old.extent}, which {factor} does not divide; only a dimension's "
                "outermost digit pads, so block an inner digit by a divisor of its extent"
            )
        outer = None if old.extent is None or factor is None else math.ceil(old.extent / factor)
        coarse = Digit(into[0], old.dim, old.stride * (factor or 0), outer, old.band)
        self.replace(position, [coarse, Digit(into[1], old.dim, old.stride, factor)])

    def split(self, name: str, parts: int | None, into: tuple[str, str]) -> None:
        """Cut the whole dimension's digit ``name`` into the band index (extent ``parts``) and the index
        within the band, in place."""
        old = self.cut(name)
        position = self.find(name)
        if old.extent is not None or old.stride != 1 or old.band:
            raise ValueError(f"{self.where}: split cuts a whole dimension's digit; {name!r} is already cut")
        bands = Digit(into[0], old.dim, 1, parts, band=True)
        self.replace(position, [bands, Digit(into[1], old.dim, 1, None)])

    def merge(self, pair: tuple[str, str], into: str, *, band: bool) -> None:
        """Merge the coarse and fine digits ``pair`` of one dimension back into one digit ``into``."""
        coarse, fine = (self.cut(name) for name in pair)
        adjacent = (
            fine.stride == 1 and fine.extent is None and not fine.band
            if band
            else fine.extent is None or coarse.stride == fine.stride * fine.extent
        )
        if coarse.dim != fine.dim or coarse.band != band or not adjacent:
            raise ValueError(f"{self.where}: {list(pair)} are not the two digits of one {'split' if band else 'block'}")
        extent = None if band or coarse.extent is None or fine.extent is None else coarse.extent * fine.extent
        del self.digits[self.find(fine.name)]
        self.replace(self.find(coarse.name), [Digit(into, coarse.dim, fine.stride, extent)])

    def distribute(self, name: str, grid_dim: int) -> None:
        """Hand digit ``name`` (extent ``P_g``) to ``grid_dim``: it becomes the rank coordinate and leaves the tuple."""
        digit = self.cut(name)
        position = self.find(name)
        if self.grid is not None:
            if grid_dim >= len(self.grid):
                raise ValueError(f"{self.where}: grid_dim {grid_dim} does not exist (the grid is {list(self.grid)})")
            if digit.extent != self.grid[grid_dim]:
                raise ValueError(
                    f"{self.where}: a distributed digit must have extent P{grid_dim} = {self.grid[grid_dim]}; "
                    f"{name!r} has {'an unknown extent' if digit.extent is None else digit.extent} (split or block "
                    f"its dimension by P{grid_dim} first)"
                )
        if any(g == grid_dim for _, g in self.distributed):
            raise ValueError(f"{self.where}: grid_dim {grid_dim} is already distributed; each drives one dimension")
        if any(d.dim == digit.dim for d, _ in self.distributed):
            raise ValueError(f"{self.where}: dimension {digit.dim} is already distributed")
        self.distributed.append((digit, grid_dim))
        del self.digits[position]

    def axis_split(self, digit: Digit, grid_dim: int, where: str) -> AxisSplit:
        """The split of the distributed ``digit`` against its dimension's local digits."""
        own = sorted((d.shape() for d in self.digits if d.dim == digit.dim), key=lambda s: (not s[3], -s[1]))
        if digit.band:
            if own != [(digit.dim, 1, None, False)]:
                raise ValueError(
                    f"{where}: a split dimension's local band must stay whole; dimension {digit.dim}'s local digits "
                    f"are {own}"
                )
            return AxisSplit(grid_dim, Scheme.BLOCK, 1)
        parts = digit.extent if digit.extent is not None else 0
        width = digit.stride
        wanted = [(digit.dim, width * parts, None, False)] + ([(digit.dim, 1, width, False)] if width > 1 else [])
        if own != wanted:
            raise ValueError(
                f"{where}: dimension {digit.dim} must be blocked by b*P{grid_dim} and its fine digit by b for a "
                f"block-cyclic distribution of width b (block by P{grid_dim} for cyclic, split by P{grid_dim} for "
                f"contiguous bands); its local digits are {own}"
            )
        return AxisSplit(grid_dim, Scheme.CYCLIC if width == 1 else Scheme.BLOCK_CYCLIC, width)

    def index_map(self, shape: Sequence[str], where: str) -> IndexMap:
        """The rank-local layout of an array whose rank-local logical extents are ``shape``: the remaining
        digits, each distributed digit's stride divided out of the coarser digits of its dimension."""
        if self.grid is None:
            raise ValueError(f"{where}: a layout needs the factors evaluated; run the operations on a grid")
        if len(shape) != self.ndim:
            raise ValueError(f"{where}: the shape {list(shape)} has {len(shape)} dimension(s), not {self.ndim}")
        bands = [digit.name for digit in self.digits if digit.band]
        if bands:
            raise ValueError(f"{where}: {bands} split a dimension into bands for distribution; distribute them")
        axes = []
        for digit in self.digits:
            stride = digit.stride
            for rank, _ in self.distributed:
                if not rank.band and rank.dim == digit.dim and stride >= rank.stride * (rank.extent or 1):
                    stride //= rank.extent or 1
            only = sum(other.dim == digit.dim for other in self.digits) == 1 and stride == 1
            axes.append(
                Axis(digit.name, digit.dim, stride, None if only else digit.extent, self.pads.get(digit.name, 0))
            )
        return IndexMap(tuple(str(extent) for extent in shape), tuple(axes), tuple(self.renumbers))

    def lower(self, where: str, array: str) -> list[AxisSplit | None]:
        """Per dimension its split (None = not split); the rank-local digits must end row-major."""
        axes: list[AxisSplit | None] = [None] * self.ndim
        for digit, grid_dim in self.distributed:
            axes[digit.dim] = self.axis_split(digit, grid_dim, where)
        row_major = sorted(self.digits, key=Digit.coarseness)
        if row_major != self.digits:
            order = [digit.name for digit in row_major]
            raise ValueError(
                f"{where}: the rank-local digits {list(self.order())} are not row-major, which is a local relayout "
                f'and not implemented yet; append {{"action": "permute", "array": "{array}", "order": {order}}}'
            )
        return axes


# ---- the program state ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ArraySpec:
    """A kernel array as a request's program starts from it: whole, on the host, row-major, at the
    problem precision. ``members`` names a struct array's fields; ``native`` marks a struct the kernel
    also takes as one element type (a complex array is ``("re", "im")``, native): it is laid out and cast
    like any array, and ``unzip`` still splits it into its members."""

    name: str
    ndim: int
    sparse: bool = False
    members: tuple[str, ...] = ()
    native: bool = False
    #: Each dimension's extent where it is known without the shape (a zipped array's field), else None.
    extents: tuple[int | None, ...] = ()
    #: The problem element type's numpy name (:data:`relayout.convert.ELEMENTS`); None when unknown.
    element: str | None = None


class ArrayState:
    """One array while the program runs."""

    __slots__ = (
        "digits",
        "element",
        "field_dim",
        "fields",
        "format",
        "location",
        "member_elements",
        "members",
        "name",
        "native",
        "precision",
        "replicated",
        "sparse",
        "touched",
    )

    def __init__(
        self,
        spec: ArraySpec,
        grid: Sequence[int] | None,
        where: str,
        members: tuple[tuple[str, Precision | None], ...] = (),
    ) -> None:
        self.name = spec.name
        self.sparse = spec.sparse
        self.digits = DigitTuple(spec.ndim, grid, where, spec.extents)
        #: The arrays a zip of one precision stacked along dimension ``field_dim``, in order.
        self.fields: tuple[str, ...] = ()
        self.field_dim = 0
        #: A struct's fields with their precisions (None = the problem precision).
        self.members = members or tuple((member, None) for member in spec.members)
        #: A zipped struct's members' problem element types, in order (empty: a native struct's are its component).
        self.member_elements: tuple[str | None, ...] = ()
        self.native = spec.native
        self.precision: Precision | None = None
        self.element = spec.element
        self.format: str | None = None
        self.location: Location | None = None
        self.replicated = False
        #: Whether a digit operation, distribute or replicate has touched it.
        self.touched = False

    @property
    def element_type(self) -> str | None:
        """The element type's numpy name as the casts so far leave it; None when unknown."""
        return self.element if self.precision is None else self.precision.element


class State:
    """The arrays a program has made so far, over ``grid`` (None = names and shapes only)."""

    __slots__ = ("applied", "arrays", "grid", "where")

    def __init__(self, arrays: Sequence[ArraySpec], grid: Sequence[int] | None, where: str) -> None:
        self.grid = None if grid is None else tuple(grid)
        self.where = where
        self.arrays = {spec.name: ArrayState(spec, self.grid, where) for spec in arrays}
        #: The operations applied, in order.
        self.applied: list[Operation] = []

    def array(self, name: str, op: Operation) -> ArrayState:
        """Array ``name`` for ``op``: it must exist now, and be dense unless ``op`` applies to sparse arrays."""
        state = self.arrays.get(name)
        if state is None:
            raise ValueError(f"{self.where}: no array {name!r}; the arrays are {sorted(self.arrays)}")
        if state.sparse and not op.sparse:
            raise ValueError(f"{self.where}: {op.action.value} applies to dense arrays only; {name!r} is sparse")
        return state

    def digits(self, name: str, op: Operation) -> DigitTuple:
        """Array ``name``'s digit tuple, for a digit operation."""
        state = self.array(name, op)
        if state.members and not state.native:
            raise ValueError(f"{self.where}: {name!r} is a struct; zip or unzip before laying it out")
        if state.replicated:
            raise ValueError(f"{self.where}: {name!r} is replicated; a replicated array keeps its layout")
        state.touched = True
        state.digits.where = self.where
        return state.digits

    def fresh(self, name: str) -> None:
        """Refuse ``name`` for a new array when it is taken."""
        if name in self.arrays:
            raise ValueError(f"{self.where}: an array {name!r} exists already")

    def layout(self, name: str, shape: Sequence[str]) -> IndexMap:
        """Array ``name``'s rank-local layout for rank-local logical extents ``shape``."""
        if name not in self.arrays:
            raise ValueError(f"no array {name!r}; the arrays are {sorted(self.arrays)}")
        return self.arrays[name].digits.index_map(shape, f"array {name!r}")

    def untouched(self, name: str, op: Operation) -> ArrayState:
        """Array ``name`` before any layout: zip and unzip regroup whole arrays."""
        state = self.array(name, op)
        if state.touched:
            raise ValueError(f"{self.where}: {op.action.value} regroups whole arrays; {name!r} is already laid out")
        return state


def run(operations: Sequence[Operation], arrays: Sequence[ArraySpec], grid: Sequence[int] | None, where: str) -> State:
    """Run ``operations`` top to bottom over ``arrays``: each sees what the previous ones left, and the
    first that cannot apply is a request fault naming its entry."""
    state = State(arrays, grid, where)
    for index, op in enumerate(operations):
        state.where = f"{where}[{index}] ({op.action.value})"
        op.apply(state)
        state.applied.append(op)
    return state


def lower(state: State) -> dict[str, list[AxisSplit | None] | None]:
    """Each laid-out array's split per dimension (None = replicated) from a run over a grid; an array no
    operation laid out is left out. An operation that does not lower yet is refused by name."""
    refused = next((op for op in state.applied if not op.lowers), None)
    if refused is not None:
        lowered = ", ".join(action.value for action, cls in OPERATIONS.entries.items() if cls.lowers)
        raise ValueError(f"{refused.action.value} is not implemented yet (only {lowered} lower today)")
    out: dict[str, list[AxisSplit | None] | None] = {}
    for name, array in state.arrays.items():
        if array.replicated:
            out[name] = None
        elif array.touched:
            out[name] = array.digits.lower(f"array {name!r}", name)
    return out


def inverse_sequence(operations: Sequence[Operation], ndim: int) -> tuple[Operation, ...]:
    """The operations that undo the digit operations ``operations`` on one ``ndim``-dimensional array:
    reversed, each inverted against the digit names before it."""
    names = {op.array for op in operations if isinstance(op, DigitOperation)}
    if len(names) != 1 or not all(isinstance(op, DigitOperation) for op in operations):
        raise ValueError("inverse_sequence inverts the digit operations of one array")
    (array,) = names
    state = State([ArraySpec(array, ndim)], None, "inverse")
    inverses: list[Operation] = []
    for op in operations:
        before = state.arrays[array].digits.order()
        op.apply(state)
        inverses.append(op.inverse(before))
    return tuple(reversed(inverses))


# ---- the operations ----------------------------------------------------------------------------------


@operation
@dataclass(frozen=True, slots=True)
class Cast(Operation):
    """Change array ``array``'s element type to ``to``; the judge converts it in before the call and out
    after it."""

    action: ClassVar[Action] = Action.CAST
    array: ArrayName
    to: Precision

    def apply(self, state: State) -> None:
        target = state.array(self.array, self)
        if target.members and not target.native:
            raise ValueError(f"{state.where}: {self.array!r} is a struct; cast its fields before zipping them")
        target.precision = self.to
        target.members = tuple((member, self.to) for member, _ in target.members)


@operation
@dataclass(frozen=True, slots=True)
class Format(Operation):
    """Store sparse array ``array`` in format ``to``."""

    action: ClassVar[Action] = Action.FORMAT
    sparse: ClassVar[bool] = True
    array: ArrayName
    to: FormatLabel

    def apply(self, state: State) -> None:
        target = state.array(self.array, self)
        if not target.sparse:
            raise ValueError(f"{state.where}: format applies to sparse arrays; {self.array!r} is dense")
        target.format = self.to


@operation
@dataclass(frozen=True, slots=True)
class Zip(Operation):
    """Fuse whole arrays ``arrays`` of one shape into one array ``into``, the inverse of :class:`Unzip`.
    Arrays of one element type (:attr:`ArrayState.element_type`) are stacked along a new dimension, the field,
    at position ``dim`` (None: last; ``into[..., k]`` is ``arrays[k]``), which the digit operations then place
    like any other (field-outer is SoA, field-inner AoS, blocked between AoSoA); arrays of different element
    types become one array of structs, one member per array in order, each keeping its precision (and take
    no ``dim``)."""

    action: ClassVar[Action] = Action.ZIP
    lowers: ClassVar[bool] = False
    arrays: ArrayNames
    into: ArrayName
    dim: Dim | None = None

    def apply(self, state: State) -> None:
        fields = [state.untouched(name, self) for name in self.arrays]
        if any(field.members or field.fields for field in fields):
            raise ValueError(f"{state.where}: a zip fuses plain arrays; unzip a struct or a zipped array first")
        ndims = {field.digits.ndim for field in fields}
        if len(ndims) != 1:
            raise ValueError(f"{state.where}: zipped arrays must have one shape; {list(self.arrays)} do not")
        locations = {field.location for field in fields}
        if len(locations) != 1:
            raise ValueError(f"{state.where}: zipped arrays must be placed alike; {list(self.arrays)} are not")
        state.fresh(self.into)
        for name in self.arrays:
            del state.arrays[name]
        ndim, elements = ndims.pop(), {field.element_type for field in fields}
        position = ndim if self.dim is None else self.dim
        if position > ndim:
            raise ValueError(
                f"{state.where}: {ndim}-dimensional arrays stack along dimension 0 to {ndim}, not {position}"
            )
        if len(elements) != 1 and self.dim is not None:
            raise ValueError(f"{state.where}: arrays of different element types zip into a struct, which has no dim")
        if len(elements) == 1:
            extents = [None] * ndim
            extents.insert(position, len(fields))
            spec = ArraySpec(self.into, ndim + 1, extents=tuple(extents), element=elements.pop())
            zipped = ArrayState(spec, state.grid, state.where)
            zipped.fields, zipped.field_dim = tuple(self.arrays), position
        else:
            members = tuple((field.name, field.precision) for field in fields)
            zipped = ArrayState(ArraySpec(self.into, ndim), state.grid, state.where, members)
            zipped.member_elements = tuple(field.element for field in fields)
        zipped.location = locations.pop()
        state.arrays[self.into] = zipped


@operation
@dataclass(frozen=True, slots=True)
class Unzip(Operation):
    """Split the whole array ``array`` into the arrays ``into``, the inverse of :class:`Zip`: a struct into
    one array per member (a complex array is the struct ``(re, im)``), a zipped array along its field, or,
    with ``dim``, any array along dimension ``dim``, one array per index (its extent must be
    ``len(into)``: ``A[N, M, 4]`` into four ``[N, M]`` arrays). Each keeps the precision and placement it
    had."""

    action: ClassVar[Action] = Action.UNZIP
    lowers: ClassVar[bool] = False
    array: ArrayName
    into: ArrayNames
    dim: Dim | None = None

    def apply(self, state: State) -> None:
        whole = state.untouched(self.array, self)
        ndim = whole.digits.ndim
        if self.dim is not None:
            if self.dim >= ndim or (whole.members and not whole.native):
                raise ValueError(f"{state.where}: {self.array!r} has no dimension {self.dim} to split along")
            extent = whole.digits.digits[self.dim].extent
            parts = [(name, whole.precision, whole.element) for name in self.into]
            ndim -= 1
        elif whole.fields:
            extent, ndim = len(whole.fields), ndim - 1
            parts = [(name, whole.precision, whole.element) for name in self.into]
        else:
            component = ELEMENTS[whole.element].component if whole.element in ELEMENTS else None
            elements = whole.member_elements or (component,) * len(whole.members)
            extent, parts = (
                len(whole.members),
                [(name, p, e) for name, (_, p), e in zip(self.into, whole.members, elements, strict=False)],
            )
            if not whole.members:
                raise ValueError(
                    f"{state.where}: {self.array!r} is neither a struct nor zipped; name the dim to split along"
                )
        if extent is not None and extent != len(self.into):
            raise ValueError(f"{state.where}: {self.array!r} splits into {extent} array(s); name one per part")
        del state.arrays[self.array]
        for name, precision, element in parts:
            state.fresh(name)
            state.arrays[name] = ArrayState(ArraySpec(name, ndim, element=element), state.grid, state.where)
            state.arrays[name].precision = precision
            state.arrays[name].location = whole.location


class DigitOperation(Operation):
    """An operation on one array's digits; the inverse of each is another digit operation."""

    __slots__ = ()
    array: ArrayName


@operation
@dataclass(frozen=True, slots=True)
class Block(DigitOperation):
    """Cut digit ``digit`` ``(d, s, e)`` into ``into``: coarse ``(d, s*b, ceil(e/b))`` and fine ``(d, s, b)``."""

    action: ClassVar[Action] = Action.BLOCK
    array: ArrayName
    digit: DigitName
    factor: Expr
    into: DigitPair

    def apply(self, state: State) -> None:
        digits = state.digits(self.array, self)
        digits.block(self.digit, digits.eval(self.factor), self.into)

    def inverse(self, before: tuple[str, ...]) -> "Unblock":
        return Unblock(self.array, self.into, self.factor, self.digit)


@operation
@dataclass(frozen=True, slots=True)
class Unblock(DigitOperation):
    """Merge the coarse and fine digits ``digits`` of a ``block`` by ``factor`` back into one digit ``into``."""

    action: ClassVar[Action] = Action.UNBLOCK
    array: ArrayName
    digits: DigitPair
    factor: Expr
    into: DigitName

    def apply(self, state: State) -> None:
        digits = state.digits(self.array, self)
        fine = digits.digits[digits.find(self.digits[1])]
        factor = digits.eval(self.factor)
        if factor is not None and fine.extent != factor:
            raise ValueError(f"{state.where}: {self.digits[1]!r} has extent {fine.extent}, not the factor {factor}")
        digits.merge(self.digits, self.into, band=False)

    def inverse(self, before: tuple[str, ...]) -> Block:
        return Block(self.array, self.into, self.factor, self.digits)


@operation
@dataclass(frozen=True, slots=True)
class Split(DigitOperation):
    """Cut a whole dimension's digit ``digit`` into ``parts`` contiguous load-balanced bands (sizes differ
    by at most one): ``into`` names the band index (extent ``parts``) and the index within the band."""

    action: ClassVar[Action] = Action.SPLIT
    array: ArrayName
    digit: DigitName
    parts: Expr
    into: DigitPair

    def apply(self, state: State) -> None:
        digits = state.digits(self.array, self)
        digits.split(self.digit, digits.eval(self.parts), self.into)

    def inverse(self, before: tuple[str, ...]) -> "Unsplit":
        return Unsplit(self.array, self.into, self.parts, self.digit)


@operation
@dataclass(frozen=True, slots=True)
class Unsplit(DigitOperation):
    """Merge the band index and the index within the band ``digits`` of a ``split`` back into one digit."""

    action: ClassVar[Action] = Action.UNSPLIT
    array: ArrayName
    digits: DigitPair
    parts: Expr
    into: DigitName

    def apply(self, state: State) -> None:
        state.digits(self.array, self).merge(self.digits, self.into, band=True)

    def inverse(self, before: tuple[str, ...]) -> Split:
        return Split(self.array, self.into, self.parts, self.digits)


@operation
@dataclass(frozen=True, slots=True)
class Permute(DigitOperation):
    """Reorder array ``array``'s digit tuple to ``order``, every current digit named once."""

    action: ClassVar[Action] = Action.PERMUTE
    array: ArrayName
    order: DigitNames

    def apply(self, state: State) -> None:
        digits = state.digits(self.array, self)
        if sorted(self.order) != sorted(digits.order()):
            raise ValueError(
                f"{state.where}: permute must name every digit once; the digits are {list(digits.order())}"
            )
        digits.digits[:] = [digits.digits[digits.find(name)] for name in self.order]

    def inverse(self, before: tuple[str, ...]) -> "Permute":
        return Permute(self.array, DigitNames(before))


@operation
@dataclass(frozen=True, slots=True)
class Pad(DigitOperation):
    """Grow digit ``digit`` by ``amount`` dead cells."""

    action: ClassVar[Action] = Action.PAD
    lowers: ClassVar[bool] = False
    array: ArrayName
    digit: DigitName
    amount: Expr

    def apply(self, state: State) -> None:
        digits = state.digits(self.array, self)
        digits.pad(self.digit, digits.eval(self.amount))

    def inverse(self, before: tuple[str, ...]) -> "Unpad":
        return Unpad(self.array, self.digit, self.amount)


@operation
@dataclass(frozen=True, slots=True)
class Unpad(DigitOperation):
    """Drop ``amount`` dead cells from digit ``digit``."""

    action: ClassVar[Action] = Action.UNPAD
    lowers: ClassVar[bool] = False
    array: ArrayName
    digit: DigitName
    amount: Expr

    def apply(self, state: State) -> None:
        digits = state.digits(self.array, self)
        amount = digits.eval(self.amount)
        digits.pad(self.digit, None if amount is None else -amount)

    def inverse(self, before: tuple[str, ...]) -> Pad:
        return Pad(self.array, self.digit, self.amount)


@operation
@dataclass(frozen=True, slots=True)
class Shuffle(DigitOperation):
    """Renumber the digits ``digits`` by the bijection ``f`` (one expression per digit, over their names);
    the caller supplies its inverse ``f_inv``, checked exhaustively where the digits' extents are known
    (otherwise the kernels bound every value ``f_inv`` gives back, so a wrong one cannot leave the buffer)."""

    action: ClassVar[Action] = Action.SHUFFLE
    lowers: ClassVar[bool] = False
    array: ArrayName
    digits: DigitNames
    f: IndexExprs
    f_inv: IndexExprs

    def validate(self) -> None:
        if not len(self.f) == len(self.f_inv) == len(self.digits):
            raise ValueError(f"f and f_inv need one expression per shuffled digit ({len(self.digits)})")

    def apply(self, state: State) -> None:
        state.digits(self.array, self).renumber(self.digits, self.f, self.f_inv)

    def inverse(self, before: tuple[str, ...]) -> "Shuffle":
        return Shuffle(self.array, self.digits, self.f_inv, self.f)


@operation
@dataclass(frozen=True, slots=True)
class LinearShuffle(DigitOperation):
    """Renumber the digits ``digits`` (extents ``2**bits``) by an invertible linear map over GF(2) on their
    bits (Linear Layouts, arXiv:2505.23819): the input bit vector is each digit's bits, least significant
    first, in ``digits`` order. XOR swizzles are this form; the inverse is the inverse matrix."""

    action: ClassVar[Action] = Action.LINEAR_SHUFFLE
    lowers: ClassVar[bool] = False
    array: ArrayName
    digits: DigitNames
    bits: Bits
    matrix: BitMatrix

    def validate(self) -> None:
        if len(self.bits) != len(self.digits):
            raise ValueError(f"bits needs one width per shuffled digit ({len(self.digits)})")
        if len(self.matrix) != sum(self.bits):
            raise ValueError(f"the matrix must be {sum(self.bits)} x {sum(self.bits)} (sum of bits)")
        if gf2_inverse(self.matrix) is None:
            raise ValueError("the matrix is singular over GF(2), so the shuffle is not a bijection")

    def apply(self, state: State) -> None:
        digits = state.digits(self.array, self)
        for name, width in zip(self.digits, self.bits, strict=True):
            extent = digits.digits[digits.find(name)].extent
            if digits.grid is not None and extent != 2**width:
                raise ValueError(f"{state.where}: {name!r} has extent {extent}, not 2**{width}")
        inverse = gf2_inverse(self.matrix) or ()
        digits.renumber(
            self.digits, xor_exprs(self.digits, self.bits, self.matrix), xor_exprs(self.digits, self.bits, inverse)
        )

    def inverse(self, before: tuple[str, ...]) -> "LinearShuffle":
        inverse = gf2_inverse(self.matrix)
        if inverse is None:
            raise ValueError("a singular GF(2) matrix has no inverse")
        return LinearShuffle(self.array, self.digits, self.bits, BitMatrix(inverse))


@operation
@dataclass(frozen=True, slots=True)
class Distribute(DigitOperation):
    """Hand a digit of extent ``P_g`` to grid dimension ``grid_dim``: it becomes the rank coordinate and
    leaves the local tuple. Name the ``digit``, or use the short form ``dim`` + ``scheme`` (+ ``block_size``
    for ``block_cyclic``), which cuts dimension ``dim``'s whole digit and distributes the rank digit
    ``r<dim>``: ``block`` = contiguous balanced bands, ``cyclic`` = round robin, ``block_cyclic`` = round robin
    of blocks of ``block_size``."""

    action: ClassVar[Action] = Action.DISTRIBUTE
    array: ArrayName
    grid_dim: GridDim
    digit: DigitName | None = None
    dim: Dim | None = None
    scheme: Scheme | None = None
    block_size: Width | None = None

    def validate(self) -> None:
        if (self.digit is None) == (self.dim is None and self.scheme is None):
            raise ValueError("name either the digit, or the dim and scheme of the short form")
        if self.digit is None and (self.dim is None or self.scheme is None):
            raise ValueError("the short form needs both dim and scheme")
        if (self.block_size is not None) != (self.scheme is Scheme.BLOCK_CYCLIC):
            raise ValueError("block_size goes with scheme block_cyclic, and only with it")

    def apply(self, state: State) -> None:
        if self.digit is not None:
            state.digits(self.array, self).distribute(self.digit, self.grid_dim)
            return
        for op in self.expanded(state):
            op.apply(state)

    def expanded(self, state: State) -> list[DigitOperation]:
        """The short form's primitive operations on dimension ``dim``'s whole digit."""
        dim = self.dim if self.dim is not None else 0
        digits = state.digits(self.array, self)
        whole = [d.name for d in digits.digits if d.dim == dim and d.extent is None and d.stride == 1 and not d.band]
        own = [d for d in digits.digits if d.dim == dim]
        if dim >= digits.ndim or len(own) != 1 or not whole:
            raise ValueError(f"{state.where}: the short form distributes a whole dimension; dimension {dim} is not one")
        name, rank, symbol = whole[0], f"r{dim}", GRID_SYMBOL.format(self.grid_dim)
        array, into = ArrayName(self.array), DigitName(rank)
        if self.scheme is Scheme.BLOCK:
            cuts: list[DigitOperation] = [Split(array, DigitName(name), Expr(symbol), DigitPair((rank, name)))]
        elif self.scheme is Scheme.CYCLIC:
            cuts = [Block(array, DigitName(name), Expr(symbol), DigitPair((name, rank)))]
        else:
            cycle, width = f"c{dim}", self.block_size or 1
            cuts = [
                Block(array, DigitName(name), Expr(f"{width}*{symbol}"), DigitPair((name, cycle))),
                Block(array, DigitName(cycle), Expr(str(width)), DigitPair((rank, f"b{dim}"))),
            ]
        return [*cuts, Distribute(array, self.grid_dim, digit=into)]

    def inverse(self, before: tuple[str, ...]) -> NoReturn:
        raise ValueError("a distribute is undone by the gather; invert the operations before it")


@operation
@dataclass(frozen=True, slots=True)
class Replicate(Operation):
    """Hold array ``array`` whole on every rank."""

    action: ClassVar[Action] = Action.REPLICATE
    sparse: ClassVar[bool] = True
    array: ArrayName

    def apply(self, state: State) -> None:
        target = state.array(self.array, self)
        if target.digits.distributed:
            raise ValueError(f"{state.where}: {self.array!r} is distributed; it cannot also be replicated")
        if not target.digits.whole():
            raise ValueError(
                f"{state.where}: {self.array!r} is laid out ({list(target.digits.order())}); a replicated array "
                "keeps its layout, so replicate it before any digit operation"
            )
        target.replicated = True
        target.touched = True


@operation
@dataclass(frozen=True, slots=True)
class Place(Operation):
    """Keep array ``array`` in ``on`` memory (``host`` or ``device``) during the call."""

    action: ClassVar[Action] = Action.PLACE
    sparse: ClassVar[bool] = True
    array: ArrayName
    on: Location

    def apply(self, state: State) -> None:
        state.array(self.array, self).location = self.on


def gf2_inverse(matrix: Sequence[Sequence[int]]) -> tuple[tuple[int, ...], ...] | None:
    """The inverse of a square 0/1 matrix over GF(2) by Gauss-Jordan elimination, or None when singular."""
    n = len(matrix)
    rows = [[*row, *(int(k == r) for k in range(n))] for r, row in enumerate(matrix)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if rows[r][col]), None)
        if pivot is None:
            return None
        rows[col], rows[pivot] = rows[pivot], rows[col]
        for r in range(n):
            if r != col and rows[r][col]:
                rows[r] = [a ^ b for a, b in zip(rows[r], rows[col], strict=True)]
    return tuple(tuple(row[n:]) for row in rows)


def xor_exprs(names: Sequence[str], bits: Sequence[int], matrix: Sequence[Sequence[int]]) -> tuple[str, ...]:
    """The values of digits ``names`` (``bits`` wide) after the GF(2) map ``matrix`` on their bits, least
    significant first in ``names`` order: one expression per digit."""
    inputs = [f"(({name} >> {b}) & 1)" for name, width in zip(names, bits, strict=True) for b in range(width)]
    outputs = []
    for row in matrix:
        terms = [inputs[j] for j, bit in enumerate(row) if bit]
        outputs.append(" ^ ".join(terms) if terms else "0")
    exprs, start = [], 0
    for width in bits:
        parts = [f"(({outputs[start + b]}) << {b})" if b else f"({outputs[start]})" for b in range(width)]
        exprs.append(" | ".join(parts) if parts else "0")
        start += width
    return tuple(exprs)


#: The largest digit space a shuffle's ``f_inv(f(x)) == x`` is checked over exhaustively.
BIJECTION_CHECK_LIMIT = 1 << 20


def check_bijection(
    names: Sequence[str], extents: Sequence[int], f: Sequence[str], f_inv: Sequence[str], where: str
) -> None:
    """Refuse ``f`` unless it maps the digit space of ``extents`` into itself and ``f_inv`` undoes it."""
    for values in itertools.product(*(range(extent) for extent in extents)):
        bound = dict(zip(names, values, strict=True))
        image = [eval_index(expr, bound, where) for expr in f]
        if any(not 0 <= v < e for v, e in zip(image, extents, strict=True)):
            raise ValueError(f"{where}: f maps {list(values)} to {image}, outside the digits' extents {list(extents)}")
        back = [eval_index(expr, dict(zip(names, image, strict=True)), where) for expr in f_inv]
        if back != list(values):
            raise ValueError(f"{where}: f_inv(f({list(values)})) = {back}; f_inv must undo f")


# ---- defaults and the request ------------------------------------------------------------------------


def normal_split(axis: AxisSplit) -> AxisSplit:
    """The split :func:`lower` gives back for ``axis``: block and cyclic carry width 1, and a block-cyclic
    of width 1 is cyclic."""
    scheme = Scheme.CYCLIC if axis.scheme is Scheme.BLOCK_CYCLIC and axis.block_size == 1 else axis.scheme
    return AxisSplit(axis.grid_dim, scheme, axis.block_size if scheme is Scheme.BLOCK_CYCLIC else 1)


def operations_for_axes(array: str, axes: Sequence[AxisSplit | None]) -> tuple[Operation, ...]:
    """The operations whose lowering is ``axes`` for ``array`` (None = not split; all None = replicated):
    one short-form ``distribute`` per split dimension, lowered back on a grid of 2s as a self-check."""
    normal = [None if axis is None else normal_split(axis) for axis in axes]
    if all(axis is None for axis in normal):
        return (Replicate(ArrayName(array)),)
    ops = tuple(
        Distribute(
            ArrayName(array),
            GridDim(axis.grid_dim),
            dim=Dim(dim),
            scheme=axis.scheme,
            block_size=Width(axis.block_size) if axis.scheme is Scheme.BLOCK_CYCLIC else None,
        )
        for dim, axis in enumerate(normal)
        if axis is not None
    )
    sample = [2] * (1 + max(axis.grid_dim for axis in normal if axis is not None))
    lowered = lower(run(ops, [ArraySpec(array, len(axes))], sample, "default layout"))
    if lowered[array] != normal:
        raise AssertionError(f"operations_for_axes({array!r}, {list(axes)}) lowers to {lowered[array]}")
    return ops


@dataclass(frozen=True, slots=True)
class Startup:
    """What a request settles before any array is touched: the libraries to ``load`` and the processor
    ``grid``'s extents over the rank count ``P`` (None = not a distributed run)."""

    load: tuple[str, ...] = ()
    grid: tuple[str, ...] | None = None


@dataclass(frozen=True, slots=True)
class Requests:
    """A request: its :class:`Startup` and its ``operations``, run top to bottom (:func:`run`)."""

    startup: Startup = field(default_factory=Startup)
    operations: tuple[Operation, ...] = ()

    @property
    def distributed(self) -> bool:
        """Whether this asks for a distributed run: it names a grid."""
        return self.startup.grid is not None

    def grid_at(self, ranks: int) -> tuple[int, ...] | None:
        """The grid's extents at ``ranks`` ranks; None when there is no grid or it does not span ``ranks``."""
        grid = self.startup.grid
        return None if grid is None else grid_extents(grid, ranks, "startup.grid")

    def run(self, arrays: Sequence[ArraySpec], ranks: int | None = None) -> State:
        """The operations run over ``arrays``: on the grid at ``ranks`` ranks; without ``ranks``, names and
        shapes only for a distributed request (its factors read the grid) and evaluated for a single-node one."""
        grid: tuple[int, ...] | None = None if self.distributed else ()
        if ranks is not None:
            if self.startup.grid is None:
                raise ValueError("operations distribute nothing without startup.grid")
            grid = self.grid_at(ranks)
            if grid is None:
                raise ValueError(f"startup.grid {list(self.startup.grid)} does not span {ranks} rank(s)")
        return run(self.operations, arrays, grid, "operations")

    def named(self) -> set[str]:
        """Every array name an operation names."""
        names: set[str] = set()
        for op in self.operations:
            values = dataclasses.asdict(op)  # type: ignore[call-overload]  # every operation is a dataclass
            for spec in dataclasses.fields(op):  # type: ignore[arg-type]
                if spec.type is ArrayName:
                    names.add(values[spec.name])
                elif spec.type is ArrayNames:
                    names.update(values[spec.name])
        return names

    def precisions(self) -> dict[str, Precision]:
        """Each array a ``cast`` names, with the precision its last cast leaves."""
        return {op.array: op.to for op in self.operations if isinstance(op, Cast)}

    def formats(self) -> dict[str, str]:
        """Each sparse array a ``format`` names, with the format its last one leaves."""
        return {op.array: op.to for op in self.operations if isinstance(op, Format)}

    def locations(self) -> dict[str, Location]:
        """Each array a ``place`` names, with the memory its last one leaves."""
        return {op.array: op.on for op in self.operations if isinstance(op, Place)}
