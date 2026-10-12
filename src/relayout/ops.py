# Copyright 2026 ETH Zurich and the relayout authors.
"""Layout op sequences: the request schema of a relayout and its lowering to a distribution.

A layout is the mixed-radix digit algebra of dace extended ``dace/libraries/layout/algebra.py``. An
array starts as one digit per dimension, named ``i0, i1, ...``; every op addresses digits by name:
``block`` and ``split`` cut a digit into two named ones, ``permute`` reorders the tuple by names and
``distribute`` hands a digit of extent ``P_g`` to grid dimension ``g``. Dimensions are numpy's, over a
C-contiguous buffer; a Fortran kernel sees it with reversed extents, a codegen step after the ops.
:func:`lower_layout` lowers a sequence to one :class:`AxisSplit` per dimension; a sequence it does not
cover yet (pad, shuffle, a non-row-major local order) is refused by name, never approximated.

Every per-array op is one class registered under its JSON name (:func:`layout_op`) with its inverse
and its effect on the :class:`DigitTuple`; a field's type (:data:`FIELD_CODECS`) is how it is read from
and written to JSON, so adding an op is one class. ``zip`` / ``unzip`` turn several arrays into one or
one into several, so they are group ops (:data:`GROUP_OPS`) outside any array's sequence.

The processor grid is a list of extent expressions over the rank count ``P`` (:func:`grid_extents`), so
one request is evaluated at every rank count a scaling run measures.
"""

import ast
import dataclasses
import enum
import math
import operator
import typing
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any, ClassVar, NamedTuple, NewType, NoReturn

__all__ = [
    "DIM_DIGIT",
    "FIELD_CODECS",
    "GRID_SYMBOL",
    "GROUP_OPS",
    "LAYOUT_OPS",
    "RANKS_SYMBOL",
    "AxisSplit",
    "Block",
    "Digit",
    "DigitTuple",
    "Distribute",
    "GroupOp",
    "LayoutOp",
    "LinearShuffle",
    "Pad",
    "Permute",
    "Registry",
    "RegistryError",
    "Scheme",
    "Shuffle",
    "Split",
    "Unblock",
    "Unpad",
    "Unsplit",
    "Unzip",
    "Zip",
    "check_index_expr",
    "eval_index",
    "expr_json",
    "from_json",
    "gf2_inverse",
    "grid_extents",
    "grid_names",
    "group_op",
    "groups_from_json",
    "groups_json",
    "inverse_sequence",
    "layout_op",
    "lower_layout",
    "op_json",
    "ops_for_axes",
    "parse_grid",
    "parse_op",
    "to_json",
]

#: Grid dimension ``g``'s extent in a factor/amount expression: ``P0``, ``P1``, ...
GRID_SYMBOL = "P{}"
#: The rank count in a grid extent expression.
RANKS_SYMBOL = "P"
#: The name of dimension ``d``'s digit before any op.
DIM_DIGIT = "i{}"


class Scheme(enum.Enum):
    """The per-axis split a layout lowers to; the value is its JSON spelling."""

    BLOCK = "block"
    BLOCK_CYCLIC = "block_cyclic"
    CYCLIC = "cyclic"


def grid_names(grid: Sequence[int]) -> dict[str, int]:
    """``{"P0": grid[0], ...}``: the names a factor expression reads."""
    return {GRID_SYMBOL.format(g): int(p) for g, p in enumerate(grid)}


# ---- index expressions -----------------------------------------------------------------------------

#: The operators an index expression may use: integer arithmetic, floor and mod, and the bitwise ones
#: XOR swizzles need.
BINOP_FUNCTIONS: dict[type[ast.operator], Callable[[int, int], int]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.BitXor: operator.xor,
    ast.BitAnd: operator.and_,
    ast.BitOr: operator.or_,
    ast.LShift: operator.lshift,
    ast.RShift: operator.rshift,
}


def int_root(value: int, degree: int) -> int:
    """The floor of the ``degree``-th root of ``value`` (``root(P, 2)`` is a square grid's edge)."""
    if value < 0 or degree < 1:
        raise ValueError(f"root({value}, {degree}) needs a non-negative value and a degree >= 1")
    guess = round(value ** (1 / degree))
    while guess**degree > value:
        guess -= 1
    while (guess + 1) ** degree <= value:
        guess += 1
    return guess


#: Function name -> (arity, implementation).
INDEX_CALLS: dict[str, tuple[int, Callable[..., int]]] = {
    "min": (2, min),
    "max": (2, max),
    "root": (2, int_root),
}


def check_index_expr(text: str, where: str) -> str:
    """``text`` when it is an integer expression of the allowed form, else a request fault."""
    try:
        tree = ast.parse(text, mode="eval")
    except SyntaxError:
        raise ValueError(f"{where}: {text!r} is not an expression") from None
    for walked in ast.walk(tree.body):
        node = walked.op if isinstance(walked, (ast.BinOp, ast.UnaryOp)) else walked
        allowed = (
            isinstance(node, (ast.Name, ast.Load, ast.USub, *BINOP_FUNCTIONS))
            or (isinstance(node, ast.Constant) and isinstance(node.value, int) and not isinstance(node.value, bool))
            or (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name)
                and node.func.id in INDEX_CALLS
                and len(node.args) == INDEX_CALLS[node.func.id][0]
                and not node.keywords
            )
        )
        if not allowed:
            raise ValueError(
                f"{where}: {text!r} uses {type(node).__name__}; an index expression takes ints, names, "
                f"+ - * // % ^ & | << >> and min(a, b), max(a, b), root(a, k)"
            )
    return text


def eval_index(text: str, names: Mapping[str, int], where: str) -> int:
    """The value of the index expression ``text`` (:func:`check_index_expr`'s form) with ``names`` bound;
    ``//`` and ``%`` floor like Python's."""

    def value(node: ast.expr) -> int:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return node.value
        if isinstance(node, ast.Name):
            if node.id not in names:
                raise ValueError(f"{where}: {text!r} names {node.id!r}, which is not bound here ({sorted(names)})")
            return names[node.id]
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in BINOP_FUNCTIONS:
            return BINOP_FUNCTIONS[type(node.op)](value(node.left), value(node.right))
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in INDEX_CALLS:
            return INDEX_CALLS[node.func.id][1](*(value(arg) for arg in node.args))
        raise ValueError(f"{where}: {text!r} is not an index expression")

    return value(ast.parse(check_index_expr(text, where), mode="eval").body)


# ---- the processor grid ------------------------------------------------------------------------------


def parse_grid(raw: object, where: str) -> tuple[str, ...]:
    """A grid request: a non-empty list of extent expressions over ``P`` (ints allowed)."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where} must be a non-empty list of extents over {RANKS_SYMBOL}; got {raw!r}")
    return tuple(parse_expr(extent, f"{where}[{g}]") for g, extent in enumerate(raw))


def grid_extents(grid: Sequence[str], ranks: int, where: str) -> tuple[int, ...] | None:
    """The grid ``grid`` spans at ``ranks`` ranks, or None when it does not span exactly that many (a
    square grid at an odd rank count); an extent below 1 is a request fault."""
    extents = tuple(eval_index(extent, {RANKS_SYMBOL: ranks}, f"{where}[{g}]") for g, extent in enumerate(grid))
    if any(extent < 1 for extent in extents):
        raise ValueError(f"{where} {list(grid)} has an extent below 1 at {RANKS_SYMBOL} = {ranks}: {list(extents)}")
    return extents if math.prod(extents) == ranks else None


# ---- op fields: one JSON codec per field type -----------------------------------------------------

#: A digit's name: an identifier other than the grid and rank symbols.
DigitName = NewType("DigitName", str)
#: Two or more distinct digit names (a permutation, the digits a shuffle renumbers).
DigitNames = NewType("DigitNames", tuple[str, ...])
#: The two digits a cut leaves, coarse then fine.
DigitPair = NewType("DigitPair", tuple[str, str])
#: A processor-grid dimension.
GridDim = NewType("GridDim", int)
#: An integer expression over ``P0, P1, ...`` (a factor, an amount, a part count).
Expr = NewType("Expr", str)
#: One index expression per shuffled digit, over the shuffled digits' names and ``P0, P1, ...``.
IndexExprs = NewType("IndexExprs", tuple[str, ...])
#: The bit width of each shuffled digit (its extent is ``2**bits``).
Bits = NewType("Bits", tuple[int, ...])
#: A square 0/1 matrix over GF(2): row ``r`` is output bit ``r`` as the XOR of the input bits it marks.
BitMatrix = NewType("BitMatrix", tuple[tuple[int, ...], ...])


def is_reserved(name: str) -> bool:
    """Whether ``name`` is the rank symbol or a grid symbol ``P<g>``."""
    return name == RANKS_SYMBOL or (name[:1] == RANKS_SYMBOL and name[1:].isdigit())


def parse_digit_name(value: object, where: str) -> str:
    if not isinstance(value, str) or not value.isidentifier() or is_reserved(value):
        raise ValueError(f"{where} must be a digit name (an identifier other than P, P0, P1, ...); got {value!r}")
    return value


def parse_digit_names(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of digit names; got {value!r}")
    names = tuple(parse_digit_name(v, f"{where}[{k}]") for k, v in enumerate(value))
    if len(set(names)) != len(names):
        raise ValueError(f"{where} names a digit twice: {list(names)}")
    return names


def parse_digit_pair(value: object, where: str) -> tuple[str, ...]:
    names = parse_digit_names(value, where)
    if len(names) != len(("coarse", "fine")):
        raise ValueError(f"{where} must name the two digits the cut leaves, coarse then fine; got {list(names)}")
    return names


def non_negative_int(value: object, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{where} must be a non-negative int; got {value!r}")
    return value


def int_list(value: object, where: str) -> tuple[int, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{where} must be a non-empty list of non-negative ints; got {value!r}")
    return tuple(non_negative_int(v, f"{where}[{k}]") for k, v in enumerate(value))


def parse_expr(value: object, where: str) -> str:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise ValueError(f"{where} must be an int or an integer expression; got {value!r}")
    return check_index_expr(str(value), where)


def parse_index_exprs(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or not all(isinstance(v, str) for v in value):
        raise ValueError(f"{where} must be a list of index expressions over the shuffled digits; got {value!r}")
    return tuple(check_index_expr(v, f"{where}[{k}]") for k, v in enumerate(value))


def parse_bit_matrix(value: object, where: str) -> tuple[tuple[int, ...], ...]:
    rows = value if isinstance(value, list) else None
    if not rows or not all(isinstance(r, list) and len(r) == len(rows) and set(r) <= {0, 1} for r in rows):
        raise ValueError(f"{where} must be a square 0/1 matrix (a list of rows); got {value!r}")
    return tuple(tuple(int(b) for b in r) for r in rows)


def expr_json(text: str) -> int | str:
    """``text`` back as JSON: a plain integer literal as an int."""
    return int(text) if text.lstrip("-").isdigit() else text


def plain_json(value: object) -> object:
    """A tuple field as a JSON list (nested for a matrix)."""
    return [plain_json(v) for v in value] if isinstance(value, tuple) else value


class FieldCodec(NamedTuple):
    """How one field type is read from JSON (a request fault naming ``where`` when malformed) and written."""

    parse: Callable[[object, str], Any]
    dump: Callable[[Any], object] = plain_json


#: Field type -> codec. A registered op may only declare fields of these types (checked at import).
FIELD_CODECS: dict[object, FieldCodec] = {
    DigitName: FieldCodec(parse_digit_name),
    DigitNames: FieldCodec(parse_digit_names),
    DigitPair: FieldCodec(parse_digit_pair),
    GridDim: FieldCodec(non_negative_int),
    Expr: FieldCodec(parse_expr, expr_json),
    IndexExprs: FieldCodec(parse_index_exprs),
    Bits: FieldCodec(int_list),
    BitMatrix: FieldCodec(parse_bit_matrix),
}


# ---- the layout op registry ------------------------------------------------------------------------


class LayoutOp:
    """One op of an array's layout sequence. A subclass is a frozen slotted dataclass registered with
    :func:`layout_op`; it names its JSON ``op``, applies itself to a :class:`DigitTuple` and inverts itself
    given the digit names before it. ``lowers`` False = parsed and checked, refused at lowering as not
    implemented yet."""

    __slots__ = ()
    op: ClassVar[str]
    lowers: ClassVar[bool] = True

    def validate(self, where: str) -> None:
        """Checks across fields, past each field's own codec (a request fault naming ``where``)."""

    def apply(self, digits: "DigitTuple") -> None:
        raise NotImplementedError

    def inverse(self, before: tuple[str, ...]) -> "LayoutOp":
        """The op undoing this one, given the digit names before it."""
        raise NotImplementedError


class RegistryError(ValueError):
    """An op class breaks the registry's contract, or its name is taken."""


class Registry[T]:
    """Op name -> op class, in registration order."""

    __slots__ = ("entries", "name")

    def __init__(self, name: str) -> None:
        self.name = name
        self.entries: dict[str, T] = {}

    def add(self, key: str, entry: T) -> None:
        if key in self.entries:
            raise RegistryError(f"{self.name} {key!r} is registered twice")
        self.entries[key] = entry

    def keys(self) -> tuple[str, ...]:
        return tuple(self.entries)


#: JSON ``op`` name -> the per-array layout op class.
LAYOUT_OPS: Registry[type[LayoutOp]] = Registry("layout op")


def op_fields(cls: type) -> dict[str, FieldCodec]:
    """``cls``'s dataclass fields with their codecs; a field of an unregistered type is refused."""
    hints = typing.get_type_hints(cls)
    codecs: dict[str, FieldCodec] = {}
    for spec in dataclasses.fields(cls):
        codec = FIELD_CODECS.get(hints[spec.name])
        if codec is None:
            raise RegistryError(f"{cls.__name__}.{spec.name}: field type {hints[spec.name]!r} has no JSON codec")
        codecs[spec.name] = codec
    return codecs


def layout_op[C: type[LayoutOp]](cls: C) -> C:
    """Register a layout op class under its ``op`` name; it must define its own ``apply`` and ``inverse``."""
    op_fields(cls)
    missing = [method for method in ("apply", "inverse") if method not in vars(cls)]
    if missing:
        raise RegistryError(f"layout op {cls.op!r} must define {' and '.join(missing)}")
    LAYOUT_OPS.add(cls.op, cls)
    return cls


def from_json[T](cls: type[T], raw: Mapping[str, object], where: str) -> T:
    """An op of ``cls`` from its JSON object (the ``op`` key names the op, it is not a field)."""
    codecs = op_fields(cls)
    unknown = sorted(set(raw) - set(codecs) - {"op"})
    if unknown:
        raise ValueError(f"{where} has unknown field(s) {unknown}; its fields are {sorted(codecs)}")
    return cls(**{name: codec.parse(raw.get(name), f"{where}.{name}") for name, codec in codecs.items()})


def to_json(op: LayoutOp) -> dict[str, object]:
    """``op``'s fields as JSON (without the ``op`` key)."""
    values = dataclasses.asdict(op)  # type: ignore[call-overload]  # every registered op is a dataclass
    return {name: codec.dump(values[name]) for name, codec in op_fields(type(op)).items()}


def parse_op(raw: object, where: str) -> LayoutOp:
    """One layout op from its JSON object; a malformed op is a request fault naming ``where``."""
    if not isinstance(raw, dict) or not isinstance(raw.get("op"), str):
        raise ValueError(f"{where} must be an object with an 'op' name; got {raw!r}")
    cls = LAYOUT_OPS.entries.get(raw["op"])
    if cls is None:
        raise ValueError(f"{where}.op {raw['op']!r} is not a layout op; ops are {', '.join(LAYOUT_OPS.keys())}")
    op = from_json(cls, raw, where)
    op.validate(where)
    return op


def op_json(op: LayoutOp) -> dict[str, object]:
    """``op`` as the JSON object :func:`parse_op` reads."""
    return {"op": op.op, **to_json(op)}


# ---- the digit tuple ---------------------------------------------------------------------------------


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
    """One array's digit tuple while its ops apply: the ops' only state. ``grid`` None = names and shapes
    only (an expression is not evaluated, its digit's extent stays unknown)."""

    __slots__ = ("digits", "distributed", "grid", "names", "ndim", "where")

    def __init__(self, ndim: int, grid: Sequence[int] | None, where: str) -> None:
        self.ndim = ndim
        self.grid = None if grid is None else tuple(grid)
        self.names = {} if grid is None else grid_names(grid)
        self.digits = [Digit(DIM_DIGIT.format(d), d, 1, None) for d in range(ndim)]
        #: (the distributed digit, its grid dim), in distribute order.
        self.distributed: list[tuple[Digit, int]] = []
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

    def find(self, name: str) -> int:
        for position, digit in enumerate(self.digits):
            if digit.name == name:
                return position
        raise ValueError(f"{self.where}: no digit {name!r}; the digits are {list(self.order())}")

    def replace(self, position: int, new: Sequence[Digit]) -> None:
        """Put ``new`` where the digit at ``position`` was; their names must not clash with the others'."""
        others = {digit.name for k, digit in enumerate(self.digits) if k != position}
        clash = sorted(others & {digit.name for digit in new})
        if clash:
            raise ValueError(f"{self.where}: digit name(s) {clash} are taken; the digits are {list(self.order())}")
        self.digits[position : position + 1] = list(new)

    def block(self, name: str, factor: int | None, into: tuple[str, str]) -> None:
        """Cut digit ``name`` ``(d, s, e)`` into coarse ``(d, s*b, ceil(e/b))`` and fine ``(d, s, b)``, in place."""
        position = self.find(name)
        old = self.digits[position]
        outer = None if old.extent is None or factor is None else math.ceil(old.extent / factor)
        coarse = Digit(into[0], old.dim, old.stride * (factor or 0), outer, old.band)
        self.replace(position, [coarse, Digit(into[1], old.dim, old.stride, factor)])

    def split(self, name: str, parts: int | None, into: tuple[str, str]) -> None:
        """Cut the whole dimension's digit ``name`` into the band index (extent ``parts``) and the index
        within the band, in place."""
        position = self.find(name)
        old = self.digits[position]
        if old.extent is not None or old.stride != 1 or old.band:
            raise ValueError(f"{self.where}: split cuts a whole dimension's digit; {name!r} is already cut")
        bands = Digit(into[0], old.dim, 1, parts, band=True)
        self.replace(position, [bands, Digit(into[1], old.dim, 1, None)])

    def merge(self, pair: tuple[str, str], into: str, *, band: bool) -> None:
        """Merge the coarse and fine digits ``pair`` of one dimension back into one digit ``into``."""
        coarse, fine = (self.digits[self.find(name)] for name in pair)
        if coarse.dim != fine.dim or coarse.band != band:
            raise ValueError(f"{self.where}: {list(pair)} are not the two digits of one {'split' if band else 'block'}")
        extent = None if band or coarse.extent is None or fine.extent is None else coarse.extent * fine.extent
        del self.digits[self.find(fine.name)]
        self.replace(self.find(coarse.name), [Digit(into, coarse.dim, fine.stride, extent)])

    def distribute(self, name: str, grid_dim: int) -> None:
        """Hand digit ``name`` (extent ``P_g``) to ``grid_dim``: it becomes the rank coordinate and leaves the tuple."""
        position = self.find(name)
        digit = self.digits[position]
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

    def lower(self, where: str) -> list[AxisSplit | None]:
        """Per dimension its split (None = not split); the rank-local digits must end row-major."""
        axes: list[AxisSplit | None] = [None] * self.ndim
        for digit, grid_dim in self.distributed:
            axes[digit.dim] = self.axis_split(digit, grid_dim, where)
        row_major = sorted(self.digits, key=Digit.coarseness)
        if row_major != self.digits:
            order = [digit.name for digit in row_major]
            raise ValueError(
                f"{where}: the rank-local digits {list(self.order())} are not row-major, which is a local relayout "
                f'and not implemented yet; append {{"op": "permute", "order": {order}}}'
            )
        return axes


def inverse_sequence(ops: Sequence[LayoutOp], ndim: int) -> tuple[LayoutOp, ...]:
    """The sequence that undoes ``ops`` on an ``ndim``-dimensional array: reversed, each op inverted."""
    digits = DigitTuple(ndim, None, "inverse")
    inverses: list[LayoutOp] = []
    for op in ops:
        before = digits.order()
        op.apply(digits)
        inverses.append(op.inverse(before))
    return tuple(reversed(inverses))


# ---- the layout ops ----------------------------------------------------------------------------------


@layout_op
@dataclass(frozen=True, slots=True)
class Block(LayoutOp):
    """Cut digit ``digit`` ``(d, s, e)`` into ``into``: coarse ``(d, s*b, ceil(e/b))`` and fine ``(d, s, b)``."""

    op: ClassVar[str] = "block"
    digit: DigitName
    factor: Expr
    into: DigitPair

    def apply(self, digits: DigitTuple) -> None:
        digits.block(self.digit, digits.eval(self.factor), self.into)

    def inverse(self, before: tuple[str, ...]) -> "Unblock":
        return Unblock(self.into, self.factor, self.digit)


@layout_op
@dataclass(frozen=True, slots=True)
class Unblock(LayoutOp):
    """Merge the coarse and fine digits ``digits`` of a ``block`` by ``factor`` back into one digit ``into``."""

    op: ClassVar[str] = "unblock"
    digits: DigitPair
    factor: Expr
    into: DigitName

    def apply(self, digits: DigitTuple) -> None:
        fine = digits.digits[digits.find(self.digits[1])]
        factor = digits.eval(self.factor)
        if factor is not None and fine.extent != factor:
            raise ValueError(f"{digits.where}: {self.digits[1]!r} has extent {fine.extent}, not the factor {factor}")
        digits.merge(self.digits, self.into, band=False)

    def inverse(self, before: tuple[str, ...]) -> Block:
        return Block(self.into, self.factor, self.digits)


@layout_op
@dataclass(frozen=True, slots=True)
class Split(LayoutOp):
    """Cut a whole dimension's digit ``digit`` into ``parts`` load-balanced contiguous bands (sizes differ by
    at most one): ``into`` names the band index (extent ``parts``) and the index within the band."""

    op: ClassVar[str] = "split"
    digit: DigitName
    parts: Expr
    into: DigitPair

    def apply(self, digits: DigitTuple) -> None:
        digits.split(self.digit, digits.eval(self.parts), self.into)

    def inverse(self, before: tuple[str, ...]) -> "Unsplit":
        return Unsplit(self.into, self.parts, self.digit)


@layout_op
@dataclass(frozen=True, slots=True)
class Unsplit(LayoutOp):
    """Merge the band index and the index within the band ``digits`` of a ``split`` back into one digit."""

    op: ClassVar[str] = "unsplit"
    digits: DigitPair
    parts: Expr
    into: DigitName

    def apply(self, digits: DigitTuple) -> None:
        digits.merge(self.digits, self.into, band=True)

    def inverse(self, before: tuple[str, ...]) -> Split:
        return Split(self.into, self.parts, self.digits)


@layout_op
@dataclass(frozen=True, slots=True)
class Permute(LayoutOp):
    """Reorder the digit tuple to ``order``, every current digit named once."""

    op: ClassVar[str] = "permute"
    order: DigitNames

    def apply(self, digits: DigitTuple) -> None:
        if sorted(self.order) != sorted(digits.order()):
            raise ValueError(
                f"{digits.where}: permute must name every digit once; the digits are {list(digits.order())}"
            )
        digits.digits[:] = [digits.digits[digits.find(name)] for name in self.order]

    def inverse(self, before: tuple[str, ...]) -> "Permute":
        return Permute(DigitNames(before))


@layout_op
@dataclass(frozen=True, slots=True)
class Pad(LayoutOp):
    """Grow digit ``digit`` by ``amount`` dead cells."""

    op: ClassVar[str] = "pad"
    lowers: ClassVar[bool] = False
    digit: DigitName
    amount: Expr

    def apply(self, digits: DigitTuple) -> None:
        digits.find(self.digit)

    def inverse(self, before: tuple[str, ...]) -> "Unpad":
        return Unpad(self.digit, self.amount)


@layout_op
@dataclass(frozen=True, slots=True)
class Unpad(LayoutOp):
    """Drop ``amount`` dead cells from digit ``digit``."""

    op: ClassVar[str] = "unpad"
    lowers: ClassVar[bool] = False
    digit: DigitName
    amount: Expr

    def apply(self, digits: DigitTuple) -> None:
        digits.find(self.digit)

    def inverse(self, before: tuple[str, ...]) -> Pad:
        return Pad(self.digit, self.amount)


@layout_op
@dataclass(frozen=True, slots=True)
class Shuffle(LayoutOp):
    """Renumber the digits ``digits`` by the bijection ``f`` (one expression per digit, over their names);
    the agent supplies its inverse ``f_inv``, checked exhaustively before the build."""

    op: ClassVar[str] = "shuffle"
    lowers: ClassVar[bool] = False
    digits: DigitNames
    f: IndexExprs
    f_inv: IndexExprs

    def validate(self, where: str) -> None:
        if not len(self.f) == len(self.f_inv) == len(self.digits):
            raise ValueError(f"{where}: f and f_inv need one expression per shuffled digit ({len(self.digits)})")

    def apply(self, digits: DigitTuple) -> None:
        for name in self.digits:
            digits.find(name)

    def inverse(self, before: tuple[str, ...]) -> "Shuffle":
        return Shuffle(self.digits, self.f_inv, self.f)


@layout_op
@dataclass(frozen=True, slots=True)
class LinearShuffle(LayoutOp):
    """Renumber the digits ``digits`` (extents ``2**bits``) by an invertible linear map over GF(2) on their
    bits (Linear Layouts, arXiv:2505.23819): the input bit vector is each digit's bits, least significant
    first, in ``digits`` order. XOR swizzles are this form; the inverse is the inverse matrix."""

    op: ClassVar[str] = "linear_shuffle"
    lowers: ClassVar[bool] = False
    digits: DigitNames
    bits: Bits
    matrix: BitMatrix

    def validate(self, where: str) -> None:
        if len(self.bits) != len(self.digits):
            raise ValueError(f"{where}: bits needs one width per shuffled digit ({len(self.digits)})")
        if len(self.matrix) != sum(self.bits):
            raise ValueError(f"{where}: the matrix must be {sum(self.bits)} x {sum(self.bits)} (sum of bits)")
        if gf2_inverse(self.matrix) is None:
            raise ValueError(f"{where}: the matrix is singular over GF(2), so the shuffle is not a bijection")

    def apply(self, digits: DigitTuple) -> None:
        for name in self.digits:
            digits.find(name)

    def inverse(self, before: tuple[str, ...]) -> "LinearShuffle":
        inverse = gf2_inverse(self.matrix)
        if inverse is None:
            raise ValueError("a singular GF(2) matrix has no inverse")
        return LinearShuffle(self.digits, self.bits, BitMatrix(inverse))


@layout_op
@dataclass(frozen=True, slots=True)
class Distribute(LayoutOp):
    """Hand digit ``digit`` (extent ``P_g``) to grid dimension ``grid_dim``: it becomes the rank coordinate
    and leaves the local tuple."""

    op: ClassVar[str] = "distribute"
    digit: DigitName
    grid_dim: GridDim

    def apply(self, digits: DigitTuple) -> None:
        digits.distribute(self.digit, self.grid_dim)

    def inverse(self, before: tuple[str, ...]) -> NoReturn:
        raise ValueError("a distribute is undone by the gather; invert the sequence before it")


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


# ---- the group ops -----------------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class GroupOp:
    """One op that turns several arrays into one or one into several. Requested as ``{name: [fields]}``
    under its own key; it has no inverse: each direction is its own request."""

    op: ClassVar[str]
    name: str
    fields: tuple[str, ...]


#: Request key -> the group op class.
GROUP_OPS: Registry[type[GroupOp]] = Registry("group op")


def group_op[C: type[GroupOp]](cls: C) -> C:
    """Register a group op class under its ``op`` name, which is also its request key."""
    GROUP_OPS.add(cls.op, cls)
    return cls


@group_op
@dataclass(frozen=True, slots=True)
class Zip(GroupOp):
    """Fuse ``fields`` (arrays of one logical shape) into one array ``name`` of C structs, one member per
    field in order, each at its own (requested) precision."""

    op: ClassVar[str] = "zip"


@group_op
@dataclass(frozen=True, slots=True)
class Unzip(GroupOp):
    """Split the struct array ``name`` into one array per member, named ``fields`` in member order. A
    complex array is the struct ``(re, im)``."""

    op: ClassVar[str] = "unzip"


def groups_from_json(key: str, raw: object, where: str) -> list[GroupOp]:
    """The ``{name: [fields]}`` object of group op ``key``."""
    cls = GROUP_OPS.entries[key]
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object {{name: [two or more arrays]}}; got {raw!r}")
    groups: list[GroupOp] = []
    for name, fields in raw.items():
        if not isinstance(fields, list) or len(fields) < 2 or not all(isinstance(f, str) and f for f in fields):
            raise ValueError(f"{where}[{name!r}] must be a list of two or more arrays; got {fields!r}")
        if len(set(fields)) != len(fields):
            raise ValueError(f"{where}[{name!r}] names an array twice: {fields!r}")
        groups.append(cls(str(name), tuple(fields)))
    return groups


def groups_json(groups: Sequence[GroupOp]) -> dict[str, dict[str, list[str]]]:
    """``groups`` as ``{key: {name: [fields]}}``, one entry per requested key."""
    out: dict[str, dict[str, list[str]]] = {}
    for group in groups:
        out.setdefault(group.op, {})[group.name] = list(group.fields)
    return out


# ---- lowering ----------------------------------------------------------------------------------------


def lower_layout(ops: Sequence[LayoutOp], ndim: int, grid: Sequence[int], where: str) -> list[AxisSplit | None]:
    """Lower one array's op sequence over ``grid`` to its per-dimension split (None = not split), or raise
    the request fault that says why it cannot be (yet).

    Lowerable: blocks, splits, their inverses and permutes, then distributes of a split's band index
    (``block``, contiguous bands) or of a digit of extent ``P_g`` sitting between a coarse digit of stride
    ``s*P_g`` and, when ``s > 1``, a fine digit of extent ``s`` (``block_cyclic`` of width ``s``; ``cyclic``
    at ``s == 1``). The rank-local digits must end in row-major order."""
    refused = next((k for k, op in enumerate(ops) if not op.lowers), None)
    if refused is not None:
        lowered = ", ".join(key for key in LAYOUT_OPS.keys() if LAYOUT_OPS.entries[key].lowers)
        raise ValueError(
            f"{where}[{refused}]: layout op {ops[refused].op!r} is not implemented yet (only {lowered} lower today)"
        )
    digits = DigitTuple(ndim, grid, where)
    for index, op in enumerate(ops):
        digits.where = f"{where}[{index}]"
        op.apply(digits)
    return digits.lower(where)


def normal_split(axis: AxisSplit) -> AxisSplit:
    """The split :func:`lower_layout` gives back for ``axis``: block and cyclic carry width 1, and a
    block-cyclic of width 1 is cyclic."""
    scheme = Scheme.CYCLIC if axis.scheme is Scheme.BLOCK_CYCLIC and axis.block_size == 1 else axis.scheme
    return AxisSplit(axis.grid_dim, scheme, axis.block_size if scheme is Scheme.BLOCK_CYCLIC else 1)


def ops_for_axes(axes: Sequence[AxisSplit | None]) -> tuple[LayoutOp, ...]:
    """The op sequence whose lowering is ``axes`` (None = not split): the inverse of :func:`lower_layout`,
    used to show a kernel's default layout. Dimension ``d``'s rank digit is ``r<d>``; it is lowered back on
    a grid of 2s as a self-check."""
    normal = [None if axis is None else normal_split(axis) for axis in axes]
    ops: list[LayoutOp] = []
    for dim, axis in enumerate(normal):
        if axis is None:
            continue
        whole, rank, symbol = DIM_DIGIT.format(dim), f"r{dim}", GRID_SYMBOL.format(axis.grid_dim)
        if axis.scheme is Scheme.BLOCK:
            ops.append(Split(DigitName(whole), Expr(symbol), DigitPair((rank, whole))))
        elif axis.scheme is Scheme.CYCLIC:
            ops.append(Block(DigitName(whole), Expr(symbol), DigitPair((whole, rank))))
        else:
            cycle, width = f"c{dim}", f"b{dim}"
            ops += [
                Block(DigitName(whole), Expr(f"{axis.block_size}*{symbol}"), DigitPair((whole, cycle))),
                Block(DigitName(cycle), Expr(str(axis.block_size)), DigitPair((rank, width))),
            ]
        ops.append(Distribute(DigitName(rank), GridDim(axis.grid_dim)))
    sample = [2] * (1 + max((axis.grid_dim for axis in normal if axis is not None), default=0))
    lowered = lower_layout(ops, len(axes), sample, "default layout")
    if lowered != normal:
        raise AssertionError(f"ops_for_axes({list(axes)}) lowers to {lowered}")
    return tuple(ops)
