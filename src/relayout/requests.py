# Copyright 2026 ETH Zurich and the relayout authors.
"""The JSON form of a relayout request, mapped one to one onto :mod:`relayout.ops`.

``{"startup": {"load": [...], "grid": [...]}, "operations": [{"action": ..., ...}, ...]}`` is a
:class:`~relayout.ops.Requests`: ``startup`` is its :class:`~relayout.ops.Startup`, and each operation
object is the operation class registered under its ``action``, with the same field names. A field's type
names its codec here (:data:`FIELD_CODECS`): how it is read, written back and described in the JSON
Schema (:func:`requests_schema`). A malformed request is a ``ValueError`` naming the entry.
"""

import dataclasses
import enum
import types
import typing
from collections.abc import Callable, Mapping
from typing import Any, NamedTuple

from relayout.ops import (
    OPERATIONS,
    RANKS_SYMBOL,
    Action,
    ArrayName,
    ArrayNames,
    BitMatrix,
    Bits,
    DigitName,
    DigitNames,
    DigitPair,
    Dim,
    Expr,
    FormatLabel,
    GridDim,
    IndexExprs,
    Location,
    Operation,
    Precision,
    RegistryError,
    Requests,
    Scheme,
    Startup,
    Width,
    check_index_expr,
    is_reserved,
)

__all__ = [
    "FIELD_CODECS",
    "FieldCodec",
    "expr_json",
    "operation_json",
    "operations_schema",
    "parse_grid",
    "parse_operation",
    "parse_operations",
    "parse_requests",
    "requests_json",
    "requests_schema",
]


def parse_name(value: object, where: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{where} must be a name; got {value!r}")
    return value


def parse_names(value: object, where: str) -> tuple[str, ...]:
    if not isinstance(value, list) or len(value) < len(("one", "two")):
        raise ValueError(f"{where} must be a list of two or more names; got {value!r}")
    names = tuple(parse_name(v, f"{where}[{k}]") for k, v in enumerate(value))
    if len(set(names)) != len(names):
        raise ValueError(f"{where} names one twice: {list(names)}")
    return names


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


def positive_int(value: object, where: str) -> int:
    number = non_negative_int(value, where)
    if number < 1:
        raise ValueError(f"{where} must be >= 1; got {value!r}")
    return number


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


def enum_codec[E: enum.Enum](cls: type[E]) -> "FieldCodec":
    """The codec of an enum field: its values on the wire."""

    def parse(value: object, where: str) -> E:
        try:
            return cls(value)
        except ValueError:
            raise ValueError(f"{where} must be one of {[e.value for e in cls]}; got {value!r}") from None

    return FieldCodec(parse, lambda member: member.value, {"enum": [e.value for e in cls]})


def expr_json(text: str) -> int | str:
    """``text`` back as JSON: a plain integer literal as an int."""
    return int(text) if text.lstrip("-").isdigit() else text


def plain_json(value: object) -> object:
    """A tuple field as a JSON list (nested for a matrix)."""
    return [plain_json(v) for v in value] if isinstance(value, tuple) else value


class FieldCodec(NamedTuple):
    """How one field type is read from JSON (a request fault naming ``where`` when malformed), written
    back, and described in the JSON Schema."""

    parse: Callable[[object, str], Any]
    dump: Callable[[Any], object] = plain_json
    schema: Mapping[str, object] = types.MappingProxyType({})


NAME_SCHEMA = {"type": "string", "minLength": 1}
EXPR_SCHEMA = {"type": ["integer", "string"]}
#: Field type -> codec. An operation may only declare fields of these types (or ``T | None`` of them).
FIELD_CODECS: dict[object, FieldCodec] = {
    ArrayName: FieldCodec(parse_name, schema=NAME_SCHEMA),
    ArrayNames: FieldCodec(parse_names, schema={"type": "array", "items": NAME_SCHEMA, "minItems": 2}),
    DigitName: FieldCodec(parse_digit_name, schema=NAME_SCHEMA),
    DigitNames: FieldCodec(parse_digit_names, schema={"type": "array", "items": NAME_SCHEMA, "minItems": 1}),
    DigitPair: FieldCodec(
        parse_digit_pair, schema={"type": "array", "items": NAME_SCHEMA, "minItems": 2, "maxItems": 2}
    ),
    Dim: FieldCodec(non_negative_int, schema={"type": "integer", "minimum": 0}),
    GridDim: FieldCodec(non_negative_int, schema={"type": "integer", "minimum": 0}),
    Width: FieldCodec(positive_int, schema={"type": "integer", "minimum": 1}),
    Expr: FieldCodec(parse_expr, expr_json, EXPR_SCHEMA),
    IndexExprs: FieldCodec(parse_index_exprs, schema={"type": "array", "items": {"type": "string"}}),
    Bits: FieldCodec(int_list, schema={"type": "array", "items": {"type": "integer", "minimum": 0}}),
    BitMatrix: FieldCodec(
        parse_bit_matrix, schema={"type": "array", "items": {"type": "array", "items": {"enum": [0, 1]}}}
    ),
    FormatLabel: FieldCodec(parse_name, schema=NAME_SCHEMA),
    Scheme: enum_codec(Scheme),
    Location: enum_codec(Location),
    Precision: enum_codec(Precision),
}


def field_codec(hint: object) -> tuple[FieldCodec | None, bool]:
    """``hint``'s codec and whether the field is optional (``T | None``)."""
    if isinstance(hint, types.UnionType) and type(None) in hint.__args__:
        (inner,) = (arg for arg in hint.__args__ if arg is not type(None))
        return FIELD_CODECS.get(inner), True
    return FIELD_CODECS.get(hint), False


def op_fields(cls: type) -> dict[str, tuple[FieldCodec, bool]]:
    """``cls``'s dataclass fields with their codecs and optionality; a field of an unregistered type is refused."""
    hints = typing.get_type_hints(cls)
    out: dict[str, tuple[FieldCodec, bool]] = {}
    for spec in dataclasses.fields(cls):
        codec, optional = field_codec(hints[spec.name])
        if codec is None:
            raise RegistryError(f"{cls.__name__}.{spec.name}: field type {hints[spec.name]!r} has no JSON codec")
        out[spec.name] = (codec, optional)
    return out


def parse_operation(raw: object, where: str) -> Operation:
    """One operation from its JSON object; a malformed one is a request fault naming ``where``."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object with an 'action'; got {raw!r}")
    try:
        action = Action(raw.get("action"))
    except ValueError:
        raise ValueError(
            f"{where}.action must be one of {[a.value for a in OPERATIONS.entries]}; got {raw.get('action')!r}"
        ) from None
    cls = OPERATIONS.entries[action]
    fields = op_fields(cls)
    unknown = sorted(set(raw) - set(fields) - {"action"})
    if unknown:
        raise ValueError(f"{where} has unknown field(s) {unknown}; {action.value} takes {sorted(fields)}")
    values = {
        name: None if optional and raw.get(name) is None else codec.parse(raw.get(name), f"{where}.{name}")
        for name, (codec, optional) in fields.items()
    }
    try:
        return cls(**values)
    except ValueError as exc:
        raise ValueError(f"{where}: {exc}") from None


def parse_operations(raw: object, where: str) -> tuple[Operation, ...]:
    """The ``operations`` list."""
    if not isinstance(raw, list):
        raise ValueError(f"{where} must be a list of operations; got {raw!r}")
    return tuple(parse_operation(entry, f"{where}[{k}]") for k, entry in enumerate(raw))


def operation_json(op: Operation) -> dict[str, object]:
    """``op`` as the JSON object :func:`parse_operation` reads (unset optional fields left out)."""
    values = dataclasses.asdict(op)  # type: ignore[call-overload]  # every registered operation is a dataclass
    out: dict[str, object] = {"action": op.action.value}
    for name, (codec, _optional) in op_fields(type(op)).items():
        if values[name] is not None:
            out[name] = codec.dump(values[name])
    return out


def operations_schema() -> dict[str, object]:
    """The JSON Schema of an ``operations`` list: one branch per registered action."""
    branches = []
    for action, cls in OPERATIONS.entries.items():
        fields = op_fields(cls)
        branches.append(
            {
                "type": "object",
                "description": (cls.__doc__ or "").split("\n\n")[0].replace("\n", " ").strip(),
                "properties": {
                    "action": {"const": action.value},
                    **{name: dict(codec.schema) for name, (codec, _) in fields.items()},
                },
                "required": ["action", *(name for name, (_, optional) in fields.items() if not optional)],
                "additionalProperties": False,
            }
        )
    return {"type": "array", "items": {"oneOf": branches}}


def parse_grid(raw: object, where: str) -> tuple[str, ...]:
    """A grid: a non-empty list of extent expressions over ``P`` (ints allowed)."""
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"{where} must be a non-empty list of extents over {RANKS_SYMBOL}; got {raw!r}")
    return tuple(parse_expr(extent, f"{where}[{g}]") for g, extent in enumerate(raw))


def parse_startup(raw: object, where: str) -> Startup:
    """``startup``: ``load`` (library names) and ``grid`` (extents over ``P``), both optional."""
    if raw is None:
        return Startup()
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object {{load, grid}}; got {raw!r}")
    unknown = sorted(set(raw) - {"load", "grid"})
    if unknown:
        raise ValueError(f"{where} has unknown field(s) {unknown}; it takes load and grid")
    load = raw.get("load") or []
    if not isinstance(load, list) or not all(isinstance(name, str) and name for name in load):
        raise ValueError(f"{where}.load must be a list of library names; got {load!r}")
    grid = None if raw.get("grid") is None else parse_grid(raw["grid"], f"{where}.grid")
    return Startup(tuple(load), grid)


def parse_requests(raw: object) -> Requests:
    """A request from its JSON object (None = no request)."""
    if raw is None:
        return Requests()
    if not isinstance(raw, dict):
        raise ValueError(f"requests must be an object {{startup, operations}}; got {type(raw).__name__}")
    unknown = sorted(set(raw) - {"startup", "operations"})
    if unknown:
        raise ValueError(f"requests has unknown field(s) {unknown}; it takes startup and operations")
    return Requests(
        parse_startup(raw.get("startup"), "startup"), parse_operations(raw.get("operations", []), "operations")
    )


def requests_json(requests: Requests) -> dict[str, object]:
    """``requests`` as the JSON :func:`parse_requests` reads, empty parts left out (``{}`` = none)."""
    startup: dict[str, object] = {}
    if requests.startup.load:
        startup["load"] = list(requests.startup.load)
    if requests.startup.grid is not None:
        startup["grid"] = [expr_json(extent) for extent in requests.startup.grid]
    out: dict[str, object] = {}
    if startup:
        out["startup"] = startup
    if requests.operations:
        out["operations"] = [operation_json(op) for op in requests.operations]
    return out


def requests_schema() -> dict[str, object]:
    """The JSON Schema of a request."""
    return {
        "type": "object",
        "properties": {
            "startup": {
                "type": "object",
                "properties": {
                    "load": {"type": "array", "items": NAME_SCHEMA},
                    "grid": {"type": "array", "items": EXPR_SCHEMA, "minItems": 1},
                },
                "additionalProperties": False,
            },
            "operations": operations_schema(),
        },
        "additionalProperties": False,
    }


# Every registered operation's fields have codecs, so a new operation without one fails at import.
for registered in OPERATIONS.entries.values():
    op_fields(registered)
