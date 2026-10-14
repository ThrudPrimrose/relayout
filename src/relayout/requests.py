# Copyright 2026 ETH Zurich and the relayout authors.
"""The JSON form of a relayout request, mapped one to one onto :mod:`relayout.ops`.

``{"startup": {"load": [...], "grid": [...]}, "operations": [{"action": ..., ...}, ...]}`` is a
:class:`~relayout.ops.Requests`: ``startup`` is its :class:`~relayout.ops.Startup`, and each operation
object is the operation class registered under its ``action``, with the same field names. Every field type
validates itself (:mod:`relayout.ops`), so parsing, writing back and the JSON Schema (:func:`requests_schema`)
all come from the pydantic models. A malformed request is a ``ValueError`` naming the entry.
"""

from typing import Any

from pydantic import TypeAdapter, ValidationError
from pydantic.dataclasses import dataclass

from relayout.ops import OPERATION_CONFIG, OPERATIONS, Action, Operation, Requests, Startup, expr_json

__all__ = [
    "expr_json",
    "operation_json",
    "operations_schema",
    "parse_operation",
    "parse_operations",
    "parse_requests",
    "requests_json",
    "requests_schema",
]

#: Action -> the validator of its operation class.
ADAPTERS: dict[Action, TypeAdapter[Any]] = {action: TypeAdapter(cls) for action, cls in OPERATIONS.entries.items()}
STARTUP = TypeAdapter(Startup)


@dataclass(frozen=True, slots=True, config=OPERATION_CONFIG)
class RequestFile:
    """A request as written: its startup, and its operations, each read by its ``action``."""

    startup: Startup = Startup()
    operations: tuple[dict[str, Any], ...] = ()


FILE = TypeAdapter(RequestFile)


def refusal(where: str, exc: ValidationError) -> ValueError:
    """A pydantic refusal as a request fault naming ``where``."""
    return ValueError(f"{where}: {exc}")


def parse_operation(raw: object, where: str) -> Operation:
    """One operation from its JSON object; a malformed one is a request fault naming ``where``."""
    if not isinstance(raw, dict):
        raise ValueError(f"{where} must be an object with an 'action'; got {raw!r}")
    fields = dict(raw)
    try:
        action = Action(fields.pop("action", None))
    except ValueError:
        raise ValueError(
            f"{where}.action must be one of {[a.value for a in OPERATIONS.entries]}; got {raw.get('action')!r}"
        ) from None
    try:
        return ADAPTERS[action].validate_python(fields)
    except ValidationError as exc:
        raise refusal(where, exc) from None


def parse_operations(raw: object, where: str) -> tuple[Operation, ...]:
    """The ``operations`` list."""
    if not isinstance(raw, (list, tuple)):
        raise ValueError(f"{where} must be a list of operations; got {raw!r}")
    return tuple(parse_operation(entry, f"{where}[{k}]") for k, entry in enumerate(raw))


def parse_requests(raw: object) -> Requests:
    """A request from its JSON object (None = no request)."""
    if raw is None:
        return Requests()
    try:
        request = FILE.validate_python(raw)
    except ValidationError as exc:
        raise refusal("requests", exc) from None
    return Requests(request.startup, parse_operations(request.operations, "operations"))


def operation_json(op: Operation) -> dict[str, object]:
    """``op`` as the JSON object :func:`parse_operation` reads (unset optional fields left out)."""
    fields = ADAPTERS[op.action].dump_python(op, mode="json", exclude_none=True)
    return {"action": op.action.value, **fields}


def requests_json(requests: Requests) -> dict[str, object]:
    """``requests`` as the JSON :func:`parse_requests` reads, empty parts left out (``{}`` = none)."""
    out: dict[str, object] = {}
    startup = STARTUP.dump_python(requests.startup, mode="json", exclude_defaults=True)
    if startup:
        out["startup"] = startup
    if requests.operations:
        out["operations"] = [operation_json(op) for op in requests.operations]
    return out


def hoisted(schema: dict[str, Any], definitions: dict[str, Any]) -> dict[str, Any]:
    """``schema`` without its ``$defs``, which move into ``definitions`` (the refs point at the root)."""
    definitions.update(schema.pop("$defs", {}))
    return schema


def operation_schema(action: Action, definitions: dict[str, Any]) -> dict[str, Any]:
    cls = OPERATIONS.entries[action]
    schema = hoisted(ADAPTERS[action].json_schema(), definitions)
    schema.pop("title", None)
    return {
        **schema,
        "description": (cls.__doc__ or "").split("\n\n")[0].replace("\n", " ").strip(),
        "properties": {"action": {"const": action.value}, **schema["properties"]},
        "required": ["action", *schema.get("required", [])],
    }


def operations_schema(definitions: dict[str, Any] | None = None) -> dict[str, object]:
    """The JSON Schema of an ``operations`` list: one branch per registered action. The branches' shared
    definitions go into ``definitions`` (into the list schema's own ``$defs`` when none is given)."""
    shared: dict[str, Any] = {} if definitions is None else definitions
    branches = [operation_schema(action, shared) for action in OPERATIONS.entries]
    schema: dict[str, object] = {"type": "array", "items": {"oneOf": branches}}
    return schema if definitions is not None or not shared else {**schema, "$defs": shared}


def requests_schema() -> dict[str, object]:
    """The JSON Schema of a request."""
    definitions: dict[str, Any] = {}
    startup = hoisted(STARTUP.json_schema(), definitions)
    startup.pop("title", None)
    return {
        "type": "object",
        "properties": {"startup": startup, "operations": operations_schema(definitions)},
        "additionalProperties": False,
        "$defs": definitions,
    }
