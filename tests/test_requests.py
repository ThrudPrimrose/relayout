# Copyright 2026 ETH Zurich and the relayout authors.
"""The JSON form of a request: one object per operation, mapped one to one onto the operation classes,
its refusals, and the JSON Schema generated from the registry."""

import pytest

from relayout.ops import OPERATIONS, Block, Cast, Distribute, Precision, Requests, Scheme, Startup
from relayout.requests import operation_json, parse_operation, parse_requests, requests_json, requests_schema

REQUEST = {
    "startup": {"load": ["openblas"], "grid": ["P"]},
    "operations": [
        {"action": "cast", "array": "A", "to": "fp32"},
        {"action": "unzip", "array": "Z", "into": ["Z_re", "Z_im"]},
        {"action": "zip", "arrays": ["Z_im", "B"], "into": "W"},
        {"action": "block", "array": "A", "digit": "i0", "factor": "32*P0", "into": ["i0", "c"]},
        {"action": "block", "array": "A", "digit": "c", "factor": 32, "into": ["r", "b"]},
        {"action": "distribute", "array": "A", "grid_dim": 0, "digit": "r"},
        {"action": "distribute", "array": "W", "grid_dim": 0, "dim": 0, "scheme": "block"},
        {"action": "replicate", "array": "x"},
        {"action": "place", "array": "A", "on": "device"},
        {"action": "format", "array": "M", "to": "csr"},
    ],
}

#: One sample per action, so every registered operation is round-tripped.
SAMPLES = {
    "cast": {"action": "cast", "array": "A", "to": "bf16"},
    "format": {"action": "format", "array": "M", "to": "bsr:4"},
    "zip": {"action": "zip", "arrays": ["u", "v"], "into": "uv"},
    "unzip": {"action": "unzip", "array": "z", "into": ["re", "im"]},
    "block": {"action": "block", "array": "A", "digit": "i0", "factor": "2*P0", "into": ["o", "f"]},
    "unblock": {"action": "unblock", "array": "A", "digits": ["o", "f"], "factor": 2, "into": "i0"},
    "split": {"action": "split", "array": "A", "digit": "i0", "parts": "P0", "into": ["r", "l"]},
    "unsplit": {"action": "unsplit", "array": "A", "digits": ["r", "l"], "parts": "P0", "into": "i0"},
    "permute": {"action": "permute", "array": "A", "order": ["i1", "i0"]},
    "pad": {"action": "pad", "array": "A", "digit": "i1", "amount": "P0 - 1"},
    "unpad": {"action": "unpad", "array": "A", "digit": "i1", "amount": 3},
    "shuffle": {"action": "shuffle", "array": "A", "digits": ["i1"], "f": ["(i1 + 3) % 8"], "f_inv": ["(i1 + 5) % 8"]},
    "linear_shuffle": {
        "action": "linear_shuffle",
        "array": "A",
        "digits": ["i0", "i1"],
        "bits": [1, 1],
        "matrix": [[1, 0], [1, 1]],
    },
    "distribute": {
        "action": "distribute",
        "array": "A",
        "grid_dim": 0,
        "dim": 1,
        "scheme": "block_cyclic",
        "block_size": 8,
    },
    "replicate": {"action": "replicate", "array": "x"},
    "place": {"action": "place", "array": "A", "on": "host"},
}


def test_a_request_is_its_operation_classes() -> None:
    requests = parse_requests(REQUEST)
    assert requests.startup == Startup(("openblas",), ("P",))
    assert requests.operations[0] == Cast("A", Precision.FP32)
    assert requests.operations[3] == Block("A", "i0", "32*P0", ("i0", "c"))
    assert requests.operations[6] == Distribute("W", 0, dim=0, scheme=Scheme.BLOCK)
    assert requests_json(requests) == REQUEST
    assert requests_json(Requests()) == {}


def test_every_registered_action_round_trips_through_json() -> None:
    """Adding an operation is one class: its JSON comes from its fields."""
    assert {action.value for action in OPERATIONS.entries} == set(SAMPLES)
    for raw in SAMPLES.values():
        assert operation_json(parse_operation(raw, "t")) == raw


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"action": "bogus"}, r"operations\[0\].action must be one of"),
        ({"action": "cast", "array": "A", "to": "fp42"}, r"operations\[0\].to must be one of"),
        ({"action": "block", "array": "A", "digit": "P0", "factor": 2, "into": ["a", "b"]}, "other than P, P0"),
        ({"action": "block", "array": "A", "digit": "i0", "factor": 2, "into": ["a"]}, "two digits the cut leaves"),
        (
            {"action": "block", "array": "A", "digit": "i0", "factor": 4, "into": ["a", "b"], "dim": 1},
            r"unknown field\(s\) \['dim'\]",
        ),
        ({"action": "shuffle", "array": "A", "digits": ["i0"], "f": ["i0 ** 2"], "f_inv": ["i0"]}, "uses Pow"),
        ({"action": "distribute", "array": "A", "grid_dim": 0}, r"operations\[0\]: name either the digit"),
        ({"action": "zip", "arrays": ["u"], "into": "uv"}, "two or more names"),
        ({"action": "place", "array": "A", "on": "gpu"}, "must be one of"),
    ],
)
def test_a_malformed_operation_is_refused_by_its_entry(raw: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_requests({"operations": [raw]})


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"startup": {"grid": []}}, "non-empty list of extents over P"),
        ({"startup": {"grid": ["P ** 2"]}}, "uses Pow"),
        ({"startup": {"load": "openblas"}}, "list of library names"),
        ({"startup": {"compiler": "gcc"}}, "takes load and grid"),
        ({"layout": {}}, "takes startup and operations"),
        ({"operations": {}}, "must be a list of operations"),
    ],
)
def test_a_malformed_request_is_refused(raw: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_requests(raw)


def test_the_json_schema_has_one_branch_per_action() -> None:
    schema = requests_schema()
    branches = schema["properties"]["operations"]["items"]["oneOf"]  # type: ignore[index]
    by_action = {branch["properties"]["action"]["const"]: branch for branch in branches}
    assert set(by_action) == set(SAMPLES)
    assert by_action["cast"]["properties"]["to"] == {"enum": ["fp64", "fp32", "fp16", "bf16", "fp8_e4m3", "fp8_e5m2"]}
    assert by_action["distribute"]["required"] == ["action", "array", "grid_dim"]
    assert by_action["block"]["additionalProperties"] is False


if __name__ == "__main__":
    test_a_request_is_its_operation_classes()
    test_every_registered_action_round_trips_through_json()
    test_a_malformed_operation_is_refused_by_its_entry({"action": "bogus"}, r"operations\[0\].action must be one of")
    test_a_malformed_request_is_refused({"layout": {}}, "takes startup and operations")
    test_the_json_schema_has_one_branch_per_action()
