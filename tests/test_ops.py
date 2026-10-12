# Copyright 2026 ETH Zurich and the relayout authors.
"""Layout op sequences: the registry, JSON, inverses, the grid and the lowering to one split per dimension.

The kernel's default distribution is shown as an op list (``ops_for_axes``), so lowering that list
must give the default back; a sequence the lowering does not cover is refused by name.
"""

import itertools

import pytest

from relayout.ops import (
    GROUP_OPS,
    LAYOUT_OPS,
    AxisSplit,
    Distribute,
    LinearShuffle,
    Scheme,
    Unzip,
    Zip,
    eval_index,
    gf2_inverse,
    grid_extents,
    groups_from_json,
    groups_json,
    inverse_sequence,
    lower_layout,
    op_json,
    ops_for_axes,
    parse_grid,
    parse_op,
)

AXES = (
    None,
    AxisSplit(0, Scheme.BLOCK, 1),
    AxisSplit(0, Scheme.CYCLIC, 1),
    AxisSplit(0, Scheme.BLOCK_CYCLIC, 4),
    AxisSplit(1, Scheme.BLOCK, 1),
    AxisSplit(1, Scheme.BLOCK_CYCLIC, 3),
)


def distribute(digit: str, grid_dim: int = 0) -> dict[str, object]:
    return {"op": "distribute", "digit": digit, "grid_dim": grid_dim}


def block(digit: str, factor: int | str, into: list[str]) -> dict[str, object]:
    return {"op": "block", "digit": digit, "factor": factor, "into": into}


def split(digit: str, parts: int | str, into: list[str]) -> dict[str, object]:
    return {"op": "split", "digit": digit, "parts": parts, "into": into}


def lowered(ops: list[dict[str, object]], ndim: int, grid: list[int]) -> list[AxisSplit | None]:
    return lower_layout([parse_op(op, "t") for op in ops], ndim, grid, "t")


def test_the_three_schemes_lower_from_their_op_sequences() -> None:
    assert lowered([split("i0", "P0", ["r", "i0"]), distribute("r")], 2, [4]) == [AxisSplit(0, Scheme.BLOCK, 1), None]
    assert lowered([block("i1", "P0", ["i1", "r"]), distribute("r")], 2, [4]) == [
        None,
        AxisSplit(0, Scheme.CYCLIC, 1),
    ]
    blocked = [block("i0", "8*P0", ["i0", "c"]), block("c", 8, ["r", "b"]), distribute("r")]
    assert lowered(blocked, 2, [4]) == [AxisSplit(0, Scheme.BLOCK_CYCLIC, 8), None]


def test_a_grid_factor_re_evaluates_at_every_rank_count() -> None:
    """One request stays valid when a scaling run re-sizes the grid: ``P0`` is read at each P."""
    ops = [block("i0", "2*P0", ["i0", "c"]), block("c", 2, ["r", "b"]), distribute("r")]
    for p in (1, 2, 4, 8):
        assert lowered(ops, 1, [p]) == [AxisSplit(0, Scheme.BLOCK_CYCLIC, 2)]


def test_a_grid_is_extents_over_the_rank_count() -> None:
    """A square grid spans only the square rank counts; a 1-D grid every one."""
    square = parse_grid(["root(P, 2)", "root(P, 2)"], "grid")
    assert grid_extents(square, 16, "grid") == (4, 4)
    assert grid_extents(square, 8, "grid") is None
    assert grid_extents(parse_grid(["P // 2", 2], "grid"), 8, "grid") == (4, 2)
    assert grid_extents(parse_grid(["P"], "grid"), 3, "grid") == (3,)
    with pytest.raises(ValueError, match="extent below 1"):
        grid_extents(parse_grid(["P // 2", 2], "grid"), 1, "grid")


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_default_layout_raises_to_ops_that_lower_back(ndim: int) -> None:
    """``ops_for_axes`` is what an agent is shown; lowering it must be the descriptor it came from."""
    for axes in itertools.product(AXES, repeat=ndim):
        grid_dims = [axis.grid_dim for axis in axes if axis is not None]
        if len(grid_dims) != len(set(grid_dims)):
            continue
        ops = ops_for_axes(axes)
        for grid in ([1, 1], [2, 3], [4, 4]):
            assert lower_layout(ops, ndim, grid, "t") == list(axes), (axes, grid)


def test_a_non_row_major_local_order_is_refused_with_the_permute_that_fixes_it() -> None:
    ops = [block("i0", 4, ["o", "f"]), {"op": "permute", "order": ["f", "o", "i1"]}]
    with pytest.raises(ValueError, match=r"append \{\"op\": \"permute\", \"order\": \['o', 'f', 'i1'\]\}"):
        lowered(ops, 2, [4])


@pytest.mark.parametrize(
    ("ops", "match"),
    [
        ([{"op": "pad", "digit": "i0", "amount": 4}], "'pad' is not implemented yet"),
        ([{"op": "shuffle", "digits": ["i0"], "f": ["i0"], "f_inv": ["i0"]}], "'shuffle' is not implemented yet"),
        ([block("i0", 3, ["i0", "r"]), distribute("r")], "must have extent P0 = 4"),
        ([distribute("i0")], "has an unknown extent"),
        (
            [block("i0", "2*P0", ["o", "c"]), block("c", 2, ["r", "b"]), block("o", 3, ["o", "x"]), distribute("r")],
            "must be blocked by",
        ),
        ([split("i0", 2, ["a", "b"]), split("a", 2, ["x", "y"])], "already cut"),
        ([block("i2", 4, ["a", "b"])], "no digit 'i2'"),
        ([block("i0", 4, ["i1", "b"])], r"name\(s\) \['i1'\] are taken"),
        ([split("i0", "P1", ["r", "i0"]), distribute("r", 3)], "grid_dim 3 does not exist"),
        ([{"op": "permute", "order": ["i1"]}], "must name every digit once"),
    ],
)
def test_a_sequence_that_cannot_lower_is_refused_by_name(ops: list[dict[str, object]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        lowered(ops, 2, [4, 2])


def test_every_op_has_its_inverse() -> None:
    """pad <-> unpad, permute <-> the order before it, block <-> unblock, shuffle f <-> f_inv."""
    shuffle = {"op": "shuffle", "digits": ["i0", "i1"], "f": ["i0", "i1 ^ (i0 % 8)"], "f_inv": ["i0", "i1 ^ (i0 % 8)"]}
    ops = tuple(
        parse_op(raw, "t")
        for raw in (
            {"op": "pad", "digit": "i0", "amount": 4},
            {"op": "permute", "order": ["i1", "i0"]},
            shuffle,
            block("i1", "P0", ["o", "f"]),
        )
    )
    inverted = inverse_sequence(ops, 2)
    assert [op_json(op) for op in inverted] == [
        {"op": "unblock", "digits": ["o", "f"], "factor": "P0", "into": "i1"},
        shuffle,
        {"op": "permute", "order": ["i0", "i1"]},
        {"op": "unpad", "digit": "i0", "amount": 4},
    ]
    with pytest.raises(ValueError, match="undone by the gather"):
        inverse_sequence((parse_op(split("i0", 2, ["r", "i0"]), "t"), parse_op(distribute("r"), "t")), 1)


def test_every_registered_op_round_trips_through_json_and_inverts() -> None:
    """Adding an op is one registered class: its JSON and its inverse come from the registry."""
    samples = {
        "block": block("i0", "2*P0", ["o", "f"]),
        "unblock": {"op": "unblock", "digits": ["i0", "i1"], "factor": 2, "into": "x"},
        "split": split("i0", "P0", ["r", "l"]),
        "unsplit": {"op": "unsplit", "digits": ["i0", "i1"], "parts": "P0", "into": "x"},
        "permute": {"op": "permute", "order": ["i1", "i0"]},
        "pad": {"op": "pad", "digit": "i1", "amount": "P0 - 1"},
        "unpad": {"op": "unpad", "digit": "i1", "amount": 3},
        "shuffle": {"op": "shuffle", "digits": ["i1"], "f": ["(i1 + 3) % 8"], "f_inv": ["(i1 + 5) % 8"]},
        "linear_shuffle": {"op": "linear_shuffle", "digits": ["i0", "i1"], "bits": [1, 1], "matrix": [[1, 0], [1, 1]]},
        "distribute": distribute("i0"),
    }
    assert set(samples) == set(LAYOUT_OPS.keys())
    for raw in samples.values():
        op = parse_op(raw, "t")
        assert op_json(op) == raw
        if not isinstance(op, Distribute):
            after = tuple(raw.get("order", ("i0", "i1")))  # the digit names a permute leaves
            assert op.inverse(("i0", "i1")).inverse(after) == op


def test_a_linear_shuffle_inverts_its_matrix_over_gf2() -> None:
    """The XOR swizzle ``col' = col ^ row`` on 2-bit indices is its own inverse; a general one is not."""
    swizzle = ((1, 0, 0, 0), (0, 1, 0, 0), (1, 0, 1, 0), (0, 1, 0, 1))
    assert gf2_inverse(swizzle) == swizzle
    skew = ((1, 1, 0), (0, 1, 1), (0, 0, 1))
    inverse = gf2_inverse(skew)
    assert inverse is not None
    product = [[sum(skew[r][k] * inverse[k][c] for k in range(3)) % 2 for c in range(3)] for r in range(3)]
    assert product == [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
    assert gf2_inverse(((1, 1), (1, 1))) is None
    raw = {"op": "linear_shuffle", "digits": ["i0", "i1"], "bits": [2, 1], "matrix": [list(r) for r in skew]}
    op = parse_op(raw, "t")
    assert isinstance(op, LinearShuffle)
    assert op.inverse(("i0", "i1")).matrix == inverse


def test_unblock_and_unsplit_undo_their_cuts_in_the_digit_tuple() -> None:
    ops = [
        block("i0", 4, ["o", "f"]),
        {"op": "unblock", "digits": ["o", "f"], "factor": 4, "into": "i0"},
        split("i0", 3, ["a", "b"]),
        {"op": "unsplit", "digits": ["a", "b"], "parts": 3, "into": "i0"},
        split("i0", "P0", ["r", "i0"]),
        distribute("r"),
    ]
    assert lowered(ops, 1, [2]) == [AxisSplit(0, Scheme.BLOCK, 1)]


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"op": "bogus"}, "not a layout op"),
        ({"op": "permute", "order": ["i0", "i0"]}, "names a digit twice"),
        ({"op": "block", "digit": "P0", "factor": 2, "into": ["a", "b"]}, "other than P, P0"),
        ({"op": "block", "digit": "i0", "factor": 2, "into": ["a"]}, "two digits the cut leaves"),
        ({"op": "shuffle", "digits": ["i0"], "f": ["i0"]}, "f_inv must be a list of index expressions"),
        ({"op": "shuffle", "digits": ["i0", "i1"], "f": ["i0"], "f_inv": ["i0"]}, "one expression per"),
        ({"op": "shuffle", "digits": ["i0"], "f": ["i0 ** 2"], "f_inv": ["i0"]}, "uses Pow"),
        ({"op": "block", "digit": "i0", "factor": 4, "into": ["a", "b"], "dim": 1}, r"unknown field\(s\) \['dim'\]"),
        ({"op": "linear_shuffle", "digits": ["i0"], "bits": [2], "matrix": [[1, 1], [1, 1]]}, "singular"),
    ],
)
def test_a_malformed_op_is_refused_by_name(raw: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_op(raw, "t")


def test_zip_and_unzip_are_group_ops_requested_as_name_to_fields() -> None:
    """Zip fuses arrays into one struct array and unzip splits one; neither is the other's inverse op."""
    assert set(GROUP_OPS.keys()) == {"zip", "unzip"}
    (zipped,) = groups_from_json("zip", {"uv": ["u", "v"]}, "zip")
    assert zipped == Zip("uv", ("u", "v"))
    assert groups_json([zipped, Unzip("z", ("re", "im"))]) == {"zip": {"uv": ["u", "v"]}, "unzip": {"z": ["re", "im"]}}
    assert "inverse" not in vars(Unzip)
    with pytest.raises(ValueError, match="two or more arrays"):
        groups_from_json("unzip", {"z": ["re"]}, "unzip")


def test_an_index_expression_evaluates_with_python_floor_semantics() -> None:
    names = {"i0": -3, "P0": 4}
    assert eval_index("i0 // 2 + i0 % P0", names, "t") == -2 + 1
    assert eval_index("(5 ^ 3) << 1 | min(P0, 7)", names, "t") == ((5 ^ 3) << 1) | 4
    assert eval_index("root(P0 * 16, 3)", names, "t") == 4
    with pytest.raises(ValueError, match="names 'N'"):
        eval_index("N + 1", names, "t")
    with pytest.raises(ValueError, match="uses Call"):
        eval_index("min(P0)", names, "t")


if __name__ == "__main__":
    test_the_three_schemes_lower_from_their_op_sequences()
    test_a_grid_factor_re_evaluates_at_every_rank_count()
    test_a_grid_is_extents_over_the_rank_count()
    test_the_default_layout_raises_to_ops_that_lower_back(1)
    test_the_default_layout_raises_to_ops_that_lower_back(2)
    test_the_default_layout_raises_to_ops_that_lower_back(3)
    test_a_non_row_major_local_order_is_refused_with_the_permute_that_fixes_it()
    test_a_sequence_that_cannot_lower_is_refused_by_name([{"op": "pad", "digit": "i0", "amount": 4}], "'pad' is not")
    test_a_sequence_that_cannot_lower_is_refused_by_name([distribute("i0")], "has an unknown extent")
    test_a_sequence_that_cannot_lower_is_refused_by_name([block("i2", 4, ["a", "b"])], "no digit 'i2'")
    test_every_op_has_its_inverse()
    test_every_registered_op_round_trips_through_json_and_inverts()
    test_a_linear_shuffle_inverts_its_matrix_over_gf2()
    test_unblock_and_unsplit_undo_their_cuts_in_the_digit_tuple()
    test_a_malformed_op_is_refused_by_name({"op": "bogus"}, "not a layout op")
    test_a_malformed_op_is_refused_by_name({"op": "permute", "order": ["i0", "i0"]}, "names a digit twice")
    test_zip_and_unzip_are_group_ops_requested_as_name_to_fields()
    test_an_index_expression_evaluates_with_python_floor_semantics()
