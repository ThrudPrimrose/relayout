# Copyright 2026 ETH Zurich and the relayout authors.
"""Layout op sequences: the registry, JSON, inverses and the lowering to one split per dimension.

The kernel's default distribution is shown as an op list (``ops_for_axes``), so lowering that list
must give the default back; a sequence the lowering does not cover is refused by name.
"""

import itertools

import pytest

from relayout.ops import (
    GROUP_OPS,
    LAYOUT_OPS,
    Distribute,
    LinearShuffle,
    Scheme,
    Split,
    Unzip,
    Zip,
    eval_index,
    from_json,
    gf2_inverse,
    inverse_sequence,
    lower_layout,
    op_json,
    ops_for_axes,
    parse_op,
    to_json,
)

AXES = (None, (0, "block", 1), (0, "cyclic", 1), (0, "block_cyclic", 4), (1, "block", 1), (1, "block_cyclic", 3))


def distribute(digit: int, grid_dim: int = 0) -> dict[str, object]:
    return {"op": "distribute", "digit": digit, "grid_dim": grid_dim}


def block(dim: int, factor: int | str) -> dict[str, object]:
    return {"op": "block", "dim": dim, "factor": factor}


type Plain = tuple[int, str, int] | None


def plain(splits: list[Split | None]) -> list[Plain]:
    """Splits as ``(grid_dim, scheme name, block_size)``, for readable expectations."""
    return [None if s is None else (s.grid_dim, s.scheme.value, s.block_size) for s in splits]


def lowered(ops: list[dict[str, object]], ndim: int, grid: list[int]) -> list[Plain]:
    return plain(lower_layout([parse_op(op, "t") for op in ops], ndim, grid, "t"))


def test_the_three_schemes_lower_from_their_op_sequences() -> None:
    assert lowered([distribute(0)], 2, [4]) == [(0, "block", 1), None]
    assert lowered([block(1, "P0"), distribute(2)], 2, [4]) == [None, (0, "cyclic", 1)]
    blocked = [block(0, "8*P0"), block(0, 8), {"op": "permute", "perm": [0, 2, 3, 1]}, distribute(1)]
    assert lowered(blocked, 2, [4]) == [(0, "block_cyclic", 8), None]


def test_a_grid_factor_re_evaluates_at_every_rank_count() -> None:
    """One request stays valid when a scaling run re-sizes the grid: ``P0`` is read at each P."""
    ops = [block(0, "2*P0"), block(0, 2), distribute(1)]
    for p in (1, 2, 4, 8):
        assert lowered(ops, 1, [p]) == [(0, "block_cyclic", 2)]


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_default_layout_raises_to_ops_that_lower_back(ndim: int) -> None:
    """``ops_for_axes`` is what an agent is shown; lowering it must be the descriptor it came from."""
    for axes in itertools.product(AXES, repeat=ndim):
        grid_dims = [axis[0] for axis in axes if axis is not None]
        if len(grid_dims) != len(set(grid_dims)):
            continue
        ops = ops_for_axes([None if a is None else Split(a[0], Scheme(a[1]), a[2]) for a in axes])
        for grid in ([1, 1], [2, 3], [4, 4]):
            expected = [None if a is None else (a[0], a[1], 1 if a[1] in ("block", "cyclic") else a[2]) for a in axes]
            assert plain(lower_layout(ops, ndim, grid, "t")) == expected, (axes, grid)


def test_a_non_row_major_local_order_is_refused_with_the_permute_that_fixes_it() -> None:
    ops = [block(0, "4*P0"), block(0, 4), distribute(2)]  # dimension 1's digit sits between dimension 0's
    with pytest.raises(ValueError, match=r'append \{"op": "permute", "perm": \[0, 2, 1\]\}'):
        lowered(ops, 2, [4])


@pytest.mark.parametrize(
    ("ops", "match"),
    [
        ([{"op": "pad", "dim": 0, "amount": 4}], "'pad' is not implemented yet"),
        ([{"op": "shuffle", "dims": [0], "f": ["i0"], "f_inv": ["i0"]}], "'shuffle' is not implemented yet"),
        ([block(0, 3), distribute(2)], "must have extent P0 = 4"),
        ([block(0, 8), distribute(0)], "coarse digit of a blocked dimension"),
        ([distribute(0), distribute(0, 1)], "dimension 0 is already distributed"),
        ([block(2, 4)], "dimension 2 does not exist"),
        ([distribute(0, 3)], "grid_dim 3 does not exist"),
        ([{"op": "permute", "perm": [1, 0, 2]}], "permute of 3 positions over a 2-digit tuple"),
    ],
)
def test_a_sequence_that_cannot_lower_is_refused_by_name(ops: list[dict[str, object]], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        lowered(ops, 2, [4, 2])


def test_every_op_has_its_inverse() -> None:
    """pad <-> unpad, permute <-> its inverse permutation, block <-> unblock, shuffle f <-> f_inv."""
    shuffle = parse_op(
        {"op": "shuffle", "dims": [0, 1], "f": ["i0", "i1 ^ (i0 % 8)"], "f_inv": ["i0", "i1 ^ (i0 % 8)"]}, "t"
    )
    ops = (
        parse_op({"op": "pad", "dim": 0, "amount": 4}, "t"),
        parse_op({"op": "permute", "perm": [2, 0, 1]}, "t"),
        parse_op(block(1, "P0"), "t"),
        shuffle,
    )
    inverted = inverse_sequence(ops)
    assert [op_json(op) for op in inverted] == [
        {"op": "shuffle", "dims": [0, 1], "f": ["i0", "i1 ^ (i0 % 8)"], "f_inv": ["i0", "i1 ^ (i0 % 8)"]},
        {"op": "unblock", "dim": 1, "factor": "P0"},
        {"op": "permute", "perm": [1, 2, 0]},
        {"op": "unpad", "dim": 0, "amount": 4},
    ]
    assert inverse_sequence(inverted)[1:] == ops[1:]
    with pytest.raises(ValueError, match="undone by the gather"):
        inverse_sequence((Distribute(0, 0),))


def test_every_registered_op_round_trips_through_json_and_inverts() -> None:
    """Adding an op is one registered class: its JSON and its inverse come from the registry."""
    samples = {
        "block": block(0, "2*P0"),
        "unblock": {"op": "unblock", "dim": 0, "factor": 2},
        "permute": {"op": "permute", "perm": [1, 0]},
        "pad": {"op": "pad", "dim": 1, "amount": "P0 - 1"},
        "unpad": {"op": "unpad", "dim": 1, "amount": 3},
        "shuffle": {"op": "shuffle", "dims": [1], "f": ["(i0 + 3) % 8"], "f_inv": ["(i0 + 5) % 8"]},
        "linear_shuffle": {"op": "linear_shuffle", "dims": [0, 1], "bits": [1, 1], "matrix": [[1, 0], [1, 1]]},
        "distribute": distribute(0),
    }
    assert set(samples) == set(LAYOUT_OPS.keys())
    for raw in samples.values():
        op = parse_op(raw, "t")
        assert op_json(op) == raw
        if raw["op"] != "distribute":
            assert inverse_sequence(inverse_sequence((op,))) == (op,)


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
    op = parse_op({"op": "linear_shuffle", "dims": [0, 1], "bits": [2, 1], "matrix": [list(r) for r in skew]}, "t")
    assert isinstance(op, LinearShuffle)
    assert op.inverse().matrix == inverse


def test_unblock_undoes_block_in_the_digit_tuple() -> None:
    ops = [block(0, 4), {"op": "unblock", "dim": 0, "factor": 4}, distribute(0)]
    assert lowered(ops, 1, [2]) == [(0, "block", 1)]


@pytest.mark.parametrize(
    ("raw", "match"),
    [
        ({"op": "bogus"}, "not a layout op"),
        ({"op": "permute", "perm": [0, 0]}, "permutation of 0..n-1"),
        ({"op": "shuffle", "dims": [0], "f": ["i0"]}, "f_inv must be a list of index expressions"),
        ({"op": "shuffle", "dims": [0, 1], "f": ["i0"], "f_inv": ["i0"]}, "one expression per"),
        ({"op": "shuffle", "dims": [0], "f": ["i0 ** 2"], "f_inv": ["i0"]}, "uses Pow"),
        ({"op": "block", "dim": 0, "factor": 4, "axis": 1}, r"unknown field\(s\) \['axis'\]"),
        ({"op": "linear_shuffle", "dims": [0], "bits": [2], "matrix": [[1, 1], [1, 1]]}, "singular"),
    ],
)
def test_a_malformed_op_is_refused_by_name(raw: dict[str, object], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        parse_op(raw, "t")


def test_zip_and_unzip_are_group_ops_of_their_own() -> None:
    """Zip fuses arrays into one struct array and unzip splits one; neither is the other's inverse op."""
    assert set(GROUP_OPS.keys()) == {"zip", "unzip"}
    zipped = from_json(Zip, {"name": "uv", "fields": ["u", "v"]}, "", "t")
    assert to_json(zipped) == {"name": "uv", "fields": ["u", "v"]}
    assert "inverse" not in vars(Unzip)
    with pytest.raises(ValueError, match="two or more arrays"):
        from_json(Unzip, {"name": "z", "fields": ["re"]}, "", "t")


def test_an_index_expression_evaluates_with_python_floor_semantics() -> None:
    names = {"i0": -3, "P0": 4}
    assert eval_index("i0 // 2 + i0 % P0", names, "t") == -2 + 1
    assert eval_index("(5 ^ 3) << 1 | min(P0, 7)", names, "t") == ((5 ^ 3) << 1) | 4
    with pytest.raises(ValueError, match="names 'N'"):
        eval_index("N + 1", names, "t")


if __name__ == "__main__":
    test_the_three_schemes_lower_from_their_op_sequences()
    test_a_grid_factor_re_evaluates_at_every_rank_count()
    test_the_default_layout_raises_to_ops_that_lower_back(1)
    test_the_default_layout_raises_to_ops_that_lower_back(2)
    test_the_default_layout_raises_to_ops_that_lower_back(3)
    test_a_non_row_major_local_order_is_refused_with_the_permute_that_fixes_it()
    test_a_sequence_that_cannot_lower_is_refused_by_name([{"op": "pad", "dim": 0, "amount": 4}], "'pad' is not")
    test_a_sequence_that_cannot_lower_is_refused_by_name([block(0, 3), distribute(2)], "must have extent P0 = 4")
    test_a_sequence_that_cannot_lower_is_refused_by_name([block(2, 4)], "dimension 2 does not exist")
    test_every_op_has_its_inverse()
    test_every_registered_op_round_trips_through_json_and_inverts()
    test_a_linear_shuffle_inverts_its_matrix_over_gf2()
    test_unblock_undoes_block_in_the_digit_tuple()
    test_a_malformed_op_is_refused_by_name({"op": "bogus"}, "not a layout op")
    test_a_malformed_op_is_refused_by_name({"op": "permute", "perm": [0, 0]}, "permutation of 0..n-1")
    test_zip_and_unzip_are_group_ops_of_their_own()
    test_an_index_expression_evaluates_with_python_floor_semantics()
