# Copyright 2026 ETH Zurich and the relayout authors.
"""The operations a request runs: their effect top to bottom, the grid, and the lowering to one split
per distributed dimension.

A kernel's default distribution is shown as short-form ``distribute`` operations
(``operations_for_axes``), so lowering them must give the default back; a program the lowering does
not cover is refused by name.
"""

import itertools

import pytest

from relayout.ops import (
    ArraySpec,
    AxisSplit,
    Block,
    Cast,
    Dim,
    Distribute,
    Format,
    LinearShuffle,
    Location,
    Pad,
    Permute,
    Place,
    Precision,
    Replicate,
    Requests,
    Scheme,
    Shuffle,
    Split,
    Startup,
    Unblock,
    Unsplit,
    Unzip,
    Zip,
    eval_index,
    gf2_inverse,
    grid_extents,
    inverse_sequence,
    lower,
    operations_for_axes,
    run,
)

AXES = (
    None,
    AxisSplit(0, Scheme.BLOCK, 1),
    AxisSplit(0, Scheme.CYCLIC, 1),
    AxisSplit(0, Scheme.BLOCK_CYCLIC, 4),
    AxisSplit(1, Scheme.BLOCK, 1),
    AxisSplit(1, Scheme.BLOCK_CYCLIC, 3),
)
A2 = [ArraySpec("A", 2)]


def lowered(ops: list, arrays: list[ArraySpec], grid: list[int]) -> dict:
    return lower(run(ops, arrays, grid, "operations"))


def test_the_three_schemes_lower_from_their_primitive_operations() -> None:
    bands = [Split("A", "i0", "P0", ("r", "i0")), Distribute("A", 0, digit="r")]
    assert lowered(bands, A2, [4]) == {"A": [AxisSplit(0, Scheme.BLOCK, 1), None]}
    cyclic = [Block("A", "i1", "P0", ("i1", "r")), Distribute("A", 0, digit="r")]
    assert lowered(cyclic, A2, [4]) == {"A": [None, AxisSplit(0, Scheme.CYCLIC, 1)]}
    blocked = [Block("A", "i0", "8*P0", ("i0", "c")), Block("A", "c", "8", ("r", "b")), Distribute("A", 0, digit="r")]
    assert lowered(blocked, A2, [4]) == {"A": [AxisSplit(0, Scheme.BLOCK_CYCLIC, 8), None]}


def test_the_short_form_is_the_primitive_operations() -> None:
    """``distribute dim scheme`` cuts the whole dimension and distributes its rank digit ``r<dim>``."""
    for scheme, width, split in (
        (Scheme.BLOCK, None, AxisSplit(0, Scheme.BLOCK, 1)),
        (Scheme.CYCLIC, None, AxisSplit(0, Scheme.CYCLIC, 1)),
        (Scheme.BLOCK_CYCLIC, 4, AxisSplit(0, Scheme.BLOCK_CYCLIC, 4)),
    ):
        short = Distribute("A", 0, dim=0, scheme=scheme, block_size=width)
        assert lowered([short], A2, [2]) == {"A": [split, None]}


def test_a_grid_factor_re_evaluates_at_every_rank_count() -> None:
    """One request stays valid when a scaling run re-sizes the grid: ``P0`` is read at each P."""
    ops = [Block("A", "i0", "2*P0", ("i0", "c")), Block("A", "c", "2", ("r", "b")), Distribute("A", 0, digit="r")]
    for p in (1, 2, 4, 8):
        assert lowered(ops, [ArraySpec("A", 1)], [p]) == {"A": [AxisSplit(0, Scheme.BLOCK_CYCLIC, 2)]}


def test_a_grid_is_extents_over_the_rank_count() -> None:
    """A square grid spans only the square rank counts; a 1-D grid every one."""
    square = Requests(Startup(grid=("root(P, 2)", "root(P, 2)")))
    assert square.grid_at(16) == (4, 4)
    assert square.grid_at(8) is None
    assert grid_extents(("P // 2", "2"), 8, "grid") == (4, 2)
    with pytest.raises(ValueError, match="extent below 1"):
        grid_extents(("P // 2", "2"), 1, "grid")
    with pytest.raises(ValueError, match="does not span 8 rank"):
        square.run(A2, 8)


@pytest.mark.parametrize("ndim", [1, 2, 3])
def test_the_default_layout_raises_to_operations_that_lower_back(ndim: int) -> None:
    """``operations_for_axes`` is what a caller is shown; lowering it must be the split it came from."""
    for axes in itertools.product(AXES, repeat=ndim):
        grid_dims = [axis.grid_dim for axis in axes if axis is not None]
        if len(grid_dims) != len(set(grid_dims)):
            continue
        ops = operations_for_axes("A", axes)
        for grid in ([1, 1], [2, 3], [4, 4]):
            expected = None if all(a is None for a in axes) else list(axes)
            assert lowered(list(ops), [ArraySpec("A", ndim)], grid) == {"A": expected}, (axes, grid)


def test_operations_see_what_the_previous_ones_left() -> None:
    """Names a cut makes are usable on the next line; a struct unzipped into two arrays is gone."""
    arrays = [ArraySpec("A", 2), ArraySpec("Z", 1, members=("re", "im")), ArraySpec("x", 1)]
    ops = [
        Cast("A", Precision.FP32),
        Block("A", "i0", "4", ("o", "f")),
        Permute("A", ("o", "f", "i1")),
        Unzip("Z", ("Z_re", "Z_im")),
        Cast("Z_re", Precision.FP32),
        Zip(("Z_re", "Z_im"), "W"),
        Replicate("x"),
        Place("A", Location.DEVICE),
    ]
    state = run(ops, arrays, None, "operations")
    assert sorted(state.arrays) == ["A", "W", "x"]
    assert state.arrays["A"].digits.order() == ("o", "f", "i1")
    assert state.arrays["A"].precision is Precision.FP32
    assert state.arrays["A"].location is Location.DEVICE
    assert state.arrays["W"].members == (("Z_re", Precision.FP32), ("Z_im", None))
    assert state.arrays["x"].replicated


def test_a_zip_of_one_precision_is_a_field_dimension() -> None:
    """Three components of one precision become a last dimension of extent 3 the digit operations place:
    field-outer is SoA, and unzip splits it back; mixed precisions make a struct."""
    xyz = [ArraySpec(name, 1) for name in ("x", "y", "z")]
    soa = [Zip(("x", "y", "z"), "p"), Permute("p", ("i1", "i0"))]
    state = run(soa, xyz, (), "operations")
    assert state.arrays["p"].fields == ("x", "y", "z")
    layout = state.layout("p", ["N", "3"])
    assert layout.view() == ((None, None), (1, 0))
    back = run([Zip(("x", "y", "z"), "p"), Unzip("p", ("u", "v", "w"))], xyz, (), "operations")
    assert sorted(back.arrays) == ["u", "v", "w"]
    mixed = run([Cast("x", Precision.FP32), Zip(("x", "y"), "q")], xyz, (), "operations")
    assert mixed.arrays["q"].members == (("x", Precision.FP32), ("y", None))
    with pytest.raises(ValueError, match="splits into 3 array"):
        run([Zip(("x", "y", "z"), "p"), Unzip("p", ("u", "v"))], xyz, (), "operations")
    first = run([Zip(("x", "y", "z"), "p", dim=Dim(0))], xyz, (), "operations").arrays["p"]
    assert first.field_dim == 0 and first.digits.digits[0].extent == 3
    with pytest.raises(ValueError, match="struct, which has no dim"):
        run([Cast("x", Precision.FP32), Zip(("x", "y"), "q", dim=Dim(0))], xyz, (), "operations")


def test_a_zip_compares_element_types_not_casts() -> None:
    """Problem element types decide too: float64 and float32 make a struct uncast, and a cast to the
    other's type stacks; unzipped parts keep the element type (a complex array's parts its component)."""
    mixed = [ArraySpec("a", 1, element="float64"), ArraySpec("b", 1, element="float32")]
    assert run([Zip(("a", "b"), "z")], mixed, (), "operations").arrays["z"].members
    stacked = run([Cast("a", Precision.FP32), Zip(("a", "b"), "z")], mixed, (), "operations").arrays["z"]
    assert stacked.fields == ("a", "b") and stacked.element_type == "float32"
    complex_z = [ArraySpec("Z", 1, members=("re", "im"), native=True, element="complex128")]
    parts = run([Unzip("Z", ("re", "im"))], complex_z, (), "operations").arrays
    assert [part.element_type for part in parts.values()] == ["float64", "float64"]


def test_unzip_along_a_dimension_splits_an_array() -> None:
    """One array per index of a small dimension, the inverse of a zip along it (the extent of a dimension
    no zip made is the frontend's to check)."""
    state = run([Unzip("q", ("q_rain", "q_ice"), dim=Dim(1))], [ArraySpec("q", 2)], (), "operations")
    assert {name: array.digits.ndim for name, array in state.arrays.items()} == {"q_rain": 1, "q_ice": 1}
    with pytest.raises(ValueError, match="no dimension 2"):
        run([Unzip("q", ("a", "b"), dim=Dim(2))], [ArraySpec("q", 2)], (), "operations")


def test_a_complex_array_is_one_element_and_a_struct() -> None:
    """A native struct (complex) is laid out and cast like any array, and unzips into its members,
    which keep its precision and placement."""
    complex_z = [ArraySpec("Z", 1, members=("re", "im"), native=True)]
    short = Distribute("Z", 0, dim=0, scheme=Scheme.BLOCK)
    assert lowered([Cast("Z", Precision.FP32), short], complex_z, [2]) == {"Z": [AxisSplit(0, Scheme.BLOCK, 1)]}
    ops = [Cast("Z", Precision.FP32), Place("Z", Location.DEVICE), Unzip("Z", ("Z_re", "Z_im"))]
    state = run(ops, complex_z, None, "operations")
    assert [(a.precision, a.location) for a in state.arrays.values()] == [(Precision.FP32, Location.DEVICE)] * 2


def test_a_non_row_major_local_order_is_refused_with_the_permute_that_fixes_it() -> None:
    ops = [Block("A", "i0", "4", ("o", "f")), Permute("A", ("f", "o", "i1"))]
    expected = r'append \{"action": "permute", "array": "A", "order": \[\'o\', \'f\', \'i1\'\]\}'
    with pytest.raises(ValueError, match=expected):
        lowered(ops, A2, [4])


@pytest.mark.parametrize(
    ("ops", "match"),
    [
        ([Pad("A", "i0", "4")], "pad is not implemented yet"),
        ([Zip(("A", "B"), "W")], "zip is not implemented yet"),
        ([Block("A", "i0", "3", ("i0", "r")), Distribute("A", 0, digit="r")], "must have extent P0 = 4"),
        ([Distribute("A", 0, digit="i0")], "has an unknown extent"),
        ([Split("A", "i0", "2", ("a", "b")), Split("A", "a", "2", ("x", "y"))], "already cut"),
        ([Block("A", "i2", "4", ("a", "b"))], "no digit 'i2'"),
        ([Block("A", "i0", "4", ("i1", "b"))], r"name\(s\) \['i1'\] are taken"),
        ([Split("A", "i0", "P1", ("r", "i0")), Distribute("A", 3, digit="r")], "grid_dim 3 does not exist"),
        ([Permute("A", ("i1",))], "must name every digit once"),
        ([Cast("Q", Precision.FP32)], "no array 'Q'"),
        ([Block("M", "i0", "2", ("a", "b"))], "block applies to dense arrays only"),
        ([Format("A", "csr")], "format applies to sparse arrays"),
        ([Distribute("A", 0, dim=0, scheme=Scheme.BLOCK), Replicate("A")], "cannot also be replicated"),
        ([Block("A", "i0", "2", ("o", "f")), Zip(("A", "B"), "W")], "regroups whole arrays"),
        ([Unzip("A", ("a", "b"))], "neither a struct nor zipped"),
        ([Block("A", "i0", "4", ("o", "f")), Replicate("A")], "replicate it before any digit operation"),
        (
            [Block("A", "i0", "8", ("a", "b")), Block("A", "b", "4", ("c", "d")), Unblock("A", ("a", "d"), "4", "x")],
            "not the two digits of one block",
        ),
        (
            [Split("A", "i0", "2", ("a", "b")), Block("A", "b", "4", ("c", "d")), Unsplit("A", ("a", "c"), "2", "x")],
            "not the two digits of one split",
        ),
        ([Place("A", Location.DEVICE), Zip(("A", "B"), "W")], "placed alike"),
        ([Pad("A", "i0", "2"), Block("A", "i0", "4", ("o", "f"))], "padded or shuffled; cut or merge digits before"),
    ],
)
def test_an_operation_that_cannot_apply_is_refused_by_name(ops: list, match: str) -> None:
    arrays = [ArraySpec("A", 2), ArraySpec("B", 2), ArraySpec("M", 2, sparse=True)]
    with pytest.raises(ValueError, match=match):
        lowered(ops, arrays, [4, 2])


@pytest.mark.parametrize(
    ("make", "match"),
    [
        (lambda: Distribute("A", 0), "either the digit, or the dim and scheme"),
        (lambda: Distribute("A", 0, digit="r", dim=0, scheme=Scheme.BLOCK), "either the digit"),
        (lambda: Distribute("A", 0, dim=0), "needs both dim and scheme"),
        (lambda: Distribute("A", 0, dim=0, scheme=Scheme.BLOCK_CYCLIC), "block_size goes with"),
        (lambda: Shuffle("A", ("i0", "i1"), ("i0",), ("i0",)), "one expression per"),
        (lambda: LinearShuffle("A", ("i0",), (2,), ((1, 1), (1, 1))), "singular"),
        (lambda: Zip(("A", "A"), "W"), "names one twice"),
        (lambda: Block("A", "i0", "2", ("x",)), "two digits the cut leaves"),
        (lambda: Block("A", "P0", "2", ("a", "b")), "other than P, P0"),
        (lambda: Block("A", "i0", "2 ** P0", ("a", "b")), "uses Pow"),
    ],
)
def test_a_malformed_operation_is_refused_when_it_is_made(make, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        make()


def test_every_digit_operation_has_its_inverse() -> None:
    """pad <-> unpad, permute <-> the order before it, block <-> unblock, split <-> unsplit, shuffle f <-> f_inv."""
    shuffle = Shuffle("A", ("i0", "f"), ("i0", "f ^ (i0 % 8)"), ("i0", "f ^ (i0 % 8)"))
    ops = (Block("A", "i1", "P0", ("o", "f")), Pad("A", "i0", "4"), Permute("A", ("f", "o", "i0")), shuffle)
    assert inverse_sequence(ops, 2) == (
        shuffle,
        Permute("A", ("i0", "o", "f")),
        ops[1].inverse(()),
        Unblock("A", ("o", "f"), "P0", "i1"),
    )
    assert Split("A", "i0", "P0", ("r", "l")).inverse(()) == Unsplit("A", ("r", "l"), "P0", "i0")
    with pytest.raises(ValueError, match="undone by the gather"):
        inverse_sequence((Split("A", "i0", "2", ("r", "i0")), Distribute("A", 0, digit="r")), 1)


def test_unblock_and_unsplit_undo_their_cuts_in_the_digit_tuple() -> None:
    ops = [
        Block("A", "i0", "4", ("o", "f")),
        Unblock("A", ("o", "f"), "4", "i0"),
        Split("A", "i0", "3", ("a", "b")),
        Unsplit("A", ("a", "b"), "3", "i0"),
        Distribute("A", 0, dim=0, scheme=Scheme.BLOCK),
    ]
    assert lowered(ops, [ArraySpec("A", 1)], [2]) == {"A": [AxisSplit(0, Scheme.BLOCK, 1)]}


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
    assert LinearShuffle("A", ("i0", "i1"), (2, 1), skew).inverse(()).matrix == inverse


def test_a_request_reads_its_precisions_formats_and_locations() -> None:
    requests = Requests(
        Startup(("openblas",), ("P",)),
        (Cast("A", Precision.FP16), Cast("A", Precision.FP32), Format("M", "csr"), Place("A", Location.DEVICE)),
    )
    assert requests.distributed
    assert requests.precisions() == {"A": Precision.FP32}
    assert requests.formats() == {"M": "csr"}
    assert requests.locations() == {"A": Location.DEVICE}


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
    test_the_three_schemes_lower_from_their_primitive_operations()
    test_the_short_form_is_the_primitive_operations()
    test_a_grid_factor_re_evaluates_at_every_rank_count()
    test_a_grid_is_extents_over_the_rank_count()
    test_the_default_layout_raises_to_operations_that_lower_back(1)
    test_the_default_layout_raises_to_operations_that_lower_back(2)
    test_the_default_layout_raises_to_operations_that_lower_back(3)
    test_operations_see_what_the_previous_ones_left()
    test_a_zip_of_one_precision_is_a_field_dimension()
    test_a_zip_compares_element_types_not_casts()
    test_unzip_along_a_dimension_splits_an_array()
    test_a_complex_array_is_one_element_and_a_struct()
    test_a_non_row_major_local_order_is_refused_with_the_permute_that_fixes_it()
    test_an_operation_that_cannot_apply_is_refused_by_name([Pad("A", "i0", "4")], "pad is not implemented yet")
    test_an_operation_that_cannot_apply_is_refused_by_name([Cast("Q", Precision.FP32)], "no array 'Q'")
    test_a_malformed_operation_is_refused_when_it_is_made(lambda: Distribute("A", 0), "either the digit")
    test_every_digit_operation_has_its_inverse()
    test_unblock_and_unsplit_undo_their_cuts_in_the_digit_tuple()
    test_a_linear_shuffle_inverts_its_matrix_over_gf2()
    test_a_request_reads_its_precisions_formats_and_locations()
    test_an_index_expression_evaluates_with_python_floor_semantics()
