# Copyright 2026 ETH Zurich and the relayout authors.
"""The library-call rules against numpy: a call on a permuted operand's buffer, with its axes renamed,
equals the call on the logical operand."""

import itertools

import numpy as np
import pytest

from relayout.rules import contracted_axes, einsum_operand, flips_transpose, transpose_axes

X = np.arange(2 * 3 * 4, dtype=np.float64).reshape(2, 3, 4)
PERMS = list(itertools.permutations(range(3)))


@pytest.mark.parametrize("perm", PERMS)
def test_transpose_einsum_tensordot_and_sum_absorb_a_permutation(perm: tuple[int, ...]) -> None:
    stored = X.transpose(perm)
    for axes in PERMS:
        np.testing.assert_array_equal(stored.transpose(transpose_axes(axes, perm)), X.transpose(axes))
    w = np.arange(12.0).reshape(3, 4)
    np.testing.assert_array_equal(
        np.einsum(f"{einsum_operand('ijk', perm)},jk->i", stored, w), np.einsum("ijk,jk->i", X, w)
    )
    np.testing.assert_array_equal(
        np.tensordot(stored, w, axes=(contracted_axes((1, 2), perm), (0, 1))), np.tensordot(X, w, axes=((1, 2), (0, 1)))
    )
    np.testing.assert_array_equal(stored.sum(axis=contracted_axes((0, 2), perm)), X.sum(axis=(0, 2)))


def test_a_blas_operand_flips_its_flag_for_a_transpose_only() -> None:
    assert flips_transpose((1, 0))
    assert not flips_transpose((0, 1))
    with pytest.raises(ValueError, match="2-D permutation"):
        flips_transpose((0, 2, 1))


if __name__ == "__main__":
    for perm in PERMS:
        test_transpose_einsum_tensordot_and_sum_absorb_a_permutation(perm)
    test_a_blas_operand_flips_its_flag_for_a_transpose_only()
