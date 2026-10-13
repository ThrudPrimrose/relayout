# Copyright 2026 ETH Zurich and the relayout authors.
"""The global assignment: the DP is the enumeration's optimum, the model prices a transposed reader's
layout lower, and every applied trajectory (permuted or blocked, with paid boundaries, read-only
returns and loop bodies) computes what the program computed."""

import itertools
import random

import dace
import numpy as np
import pytest
from conftest import variant

from relayout.frontends.sdfg import prepare
from relayout.search.assign import (
    AssignmentCosts,
    apply_assignment,
    brute_force_trajectories,
    conflict_report,
    eval_costs,
    greedy_assignment,
    model_costs,
    per_array_dp,
)
from relayout.search.cost import LogGP
from relayout.search.layouts import IDENTITY, Layout, blockings, permutations
from relayout.search.program import Kernel, kernel_per_state, line_graph, locked_transitions

N = dace.symbol("N")

CPU = LogGP(L=100e-9, o=0.0, g=5e-9, G=1e-11, line_bytes=64, sector_bytes=64, bw_saturated=1e11, bw_core=2e10)


@dace.program
def conflict(A: dace.float64[N, N], B: dace.float64[N, N], C: dace.float64[N, N], D: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = A[i, j] * 2.0
    for i, j in dace.map[0:N, 0:N]:
        C[i, j] = B[j, i] + A[i, j]
    for i, j in dace.map[0:N, 0:N]:
        D[i, j] = B[j, i] * 0.5 + C[i, N - 1 - j] + A[i, j]


def expected(a: np.ndarray) -> dict[str, np.ndarray]:
    b = 2.0 * a
    c = b.T + a
    return {"B": b, "C": c, "D": 0.5 * b.T + c[:, ::-1] + a}


def kernels_of(name: str) -> tuple[dace.SDFG, list[Kernel]]:
    sdfg = prepare(conflict.to_sdfg(simplify=True))
    variant(sdfg, name)
    kernel_per_state(sdfg)
    return sdfg, line_graph(sdfg)


def run_and_check(sdfg: dace.SDFG, n: int = 16) -> None:
    a = np.random.default_rng(n).random((n, n))
    out = {k: np.zeros((n, n)) for k in "BCD"}
    sdfg(A=a, **out, N=n)
    for k, value in expected(a).items():
        np.testing.assert_allclose(out[k], value, err_msg=k)


ID = Layout(IDENTITY)


def perm(array: str) -> Layout:
    return permutations(array, 2)[1]


def block(array: str) -> Layout:
    return blockings(array, 2, (4,))[1]


def random_costs(rng: random.Random, kernels: int, tags: int) -> AssignmentCosts:
    layouts = {"X": [ID] + [Layout(f"t{k}", perm("X").operations) for k in range(1, tags)]}
    names = [layout.tag for layout in layouts["X"]]
    return AssignmentCosts(
        layouts,
        {("X", k, t): rng.random() for k in range(kernels) for t in names},
        {("X", a, b): rng.random() for a, b in itertools.permutations(names, 2)},
        {"X": rng.random() < 0.5},
        {"X": rng.randrange(kernels)},
    )


def test_the_dp_is_the_enumerations_optimum_and_greedy_no_better() -> None:
    rng = random.Random(0)
    for _ in range(200):
        kernels = rng.randint(1, 5)
        costs = random_costs(rng, kernels, rng.randint(1, 3))
        locked = {k for k in range(1, kernels) if rng.random() < 0.3}
        for changes in (True, False):
            dp = per_array_dp(costs, kernels, changes, locked)["X"]
            oracle = brute_force_trajectories(costs, kernels, changes, locked)["X"]
            assert dp.cost == pytest.approx(oracle.cost) and dp.tags == oracle.tags
        assert (
            greedy_assignment(costs, kernels, locked)["X"].cost
            >= per_array_dp(costs, kernels, True, locked)["X"].cost - 1e-12
        )


def test_the_model_prices_the_transposed_readers_layout_lower() -> None:
    sdfg, kernels = kernels_of("model")
    layouts = {"B": permutations("B", 2)}
    costs = model_costs(sdfg, kernels, layouts, {"N": 256}, CPU, cores=1)
    perm = layouts["B"][1].tag
    assert costs.node_cost[("B", 0, IDENTITY)] < costs.node_cost[("B", 0, perm)]
    assert costs.node_cost[("B", 1, perm)] < costs.node_cost[("B", 1, IDENTITY)]
    assert costs.entry_needed == {"B": False} and costs.last_write == {"B": 0}
    (row,) = conflict_report(costs, len(kernels))
    assert row.conflicting and row.global_cost <= min(row.greedy_cost, row.single_cost)


TRAJECTORIES = {
    # one paid boundary after B's only write: the readers work on the copy
    "boundary": ({"B": [ID, perm("B"), perm("B")]}, {"B": ["B", "B_perm10"]}, 1),
    # B written in its layout (no conversion in: the write covers it), converted back after it
    "exit": ({"B": [perm("B")] * 3}, {"B": ["B_perm10"]}, 1),
    # A is only read: one copy converted in, read again for free after the program's A
    "read_only": ({"A": [perm("A"), ID, perm("A")]}, {"A": ["A_perm10", "A", "A_perm10"]}, 1),
    # a blocked reader of B (the program's B still current after it), and C written in its own layout
    # and converted back for its reader
    "blocked": (
        {"B": [ID, block("B"), ID], "C": [ID, perm("C"), ID]},
        {"B": ["B", "B_block0x4", "B"], "C": ["C_perm10", "C"]},
        2,
    ),
}


@pytest.mark.parametrize("case", list(TRAJECTORIES))
def test_an_applied_trajectory_computes_the_program(case: str) -> None:
    trajectory, holders, conversions = TRAJECTORIES[case]
    sdfg, kernels = kernels_of(case)
    applied = apply_assignment(sdfg, kernels, trajectory)
    sdfg.validate()
    assert applied.holders == holders
    assert len(applied.conversions) == conversions
    for array in holders:
        for kernel, layout in zip(kernels, trajectory[array], strict=True):
            if kernel.touches(f"{array}_{layout.tag}") or not layout.is_identity:
                assert not kernel.touches(array)
    run_and_check(sdfg)


@dace.program
def looped(A: dace.float64[N, N], B: dace.float64[N, N]):
    for _ in range(3):
        for i, j in dace.map[0:N, 0:N]:
            B[i, j] = A[j, i] + B[i, j]
        for i, j in dace.map[0:N, 0:N]:
            A[i, j] = B[j, i] * 0.5


def test_a_loop_body_takes_one_layout() -> None:
    sdfg = looped.to_sdfg(simplify=True)
    kernel_per_state(sdfg)
    kernels = line_graph(sdfg)
    assert locked_transitions(kernels) == {1}
    with pytest.raises(ValueError, match="inside a loop body"):
        apply_assignment(sdfg, kernels, {"B": [ID, perm("B")]})
    apply_assignment(sdfg, kernels, {"B": [perm("B")] * 2})
    rng = np.random.default_rng(1)
    a, b = rng.random((8, 8)), rng.random((8, 8))
    want_a, want_b = a.copy(), b.copy()
    for _ in range(3):
        want_b += want_a.T
        want_a = 0.5 * want_b.T
    sdfg(A=a, B=b, N=8)
    np.testing.assert_allclose(b, want_b)
    np.testing.assert_allclose(a, want_a)


def test_measured_costs_fill_the_table() -> None:
    sdfg, kernels = kernels_of("measured")
    costs = eval_costs(sdfg, kernels, {"B": permutations("B", 2)}, {"N": 64}, CPU, reps=3, warmup=1)
    costs.check(len(kernels))
    assert all(v > 0 for (_, k, _), v in costs.node_cost.items())


if __name__ == "__main__":
    test_the_dp_is_the_enumerations_optimum_and_greedy_no_better()
    test_the_model_prices_the_transposed_readers_layout_lower()
    test_an_applied_trajectory_computes_the_program("boundary")
    test_an_applied_trajectory_computes_the_program("exit")
    test_an_applied_trajectory_computes_the_program("read_only")
    test_an_applied_trajectory_computes_the_program("blocked")
    test_a_loop_body_takes_one_layout()
    test_measured_costs_fill_the_table()
