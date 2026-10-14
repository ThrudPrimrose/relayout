# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drive the DaCe ports of the SC26 reference kernels through the brute-force layout sweep.

Each kernel enumerates its global layout candidates (dimension permutations of the array whose
orientation is the layout decision), and the sweep compiles/runs/verifies every candidate against
the numpy oracle. The invariant asserted here is CORRECTNESS: every transparent candidate reproduces
the oracle and ``best()`` returns a correct one. Timing is not asserted (noisy on a shared host)."""

from relayout.search.candidates import build
from relayout.corpus import k04_mvt, k11_thomas, k12_dlt_stencil, k15_colstore
from relayout.search.measure import best, compute_timer, sweep, time_call
from tests.layout_candidates import (
    k04_mvt_candidates,
    k11_thomas_candidates,
    k12_dlt_stencil_candidates,
    k15_colstore_candidates,
)


def test_k04_mvt_layout_sweep_all_verify():
    """mvt reads A in both orientations; every global permutation of A reproduces x1/x2."""
    n = 24
    inputs = k04_mvt.make_inputs(n)
    reference = k04_mvt.oracle(inputs["A"], inputs["y1"], inputs["y2"])
    candidates = build(k04_mvt.mvt, k04_mvt_candidates())

    results = sweep(candidates, k04_mvt.run_closure(inputs, n), reference, timer=time_call, reps=1, warmup=0)
    assert {c.name for c in candidates} == {"identity", "A:perm10"}
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert best(results) is not None


def test_k15_colstore_layout_sweep_all_verify():
    """Row-store vs column-store of the table T: both column-sum candidates reproduce the oracle."""
    n, c = 32, 8
    inputs = k15_colstore.make_inputs(n, c)
    reference = k15_colstore.oracle(inputs["T"])
    candidates = build(k15_colstore.colsum, k15_colstore_candidates())

    results = sweep(candidates, k15_colstore.run_closure(inputs, n, c), reference, timer=time_call, reps=1, warmup=0)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert best(results) is not None


def test_k11_thomas_layout_sweep_timed():
    """Batched Thomas solve, interleaved vs system-major coefficient layout. Both verify, and the
    sweep is timed with the compute-region timer (excludes the relayout copies)."""
    k, nb = 12, 40
    inputs = k11_thomas.make_inputs(k, nb)
    reference = k11_thomas.oracle(inputs["a"], inputs["b"], inputs["c"], inputs["d"])
    candidates = build(k11_thomas.thomas, k11_thomas_candidates())

    results = sweep(candidates, k11_thomas.run_closure(inputs, k, nb), reference, timer=compute_timer, reps=2, warmup=1)
    assert {c.name for c in candidates} == {"interleaved", "system_major"}
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    # every correct candidate has a compute-region time recorded
    assert all(r.seconds is not None and r.seconds >= 0.0 for r in results)
    assert best(results) is not None


def test_k12_dlt_stencil_lifted_sweep_all_verify():
    """DLT stencil: x as it is and dimension-lifted to [V, N/V] reproduce the 3-point stencil."""
    n = 32  # divisible by every block factor
    inputs = k12_dlt_stencil.make_inputs(n)
    reference = k12_dlt_stencil.oracle(inputs["x"])
    candidates = build(k12_dlt_stencil.stencil, k12_dlt_stencil_candidates())
    assert {c.name for c in candidates} == {"identity", "x:dlt4", "x:dlt8"}

    results = sweep(candidates, k12_dlt_stencil.run_closure(inputs, n), reference, timer=time_call, reps=1, warmup=0)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert best(results) is not None


if __name__ == "__main__":
    test_k04_mvt_layout_sweep_all_verify()
    test_k15_colstore_layout_sweep_all_verify()
    test_k11_thomas_layout_sweep_timed()
    test_k12_dlt_stencil_lifted_sweep_all_verify()
    print("kernel sweep tests PASS")
