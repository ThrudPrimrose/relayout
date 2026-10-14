# Copyright 2026 ETH Zurich and the relayout authors.
"""The SDFG rewrite on the forms the old per-transformation passes broke on (ported from dace@extended's
layout tests): reductions (WCR) keep accumulating, an array a nested SDFG reads and writes is laid out
once, and an interstate subscript follows the layout."""

import numpy as np
import pytest

import dace
from tests.conftest import variant
from relayout.frontends.sdfg import physical_subset, prepare, relayout_array, rewrite_subscripts
from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Block, Pad, Permute, Shuffle, run

NI, NJ, NK = (dace.symbol(s) for s in ("NI", "NJ", "NK"))


def layout(ops: list, shape: list[str]) -> IndexMap:
    return run(ops, [ArraySpec("A", len(shape))], (), "operations").layout("A", shape)


@dace.program
def reduce_collapsed(A: dace.float64[NI, NJ, NK], out: dace.float64[NI, NK]):  # noqa: N803
    for i, j, k in dace.map[0:NI, 0:NJ, 0:NK]:
        out[i, k] += A[i, j, k]


@dace.program
def reduce_nested(A: dace.float64[NI, NJ, NK], out: dace.float64[NI, NK]):  # noqa: N803
    for i, k in dace.map[0:NI, 0:NK]:
        for j in dace.map[0:NJ]:
            out[i, k] += A[i, j, k]


REDUCTION_LAYOUTS = {
    "permute": ([Permute("A", ("i2", "i0", "i1"))], [Permute("A", ("i1", "i0"))]),
    "block": ([Block("A", "i1", "2", ("o", "f"))], [Block("A", "i0", "4", ("o", "f"))]),
    "pad+shuffle": (
        [Pad("A", "i2", "1")],
        [Block("A", "i0", "4", ("o", "f")), Shuffle("A", ("f",), ("3 - f",), ("3 - f",))],
    ),
}


@pytest.mark.parametrize("program", [reduce_collapsed, reduce_nested])
@pytest.mark.parametrize("case", sorted(REDUCTION_LAYOUTS))
def test_a_reduction_keeps_accumulating(program: object, case: str) -> None:
    a_ops, out_ops = REDUCTION_LAYOUTS[case]
    sdfg = prepare(program.to_sdfg(simplify=True))  # type: ignore[attr-defined]
    variant(sdfg, case.replace("+", "_"))
    relayout_array(sdfg, "A", layout(a_ops, ["NI", "NJ", "NK"]))
    relayout_array(sdfg, "out", layout(out_ops, ["NI", "NK"]))
    a = np.random.default_rng(0).random((8, 5, 6))
    out = np.zeros((8, 6))
    sdfg(A=a, out=out, NI=8, NJ=5, NK=6)
    np.testing.assert_allclose(out, a.sum(axis=1))
    assert any(e.data.wcr is not None for s in sdfg.all_sdfgs_recursive() for st in s.states() for e in st.edges())


def readwrite_nested() -> dace.SDFG:
    """An array a nested SDFG reads and writes under one inner name."""
    inner = dace.SDFG("inner_readwrite")
    inner.add_array("X", [NI], dace.float64)
    state = inner.add_state(is_start_block=True)
    state.add_mapped_tasklet(
        "t", {"i": "0:NI"}, {"a": dace.Memlet("X[i]")}, "b = a + 1.0", {"b": dace.Memlet("X[i]")}, external_edges=True
    )
    outer = dace.SDFG("outer_readwrite")
    outer.add_array("X", [NI], dace.float64, transient=False)
    top = outer.add_state(is_start_block=True)
    node = top.add_nested_sdfg(inner, {"X"}, {"X"})
    top.add_edge(top.add_read("X"), None, node, "X", dace.Memlet("X[0:NI]"))
    top.add_edge(node, "X", top.add_write("X"), None, dace.Memlet("X[0:NI]"))
    return outer


def test_a_nested_read_write_array_is_laid_out_once() -> None:
    sdfg = readwrite_nested()
    relayout_array(sdfg, "X", layout([Pad("A", "i0", "3")], ["NI"]))
    (inner,) = (n.sdfg for n, _ in sdfg.all_nodes_recursive() if isinstance(n, dace.nodes.NestedSDFG))
    assert str(inner.arrays["X"].shape[0]) == "NI + 3"
    x = np.arange(5.0)
    sdfg(X=x, NI=5)
    np.testing.assert_array_equal(x, np.arange(5.0) + 1.0)


def test_an_interstate_subscript_follows_the_layout() -> None:
    blocked = layout([Block("A", "i0", "4", ("o", "f"))], ["N"])
    assert rewrite_subscripts("A[i + 1] * B[i] + A[j]", "A", "laid_A", blocked.physical) == (
        "laid_A[(i + 1) // 4, (i + 1) % 4] * B[i] + laid_A[j // 4, j % 4]"
    )
    box = dace.subsets.Range.from_string("0:N")
    assert str(physical_subset(box, blocked)) == "0:__int_floor(N + 3, 4), 0:4"


if __name__ == "__main__":
    for program in (reduce_collapsed, reduce_nested):
        for case in sorted(REDUCTION_LAYOUTS):
            test_a_reduction_keeps_accumulating(program, case)
    test_a_nested_read_write_array_is_laid_out_once()
    test_an_interstate_subscript_follows_the_layout()
