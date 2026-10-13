# Copyright 2026 ETH Zurich and the relayout authors.
"""The LogGP model: a fit recovers its line, a nest's traffic follows its strides, and the time takes
the binding term."""

import dace
import numpy as np
import pytest

from relayout.search.cost import LogGP, count_nest, fit_message_size, nest_memory_time, replayed_blocks, validate
from relayout.search.program import nest_entries

N = dace.symbol("N")

CPU = LogGP(L=100e-9, o=0.0, g=5e-9, G=1e-11, line_bytes=64, sector_bytes=64, bw_saturated=1e11, bw_core=2e10)


@dace.program
def copy_rows(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = A[i, j]


@dace.program
def copy_columns(A: dace.float64[N, N], B: dace.float64[N, N]):
    for i, j in dace.map[0:N, 0:N]:
        B[i, j] = A[j, i]


def counts(program: dace.frontend.python.parser.DaceProgram) -> dict[str, tuple[float, float]]:
    sdfg = program.to_sdfg(simplify=True)
    (state,) = [s for s in sdfg.states() if nest_entries(s)]
    nest = count_nest(state, nest_entries(state)[0], CPU.line_bytes, CPU.sector_bytes)
    return {a: (nest.messages({"N": 64}, a), nest.bytes_moved({"N": 64}, a)) for a in nest.arrays}


def test_a_fit_recovers_its_line_and_validates_against_the_parameters() -> None:
    sizes = [64, 256, 1024, 4096]
    fit = fit_message_size(sizes, [CPU.L + n / CPU.bw_core for n in sizes])
    assert fit.alpha == pytest.approx(CPU.L) and fit.beta == pytest.approx(1 / CPU.bw_core) and fit.residual < 1e-9
    assert validate(CPU, fit, peak_bytes_per_s=2e11, knee=10.0) == []
    assert len(validate(CPU, fit, peak_bytes_per_s=5e10, knee=40.0)) == 2


def test_a_transposed_read_touches_a_line_per_element_and_a_write_moves_twice() -> None:
    rows, columns = counts(copy_rows), counts(copy_columns)
    elements = 64 * 64
    assert rows["A"][0] == 1 + 63 + 64 * 63 / 8  # 8 doubles a line, a new line per row
    assert columns["A"][0] == pytest.approx(elements, rel=0.02)
    assert rows["B"][1] == pytest.approx(2 * rows["A"][1])


def test_the_time_is_the_binding_term() -> None:
    assert nest_memory_time(CPU, 1e9, 1.0, 1.0) == pytest.approx(1e9 * CPU.G)
    assert nest_memory_time(CPU, 64.0, 1e6, 1.0) == pytest.approx(1e6 * CPU.L)


def test_replayed_blocks_bound_an_indirect_access() -> None:
    assert replayed_blocks(np.array([0, 1, 8, 9, 0, 1]), 8) == (pytest.approx(3 / 6), pytest.approx(2 / 6))


if __name__ == "__main__":
    test_a_fit_recovers_its_line_and_validates_against_the_parameters()
    test_a_transposed_read_touches_a_line_per_element_and_a_write_moves_twice()
    test_the_time_is_the_binding_term()
    test_replayed_blocks_bound_an_indirect_access()
