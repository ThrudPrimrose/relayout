# Copyright 2026 ETH Zurich and the relayout authors.
"""Sparse formats from one canonical CSR: each format decoded back to a dense matrix equals the CSR's, every
format converts to every other (relayout.sparse.decode / convert), a program written for one format is called
with another through relayout.sparse.boundary, and the gather / scatter kernels move values (with a cast) as
the numpy gather does."""

import ctypes
import os
import pathlib
import tempfile

import numpy as np
import pytest

from relayout.convert import Gather, compile_host, host_source
import dace
from relayout.sparse import Csr, SparseFormat, boundary, convert, gather, plan, scatter

LABELS = ("csr", "csc", "coo", "bsr:2", "bsr:3", "dia", "ell")


def random_csr(rows: int, cols: int, density: float, seed: int) -> tuple[Csr, np.ndarray]:
    rng = np.random.default_rng(seed)
    dense = np.where(rng.random((rows, cols)) < density, rng.standard_normal((rows, cols)), 0.0)
    row, col = np.nonzero(dense)
    indptr = np.concatenate([[0], np.cumsum(np.bincount(row, minlength=rows))]).astype(np.int64)
    return Csr((rows, cols), indptr, col.astype(np.int64)), dense[row, col]


def decode(scheme, data: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """The dense matrix a format's buffers describe, read the way its ABI documents them."""
    rows, cols = shape
    out = np.zeros(shape)
    b = scheme.buffers
    name = scheme.format.name
    if name in ("csr", "csc"):
        for major in range(rows if name == "csr" else cols):
            for k in range(b["indptr"][major], b["indptr"][major + 1]):
                r, c = (major, b["indices"][k]) if name == "csr" else (b["indices"][k], major)
                out[r, c] = data[k]
    elif name == "coo":
        out[b["row"], b["col"]] = data
    elif name == "bsr":
        e = scheme.scalars["bs"]
        for br in range(scheme.scalars["mb"]):
            for k in range(b["indptr"][br], b["indptr"][br + 1]):
                bc = b["indices"][k]
                out[br * e : (br + 1) * e, bc * e : (bc + 1) * e] += data[k]
    elif name == "dia":
        for d, offset in enumerate(b["offsets"]):
            for j in range(cols):
                if 0 <= j - offset < rows:
                    out[j - offset, j] += data[d, j]
    else:
        for r in range(rows):
            for s in range(scheme.scalars["width"]):
                if b["indices"][r, s] >= 0:
                    out[r, b["indices"][r, s]] += data[r, s]
    return out


@pytest.mark.parametrize("label", LABELS)
def test_every_format_holds_the_csr_matrix(label: str) -> None:
    csr, values = random_csr(12, 18, 0.2, seed=len(label))
    dense = np.zeros(csr.shape)
    dense[csr.rows(), csr.indices] = values
    scheme = plan(csr, SparseFormat.parse(label))
    stored = gather(scheme, values)
    np.testing.assert_array_equal(decode(scheme, stored, csr.shape), dense)
    np.testing.assert_array_equal(scatter(scheme, stored, csr.nnz), values)


def test_a_malformed_pattern_or_label_is_refused() -> None:
    with pytest.raises(ValueError, match="ascending columns"):
        Csr((2, 3), np.array([0, 2, 2]), np.array([1, 0]))
    with pytest.raises(ValueError, match="does not tile"):
        plan(random_csr(5, 6, 0.5, 0)[0], SparseFormat.parse("bsr:2"))
    for label in ("bsr", "csr:2", "bsr:0", "hyb"):
        with pytest.raises(ValueError, match="sparse format"):
            SparseFormat.parse(label)


def test_the_gather_kernels_cast_and_pad_like_numpy(tmp_path: pathlib.Path) -> None:
    csr, values = random_csr(10, 10, 0.3, seed=3)
    scheme = plan(csr, SparseFormat.parse("ell"))
    source = host_source([], [Gather("v", "float64", "float32")])
    lib = ctypes.CDLL(str(compile_host(source, tmp_path / "g.so", os.environ.get("CC", "cc"))))
    table = np.ascontiguousarray(scheme.table.ravel())
    stored = np.full(table.size, np.nan, dtype=np.float32)
    pointers = (ctypes.c_void_p(a.ctypes.data) for a in (values, table, stored))
    lib.relayout_gather_v(*pointers, ctypes.c_int64(table.size))
    np.testing.assert_array_equal(stored, gather(scheme, values).astype(np.float32).ravel())
    back = np.zeros_like(values)
    lib.relayout_scatter_v(*(ctypes.c_void_p(a.ctypes.data) for a in (stored, table, back)), ctypes.c_int64(table.size))
    np.testing.assert_array_equal(back, values.astype(np.float32).astype(np.float64))


@pytest.mark.parametrize(("given", "wanted"), [(g, w) for g in LABELS for w in LABELS])
def test_every_format_converts_to_every_other(given: str, wanted: str) -> None:
    csr, values = random_csr(6, 6, 0.35, 5)
    source = plan(csr, SparseFormat.parse(given))
    scheme, stored, _ = convert(
        SparseFormat.parse(given), source.buffers, gather(source, values), (6, 6), SparseFormat.parse(wanted)
    )
    np.testing.assert_allclose(
        decode(scheme, stored, (6, 6)), decode(plan(csr, SparseFormat.parse("csr")), values, (6, 6))
    )


def ell_spmv(
    A_indices: np.ndarray, A_data: np.ndarray, x: np.ndarray, y: np.ndarray, N: int, M: int, A_width: int
) -> None:  # noqa: N803
    """y = A x with A in ELL (padding: index -1)."""
    for r in range(N):
        for s in range(A_width):
            if A_indices[r, s] >= 0:
                y[r] += A_data[r, s] * x[A_indices[r, s]]


NR, NC, W = (dace.symbol(s) for s in ("N", "M", "A_width"))


@dace.program
def ell_spmv_sdfg(A_indices: dace.int64[NR, W], A_data: dace.float64[NR, W], x: dace.float64[NC], y: dace.float64[NR]):  # noqa: N803
    for r in range(NR):
        for s in range(W):
            if A_indices[r, s] >= 0:
                y[r] += A_data[r, s] * x[A_indices[r, s]]


def scale_values(A_indptr: np.ndarray, A_indices: np.ndarray, A_data: np.ndarray, N: int, M: int) -> None:  # noqa: N803
    A_data *= 2.0


def test_a_program_written_for_one_format_is_called_with_another() -> None:
    """An ELL SpMV (Python, and as an SDFG) called with a CSR matrix; and a program writing a CSC matrix's
    values, called with COO, whose values come back."""
    csr, values = random_csr(7, 5, 0.4, 6)
    x = np.random.default_rng(7).random(5)
    want = decode(plan(csr, SparseFormat.parse("csr")), values, (7, 5)) @ x
    given = {"A_indptr": csr.indptr, "A_indices": csr.indices, "A_data": values}
    for program in (ell_spmv, ell_spmv_sdfg.to_sdfg()):
        y = np.zeros(7)
        boundary(program, "A", SparseFormat.parse("csr"), SparseFormat.parse("ell"), ("N", "M"))(
            **given, x=x, y=y, N=7, M=5
        )
        np.testing.assert_allclose(y, want)
    coo = plan(csr, SparseFormat.parse("coo"))
    data = values.copy()
    call = boundary(scale_values, "A", SparseFormat.parse("coo"), SparseFormat.parse("csc"), ("N", "M"), written=True)
    call(A_row=coo.buffers["row"], A_col=coo.buffers["col"], A_data=data, N=7, M=5)
    np.testing.assert_allclose(data, 2.0 * values)
    order = np.random.default_rng(3).permutation(values.size)
    row, col = coo.buffers["row"][order], coo.buffers["col"][order]
    row, col = np.append(row, row[0]), np.append(col, col[0])
    data = np.append(values[order], 1.0)  # unsorted, and its first entry twice
    call(A_row=row, A_col=col, A_data=data, N=7, M=5)
    want = np.append(2.0 * values[order], 0.0)
    want[0] = 2.0 * (values[order][0] + 1.0)
    np.testing.assert_allclose(data, want)


if __name__ == "__main__":
    for label in LABELS:
        test_every_format_holds_the_csr_matrix(label)
    test_a_malformed_pattern_or_label_is_refused()
    with tempfile.TemporaryDirectory() as tmp:
        test_the_gather_kernels_cast_and_pad_like_numpy(pathlib.Path(tmp))
    for given in LABELS:
        for wanted in LABELS:
            test_every_format_converts_to_every_other(given, wanted)
    test_a_program_written_for_one_format_is_called_with_another()
