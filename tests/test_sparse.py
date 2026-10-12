# Copyright 2026 ETH Zurich and the relayout authors.
"""Sparse formats from one canonical CSR: each format decoded back to a dense matrix equals the CSR's,
and the gather / scatter kernels move values (with a cast) as the numpy gather does."""

import ctypes
import os
import pathlib
import tempfile

import numpy as np
import pytest

from relayout.convert import Gather, compile_host, host_source
from relayout.sparse import Csr, SparseFormat, gather, plan, scatter

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


if __name__ == "__main__":
    for label in LABELS:
        test_every_format_holds_the_csr_matrix(label)
    test_a_malformed_pattern_or_label_is_refused()
    with tempfile.TemporaryDirectory() as tmp:
        test_the_gather_kernels_cast_and_pad_like_numpy(pathlib.Path(tmp))
