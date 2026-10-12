# Copyright 2026 ETH Zurich and the relayout authors.
"""The conversion kernels against numpy: element casts bit for bit (ml_dtypes for bfloat16 and fp8),
layouts cell for cell, both directions, built with the host C compiler (``$CC``, default ``cc``); the
device kernels when cupy has a GPU."""

import ctypes
import itertools
import os
import pathlib
import tempfile

import ml_dtypes
import numpy as np
import pytest

from relayout.convert import Conversion, compile_host, device_source, host_source
from relayout.layout import IndexMap
from relayout.ops import ArraySpec, Block, Pad, Permute, eval_index, run

NARROW = {
    "float16": np.dtype(np.float16),
    "bfloat16": np.dtype(ml_dtypes.bfloat16),
    "float8_e4m3fn": np.dtype(ml_dtypes.float8_e4m3fn),
    "float8_e5m2": np.dtype(ml_dtypes.float8_e5m2),
}


def build(conversions: list[Conversion], tmp: pathlib.Path) -> ctypes.CDLL:
    return ctypes.CDLL(str(compile_host(host_source(conversions), tmp / "convert.so", os.environ.get("CC", "cc"))))


def call(lib: ctypes.CDLL, symbol: str, conversion: Conversion, src: np.ndarray, dst: np.ndarray, **sizes: int) -> None:
    """Kernel ``symbol`` of ``conversion`` on ``src`` -> ``dst``, the shape's symbols bound by name."""
    fn = lib[symbol]
    fn.restype = None
    args = (ctypes.c_int64(sizes[name]) for name in conversion.symbols())
    fn(ctypes.c_void_p(src.ctypes.data), ctypes.c_void_p(dst.ctypes.data), *args)


def probes(source: np.dtype) -> np.ndarray:
    """Random values over many binades, every format's edges, ties, subnormals, signed zero, Inf, NaN."""
    rng = np.random.default_rng(7)
    values = rng.standard_normal(4000) * np.exp2(rng.integers(-30, 20, 4000))
    edges = [0.0, -0.0, np.inf, -np.inf, np.nan, 448.0, 449.0, 464.0, 65504.0, 65519.9, 65520.0, 57344.0, 61440.0]
    for fmt in NARROW.values():
        info = ml_dtypes.finfo(fmt)
        grid = np.array([info.tiny, info.smallest_subnormal, float(info.max)], dtype=np.float64)
        edges += [*grid, *(grid * 1.5), *(grid / 2), *(grid * 0.75)]
        steps = np.arange(-16, 16, dtype=np.float64)
        edges += list((steps + 0.5) * float(info.eps))  # ties between neighbours near 1
    with np.errstate(over="ignore"):  # the overflow edges are the point
        return np.concatenate([values, edges]).astype(source)


@pytest.mark.parametrize("source", ["float64", "float32"])
def test_every_narrow_format_rounds_once_like_numpy(source: str, tmp_path: pathlib.Path) -> None:
    flat = IndexMap.identity(["n"])
    conversions = [Conversion(f"to_{name}", flat, source, name) for name in NARROW]
    lib = build(conversions, tmp_path)
    values = probes(np.dtype(source))
    for name, dtype in NARROW.items():
        out = np.empty(values.size, dtype=dtype)
        call(lib, f"relayout_in_to_{name}", conversions[0], values, out, n=values.size)
        with np.errstate(over="ignore"):
            expected = values.astype(dtype)
        bits = f"u{dtype.itemsize}"
        both_nan = np.isnan(out.astype(np.float64)) & np.isnan(expected.astype(np.float64))
        same = (out.view(bits) == expected.view(bits)) | both_nan
        bad = np.flatnonzero(~same)
        assert bad.size == 0, f"{source} -> {name}: {values[bad[:5]]} gave {out[bad[:5]]}, numpy {expected[bad[:5]]}"
        back = np.empty(values.size, dtype=np.dtype(source))
        call(lib, f"relayout_out_to_{name}", conversions[0], out, back, n=values.size)
        np.testing.assert_array_equal(back, expected.astype(np.dtype(source)))


def layout_for(ops: list, shape: tuple[int, ...]) -> IndexMap:
    return run(ops, [ArraySpec("A", len(shape))], (), "operations").layout("A", ["N", "M"])


def placed(layout: IndexMap, logical: np.ndarray) -> np.ndarray:
    """The laid-out buffer of ``logical`` computed cell by cell from the layout's own index maps."""
    names = {"N": logical.shape[0], "M": logical.shape[1]}
    extents = tuple(eval_index(e, names, "t") for e in layout.extents())
    out = np.zeros(extents, dtype=logical.dtype)
    position = layout.physical(["x0", "x1"])
    for index in itertools.product(*(range(s) for s in logical.shape)):
        cell = tuple(eval_index(e, {**names, "x0": index[0], "x1": index[1]}, "t") for e in position)
        out[cell] = logical[index]
    return out


def test_a_layout_and_a_cast_are_one_pass_both_ways(tmp_path: pathlib.Path) -> None:
    ops = [Block("A", "i1", "4", ("o", "f")), Pad("A", "i0", "3"), Permute("A", ("o", "i0", "f"))]
    layout = layout_for(ops, (5, 10))
    conversion = Conversion("A", layout, "float64", "float32")
    lib = build([conversion], tmp_path)
    logical = np.random.default_rng(1).standard_normal((5, 10))
    stored = np.full(placed(layout, logical).shape, np.nan, dtype=np.float32)
    call(lib, "relayout_in_A", conversion, logical, stored, N=5, M=10)
    np.testing.assert_array_equal(stored, placed(layout, logical.astype(np.float32)))
    back = np.full_like(logical, np.nan)
    call(lib, "relayout_out_A", conversion, stored, back, N=5, M=10)
    np.testing.assert_array_equal(back, logical.astype(np.float32).astype(np.float64))


def test_complex_arrays_cast_their_parts(tmp_path: pathlib.Path) -> None:
    conversion = Conversion("z", layout_for([Permute("A", ("i1", "i0"))], (3, 4)), "complex128", "complex64")
    lib = build([conversion], tmp_path)
    z = np.random.default_rng(2).standard_normal((3, 4)) + 1j * np.random.default_rng(3).standard_normal((3, 4))
    stored = np.empty((4, 3), dtype=np.complex64)
    call(lib, "relayout_in_z", conversion, z, stored, N=3, M=4)
    np.testing.assert_array_equal(stored, z.T.astype(np.complex64))


def test_the_device_kernels_match_the_host(tmp_path: pathlib.Path) -> None:
    cp = pytest.importorskip("cupy")
    if cp.cuda.runtime.getDeviceCount() == 0:
        pytest.skip("no GPU")
    ops = [Block("A", "i1", "4", ("o", "f")), Pad("A", "i0", "3"), Permute("A", ("o", "i0", "f"))]
    layout = layout_for(ops, (5, 10))
    module = cp.RawModule(code=device_source([Conversion("A", layout, "float64", "bfloat16")]), options=("-std=c++17",))
    logical = np.random.default_rng(4).standard_normal((5, 10))
    expected = placed(layout, logical.astype(ml_dtypes.bfloat16)).view(np.uint16)
    stored = cp.zeros(expected.shape, dtype=cp.uint16)
    module.get_function("relayout_in_A")((32,), (128,), (cp.asarray(logical), stored, np.int64(10), np.int64(5)))
    np.testing.assert_array_equal(cp.asnumpy(stored), expected)


if __name__ == "__main__":
    for source in ("float64", "float32"):
        with tempfile.TemporaryDirectory() as tmp:
            test_every_narrow_format_rounds_once_like_numpy(source, pathlib.Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_a_layout_and_a_cast_are_one_pass_both_ways(pathlib.Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        test_complex_arrays_cast_their_parts(pathlib.Path(tmp))
    with tempfile.TemporaryDirectory() as tmp:
        try:
            test_the_device_kernels_match_the_host(pathlib.Path(tmp))
        except pytest.skip.Exception as skipped:
            print(f"skipped: {skipped}")
