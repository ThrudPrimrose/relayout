# Copyright 2026 ETH Zurich and the relayout authors.
"""Sparse formats as layouts of one canonical CSR matrix.

A format conversion splits into its structure and its values. :func:`plan` builds the structure once
from the CSR pattern: the format's index buffers, its scalars, and a ``table`` the shape of its value
buffer naming, for every slot, the CSR entry it holds (-1 = padding). Converting values is then the
gather ``values[k] = csr_data[table[k]]`` (0 where padding), and back the scatter
``csr_data[table[k]] = values[k]``: the same kernels for every format (:class:`relayout.convert.Gather`),
fused with a cast like a dense conversion. The canonical CSR has int64 indices, no duplicates and
columns ascending within a row.
"""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["FORMATS", "Csr", "SparseFormat", "SparsePlan", "boundary", "convert", "decode", "gather", "plan", "scatter"]

type Index = npt.NDArray[np.int64]

#: Format name -> its index buffers (roles) in ABI order, before the value buffer ``data``.
FORMATS: dict[str, tuple[str, ...]] = {
    "csr": ("indptr", "indices"),
    "csc": ("indptr", "indices"),
    "coo": ("row", "col"),
    "bsr": ("indptr", "indices"),
    "dia": ("offsets",),
    "ell": ("indices",),
}


@dataclass(frozen=True, slots=True)
class SparseFormat:
    """A format by its label: ``name`` and, for ``bsr``, the square ``block`` edge (``bsr:4``)."""

    name: str
    block: int = 0

    @classmethod
    def parse(cls, label: str) -> "SparseFormat":
        name, _, block = label.partition(":")
        if name not in FORMATS:
            raise ValueError(f"sparse format {label!r}: the formats are {sorted(FORMATS)}")
        if (name == "bsr") != bool(block):
            raise ValueError(f"sparse format {label!r}: bsr takes a block edge (bsr:4), the others none")
        if block and (not block.isdigit() or int(block) < 1):
            raise ValueError(f"sparse format {label!r}: the block edge must be a positive int")
        return cls(name, int(block or 0))

    @property
    def label(self) -> str:
        return f"{self.name}:{self.block}" if self.block else self.name


@dataclass(frozen=True, slots=True)
class Csr:
    """A canonical CSR pattern: ``shape``, ``indptr`` (rows + 1) and ``indices`` (nnz)."""

    shape: tuple[int, int]
    indptr: Index
    indices: Index

    def __post_init__(self) -> None:
        rows, cols = self.shape
        if self.indptr.shape != (rows + 1,) or self.indptr[0] != 0 or self.indptr[-1] != self.indices.size:
            raise ValueError(f"indptr must have {rows + 1} entries from 0 to nnz = {self.indices.size}")
        if self.indices.size and (self.indices.min() < 0 or self.indices.max() >= cols):
            raise ValueError(f"column indices must lie in [0, {cols})")
        boundary = np.zeros(max(self.indices.size - 1, 0), dtype=bool)
        starts = self.indptr[1:-1]
        boundary[starts[(starts > 0) & (starts < self.indices.size)] - 1] = True
        if not ((np.diff(self.indices) > 0) | boundary).all():
            raise ValueError("a canonical CSR has strictly ascending columns within each row")

    @property
    def nnz(self) -> int:
        return int(self.indices.size)

    def rows(self) -> Index:
        """The row of every entry, in storage order."""
        return np.repeat(np.arange(self.shape[0], dtype=np.int64), np.diff(self.indptr))


@dataclass(frozen=True, slots=True)
class SparsePlan:
    """One pattern in one format: its index ``buffers`` by role, its ``scalars``, and the value buffer's
    ``table`` (the CSR entry each slot holds, -1 = padding)."""

    format: SparseFormat
    buffers: dict[str, Index]
    scalars: dict[str, int]
    table: Index


def counts_to_indptr(counts: Index) -> Index:
    return np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64)])


def plan(csr: Csr, fmt: SparseFormat) -> SparsePlan:
    """The structure of ``csr``'s pattern in ``fmt`` and the table its values are gathered by."""
    rows, cols = csr.shape
    entry = np.arange(csr.nnz, dtype=np.int64)
    row, col = csr.rows(), csr.indices
    if fmt.name == "csr":
        return SparsePlan(fmt, {"indptr": csr.indptr.copy(), "indices": col.copy()}, {}, entry)
    if fmt.name == "coo":
        return SparsePlan(fmt, {"row": row, "col": col.copy()}, {}, entry)
    if fmt.name == "csc":
        order = np.argsort(col, kind="stable")  # rows ascend within a column: CSR walks rows in order
        indptr = counts_to_indptr(np.bincount(col, minlength=cols))
        return SparsePlan(fmt, {"indptr": indptr, "indices": row[order]}, {}, order)
    if fmt.name == "bsr":
        b = fmt.block
        if rows % b or cols % b:
            raise ValueError(f"bsr:{b} does not tile the {rows} x {cols} matrix")
        block_cols = cols // b
        key = (row // b) * block_cols + col // b
        blocks, slot = np.unique(key, return_inverse=True)
        table = np.full((blocks.size, b, b), -1, dtype=np.int64)
        table[slot, row % b, col % b] = entry
        indptr = counts_to_indptr(np.bincount(blocks // block_cols, minlength=rows // b))
        buffers = {"indptr": indptr, "indices": blocks % block_cols}
        return SparsePlan(fmt, buffers, {"bs": b, "mb": rows // b, "nnzb": int(blocks.size)}, table)
    if fmt.name == "dia":
        diagonal = col - row
        offsets = np.unique(diagonal)
        table = np.full((offsets.size, cols), -1, dtype=np.int64)
        table[np.searchsorted(offsets, diagonal), col] = entry
        return SparsePlan(fmt, {"offsets": offsets}, {"ndiag": int(offsets.size)}, table)
    width = int(np.diff(csr.indptr).max()) if rows else 0  # ell
    slot = entry - csr.indptr[row]
    indices = np.full((rows, width), -1, dtype=np.int64)
    table = np.full((rows, width), -1, dtype=np.int64)
    indices[row, slot] = col
    table[row, slot] = entry
    return SparsePlan(fmt, {"indices": indices}, {"width": width}, table)


def gather(scheme: SparsePlan, values: npt.NDArray[np.generic]) -> npt.NDArray[np.generic]:
    """``scheme``'s value buffer holding the CSR ``values``, zero in padding (the numpy form of the
    :class:`relayout.convert.Gather` kernel)."""
    out = np.zeros(scheme.table.shape, dtype=values.dtype)
    used = scheme.table >= 0
    out[used] = values[scheme.table[used]]
    return out


def scatter(scheme: SparsePlan, stored: npt.NDArray[np.generic], nnz: int) -> npt.NDArray[np.generic]:
    """The CSR values ``scheme``'s value buffer ``stored`` holds."""
    out = np.zeros(nnz, dtype=stored.dtype)
    used = scheme.table >= 0
    out[scheme.table[used]] = stored[used]
    return out


def coordinates(
    fmt: SparseFormat, buffers: Mapping[str, Index], shape: tuple[int, int], cells: int
) -> tuple[Index, Index, Index]:
    """``(row, col, slot)`` of every value slot ``fmt``'s ``buffers`` hold for a ``shape`` matrix whose value
    buffer has ``cells`` slots: a slot outside the matrix (ell's -1, dia's overhang) is left out."""
    rows, cols = shape
    if fmt.name in ("csr", "csc"):
        outer = np.repeat(np.arange(rows if fmt.name == "csr" else cols, dtype=np.int64), np.diff(buffers["indptr"]))
        inner = np.asarray(buffers["indices"], dtype=np.int64)
        row, col = (outer, inner) if fmt.name == "csr" else (inner, outer)
        return row, col, np.arange(row.size, dtype=np.int64)
    if fmt.name == "coo":
        row = np.asarray(buffers["row"], dtype=np.int64)
        return row, np.asarray(buffers["col"], dtype=np.int64), np.arange(row.size, dtype=np.int64)
    if fmt.name == "bsr":
        b = fmt.block
        block_row = np.repeat(np.arange(rows // b, dtype=np.int64), np.diff(buffers["indptr"]))
        block_col = np.asarray(buffers["indices"], dtype=np.int64)
        k, i, j = np.indices((block_row.size, b, b), dtype=np.int64).reshape(3, -1)
        return block_row[k] * b + i, block_col[k] * b + j, np.arange(k.size, dtype=np.int64)
    if fmt.name == "dia":
        offsets = np.asarray(buffers["offsets"], dtype=np.int64)
        d, col = np.indices((offsets.size, cols), dtype=np.int64).reshape(2, -1)
        row = col - offsets[d]
        inside = (row >= 0) & (row < rows)
        return row[inside], col[inside], np.arange(d.size, dtype=np.int64)[inside]
    indices = np.asarray(buffers["indices"], dtype=np.int64)  # ell
    row = np.repeat(np.arange(rows, dtype=np.int64), indices.shape[1] if indices.ndim == 2 else 0)
    flat = indices.reshape(-1)
    used = flat >= 0
    return row[used], flat[used], np.arange(flat.size, dtype=np.int64)[used]


def decode(
    fmt: SparseFormat, buffers: Mapping[str, Index], values: npt.NDArray[np.generic], shape: tuple[int, int]
) -> tuple[Csr, npt.NDArray[np.generic]]:
    """The canonical CSR of a matrix ``fmt`` stores (index ``buffers``, value buffer ``values``): entries
    sorted by row and column, duplicates summed. A blocked or diagonal format's padding comes back as explicit
    zeros (its stored cells are all the pattern it has)."""
    flat = np.asarray(values).reshape(-1)
    row, col, slot = coordinates(fmt, buffers, shape, flat.size)
    order = np.lexsort((col, row))
    row, col, data = row[order], col[order], flat[slot[order]]
    first = np.ones(row.size, dtype=bool)
    first[1:] = (row[1:] != row[:-1]) | (col[1:] != col[:-1])
    starts = np.flatnonzero(first)
    summed = np.add.reduceat(data, starts) if data.size else data
    indptr = counts_to_indptr(np.bincount(row[starts], minlength=shape[0]))
    return Csr(shape, indptr, col[starts]), summed


def convert(
    given: SparseFormat,
    buffers: Mapping[str, Index],
    values: npt.NDArray[np.generic],
    shape: tuple[int, int],
    wanted: SparseFormat,
) -> tuple[SparsePlan, npt.NDArray[np.generic], Csr]:
    """The matrix ``given`` stores, in ``wanted``: its plan (index buffers, scalars, table), its value buffer,
    and the canonical CSR between them."""
    csr, data = decode(given, buffers, values, shape)
    scheme = plan(csr, wanted)
    return scheme, gather(scheme, data), csr


def boundary(
    program: Callable[..., object],
    name: str,
    given: SparseFormat,
    wanted: SparseFormat,
    shape: tuple[str, str],
    written: bool = False,
) -> Callable[..., object]:
    """``program`` (a Python function or an SDFG, called with keyword arguments) written for sparse array
    ``name`` in ``wanted``, called instead with it in ``given``: the members ``<name>_<role>`` and
    ``<name>_data`` the caller passes in ``given`` (sizes in the arguments named by ``shape``) are converted to
    ``wanted``'s, with its scalars (``<name>_width``, ...), before the call; ``written``, the values the program
    leaves are converted back into the caller's ``<name>_data`` after it."""

    def call(**arguments: object) -> object:
        roles = FORMATS[given.name]
        matrix = (int(arguments[shape[0]]), int(arguments[shape[1]]))  # type: ignore[call-overload]
        buffers = {role: np.asarray(arguments.pop(f"{name}_{role}")) for role in roles}
        data = np.asarray(arguments.pop(f"{name}_data"))
        scheme, stored, csr = convert(given, buffers, data, matrix, wanted)
        arguments |= {f"{name}_{role}": buffer for role, buffer in scheme.buffers.items()}
        arguments |= {f"{name}_{scalar}": value for scalar, value in scheme.scalars.items()}
        arguments[f"{name}_data"] = stored
        result = program(**arguments)
        if written:
            back = plan(csr, given)
            data[...] = gather(back, scatter(scheme, stored, csr.nnz)).reshape(data.shape)
        return result

    return call
