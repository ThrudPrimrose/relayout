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

from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

__all__ = ["FORMATS", "Csr", "SparseFormat", "SparsePlan", "gather", "plan", "scatter"]

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
