# Copyright 2026 ETH Zurich and the relayout authors.
"""Where each cell of an array's logical index space lives in a target buffer: :class:`Laid` (a layout of
its own) or :class:`Field` (one index of a zipped or split array's dimension), and which logical boxes
stay one contiguous run there.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from dace import data, subsets, symbolic

from relayout.layout import IndexMap, divides
from relayout.passes.sdfg.common import LayoutError, as_range, expr

__all__ = ["Field", "Laid", "Placement", "feeding_dims", "in_order", "physical_subset", "trailing_axes"]


def feeding_dims(layout: IndexMap, axis: int) -> set[int]:
    """The logical dimensions physical axis ``axis``'s value depends on: its own, and through a
    renumbering every dimension of the axes renumbered with it."""
    name = layout.axes[axis].name
    group = {name}.union(*(set(r.axes) for r in layout.renumbers if name in r.axes))
    return {a.dim for a in layout.axes if a.name in group}


def physical_subset(subset: subsets.Subset | None, layout: IndexMap) -> subsets.Range:
    """``subset`` (one range per logical dimension) as a range of the laid-out buffer: exact for a point
    and for any range of a permuted or padded array, else the whole extent of every axis a ranged
    dimension spreads over."""
    ranges = list(as_range(subset).ranges)
    if len(ranges) != len(layout.shape):
        raise LayoutError(f"a subset of rank {len(ranges)} on a rank-{len(layout.shape)} array: {subset}")
    if layout.view() is not None:
        return subsets.Range([ranges[axis.dim] for axis in layout.axes])
    point = [symbolic.equal_valued(begin, end) for begin, end, _ in ranges]
    position = layout.physical([str(begin) for begin, _, _ in ranges])
    out = []
    for k, (value, extent) in enumerate(zip(position, layout.extents(), strict=True)):
        if all(point[dim] for dim in feeding_dims(layout, k)):
            out.append((expr(value), expr(value), 1))
        else:
            out.append((0, expr(extent) - 1, 1))
    return subsets.Range(out)


class Placement(Protocol):
    """Where each cell of an array's logical index space lives in a target buffer: a layout of its own
    (:class:`Laid`) or one index of a zipped array's field dimension (:class:`Field`)."""

    def point(self, index: Sequence[str]) -> list[str]:
        """The target position of the logical index ``index``."""
        ...

    def box(self, subset: subsets.Subset | None) -> subsets.Range:
        """The target range covering the logical box ``subset``."""
        ...

    def view_strides(self, varying: Sequence[int], target: data.Array) -> list[object] | None:
        """The strides a view of a box (its ``varying`` dimensions, in order) takes in ``target``, or None
        when the box is no strided slice of it."""
        ...

    def in_order(self, subset: subsets.Subset | None, target: data.Array) -> bool:
        """Whether the target cells of the box are one contiguous run in the box's own order."""
        ...

    def permutation(self) -> tuple[int, ...] | None:
        """The axis permutation the target stores the array in (a library rule's input), else None."""
        ...

    def nested(self, shape: Sequence[str]) -> "Placement":
        """The same placement for a nested SDFG's array of logical ``shape``."""
        ...


@dataclass(frozen=True, slots=True)
class Laid:
    """An array in a buffer of its own laid out by ``layout``."""

    layout: IndexMap

    def point(self, index: Sequence[str]) -> list[str]:
        return list(self.layout.physical(index))

    def box(self, subset: subsets.Subset | None) -> subsets.Range:
        return physical_subset(subset, self.layout)

    def view_strides(self, varying: Sequence[int], target: data.Array) -> list[object] | None:
        if self.layout.view() is None:
            return None
        axis_of = {axis.dim: k for k, axis in enumerate(self.layout.axes)}
        return [target.strides[axis_of[d]] for d in varying]

    def in_order(self, subset: subsets.Subset | None, target: data.Array) -> bool:
        return in_order(subset, self.layout)

    def permutation(self) -> tuple[int, ...] | None:
        return self.layout.permutation()

    def nested(self, shape: Sequence[str]) -> "Laid":
        return Laid(self.layout.with_shape(shape))


@dataclass(frozen=True, slots=True)
class Field:
    """An array stored at ``index`` of a larger array's dimension ``dim`` (None = after the last): a field
    of a zipped array, or one part of an array split along a dimension."""

    index: int
    dim: int | None = None

    def insert[T](self, values: Sequence[T], value: T) -> list[T]:
        at = len(values) if self.dim is None else self.dim
        return [*values[:at], value, *values[at:]]

    def point(self, index: Sequence[str]) -> list[str]:
        return self.insert(index, str(self.index))

    def box(self, subset: subsets.Subset | None) -> subsets.Range:
        return subsets.Range(self.insert(list(as_range(subset).ranges), (self.index, self.index, 1)))

    def view_strides(self, varying: Sequence[int], target: data.Array) -> list[object] | None:
        dims = self.insert(list(range(len(target.shape) - 1)), -1)
        return [target.strides[dims.index(d)] for d in varying]

    def in_order(self, subset: subsets.Subset | None, target: data.Array) -> bool:
        return all(symbolic.equal_valued(b, e) for b, e, _ in as_range(subset).ranges) or symbolic.equal_valued(
            target.shape[-1], 1
        )

    def permutation(self) -> tuple[int, ...] | None:
        return None

    def nested(self, shape: Sequence[str]) -> "Field":
        return self


def in_order(subset: subsets.Subset | None, layout: IndexMap) -> bool:
    """Whether the laid-out cells of the logical box ``subset`` are one contiguous run in the box's own
    (row-major) order, so a transfer can use them as they are: a permuted or padded layout whose
    varying axes keep their order and are whole after the first; a cut never (unless a point)."""
    ranges = list(as_range(subset).ranges)
    view = layout.view()
    if view is None:
        suffix = trailing_axes(subset, layout)
        if suffix is None:
            return False
        renumbered = {name for renumber in layout.renumbers for name in renumber.axes}
        axes = [layout.axes[k] for k in suffix]
        clean = not any(a.pad or a.name in renumbered for a in axes)
        ordered = [(a.dim, -a.stride) for a in axes] == sorted((a.dim, -a.stride) for a in axes)
        cuts = [[a for a in layout.axes if a.dim == d] for d in {a.dim for a in axes}]
        dense = all(divides(own, layout.shape[own[0].dim]) for own in cuts if len(own) > 1)  # no ragged dead cells
        return clean and ordered and dense
    extents = [expr(extent) for extent in layout.extents()]
    physical = [ranges[axis.dim] for axis in layout.axes]
    varying = [k for k, (begin, end, _) in enumerate(physical) if not symbolic.equal_valued(begin, end)]
    if not varying:
        return True
    dims = [layout.axes[k].dim for k in varying]
    whole = all(
        symbolic.equal_valued(physical[k][0], 0) and symbolic.equal_valued(physical[k][1], extents[k] - 1)
        for k in range(varying[0] + 1, len(physical))
    )
    return dims == sorted(dims) and whole


def trailing_axes(subset: subsets.Subset | None, layout: IndexMap) -> list[int] | None:
    """The physical axes the logical box ``subset`` spans, when its cells are one contiguous run of the
    laid-out buffer (dead or renumbered cells included): every ranged dimension whole, and the axes it
    feeds the trailing ones, fed by ranged dimensions only. None when the run is not contiguous; [] for a point."""
    ranges = list(as_range(subset).ranges)
    ranged = {d for d, (begin, end, _) in enumerate(ranges) if not symbolic.equal_valued(begin, end)}
    whole = all(
        symbolic.equal_valued(ranges[d][0], 0) and symbolic.equal_valued(ranges[d][1], expr(layout.shape[d]) - 1)
        for d in ranged
    )
    spanned = [k for k in range(len(layout.axes)) if feeding_dims(layout, k) & ranged]
    if not whole or any(not feeding_dims(layout, k) <= ranged for k in spanned):
        return None
    return spanned if spanned == list(range(len(layout.axes) - len(spanned), len(layout.axes))) else None
