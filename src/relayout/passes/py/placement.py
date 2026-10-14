# Copyright 2026 ETH Zurich and the relayout authors.
"""Where each cell of an array's logical index space lives in a target buffer, as Python source: :class:`Laid`
(a layout of its own), :class:`Field` (one index of a zipped array's dimension) or :class:`Member` (one member
of a zipped array of structs). The Python counterpart of :mod:`relayout.passes.sdfg.placement`.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

from relayout.layout import IndexMap

__all__ = ["Field", "Laid", "Member", "Placement"]


class Placement(Protocol):
    """How a logical index of an array reads in the target buffer that holds it."""

    def access(self, target: str, index: Sequence[str]) -> str:
        """The element of ``target`` holding the logical index ``index``."""
        ...

    def view(self, target: str) -> str | None:
        """``target`` read as the logical array without a copy, or None when numpy cannot view it so."""
        ...

    def dtype(self, target: str) -> str:
        """The element type of the array as ``target`` holds it."""
        ...

    def layout(self) -> IndexMap | None:
        """The layout the target stores the array in on its own (an MPI buffer of it may go as it is), else None."""
        ...


@dataclass(frozen=True, slots=True)
class Laid:
    """An array in a buffer of its own laid out by ``layout``."""

    index_map: IndexMap

    def access(self, target: str, index: Sequence[str]) -> str:
        return f"{target}[{', '.join(self.index_map.physical(index))}]"

    def view(self, target: str) -> str | None:
        view = self.index_map.view()
        if view is None:
            return None
        live, order = view
        text = target
        if any(extent is not None for extent in live):
            text += "[" + ", ".join(":" if extent is None else f":{extent}" for extent in live) + "]"
        if order != tuple(range(len(order))):
            text += f".transpose({', '.join(map(str, order))})"
        return text

    def dtype(self, target: str) -> str:
        return f"{target}.dtype"

    def layout(self) -> IndexMap | None:
        return self.index_map


@dataclass(frozen=True, slots=True)
class Field:
    """An array stored at ``index`` of a larger array's dimension ``dim`` (None = after the last)."""

    index: int
    dim: int | None = None

    def access(self, target: str, index: Sequence[str]) -> str:
        at = len(index) if self.dim is None else self.dim
        return f"{target}[{', '.join([*index[:at], str(self.index), *index[at:]])}]"

    def view(self, target: str) -> str | None:
        if self.dim is None:
            return f"{target}[..., {self.index}]"
        return f"{target}[{', '.join([':'] * self.dim + [str(self.index)])}]"

    def dtype(self, target: str) -> str:
        return f"{target}.dtype"

    def layout(self) -> IndexMap | None:
        return None


@dataclass(frozen=True, slots=True)
class Member:
    """An array stored as member ``member`` of a structured array (one numpy field per zipped array)."""

    member: str

    def access(self, target: str, index: Sequence[str]) -> str:
        return f"{target}[{self.member!r}][{', '.join(index)}]"

    def view(self, target: str) -> str | None:
        return f"{target}[{self.member!r}]"

    def dtype(self, target: str) -> str:
        return f"{target}.dtype[{self.member!r}]"

    def layout(self) -> IndexMap | None:
        return None
