"""Layout candidates for a corpus program (:mod:`relayout.corpus`): name a way to lay an array out
(:func:`layouts_of`, :func:`shuffled`, :func:`zip_fields`) and turn a program plus its candidates into the
sweep candidates :func:`relayout.search.measure.sweep` verifies and ranks (:func:`build`).
"""

import functools
import re
from collections.abc import Callable, Mapping, Sequence

import dace

from relayout.passes.sdfg.zip_arrays import zip_arrays
from relayout.ops import Block, Permute, Shuffle
from relayout.search.layouts import Layout, lay_out
from relayout.search.measure import Candidate, candidate_name

#: Lays one candidate out on the SDFG of its program.
type Apply = Callable[[dace.SDFG], object]


def apply_layout(sdfg: dace.SDFG, array: str, layout: Layout) -> None:
    if not layout.is_identity:
        lay_out(sdfg, array, layout)


def layouts_of(array: str, layouts: Sequence[Layout]) -> dict[str, Apply]:
    """One candidate per layout of ``array``, named as the search names it (``identity`` or ``array:tag``)."""
    return {
        candidate_name(array, layout): functools.partial(apply_layout, array=array, layout=layout) for layout in layouts
    }


def shuffled(array: str, tag: str, f: str, f_inv: str) -> Layout:
    """``array``'s first dimension renumbered by the bijection ``f`` of ``i0`` (inverse ``f_inv``)."""
    return Layout(tag, (Shuffle(array, ("i0",), (f,), (f_inv,)),))


def zip_fields(sdfg: dace.SDFG, fields: Sequence[str], into: str, width: int | None = None) -> None:
    """Zip the 1-D ``fields`` into ``into``: AoS ``[N, F]``, or with ``width`` AoSoA ``[N/width, F, width]``."""
    zip_arrays(sdfg, fields, into)
    if width is not None:
        tiles = (Block(into, "i0", str(width), ("o", "l")), Permute(into, ("o", "i1", "l")))
        lay_out(sdfg, into, Layout(f"aosoa{width}", tiles))


def build(program: dace.frontend.python.parser.DaceProgram, candidates: Mapping[str, Apply]) -> list[Candidate]:
    """The sweep's candidates: each a fresh simplified SDFG of ``program`` laid out by its apply, its SDFGs named
    after it (dace keeps one library per build folder in a process)."""

    def make(name: str, apply: Apply) -> dace.SDFG:
        sdfg = program.to_sdfg(simplify=True)
        apply(sdfg)
        suffix = re.sub(r"\W", "_", name)
        for inner in sdfg.all_sdfgs_recursive():
            inner.name = f"{inner.name}_{suffix}"
        return sdfg

    return [Candidate(name, functools.partial(make, name, apply)) for name, apply in candidates.items()]
