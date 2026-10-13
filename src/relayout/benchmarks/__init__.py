"""Layout benchmarks: the SC26 layout witness kernels, copied from spcl/dace@extended tests/transformations/layout/kernels.

Each module holds a DaCe program, a numpy oracle, make_inputs, and the layout candidates it exercises
(``candidates()``: a name per way to lay the program out, and ``run_closure``), so
:func:`relayout.search.measure.sweep` can verify and rank them over :func:`build`. Later these move into hpcagent-bench as layout variants of its kernels.

ORDER runs from simple to complicated: dense elementwise, then transposes and AoS/SoA, then reductions
where the layout is global across nests, then indirection, then the OMEN cross-phase kernels.
"""

import functools
import re
from collections.abc import Callable, Mapping, Sequence

import dace

from relayout.frontends.sdfg import zip_arrays
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


ORDER = (
    "k01_stream_triad",  # Block on read operands; bandwidth instrument, no layout win expected
    "e1_matrix_add",  # Permute + Block on one operand; SC26 running example
    "k02_transpose_blocked",  # Block/Permute; GPU shared-memory transpose, swizzle candidate
    "k15_colstore",  # row vs column store
    "k03_complex_conjugate_zip",  # Zip: AoS / SoA / AoSoA of complex
    "k09_particle_field_affinity",  # Zip/Block AoSoA of particle fields
    "k13_conv1x1_channel_layouts",  # NCHW / NHWC / nChw16c
    "k04_mvt",  # one Permute must serve two nests
    "atax",  # same, PolyBench
    "bicg",  # same, PolyBench
    "k12_dlt_stencil",  # dimension-lifted transposition of a 1D stencil
    "k11_thomas",  # batched tridiagonal, interleaved vs system-major
    "k16_nbody_force",  # layout + tiling over O(N^2)
    "k17_knn_bruteforce",  # layout + tiling over an argmin reduction
    "k14_eytzinger_search",  # Shuffle (value permutation)
    "k07_icon_stencil",  # vertical-first vs horizontal-first
    "icon_one_loop",  # ICON vertical stencil, multi-output
    "k05_spmv",  # ELL gather: Pad + Block + Permute
    "k06_gather_accumulate_scatter",  # Shuffle through gather/scatter (QE addusxx_g)
    "k08_mesh_scatter",  # Pad / Permute / Shuffle through indirection (OP2 res_calc)
    "e6_indirect_stencil",  # ICON connectivity gather
    "icon_zekinh",  # double indirection
    "k10_omen_windowed_contraction",  # OMEN SSE window, cross-phase Permute
    "k18_scattering_self_energies",  # k10 with the neighbor indirection (npbench sselfeng)
)
