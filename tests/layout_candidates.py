# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""The layout candidates of every :mod:`relayout.corpus` program: a name per way to lay the program out
(:func:`relayout.search.candidates.build` turns them into sweep candidates). ``CANDIDATES`` maps a corpus
module name to the function that returns its candidates."""

import itertools

from relayout.corpus import k05_spmv, k09_particle_field_affinity
from relayout.search.candidates import apply_layout, layouts_of, shuffled, zip_fields
from relayout.passes.sdfg.prepare import prepare
from relayout.ops import Block, Pad, Permute, Shuffle
from relayout.search.layouts import IDENTITY, Layout, blockings, permutations


def atax_candidates():
    """The global layout candidates for this kernel: every dimension permutation of the shared ``A``
    (row-major vs column-major storage). All are transparent, so every candidate reproduces the
    oracle."""
    return layouts_of("A", permutations("A", 2))


def bicg_candidates():
    """The global layout candidates for this kernel: every dimension permutation of ``A`` (row-major
    vs column-major), the read-orientation conflict the pair nest exposes."""
    return layouts_of("A", permutations("A", 2))


def e1_matrix_add_candidates():
    """The Permute family (row- vs column-major storage of ``B``) and the Block family (rows of ``B`` in
    contiguous ``[T, N]`` panels for T in {8, 16, 32})."""
    cands = layouts_of("B", permutations("B", 2))
    cands.update(layouts_of("B", blockings("B", 2, (8, 16, 32))))
    return cands


def e6_indirect_stencil_candidates():
    """The global layout candidates, each a single-array dimension permutation (identity included):

      * ``w`` / ``z_w_v`` -- gathered field orientation, ``[N, NL]`` (horizontal-first) vs ``[NL, N]``
        (vertical-first). The permute carries through the indirection's nested SDFG.
      * ``cell_idx`` / ``vert_idx`` -- connectivity-table layout, ``[NE, 2]`` (AoS) vs ``[2, NE]`` (SoA).

    Every permute is transparent, so all candidates reproduce the oracle; the sweep only picks the fastest physical layout.
    """
    cands = {}
    for arr in ("w", "z_w_v", "cell_idx", "vert_idx"):
        cands.update(layouts_of(arr, permutations(arr, 2)))
    return cands


def icon_one_loop_candidates():
    """The global layout candidates: every dimension permutation of each read field (identity
    included). ``vn`` is the vertical-stride-sensitive field read at ``jk`` and ``jk-1``; ``vt`` and
    ``wgtfac_e`` are the level-difference operands. Every permute is transparent, so all candidates
    reproduce the oracle -- the sweep only picks the fastest."""
    cands = {}
    for arr in ("vn", "vt", "wgtfac_e"):
        cands.update(layouts_of(arr, permutations(arr, 3)))
    return cands


def icon_zekinh_candidates():
    """The global layout candidates, each a single-array dimension permutation (identity included):

      * ``z_kin_hor_e`` -- gathered-field orientation over ``[NB, NLEV, NPROMA]`` (blocked ICON default
        vs level-contiguous, etc.). The permute carries through the double indirection's nested SDFG.
      * ``edge_idx`` / ``edge_blk`` -- connectivity-table orientation over ``[NB, NPROMA, 3]``. Read by
        affine subscripts, so every permutation carries (like e6's ``cell_idx`` / ``vert_idx``).

    Every permute is transparent, so all candidates reproduce the oracle; the sweep only picks the fastest physical layout.

    """
    cands = {}
    for arr in ("z_kin_hor_e", "edge_idx", "edge_blk"):
        cands.update(layouts_of(arr, permutations(arr, 3)))
    return cands


def k01_stream_triad_candidates():
    """Block family over each read operand (unblocked + ``[N/T, T]`` tiles for T in {8, 16})."""
    cands = {}
    for arr in ("b", "c"):
        blocks = [Layout(f"block{t}", (Block(arr, "i0", str(t), ("o", "f")),)) for t in (8, 16)]
        cands.update(layouts_of(arr, [Layout(IDENTITY), *blocks]))
    return cands


def k02_transpose_blocked_candidates():
    """The Permute family (row- vs column-major storage of ``A``) and the Block family (rows of ``A`` in
    contiguous ``[T, N]`` panels for T in {8, 16, 32})."""
    cands = layouts_of("A", permutations("A", 2))
    cands.update(layouts_of("A", blockings("A", 2, (8, 16, 32))))
    return cands


def k03_complex_conjugate_zip_candidates():
    """AoS / SoA / AoSoA layouts of the (re, im) complex array: Zip, and Zip then Block (AoSoA)."""
    return {
        "unzipped_soa": lambda sdfg: None,
        "zipped_aos": lambda sdfg: zip_fields(sdfg, ["re", "im"], "cplx"),
        "aosoa_4": lambda sdfg: zip_fields(sdfg, ["re", "im"], "cplx", 4),
        "aosoa_8": lambda sdfg: zip_fields(sdfg, ["re", "im"], "cplx", 8),
    }


def k04_mvt_candidates():
    """The global layout candidates for this kernel: every dimension permutation of ``A``."""
    return layouts_of("A", permutations("A", 2))


def k05_spmv_candidates():
    """The global layout candidates for the ELL matrices:

      * ``eval_:perm10`` / ``ecol:perm10`` -- row-major ELL ``[NR, W]`` vs column-major SELL slab ``[W, NR]``
        (Permute of the value / index matrix).
      * ``eval_:block0x*`` -- row tiling of the value matrix (Block of the row axis ``NR``).
      * ``eval_:pad_w`` -- grow the ragged width ``W`` to ``2 * W`` lanes (Pad); the extra lanes are never
        read (the map iterates ``0:W``), so the pad is transparent.

    Every candidate is transparent, so all reproduce the oracle."""
    cands = layouts_of("eval_", permutations("eval_", 2))
    cands.update(layouts_of("ecol", permutations("ecol", 2)))
    cands.update(layouts_of("eval_", blockings("eval_", 2, (8, 16, 32))))
    cands.update(layouts_of("eval_", [Layout("pad_w", (Pad("eval_", "i1", str(k05_spmv.W)),))]))
    return cands


def k06_gather_accumulate_scatter_candidates():
    """The layout candidates: renumber ``x`` by each bijection ``sigma`` (plus the unshuffled base). The
    XOR swizzle ``i ^ 3`` permutes within each block of 4 (``3 - f`` on the block offset ``f``, so ``N``
    a multiple of 4); the cyclic shift holds for any ``N``. Every Shuffle is transparent, so all candidates
    reproduce the oracle."""
    xor = Layout("gas_xor", (Block("x", "i0", "4", ("o", "f")), Shuffle("x", ("f",), ("3 - f",), ("3 - f",))))
    cyclic = shuffled("x", "gas_cyc", "(i0 + 1) % N", "(i0 + N - 1) % N")
    return layouts_of("x", [Layout(IDENTITY), xor, cyclic])


def k07_icon_stencil_candidates():
    """The global layout candidates: every dimension permutation of ``A`` (vertical-first, etc.)."""
    return layouts_of("A", permutations("A", 3))


def k08_mesh_scatter_candidates():
    """The global layout candidates:

      * ``identity``    -- AoS ``[NC, V]`` sign table, natural cell order.
      * ``tsg:perm10``  -- SoA ``[V, NC]`` sign table (Permute).
      * ``tsg:pad2`` / ``tsg:pad4`` -- grow the valence axis to ``[NC, V+p]`` (Pad).
      * ``x:mesh_cyc`` / ``x:mesh_rev`` -- renumber the cell field ``x`` by a closed-form mesh reorder
        (Shuffle): a cyclic shift and a reflection, bijections on ``[0, NC)`` for any ``NC``.

    Every candidate is transparent, so all reproduce the oracle -- the sweep only picks the layout.
    """
    cands = layouts_of("tsg", permutations("tsg", 2))
    cands.update(layouts_of("tsg", [Layout(f"pad{extra}", (Pad("tsg", "i1", str(extra)),)) for extra in (2, 4)]))
    mesh = [
        shuffled("x", "mesh_cyc", "(i0 + 1) % NC", "(i0 + NC - 1) % NC"),
        shuffled("x", "mesh_rev", "NC - 1 - i0", "NC - 1 - i0"),
    ]
    cands.update(layouts_of("x", mesh))
    return cands


def k09_particle_field_affinity_candidates():
    """The global layout candidates for the particle state:

    * ``soa``     -- identity: keep the F fields fully unzipped ([N] each).
    * ``aos``     -- Zip all fields into one ``[N, F]`` array ``P`` (field-minor AoS).
    * ``aosoa_V`` -- the Zip, then a Block of the particle axis by V -> ``P`` is ``[N/V, F, V]``.
    """
    return {
        "soa": lambda sdfg: None,
        "aos": lambda sdfg: zip_fields(sdfg, k09_particle_field_affinity.FIELDS, "P"),
        "aosoa_4": lambda sdfg: zip_fields(prepare(sdfg), k09_particle_field_affinity.FIELDS, "P", 4),
        "aosoa_8": lambda sdfg: zip_fields(prepare(sdfg), k09_particle_field_affinity.FIELDS, "P", 8),
    }


def k10_omen_windowed_contraction_candidates():
    """The global layout candidates: the dimension order of ``G``'s two batch axes -- consumer
    order (NA, NE) identity vs producer order (NE, NA) via a transparent Permute."""

    def producer_order(sdfg):
        apply_layout(sdfg, "G", Layout("ea", (Permute("G", ("i1", "i0", "i2", "i3")),)))

    return {"ae_consumer": (lambda sdfg: None), "ea_producer": producer_order}


def k11_thomas_candidates():
    """interleaved (K,NB) vs system-major (NB,K): permute all four coefficient arrays."""

    def system_major(sdfg):
        for array in "abcd":
            apply_layout(sdfg, array, Layout("system_major", (Permute(array, ("i1", "i0")),)))

    return {"interleaved": (lambda sdfg: None), "system_major": system_major}


def k12_dlt_stencil_candidates():
    """``x`` as it is and dimension-lifted ([V, N/V]) for V in {4, 8}."""
    lifted = [Layout(f"dlt{v}", (Block("x", "i0", str(v), ("o", "f")), Permute("x", ("f", "o")))) for v in (4, 8)]
    return layouts_of("x", [Layout(IDENTITY), *lifted])


K13_CONV1X1_CHANNEL_LAYOUTS_CBLOCK = (
    16  # nChw16c SIMD-lane panel width (C must be divisible by it for the Block candidate)
)


def k13_conv1x1_channel_layouts_candidates():
    """Permute family over X (NCHW and its 23 other dimension orders) plus the nChw16c Block of X's C axis."""
    orders = [order for order in itertools.permutations(range(4)) if order != (0, 1, 2, 3)]
    permuted = [Layout(f"perm{''.join(map(str, o))}", (Permute("X", tuple(f"i{d}" for d in o)),)) for o in orders]
    panels = (
        Block("X", "i1", str(K13_CONV1X1_CHANNEL_LAYOUTS_CBLOCK), ("cb", "cl")),
        Permute("X", ("i0", "cb", "i2", "i3", "cl")),
    )
    return layouts_of("X", [Layout(IDENTITY), *permuted, Layout(f"nChw{K13_CONV1X1_CHANNEL_LAYOUTS_CBLOCK}c", panels)])


#: The Eytzinger stand-in (median-to-front) plus two more closed-form transparent reorderings; each a
#: bijection on ``[0, N)`` for any ``N``.
K14_EYTZINGER_SEARCH_SHUFFLES = (
    shuffled("A", "eytzinger", "(i0 + N // 2) % N", "(i0 + N - N // 2) % N"),
    shuffled("A", "cyc", "(i0 + 1) % N", "(i0 + N - 1) % N"),
    shuffled("A", "rev", "N - 1 - i0", "N - 1 - i0"),
)


def k14_eytzinger_search_candidates():
    """The global layout candidates for this kernel: the unshuffled key array plus
    each registered element reordering of ``A`` (Eytzinger stand-in, cyclic, reversal)."""
    return layouts_of("A", [Layout(IDENTITY), *K14_EYTZINGER_SEARCH_SHUFFLES])


def k15_colstore_candidates():
    return layouts_of("T", permutations("T", 2))


def k16_nbody_force_candidates():
    """AoS (N,3) identity vs SoA (3,N): one transparent Permute of ``pos``'s two axes."""
    return layouts_of("pos", permutations("pos", 2))  # aos: permute_pos_01, soa: permute_pos_10


def k17_knn_bruteforce_candidates():
    """AoS (NR,D) identity vs SoA (D,NR): one transparent Permute of ``ref``'s two axes."""
    return layouts_of("ref", permutations("ref", 2))  # permute_ref_01 (aos), permute_ref_10 (soa)


def k18_scattering_self_energies_candidates():
    """The batch-axis order of ``G``: energy-outer (identity) vs atom-outer (swap NE and NA, so the
    gathered neighbour's energy window is contiguous) -- one transparent Permute, the k10 decision."""

    def atom_outer(sdfg):
        apply_layout(sdfg, "G", Layout("atom_outer", (Permute("G", ("i0", "i2", "i1", "i3", "i4")),)))

    return {"e_outer": (lambda sdfg: None), "atom_outer": atom_outer}


#: corpus module name -> its layout candidates.
CANDIDATES = {
    "atax": atax_candidates,
    "bicg": bicg_candidates,
    "e1_matrix_add": e1_matrix_add_candidates,
    "e6_indirect_stencil": e6_indirect_stencil_candidates,
    "icon_one_loop": icon_one_loop_candidates,
    "icon_zekinh": icon_zekinh_candidates,
    "k01_stream_triad": k01_stream_triad_candidates,
    "k02_transpose_blocked": k02_transpose_blocked_candidates,
    "k03_complex_conjugate_zip": k03_complex_conjugate_zip_candidates,
    "k04_mvt": k04_mvt_candidates,
    "k05_spmv": k05_spmv_candidates,
    "k06_gather_accumulate_scatter": k06_gather_accumulate_scatter_candidates,
    "k07_icon_stencil": k07_icon_stencil_candidates,
    "k08_mesh_scatter": k08_mesh_scatter_candidates,
    "k09_particle_field_affinity": k09_particle_field_affinity_candidates,
    "k10_omen_windowed_contraction": k10_omen_windowed_contraction_candidates,
    "k11_thomas": k11_thomas_candidates,
    "k12_dlt_stencil": k12_dlt_stencil_candidates,
    "k13_conv1x1_channel_layouts": k13_conv1x1_channel_layouts_candidates,
    "k14_eytzinger_search": k14_eytzinger_search_candidates,
    "k15_colstore": k15_colstore_candidates,
    "k16_nbody_force": k16_nbody_force_candidates,
    "k17_knn_bruteforce": k17_knn_bruteforce_candidates,
    "k18_scattering_self_energies": k18_scattering_self_energies_candidates,
}
