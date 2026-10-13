# Copyright 2019-2026 ETH Zurich and the DaCe authors. All rights reserved.
"""Drive the ported SC26 reference kernels through the layout sweep, verifying every candidate
against the numpy oracle.

Each kernel exercises a different primitive family: e1 matrix-add (the Fig.4 running example, Permute
+ Block on a read operand), k01 STREAM triad (Block, the fragmentation instrument), k02 transpose
(Block + Permute), k03 complex conjugate (Zip SoA/AoS/AoSoA), k05 SpMV (ELL sparse gather: Pad +
Block + Permute + indirection), k06 gather-accumulate-scatter (Shuffle, complex128), k07 ICON stencil
(Permute 3D), k08 mesh edge-residual (padded-gather scatter: Pad + Permute + Shuffle through an
indirection), k09 particle fields (Zip AoSoA), k10 OMEN windowed contraction (cross-phase Permute of
a transient, complex128), k13 1x1 conv (Permute + nChw16c Block), k14 Eytzinger search (Shuffle), e6
ICON indirect stencil (connectivity-table gather, transparent Permute). For the kernels with a large
candidate set (k02: 7, k13: 25) a representative SUBSET is swept so the test stays fast -- the
module's ``candidates()`` still enumerates the full family for a real (timed) sweep. Correctness is
the invariant; timing is not asserted."""

from relayout.benchmarks import (
    atax,
    bicg,
    e1_matrix_add,
    e6_indirect_stencil,
    icon_one_loop,
    icon_zekinh,
    k01_stream_triad,
    k02_transpose_blocked,
    k03_complex_conjugate_zip,
    k05_spmv,
    k06_gather_accumulate_scatter,
    k07_icon_stencil,
    k08_mesh_scatter,
    k09_particle_field_affinity,
    k10_omen_windowed_contraction,
    k13_conv1x1_channel_layouts,
    k14_eytzinger_search,
    k16_nbody_force,
    k17_knn_bruteforce,
    k18_scattering_self_energies,
)
from relayout.benchmarks import build as build_candidates
from relayout.search.measure import best, sweep, time_call


def check(program, candidates, run, reference, pick=None):
    """Every candidate (or the ``pick`` subset) reproduces ``reference``; timed once, not asserted."""
    chosen = candidates if pick is None else {name: candidates[name] for name in pick}
    results = sweep(build_candidates(program, chosen), run, reference, timer=time_call, reps=1, warmup=0)
    assert all(r.correct for r in results), [(r.name, r.error) for r in results]
    assert best(results) is not None


def test_atax_permute():
    inp = atax.make_inputs(32, 32)
    check(atax.atax, atax.candidates(), atax.run_closure(inp, 32, 32), atax.oracle(inp["A"], inp["x"]))


def test_bicg_permute():
    inp = bicg.make_inputs(32, 30)
    check(bicg.bicg, bicg.candidates(), bicg.run_closure(inp, 32, 30), bicg.oracle(inp["A"], inp["p"], inp["r"]))


def test_icon_zekinh_double_indirect_permute():
    inp = icon_zekinh.make_inputs(2, 8, 6)
    check(
        icon_zekinh.zekinh,
        icon_zekinh.candidates(),
        icon_zekinh.run_closure(inp, 2, 8, 6),
        icon_zekinh.oracle(inp["e_bln"], inp["edge_idx"], inp["edge_blk"], inp["z_kin_hor_e"]),
    )


def test_icon_one_loop_vertical_permute():
    inp = icon_one_loop.make_inputs(2, 16, 8)
    check(
        icon_one_loop.one_loop,
        icon_one_loop.candidates(),
        icon_one_loop.run_closure(inp, 2, 16, 8),
        icon_one_loop.oracle(inp["vn"], inp["wgtfac_e"], inp["vt"]),
    )


def test_e1_matrix_add_permute_and_block():
    inp = e1_matrix_add.make_inputs(64)
    check(
        e1_matrix_add.matrix_add,
        e1_matrix_add.candidates(),
        e1_matrix_add.run_closure(inp, 64),
        e1_matrix_add.oracle(inp["A"], inp["B"]),
    )


def test_k05_spmv_ell_pad_block_permute():
    inp = k05_spmv.make_inputs(8)
    check(
        k05_spmv.spmv_ell,
        k05_spmv.candidates(),
        k05_spmv.run_closure(inp, 8),
        k05_spmv.oracle(inp["ecol"], inp["eval_"], inp["x"]),
    )


def test_k08_mesh_scatter_pad_permute_shuffle():
    inp = k08_mesh_scatter.make_inputs(8)
    check(
        k08_mesh_scatter.mesh_edge_residual,
        k08_mesh_scatter.candidates(),
        k08_mesh_scatter.run_closure(inp, 8),
        k08_mesh_scatter.oracle(inp["x"], inp["e0"], inp["e1"], inp["tab"], inp["tsg"]),
    )


def test_e6_indirect_stencil_permute():
    inp = e6_indirect_stencil.make_inputs(16, 16, 24, 8)
    fields = ["w", "z_w_v", "cell_idx", "vert_idx", "vn_ie", "z_vt_ie", "inv_dual", "inv_primal", "tangent"]
    check(
        e6_indirect_stencil.indirect_stencil,
        e6_indirect_stencil.candidates(),
        e6_indirect_stencil.run_closure(inp, 16, 16, 24, 8),
        e6_indirect_stencil.oracle(*[inp[f] for f in fields]),
    )


def test_k01_stream_triad_block():
    inp = k01_stream_triad.make_inputs(64)
    check(
        k01_stream_triad.triad,
        k01_stream_triad.candidates(),
        k01_stream_triad.run_closure(inp, 64),
        k01_stream_triad.oracle(inp["b"], inp["c"]),
    )


def test_k02_transpose_block_and_permute():
    inp = k02_transpose_blocked.make_inputs(32)
    check(
        k02_transpose_blocked.transpose,
        k02_transpose_blocked.candidates(),
        k02_transpose_blocked.run_closure(inp, 32),
        k02_transpose_blocked.oracle(inp["A"]),
        pick=["identity", "A:block0x8", "A:perm10"],
    )


def test_k03_complex_conjugate_zip():
    inp = k03_complex_conjugate_zip.make_inputs(32)
    check(
        k03_complex_conjugate_zip.conjugate,
        k03_complex_conjugate_zip.candidates(),
        k03_complex_conjugate_zip.run_closure(inp, 32),
        k03_complex_conjugate_zip.oracle(inp["re"], inp["im"]),
    )


def test_k06_gather_accumulate_scatter_shuffle():
    inp = k06_gather_accumulate_scatter.make_inputs(16)
    check(
        k06_gather_accumulate_scatter.gather_accumulate_scatter,
        k06_gather_accumulate_scatter.candidates(),
        k06_gather_accumulate_scatter.run_closure(inp, 16),
        k06_gather_accumulate_scatter.oracle(inp["x"], inp["y"]),
    )


def test_k07_icon_stencil_permute():
    inp = k07_icon_stencil.make_inputs(6, 6, 6)
    check(
        k07_icon_stencil.stencil,
        k07_icon_stencil.candidates(),
        k07_icon_stencil.run_closure(inp, 6, 6, 6),
        k07_icon_stencil.oracle(inp["A"]),
        pick=["identity", "A:perm210", "A:perm120"],
    )


def test_k09_particle_field_affinity_aosoa():
    inp = k09_particle_field_affinity.make_inputs(32)
    fields = ["x", "y", "z", "vx", "vy", "vz", "q", "m"]
    check(
        k09_particle_field_affinity.particle_step,
        k09_particle_field_affinity.candidates(),
        k09_particle_field_affinity.run_closure(inp, 32),
        k09_particle_field_affinity.oracle(*[inp[f] for f in fields]),
    )


def test_k10_omen_windowed_contraction_permute():
    inp = k10_omen_windowed_contraction.make_inputs(3, 8)
    check(
        k10_omen_windowed_contraction.omen,
        k10_omen_windowed_contraction.candidates(),
        k10_omen_windowed_contraction.run_closure(inp, 3, 8),
        k10_omen_windowed_contraction.oracle(inp["H"], inp["X"], inp["D"]),
    )


def test_k13_conv1x1_channel_layouts():
    inp = k13_conv1x1_channel_layouts.make_inputs(2, 16, 4, 4, 8)
    check(
        k13_conv1x1_channel_layouts.conv1x1,
        k13_conv1x1_channel_layouts.candidates(),
        k13_conv1x1_channel_layouts.run_closure(inp, 2, 16, 4, 4, 8),
        k13_conv1x1_channel_layouts.oracle(inp["X"], inp["Wt"]),
        pick=["identity", "X:perm0231", "X:nChw16c"],
    )


def test_k14_eytzinger_search_shuffle():
    inp = k14_eytzinger_search.make_inputs(15)
    check(
        k14_eytzinger_search.eytzinger_search,
        k14_eytzinger_search.candidates(),
        k14_eytzinger_search.run_closure(inp, 15),
        k14_eytzinger_search.oracle(inp["A"]),
    )


def test_k16_nbody_force_permute():
    inp = k16_nbody_force.make_inputs(32)
    check(
        k16_nbody_force.nbody_force,
        k16_nbody_force.candidates(),
        k16_nbody_force.run_closure(inp, 32),
        k16_nbody_force.oracle(inp["pos"], inp["vel"], inp["mass"]),
    )


def test_k17_knn_bruteforce_permute():
    inp = k17_knn_bruteforce.make_inputs(32, 12)
    check(
        k17_knn_bruteforce.knn,
        k17_knn_bruteforce.candidates(),
        k17_knn_bruteforce.run_closure(inp, 32, 12),
        k17_knn_bruteforce.oracle(inp["ref"], inp["query"]),
    )


def test_k18_scattering_self_energies_permute():
    inp = k18_scattering_self_energies.make_inputs(6, 4)
    check(
        k18_scattering_self_energies.sselfeng,
        k18_scattering_self_energies.candidates(),
        k18_scattering_self_energies.run_closure(inp, 6, 4),
        k18_scattering_self_energies.oracle(inp["neigh_idx"], inp["dH"], inp["G"], inp["D"]),
    )


if __name__ == "__main__":
    test_atax_permute()
    test_bicg_permute()
    test_icon_zekinh_double_indirect_permute()
    test_icon_one_loop_vertical_permute()
    test_e1_matrix_add_permute_and_block()
    test_k05_spmv_ell_pad_block_permute()
    test_k08_mesh_scatter_pad_permute_shuffle()
    test_e6_indirect_stencil_permute()
    test_k01_stream_triad_block()
    test_k02_transpose_block_and_permute()
    test_k03_complex_conjugate_zip()
    test_k06_gather_accumulate_scatter_shuffle()
    test_k07_icon_stencil_permute()
    test_k09_particle_field_affinity_aosoa()
    test_k10_omen_windowed_contraction_permute()
    test_k13_conv1x1_channel_layouts()
    test_k14_eytzinger_search_shuffle()
    test_k16_nbody_force_permute()
    test_k17_knn_bruteforce_permute()
    test_k18_scattering_self_energies_permute()
    print("kernel ports tests PASS")
