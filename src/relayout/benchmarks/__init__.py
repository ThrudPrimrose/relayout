"""Layout benchmarks: the SC26 layout witness kernels, copied from spcl/dace@extended tests/transformations/layout/kernels.

Each module holds a DaCe program, a numpy oracle, make_inputs, and the layout candidates it exercises
(``candidates()`` / ``run_closure``), so ``dace.transformation.layout.brute_force.sweep`` can enumerate,
verify and rank layouts. Later these move into hpcagent-bench as layout variants of its kernels.

ORDER runs from simple to complicated: dense elementwise, then transposes and AoS/SoA, then reductions
where the layout is global across nests, then indirection, then the OMEN cross-phase kernels.
"""

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
