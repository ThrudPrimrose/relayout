# relayout

Layout transformation sequences on DaCe SDFGs. Targets: GPU linear layouts, and distributed runs where
layout and data distribution change together (MPI, NCCL/RCCL).

## Install

```
uv sync --extra mpi --group dev        # + --extra bench for hpcagent-bench kernels
```

One dace for everything: `spcl/dace@extended` (overrides the pins in dace-fortran and hpcagent-bench).
Fortran tests need LLVM `flang` on PATH.

## Layout

- `src/relayout/kernels/omen_sse/` OMEN scattering self-energy (SC19 Gordon Bell), Python and Fortran,
  single node and MPI (G energy-distributed in, Sigma momentum-distributed out, one Alltoall).
  `reference()` is the closed form the SC19 transformation sequence ends at.
- `src/relayout/benchmarks/` SC26 layout witness kernels from dace@extended, `ORDER` simple to complicated.
- `tests/` numpy oracles; MPI tests run as `mpirun -n 2 uv run pytest tests/test_omen_sse_mpi.py`.

## Plan

1. OMEN SSE: reproduce the SC19 sequence step by step, one oracle test per step.
2. Same sequence on GPU, with layouts expressed as linear layouts.
3. NCCL/RCCL library nodes beside the MPI ones; Alltoall on device buffers.
4. Benchmarks into hpcagent-bench as layout variants.
