#!/bin/bash
# The zekin staging experiment (zekin_staging.py) on one MI300A of Beverin (CSCS): klon- vs klev-first layout,
# fp64 / fp32 / fp16, unstaged / staged loads / staged loads and stores. Extra arguments go to the script, e.g.
#   sbatch zekin_staging.sbatch --nproma 270764 --reps 100
# ENV_TAR packs the dace (extended + StageThreadBlockData), FP-Arena and hpcagent-bench checkouts and the venv the
# script runs in, built under /dev/shm on the login node; the job unpacks it at the same paths in the node's /dev/shm
# (one file on capstor instead of ~25k inodes). RELAYOUT is this checkout (installed editable in the venv). Results
# land in OUT: zekin_staging.json and the zekin_staging.md table.
#SBATCH -J zekin-staging
#SBATCH -p mi300
#SBATCH -A a-g200
#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --exclusive
#SBATCH -t 03:00:00
#SBATCH -o /capstor/scratch/cscs/ybudanaz/x86_64/zekin-staging-runs/%x-%j.out
set -euo pipefail
ulimit -c 0

SCRATCH_ROOT=/capstor/scratch/cscs/ybudanaz/x86_64
ENV_TAR=${ENV_TAR:-$SCRATCH_ROOT/zekin-staging-runs/zekin-staging-env.tar.zst}
RELAYOUT=${RELAYOUT:-$SCRATCH_ROOT/relayout}
OUT=${OUT:-$SCRATCH_ROOT/zekin-staging-runs/$SLURM_JOB_ID}
DACE_TREE=/dev/shm/ybud-stage FP_ARENA_TREE=/dev/shm/ybud-fparena HB_TREE=/dev/shm/ybud-hb-conn
VENV=/dev/shm/ybud-stage-venv
PY=$VENV/bin/python

tar --zstd -xf "$ENV_TAR" -C /dev/shm
trap 'rm -rf "$DACE_TREE" "$FP_ARENA_TREE" "$HB_TREE" "$VENV" "$DACE_default_build_folder"' EXIT

# gcc 14 for the host side (the system gcc is 7.5, no C++20), hipcc for the kernels.
export PATH=$SCRATCH_ROOT/ybud-gcc14:/opt/rocm/bin:$PATH CC=gcc-14 CXX=g++-14
export DACE_compiler_cuda_backend=hip DACE_compiler_cuda_hip_arch=gfx942
export DACE_default_build_folder=/tmp/zekin-staging-dacecache-$SLURM_JOB_ID
export OMP_NUM_THREADS=8

mkdir -p "$OUT"
cd "$OUT"  # outside every checkout: a checkout's top-level package directory would shadow the installed one
echo "dace $(git -C "$DACE_TREE" rev-parse --short HEAD), fp-arena $(git -C "$FP_ARENA_TREE" rev-parse --short HEAD)," \
     "hpcagent-bench $(git -C "$HB_TREE" rev-parse --short HEAD), relayout $(git -C "$RELAYOUT" rev-parse --short HEAD)"
rocm-smi --showproductname 2>/dev/null | grep -m1 "Card series" || true
# Run a snapshot: dace re-reads the kernel's source while the job runs, so an edit of the checkout must not reach it.
cp "$RELAYOUT/examples/zekin_staging/zekin_staging.py" "$OUT/zekin_staging.py"
srun -n 1 "$PY" "$OUT/zekin_staging.py" --out "$OUT/zekin_staging.json" "$@"
