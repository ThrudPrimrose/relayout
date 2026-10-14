# relayout

Data layout transformations for numpy and DaCe programs, kept outside DaCe's core (as
[FP-Arena](https://github.com/spcl/FP-Arena) keeps its floating-point types). One request says how a
kernel's arrays should be laid out, cast and distributed; relayout checks it, rewrites the program
for it (a numpy / mpi4py program or an SDFG) and generates the kernels that convert the data.

## Install

```
uv sync --extra mpi --group dev        # add --extra fortran for the Fortran kernels (needs LLVM flang)
```

relayout follows dace's `extended` branch; dace, islpy, z3 and numpy are dependencies.

## Quickstart: OMEN's self-energy (SC19)

[`examples/omen_quickstart.py`](examples/omen_quickstart.py) lays out the scattering self-energy
kernel of OMEN (Ziogas et al., SC'19) on one node and distributes it as SC19 does:

```python
from relayout.passes.sdfg.apply_requests import apply_requests
from relayout.corpus.omen_sse.sse import sse
from relayout.ops import ArraySpec, lower
from relayout.requests import parse_requests

local = parse_requests({"operations": [
    {"action": "permute", "array": "G", "order": ["i2", "i0", "i1", "i3", "i4"]},   # atom-major G
    {"action": "permute", "array": "dH", "order": ["i0", "i1", "i3", "i4", "i2"]},  # direction innermost
]})
sdfg = sse.to_sdfg(simplify=True)
apply_requests(sdfg, local)       # the SDFG now runs on the laid-out G and dH; arguments keep their layout
sdfg(**arrays, **sizes)           # Sigma equals omen_sse.reference(...)

distributed = parse_requests({"startup": {"grid": ["P"]}, "operations": [
    {"action": "distribute", "array": "G", "grid_dim": 0, "dim": 1, "scheme": "block"},      # energies in
    {"action": "distribute", "array": "Sigma", "grid_dim": 0, "dim": 0, "scheme": "block"},  # momenta out
    *({"action": "replicate", "array": a} for a in ("neigh_idx", "dH", "D")),
]})
lower(distributed.run([ArraySpec(name, 5) for name in ...], ranks=4))   # each array's split per dimension
```

## What is in it

| Module | What it does |
| --- | --- |
| `relayout.ops` | the request: `startup` (libraries, a processor grid over the rank count `P`) and `operations` run top to bottom (`cast`, `format`, `zip` / `unzip`, the digit cuts `block` / `split` / `permute` / `pad` / `shuffle` / `linear_shuffle` and their inverses, `distribute`, `replicate`, `place`); the lowering of a distribution to one split per dimension |
| `relayout.requests` | the request's JSON, one to one with `relayout.ops`, and its JSON Schema (both from pydantic) |
| `relayout.layout` | `IndexMap`: one array's layout as index arithmetic, logical index to physical and back; the model everything below reads |
| `relayout.convert` | conversion kernels: one gather that casts, lays out and changes sparse formats in a single pass, as C with OpenMP and as a CUDA / HIP kernel; numpy forms for tests |
| `relayout.sparse` | sparse formats (csr, csc, coo, bsr, dia, ell) planned once from a canonical CSR, every format decoded back to it and converted to every other; values move through the gather; `boundary` calls a program (Python or SDFG) written for one format with another, converting at the call |
| `relayout.rules` | how a library call absorbs a permuted operand: transpose, einsum, contraction / reduction axes, BLAS transpose flags (gemm, matmul, syrk) and triangles (syrk, symm) |
| `relayout.passes.py` | Python passes, one module per SDFG pass of the same name and contract (an argument keeps its array, converted in and out; a local is rewritten in place): `prepare`, `relayout_array`, `zip_arrays` / `unzip_array` (a struct zip is a numpy structured array), `cast_array`, `pack_mpi_buffers` (mpi4py buffers, `Irecv` unpacked after its `Wait`), `unblock_dimensions` / `untile_blocked_loops`, `canonicalize_schedule` (loops tiled to the blocks, independent nests ordered by stride), `apply_requests`; shared by them: `common`, `placement`, `rewrite_accesses` |
| `relayout.passes.sdfg` | SDFG passes, one module each: `prepare` (untile, canonicalize, packed arrays only), `relayout_array` (one rewrite for every layout: permute, block, split, pad, shuffle; library nodes by their rules), `zip_arrays` / `unzip_array` (along a dimension, data-dependent indices select their part; different element types zip into an array of structs), `cast_array` (FP-Arena, extra `fparena`), `pack_mpi_buffers` (MPI buffers through packed transients), `unblock_dimensions` / `untile_blocked_loops` (hand-blocked arrays and tiled loops flattened), `canonicalize_schedule` (maps tiled to the blocks their arrays are cut into, tiles over flat arrays merged back, map nests ordered by stride), `apply_requests` (a request's operations, in order); shared by them: `common`, `placement`, `rewrite_accesses`, `library_rules` |
| `relayout.search` | which layout each array takes: a program as a line of kernels, each kernel's layouts costed by the LogGP model or measured, one trajectory per array chosen globally (Viterbi DP, with greedy and enumeration as checks) and applied with conversions between segments |
| `relayout.corpus` (`corpus/`) | the input programs: the OMEN SSE kernel (Python, Fortran, MPI), the layout witness kernels and Fortran inputs, each with a numpy oracle; the layouts tried on them live in `tests/layout_candidates.py` |

The request format and the lowering are described in [docs/layout.md](docs/layout.md).

## Tests

`pytest` runs everything on one process; the MPI tests also run under
`mpirun -n 2 pytest -m mpi tests`, and the device kernels
where cupy finds a GPU.
