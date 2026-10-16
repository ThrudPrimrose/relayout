# Copyright 2026 ETH Zurich and the relayout authors.
"""Layout, precision and shared-memory staging on ICON's ``z_ekinh`` gather (velocity tendencies).

The kernel is hpcagent-bench's ``zekin_gather`` -- the kinetic energy of velocity_tendencies interpolated from edges
to cells through the mesh-local connectivity tables -- with its inputs (``initialize``) and its numpy reference
from hpcagent-bench::

    z_ekinh[jb, jk, jc] = sum_m e_bln[jb, m, jc] * z_kin_hor_e[edge_blk[m, jb, jc], jk, edge_idx[m, jb, jc]]

written per element, so canonicalization fuses it into one kernel. One block (``nblks = 1``, declared SMALL and
specialized), a huge ``nproma`` and ``nlev = 144`` levels (nine 16-level thread-block rows, no partial tile). Every configuration is one pipeline:

1. relayout lays ``z_kin_hor_e`` and ``z_ekinh`` out ``klon``-first (``[nblks, nlev, nproma]``, ICON's own order:
   nproma contiguous) or ``klev``-first (``[nblks, nproma, nlev]``: the structured vertical axis contiguous); the
   arguments are laid out in place (``convert=False``), so the caller passes laid-out data and the kernel converts
   nothing;
2. CPF for the GPU (canonicalize, offload, finalize), then ``CoarsenKernelThreads``: each thread computes a
   ``--coarsen`` tile of consecutive levels (and cells, ``6x2``), so a thread block covers that many times more levels
   and reuses the connectivity tables and weights (which do not depend on the level) that many times more
   (1 = one element per thread; 9 = a 16-row block covers one whole 144-level column);
3. FP-Arena lowers every floating-point argument to fp32 or fp16 (the interface itself, the caller passes
   low-precision data);
4. ``StageThreadBlockData`` stages the tile each thread block reads (``loads``), and also the tile it writes
   (``loads+stores``), through shared memory with block-collective, 16-byte-word copies and one barrier each way.

After the layout, relayout's ``canonicalize_schedule`` makes the parameter that walks the laid-out fields' unit-stride
axis the innermost one, so the kernel's ``threadIdx.x`` walks it in both layouts.

Each configuration runs inside a host harness SDFG: inputs copied to the device once, then ``--reps`` timed samples
(after one warm-up sample), each launching the kernel back to back enough times that its kernels take at least
``--sample-ms`` of GPU time (calibrated from single launches; every launch is timed by GPU events), then the output
copied back and checked against the reference. The default size (16M cells x 144 levels, 18.4 GB per fp64 level
field) is about the most the 128 GB of an MI300A holds: the reference, the input and output fields of the
configuration's precision and layout and their device copies; every other copy is made in chunks or regenerated.

Run: ``python examples/zekin_staging/zekin_staging.py --out results.json`` on a GPU node (needs hpcagent-bench
importable and the ``fparena`` extra); ``zekin_staging.sbatch`` is the job for the MI300A partition of Beverin.
"""

import argparse
import copy
import itertools
import json
import math
import pathlib
import statistics
import time
import traceback
from collections.abc import Sequence

import dace
import numpy as np
import sympy
from dace import dtypes, symbolic
from dace.sdfg.state import LoopRegion
from dace.transformation.passes.canonicalize.finalize import finalize_for_target, offload_to_gpu
from dace.transformation.passes.canonicalize.pipeline import canonicalize
from dace.transformation.passes.gpu_specialization.coarsen_kernel_threads import CoarsenKernelThreads
from dace.transformation.passes.gpu_specialization.stage_thread_block_data import StageThreadBlockData
from fp_arena.transformations import change_and_propagate_fp_types
from hpcagent_bench.benchmarks.scientific_computing.unstructured_grids.zekin_gather import zekin_gather as hb_inputs
from hpcagent_bench.benchmarks.scientific_computing.unstructured_grids.zekin_gather import (
    zekin_gather_numpy as hb_reference,
)
from hpcagent_bench.support.counter_generator import CounterGenerator

from relayout.ops import Permute, Requests
from relayout.passes.sdfg.apply_requests import apply_requests
from relayout.passes.sdfg.canonicalize_schedule import canonicalize_schedule

#: 64-bit sizes: a level field has more than 2**31 elements at the default size.
NB, NLEV, NPROMA = (dace.symbol(name, dace.int64) for name in ("NB", "NLEV", "NPROMA"))

#: The fields with a level axis, which the layout moves.
LEVEL_FIELDS = ("z_kin_hor_e", "z_ekinh")
#: Storage order of the level fields, as relayout digits of ``[nblks, nlev, nproma]``.
LAYOUTS = {"klon": ("i0", "i1", "i2"), "klev": ("i0", "i2", "i1")}
PRECISIONS = {"fp64": dace.float64, "fp32": dace.float32, "fp16": dace.float16}
#: Staging variants: (stage loads, stage stores).
VARIANTS = {"base": None, "loads": (True, False), "loads+stores": (True, True)}
#: The symbol of the launches per timed sample.
BATCH = "BATCH"
#: Normalized max error each precision is checked to (inputs and arithmetic rounded to it).
TOLERANCE = {"fp64": 1e-12, "fp32": 1e-5, "fp16": 1e-2}


@dace.program
def zekin(
    e_bln: dace.float64[NB, 3, NPROMA],
    edge_idx: dace.int32[3, NB, NPROMA],
    edge_blk: dace.int32[3, NB, NPROMA],
    z_kin_hor_e: dace.float64[NB, NLEV, NPROMA],
    z_ekinh: dace.float64[NB, NLEV, NPROMA],
):
    for jb, jk, jc in dace.map[0:NB, 0:NLEV, 0:NPROMA]:
        z_ekinh[jb, jk, jc] = (
            e_bln[jb, 0, jc] * z_kin_hor_e[edge_blk[0, jb, jc], jk, edge_idx[0, jb, jc]]
            + e_bln[jb, 1, jc] * z_kin_hor_e[edge_blk[1, jb, jc], jk, edge_idx[1, jb, jc]]
            + e_bln[jb, 2, jc] * z_kin_hor_e[edge_blk[2, jb, jc], jk, edge_idx[2, jb, jc]]
        )


def oracle(e_bln: np.ndarray, edge_idx: np.ndarray, edge_blk: np.ndarray, z_kin_hor_e: np.ndarray) -> np.ndarray:
    """The reference, vectorized over the levels (hpcagent-bench's transposes element by element, which only
    a small size affords; :func:`check_against_hpcagent_bench` ties the two together)."""
    out = np.zeros_like(z_kin_hor_e)
    for m in range(3):
        gathered = z_kin_hor_e[edge_blk[m], :, edge_idx[m]]  # (nblks, nproma, nlev)
        out += e_bln[:, m, :][:, None, :] * gathered.transpose(0, 2, 1)
    return out


def level_field(seed: int, nblks: int, nlev: int, nproma: int) -> np.ndarray:
    """hpcagent-bench's ``z_kin_hor_e`` of ``initialize`` (its generator's first draw), regenerated on its own so only
    one copy of it at a time is held."""
    return CounterGenerator(seed).random((nblks, nlev, nproma))


def laid_out(field: np.ndarray, dtype: np.dtype, order: Sequence[int]) -> np.ndarray:
    """``field`` in ``dtype`` and the storage order ``order``: an array of its own (dace takes no views), ``field``
    itself if it already is one."""
    field = field.astype(dtype, copy=False)
    return field if list(order) == sorted(order) else field.transpose(order).copy()


def max_error(out: np.ndarray, order: Sequence[int], expected: np.ndarray, chunk: int = 1 << 20) -> float:
    """Max absolute difference of the laid-out ``out`` from ``expected`` (``[nblks, nlev, nproma]``), compared in
    fp64 a chunk of cells at a time (a whole fp64 copy of ``out`` would not fit)."""
    got = out.transpose(np.argsort(order))
    nproma = expected.shape[-1]
    return max(
        float(np.max(np.abs(got[..., c : c + chunk].astype(np.float64) - expected[..., c : c + chunk])))
        for c in range(0, nproma, chunk)
    )


def check_against_hpcagent_bench(nlev: int = 6, nproma: int = 97) -> None:
    """The program and the vectorized oracle compute hpcagent-bench's reference; :func:`level_field` is its input."""
    arrays = hb_inputs.initialize(1, nlev, nproma, rng=CounterGenerator(7))
    e_bln, edge_idx, edge_blk, z_kin_hor_e, z_ekinh = (np.ascontiguousarray(a) for a in arrays)
    np.testing.assert_array_equal(level_field(7, 1, nlev, nproma), z_kin_hor_e)
    hb_reference.zekin_gather(e_bln, edge_idx, edge_blk, z_kin_hor_e, z_ekinh, 1, nlev, nproma)
    np.testing.assert_allclose(oracle(e_bln, edge_idx, edge_blk, z_kin_hor_e), z_ekinh, rtol=1e-13)
    out = np.zeros_like(z_ekinh)
    zekin.to_sdfg(simplify=True)(
        e_bln=e_bln,
        edge_idx=edge_idx,
        edge_blk=edge_blk,
        z_kin_hor_e=z_kin_hor_e,
        z_ekinh=out,
        NB=1,
        NLEV=nlev,
        NPROMA=nproma,
    )
    np.testing.assert_allclose(out, z_ekinh, rtol=1e-13)


def kernel_maps(sdfg: dace.SDFG) -> list[dace.nodes.MapEntry]:
    return [
        node
        for node, _ in sdfg.all_nodes_recursive()
        if isinstance(node, dace.nodes.MapEntry) and node.map.schedule == dtypes.ScheduleType.GPU_Device
    ]


def coarsening(spec: str) -> dict[str, int]:
    """Elements per thread of ``spec``: ``LEVELS`` or ``LEVELSxCELLS`` (``6x2``: 6 levels by 2 cells)."""
    levels, _, cells = spec.partition("x")
    return {"jk": int(levels), "jc": int(cells or 1)}


def build(layout: str, precision: str, variant: str, coarsen: str, nblks: int) -> tuple[dace.SDFG, list[str]]:
    """The kernel SDFG of one configuration and the arrays staging moved through shared memory."""
    sdfg = zekin.to_sdfg(simplify=True)
    sdfg.name = f"zekin_{layout}_{precision}_{variant.replace('+', '_')}_c{coarsen}"
    if layout != "klon":
        requests = Requests(operations=tuple(Permute(name, LAYOUTS[layout]) for name in LEVEL_FIELDS))
        apply_requests(sdfg, requests, convert=False)
        # The laid-out fields' unit-stride axis becomes the innermost map parameter (threadIdx.x after offload).
        canonicalize_schedule(sdfg, set(LEVEL_FIELDS))
    sdfg.set_symbol_scale("NB", symbolic.Scale.SMALL)
    sdfg.specialize({"NB": nblks})
    canonicalize(sdfg, validate=True, validate_all=False, target="gpu")
    offload_to_gpu(sdfg)
    finalize_for_target(sdfg, "gpu", validate=True)
    # The levels first: the tables a block stages do not depend on them, so a block covering more levels reuses them
    # more; cells too for a 2-D tile per thread.
    CoarsenKernelThreads(factors=coarsening(coarsen)).apply_pass(sdfg, {})
    if precision != "fp64":
        target = PRECISIONS[precision]
        pins = {name: target for name, desc in sdfg.arrays.items() if not desc.transient and desc.dtype == dace.float64}
        change_and_propagate_fp_types(sdfg, pins, preserve_interface=False)
    staged: list[str] = []
    if VARIANTS[variant] is not None:
        loads, stores = VARIANTS[variant]
        result = StageThreadBlockData(stage_loads=loads, stage_stores=stores).apply_pass(sdfg, {})
        staged = sorted(itertools.chain.from_iterable((result or {}).values()))
    for entry in kernel_maps(sdfg):
        entry.map.instrument = dtypes.InstrumentationType.GPU_Events
    sdfg.validate()
    return sdfg, staged


def batched(kernel: dace.SDFG, inputs: list[str], outputs: list[str], mapping: dict) -> dace.SDFG:
    """``kernel`` launched ``BATCH`` times back to back (``BATCH`` a symbol: set per call, no recompile)."""
    batch = dace.SDFG(f"{kernel.name}_batch")
    batch.add_symbol(BATCH, dace.int64)
    for name in sorted(set(mapping) - set(map(str, kernel.constants))):
        batch.add_symbol(name, kernel.symbols[name])
    for name in inputs + outputs:
        desc = copy.deepcopy(kernel.arrays[name])
        desc.set_shape([sympy.sympify(e).subs(mapping_constants(mapping)) for e in desc.shape])
        batch.add_datadesc(name, desc)
    loop = LoopRegion("launches", f"__b < {BATCH}", "__b", "__b = 0", "__b = __b + 1")
    batch.add_node(loop, is_start_block=True)
    state = loop.add_state("launch", is_start_block=True)
    call = state.add_nested_sdfg(kernel, set(inputs), set(outputs), mapping)
    for name in inputs:
        state.add_edge(state.add_read(name), None, call, name, dace.Memlet.from_array(name, batch.arrays[name]))
    for name in outputs:
        state.add_edge(call, name, state.add_write(name), None, dace.Memlet.from_array(name, batch.arrays[name]))
    return batch


def mapping_constants(mapping: dict) -> dict:
    return {symbolic.symbol(k): v for k, v in mapping.items() if isinstance(v, int)}


def host_harness(kernel: dace.SDFG, reps: int) -> dace.SDFG:
    """Device copies of host arrays (one copy in, one out) and ``reps`` samples between them, each launching
    ``kernel`` (its kernels timed by GPU events) ``BATCH`` times back to back."""
    constants = {symbolic.symbol(k): int(v) for k, v in kernel.constants.items()}
    outer = dace.SDFG(f"{kernel.name}_harness")
    outer.add_symbol(BATCH, dace.int64)
    for name in sorted(map(str, kernel.free_symbols)):
        outer.add_symbol(name, kernel.symbols[name])
    arguments = [name for name, desc in kernel.arrays.items() if not desc.transient]
    for name in arguments:
        desc = kernel.arrays[name]
        shape = [sympy.sympify(extent).subs(constants) for extent in desc.shape]
        outer.add_array(name, shape, desc.dtype)
        outer.add_array(f"dev_{name}", shape, desc.dtype, storage=dtypes.StorageType.GPU_Global, transient=True)
    copy_in = outer.add_state("copy_in", is_start_block=True)
    for name in arguments:
        copy_in.add_nedge(
            copy_in.add_read(name), copy_in.add_write(f"dev_{name}"), dace.Memlet.from_array(name, outer.arrays[name])
        )
    written = {node.data for state in kernel.states() for node in state.data_nodes() if state.in_degree(node) > 0}
    outputs = [name for name in arguments if name in written]
    inputs = [name for name in arguments if name not in outputs]
    mapping = {
        **{name: name for name in map(str, kernel.free_symbols)},
        **{str(k): int(v) for k, v in constants.items()},
    }
    batch = batched(kernel, inputs, outputs, mapping)
    loop = LoopRegion("samples", f"__rep < {reps}", "__rep", "__rep = 0", "__rep = __rep + 1")
    outer.add_node(loop)
    sample = loop.add_state("sample", is_start_block=True)
    outer.add_edge(copy_in, loop, dace.InterstateEdge())
    symbols = {name: name for name in [BATCH, *map(str, kernel.free_symbols)]}
    call = sample.add_nested_sdfg(batch, set(inputs), set(outputs), symbols)
    for name in inputs:
        memlet = dace.Memlet.from_array(f"dev_{name}", outer.arrays[f"dev_{name}"])
        sample.add_edge(sample.add_read(f"dev_{name}"), None, call, name, memlet)
    for name in outputs:
        memlet = dace.Memlet.from_array(f"dev_{name}", outer.arrays[f"dev_{name}"])
        sample.add_edge(call, name, sample.add_write(f"dev_{name}"), None, memlet)
    copy_out = outer.add_state("copy_out")
    outer.add_edge(loop, copy_out, dace.InterstateEdge())
    for name in outputs:
        copy_out.add_nedge(
            copy_out.add_read(f"dev_{name}"), copy_out.add_write(name), dace.Memlet.from_array(name, outer.arrays[name])
        )
    return outer


def launch_times_ms(harness: dace.SDFG) -> list[float]:
    """The GPU-event time of every kernel launch of the latest run, in milliseconds, in launch order."""
    report = harness.get_latest_report()
    times: list[float] = []
    for events in report.durations.values():  # only the kernel maps are instrumented
        for threads in events.values():
            for values in threads.values():
                times.extend(values)
    return times


def run(args: argparse.Namespace) -> list[dict]:
    check_against_hpcagent_bench()
    arrays = hb_inputs.initialize(args.nblks, args.nlev, args.nproma, rng=CounterGenerator(args.seed))
    e_bln, edge_idx, edge_blk, z_kin_hor_e = (np.ascontiguousarray(a) for a in arrays[:4])
    del arrays
    expected = oracle(e_bln, edge_idx, edge_blk, z_kin_hor_e)
    del z_kin_hor_e  # regenerated per layout and precision: one copy of a level field at a time
    scale = float(np.max(np.abs(expected)))
    sizes = {"NLEV": args.nlev, "NPROMA": args.nproma}
    rows = []
    for layout, precision in itertools.product(args.layouts, args.precisions):
        npt = PRECISIONS[precision].as_numpy_dtype()
        order = [int(d[1:]) for d in LAYOUTS[layout]]
        inputs = {
            "e_bln": e_bln.astype(npt),
            "edge_idx": edge_idx,
            "edge_blk": edge_blk,
            "z_kin_hor_e": laid_out(level_field(args.seed, args.nblks, args.nlev, args.nproma), npt, order),
        }
        out = np.empty(inputs["z_kin_hor_e"].shape, dtype=npt)
        for coarsen, variant, chiplets in itertools.product(args.coarsen, args.variants, args.chiplets):
            config = {
                "layout": layout,
                "precision": precision,
                "variant": variant,
                "coarsen": coarsen,
                "chiplets": chiplets,
            }
            try:
                out.fill(0)
                started = time.perf_counter()
                kernel, staged = build(layout, precision, variant, coarsen, args.nblks)
                harness = host_harness(kernel, args.reps + 1)
                # The thread blocks of a kernel spread over that many chiplets (XCDs) in contiguous chunks; 1 = off.
                with dace.config.set_temporary("compiler", "cuda", "chiplet_number", value=chiplets):
                    compiled = harness.compile()
                build_s = time.perf_counter() - started
                # Calibrate: enough back-to-back launches per sample that a sample takes at least --sample-ms.
                compiled(**inputs, z_ekinh=out, **sizes, **{BATCH: 1})
                one = statistics.median(launch_times_ms(harness)[args.warmup :])
                launches = max(1, math.ceil(args.sample_ms / one))
                out.fill(0)
                compiled(**inputs, z_ekinh=out, **sizes, **{BATCH: launches})
                times = launch_times_ms(harness)[launches:]  # the first sample warms up
                samples = [sum(times[k : k + launches]) for k in range(0, len(times), launches)]
                error = max_error(out, order, expected) / scale
                itemsize = np.dtype(npt).itemsize
                cells = args.nblks * args.nproma
                moved = cells * (3 * itemsize + 6 * 4) + 2 * cells * args.nlev * itemsize  # every array once
                best = min(times)
                row = {
                    **config,
                    "staged": staged,
                    "median_ms": statistics.median(times),
                    "min_ms": best,
                    "samples": len(times),
                    "launches_per_sample": launches,
                    "sample_ms": statistics.median(samples),
                    "effective_GBps": moved / (best * 1e-3) / 1e9,
                    "max_error": error,
                    "correct": error <= TOLERANCE[precision],
                    "build_s": build_s,
                }
            except Exception as failure:  # noqa: BLE001 -- a configuration's failure is a result; the sweep goes on
                traceback.print_exc()
                row = {**config, "failed": f"{type(failure).__name__}: {failure}", "correct": False}
            compiled = harness = None  # its device copies, before the next configuration allocates its own
            rows.append(row)
            print(json.dumps(row), flush=True)
        del inputs, out
    return rows


def table(rows: Sequence[dict]) -> str:
    """Markdown: every configuration's time, its speedup over the unstaged kernel of the same layout, precision,
    coarsening and chiplet distribution, and over klon fp64 uncoarsened unstaged undistributed."""

    def key(r: dict) -> tuple:
        return r["layout"], r["precision"], r["coarsen"], r["chiplets"]

    failed = [r for r in rows if "failed" in r]
    rows = [r for r in rows if "failed" not in r]
    base = {key(r): r["median_ms"] for r in rows if r["variant"] == "base"}
    reference = base.get(("klon", "fp64", "1", 1))
    columns = "layout precision coarsen chiplets variant median_ms GB/s vs_unstaged vs_klon_fp64 max_error staged"
    lines = [
        "| " + " | ".join(c.replace("_", " ") for c in columns.split()) + " |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|---:|---|",
    ]

    def speedup(over: float | None, r: dict) -> str:
        return f"{over / r['median_ms']:.2f}x" if over else "-"

    for r in rows:
        lines.append(
            f"| {r['layout']} | {r['precision']} | {r['coarsen']} | {r['chiplets']} | {r['variant']} | "
            f"{r['median_ms']:.3f} | "
            f"{r['effective_GBps']:.0f} | {speedup(base.get(key(r)), r)} | "
            f"{speedup(reference, r)} | {r['max_error']:.1e}"
            f"{'' if r['correct'] else ' WRONG'} | {', '.join(r['staged'])} |"
        )
    lines += [f"\nFAILED {' '.join(str(v) for v in key(r))} {r['variant']}: {r['failed']}" for r in failed]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--nproma",
        type=int,
        default=16_000_000,
        help="cells per block: an fp64 field of nlev=144 levels is 18.4 GB, about the most the 128 GB of an MI300A "
        "holds with the device copies and the reference",
    )
    parser.add_argument("--nlev", type=int, default=144)
    parser.add_argument("--nblks", type=int, default=1)
    parser.add_argument("--reps", type=int, default=50, help="timed samples per configuration (one more warms up)")
    parser.add_argument("--sample-ms", type=float, default=100.0, help="least GPU time of one sample")
    parser.add_argument("--warmup", type=int, default=1, help="calibration samples dropped")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--layouts", nargs="+", default=list(LAYOUTS), choices=list(LAYOUTS))
    parser.add_argument("--precisions", nargs="+", default=list(PRECISIONS), choices=list(PRECISIONS))
    parser.add_argument("--variants", nargs="+", default=list(VARIANTS), choices=list(VARIANTS))
    parser.add_argument(
        "--coarsen",
        nargs="+",
        default=["1", "2", "3", "4", "6", "8", "9", "16", "2x2", "3x2", "4x2", "6x2", "8x2", "9x2"],
        help="elements per thread, LEVELS or LEVELSxCELLS (CoarsenKernelThreads factors); a 16-row block covers 16 x "
        "LEVELS levels, so 9 is one whole column of 144",
    )
    parser.add_argument(
        "--chiplets",
        nargs="+",
        type=int,
        default=[1, 6],
        help="chiplets (XCDs) the thread blocks of a kernel are distributed over (compiler.cuda.chiplet_number): 1 is "
        "no distribution, 6 the XCDs of an MI300A",
    )
    parser.add_argument("--out", type=pathlib.Path, default=pathlib.Path("zekin_staging.json"))
    args = parser.parse_args()
    rows = run(args)
    args.out.write_text(
        json.dumps({"arguments": {k: str(v) for k, v in vars(args).items()}, "results": rows}, indent=1)
    )
    markdown = table(rows)
    args.out.with_suffix(".md").write_text(markdown + "\n")
    print(markdown)
    if not all(r["correct"] for r in rows):
        raise SystemExit("some configuration computed a wrong result")


if __name__ == "__main__":
    main()
