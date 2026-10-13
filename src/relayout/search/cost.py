# Copyright 2026 ETH Zurich and the relayout authors.
"""What a map nest costs in a layout, by the LogGP model of one memory level.

A nest moves ``B`` bytes in ``M`` messages (cache lines, GPU 128 B lines): its time is
``max(B * G, M * L / C) + o * M`` (:func:`nest_memory_time`), bandwidth-bound or latency-bound by the
``C`` requests it keeps in flight. A layout changes ``B`` and ``M`` only: :func:`count_nest` counts the
new blocks each array's accesses touch per iteration from the index expressions and the laid-out
strides. A message is a request (``line_bytes``), a transfer is a sector (``sector_bytes``, smaller on a
GPU), and a written block moves twice (fetched, written back).

The parameters come from a microbenchmark: :func:`fit_message_size` fits ``T(n) = alpha + beta * n``
over message sizes and :func:`validate` cross-checks the fit against the measured parameters.
"""

import copy
import math
from collections.abc import Sequence
from dataclasses import dataclass

import dace
import numpy
import sympy
from dace import data, nodes, subsets, symbolic
from dace.sdfg.narrowing import SymbolicLike, as_expr
from dace.sdfg.state import SDFGState

from relayout.search.program import symbol_values

__all__ = [
    "ArrayTraffic",
    "Fit",
    "LogGP",
    "NestCounts",
    "count_nest",
    "exposed_concurrency",
    "fit_message_size",
    "gap_from_bandwidth",
    "nest_memory_time",
    "relayout_time",
    "replayed_blocks",
    "validate",
]

#: Storage that is no memory traffic: registers and GPU shared memory.
LOCAL_STORAGE = frozenset({dace.StorageType.Register, dace.StorageType.GPU_Shared})

GPU_SCHEDULES = frozenset({dace.ScheduleType.GPU_Device, dace.ScheduleType.GPU_ThreadBlock})

#: Schedules whose iterations run concurrently.
PARALLEL_SCHEDULES = GPU_SCHEDULES | {dace.ScheduleType.CPU_Multicore, dace.ScheduleType.Default}


@dataclass(frozen=True, slots=True)
class LogGP:
    """The LogGP parameters of one memory level, in seconds and seconds per byte."""

    L: float  # unloaded round trip of one line
    o: float  # overhead per message
    g: float  # minimum interval between two requests of one core
    G: float  # gap per byte: 1 / bw_saturated
    line_bytes: int  # what one request covers (64 B x86, 128 B NVIDIA/AMD)
    sector_bytes: int  # what one transfer moves (the line on x86, a 32 B sector on a GPU)
    bw_saturated: float  # bytes/s of all cores
    bw_core: float  # bytes/s of one core
    c_core: float | None = None  # measured outstanding-miss knee of one core; L / g when unmeasured

    @property
    def core_mlp(self) -> float:
        """Demand misses one core keeps in flight."""
        return self.L / self.g if self.c_core is None else self.c_core

    @property
    def core_stream_mlp(self) -> float:
        """Requests one core keeps in flight on a prefetched stream (Little's law)."""
        return self.bw_core * self.L / self.line_bytes


def gap_from_bandwidth(bytes_per_s: float) -> float:
    return 1.0 / bytes_per_s


def nest_memory_time(p: LogGP, bytes_moved: float, messages: float, concurrency: float) -> float:
    """``max(bytes_moved * G, messages * L / concurrency) + o * messages``: ``messages`` counted at request
    granularity, ``bytes_moved`` at transfer granularity."""
    return max(bytes_moved * p.G, messages * p.L / concurrency) + p.o * messages


def relayout_time(desc: data.Array, p: LogGP) -> sympy.Expr:
    """A relayout streaming the array in and out at the saturated bandwidth (a lower bound)."""
    return 2 * as_expr(desc.total_size) * desc.dtype.bytes / p.bw_saturated


@dataclass(frozen=True, slots=True)
class Fit:
    """``T(n) = alpha + beta * n`` over a message-size sweep, with its relative RMS residual."""

    alpha: float  # s; identifies with L
    beta: float  # s/byte; identifies with 1 / bw_core
    residual: float


def fit_message_size(sizes: Sequence[int], times: Sequence[float]) -> Fit:
    """Least squares weighted by ``1 / T`` (timing error is multiplicative)."""
    if len(sizes) != len(times) or len(sizes) < 2:
        raise ValueError("fit_message_size needs at least two (size, time) points of equal length")
    x, y = numpy.asarray(sizes, dtype=float), numpy.asarray(times, dtype=float)
    w = numpy.where(y > 0, 1.0 / numpy.where(y > 0, y, 1.0), 0.0)
    sw, swx, swy = w.sum(), (w * x).sum(), (w * y).sum()
    denominator = sw * (w * x * x).sum() - swx * swx
    if denominator == 0:
        raise ValueError("fit_message_size: all sizes are equal")
    beta = (sw * (w * x * y).sum() - swx * swy) / denominator
    alpha = (swy - beta * swx) / sw
    spread = (w * (y - swy / sw) ** 2).sum()
    error = (w * (y - alpha - beta * x) ** 2).sum()
    return Fit(float(alpha), float(beta), math.sqrt(error / spread) if spread > 0 else 0.0)


def relative_gap(a: float, b: float) -> float:
    scale = max(abs(a), abs(b))
    return abs(a - b) / scale if scale > 0 else 0.0


def validate(p: LogGP, fit: Fit, peak_bytes_per_s: float, knee: float, tolerance: float = 0.2) -> list[str]:
    """The reasons to reject ``p`` (empty: accept): each parameter checked a second, independent way."""
    if min(p.L, p.G, p.g) <= 0:
        return ["L, G and g must be positive"]
    reasons = []
    if relative_gap(fit.alpha, p.L) > tolerance:
        reasons.append(f"the fit's intercept {fit.alpha:.3g} s is not the measured L {p.L:.3g} s")
    if relative_gap(fit.beta, 1.0 / p.bw_core) > tolerance:
        reasons.append(f"the fit's slope {fit.beta:.3g} s/B is not one core's 1 / bandwidth {1.0 / p.bw_core:.3g} s/B")
    if knee > p.L / p.g * (1 + tolerance):
        reasons.append(f"the outstanding-miss knee {knee:.3g} exceeds the issue cap L / g = {p.L / p.g:.3g}")
    if p.bw_saturated > peak_bytes_per_s:
        reasons.append(f"the saturated bandwidth {p.bw_saturated:.3g} B/s exceeds the peak {peak_bytes_per_s:.3g} B/s")
    return reasons


@dataclass(frozen=True, slots=True)
class ArrayTraffic:
    """New blocks one array's accesses touch per iteration of a nest (symbolic)."""

    messages: sympy.Expr  # at request granularity
    sectors: sympy.Expr  # at transfer granularity
    bytes_moved: sympy.Expr  # sectors * sector_bytes, twice for a written array


@dataclass(frozen=True, slots=True)
class NestCounts:
    """A nest's iterations and the per-iteration traffic of each array in memory."""

    iterations: sympy.Expr
    arrays: dict[str, ArrayTraffic]

    def messages(self, symbols: dict[str, int], name: str | None = None) -> float:
        return self.total(symbols, [t.messages for n, t in self.arrays.items() if name in (None, n)])

    def bytes_moved(self, symbols: dict[str, int], name: str | None = None) -> float:
        return self.total(symbols, [t.bytes_moved for n, t in self.arrays.items() if name in (None, n)])

    def total(self, symbols: dict[str, int], per_iteration: list[sympy.Expr]) -> float:
        return float(symbolic.evaluate(sympy.Add(*per_iteration, 0) * self.iterations, symbol_values(symbols)))


def nest_maps(state: SDFGState, entry: nodes.MapEntry) -> list[nodes.MapEntry]:
    """``entry`` and every map inside its scope, outermost first."""
    scopes = state.scope_dict()

    def depth(node: nodes.MapEntry) -> int:
        level, outer = 0, scopes[node]
        while outer is not None:
            level, outer = level + 1, scopes[outer]
        return level

    inner = [n for n in state.scope_subgraph(entry).nodes() if isinstance(n, nodes.MapEntry)]
    return sorted(inner, key=depth)


def leaf_memlets(state: SDFGState, entry: nodes.MapEntry) -> list[tuple[str, dace.Memlet, bool]]:
    """``(array, memlet, written)`` for each point access of a code node in ``entry``'s scope. A tasklet input
    fed by a transient scalar (``A[i] -> A_index -> tasklet``) is the access of what fills the scalar."""
    sdfg = state.sdfg
    out = []
    for edge in state.scope_subgraph(entry).edges():
        if edge.data.is_empty():
            continue
        if isinstance(edge.dst, nodes.CodeNode):
            memlet = edge.data
            source = edge.src
            if isinstance(source, nodes.AccessNode) and is_register(sdfg, source.data):
                feeding = state.in_edges(source)
                memlet = feeding[0].data if len(feeding) == 1 and not feeding[0].data.is_empty() else memlet
            out.append((memlet, False))
        elif isinstance(edge.src, nodes.CodeNode):
            out.append((edge.data, True))
    return [(m.data, m, w) for m, w in out if m.data is not None and not is_register(sdfg, m.data)]


def is_register(sdfg: dace.SDFG, name: str) -> bool:
    desc = sdfg.arrays.get(name)
    return isinstance(desc, data.Scalar) and desc.transient


def union(a: subsets.Range, b: subsets.Range) -> subsets.Range:
    merged = subsets.union(a, b)
    if isinstance(merged, subsets.Range):
        return merged
    return subsets.Range(
        [
            (sympy.Min(ab, bb), sympy.Max(ae, be), astep if astep == bstep else 1)
            for (ab, ae, astep), (bb, be, bstep) in zip(a.ranges, b.ranges, strict=True)
        ]
    )


def iterations(begin: SymbolicLike, end: SymbolicLike, step: SymbolicLike) -> sympy.Expr:
    return symbolic.int_floor(as_expr(end) - as_expr(begin), as_expr(step)) + 1


def blocks_per_iteration(
    name: str,
    index: Sequence[SymbolicLike],
    strides: Sequence[SymbolicLike],
    loops: list[tuple[str, tuple]],
    block: int,
) -> sympy.Expr:
    """The average new blocks of ``block`` elements an access at ``index`` touches per iteration of
    ``loops`` (outermost first): each step of a loop moves the address by its stride, a fraction
    ``min(1, |stride| / block)`` of a new block, every time the loops outside it come around."""
    address = sympy.Add(*[as_expr(i) * as_expr(s) for i, s in zip(index, strides, strict=True)])
    extents = [iterations(*rng) for _, rng in loops]
    total = sympy.Integer(1)
    for depth, (param, (_, _, step)) in enumerate(loops):
        moved, symbol, delta = symbolic.equalize_symbols_across(address, symbolic.symbol(param), as_expr(step))
        stride = sympy.simplify(moved.subs(symbol, symbol + delta) - moved)
        if symbol in stride.free_symbols:
            raise ValueError(
                f"{name!r} is indexed by {list(index)}, not affine in {param!r} (a // or %, or a product of "
                f"parameters): replay its accesses with replayed_blocks and pass them as replayed counts"
            )
        outer = sympy.Mul(*extents[:depth])
        total += outer * (extents[depth] - 1) * sympy.Min(1, sympy.Abs(stride) / block)
    average = as_expr(symbolic.simplify(total / sympy.Mul(*extents)))
    leaked = {str(s) for s in average.free_symbols} & {param for param, _ in loops}
    if leaked:
        raise ValueError(f"{name!r} is indexed by {list(index)}, not affine in {sorted(leaked)} together")
    return average


def count_nest(
    state: SDFGState,
    entry: nodes.MapEntry,
    line_bytes: int,
    sector_bytes: int,
    replayed: dict[str, tuple[float, float]] | None = None,
) -> NestCounts:
    """The traffic of the nest at ``entry``. An array behind a data-dependent (dynamic) access has no
    static count: pass its per-iteration ``(messages, sectors)`` in ``replayed`` (:func:`replayed_blocks`)."""
    sdfg = state.sdfg
    loops = [(p, rng) for m in nest_maps(state, entry) for p, rng in zip(m.map.params, m.map.range, strict=True)]
    accessed: dict[str, subsets.Range] = {}
    written: set[str] = set()
    dynamic: set[str] = set()
    for name, memlet, writes in leaf_memlets(state, entry):
        subset = memlet.subset
        if not isinstance(subset, subsets.Range):
            raise TypeError(f"{name!r} is accessed by {subset}, not a range")
        accessed[name] = union(accessed[name], subset) if name in accessed else copy.deepcopy(subset)
        written.update([name] if writes else [])
        dynamic.update([name] if memlet.dynamic else [])
    replayed = replayed or {}
    arrays = {}
    for name, subset in accessed.items():
        desc = sdfg.arrays[name]
        if desc.storage in LOCAL_STORAGE:
            continue
        if name in replayed:
            messages, sectors = (sympy.sympify(v) for v in replayed[name])
        elif name in dynamic:
            raise ValueError(f"{name!r} is accessed through a data-dependent memlet: pass its replayed counts")
        else:
            index = [begin for begin, _, _ in subset.ranges]
            counts = []
            for block in (line_bytes, sector_bytes):
                elements, span = max(1, block // desc.dtype.bytes), max(1, -(-desc.dtype.bytes // block))
                counts.append(blocks_per_iteration(name, index, desc.strides, loops, elements) * span)
            messages, sectors = counts
        moved = sectors * sector_bytes * (2 if name in written else 1)
        arrays[name] = ArrayTraffic(messages, sectors, moved)
    return NestCounts(sympy.Mul(*[iterations(*rng) for _, rng in loops]), arrays)


def replayed_blocks(indices: numpy.ndarray, block_elements: int) -> tuple[float, float]:
    """Blocks per access of an indirect access replayed from its index sequence (in traversal order):
    ``(streaming, distinct)``, the upper bound with no reuse beyond the last block and the lower bound
    with every block reused."""
    if block_elements < 1:
        raise ValueError(f"block_elements must be positive, got {block_elements}")
    if indices.ndim != 1:
        raise ValueError(f"replay a 1-D access sequence in traversal order, got shape {indices.shape}")
    if indices.size == 0:
        return 0.0, 0.0
    blocks = indices // block_elements
    streaming = (1 + int(numpy.count_nonzero(blocks[1:] != blocks[:-1]))) / indices.size
    return streaming, int(numpy.unique(blocks).size) / indices.size


def exposed_concurrency(state: SDFGState, entry: nodes.MapEntry, p: LogGP, cores: int | None) -> float:
    """The requests the nest keeps in flight: one core's on a sequential nest, ``cores`` times that on a
    parallel one (unbounded when ``cores`` is None). A GPU counts demand misses, a CPU its prefetched
    streams."""
    schedules = {m.map.schedule for m in nest_maps(state, entry)}
    unit = p.core_mlp if schedules & GPU_SCHEDULES else p.core_stream_mlp
    if not schedules & PARALLEL_SCHEDULES:
        return unit
    return math.inf if cores is None else cores * unit
