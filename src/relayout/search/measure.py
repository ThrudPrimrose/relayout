# Copyright 2026 ETH Zurich and the relayout authors.
"""Measured layouts: every candidate compiled, run, checked against the program's own result, and the
correct ones timed (the cost model's ground truth, and the search when no model is trusted).

A candidate's conversions are no part of its time: :func:`compute_timer` times the states that compute,
not the ``relayout_in`` / ``relayout_out`` conversions around them. Times whose spread exceeds
:data:`CONTENDED` are kept but marked contended.
"""

import contextlib
import copy
import statistics
import time
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import dace
import numpy
from dace import nodes

from relayout.passes.sdfg.common import CONVERSION
from relayout.ops import Location
from relayout.search.layouts import IDENTITY, Layout, lay_out
from relayout.search.program import Kernel, extract_nest, nest_arguments, rename, written_arrays

__all__ = [
    "CONTENDED",
    "Candidate",
    "NestEvaluation",
    "SweepResult",
    "Timing",
    "best",
    "compute_timer",
    "evaluate_nest",
    "sweep",
    "time_call",
]

#: The relative spread of repeated timings above which a median is contended.
CONTENDED = 0.10


type Outputs = dict[str, numpy.ndarray]
type Run = Callable[[dace.SDFG], Outputs]


@dataclass(frozen=True, slots=True)
class Timing:
    """Seconds of repeated runs."""

    samples: tuple[float, ...]

    @property
    def median(self) -> float:
        return statistics.median(self.samples)

    @property
    def contended(self) -> bool:
        low = min(self.samples)
        return low > 0 and (max(self.samples) - low) / low > CONTENDED


type Timer = Callable[[dace.SDFG, Run, int, int], Timing]


@dataclass(slots=True)
class SweepResult:
    """One candidate: whether it computed the reference, its time, or why it failed; ``order`` is its
    place in the enumeration, which breaks ties."""

    name: str
    order: int
    correct: bool = False
    timing: Timing | None = None
    error: str | None = None

    @property
    def seconds(self) -> float | None:
        return None if self.timing is None else self.timing.median


def time_call(sdfg: dace.SDFG, run: Run, reps: int, warmup: int) -> Timing:
    """The wall time of whole calls."""
    for _ in range(warmup):
        run(sdfg)
    samples = []
    for _ in range(reps):
        start = time.perf_counter()
        run(sdfg)
        samples.append(time.perf_counter() - start)
    return Timing(tuple(samples))


def is_conversion(state: dace.SDFGState) -> bool:
    entries = [n for n in state.nodes() if isinstance(n, nodes.MapEntry)]
    return bool(entries) and all(e.map.label.startswith(CONVERSION) for e in entries)


def on_gpu(state: dace.SDFGState) -> bool:
    return any(
        isinstance(n, nodes.MapEntry) and n.map.schedule in dace.dtypes.GPU_SCHEDULES
        for n, _ in state.all_nodes_recursive()
    )


def compute_timer(sdfg: dace.SDFG, run: Run, reps: int, warmup: int) -> Timing:
    """The time of the states that compute (host timers, or GPU events on a GPU state), not of the
    conversions; instruments ``sdfg``. Every run must write a fresh report."""
    for state in sdfg.states():
        if not is_conversion(state) and any(isinstance(n, nodes.CodeNode) for n in state.nodes()):
            state.instrument = dace.InstrumentationType.GPU_Events if on_gpu(state) else dace.InstrumentationType.Timer
    for _ in range(warmup):
        run(sdfg)
    samples = []
    for rep in range(reps):
        previous = sdfg.get_latest_report_path()
        run(sdfg)
        path = sdfg.get_latest_report_path()
        report = sdfg.get_latest_report()
        if path is None or path == previous or report is None:
            raise RuntimeError(f"compute_timer: run {rep} of {sdfg.name!r} wrote no fresh instrumentation report")
        durations = [
            sum(values) / len(values)
            for by_state in report.durations.values()
            for by_node in by_state.values()
            for values in by_node.values()
            if values
        ]
        if not durations:
            raise RuntimeError(f"compute_timer: {sdfg.name!r} has no state that computes")
        samples.append(sum(durations) * 1e-3)
    return Timing(tuple(samples))


@contextlib.contextmanager
def one_stream() -> Iterator[None]:
    """dace emits one GPU stream, so events around a call see all its kernels; compile inside."""
    with dace.config.set_temporary("compiler", "cuda", "max_concurrent_streams", value=-1):
        yield


@dataclass(frozen=True, slots=True)
class Candidate:
    """A named way to build one SDFG of the sweep."""

    name: str
    build: Callable[[], dace.SDFG]


def sweep(
    candidates: list[Candidate],
    run: Run,
    reference: Outputs,
    timer: Timer = compute_timer,
    reps: int = 10,
    warmup: int = 2,
    location: Location = Location.HOST,
) -> list[SweepResult]:
    """Build, run and check every candidate against ``reference``, then time the correct ones back to
    back; ranked correct first, fastest first. A candidate that fails to build or run is incorrect with
    its error; a timing failure keeps it correct and untimed."""
    results = []
    checked: list[tuple[SweepResult, dace.SDFG]] = []
    with one_stream() if location is Location.DEVICE else contextlib.nullcontext():
        for order, candidate in enumerate(candidates):
            result = SweepResult(candidate.name, order)
            results.append(result)
            try:
                sdfg = candidate.build()
                out = run(sdfg)
            except Exception as error:  # noqa: BLE001  # any build or run failure disqualifies the candidate alone
                result.error = f"{type(error).__name__}: {error}"
                continue
            result.correct = all(k in out and numpy.allclose(out[k], v) for k, v in reference.items())
            if result.correct:
                checked.append((result, sdfg))
        for result, sdfg in checked:
            try:
                result.timing = timer(sdfg, run, reps, warmup)
            except Exception as error:  # noqa: BLE001  # an untimed candidate is still correct
                result.error = f"timing: {type(error).__name__}: {error}"
    results.sort(key=lambda r: (not r.correct, r.seconds if r.seconds is not None else float("inf")))
    return results


def best(results: list[SweepResult], noise: float = CONTENDED) -> SweepResult | None:
    """The fastest correct candidate, the earliest enumerated among those within ``noise`` of it (the
    earliest correct one when none was timed); None when none is correct."""
    timed = [r for r in results if r.correct and r.seconds is not None and r.seconds > 0]
    if not timed:
        return min((r for r in results if r.correct), key=lambda r: r.order, default=None)
    fastest = min(r.seconds for r in timed if r.seconds is not None)
    return min(
        (r for r in timed if r.seconds is not None and r.seconds <= fastest * (1 + noise)), key=lambda r: r.order
    )


@dataclass(frozen=True, slots=True)
class NestEvaluation:
    """One kernel's candidates ranked: ``results[k].name`` is ``array:tag`` or ``identity``."""

    nest: dace.SDFG
    results: list[SweepResult]

    def by_name(self) -> dict[str, SweepResult]:
        return {r.name: r for r in self.results}


def candidate_name(array: str, layout: Layout) -> str:
    return IDENTITY if layout.is_identity else f"{array}:{layout.tag}"


def evaluate_nest(
    kernel: Kernel,
    layouts: dict[str, list[Layout]],
    symbols: dict[str, int],
    provided: dict[str, numpy.ndarray] | None = None,
    timer: Timer = compute_timer,
    reps: int = 10,
    warmup: int = 2,
    seed: int = 0,
    name: str | None = None,
) -> NestEvaluation:
    """The kernel cut out, run as it is for the reference, and every array the kernel touches laid out by
    each of its non-identity ``layouts`` (one array at a time) and swept."""
    nest = extract_nest(kernel, name or f"{kernel.state.sdfg.name}_k{kernel.index}")
    outputs = sorted(written_arrays(nest))
    if not outputs:
        raise ValueError(f"evaluate_nest: {nest.name!r} writes no argument to check candidates against")
    arguments = nest_arguments(nest, symbols, provided, seed)
    call_symbols = {k: v for k, v in symbols.items() if k in nest.symbols or k in nest.free_symbols}

    def run(sdfg: dace.SDFG) -> Outputs:
        args = {k: v.copy() for k, v in arguments.items()}
        sdfg(**args, **call_symbols)
        return {k: args[k] for k in outputs}

    def build(array: str, layout: Layout) -> Callable[[], dace.SDFG]:
        def make() -> dace.SDFG:
            sdfg = copy.deepcopy(nest)
            rename(sdfg, f"{nest.name}_{candidate_name(array, layout).replace(':', '_')}")
            if not layout.is_identity:
                lay_out(sdfg, array, layout)
            return sdfg

        return make

    reference_sdfg = copy.deepcopy(nest)
    rename(reference_sdfg, f"{nest.name}_reference")
    reference = run(reference_sdfg)
    candidates = [Candidate(IDENTITY, build("", Layout(IDENTITY)))] + [
        Candidate(candidate_name(array, layout), build(array, layout))
        for array in sorted(layouts)
        if array in nest.arrays
        for layout in layouts[array]
        if not layout.is_identity
    ]
    return NestEvaluation(nest, sweep(candidates, run, reference, timer, reps, warmup))
