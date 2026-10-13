# Copyright 2026 ETH Zurich and the relayout authors.
"""One layout per array per kernel of the line, chosen globally and applied.

A trajectory gives an array a layout at every kernel. It costs the kernels' times in their layouts, a
relayout at every change, a conversion in before the first kernel when the program's value is needed
there, and one back after the last write (:func:`trajectory_cost`). :func:`per_array_dp` finds each
array's cheapest trajectory exactly (Viterbi), :func:`brute_force_trajectories` checks it by
enumeration, and :func:`greedy_assignment` is the per-kernel baseline. The costs come from the LogGP
model (:func:`model_costs`) or from measurement (:func:`eval_costs`).

:func:`apply_assignment` carries a choice out: each run of kernels in one layout works on a buffer in
that layout, converted from the previous one at the boundary (before a loop, never inside it), and the
program's own buffer is restored after the last write.
"""

import itertools
from dataclasses import dataclass, field

import dace
from dace import symbolic
from dace.sdfg import propagation

from relayout.frontends.schedule import canonicalize_schedule
from relayout.frontends.sdfg import CONVERSION, Laid, covers, laid_out_descriptor, rewrite_accesses, transfer
from relayout.layout import IndexMap
from relayout.search.cost import LogGP, count_nest, exposed_concurrency, nest_memory_time, relayout_time
from relayout.search.layouts import Layout, lay_out
from relayout.search.measure import candidate_name, evaluate_nest
from relayout.search.program import Kernel, extract_nest, loop_spans, nest_entries, symbol_values

__all__ = [
    "AppliedAssignment",
    "ArrayTrajectory",
    "AssignmentCosts",
    "ConflictRow",
    "apply_assignment",
    "apply_region_layout",
    "brute_force_trajectories",
    "conflict_report",
    "eval_costs",
    "greedy_assignment",
    "model_costs",
    "per_array_dp",
    "to_assignment",
    "trajectory_cost",
]

type NodeKey = tuple[str, int, str]  # array, kernel, layout tag
type EdgeKey = tuple[str, str, str]  # array, from tag, to tag


@dataclass(slots=True)
class AssignmentCosts:
    """Each array's candidate layouts (the identity first), the time of every kernel in every layout of
    every array, the relayout times, whether the first kernel needs the program's value
    (``entry_needed``), the kernel of the last write, and the measured costs not to trust."""

    layouts: dict[str, list[Layout]]
    node_cost: dict[NodeKey, float]
    relayout_cost: dict[EdgeKey, float]
    entry_needed: dict[str, bool] = field(default_factory=dict)
    last_write: dict[str, int] = field(default_factory=dict)
    untrusted: set[NodeKey] = field(default_factory=set)

    def check(self, kernels: int) -> None:
        for array, layouts in self.layouts.items():
            tags = [layout.tag for layout in layouts]
            if not layouts or not layouts[0].is_identity or len(set(tags)) != len(tags):
                raise ValueError(f"{array!r}: the layouts must be distinct and start with the identity, got {tags}")
            missing = [(array, k, t) for k in range(kernels) for t in tags if (array, k, t) not in self.node_cost]
            missing += [
                (array, a, b) for a, b in itertools.permutations(tags, 2) if (array, a, b) not in self.relayout_cost
            ]
            if missing:
                raise ValueError(f"AssignmentCosts misses {missing[:3]}")
            if not 0 <= self.last_write.get(array, 0) < kernels:
                raise ValueError(f"{array!r}: last write at kernel {self.last_write[array]} of {kernels}")


@dataclass(frozen=True, slots=True)
class ArrayTrajectory:
    array: str
    tags: list[str]
    cost: float


def trajectory_cost(costs: AssignmentCosts, array: str, tags: list[str]) -> float:
    """The kernels' times, a relayout at every change, the conversion in (when needed and the first
    layout is not the identity) and the conversion back after the last write; a change back to the
    identity after the last write is that conversion back, paid once."""
    identity = costs.layouts[array][0].tag
    last = costs.last_write.get(array)
    total = sum(costs.node_cost[(array, k, tag)] for k, tag in enumerate(tags))
    if costs.entry_needed.get(array, False) and tags[0] != identity:
        total += costs.relayout_cost[(array, identity, tags[0])]
    if last is not None and tags[last] != identity:
        total += costs.relayout_cost[(array, tags[last], identity)]
    restored = last is not None and tags[last] == identity
    for k in range(len(tags) - 1):
        if tags[k] != tags[k + 1] and not (last is not None and k >= last and tags[k + 1] == identity and not restored):
            total += costs.relayout_cost[(array, tags[k], tags[k + 1])]
        restored = restored or (last is not None and k + 1 >= last and tags[k + 1] == identity)
    return total


def node_cost(costs: AssignmentCosts, array: str, tags: list[str], k: int, j: int) -> float:
    """Kernel ``k`` in layout ``tags[j]``, with the conversion in (at the first kernel) and back (at the
    last write) a layout other than the identity pays there."""
    cost = costs.node_cost[(array, k, tags[j])]
    if k == 0 and costs.entry_needed.get(array, False) and j != 0:
        cost += costs.relayout_cost[(array, tags[0], tags[j])]
    if k == costs.last_write.get(array) and j != 0:
        cost += costs.relayout_cost[(array, tags[j], tags[0])]
    return cost


def per_array_dp(
    costs: AssignmentCosts, kernels: int, allow_changes: bool = True, locked: set[int] | None = None
) -> dict[str, ArrayTrajectory]:
    """Each array's cheapest trajectory (:func:`trajectory_cost`), ties to the earliest enumerated; a
    kernel in ``locked`` keeps the layout of the one before (a loop body: :func:`locked_transitions`),
    and without ``allow_changes`` an array keeps one layout throughout."""
    locked = locked or set()
    costs.check(kernels)
    solution = {}
    for array, layouts in costs.layouts.items():
        tags = [layout.tag for layout in layouts]
        last = costs.last_write.get(array)

        # best[(j, restored)]: the cheapest (cost, path) ending in layout j, restored = the identity was
        # reached at or after the last write; the future depends on (j, restored) alone
        best: dict[tuple[int, bool], tuple[float, tuple[int, ...]]] = {
            (j, last == 0 and j == 0): (node_cost(costs, array, tags, 0, j), (j,)) for j in range(len(tags))
        }
        for k in range(1, kernels):
            following: dict[tuple[int, bool], tuple[float, tuple[int, ...]]] = {}
            for (i, restored), (cost, path) in best.items():
                for j in range(len(tags)):
                    if i != j and (not allow_changes or k in locked):
                        continue
                    exit_here = last is not None and k - 1 >= last and j == 0 and not restored
                    edge = 0.0 if i == j or exit_here else costs.relayout_cost[(array, tags[i], tags[j])]
                    key = (j, restored or (last is not None and k >= last and j == 0))
                    candidate = (cost + edge + node_cost(costs, array, tags, k, j), (*path, j))
                    if key not in following or candidate < following[key]:
                        following[key] = candidate
            best = following
        cost, path = min(best.values())
        solution[array] = ArrayTrajectory(array, [tags[j] for j in path], cost)
    return solution


def brute_force_trajectories(
    costs: AssignmentCosts, kernels: int, allow_changes: bool = True, locked: set[int] | None = None, cap: int = 10**6
) -> dict[str, ArrayTrajectory]:
    """:func:`per_array_dp` by enumerating every trajectory (at most ``cap`` per array)."""
    locked = locked or set()
    costs.check(kernels)
    solution = {}
    for array, layouts in costs.layouts.items():
        tags = [layout.tag for layout in layouts]
        if allow_changes and len(tags) ** kernels > cap:
            raise ValueError(f"{array!r}: {len(tags) ** kernels} trajectories exceed the cap {cap}")
        options = itertools.product(tags, repeat=kernels) if allow_changes else ((t,) * kernels for t in tags)
        found: ArrayTrajectory | None = None
        for option in options:
            if any(option[k] != option[k - 1] for k in locked):
                continue
            cost = trajectory_cost(costs, array, list(option))
            if found is None or cost < found.cost:
                found = ArrayTrajectory(array, list(option), cost)
        assert found is not None
        solution[array] = found
    return solution


def greedy_assignment(
    costs: AssignmentCosts, kernels: int, locked: set[int] | None = None
) -> dict[str, ArrayTrajectory]:
    """Each kernel (each loop body, as one) in its own cheapest layout, paying the conversions that follows."""
    locked = locked or set()
    costs.check(kernels)
    runs: list[list[int]] = []
    for k in range(kernels):
        if k in locked and runs:
            runs[-1].append(k)
        else:
            runs.append([k])
    solution = {}
    for array, layouts in costs.layouts.items():
        tags = [layout.tag for layout in layouts]
        trajectory = []
        for run in runs:
            pick = min(tags, key=lambda t: (sum(costs.node_cost[(array, k, t)] for k in run), tags.index(t)))
            trajectory += [pick] * len(run)
        solution[array] = ArrayTrajectory(array, trajectory, trajectory_cost(costs, array, trajectory))
    return solution


@dataclass(frozen=True, slots=True)
class ConflictRow:
    """An array's per-kernel preferences, its chosen trajectory, and the greedy, global and one-layout
    costs; ``untrusted`` when a chosen cost was measured contended."""

    array: str
    preferred: list[str]
    chosen: list[str]
    greedy_cost: float
    global_cost: float
    single_cost: float
    untrusted: bool

    @property
    def conflicting(self) -> bool:
        return len(set(self.preferred)) > 1


def conflict_report(costs: AssignmentCosts, kernels: int, locked: set[int] | None = None) -> list[ConflictRow]:
    greedy = greedy_assignment(costs, kernels, locked)
    chosen = per_array_dp(costs, kernels, True, locked)
    single = per_array_dp(costs, kernels, False, locked)
    return [
        ConflictRow(
            array,
            greedy[array].tags,
            chosen[array].tags,
            greedy[array].cost,
            chosen[array].cost,
            single[array].cost,
            any((array, k, t) in costs.untrusted for k, t in enumerate(chosen[array].tags)),
        )
        for array in sorted(costs.layouts)
    ]


def to_assignment(
    trajectories: dict[str, ArrayTrajectory], layouts: dict[str, list[Layout]]
) -> dict[str, list[Layout]]:
    """The trajectories as :func:`apply_assignment` takes them; all-identity ones left out."""
    out = {}
    for array, trajectory in trajectories.items():
        by_tag = {layout.tag: layout for layout in layouts[array]}
        if any(not by_tag[t].is_identity for t in trajectory.tags):
            out[array] = [by_tag[t] for t in trajectory.tags]
    return out


def reads(state: dace.SDFGState, array: str) -> bool:
    """Whether ``state`` reads ``array``: an access node it leaves, or a reduction onto it."""
    return any(n.data == array and state.out_degree(n) > 0 for n in state.data_nodes()) or any(
        e.data.data == array and e.data.wcr is not None for e in state.edges()
    )


def liveness(kernels: list[Kernel], arrays: list[str]) -> tuple[dict[str, bool], dict[str, int]]:
    """Per array: whether its first kernel needs its value, and the kernel of its last write."""
    entry, last = {}, {}
    for array in arrays:
        touching = [k for k in kernels if k.touches(array)]
        if touching:
            first = touching[0].state
            entry[array] = reads(first, array) or not covers(first, array)
        writes = [k.index for k in kernels if k.writes(array)]
        if writes:
            last[array] = max(writes)
    return entry, last


def relayout_costs(
    sdfg: dace.SDFG, layouts: dict[str, list[Layout]], symbols: dict[str, int], p: LogGP
) -> dict[EdgeKey, float]:
    out = {}
    for array, options in layouts.items():
        seconds = float(symbolic.evaluate(relayout_time(sdfg.arrays[array], p), symbol_values(symbols)))
        out.update({(array, a.tag, b.tag): seconds for a, b in itertools.permutations(options, 2)})
    return out


def model_costs(
    sdfg: dace.SDFG,
    kernels: list[Kernel],
    layouts: dict[str, list[Layout]],
    symbols: dict[str, int],
    p: LogGP,
    cores: int | None = None,
) -> AssignmentCosts:
    """Costs from the LogGP model: each kernel cut out, the array laid out in place and the maps
    re-tiled, its traffic counted. ``cores`` bounds a parallel nest's concurrency; unbounded, the latency
    term vanishes, and with it the cost of a strided access on a GPU, so pass the GPU's compute units."""
    node_cost: dict[NodeKey, float] = {}
    for kernel in kernels:
        for array, options in layouts.items():
            for layout in options:
                if not kernel.touches(array):
                    node_cost[(array, kernel.index, layout.tag)] = 0.0
                    continue
                nest = extract_nest(kernel, f"model_k{kernel.index}_{array}_{layout.tag}")
                if not layout.is_identity:
                    nest.arrays[array].transient = True
                    lay_out(nest, array, layout)
                (state,) = nest.states()
                (entry,) = nest_entries(state)
                counts = count_nest(state, entry, p.line_bytes, p.sector_bytes)
                moved, messages = counts.bytes_moved(symbols, array), counts.messages(symbols, array)
                node_cost[(array, kernel.index, layout.tag)] = nest_memory_time(
                    p, moved, messages, exposed_concurrency(state, entry, p, cores)
                )
    entry, last = liveness(kernels, list(layouts))
    return AssignmentCosts(layouts, node_cost, relayout_costs(sdfg, layouts, symbols, p), entry, last)


def eval_costs(
    sdfg: dace.SDFG,
    kernels: list[Kernel],
    layouts: dict[str, list[Layout]],
    symbols: dict[str, int],
    p: LogGP,
    reps: int = 10,
    warmup: int = 2,
    seed: int = 0,
) -> AssignmentCosts:
    """Costs measured kernel by kernel (:func:`evaluate_nest`); a candidate that fails is an error, a
    contended one is marked untrusted. Relayouts are still modelled by ``p``."""
    node_cost: dict[NodeKey, float] = {}
    untrusted: set[NodeKey] = set()
    for kernel in kernels:
        touched = {array: options for array, options in layouts.items() if kernel.touches(array)}
        found = evaluate_nest(kernel, touched, symbols, reps=reps, warmup=warmup, seed=seed).by_name()
        failed = [r.name for r in found.values() if not r.correct or r.timing is None]
        if failed:
            raise RuntimeError(
                f"eval_costs: kernel {kernel.index}'s candidates {failed} failed: {found[failed[0]].error}"
            )
        for array, options in layouts.items():
            for layout in options:
                key = (array, kernel.index, layout.tag)
                timing = found[candidate_name(array, layout)].timing if array in touched else None
                node_cost[key] = 0.0 if timing is None else timing.median
                if timing is not None and timing.contended:
                    untrusted.add(key)
    entry, last = liveness(kernels, list(layouts))
    return AssignmentCosts(layouts, node_cost, relayout_costs(sdfg, layouts, symbols, p), entry, last, untrusted)


@dataclass(frozen=True, slots=True)
class AppliedAssignment:
    """Per array the buffer each segment works on, and the states converting between them."""

    holders: dict[str, list[str]]
    conversions: list[dace.SDFGState]


def segments(trajectory: list[Layout]) -> list[tuple[int, int, Layout]]:
    """The runs ``(start, end, layout)`` of one layout."""
    out = []
    start = 0
    for k in range(1, len(trajectory) + 1):
        if k == len(trajectory) or trajectory[k] != trajectory[start]:
            out.append((start, k, trajectory[start]))
            start = k
    return out


def apply_region_layout(
    sdfg: dace.SDFG, kernels: list[Kernel], layouts: dict[str, Layout], region: tuple[int, int]
) -> AppliedAssignment:
    """Lay arrays out for the kernels ``[start, end)`` only (whole loops), the identity elsewhere."""
    start, end = region
    if not 0 <= start < end <= len(kernels) or any(s < start < e or s < end < e for s, e in loop_spans(kernels)):
        raise ValueError(f"apply_region_layout: [{start}, {end}) is no run of whole loops of {len(kernels)} kernels")
    identity = Layout("identity")
    return apply_assignment(
        sdfg,
        kernels,
        {
            a: [identity] * start + [layout] * (end - start) + [identity] * (len(kernels) - end)
            for a, layout in layouts.items()
        },
    )


def apply_assignment(sdfg: dace.SDFG, kernels: list[Kernel], assignment: dict[str, list[Layout]]) -> AppliedAssignment:
    """Carry ``assignment`` (one layout per array per kernel) out on ``sdfg`` in place; see the module
    docstring. A loop body takes one layout; an array an interstate edge reads is refused."""
    for array, trajectory in assignment.items():
        if len(trajectory) != len(kernels):
            raise ValueError(f"apply_assignment: {array!r} has {len(trajectory)} layouts for {len(kernels)} kernels")
        if any(len({trajectory[k] for k in range(s, e)}) > 1 for s, e in loop_spans(kernels)):
            raise ValueError(f"apply_assignment: {array!r} changes layout inside a loop body (lock its transitions)")
        interstate = [e for e in sdfg.all_interstate_edges() if array in e.data.read_symbols()]
        if interstate and any(not layout.is_identity for layout in trajectory):
            raise ValueError(f"apply_assignment: an interstate edge reads {array!r}; lay it out with relayout_array")
    plans = [plan(sdfg, kernels, array, assignment[array]) for array in sorted(assignment)]
    for array_plan in plans:
        for holder, index_map, states in array_plan.rewrites:
            rewrite_accesses(sdfg, array_plan.array, holder, Laid(index_map), states)
    before: dict[int, list[Conversion]] = {}
    for array_plan in plans:
        for conversion in array_plan.conversions:
            before.setdefault(conversion.kernel, []).append(conversion)
    states = []
    for k in sorted(before):
        if k == len(kernels):
            last = kernels[-1].loop or kernels[-1].state
            state = last.parent_graph.add_state_after(last, "relayout_exit")
        else:
            first = kernels[k].loop or kernels[k].state
            graph = first.parent_graph
            state = graph.add_state_before(
                first, f"relayout_before_{first.label}", is_start_block=graph.start_block is first
            )
        for c in before[k]:
            transfer(
                state,
                f"{CONVERSION}{c.source}_{c.target}",
                c.source,
                Laid(c.source_map),
                c.target,
                Laid(c.target_map),
                c.shape,
            )
        states.append(state)
    propagation.propagate_memlets_sdfg(sdfg)
    canonicalize_schedule(sdfg)
    return AppliedAssignment({p.array: p.holders for p in plans}, states)


@dataclass(frozen=True, slots=True)
class Conversion:
    """Copy ``source`` (laid by ``source_map``) into ``target`` before kernel ``kernel`` (the end: after the last)."""

    kernel: int
    source: str
    source_map: IndexMap
    target: str
    target_map: IndexMap
    shape: list[str]


@dataclass(frozen=True, slots=True)
class Plan:
    array: str
    holders: list[str]
    rewrites: list[tuple[str, IndexMap, list[dace.SDFGState]]]
    conversions: list[Conversion]


def plan(sdfg: dace.SDFG, kernels: list[Kernel], array: str, trajectory: list[Layout]) -> Plan:
    """The buffers, rewrites and conversions of one array's trajectory, read before anything changes.
    Some buffers hold the current value: a segment whose layout one of them has works on it unconverted;
    another gets a buffer of its own layout, converted from the latest one unless its first kernel
    provably overwrites the array before reading it. A write leaves its buffer the only one current; at
    the end the program's buffer must be current."""
    desc = sdfg.arrays[array]
    shape = [str(extent) for extent in desc.shape]
    identity_map = IndexMap.identity(shape)
    current = {array: identity_map}
    holders, rewrites, conversions = [], [], []
    for start, end, layout in segments(trajectory):
        touching = [k for k in kernels[start:end] if k.touches(array)]
        if not touching:
            continue
        index_map = layout.index_map(array, shape)
        holder = next((h for h, m in current.items() if m == index_map), None)
        if holder is None:
            holder = array if layout.is_identity else f"{array}_{layout.tag}"
            if holder not in sdfg.arrays:
                sdfg.add_datadesc(holder, laid_out_descriptor(desc, index_map, transient=True))
            first = touching[0].state
            if reads(first, array) or not covers(first, array):
                source, source_map = list(current.items())[-1]
                conversions.append(Conversion(start, source, source_map, holder, index_map, shape))
        holders.append(holder)
        if holder != array:
            rewrites.append((holder, index_map, [k.state for k in touching]))
        if any(k.writes(array) for k in touching):
            current = {holder: index_map}
        else:
            current[holder] = index_map
    if array not in current and not desc.transient:
        source, source_map = list(current.items())[-1]
        conversions.append(Conversion(len(kernels), source, source_map, array, identity_map, shape))
    return Plan(array, holders, rewrites, conversions)
