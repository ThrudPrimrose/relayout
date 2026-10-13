# Copyright 2026 ETH Zurich and the relayout authors.
"""A program as the layout search sees it: a line of kernels, one map nest per state.

:func:`kernel_per_state` splits states until each holds one top-level map nest; :func:`line_graph`
reads the kernels in program order (a loop whose body is a line contributes its body's kernels, tagged
with the loop, which keeps one layout: :func:`locked_transitions`). :func:`extract_nest` cuts one kernel
out into a runnable SDFG and :func:`nest_arguments` fills its arguments, to measure or count it alone.
"""

import ast
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass

import dace
import numpy
import sympy
from dace import nodes, symbolic
from dace.sdfg.analysis.cutout import SDFGCutout
from dace.sdfg.state import ControlFlowRegion, LoopRegion, SDFGState
from dace.sdfg.utils import dfs_topological_sort
from dace.transformation.helpers import state_fission

__all__ = [
    "IndirectAccess",
    "Kernel",
    "extract_nest",
    "indirect_accesses",
    "kernel_per_state",
    "line_graph",
    "locked_transitions",
    "loop_spans",
    "nest_arguments",
    "nest_entries",
    "written_arrays",
]


class ProgramShapeError(NotImplementedError):
    """A program that is no line of map nests."""


def nest_entries(state: SDFGState) -> list[nodes.MapEntry]:
    """The top-level map entries of ``state``."""
    return [n for n in state.scope_children()[None] if isinstance(n, nodes.MapEntry)]


def kernel_per_state(sdfg: dace.SDFG) -> int:
    """Split every state holding several top-level map nests into one state per nest; returns the
    number of splits."""
    splits = 0
    while True:
        crowded = [s for s in sdfg.states() if len(nest_entries(s)) > 1]
        if not crowded:
            return splits
        excess = sum(len(nest_entries(s)) - 1 for s in crowded)
        for state in crowded:
            entries = set(nest_entries(state))
            first = next(n for n in dfs_topological_sort(state) if n in entries)
            state_fission(state.scope_subgraph(first), label=f"{state.label}_k{splits}", allow_isolated_nodes=False)
            splits += 1
        if sum(len(nest_entries(s)) - 1 for s in sdfg.states() if len(nest_entries(s)) > 1) >= excess:
            raise ProgramShapeError(
                "kernel_per_state: a state's nests share an access node or a view, so they cannot be split"
            )


@dataclass(frozen=True, slots=True)
class Kernel:
    """One map nest of the line, at position ``index``, in the body of ``loop`` (or at top level)."""

    state: SDFGState
    entry: nodes.MapEntry
    index: int
    loop: LoopRegion | None

    def touches(self, name: str) -> bool:
        return any(n.data == name for n in self.state.data_nodes())

    def writes(self, name: str) -> bool:
        return any(n.data == name and self.state.in_degree(n) > 0 for n in self.state.data_nodes())


def does_work(state: SDFGState) -> bool:
    """Whether ``state`` moves data: a node beyond access nodes and empty tasklets, or a copy."""
    for node in state.nodes():
        if not isinstance(node, nodes.AccessNode) and not (
            isinstance(node, nodes.Tasklet) and not node.in_connectors and not node.out_connectors
        ):
            return True
    return any(not e.data.is_empty() for e in state.edges())


def collect(region: ControlFlowRegion, kernels: list[Kernel], loop: LoopRegion | None) -> None:
    """Append the kernels of ``region``, a line of states and loops whose bodies are lines."""
    block = region.start_block if region.number_of_nodes() else None
    seen = set()
    while block is not None:
        if block in seen or region.out_degree(block) > 1 or region.in_degree(block) > 1:
            raise ProgramShapeError(f"line_graph: {block.label!r} branches, joins or cycles; the line holds no DAG")
        seen.add(block)
        if isinstance(block, SDFGState):
            entries = nest_entries(block)
            if len(entries) > 1:
                raise ProgramShapeError(f"line_graph: {block.label!r} holds {len(entries)} nests; kernel_per_state")
            if entries:
                kernels.append(Kernel(block, entries[0], len(kernels), loop))
            elif does_work(block):
                raise ProgramShapeError(f"line_graph: {block.label!r} does work outside a map nest")
        elif isinstance(block, LoopRegion) and loop is None:
            collect(block, kernels, block)
        else:
            raise ProgramShapeError(f"line_graph: {block.label!r} is a {type(block).__name__} (or a nested loop)")
        successors = region.out_edges(block)
        block = successors[0].dst if successors else None
    if len(seen) != region.number_of_nodes():
        raise ProgramShapeError(f"line_graph: {region.label!r} has blocks off its line")


def line_graph(sdfg: dace.SDFG) -> list[Kernel]:
    """The kernels of ``sdfg`` in program order (one nest per state: :func:`kernel_per_state`)."""
    kernels: list[Kernel] = []
    collect(sdfg, kernels, None)
    return kernels


def loop_spans(kernels: list[Kernel]) -> list[tuple[int, int]]:
    """The ``[start, end)`` runs of kernels in one loop body."""
    spans = []
    start = 0
    while start < len(kernels):
        end = start + 1
        while end < len(kernels) and kernels[end].loop is kernels[start].loop:
            end += 1
        if kernels[start].loop is not None:
            spans.append((start, end))
        start = end
    return spans


def locked_transitions(kernels: list[Kernel]) -> set[int]:
    """The kernels ``k`` that keep the layout of ``k - 1``: those inside a loop body after its first, so
    a loop runs in one layout and its relayouts land outside it."""
    return {k for start, end in loop_spans(kernels) for k in range(start + 1, end)}


def extract_nest(kernel: Kernel, name: str) -> dace.SDFG:
    """The kernel cut out into a runnable SDFG named ``name``: the arrays crossing its boundary are its
    arguments."""
    scope = kernel.state.scope_subgraph(kernel.entry).nodes()
    cut = SDFGCutout.singlestate_cutout(
        kernel.state, *scope, make_copy=True, make_side_effects_global=True, use_alibi_nodes=False
    )
    rename(cut, re.sub(r"\W", "_", name))
    cut.validate()
    return cut


def rename(sdfg: dace.SDFG, name: str) -> None:
    sdfg.name = name  # pyright: ignore[reportAttributeAccessIssue]  # dace declares SDFG.name a Property


def symbol_values(symbols: Mapping[str, int]) -> dict[symbolic.symbol | str, int | float]:
    """``symbols`` as :func:`dace.symbolic.evaluate` takes them."""
    out: dict[symbolic.symbol | str, int | float] = {k: v for k, v in symbols.items()}
    return out


def written_arrays(sdfg: dace.SDFG) -> set[str]:
    """The arguments ``sdfg`` writes."""
    return {
        n.data
        for s in sdfg.states()
        for n in s.data_nodes()
        if s.in_degree(n) > 0 and not sdfg.arrays[n.data].transient
    }


def evaluate(value: object, symbols: dict[str, int]) -> int | None:
    try:
        return int(symbolic.evaluate(sympy.sympify(value), symbol_values(symbols)))
    except (TypeError, ValueError, KeyError):
        return None


def index_bound(sdfg: dace.SDFG, symbols: dict[str, int]) -> int:
    """The smallest extent of an argument array: an integer argument may index any of them, so its fill
    stays below (8 at most)."""
    extents = [8]
    for desc in sdfg.arrays.values():
        values = [evaluate(extent, symbols) for extent in desc.shape]
        shape = [v for v in values if v is not None]
        if not desc.transient and len(shape) == len(values) and math.prod(shape) > 1:
            extents += shape
    return max(1, min(extents))


def nest_arguments(
    sdfg: dace.SDFG, symbols: dict[str, int], provided: dict[str, numpy.ndarray] | None = None, seed: int = 0
) -> dict[str, numpy.ndarray]:
    """Arguments for ``sdfg``: copies of ``provided``, the rest filled from ``seed`` (in name order)."""
    provided = provided or {}
    rng = numpy.random.default_rng(seed)
    high = index_bound(sdfg, symbols)
    out = {}
    for name in sorted(sdfg.arrays):
        desc = sdfg.arrays[name]
        if desc.transient:
            continue
        if name in provided:
            out[name] = provided[name].copy()
            continue
        shape = tuple(int(symbolic.evaluate(sympy.sympify(s), symbol_values(symbols))) for s in desc.shape)
        dtype = desc.dtype.as_numpy_dtype()
        if numpy.issubdtype(dtype, numpy.complexfloating):
            out[name] = (rng.random(shape) + 1j * rng.random(shape)).astype(dtype)
        elif numpy.issubdtype(dtype, numpy.floating):
            out[name] = rng.random(shape).astype(dtype)
        elif numpy.issubdtype(dtype, numpy.integer):
            out[name] = rng.integers(0, high, size=shape, dtype=dtype)
        else:
            raise NotImplementedError(f"nest_arguments: no fill for {name!r} of type {dtype}")
    return out


@dataclass(frozen=True, slots=True)
class IndirectAccess:
    """``data`` subscripted by a value read from ``index`` (``data[index[f(i)]]``), read or written."""

    index: str
    data: str
    written: bool


def index_source(text: str, loop_variable: str, sdfg: dace.SDFG) -> str | None:
    """``a`` when ``text`` is ``a[f(loop_variable)]`` for an array ``a`` of ``sdfg``."""
    try:
        tree = ast.parse(text, mode="eval").body
    except SyntaxError:
        return None
    if not (isinstance(tree, ast.Subscript) and isinstance(tree.value, ast.Name) and tree.value.id in sdfg.arrays):
        return None
    names = {n.id for n in ast.walk(tree.slice) if isinstance(n, ast.Name)}
    return tree.value.id if loop_variable in names else None


def indirect_accesses(sdfg: dace.SDFG) -> list[IndirectAccess]:
    """Each indirect access of a loop in ``sdfg`` (a symbol assigned ``index[f(i)]`` on the loop's edges
    subscripting ``data``), once, in program order."""
    found: dict[IndirectAccess, None] = {}
    for inner in sdfg.all_sdfgs_recursive():
        for region in inner.all_control_flow_regions():
            if not isinstance(region, LoopRegion) or not region.loop_variable:
                continue
            bound = {
                symbol: source
                for e in region.edges()
                for symbol, value in e.data.assignments.items()
                if (source := index_source(str(value), region.loop_variable, inner)) is not None
            }
            for state in region.states():
                for node in state.data_nodes():
                    edges = [(e, True) for e in state.in_edges(node)] + [(e, False) for e in state.out_edges(node)]
                    for edge, written in edges:
                        for symbol in edge.data.subset.free_symbols if edge.data.subset is not None else ():
                            index = bound.get(str(symbol))
                            if index is not None and index != node.data:
                                found[IndirectAccess(index, node.data, written)] = None
    return list(found)
