# Copyright 2026 ETH Zurich and the relayout authors.
"""Shared test helpers."""

import ast
import importlib.util
import inspect
import textwrap
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import pytest

from relayout.ops import Block, LinearShuffle, Operation, Pad, Permute, Shuffle


def mpi_or_skip() -> Any:  # noqa: ANN401  # the mpi4py.MPI module
    """mpi4py's ``MPI``, or a skip when the ``mpi`` extra or an MPI library is missing."""
    if importlib.util.find_spec("mpi4py") is None:
        pytest.skip("no mpi4py (extra 'mpi')")
    try:
        from mpi4py import MPI  # noqa: PLC0415  # optional extra, and it loads the MPI library
    except RuntimeError as exc:
        pytest.skip(f"mpi4py finds no MPI library: {str(exc).splitlines()[0]}")
    return MPI


@dataclass(frozen=True, slots=True)
class Rewritten:
    """A Python kernel after a pass: the compiled function and its source."""

    function: Callable[..., Any]
    source: str


def rewritten(function: Callable[..., Any], transform: Callable[[ast.Module], object]) -> Rewritten:
    """``function`` parsed from its source, ``transform`` applied to the module, compiled in ``function``'s module
    namespace (the rewritten function sees its module, as the original does)."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(function)))
    transform(tree)
    source = ast.unparse(ast.fix_missing_locations(tree))
    namespace = dict(function.__globals__)
    exec(compile(source, f"<rewritten {function.__name__}>", "exec"), namespace)  # noqa: S102  # the test's own kernel
    return Rewritten(namespace[function.__name__], source)


def flang_missing() -> bool:
    """The Fortran frontend needs the optional ``fortran`` extra and an LLVM flang."""
    if importlib.util.find_spec("dace_fortran") is None:
        return True
    from dace_fortran.llvm_toolchain import find_flang  # noqa: PLC0415  # optional extra

    return find_flang() is None


def variant(sdfg: object, suffix: str) -> None:
    """Name ``sdfg`` and every SDFG nested in it after one variant: dace keeps one library per build path
    in a process, so a second variant of a program under the same name would run the first's library."""
    for inner in sdfg.all_sdfgs_recursive():  # type: ignore[attr-defined]
        inner.name = f"{inner.name}_{suffix}"


#: Every single-node digit operation on a 2-D array ``A`` (rows ``i0``, columns ``i1``; ``split`` only lays bands out
#: for a distribution, tests/test_ops.py), alone and composed: the cases each
#: layout check (index map, conversion kernels, MPI transfers) runs over. Valid at any extents, so the MPI tests
#: use them on ``[P, N]``; the fixed-shape tests use :data:`LAYOUT_SHAPE`, where ``block_ragged`` leaves dead cells.
LAYOUT_CASES: dict[str, list[Operation]] = {
    "permute": [Permute("A", ("i1", "i0"))],
    "block": [Block("A", "i1", "2", ("o", "f"))],
    "block_ragged": [Block("A", "i1", "4", ("o", "f"))],
    "pad": [Pad("A", "i1", "3")],
    "shuffle": [Block("A", "i1", "2", ("o", "f")), Shuffle("A", ("f",), ("1 - f",), ("1 - f",))],
    "linear_shuffle": [
        Block("A", "i1", "2", ("o", "c")),
        Block("A", "i0", "2", ("r", "w")),
        LinearShuffle("A", ("w", "c"), (1, 1), ((1, 0), (1, 1))),
    ],
    "composed": [Block("A", "i1", "2", ("o", "f")), Pad("A", "i0", "1"), Permute("A", ("o", "i0", "f"))],
}

#: The logical shape the fixed-shape layout checks use.
LAYOUT_SHAPE = (4, 6)
