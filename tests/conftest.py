# Copyright 2026 ETH Zurich and the relayout authors.
"""Shared test helpers."""

import importlib.util
from typing import Any

import pytest


def mpi_or_skip() -> Any:  # noqa: ANN401  # the mpi4py.MPI module
    """mpi4py's ``MPI``, or a skip when the ``mpi`` extra or an MPI library is missing."""
    if importlib.util.find_spec("mpi4py") is None:
        pytest.skip("no mpi4py (extra 'mpi')")
    try:
        from mpi4py import MPI  # noqa: PLC0415  # optional extra, and it loads the MPI library
    except RuntimeError as exc:
        pytest.skip(f"mpi4py finds no MPI library: {str(exc).splitlines()[0]}")
    return MPI


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
