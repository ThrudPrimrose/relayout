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
