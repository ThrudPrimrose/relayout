# Copyright 2026 ETH Zurich and the relayout authors.
"""Python passes: a numpy / mpi4py program rewritten to run on laid-out arrays, one module per pass.

- :mod:`.rewrite_accesses`: element accesses, logical views and packed MPI buffers of laid-out arrays;
- :mod:`.regroup`: ``zip`` / ``unzip``;
- :mod:`.rewrite`: both, in order.
"""
