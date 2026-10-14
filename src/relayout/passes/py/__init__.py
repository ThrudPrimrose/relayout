# Copyright 2026 ETH Zurich and the relayout authors.
"""Python passes: a numpy / mpi4py function rewritten to run on laid-out arrays, one module per pass, each the
counterpart of the SDFG pass of the same name (:mod:`relayout.passes.sdfg`).

- :mod:`.prepare`: the function in the form the passes read (hand-tiled loops over blocked arrays untiled);
- :mod:`.relayout_array`: one array laid out by an index map (permute, block, split, pad, shuffle);
- :mod:`.zip_arrays` / :mod:`.unzip_array`: arrays regrouped along a dimension or into a structured array;
- :mod:`.cast_array`: an array in another element type;
- :mod:`.pack_mpi_buffers`: mpi4py buffers of laid-out arrays packed in logical order;
- :mod:`.unblock_dimensions` / :mod:`.untile_blocked_loops`: hand-blocked arrays and tiled loops flattened;
- :mod:`.canonicalize_schedule`: loops re-tiled to the layout's blocks and ordered by stride;
- :mod:`.apply_requests`: a request's operations, in order.

Shared by them: :mod:`.common`, :mod:`.placement`, :mod:`.rewrite_accesses`. As on an SDFG, an argument keeps
its array for the caller (converted in at the entry and out at the exit) and a local is rewritten in place.
"""
