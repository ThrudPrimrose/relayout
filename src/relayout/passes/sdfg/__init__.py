# Copyright 2026 ETH Zurich and the relayout authors.
"""SDFG passes: an SDFG rewritten to run on laid-out arrays, one module per pass.

- :mod:`.prepare`: the SDFG in the form the passes read (untiled, canonical, packed arrays only);
- :mod:`.relayout_array`: one array laid out by an index map (permute, block, split, pad, shuffle);
- :mod:`.zip_arrays` / :mod:`.unzip_array`: arrays regrouped along a dimension or into structs;
- :mod:`.cast_array`: an array in another element type (FP-Arena);
- :mod:`.pack_mpi_buffers`: MPI buffers of laid-out arrays packed where their cells are no contiguous run;
- :mod:`.unblock_dimensions` / :mod:`.untile_blocked_loops`: hand-blocked arrays and tiled loops flattened;
- :mod:`.canonicalize_schedule`: maps re-tiled to the layout's blocks and ordered by stride;
- :mod:`.apply_requests`: a request's operations, in order.

Shared by them: :mod:`.common`, :mod:`.placement`, :mod:`.rewrite_accesses`, :mod:`.library_rules`.
"""
