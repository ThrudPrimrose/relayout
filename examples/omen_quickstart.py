# Copyright 2026 ETH Zurich and the relayout authors.
"""OMEN's scattering self-energy (SC19): one request lays the kernel's arrays out on a node, and one
distributes them over ranks the way SC19 does.

Run: ``python examples/omen_quickstart.py``.
"""

import numpy as np

from relayout.frontends.sdfg import apply_requests
from relayout.corpus import omen_sse
from relayout.corpus.omen_sse.sse import sse
from relayout.ops import ArraySpec, lower
from relayout.requests import parse_requests

ARRAYS = ("neigh_idx", "dH", "G", "D", "Sigma")
SIZES = omen_sse.SIZES["S"]

# 1. On one node: G atom-major (the neighbour gather walks atoms, the energy loop reuses them), and dH
#    with its 3-D direction innermost. The request is the JSON an agent or a driver sends.
local = parse_requests(
    {
        "operations": [
            {"action": "permute", "array": "G", "order": ["i2", "i0", "i1", "i3", "i4"]},
            {"action": "permute", "array": "dH", "order": ["i0", "i1", "i3", "i4", "i2"]},
        ]
    }
)
sdfg = sse.to_sdfg(simplify=True)
print("laid out:", apply_requests(sdfg, local))
args = dict(zip(ARRAYS, omen_sse.initialize(**SIZES), strict=True))
expected = args["Sigma"].copy()
omen_sse.reference(args["neigh_idx"], args["dH"], args["G"], args["D"], expected)
sdfg(**args, **SIZES)
assert np.allclose(args["Sigma"], expected), "the laid-out kernel computes the reference"

# 2. Over ranks, as SC19: G arrives energy-distributed, Sigma leaves momentum-distributed, the rest is
#    replicated. The grid is an expression over the rank count P, evaluated at each run.
distributed = parse_requests(
    {
        "startup": {"grid": ["P"]},
        "operations": [
            {"action": "distribute", "array": "G", "grid_dim": 0, "dim": 1, "scheme": "block"},
            {"action": "distribute", "array": "Sigma", "grid_dim": 0, "dim": 0, "scheme": "block"},
            *({"action": "replicate", "array": name} for name in ("neigh_idx", "dH", "D")),
        ],
    }
)
specs = [ArraySpec(name, len(args[name].shape)) for name in ARRAYS]
for name, axes in lower(distributed.run(specs, ranks=4)).items():
    print(f"{name:9} {'replicated' if axes is None else [None if a is None else a.scheme.value for a in axes]}")
