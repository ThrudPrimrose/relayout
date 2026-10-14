# Copyright 2026 ETH Zurich and the relayout authors.
"""What a program rewrite carries out of a request, shared by the SDFG and the Python ``apply_requests``: the
``cast``, ``zip`` / ``unzip`` and digit operations of one node. The rest belongs to whoever runs the program (a
harness loads the libraries, scatters a distributed array to its ranks, replicates, places and formats), so a
request naming it is refused by name rather than half applied.
"""

from relayout.ops import Distribute, Format, Place, Replicate, Requests

__all__ = ["CALLER_OPERATIONS", "unapplied"]

#: The operations a program rewrite leaves to the caller.
CALLER_OPERATIONS = (Format, Distribute, Replicate, Place)


def unapplied(requests: Requests) -> str | None:
    """Why a program rewrite cannot carry ``requests`` out on its own, or None when it can."""
    reasons = []
    if requests.distributed:
        reasons.append("a startup grid (a distributed run)")
    if requests.startup.load:
        reasons.append(f"startup.load {list(requests.startup.load)}")
    named = sorted({op.action.value for op in requests.operations if isinstance(op, CALLER_OPERATIONS)})
    if named:
        reasons.append(f"the operations {named}")
    if not reasons:
        return None
    return (
        f"apply_requests rewrites one node's program; {' and '.join(reasons)} belong to the caller that runs it "
        "(lower a distributed request with relayout.ops.lower, format sparse arrays with relayout.sparse)"
    )
