# Copyright 2026 ETH Zurich and the relayout authors.
"""``zip``: arrays of one shape fused into one new transient, stacked along a new dimension (one element
type) or as an array of structs with one member per array (different element types).
"""

import copy
from collections.abc import Sequence

import dace
from dace import nodes
from dace.properties import CodeBlock
from dace.sdfg import propagation
from dace.sdfg.state import SDFGState

from relayout.passes.sdfg.common import (
    CONVERSION,
    LayoutError,
    array_desc,
    copy_through_tasklets,
    fully_written,
    is_read,
    is_written,
    packed_strides,
    product,
    remove_path_edge,
)
from relayout.passes.sdfg.placement import Field
from relayout.passes.sdfg.rewrite_accesses import (
    move_array,
)

__all__ = ["zip_arrays", "zip_struct"]


def zip_arrays(sdfg: dace.SDFG, fields: Sequence[str], into: str, dim: int | None = None) -> str:
    """Fuse ``sdfg``'s arrays ``fields`` (one shape, one element type) into the new transient ``into``,
    stacked along a new dimension at ``dim`` (None: last; ``into[..., k]`` is ``fields[k]``), the inverse of
    :func:`unzip_array`. Arguments keep their descriptors and are converted in and out (:func:`move_array`)."""
    descs = [array_desc(sdfg, field) for field in fields]
    shapes = {tuple(str(extent) for extent in desc.shape) for desc in descs}
    if len(shapes) != 1:
        raise LayoutError(f"zip of {list(fields)}: the arrays must share one shape")
    if len({desc.dtype for desc in descs}) != 1:
        if dim is not None:
            raise LayoutError(
                f"zip of {list(fields)}: arrays of different element types make a struct, which has no dim"
            )
        return zip_struct(sdfg, fields, into)
    shape = list(descs[0].shape)
    shape.insert(len(shape) if dim is None else dim, len(fields))
    zipped = copy.deepcopy(descs[0])
    zipped.transient = True
    zipped.set_shape(shape, strides=packed_strides(shape), total_size=product(shape))
    sdfg.add_datadesc(into, zipped, find_new_name=False)
    for k, field in enumerate(fields):
        move_array(sdfg, field, into, Field(k, dim))
    propagation.propagate_memlets_sdfg(sdfg)
    return into


def zip_struct(sdfg: dace.SDFG, fields: Sequence[str], into: str) -> str:
    """Fuse ``sdfg``'s arrays ``fields`` (one shape, different element types) into the new transient ``into``,
    an array of the struct ``relayout_zip_<into>`` with one member per field, named after it: a tasklet reads
    its member of the element, and writes it by reading the element and writing it back (so two members
    written in one state, which would race on the element, are refused, as is a reduction into one).
    Arguments keep their descriptors and are converted in and out member by member, one state each."""
    descs = {field: array_desc(sdfg, field) for field in fields}
    struct = dace.struct(f"relayout_zip_{into}", **{field: desc.dtype for field, desc in descs.items()})
    zipped = copy.deepcopy(descs[fields[0]])
    zipped.dtype, zipped.transient = struct, True
    shape = list(zipped.shape)
    zipped.set_shape(shape, strides=packed_strides(shape), total_size=product(shape))
    sdfg.add_datadesc(into, zipped, find_new_name=False)
    for edge in sdfg.all_interstate_edges():
        named = sorted(set(fields) & {str(s) for s in edge.data.read_symbols()})
        if named:
            raise LayoutError(f"zip of {list(fields)}: an interstate edge reads {named}; it would read a struct member")
    for state in sdfg.states():
        written = sorted({n.data for n in state.data_nodes() if n.data in descs and state.in_degree(n) > 0})
        if len(written) > 1:
            raise LayoutError(f"{state.label}: writes {written}, members of one struct element, in one state")
    io = {field: (is_read(sdfg, field), is_written(sdfg, field), fully_written(sdfg, field)) for field in fields}
    for state in sdfg.states():
        for field in fields:
            copy_through_tasklets(state, field, lambda memlet: True)
            for node in [n for n in state.data_nodes() if n.data == field]:
                member_access(state, node, into, field)
    shape_text = [str(extent) for extent in shape]
    for field, (read, written, whole) in io.items():
        if descs[field].transient:
            sdfg.remove_data(field, validate=False)
            continue
        if read or (written and not whole):
            member_transfer(sdfg.add_state_before(sdfg.start_block, f"zip_in_{field}"), field, into, shape_text, True)
        if written:
            sink = next(b for b in sdfg.nodes() if sdfg.out_degree(b) == 0)
            member_transfer(sdfg.add_state_after(sink, f"zip_out_{field}"), field, into, shape_text, False)
    propagation.propagate_memlets_sdfg(sdfg)
    return into


def member_access(state: SDFGState, node: nodes.AccessNode, into: str, member: str) -> None:
    """Every point access of a tasklet (Python or C++) through ``node`` moved to member ``member`` of the
    struct array ``into``: a read takes the element and picks the member, a write reads the element, sets the
    member and writes it back."""
    for edge in list(state.out_edges(node)) + list(state.in_edges(node)):
        if edge not in state.edges():
            continue
        tree = state.memlet_tree(edge)
        for leaf in list(tree.leaves()):
            reads = state.memlet_path(leaf)[0].src is node
            tasklet = leaf.dst if reads else leaf.src
            conn = leaf.dst_conn if reads else leaf.src_conn
            if not isinstance(tasklet, nodes.Tasklet) or conn is None:
                raise LayoutError(f"{state.label}: {leaf.data} reaches a struct member outside a tasklet")
            if leaf.data.wcr is not None:
                raise LayoutError(f"{state.label}: a reduction into {node.data!r} would race on its struct element")
            path = state.memlet_path(leaf)
            element = dace.Memlet(data=into, subset=copy.deepcopy(leaf.data.subset), dynamic=leaf.data.dynamic)
            code = tasklet.code.as_string
            cpp = tasklet.language is dace.Language.CPP
            ctype = state.sdfg.arrays[into].dtype.fields[member].ctype
            if reads:
                tasklet.remove_in_connector(conn)
                tasklet.add_in_connector(f"{conn}__z")
                state.add_memlet_path(
                    state.add_read(into), *[e.dst for e in path[:-1]], tasklet, dst_conn=f"{conn}__z", memlet=element
                )
                prefix = f"const {ctype} {conn} = {conn}__z.{member};" if cpp else f"{conn} = {conn}__z.{member}"
                tasklet.code = CodeBlock(f"{prefix}\n{code}", tasklet.language)
            else:
                exits = [e.src for e in path[1:]]
                tasklet.remove_out_connector(conn)
                tasklet.add_out_connector(f"{conn}__z")
                tasklet.add_in_connector(f"{conn}__zin")
                state.add_memlet_path(tasklet, *exits, state.add_write(into), src_conn=f"{conn}__z", memlet=element)
                entries = [state.entry_node(exit_) for exit_ in reversed(exits)]
                state.add_memlet_path(
                    state.add_read(into), *entries, tasklet, dst_conn=f"{conn}__zin", memlet=copy.deepcopy(element)
                )
                if cpp:
                    store = f"{conn}__z = {conn}__zin;\n{conn}__z.{member} = {conn};"
                    tasklet.code = CodeBlock(f"{ctype} {conn};\n{code}\n{store}", tasklet.language)
                else:
                    store = f"{conn}__z = {conn}__zin\n{conn}__z.{member} = {conn}"
                    tasklet.code = CodeBlock(f"{code}\n{store}", tasklet.language)
        for old in list(tree):
            remove_path_edge(state, old)
    state.remove_node(node)


def member_transfer(state: SDFGState, field: str, into: str, shape: Sequence[str], inward: bool) -> None:
    """A map over ``shape`` copying argument ``field`` into its member of ``into`` (or back)."""
    params = [f"__r{d}" for d in range(len(shape))]
    ranges = {p: f"0:{extent}" for p, extent in zip(params, shape, strict=True)}
    at = ", ".join(params)
    if inward:
        state.add_mapped_tasklet(
            f"{CONVERSION}zip_in_{field}",
            ranges,
            {"__inp": dace.Memlet(f"{field}[{at}]"), "__zin": dace.Memlet(f"{into}[{at}]")},
            f"__z = __zin\n__z.{field} = __inp",
            {"__z": dace.Memlet(f"{into}[{at}]")},
            external_edges=True,
        )
    else:
        state.add_mapped_tasklet(
            f"{CONVERSION}zip_out_{field}",
            ranges,
            {"__z": dace.Memlet(f"{into}[{at}]")},
            f"__out = __z.{field}",
            {"__out": dace.Memlet(f"{field}[{at}]")},
            external_edges=True,
        )
