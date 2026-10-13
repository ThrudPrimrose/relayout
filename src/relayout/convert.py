# Copyright 2026 ETH Zurich and the relayout authors.
"""Conversion kernels: one array moved between its logical buffer (row-major, at one element type) and
its laid-out buffer (an :class:`~relayout.layout.IndexMap`, at another element type) in one pass.

Every conversion is a gather over the laid-out cells: ``stored[p] = cast(logical[logical_index(p)])``
in, ``logical[logical_index(p)] = cast(stored[p])`` out, a dead cell (pad, a ragged block) written as
zero in and skipped out. A precision cast is the identity layout, a layout change the identity cast,
and both together are still one pass over memory. :func:`host_source` emits it as C with OpenMP,
:func:`device_source` as a CUDA / HIP kernel for NVRTC or hipRTC (``cupy.RawModule``); the element
conversions are ``runtime/relayout_fp.h``. A kernel takes the symbols of the logical shape as
``int64_t`` arguments after the two buffers, in :meth:`Conversion.symbols` order.
"""

import ast
import pathlib
import subprocess
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from relayout.expr import eval_index, evaluate, is_root, node_names
from relayout.layout import IndexMap, linear

__all__ = [
    "ELEMENTS",
    "Conversion",
    "Element",
    "Gather",
    "Group",
    "Member",
    "c_expr",
    "compile_host",
    "device_source",
    "fp_header",
    "from_layout",
    "host_source",
    "to_layout",
]


@dataclass(frozen=True, slots=True)
class Element:
    """An element type by its numpy name: its C storage type, and for a format stored as bits its
    ``(exponent bits, mantissa bits, finite-only)``; ``component`` names a complex type's part.
    ``via_float`` rounds a double to float before the format, as ml_dtypes converts (numpy's float16
    rounds a double once)."""

    name: str
    c_type: str
    bits: tuple[int, int, bool] | None = None
    component: str | None = None
    via_float: bool = False


ELEMENTS = {
    element.name: element
    for element in (
        Element("float64", "double"),
        Element("float32", "float"),
        Element("float16", "uint16_t", (5, 10, False)),
        Element("bfloat16", "uint16_t", (8, 7, False), via_float=True),
        Element("float8_e4m3fn", "uint8_t", (4, 3, True), via_float=True),
        Element("float8_e5m2", "uint8_t", (5, 2, False), via_float=True),
        Element("complex128", "relayout_c128", component="float64"),
        Element("complex64", "relayout_c64", component="float32"),
        Element("int64", "int64_t"),
        Element("int32", "int32_t"),
        Element("int16", "int16_t"),
        Element("int8", "int8_t"),
        Element("uint8", "uint8_t"),
        Element("bool", "uint8_t"),
    )
}


def fp_header() -> str:
    return (pathlib.Path(__file__).parent / "runtime" / "relayout_fp.h").read_text(encoding="utf-8")


def as_double(element: Element, value: str) -> str:
    if element.bits is None:
        return f"(double)({value})"
    ebits, mbits, finite = element.bits
    return f"relayout_widen({value}, {ebits}, {mbits}, {int(finite)})"


def cast(source: Element, target: Element, value: str) -> str:
    """The C expression converting ``value`` (a ``source`` element) to a ``target`` element."""
    if source == target:
        return value
    if (source.component is None) != (target.component is None):
        raise ValueError(f"no cast between {source.name} and {target.name}: one is complex")
    if source.component is not None and target.component is not None:
        source_part, target_part = ELEMENTS[source.component], ELEMENTS[target.component]
        parts = (cast(source_part, target_part, f"({value}).{part}") for part in ("re", "im"))
        return f"relayout_make_{target.c_type.removeprefix('relayout_')}({', '.join(parts)})"
    if target.bits is not None:
        ebits, mbits, finite = target.bits
        wide = as_double(source, value)
        if target.via_float and source.name == "float64":
            wide = f"(double)(float)({wide})"
        return f"({target.c_type})relayout_narrow({wide}, {ebits}, {mbits}, {int(finite)})"
    if source.bits is not None:
        return f"({target.c_type})({as_double(source, value)})"
    return f"({target.c_type})({value})"


C_BINOPS = {
    ast.Add: "+",
    ast.Sub: "-",
    ast.Mult: "*",
    ast.FloorDiv: "/",
    ast.Mod: "%",
    ast.BitXor: "^",
    ast.BitAnd: "&",
    ast.BitOr: "|",
    ast.LShift: "<<",
    ast.RShift: ">>",
}


def may_be_negative(node: ast.expr) -> bool:
    """Whether an index expression over non-negative names can be negative: only a subtraction or a
    negation makes one."""
    return any(isinstance(walked, (ast.Sub, ast.USub)) for walked in ast.walk(node))


def c_expr(text: str) -> str:
    """An index expression over non-negative names as C, fully parenthesized, with Python's floor ``//`` and
    ``%`` (``relayout_floordiv`` / ``relayout_floormod`` where an operand can be negative, C's otherwise)."""

    def emit(node: ast.expr) -> str:
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            return f"{node.value}LL"
        if isinstance(node, ast.Name):
            return node.id
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return f"(-{emit(node.operand)})"
        if isinstance(node, ast.BinOp) and isinstance(node.op, ast.FloorDiv | ast.Mod) and may_be_negative(node):
            helper = "relayout_floordiv" if isinstance(node.op, ast.FloorDiv) else "relayout_floormod"
            return f"{helper}({emit(node.left)}, {emit(node.right)})"
        if isinstance(node, ast.BinOp) and type(node.op) in C_BINOPS:
            return f"({emit(node.left)} {C_BINOPS[type(node.op)]} {emit(node.right)})"
        if isinstance(node, ast.Compare) and len(node.ops) == 1 and isinstance(node.ops[0], ast.Lt):
            return f"({emit(node.left)} < {emit(node.comparators[0])})"
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in ("min", "max"):
            a, b = (emit(arg) for arg in node.args)
            return f"(({a}) {'<' if node.func.id == 'min' else '>'} ({b}) ? ({a}) : ({b}))"
        if is_root(node) and not node_names(node):
            return f"{eval_index(ast.unparse(node), {}, text)}LL"
        raise ValueError(f"{text!r}: {ast.unparse(node)} has no C form here")

    return emit(ast.parse(text, mode="eval").body)


@dataclass(frozen=True, slots=True)
class Conversion:
    """Array ``name`` between its logical buffer of ``logical`` elements and its laid-out buffer of
    ``stored`` elements in ``layout``."""

    name: str
    layout: IndexMap
    logical: str
    stored: str

    def symbols(self) -> tuple[str, ...]:
        """The names the logical shape reads, the kernels' ``int64_t`` arguments in this order."""
        names: set[str] = set()
        for extent in self.layout.shape:
            names |= {node.id for node in ast.walk(ast.parse(extent, mode="eval")) if isinstance(node, ast.Name)}
        return tuple(sorted(names))

    def body(self, positions: Sequence[str], inward: bool) -> str:
        """The statement converting the cell at physical ``positions`` (C variables) one way."""
        layout = self.layout
        stored_at = c_expr(linear(positions, layout.extents()))
        logical_at = c_expr(linear(layout.logical(positions), layout.shape))
        live = " && ".join(c_expr(condition) for condition in layout.live(positions))
        source, target = ELEMENTS[self.logical], ELEMENTS[self.stored]
        if inward:
            value = cast(source, target, f"src[{logical_at}]")
            if not live:
                return f"dst[{stored_at}] = {value};"
            return f"if ({live}) dst[{stored_at}] = {value}; else dst[{stored_at}] = zero;"
        value = cast(target, source, f"src[{stored_at}]")
        return f"dst[{logical_at}] = {value};" if not live else f"if ({live}) dst[{logical_at}] = {value};"

    def types(self, inward: bool) -> tuple[str, str]:
        logical, stored = ELEMENTS[self.logical].c_type, ELEMENTS[self.stored].c_type
        return (logical, stored) if inward else (stored, logical)

    def signature(self, inward: bool, qualifier: str) -> str:
        symbols = "".join(f", int64_t {symbol}" for symbol in self.symbols())
        src, dst = self.types(inward)
        direction = "in" if inward else "out"
        return f"void relayout_{direction}_{self.name}(const {src} *{qualifier} src, {dst} *{qualifier} dst{symbols})"


def host_function(conversion: Conversion, inward: bool) -> str:
    extents = [c_expr(extent) for extent in conversion.layout.extents()]
    positions = [f"p{k}" for k in range(len(extents))]
    loops = "".join(
        f"{'    ' * (k + 1)}for (int64_t {p} = 0; {p} < {extent}; ++{p})\n"
        for k, (p, extent) in enumerate(zip(positions, extents, strict=True))
    )
    zero = f"    const {conversion.types(inward)[1]} zero = {{0}};\n    (void)zero;\n" if inward else ""
    collapse = f" collapse({len(extents)})" if len(extents) > 1 else ""
    return (
        f"{conversion.signature(inward, 'restrict')} {{\n{zero}"
        f"#pragma omp parallel for{collapse} schedule(static)\n{loops}"
        f"{'    ' * (len(extents) + 1)}{conversion.body(positions, inward)}\n}}\n"
    )


def host_source(
    conversions: Sequence[Conversion], gathers: Sequence["Gather"] = (), groups: Sequence["Group"] = ()
) -> str:
    """A C translation unit exporting ``relayout_in_<name>`` / ``relayout_out_<name>`` per conversion,
    ``relayout_gather_<name>`` / ``relayout_scatter_<name>`` per gather and ``relayout_zip_<name>`` /
    ``relayout_unzip_<name>`` per group."""
    dense = "".join(host_function(c, inward) for c in conversions for inward in (True, False))
    return fp_header() + dense + "".join(g.host() for g in gathers) + "".join(g.host() for g in groups)


def grid_stride(var: str, bound: str) -> str:
    """A device loop header handing ``var`` in ``[0, bound)`` to every thread of the grid in turn."""
    first = "(int64_t)blockIdx.x * blockDim.x + threadIdx.x"
    return f"for (int64_t {var} = {first}; {var} < {bound}; {var} += (int64_t)blockDim.x * gridDim.x)"


def device_function(conversion: Conversion, inward: bool) -> str:
    extents = [c_expr(extent) for extent in conversion.layout.extents()]
    positions = [f"p{k}" for k in range(len(extents))]
    decompose, rest = [], "t"
    for k in reversed(range(len(extents))):
        decompose.append(f"        const int64_t {positions[k]} = {rest} % {extents[k]};\n")
        if k:
            decompose.append(f"        const int64_t t{k} = {rest} / {extents[k]};\n")
            rest = f"t{k}"
    total = " * ".join(extents) or "1LL"
    zero = f"    const {conversion.types(inward)[1]} zero = {{0}};\n" if inward else ""
    return (
        f'extern "C" __global__ {conversion.signature(inward, "__restrict__")} {{\n{zero}'
        f"    const int64_t total = {total};\n"
        f"    {grid_stride('t', 'total')} {{\n"
        f"{''.join(decompose)}        {conversion.body(positions, inward)}\n    }}\n}}\n"
    )


def device_source(
    conversions: Sequence[Conversion], gathers: Sequence["Gather"] = (), groups: Sequence["Group"] = ()
) -> str:
    """CUDA / HIP source with one ``__global__`` kernel per conversion, gather or group and direction
    (grid-stride loops), named as in :func:`host_source`."""
    dense = "".join(device_function(c, inward) for c in conversions for inward in (True, False))
    return fp_header() + dense + "".join(g.device() for g in gathers) + "".join(g.device() for g in groups)


@dataclass(frozen=True, slots=True)
class Gather:
    """Values moved through a table (a sparse format's, :mod:`relayout.sparse`): in,
    ``stored[k] = cast(logical[table[k]])`` (zero where ``table[k] < 0``); out, the scatter back."""

    name: str
    logical: str
    stored: str

    def host(self) -> str:
        logical, stored = ELEMENTS[self.logical], ELEMENTS[self.stored]
        inward = cast(logical, stored, "src[table[k]]")
        outward = cast(stored, logical, "src[k]")
        return (
            f"void relayout_gather_{self.name}(const {logical.c_type} *restrict src, const int64_t *restrict table, "
            f"{stored.c_type} *restrict dst, int64_t n) {{\n    const {stored.c_type} zero = {{0}};\n"
            f"#pragma omp parallel for simd schedule(static)\n"
            f"    for (int64_t k = 0; k < n; ++k) dst[k] = table[k] < 0 ? zero : {inward};\n}}\n"
            f"void relayout_scatter_{self.name}(const {stored.c_type} *restrict src, const int64_t *restrict table, "
            f"{logical.c_type} *restrict dst, int64_t n) {{\n"
            f"#pragma omp parallel for simd schedule(static)\n"
            f"    for (int64_t k = 0; k < n; ++k) if (table[k] >= 0) dst[table[k]] = {outward};\n}}\n"
        )

    def device(self) -> str:
        logical, stored = ELEMENTS[self.logical], ELEMENTS[self.stored]
        loop = grid_stride("k", "n")
        return (
            f'extern "C" __global__ void relayout_gather_{self.name}(const {logical.c_type} *__restrict__ src, '
            f"const int64_t *__restrict__ table, {stored.c_type} *__restrict__ dst, int64_t n) {{\n"
            f"    const {stored.c_type} zero = {{0}};\n"
            f"    {loop} dst[k] = table[k] < 0 ? zero : {cast(logical, stored, 'src[table[k]]')};\n}}\n"
            f'extern "C" __global__ void relayout_scatter_{self.name}(const {stored.c_type} *__restrict__ src, '
            f"const int64_t *__restrict__ table, {logical.c_type} *__restrict__ dst, int64_t n) {{\n"
            f"    {loop} if (table[k] >= 0) dst[table[k]] = {cast(stored, logical, 'src[k]')};\n}}\n"
        )


@dataclass(frozen=True, slots=True)
class Member:
    """One part of a :class:`Group`: its array ``name`` laid out by ``layout`` (over the parts' logical
    shape) at element type ``logical``, and its element type ``stored`` inside the whole, as member
    ``field`` of a struct whole."""

    name: str
    layout: IndexMap
    logical: str
    stored: str
    field: str = ""


@dataclass(frozen=True, slots=True)
class Group:
    """Part arrays of one logical shape and the one array ``name`` that zips them, laid out by ``layout``:
    stacked along logical dimension ``dim`` of the whole (the parts' shape with their count inserted there),
    or, ``dim`` None, one struct of C type ``struct`` per element, part ``k`` its member ``parts[k].field``
    (a complex whole is the struct ``(re, im)``). ``relayout_zip_<name>`` fills the whole from the parts and
    ``relayout_unzip_<name>`` the parts from the whole, each casting between the parts' element types; both
    walk the parts' logical index space, so a laid-out buffer's dead cells keep what they held."""

    name: str
    layout: IndexMap
    parts: tuple[Member, ...]
    dim: int | None = None
    struct: str = ""

    def symbols(self) -> tuple[str, ...]:
        """The names the parts' logical shape reads, the kernels' ``int64_t`` arguments in this order."""
        return Conversion(self.name, self.parts[0].layout, "float64", "float64").symbols()

    def whole_type(self) -> str:
        return self.struct or ELEMENTS[self.parts[0].stored].c_type

    def typedef(self) -> str:
        """The C struct of a struct whole that is no builtin type (complex is the runtime header's)."""
        if not self.struct or self.struct.startswith("relayout_c"):
            return ""
        members = " ".join(f"{ELEMENTS[part.stored].c_type} {part.field};" for part in self.parts)
        return f"typedef struct {{ {members} }} {self.struct};\n"

    def cells(self, index: Sequence[str]) -> list[tuple[Member, str, str]]:
        """Per part at logical ``index``: the part, its buffer position and the whole's (with its member)."""
        out = []
        for k, part in enumerate(self.parts):
            at = c_expr(linear(part.layout.physical(index), part.layout.extents()))
            whole_index = list(index) if self.dim is None else [*index[: self.dim], str(k), *index[self.dim :]]
            whole = c_expr(linear(self.layout.physical(whole_index), self.layout.extents()))
            out.append((part, at, f"whole[{whole}]" + (f".{part.field}" if self.dim is None else "")))
        return out

    def bodies(self, index: Sequence[str]) -> tuple[str, str]:
        """The zip and the unzip statements at logical ``index``."""
        zips, unzips = [], []
        for k, (part, at, whole) in enumerate(self.cells(index)):
            logical, stored = ELEMENTS[part.logical], ELEMENTS[part.stored]
            zips.append(f"{whole} = {cast(logical, stored, f'p{k}[{at}]')};")
            unzips.append(f"p{k}[{at}] = {cast(stored, logical, whole)};")
        return " ".join(zips), " ".join(unzips)

    def signatures(self, qualifier: str) -> tuple[str, str]:
        parts_in = ", ".join(f"const {ELEMENTS[p.logical].c_type} *{qualifier} p{k}" for k, p in enumerate(self.parts))
        parts_out = ", ".join(f"{ELEMENTS[p.logical].c_type} *{qualifier} p{k}" for k, p in enumerate(self.parts))
        extents = "".join(f", int64_t {symbol}" for symbol in self.symbols())
        whole = self.whole_type()
        return (
            f"void relayout_zip_{self.name}({parts_in}, {whole} *{qualifier} whole{extents})",
            f"void relayout_unzip_{self.name}(const {whole} *{qualifier} whole, {parts_out}{extents})",
        )

    def host(self) -> str:
        shape = [c_expr(extent) for extent in self.parts[0].layout.shape]
        index = [f"i{d}" for d in range(len(shape))]
        loops = "".join(
            f"{'    ' * (d + 1)}for (int64_t {i} = 0; {i} < {n}; ++{i})\n"
            for d, (i, n) in enumerate(zip(index, shape, strict=True))
        )
        collapse = f" collapse({len(shape)})" if len(shape) > 1 else ""
        body = "    " * (len(shape) + 1)
        out = []
        for signature, statement in zip(self.signatures("restrict"), self.bodies(index), strict=True):
            out.append(
                f"{signature} {{\n#pragma omp parallel for{collapse} schedule(static)\n{loops}{body}{{ {statement} }}\n}}\n"
            )
        return self.typedef() + "".join(out)

    def device(self) -> str:
        shape = [c_expr(extent) for extent in self.parts[0].layout.shape]
        index = [f"i{d}" for d in range(len(shape))]
        decompose, rest = [], "t"
        for k in reversed(range(len(shape))):
            decompose.append(f"        const int64_t {index[k]} = {rest} % {shape[k]};\n")
            if k:
                decompose.append(f"        const int64_t t{k} = {rest} / {shape[k]};\n")
                rest = f"t{k}"
        total = " * ".join(shape) or "1LL"
        out = []
        for signature, statement in zip(self.signatures("__restrict__"), self.bodies(index), strict=True):
            out.append(
                f'extern "C" __global__ {signature} {{\n    const int64_t total = {total};\n'
                f"    {grid_stride('t', 'total')} {{\n{''.join(decompose)}        {statement}\n    }}\n}}\n"
            )
        return self.typedef() + "".join(out)


def compile_host(source: str, out: pathlib.Path, compiler: str = "cc", flags: Sequence[str] = ("-O2",)) -> pathlib.Path:
    """``source`` built as a shared library ``out`` with OpenMP (for tests and standalone use; a caller
    with its own toolchain builds :func:`host_source` itself)."""
    c_file = out.with_suffix(".c")
    c_file.write_text(source, encoding="utf-8")
    argv = [compiler, *flags, "-fopenmp", "-fPIC", "-shared", "-o", str(out), str(c_file)]
    subprocess.run(argv, check=True, capture_output=True)
    return out


#: The calls of a layout's expressions over numpy arrays.
NUMPY_CALLS = {"min": np.minimum, "max": np.maximum}


def placement(
    layout: IndexMap, sizes: Mapping[str, int]
) -> tuple[tuple[npt.NDArray[np.int64], ...], npt.NDArray[np.bool_]]:
    """For every cell of the laid-out buffer (``sizes`` binding the shape's symbols): the logical index
    it holds, one array per dimension, and whether it is live."""
    extents = [eval_index(extent, sizes, "extents") for extent in layout.extents()]
    grid = np.indices(extents, dtype=np.int64)
    positions = [f"p{k}" for k in range(len(extents))]
    names = {**sizes, **dict(zip(positions, grid, strict=True))}

    def array(expr: str) -> npt.NDArray[np.int64]:
        return np.broadcast_to(evaluate(expr, names, NUMPY_CALLS, "layout"), extents)

    live = np.ones(extents, dtype=bool)
    for condition in layout.live(positions):
        live &= array(condition)
    return tuple(array(expr) for expr in layout.logical(positions)), live


def to_layout(
    logical: npt.NDArray[np.generic], layout: IndexMap, sizes: Mapping[str, int], dtype: npt.DTypeLike | None = None
) -> npt.NDArray[np.generic]:
    """The laid-out buffer of ``logical`` at ``dtype`` (default: its own), zero in dead cells: the numpy
    form of the ``relayout_in`` kernel."""
    index, live = placement(layout, sizes)
    out = np.zeros(live.shape, dtype=logical.dtype if dtype is None else dtype)
    out[live] = logical[tuple(axis[live] for axis in index)]
    return out


def from_layout(
    stored: npt.NDArray[np.generic], layout: IndexMap, sizes: Mapping[str, int], dtype: npt.DTypeLike | None = None
) -> npt.NDArray[np.generic]:
    """The logical array ``stored`` holds at ``dtype``: the numpy form of the ``relayout_out`` kernel."""
    index, live = placement(layout, sizes)
    shape = tuple(eval_index(extent, sizes, "shape") for extent in layout.shape)
    out = np.zeros(shape, dtype=stored.dtype if dtype is None else dtype)
    out[tuple(axis[live] for axis in index)] = stored[live]
    return out
