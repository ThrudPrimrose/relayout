# Relayout requests

A request says how the data of a kernel's arrays should reach the kernel. It is a Python API in
`relayout.ops` and, one to one, a JSON form in `relayout.requests`; hpcagent-bench takes the JSON in
its submission envelope and hands it to relayout.

```json
{
  "startup": {"load": ["openblas"], "grid": ["P"]},
  "operations": [
    {"action": "cast", "array": "A", "to": "fp32"},
    {"action": "unzip", "array": "Z", "into": ["Z_re", "Z_im"]},
    {"action": "zip", "arrays": ["Z_im", "B"], "into": "W"},
    {"action": "block", "array": "A", "digit": "i0", "factor": "32*P0", "into": ["i0", "c"]},
    {"action": "block", "array": "A", "digit": "c", "factor": 32, "into": ["r", "b"]},
    {"action": "distribute", "array": "A", "grid_dim": 0, "digit": "r"},
    {"action": "distribute", "array": "W", "grid_dim": 0, "dim": 0, "scheme": "block"},
    {"action": "replicate", "array": "x"},
    {"action": "place", "array": "A", "on": "device"},
    {"action": "format", "array": "M", "to": "csr"}
  ]
}
```

| JSON | Python (`relayout.ops`) |
| --- | --- |
| the request | `Requests(startup, operations)`; `relayout.requests.parse_requests` / `requests_json` |
| `startup` | `Startup(load, grid)` |
| `{"action": "block", "array": ..., ...}` | `Block(array, digit, factor, into)`: the class registered under `Action.BLOCK`, same field names |
| a field's JSON value | its codec, chosen by the field's type (`relayout.requests.FIELD_CODECS`) |
| the JSON Schema | `relayout.requests.requests_schema()`, generated from the registry |

Adding an operation is one dataclass registered with `@operation`: its fields, its checks
(`validate`, at construction) and its effect (`apply`). Its JSON and its schema follow from the field
types.

## Startup

- `load`: library names the caller links.
- `grid`: the processor grid's extents as integer expressions over the rank count `P`, which multiply
  to `P`: `["P"]` is 1-D, `["root(P, 2)", "root(P, 2)"]` square, `["P // 2", 2]` two columns. Every
  rank count is evaluated on its own (`Requests.grid_at`); a rank count a grid does not span is not
  run with it.

## Operations

`run(operations, arrays, grid)` runs the program top to bottom over the kernel's arrays
(`ArraySpec(name, ndim, sparse, members)`). Each operation sees what the previous ones left: a digit a
cut names is usable on the next line, the arrays an `unzip` makes exist from then on, and the arrays a
`zip` fuses are gone. An array starts whole, on the host, row-major, at the problem precision. The first
operation that cannot apply is a `ValueError` naming its entry (`operations[5] (distribute): ...`).

| Action | Fields | Effect |
| --- | --- | --- |
| `cast` | `array`, `to` | the element type (`fp64` `fp32` `fp16` `bf16` `fp8_e4m3` `fp8_e5m2`) |
| `format` | `array`, `to` | a sparse array's storage format (the caller checks the label) |
| `zip` | `arrays`, `into` | fuses whole arrays of one shape into one array of structs, one member per array, each keeping its precision |
| `unzip` | `array`, `into` | splits a whole struct array (a complex array is `(re, im)`) into one array per member, each keeping its member's precision |
| `block` | `array`, `digit`, `factor`, `into` | cuts `(d, s, e)` into coarse `(d, s·b, ⌈e/b⌉)` and fine `(d, s, b)`, in place |
| `unblock` | `array`, `digits`, `factor`, `into` | merges a block's two digits |
| `split` | `array`, `digit`, `parts`, `into` | cuts a whole dimension's digit into `parts` contiguous balanced bands: the band index and the index within the band |
| `unsplit` | `array`, `digits`, `parts`, `into` | merges a split's two digits |
| `permute` | `array`, `order` | reorders the digit tuple; names every digit once |
| `pad` / `unpad` | `array`, `digit`, `amount` | adds / drops dead cells |
| `shuffle` | `array`, `digits`, `f`, `f_inv` | renumbers the digits by the bijection `f`, one expression per digit over their names |
| `linear_shuffle` | `array`, `digits`, `bits`, `matrix` | renumbers digits of extent `2^b` by an invertible linear map over GF(2) |
| `distribute` | `array`, `grid_dim`, and `digit` or `dim` + `scheme` [+ `block_size`] | hands a digit of extent `P_g` to grid dimension `g` |
| `replicate` | `array` | holds the array whole on every rank |
| `place` | `array`, `on` | host or device memory during the call |

**Digits.** An array starts as one digit per dimension, named `i0, i1, ...`, over its row-major (numpy)
index space. A digit of dimension `d` with stride `s` and extent `e` is `(i[d] // s) % e`; the tuple
read left to right is the storage order. A Fortran kernel declares the same buffer with reversed
extents (numpy `A[M, N]` is Fortran `A(N, M)`); reversing is a codegen step after the operations.

**Expressions.** `factor`, `parts`, `amount` and the grid's extents are integer expressions over the
grid extents `P0, P1, ...` (the grid's over `P`): ints, names, `+ - * // % ^ & | << >>`, `min(a, b)`,
`max(a, b)` and `root(a, k)` (the floor of the `k`-th root), with Python's floor semantics, evaluated
without `eval`.

**Distribute, short form.** `{"dim": d, "scheme": s}` cuts dimension `d`'s whole digit and distributes
its rank digit `r<d>`: `block` is `split` by `P_g`, `cyclic` is `block` by `P_g` with the fine digit
distributed, `block_cyclic` of width `b` is `block` by `b·P_g`, `block` of its fine digit by `b`, the
digit of extent `P_g` distributed (ScaLAPACK's local storage). A kernel's default distribution is
shown this way (`operations_for_axes`).

**Shuffle.** The XOR swizzle `col' = col ^ (row % 8)` is `"digits": ["i0", "i1"], "f": ["i0", "i1 ^ (i0
% 8)"]`. The caller supplies `f_inv`, checked exhaustively before any build. `linear_shuffle` is the form
of Linear Layouts (Zhou et al., arXiv:2505.23819): the input bit vector is each digit's bits, least
significant first, in `digits` order; row `r` of the matrix is output bit `r` as the XOR of the input
bits it marks. A singular matrix is refused.

**Dead cells.** `pad`, and a `block` whose factor does not divide its extent, create cells that hold no
element: inputs see them zero-filled, outputs in them are ignored.

**Inverses.** `inverse_sequence(operations, ndim)` undoes one array's digit operations: reversed, each
inverted against the digit names before it. A `distribute` is undone by the gather.

## Lowering

`lower(state)` gives each laid-out array its `AxisSplit(grid_dim, scheme, block_size)` per dimension
(None = not split; a replicated array maps to None). It lowers block, unblock, split, unsplit, permute,
distribute, replicate, cast, format and place, where every distributed digit is a split's band index
or a digit of extent `P_g` between a coarse digit of stride `s·P_g` and, for `s > 1`, a fine digit of
extent `s`, and the rank-local digits end row-major (a refusal names the `permute` to append). `pad`,
`unpad`, `shuffle`, `linear_shuffle`, `zip` and `unzip` are parsed and checked, and refused at lowering
as not implemented yet.

Ops before a `distribute` shape ownership; ops after it shape each rank's local storage. A grid
dimension no `distribute` of an array names replicates the array across it.

### ScaLAPACK coverage

`distribute` covers every dense ScaLAPACK descriptor (DTYPE=1). Per dimension with block `MB`,
source process `RSRC` and leading dimension `LLD`:

1. `block` by `MB·P`, then its fine digit by `MB`: block-cyclic as above.
2. `RSRC ≠ 0`: ScaLAPACK gives global block `g` to process `(g + RSRC) % P` and keeps its local index
   `g // P`, so the shuffle rotates within each group of `P` blocks, before the blocks:
   `f(i) = ((g // P)·P + (g + RSRC) % P)·MB + i % MB` with `g = i // MB`; `f_inv` uses `-RSRC (mod P)`.
3. `distribute` the digit of extent `P`.
4. ScaLAPACK's column-major local storage of a numpy matrix is a `permute` after the distribute. A
   Fortran kernel needs none: its reversed-extent declaration of the row-major tile is already
   column-major, over the transposed matrix.
5. `LLD > LOCr` is `pad` on the local row digit after the distribute.
6. BLACS process ordering is a property of the grid (row-major today), not of an array.

Any partition of `[N]` into `P` parts whose largest has `m` elements is `pad` by `P·m − N`, `shuffle` by
`σ`, `block` by `m`, `distribute`, where `σ` sends each element to slot `owner·m + local`. Not
expressible: halos and ghost cells (an element on two ranks is not injective), and ScaLAPACK's band and
out-of-core descriptors.

## Order of the conversions

Precision is elementwise and commutes with every layout operation except zip, so a cast fuses into
the pack and unpack kernels: one kernel per array (per zip group) each way, `dst[p] =
convert(src[L⁻¹(p)])`, writes contiguous, dead cells zero.

## Next

- **Python-AST rewrite:** the caller's numpy reference rewritten to the requested layout (each
  subscript `A[i, j]` to the laid-out index, a zip to a struct field access), the Python counterpart of
  dace's SDFG `subscript_rewrite`. hpcagent-bench then emits C from the rewritten numpy, which needs
  struct support in its numpy-to-C lowering for `zip`.
- **One algebra:** the digit tuple onto dace's symbolic `LayoutMap`
  (`dace/libraries/layout/algebra.py`), with each array's symbolic shape (`ArraySpec` extents), so `pad`
  and dead cells are exact.
- **AoSoA:** fields interleaved per block (`u0..u63, v0..v63, u64..`), as an option of `zip`.
