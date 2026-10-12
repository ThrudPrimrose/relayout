# Layout op sequences

`relayout.ops` is the schema of a relayout request and its lowering. A consumer (hpcagent-bench's
`requests.layout`, a relayout pass) sends one op list per array; relayout checks it, inverts it and
lowers it. Adding an op is one registered class (`@layout_op`): its fields' types name their JSON
codecs, and it defines `apply` (its effect on the digit tuple) and `inverse`.

## The digit tuple

An array starts as one **digit** per dimension, row-major, named `i0, i1, ...`. A digit of dimension
`d` with stride `s` and extent `e` is `(i[d] // s) % e`; the tuple read left to right is the storage
order. Every op addresses digits by name, and an op that cuts a digit names the two it leaves (`into`,
coarse then fine), so a sequence reads without tracking positions.

Dimensions are numpy's: buffers are C-contiguous and the ops act in that row-major index space. A
Fortran kernel declares the same buffer with reversed extents (numpy `A[M, N]` is Fortran `A(N, M)`);
reversing the dimensions and the accesses is a codegen step after the ops, never an op.

| Op | JSON | Effect | Inverse |
| --- | --- | --- | --- |
| block | `{"op": "block", "digit": n, "factor": b, "into": [c, f]}` | cuts `(d, s, e)` into `c = (d, s·b, ⌈e/b⌉)` and `f = (d, s, b)`, in place | unblock |
| unblock | `{"op": "unblock", "digits": [c, f], "factor": b, "into": n}` | merges a block's two digits | block |
| split | `{"op": "split", "digit": n, "parts": p, "into": [r, l]}` | cuts a whole dimension's digit into `p` contiguous load-balanced bands (sizes differ by at most one): `r` the band index, `l` the index within the band | unsplit |
| unsplit | `{"op": "unsplit", "digits": [r, l], "parts": p, "into": n}` | merges a split's two digits | split |
| permute | `{"op": "permute", "order": [n, ...]}` | reorders the tuple; names every digit once | the order before it |
| pad | `{"op": "pad", "digit": n, "amount": a}` | grows a digit by `a` dead cells | unpad |
| unpad | `{"op": "unpad", "digit": n, "amount": a}` | drops `a` dead cells | pad |
| shuffle | `{"op": "shuffle", "digits": [n, ...], "f": [...], "f_inv": [...]}` | renumbers the digits by the bijection `f`, one expression per digit over their names | `f` and `f_inv` swapped |
| linear_shuffle | `{"op": "linear_shuffle", "digits": [n, ...], "bits": [b, ...], "matrix": [[0/1, ...], ...]}` | renumbers digits of extent `2^b` by an invertible linear map over GF(2) on their bits | the inverse matrix |
| distribute | `{"op": "distribute", "digit": n, "grid_dim": g}` | digit `n` (extent `P_g`) becomes rank coordinate `g` and leaves the tuple | the gather |

`factor`, `parts` and `amount` are integer expressions over the grid extents `P0, P1, ...`, so one
sequence stays valid at every rank count. Expressions use ints, names, `+ - * // % ^ & | << >>` and
`min(a, b)`, `max(a, b)`, `root(a, k)` (the floor of the `k`-th root), with Python's floor semantics;
relayout parses and evaluates them itself, never with `eval`.

`inverse_sequence(ops, ndim)` is the sequence that undoes `ops`: reversed, each op inverted against
the digit names before it.

**Shuffle.** A shuffle maps the whole index vector of its digits: the XOR swizzle
`col' = col ^ (row % 8)` is `"digits": ["i0", "i1"], "f": ["i0", "i1 ^ (i0 % 8)"]`. The caller
supplies `f_inv`, checked exhaustively (`f_inv(f(i)) == i`) over the padded extent before any build.
`linear_shuffle` is the form of Linear Layouts (Zhou et al., arXiv:2505.23819): the input bit vector
is each digit's bits, least significant first, in `digits` order, and row `r` of the matrix is output
bit `r` as the XOR of the input bits it marks. A singular matrix is refused when parsed.

**Dead cells.** `pad`, and a `block` whose factor does not divide its extent, create cells that hold
no element: inputs see them zero-filled, outputs in them are ignored.

## The processor grid

A grid is a list of extent expressions over the rank count `P` (`parse_grid`, `grid_extents`):
`["P"]` is 1-D at every rank count, `["root(P, 2)", "root(P, 2)"]` is square, `["P // 2", 2]` keeps
two columns. At a rank count the extents must be positive and multiply to `P`; a rank count a grid
does not span (a square grid at `P = 8`) is not graded with it.

## Distribution

Ops before a `distribute` shape ownership (which global elements a rank gets); ops after it shape
each rank's local storage. A grid dimension no `distribute` of an array names replicates the array
across it; an array with no `distribute` is fully replicated.

`lower_layout(ops, ndim, grid)` lowers a sequence to one `AxisSplit(grid_dim, scheme, block_size)`
per dimension (None = not split). `ops_for_axes` is its inverse; dimension `d`'s rank digit is `r<d>`:

| Scheme | Op sequence for dimension `d` on grid dimension `g` |
| --- | --- |
| block (load-balanced bands) | `split(i<d>, P<g>, into [r<d>, i<d>])`, `distribute(r<d>, g)` |
| cyclic | `block(i<d>, P<g>, into [i<d>, r<d>])`, `distribute(r<d>, g)` |
| block-cyclic `b` | `block(i<d>, b*P<g>, into [i<d>, c<d>])`, `block(c<d>, b, into [r<d>, b<d>])`, `distribute(r<d>, g)` |
| replicated | no `distribute` on `d` |

The block-cyclic local tuple `[(d, b·P, ⌈N/bP⌉), (d, 1, b)]` is ScaLAPACK's local storage.

**What lowers today:** block, unblock, split, unsplit, permute and distribute, where every
distributed digit is a split's band index or a digit of extent `P_g` between a coarse digit of stride
`s·P_g` and, for `s > 1`, a fine digit of extent `s`, and the rank-local digits end row-major (a
refusal names the `permute` to append). `pad`, `unpad`, `shuffle` and `linear_shuffle` are parsed,
checked and inverted, and refused at lowering as not implemented yet; they follow with the pack
kernels.

**Rules.** Every op type-checks against the current tuple (the digit exists, a new name is free, a
permute names every digit once); a distributed digit's extent equals `P_g` at every measured rank
count; each grid dimension and each array dimension is distributed at most once; every shuffle passes
the inverse check; all extents are positive at every measured `P`.

### ScaLAPACK coverage

`distribute` covers every dense ScaLAPACK descriptor (DTYPE=1). Per dimension with block `MB`,
source process `RSRC` and leading dimension `LLD`:

1. `block(i, MB·P)`, `block(c, MB)`: block-cyclic as above.
2. `RSRC ≠ 0`: ScaLAPACK gives global block `g` to process `(g + RSRC) % P` and keeps its local index
   `g // P`, so the shuffle rotates within each group of `P` blocks, before the blocks:
   `f(i) = ((g // P)·P + (g + RSRC) % P)·MB + i % MB` with `g = i // MB`; `f_inv` uses `-RSRC (mod P)`.
3. `distribute` the digit of extent `P`.
4. ScaLAPACK's column-major local storage of a numpy matrix is a `permute` after the distribute. A
   Fortran kernel needs none: its reversed-extent declaration of the row-major tile is already
   column-major, over the transposed matrix.
5. `LLD > LOCr` is `pad` on the local row digit after the distribute.
6. BLACS process ordering is a property of the grid (row-major today), not of an array.

Any partition of `[N]` into `P` parts whose largest has `m` elements is `pad(i, P·m − N)`,
`shuffle(i, σ)`, `block(i, m)`, `distribute`, where `σ` sends each element to slot `owner·m + local`.
Not expressible: halos and ghost cells (an element on two ranks is not injective), and ScaLAPACK's
band and out-of-core descriptors.

## Group ops

`zip` and `unzip` change how many arrays there are, so they are not per-array ops: a consumer lists
them as `{name: [fields]}` under their own keys (`groups_from_json`), with no inverse link between
them. `zip` fuses arrays of one logical shape into one array `name` of C structs, one member per
field in order; `unzip` splits the struct array `name` into one array per member (a complex array is
the struct `(re, im)`). Unzips run before zips; the arrays an unzip creates take a layout like any
other, and a zip's layout is keyed by its `name`.

## Order of the conversions

Precision is elementwise and commutes with every op except zip, so it is defined on the logical arrays,
before zip. The cast fuses into the relayout pack and unpack kernels: one kernel per array (per zip
group) each way, `dst[p] = convert(src[L⁻¹(p)])`, writes contiguous, dead cells zero.

## Open items

- **One algebra:** `relayout.ops` simulates its digit tuple over concrete grids; moving it onto dace's
  symbolic `LayoutMap` (`dace/libraries/layout/algebra.py`), with each array's symbolic shape, gives
  the symbolic extents `pad` and dead cells need.
- **Python-AST rewrite:** a starter in the requested layout rewrites each numpy subscript `A[i, j]` to
  the laid-out index, the Python counterpart of dace's SDFG `subscript_rewrite`.
- **AoSoA:** fields interleaved per block (`u0..u63, v0..v63, u64..`), as an option of `zip`.
