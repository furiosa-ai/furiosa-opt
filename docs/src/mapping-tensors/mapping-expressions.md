# Mapping Expressions

A mapping expression is a Rust type that encodes a mapping.
Read a position's `Cell` first; then use the `Index` terms, padding kind, or out-of-bounds result to debug the access.
This section defines the constructors and supported behavior that produce those results.

## Axis Sizes

The `axes!` macro declares axis identifiers and their sizes.
The following declaration applies throughout this section:

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 512];
```

## Position Results

A mapping expression like `m![H, W]` maps each physical position to a `Cell`.
See the result before reading its terms:

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
axes![A = 8];
type E = m![A];
assert_eq!(E::map(0), i![A: 0]);
```

`Cell::Index` is a structural live result: it preserves the mapping terms, while `Index::finalize()` validates their scalar coordinates and can return an error for a modulo overshoot.
Padding and out-of-bounds positions do not contain live terms.
`Cell::Index(Index::new())` is a valid live result with no axis terms; it is distinct from `Cell::Padding` and `Cell::OutOfBounds`, so an empty `Index` never stands for a non-live position.
The `i!` macro combines assignments in the order written, which is the major-to-minor order for a composed mapping; because `Cell::combine` is non-commutative for padding, reorder assignments only when that order is intended.

## `M` Constructor Rules

Every mapping expression implements the `M` trait, which provides the buffer size and position mapping:

```rust
// Inside `furiosa_opt_std::prelude`...
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use std::fmt::Debug;
{{#include ../../../furiosa-mapping-types/src/dsl.rs:trait_m}}

/// A live tensor index: a map from axis identifiers to coordinate values.
pub struct Index { /* ... */ }

/// Results for live, padded, and out-of-range positions.
pub enum Cell { Index(Index), Padding(PaddingKind), OutOfBounds }

/// Constructs classified cells.
/// `i![A: 2, B: 3]` creates a live `Cell::Index` with A = 2 and B = 3;
/// non-live contributions remain `Cell::Padding` or `Cell::OutOfBounds`.
macro_rules! i {
    # () => {};
    /* ... */
}
```

### Host Tensor Usage

The simplest concrete type built on the `M` trait is `HostTensor<D, E>`: a host memory buffer of element type `D` whose representation is fully determined by mapping `E`.
`E` determines both the buffer size (`E::SIZE`) and the correspondence from buffer positions to cells (`E::map`).
`HostTensor<bf16, m![A, B]>` contains 4,096 elements of `bf16` data.
For a host tensor, each `Cell::Index(index)` returned by `E::map` identifies the structural `Index` stored at that physical position.
Call `Index::finalize()` to validate its scalar coordinates before treating it as a tensor index; an overshoot can fail there.
Convert `Cell::OutOfBounds` with `CellExt::finalize(PaddingKind::Top)` for reads or `CellExt::finalize(PaddingKind::Bottom)` for writes.
The formal definition of *holds* appears in [Tensor Semantics](./tensor-semantics.md).

Device tensors such as `HbmTensor` and `DmTensor` have more complex representations spanning multiple mapping expressions; see [Spatial and Temporal Dimensions](./spatial-temporal-dimensions.md) for details.

## Constructors

Mapping expressions, including the mapping `E` in `HostTensor<D, E>`, are built by composing small constructors, each of which transforms or combines simpler mappings.
These expressions use arithmetic-like operators (`/`, `%`, and `#` for padding) to concisely define the mapping between tensor and linear buffer indices.

### Symbol

A symbol is a single uppercase letter whose size comes from the shape declaration.
The mapping `m![A]` maps 8 buffer indices linearly to tensor indices along the axis:

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
#
# axes![A = 8];
#
type E = m![A]; // Symbol<Ident::A, 8>

fn test_symbol() {
    assert_eq!(E::map(0), i![A: 0]);
    assert_eq!(E::map(1), i![A: 1]);
    assert_eq!(E::map(2), i![A: 2]);
    for i in 0..E::SIZE {
        assert_eq!(E::map(i), i![A: i]);
    }
    assert_eq!(E::map(E::SIZE), Cell::OutOfBounds);
}
#
# test_symbol();
```

```rust,ignore
{{#include ../../../furiosa-mapping-types/src/dsl.rs:symbol_impl}}
```

> [!NOTE]
> For every symbol `A`, the zeroth index `i![A: 0]` is equivalent to the empty tensor index `i![]`.

### Pair

The pair mapping `m![A, B]` stores a 2D tensor with shape \\(\\{A=8, B=512\\}\\) as a buffer of 4,096 elements.
The mapping `Pair<L, R>` maps the Cartesian product of two spaces into a linear buffer where `L` is the major dimension and `R` is the minor dimension.
The size is `L::SIZE * R::SIZE`, and the mapping uses floor division and modulo to decompose indices.
If the major (`L`) result is padding, its kind wins before the minor (`R`) result is considered; an out-of-bounds result propagates when no live combination is possible.
`m![A, B, C, D]` expands to `Pair<A, Pair<B, Pair<C, D>>>` and is right-associative.

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
#
# axes![A = 8, B = 512];
#
type E = m![A, B]; // Pair<m![A], m![B]>

fn test_pair() {
    // First 512 elements hold A=0, next 512 hold A=1
    assert_eq!(E::map(0),   i![A: 0, B: 0]);
    assert_eq!(E::map(511), i![A: 0, B: 511]);
    assert_eq!(E::map(512), i![A: 1, B: 0]);
    assert_eq!(E::map(519), i![A: 1, B: 7]); // 519 == 512 * 1 + 7
    for i in 0..E::SIZE {
        assert_eq!(E::map(i), i![A: i / <m![B]>::SIZE, B: i % <m![B]>::SIZE]);
    }
    assert_eq!(E::map(E::SIZE), Cell::OutOfBounds);
}
#
# test_pair();
```

```rust,ignore
{{#include ../../../furiosa-mapping-types/src/dsl.rs:pair_impl}}
```

### Identity

The identity mapping `m![1]` creates a single-element buffer that maps buffer index `0` to the empty tensor index `i![]`.
It serves as the identity element for `Pair`: `m![1, A]` and `m![A, 1]` are both equivalent to `m![A]`.
More generally, a broadcast mapping returns the same live empty `Cell::Index(Index::new())` for every position below its size, and returns `Cell::OutOfBounds` beyond that size.

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
#
type E = m![1]; // Identity

fn test_identity() {
    assert_eq!(E::map(0), i![]);
    assert_eq!(E::map(1), Cell::OutOfBounds);
}
#
# test_identity();
```

```rust,ignore
{{#include ../../../furiosa-mapping-types/src/dsl.rs:identity_impl}}
```

### Padding

Padding aligns data to hardware requirements by adding unused buffer space.
For example, the DMA engine requires rows to start on 64-byte boundaries.
With `axes![C = 13, D = 61]`, `m![C, D]` creates misaligned rows since `61` is not divisible by `64`.
`m![C, D # 64]` fixes this by aligning each row to 64-byte boundaries, using 3 extra elements per row.

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
#
axes![C = 13, D = 61];

type E = m![C, D # 64]; // Pair<m![C], Padding<m![D], 64>>

fn test_padding() {
    assert_eq!(E::map(0),  i![C: 0, D: 0]);
    assert_eq!(E::map(60), i![C: 0, D: 60]);
    assert_eq!(E::map(61), Cell::Padding(PaddingKind::Top));
    assert_eq!(E::map(62), Cell::Padding(PaddingKind::Top));
    assert_eq!(E::map(63), Cell::Padding(PaddingKind::Top));
    assert_eq!(E::map(64), i![C: 1, D: 0]);
}
#
# test_padding();
```

```rust,ignore
{{#include ../../../furiosa-mapping-types/src/dsl.rs:padding_impl}}
```

The padded positions' content is part of the type, not just their count.
Three kinds are tracked.

- `m![A # m]` (or `m![A #{*} m]`) is top padding to size `m`.
  These positions are accessible but hold arbitrary values.
  Raw DM tensors carry this.
`#` is the shorthand; `#{*}` spells the kind out explicitly.
- `m![A #{0} m]` is zero-filled padding to size `m`.
  These positions are accessible and known to hold zero.
  The mapping records that value property; the expression itself does not perform the fill.
- `m![A #{!} m]` is bottom padding to size `m`.
  These positions are inaccessible and reads/writes are undefined behavior.
  This models addresses the compiler must avoid.

`#` defaults to top kind.
The Rust type level mirrors this via a const generic of `PaddingKind` on `Padding<L, SIZE, KIND>`.
`Padding<L, N>` is `KIND = PaddingKind::Top`, `Padding<L, N, { PaddingKind::Zero }>` is the zero-filled variant, and `Padding<L, N, { PaddingKind::Bottom }>` is inaccessible.

`Cell::Padding(kind)` reports a position inside the padded extent but outside the inner mapping.
The `kind` is `Top`, `Zero`, or `Bottom`; positions beyond the padded extent remain `Cell::OutOfBounds`.
The complete `Cell` result definition appears in [Position Results](#position-results).

### Resize

Resize constrains a mapping to a smaller logical size by truncating indices beyond the new size, discarding elements outside that range.
Unlike padding, which expands the buffer, Resize shrinks the logical view.
The mapping `m![D = 2]` takes only the first 2 elements of axis `D`, producing indices `D = 0` and `D = 1`.

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
#
axes![C = 2, D = 3];
type E = m![C, D = 2]; // Pair<m![C], Resize<m![D], 2>>

fn test_resize() {
    assert_eq!(E::map(0), i![C: 0, D: 0]);
    assert_eq!(E::map(1), i![C: 0, D: 1]);
    assert_eq!(E::map(2), i![C: 1, D: 0]);
    assert_eq!(E::map(3), i![C: 1, D: 1]);
    assert_eq!(E::map(4), Cell::OutOfBounds);
}
#
# test_resize();
```

```rust,ignore
{{#include ../../../furiosa-mapping-types/src/dsl.rs:resize_impl}}
```

### Tiling

Tiling is implemented through indexed views: `tile` validates a mapping split, then creates a metadata view without copying data.
The generic view API is `TensorView::tile<I, E2, LEN>(start)`.
`I` is the mapping used to locate the tile, `E2` is the requested view mapping, `LEN` is the number of `I` cells in the tile, and `start` is the logical starting coordinate along `I`.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
#
# axes![A = 8, B = 512];
#
let tensor = HbmTensor::<bf16, m![1], m![A, B]>::new();
let view = tensor.view(); // HbmTensorView::<'_, bf16, m![1], m![A, B]>
let tile01 = view.tile::<m![B], 2, m![A, B = 2 # 512]>(0); // HbmTensorView::<'_, bf16, m![1], m![A, B = 2 # 512]>
let tile23 = view.tile::<m![B], 2, m![A, B = 2 # 512]>(2); // HbmTensorView::<'_, bf16, m![1], m![A, B = 2 # 512]>
```

The HBM example uses `I = m![B]`, `LEN = 2`, and `E2 = m![A, B = 2 # 512]`.
The `B = 2 # 512` mapping gives the view two live `B` positions inside a 512-cell physical extent; without that footprint, the split does not validate against the source mapping.
`start = 0` and `start = 2` are logical `B` coordinates, so these views cover ranges `0..2` and `2..4`.
They are starts, not a separate tile-number API.

Read views fill cells outside the tile with `PaddingKind::Top`; mutable views require `PaddingKind::Bottom` there so writes cannot escape the tile.
HBM and DM expose tier-specific tile wrappers, while TRF, VRF, and DPE use the generic view; `HostTensor` has no view/tile API.

### Stride and Modulo

Stride (`/`) and modulo (`%`) decompose a single dimension into two: the outer (block index) and the inner (position within block).
Consider the 512-element axis `B` divided into 8 blocks of 64 elements each.
The mapping `m![B / 64, B % 64]` creates an 8 × 64 grid where the first dimension selects which block and the second dimension selects the position within that block:

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
# axes![A = 8, B = 512];
type D1 = m![B / 64]; // stride with size 8
type D2 = m![B % 64]; // modulo with size 64

type E = m![B / 64, B % 64]; // equivalent to `m![B]`

fn test_stride_modulo() {
    assert_eq!(E::map(130), i![B / 64: 2, B % 64: 2]); // block 2, position 2: B = 64*2 + 2 = 130
    assert_eq!(E::map(130), <m![B]>::map(130));               // same result as flat m![B]

    for i in 0..8 {
        assert_eq!(D1::map(i), i![B / 64: i]);
    }
    assert_eq!(D1::map(8), Cell::OutOfBounds);

    for j in 0..64 {
        assert_eq!(D2::map(j), i![B % 64: j]);
    }
    assert_eq!(D2::map(64), Cell::OutOfBounds);

    for i in 0..8 {
        for j in 0..64 {
            assert_eq!(
                E::map(64 * i + j),
                <m![B]>::map(64 * i + j),
            );
        }
    }
    assert_eq!(E::map(512), Cell::OutOfBounds);
}
#
# test_stride_modulo();
```

```rust,ignore
{{#include ../../../furiosa-mapping-types/src/dsl.rs:stride_impl}}

{{#include ../../../furiosa-mapping-types/src/dsl.rs:modulo_impl}}
```

Stride and modulo mappings can be visualized in tabular form.
Consider the mapping `m![B / 4, B % 4]` with `B::SIZE = 16`.
The following table shows how buffer indices are arranged: each row corresponds to a specific index of `B / 4` (the stride axis), and each column corresponds to an index of `B % 4` (the modulo axis):

|                 | `i![B % 4: 0]` | `i![B % 4: 1]` | `i![B % 4: 2]` | `i![B % 4: 3]` |
| --------------- | -------------- | -------------- | -------------- | -------------- |
| `i![B / 4: 0]` | `i![B: 0]`     | `i![B: 1]`     | `i![B: 2]`     | `i![B: 3]`     |
| `i![B / 4: 1]` | `i![B: 4]`     | `i![B: 5]`     | `i![B: 6]`     | `i![B: 7]`     |
| `i![B / 4: 2]` | `i![B: 8]`     | `i![B: 9]`     | `i![B: 10]`    | `i![B: 11]`    |
| `i![B / 4: 3]` | `i![B: 12]`    | `i![B: 13]`    | `i![B: 14]`    | `i![B: 15]`    |

Modulo differs from resize in how it handles buffer size:
- Resize shrinks the buffer by truncating indices beyond the new size.
- Modulo preserves the original buffer size while partitioning it into equal-sized blocks.

These operations can be nested for complex decompositions.
The following example splits `B` into three dimensions where the buffer's bit arrangement differs from that of the tensor index.

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
# axes![A = 8, B = 512];
// B's bits: 6 - 8,  0 - 4,          5
// Values:   0 - 7, 0 - 31,      0 - 1
type E = m![B / 64, B % 32, B / 32 % 2];

fn test_nested_stride() {
    assert_eq!(E::map(67), i![B: 97]); // 67 = 64*1 + 2*1 + 1 (i=1,j=1,k=1) → B = 64*1 + 1 + 32*1 = 97
    // Verify B=97 round-trips: 97/64=1, 97%32=1, (97/32)%2=1
    assert_eq!(97 / 64, 1);
    assert_eq!(97 % 32, 1);
    assert_eq!((97 / 32) % 2, 1);

    // buffer index: 64 * i + 2 * j + k (i = block, j = position within block, k = sub-block)
    // tensor index B: 64 * i + j + 32 * k (rearranges bit positions)
    for i in 0..8 {
        for j in 0..32 {
            for k in 0..2 {
                assert_eq!(
                    E::map(64 * i + 2 * j + k),
                    i![B: 64 * i + j + 32 * k],
                );
            }
        }
    }
    assert_eq!(E::map(512), Cell::OutOfBounds);
}
#
# test_nested_stride();
```

This kind of bit rearrangement maps naturally to hardware representations where address bits are reordered for bank interleaving or cache efficiency.
In binary, this rearranges bit positions: buffer `001_00001_1` becomes `B = 001_1_00001`.
The buffer groups bits as `[8:6]_[5:1]_[0]` while `B` groups them as `[8:6]_[5]_[4:0]`.

Tiling can operate on blocks rather than individual elements.
The following example tiles by block using `m![B / 32]` and creates overlapping tiles:

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# axes![A = 8, B = 512];
let tensor = HbmTensor::<bf16, m![1], m![A, B]>::new();
for i in 0..15 {
    let tile = tensor.view().tile::<m![B / 32], 2, m![A, B / 32 = 2 # 16, B % 32]>(i);
}
```

With `B = 512`, the dimension `B / 32` has 16 blocks numbered 0-15.
Each tile takes 2 consecutive blocks starting at index `i`.
Tile 0 covers blocks `{0, 1}`, tile 1 covers blocks `{1, 2}`, and so on through tile 14 covering blocks `{14, 15}`.
These tiles overlap because consecutive tiles share one block.

The tile mapping `B / 32 = 2` resizes the block dimension to 2 since each tile contains exactly 2 blocks.
When tiling with a single block, `B / 32 = 1` simplifies to the identity `m![1]` since the dimension has only one value.

### Escape

For complex mappings, define type aliases and reference them using `{ ... }`.
With separate mappings `L = m![A]` and `R = m![B]`, combining them as `m![{ L }, { R }]` produces the same result as `m![A, B]`:

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# axes![A = 8, B = 512];
type L = m![A];
type R = m![B];
type E = m![{ L }, { R }]; // equivalent to `m![A, B]`

fn test_escape() {
    for i in 0..E::SIZE {
        assert_eq!(E::map(i), <m![A, B]>::map(i));
    }
}
#
# test_escape();
```

This escape syntax breaks down complex mappings into named, reusable components.



## Equivalent Mappings

Different constructor combinations can produce the same position behavior.
This behavioral relation is distinct from structural equality of Rust mapping values.
Mappings `E1` and `E2` are *equivalent* when:
- `E1::SIZE == E2::SIZE`, and
- For every `i`, `E1::map(i) == E2::map(i)`.

The equivalence relation is reflexive, symmetric, and transitive.
The identities below describe useful behavior; they do not assert that normalization gives a unique structural representative for every equivalent input.

Normalization has a narrower tested behavior: applying it twice is idempotent; generated mapping corpora preserve their tested position and padding behavior; and normalized mappings round-trip through the mapping representation.
These tests do not prove equivalence, guarantee a unique representation, or establish universal semantic preservation.
The following identities capture common equivalences:

- **Identity of pairs**: for every `E`, `E` is equivalent both to `m![{ E }, 1]` and `m![1, { E }]`.
- **Stride-modulo decomposition**: for every `E` whose size `E::SIZE` is divisible by `n`, `E` and `m![{ E } / n, { E } % n]` are equivalent.
- **Pair projection**: for every `A` and `B`, `m![[{ A }, { B }] / B::SIZE]` is equivalent to `m![A]` and `m![[{ A }, { B }] % B::SIZE]` is equivalent to `m![B]`.
- **Associativity of pairs**: for every `E1`, `E2`, `E3`, `m![{ E1 }, { E2 }, { E3 }]`, `m![[{ E1 }, { E2 }], { E3 }]`, and `m![{ E1 }, [{ E2 }, { E3 }]]` are equivalent.
- **Idempotent operations**: for every `E`, `E` is equivalent to `m![{ E } / 1]`, to `m![{ E } # E::SIZE]`, and to `m![{ E } = E::SIZE]`.
- **Modulo by 1**: for every `E`, `m![E % 1]` is equivalent to the identity mapping `m![1]`.
