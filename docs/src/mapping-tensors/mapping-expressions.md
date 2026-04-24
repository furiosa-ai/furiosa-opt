# Mapping Expressions
A mapping expression defines where each tensor element sits in a buffer.
This page covers the available mapping constructors and the equivalences between mappings.

Consider a tensor with `axes![A = 8, B = 512]`.
The mapping expression `m![A, B]` places \\(A\\) as the major axis and \\(B\\) as the minor axis, requiring a buffer of `8 × 512 = 4096` elements.
Buffer position 0 holds \\(\\{A=0, B=0\\}\\), position 1 holds \\(\\{A=0, B=1\\}\\), and so on through all 512 elements where \\(A=0\\) before moving to \\(A=1\\).


## Axis Sizes

The `axes!` macro declares axis identifiers and their sizes.
Throughout this section, assume the following axis sizes.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, B = 512];
```

## Mapping Interface

A mapping expression like `m![H, W]` is a Rust type that describes how tensor indices map to buffer positions.
Every mapping expression implements the `M` trait, which provides the buffer size and a buffer-index-to-tensor-index mapping function:

```rust
// Inside `furiosa_visa_std::prelude`...
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::fmt::Debug;
{{#include ../../../furiosa-mapping/src/mapping.rs:trait_m}}

/// Tensor index: a map from axis identifiers to coordinate values.
pub struct Index { /* ... */ }

/// Constructs tensor indices.
/// `i![A: 2, B: 3]` creates an `Index` with A = 2 and B = 3.
macro_rules! i {
    # () => {};
    /* ... */
}
```


A mapping defines what mathematical tensor a buffer represents.
For example, `HostTensor<bf16, m![A, B]>` denotes a host memory buffer containing `m![A, B]::SIZE` elements of `bf16` data, which is 4096 elements.
We say a buffer *holds* a tensor \\(T\\) when:
- For every buffer index `i` and tensor index `ti`,
- if `m![A, B]::map(i) = Some(ti)`,
- then the `i`-th element of the buffer stores the value of tensor \\(T\\) at index `ti`.


## Constructors

Mapping expressions are built by composing small constructors, each of which transforms or combines simpler mappings.
These expressions use arithmetic-like operators (`/`, `%`, and `#` for padding) to concisely define the mapping between tensor and linear buffer indices.

### Symbol

A symbol is a single uppercase letter whose size comes from the shape declaration.
The mapping `m![A]` maps 8 buffer indices linearly to tensor indices along the axis: buffer index `0` holds `i![]` (empty tensor index), index `1` holds `i![A: 1]`, index `2` holds `i![A: 2]`, and so on:

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
#
# axes![A = 8];
#
type E = m![A]; // Symbol<Ident::A, 8>

#[test]
fn test_symbol() {
    for i in 0..E::SIZE {
        assert_eq!(E::map(i), Some(i![A: i]));
    }
    assert_eq!(E::map(E::SIZE), None);
}
```

```rust,ignore
// Trait implementation
{{#include ../../../furiosa-mapping/src/mapping.rs:symbol_impl}}
```

> [!NOTE]
> For every symbol `A`, the 0'th index `i![A: 0]` corresponds to the empty tensor index `i![]`.

### Pair

One way to store a 2D tensor with shape \\(\\{A=8, B=512\\}\\) is the pair mapping `m![A, B]`.
This creates a buffer of 4096 elements where `A` is the major axis and `B` is the minor axis.
The first 512 elements hold `A = 0` and the next 512 elements hold `A = 1`.
Buffer index `519` holds `i![A: 1, B: 7]` since `519 == 512 * 1 + 7`.

The mapping `Pair<L, R>` maps the Cartesian product of two spaces into a linear buffer where `L` is the major dimension and `R` is the minor dimension.
The size is `L::SIZE * R::SIZE`, and the mapping uses floor division and modulo to decompose indices.
`m![A, B, C, D]` expands to `Pair<A, Pair<B, Pair<C, D>>>` and is right-associative.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
#
# axes![A = 8, B = 512];
#
type E = m![A, B]; // Pair<m![A], m![B]>

#[test]
fn test_pair() {
    for i in 0..E::SIZE {
        assert_eq!(E::map(i), Some(i![A: i / <m![B]>::SIZE, B: i % <m![B]>::SIZE]));
    }
    assert_eq!(E::map(2 * <m![B]>::SIZE + 7), Some(i![A: 2, B: 7]));
    assert_eq!(E::map(E::SIZE), None);
}
```

```rust,ignore
// Trait implementation
{{#include ../../../furiosa-mapping/src/mapping.rs:pair_impl}}
```

### Identity

The identity mapping `m![1]` creates a single-element buffer that maps buffer index `0` to the empty tensor index `i![]`.
It serves as the identity element for `Pair`: `m![1, A]` and `m![A, 1]` are both equivalent to `m![A]`.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
#
type E = m![1]; // Identity

#[test]
fn test_identity() {
    assert_eq!(E::map(0), Some(i![]));
    assert_eq!(E::map(1), None);
}
```

```rust,ignore
// Trait implementation
{{#include ../../../furiosa-mapping/src/mapping.rs:identity_impl}}
```

### Padding

Padding aligns data to hardware requirements by adding unused buffer space.
For example, the DMA engine requires rows to start on 64-byte boundaries.
With `axes![C = 13, D = 61]`, `m![C, D]` creates misaligned rows since `61` is not divisible by `64`.
`m![C, D # 64]` fixes this by aligning each row to 64-byte boundaries, using 3 extra elements per row.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
#
axes![C = 13, D = 61];

type E = m![C, D # 64]; // Pair<m![C], Padding<m![D], 64>>

#[test]
fn test_padding() {
    assert_eq!(E::map(0),  Some(i![C: 0, D: 0]));
    assert_eq!(E::map(60), Some(i![C: 0, D: 60]));
    assert_eq!(E::map(61), None); // padding
    assert_eq!(E::map(62), None); // padding
    assert_eq!(E::map(63), None); // padding
    assert_eq!(E::map(64), Some(i![C: 1, D: 0]));
}
```

```rust,ignore
// Trait implementation
{{#include ../../../furiosa-mapping/src/mapping.rs:padding_impl}}
```

### Resize

Resize constrains a mapping to a smaller logical size by truncating indices beyond the new size, discarding elements outside that range. Unlike padding, which expands the buffer, Resize shrinks the logical view.
The mapping `m![D = 2]` takes only the first 2 elements of axis `D`, producing indices `D = 0` and `D = 1`.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
#
axes![C = 2, D = 3];
type E = m![C, D = 2]; // Pair<m![C], Resize<m![D], 2>>

#[test]
fn test_resize() {
    assert_eq!(E::map(0), Some(i![C: 0, D: 0]));
    assert_eq!(E::map(1), Some(i![C: 0, D: 1]));
    assert_eq!(E::map(2), Some(i![C: 1, D: 0]));
    assert_eq!(E::map(3), Some(i![C: 1, D: 1]));
    assert_eq!(E::map(4), None);
}
```

```rust,ignore
// Trait implementation
{{#include ../../../furiosa-mapping/src/mapping.rs:resize_impl}}
```

Tiling is implemented through *indexed views*, pure metadata transformations without data copies.
The `.tile()` method extracts a tile by resizing one dimension to the tile size and offsetting into the buffer.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
#
# axes![A = 8, B = 512];
#
# fn tiles() {
let tensor = unsafe { HbmTensor::<bf16, m![1], m![A, B]>::from_addr(0) };
let view = tensor.view(); // HbmTensorView::<'_, bf16, m![1], m![A, B]>
let tile01 = view.tile::<m![B], 2, m![A, B = 2 # 4]>(0); // HbmTensorView::<'_, bf16, m![1], m![A, B = 2 # 4]>
let tile23 = view.tile::<m![B], 2, m![A, B = 2 # 4]>(2); // HbmTensorView::<'_, bf16, m![1], m![A, B = 2 # 4]>
# }
```

The `.tile()` method takes three type parameters and one value parameter.
- The *tile dimension* `m![B]` specifies which dimension to divide along.
- The *tile size* `2` specifies the number of elements per tile.
- The *tile mapping* `m![A, B = 2 # 4]` defines the resulting view's mapping. The mapping `B = 2 # 4` signifies that dimension `B` has a logical size of `2` within the view but exists within a physical footprint of `4`. This is essential for preserving the original memory layout and stride calculations.
- The *starting index* specifies which tile to extract. Passing `0` captures the range `0..2` for `tile01`, while passing `2` captures the range `2..4` for `tile23`.

### Stride and Modulo

Stride (`/`) and modulo (`%`) decompose a single dimension into two: the outer (block index) and the inner (position within block).
Consider the 512-element axis `B` divided into 8 blocks of 64 elements each.
The mapping `m![B / 64, B % 64]` creates an 8 × 64 grid where the first dimension selects which block and the second dimension selects the position within that block.
Buffer index `130` corresponds to block `2` at position `2` within that block, giving tensor index `B = 64 × 2 + 2 = 130`, equal to the flat-buffer result (since `m![B / 64, B % 64]` is equivalent to `m![B]`):

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# axes![A = 8, B = 512];
type D1 = m![B / 64]; // stride with size 8
type D2 = m![B % 64]; // modulo with size 64

type E = m![B / 64, B % 64]; // equivalent to `m![B]`

#[test]
fn test_stride_modulo() {
    for i in 0..8 {
        assert_eq!(D1::map(i), Some(i![B / 64: i]));
    }
    assert_eq!(D1::map(8), None);

    for j in 0..64 {
        assert_eq!(D2::map(j), Some(i![B % 64: j]));
    }
    assert_eq!(D2::map(64), None);

    for i in 0..8 {
        for j in 0..64 {
            assert_eq!(
                E::map(64 * i + j),     // i![B / 64: i, B % 64: j]
                <m![B]>::map(64 * i + j), // equivalent to above
            );
        }
    }
    assert_eq!(E::map(512), None);
}
```

```rust,ignore
// Trait implementation
{{#include ../../../furiosa-mapping/src/mapping.rs:stride_impl}}

{{#include ../../../furiosa-mapping/src/mapping.rs:modulo_impl}}
```

Together, `m![B / 64, B % 64]` transforms axis `B` into an 8 × 64 grid.
The mapping is equivalent to `m![B]` but expresses a different logical view of the same data, revealing block structure hidden in the flat representation.

Stride and modulo mappings can be visualized in tabular form. Consider the mapping `m![B / 4, B % 4]` with `B::SIZE = 16`. The following table shows how buffer indices are arranged: each row corresponds to a specific index of `B / 4` (the stride axis), and each column corresponds to an index of `B % 4` (the modulo axis):

|                 | `i![B % 4: 0]` | `i![B % 4: 1]` | `i![B % 4: 2]` | `i![B % 4: 3]` |
| --------------- | -------------- | -------------- | -------------- | -------------- |
| `i![B / 4: 0]` | `i![B: 0]`     | `i![B: 1]`     | `i![B: 2]`     | `i![B: 3]`     |
| `i![B / 4: 1]` | `i![B: 4]`     | `i![B: 5]`     | `i![B: 6]`     | `i![B: 7]`     |
| `i![B / 4: 2]` | `i![B: 8]`     | `i![B: 9]`     | `i![B: 10]`    | `i![B: 11]`    |
| `i![B / 4: 3]` | `i![B: 12]`    | `i![B: 13]`    | `i![B: 14]`    | `i![B: 15]`    |

Stride and modulo factorize a single mapping into multiple dimensions.
The expression `m![B / n]` creates an outer dimension indexing blocks of size `n`.
The expression `m![B % n]` creates an inner dimension indexing positions within each block.

Modulo differs from resize in how it handles buffer size:
- Resize shrinks the buffer by truncating indices beyond the new size.
- Modulo preserves the original buffer size while partitioning it into equal-sized blocks.

These operations can be nested for complex decompositions.
The following example splits `B` into three dimensions where the buffer's bit layout differs from that of the tensor index.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# axes![A = 8, B = 512];
// B's bits: 6 - 8,  0 - 4,          5
// Values:   0 - 7, 0 - 31,      0 - 1
type E = m![B / 64, B % 32, B / 32 % 2];

#[test]
fn test_nested_stride() {
    for i in 0..8 {
        for j in 0..32 {
            for k in 0..2 {
                assert_eq!(
                    E::map(64 * i + 2 * j + k),
                    Some(i![B: 64 * i + j + 32 * k]),
                );
            }
        }
    }
    assert_eq!(E::map(512), None);
}
```

The buffer index decomposes as `64 * i + 2 * j + k` where `i` selects the block, `j` selects position within the block, and `k` selects the sub-block.
The tensor index `B` reconstructs as `64 * i + j + 32 * k`, which rearranges the bit positions.

For example, buffer index `67` maps to `B = 97`:
- Buffer `67 = 64 * 1 + 2 * 1 + 1` gives `i=1, j=1, k=1`
- Tensor `B = 64 * 1 + 1 + 32 * 1 = 97`
- Verify: `97 / 64 = 1`, `97 % 32 = 1`, `(97 / 32) % 2 = 1`

This kind of bit rearrangement maps naturally to hardware memory layouts where address bits are reordered for bank interleaving or cache efficiency.
In binary, this rearranges bit positions: buffer `001_00001_1` becomes `B = 001_1_00001`.
The buffer groups bits as `[8:6]_[5:1]_[0]` while `B` groups them as `[8:6]_[5]_[4:0]`.

Tiling can operate on blocks rather than individual elements.
The following example tiles by block using `m![B / 32]` and creates overlapping tiles:

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# axes![A = 8, B = 512];
let tensor = unsafe { HbmTensor::<bf16, m![1], m![A, B]>::from_addr(0) };
for i in 0..15 {
    let tile = tensor.view().tile::<m![B / 32], 2, m![A, B / 32 = 2 # 16, B % 32]>(i);
}
```

With `B = 512`, the dimension `B / 32` has 16 blocks numbered 0-15.
Each tile takes 2 consecutive blocks starting at index `i`.
Tile 0 covers blocks `{0, 1}`, tile 1 covers blocks `{1, 2}`, and so on through tile 14 covering blocks `{14, 15}`.
These tiles overlap because consecutive tiles share one block: tiles 0 and 1 both include block 1.

The tile mapping `B / 32 = 2` resizes the block dimension to 2 since each tile contains exactly 2 blocks.
When tiling with a single block, `B / 32 = 1` simplifies to the identity `m![1]` since the dimension has only one value.

### Escape

For complex mappings, define type aliases and reference them using `{ ... }`.
With separate mappings `L = m![A]` and `R = m![B]`, combining them as `m![{ L }, { R }]` produces the same result as `m![A, B]`:

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# axes![A = 8, B = 512];
type L = m![A];
type R = m![B];
type E = m![{ L }, { R }]; // equivalent to `m![A, B]`
```

This escape syntax breaks down complex mappings into named, reusable components.


### Advanced Constructors

<!-- > **TODO** (jeongmin.park): Need input on how to write this section. -->

#### Skewed Axis

A skewed axis creates a diagonal access pattern across two dimensions.
Skewed axes introduce derived axis labels defined by arithmetic differences between existing axes; for example, `B' = B - A` defines a new axis `B'` whose coordinate at any point equals `B` minus `A`.
Algorithms that process data along diagonals use this pattern, such as certain wavefront computations.

The expression `m![A, B' = 4]` with `B' = B - A` creates a mapping where each row is shifted relative to the previous one.
The `=` operator specifies the logical size after skewing. The result wraps around using modular arithmetic.

For example, with `axes![A = 4, B = 4]` and `B' = B - A`:

| (A, B') | (A, B) |
|---------|--------|
| (0, 0)  | (0, 0) |
| (0, 1)  | (0, 1) |
| (0, 2)  | (0, 2) |
| (0, 3)  | (0, 3) |
| (1, 0)  | (1, 1) |
| (1, 1)  | (1, 2) |
| (1, 2)  | (1, 3) |
| (1, 3)  | (1, 0) |

When `A = 1` and `B' = 3`, the original `B` coordinate wraps to `0` via modular arithmetic since `B = (B' + A) % 4 = (3 + 1) % 4 = 0`.

#### Indirect Sequencing

> **TODO**: Document indirect sequencing patterns for non-contiguous memory access. This advanced constructor provides index-based memory access where the sequence of buffer positions is determined by an indirection table rather than a mathematical formula.

#### Sliding (Linear Combination)

> [!NOTE]
> Linear combination expressions `$(e1:n1, ..., ed:nd)` combine multiple dimensions with specified strides. Formal definition: `size_S($(e1:n1, ..., ed:nd)) = 1 + sum_k((size_S(ek) - 1) * nk)`. The mapping `S, $(e1:n1, ..., ed:nd) |- si ~ ti` holds if there exist `si1...sid, ti1...tid` such that for all `k`: `S, ek |- sik ~ tik`, `si = sum_k(sik * nk)`, and `ti = sum_k(tik * nk)`.
>
> Linear combinations can encode outer sum: `e1 * e2` is equivalent to `$(e1 : size_S(e2), e2 : 1)`. However, outer sum is preferred because it's more resilient to axis reordering. Changing `e1 * e2` to `e2 * e1` doesn't require manual stride updates.

Sliding operations access overlapping data blocks, essential for convolutional neural networks.
Consider a buffer of 9 elements representing a tensor with shape \\(\\{N=5, F=3\\}\\) where each row is a 3-element slice that slides one element at a time.
The tensor element at \\((N, F)\\) maps to buffer index \\(N + 2F\\):

$$
\begin{array}{c|ccc}
  & F=0 & F=1 & F=2 \\\\
\hline
N=0 & 0 & 2 & 4 \\\\
N=1 & 1 & 3 & 5 \\\\
N=2 & 2 & 4 & 6 \\\\
N=3 & 3 & 5 & 7 \\\\
N=4 & 4 & 6 & 8 \\\\
\end{array}
$$

> [!NOTE]
> In this sliding pattern, a single space index can map to multiple tensor indices. For example, space index `4` maps to `{4_N}`, `{2_N, 1_F}`, and `{2_F}` simultaneously. This illustrates the non-one-to-one nature of `(S, e).maps(si, ti)`.

This can be expressed using a linear combination expression where the `N` axis has stride `1` and the `F` axis has stride `2`, yielding a total size of `1 + (5-1)*1 + (3-1)*2 = 9`.

<!-- > **TODO**: Define the Rust pseudocode syntax for linear combination expressions. -->
<!-- > -->
<!-- > Expected syntax format: -->
<!-- > ```text -->
<!-- > // Linear combination with specified strides -->
<!-- > m![$( N: 1, F: 2 )] -->
<!-- > // or similar notation to express: $(e1:n1, ..., ed:nd) -->
<!-- > ``` -->


## Equivalent Mapping

Mappings `E1` and `E2` are *equivalent* when:
- `E1::SIZE == E2::SIZE`
- For every `i`, `E1::map(i) == E2::map(i)`

The equivalence relation is reflexive, symmetric, and transitive. Examples:
* Identity of pairs: for every `E`, `E` is equivalent both to `m![{ E }, 1]` and `m![1, { E }]`.
* Stride-modulo decomposition: for every `E` whose size `E::SIZE` is divisible by `n`, `E` and `m![{ E } / n, { E } % n]` are equivalent.
* Pair projection: for every `A` and `B`, `m![[{ A }, { B }] / B::SIZE]` is equivalent to `m![A]` and `m![[{ A }, { B }] % B::SIZE]` is equivalent to `m![B]`.
* Associativity of pairs: for every `E1`, `E2`, `E3`, `m![{ E1 }, { E2 }, { E3 }]`, `m![[{ E1 }, { E2 }], { E3 }]`, and `m![{ E1 }, [{ E2 }, { E3 }]]` are equivalent.
* Idempotent operations: for every `E`, `E` is equivalent to `m![{ E } / 1]`, to `m![{ E } # E::SIZE]`, and to `m![{ E } = E::SIZE]`.
* Modulo by 1: For every `E`, `m![E % 1]` is equivalent to the identity mapping `m![1]`.
