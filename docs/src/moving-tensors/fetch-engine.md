# Fetch Engine

The Fetch Engine reads DM tensors and produces packet streams for the Tensor Unit in two stages:
- **[Fetch Sequencer](#fetch-sequencer)**: A [mathematical tensor move](../mapping-tensors/tensor-semantics.md#mathematical-tensor-move) that reads DM with per-slice sequencers and produces a packet stream.
- **[Fetch Adapter](#fetch-adapter)**: Applies optional element-wise transformations that may change the mathematical tensor.

## Interface

`BeginTensor` represents a tensor resident in DM, at the entry of the Tensor Unit pipeline.
Its `Time` is `m![1]` (no temporal iteration before the pipeline starts) and `Packet` is the element layout in DM.

`BeginTensor::fetch()` converts the input into a `FetchTensor` packet stream.
The `assert_eq!` calls enforce hardware constraints on `Cluster::SIZE`, `Slice::SIZE`, and packet alignment (see [Constraints](#constraints)).

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/fetch.rs:fetch_impl}}
```

As introduced in [Mapping Tensors](../mapping-tensors/index.md), the `Chip`, `Cluster`, `Slice`, `Time`, `Packet` mapping distributes data across space and time.
`.fetch()` preserves the `Chip`, `Cluster`, and `Slice` dimensions unchanged from the input, because each slice independently reads its own DM partition.
Later the [Switch Engine](../computing-tensors/switch-engine.md) changes the `Slice` mapping by moving data across slices.

The kernel writer chooses the output type parameters `D2`, `Time2`, and `Packet2` to configure both the Fetch Sequencer and the Fetch Adapter.
The compiler derives all hardware settings from the output types.
`D2` enables type casting such as `i8` to `i32`.
`Time2` sets the number of time steps in the output stream.
`Packet2` sets the element layout within each packet.
For performance implications of `Packet2` choices, see [Optimizations](#optimizations).

The following example fetches an `i8` matrix from DM, casting each element to `i32`.
The output `FetchTensor` streams 512 time steps, each a 32-element `i32` packet (128 bytes).
Here `D2 = i32`, `Time2 = m![A]`, and `Packet2 = m![B]`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![CH = 4, CL = 2, S = 256, A = 512, B = 32];

fn fetch_matrix_example<'l, const T: Tu>(
    input: BeginTensor<'l, T, i8, m![CH], m![CL], m![S], m![1], m![A, B]>,
) -> FetchTensor<'l, T, i32, m![CH], m![CL], m![S], m![A], m![B]> {
    input.fetch()
}
```

## Fetch Sequencer

`Chip`, `Cluster`, and `Slice` are the hardware spatial parallelism dimensions.
A Fetch Sequencer runs independently in every slice, each operating on its own local DM partition.
The example in [Interface](#interface) expresses this: `Chip = m![CH]`, `Cluster = m![CL]`, and `Slice = m![S]` (with `CH = 4`, `CL = 2`, `S = 256`) reflect a 4-chip RNGD system with 2 clusters per chip and 256 slices per cluster (2,048 slices total), each running the same sequencer pattern on its own `A×B` sub-tensor.


### Constraints

- **Hardware dimensions**: `Chip::SIZE`, `Cluster::SIZE`, and `Slice::SIZE` must match the hardware configuration (see [Sequencer](./sequencer.md#architecture)).

### Multi-Read Packet

Preparing a packet may require multiple hardware reads because packet axes may not be contiguous in DM, and the hardware reads at most 32 bytes at once.
In the [main-context](../computing-tensors/index.md#scheduling-execution-contexts), `read_size` is the largest divisor of the sequencer's `access_size` (see [Sequencer Architecture](./sequencer.md#access-size) for `access_size`) such that `D[read_size]` is 1, 2, 4, 8, 16, or 32 bytes.
In the [sub-context](../computing-tensors/index.md#scheduling-execution-contexts), `read_size` is fixed at 8 bytes.
The compiler derives `read_size` from the input and output type of `fetch()` and users do not set it directly.
Multi-read occurs whenever `Packet::SIZE > read_size`.
For example, a 24-byte packet in the main-context forces `read_size = 8` and 3 reads per packet.
The total cycle count is `Time::SIZE * (Packet::SIZE / read_size)`.

The following examples fetch the same `i4` tensor of shape `m![N, C, H, W]` (with `N=4, C=3, H=4, W=8`) using four different `Packet2` choices.
```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![N = 4, C = 3, H = 4, W = 8];

/// Sequencer config: [N = 4 : 96, C = 3 : 32, H = 4 : 8, W = 8 : 1].
/// access_size = 8; read_size = 8 (4 bytes); reads per packet = 1; cycles = 48
fn fetch_batch_1<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1], m![1], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1], m![1], m![N, C, H], m![W]> {
    input.fetch()
}

/// Sequencer config: [N = 4 : 96, C = 3 : 32, H / 2 = 2 : 16, H % 2 = 2 : 8, W = 8 : 1].
/// access_size = 16; read_size = 16 (8 bytes); reads per packet = 1; cycles = 24
fn fetch_batch_2<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1], m![1], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1], m![1], m![N, C, H / 2], m![H % 2, W]> {
    input.fetch()
}

/// Sequencer config: [N = 4 : 96, C = 3 : 32, H = 4 : 8, W = 8 : 1].
/// access_size = 32; read_size = 32 (16 bytes); reads per packet = 1; cycles = 12
fn fetch_batch_3<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1], m![1], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1], m![1], m![N, C], m![H, W]> {
    input.fetch()
}

/// Sequencer config: [N = 4 : 96, C = 3 : 32, H = 4 : 8, W = 8 : 1].
/// access_size = 96; read_size = 32 (16 bytes); reads per packet = 3; cycles = 12
fn fetch_batch_4<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1], m![1], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1], m![1], m![N], m![C, H, W]> {
    input.fetch()
}
```

### Interleaving

Interleaving combines two tensors with identical mappings into a single sequencer operation, reducing overhead when both tensors are needed for the same computation.
An explicit `Time` axis encodes alternation between the two tensors.

In the following example, the main-context creates an interleaved tensor using `begin_interleaved()`.
The first temporal iteration fetches from `lhs`, the second from `rhs`, the third from `lhs` again, and so on.
At most two tensors can be interleaved in a single fetch operation.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 32, I = 2];

/// Interleaves two input tensors into a single packet stream.
/// Useful for operations like 'input1 + input2' in the Vector Engine.
/// The interleaved BeginTensor is created via Tu.begin_interleaved().
/// The `I = 2` axis in Time encodes alternation between the two tensors.
fn fetch_interleaved<'l>(
    ctx: &'l mut Context,
    lhs: &'l DmTensor<i8, m![1], m![1], m![1], m![A, B]>,
    rhs: &'l DmTensor<i8, m![1], m![1], m![1], m![A, B]>,
) -> FetchTensor<'l, { Tu::Main }, i8, m![1], m![1], m![1], m![A, I], m![B]> {
    ctx.main.begin_interleaved::<I, _, _, _, _, _>(lhs.view(), rhs.view()).fetch()
}
```

### Optimizations

Three factors determine Fetch Sequencer throughput.

- **Input bandwidth**: `read_size` is limited by axis contiguity in DM and packet size.
  Non-adjacent axes reduce `access_size` and therefore `read_size` (see [Non-Contiguous Packets](./sequencer.md#non-contiguous-packets)).
  A packet smaller than the contiguous run also limits `read_size`.
  Padding to a larger power-of-two raises it (see [Packet padding](#example-packet-padding)).

  Furthermore, access patterns that hit the same bank 64 or more times consecutively starve the lower-priority [Commit Engine](./commit-engine.md) and [DMA Engine](./dma-engine.md) and can cause catastrophic NoC timeouts.

  See [Memory Performance](./memory-performance.md) for details.
- **Output bandwidth**: the downstream [Collect Engine](../computing-tensors/collect-engine.md) converts Fetch's packets to 32-byte *flits*, so packet sizes that don't align to 32 bytes waste bandwidth.
  A 20-byte packet fills one flit with 12 bytes of zero-padding, wasting `12 / 32 = 37.5%`.
  A 40-byte packet spans two flits (64 bytes total) and zero-pads the final 24 bytes of the second flit, wasting `24 / 64 = 37.5%`.
- **Spatial parallelism**: Distributing fetches across slices maximizes throughput.

#### Example: Packet padding

Padding `Packet2` to a larger power-of-two element count can increase `read_size`.
The three examples below fetch the same 30-byte tensor in 15, 3, and 1 cycles by growing the packet from 2 to 16 to 32 bytes:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 3, B = 5, C = 2];

/// Smallest packet: only C dimension (2 bytes). Takes 15 cycles.
fn fetch_packet_C<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1], m![1], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1], m![1], m![A, B], m![C]> {
    input.fetch()
}

/// Medium packet: B and C dimensions padded to 16 bytes. Takes 3 cycles.
fn fetch_packet_BC<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1], m![1], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1], m![1], m![A], m![[B, C] # 16]> {
    input.fetch()
}

/// Largest packet: all dimensions padded to 32 bytes. Takes 1 cycle.
fn fetch_packet_ABC<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1], m![1], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1], m![1], m![1], m![[A, B, C] # 32]> {
    input.fetch()
}
```

In these examples, padding reads beyond the actual data, but this is safe because padding values do not affect computation.
Different padding strategies produce different `FetchTensor` mappings, which may affect downstream components.

## Fetch Adapter

The Fetch Adapter applies element-wise transformations to sequencer packets through four stages: [masking](#masking), [table indexing](#table-indexing), [type casting](#type-casting), and [zero-point subtraction](#zero-point-subtraction).
The [main-context](../computing-tensors/index.md#scheduling-execution-contexts) adapter supports all four stages, while [sub-context](../computing-tensors/index.md#scheduling-execution-contexts) adapters only support zero-point subtraction.


### Masking

Masking overwrites padded elements with a neutral value so they do not influence downstream operations.
Padding is necessary because the Tensor Unit's internal data paths operate on fixed-width units (32-byte flits of 8 elements at 32 bits each), so any axis whose size is not a multiple of the flit width must be rounded up (for example, a 63-element axis is rounded up to 64).
Without masking the padded slots hold arbitrary values, which corrupts reductions like `sum` or `max`.
For instance, the Packet Reducer sums along an axis, and the padded slots must be zero for a `sum` to be correct over the valid range.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 63];

/// Pads 63 elements to 64 and masks the 64th to zero,
/// so reductions compute correctly over the 63 valid elements.
fn fetch_with_masking<'l, const T: Tu>(
    input: BeginTensor<'l, T, i8, m![1], m![1], m![1], m![1], m![A]>,
) -> FetchTensor<'l, T, i8, m![1], m![1], m![1], m![1], m![A # 64]> {
    input.fetch()
}
```

#### Masking configuration

The Fetch Engine masks innermost axes padded on both sides, written `(# n + A + # m)` where `n` is left padding, `A` is valid data, and `m` is right padding.
The hardware exposes three cases covering different padding shapes, all configured through the same three parameters:

- **`last_dim`**: dimension index that masking applies to.
  Index 0 is the innermost dimension.
- **`left_pad`**: zeroes the first `left_pad` elements when `last_dim` points to the innermost dimension (index 0).
- **`last_dim_rightmost_valid_count[0]`**: zeroes `dim0 - last_dim_rightmost_valid_count[0]` elements from the right when `last_dim` is the outermost active dimension.
  Capped at 0-255 for 4-bit types and 0-31 for `f32`, because the final packet must stay within 256 bytes.


##### Example (padding case 1)

Case 1 is the basic shape: one contiguous left pad and one contiguous right pad along the innermost axis.

![PADDING_1](./fetch-engine-padding-1.png)

- `axes![A = 32, B = 90]`
- `dtype = i8`
- `base_addr = 0`
- `Element = m![A, B # 96]`
- Configuration: `last_dim = 1`, `lpad = 2`, `last_dim_rightmost_valid_count[0] = 4`, `pad_value = 0`
- Stream mapping: `let B' = # 2 + B + # 4 in { Time: m![A, B' / 32], Packet: m![B' % 32] }`
- Sequencer configuration: `[A = 32 : 96, B' / 32 = 3 : 32, B' % 32 = 32 : 1] : 32 @ base_addr = -2`
- Packet size: `m![B' % 32]::SIZE = 32`
- Cycles: `Time::SIZE = m![A, B' / 32]::SIZE = 32 * 3 = 96`
- Result: The first 2 and last 4 values of `(# 2 + B + # 4)` are masked to `0`.

##### Example (padding case 2)

Case 2 applies the same masking as Case 1, but the padded regions are split rather than contiguous.

![PADDING_2](./fetch-engine-padding-2.png)

- `axes![A = 32, B = 90]`
- `dtype = i8`
- `base_addr = 0`
- `Element = m![A, B # 96]`
- Configuration: `last_dim = 0`, `lpad = 2`, `last_dim_rightmost_valid_count[0] = 4`, `pad_value = 0`
- Stream mapping: `let B' = # 2 + B + # 4 in { Time: m![B' / 32, A], Packet: m![B' % 32] }`
- Sequencer configuration: `[B' / 32 = 3 : 32, A = 32 : 96, B' % 32 = 32 : 1] : 32 @ base_addr = -2`
- Packet size: `m![B' % 32]::SIZE = 32`
- Cycles: `Time::SIZE = m![A, B' / 32]::SIZE = 32 * 3 = 96`
- Result: The first 2 and last 4 values of `(# 2 + B + # 4)` are masked to `0`.

##### Example (padding case 3)

Case 3 lifts the right-padding limit from Cases 1 and 2 (255 * 4-bit) by giving each entry index `i` its own `last_dim_rightmost_valid_count[i]`.
It supports `last_dim_rightmost_valid_count[0..8]` when the axis size is 8 or less.

![PADDING_3](./fetch-engine-padding-3.png)

- `axes![A = 32, B = 97]`
- `dtype = f32`
- `base_addr = 0`
- `Element = m![A, B # 128]`
- Stream mapping: `let B' = B + # 31 in { Time: m![A, B' / 16, 1], Packet: m![B' % 16] }`
- Sequencer configuration: `[A = 32 : 128, B' / 16 = 8 : 16, 1 = 1 : 0, B' % 16 = 16 : 1] : 16 @ base_addr = -2`
- Packet size: `m![B' % 16]::SIZE = 16`
- Cycles: `Time::SIZE = m![A, B' / 32]::SIZE = 32 * 8 = 256`
- Configuration: `last_dim_rightmost_valid_count_dim = 1`, `last_dim = 2`, `last_dim_rightmost_valid_count[0..8] = [16, 16, 16, 16, 16, 16, 1, 0]`, `pad_value = 0`
- Result: Of `(B # 31)`, 97 elements are valid and 31 are masked as invalid.

### Table Indexing

Table indexing provides hardware-accelerated lookup tables during the fetch stage.
Each value is treated as an index into a pre-configured table, and the corresponding table entry is output instead.
This is useful for operations that cannot be efficiently implemented with standard arithmetic, such as non-linear activation functions like Sigmoid and GeLU, or quantization schemes that use custom encoding tables.
This enables:

- **Non-linear activations**: Implements Sigmoid, GeLU, and other functions through pre-computed lookup tables.
- **Custom type casting**: Translates specialized encodings like `MXFP4` to standard formats using conversion tables.


```rust,ignore
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8];

/// Fetches with table lookup: each input value indexes into a pre-configured table.
/// Input [0, 1, 2, 3, 4, 5, 6, 7] with table[x] = 2*x
/// Output [0, 2, 4, 6, 8, 10, 12, 14]
fn fetch_with_table<'l, const T: Tu>(
    input: BeginTensor<'l, T, i8, m![1], m![1], m![1], m![1], m![A]>,
    table: &LookupTable<i8, i8>,
) -> FetchTensor<'l, T, i8, m![1], m![1], m![1], m![1], m![A]> {
    input.fetch_with_table(table)
}
```



### Type Casting

The Fetch Adapter converts element types while streaming data from DM, enabling computation on data stored at lower precision than the compute pipeline requires.
Type casting adds 1–2 cycles of latency.

RNGD supports the following conversions:

| Input | Output |
|-------|--------|
| `i4` | `i5`, `i32` |
| `i8` | `i9`, `i32` |
| `i16` | `i32` |
| `f8e4m3` | `f32` |
| `f8e5m2` | `f32` |
| `bf16` | `f32` |
| `f16` | `f32` |
| `f32` | `bf16` |

RNGD-S supports the following additional type conversions:

| Input | Output |
|-------|--------|
| `i4` | `i9` |
| `i16` | `i9` |
| `f8e4m3` | `bf16` |
| `f8e5m2` | `bf16` |

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8];

/// Fetches with type casting: converts i8 storage to i32 for computation.
/// Input:   i8 [0, 1, 2, 3, 4, 5, 6, 7]
/// Output: i32 [0, 1, 2, 3, 4, 5, 6, 7]
fn fetch_with_type_cast<'l, const T: Tu>(
    input: BeginTensor<'l, T, i8, m![1], m![1], m![1], m![1], m![A]>,
) -> FetchTensor<'l, T, i32, m![1], m![1], m![1], m![1], m![A]> {
    input.fetch()
}
```

Type casting adds an additional limit on `read_size`: the cast output per fetch must fit in a single 32-byte flit (see [Collect Engine](../computing-tensors/collect-engine.md)).

- **Valid**:
  - `i4`→`i32`, `read_size = 8 (4 bytes)`: produces 8 × 4 = 32 B
  - `i8`→`i32`, `read_size = 8 (8 bytes)`: produces 8 × 4 = 32 B
- **Invalid**:
  - `i4`→`i32`, `read_size = 16 (8 bytes)`: produces 16 × 4 = 64 B
  - `i8`→`i32`, `read_size = 16 (16 bytes)`: produces 16 × 4 = 64 B

### Zero-Point Subtraction

When converting from quantized integers to computation types, the hardware can simultaneously subtract a zero-point offset.
This is useful for asymmetric quantization, which represents data ranges more efficiently than symmetric quantization but requires subtracting the zero-point before computation.
The zero-point value must fit within the input quantized type's representable range (for example, an `i8` zero-point lies in `-128..=127`).


```rust,ignore
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8];

/// Fetches with zero-point subtraction for asymmetric quantization.
/// Input:   i8 [0, 1, 2, 3, 4, 5, 6, 7], with zero_point = 10
/// Output: i32 [-10, -9, -8, -7, -6, -5, -4, -3]
fn fetch_with_zero_point<'l, const T: Tu>(
    input: BeginTensor<'l, T, i8, m![1], m![1], m![1], m![1], m![A]>,
    zero_point: i8,
) -> FetchTensor<'l, T, i32, m![1], m![1], m![1], m![1], m![A]> {
    input.fetch_with_zero_point(zero_point)
}
```

Interleaving fetches enable subtracting different zero points from each tensor.
This is useful when combining tensors with different quantization parameters.

```rust,ignore
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, I = 2];

/// Fetches interleaved tensors with different zero points per tensor.
/// Input1: [0, 1, 2, 3, 4, 5, 6, 7], with zero_point = 100
/// Input2: [0, 1, 2, 3, 4, 5, 6, 7], with zero_point = -100
/// Output interleaved: [-100, -99, ..., 100, 101, ...]
fn fetch_interleaved_with_zero_points<'l, const T: Tu, const I: Ident>(
    input: BeginTensor<'l, T, i8, m![1], m![1], m![1], m![I], m![A]>,
    zero_points: [i8; 2],
) -> FetchTensor<'l, T, i32, m![1], m![1], m![1], m![I], m![A]> {
    input.fetch_with_zero_points(zero_points)
}
```

