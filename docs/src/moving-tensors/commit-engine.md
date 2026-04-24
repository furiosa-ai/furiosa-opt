# Commit Engine

The Commit Engine writes Tensor Unit results back to DM (Data Memory), the primary on-chip SRAM tier.
It implements a [logical tensor move](./index.md) from Tensor Unit streams to SRAM, writing each slice's result to its designated DM address.

After the Tensor Unit completes computation, results exist as streaming packets distributed across slices.
The Commit Engine transforms these packets through an adapter ([truncating](#truncating)) and writes them to DM via a [sequencer](#commit-sequencer).
This page covers the interface and examples, the adapter stages, the sequencer, sub-context operations, and performance guidelines.

## Interface

```rust,ignore
{{#include ../../../furiosa-visa-std/src/stream_tensor.rs:commit_impl}}
```

The Commit Engine mirrors the [Fetch Engine's](./fetch-engine.md) structure, but operates in reverse.

For detailed examples, see [kernel examples](../kernel-examples/fetch-commit-engine.md).

## Examples

Consider storing a matrix multiplication result `C = A * B` back to DM after computation.
The [Cast Engine](../computing-tensors/cast-engine.md) converts the Contraction Engine's `f32` packet elements to `bf16` to save space.
The Commit Engine stores the resulting tensor to DM.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![P = 256, M = 16, N = 8];

fn cast_commit<'l, const T: Tu>(
    input: AccumulationTensor<'l, T, f32, m![1], m![1], m![P], m![M], m![N]>,
) -> DmTensor<bf16, m![1], m![1], m![P], m![M, N # 16]> {
    // Cast f32 to bf16 (Cast Engine), then commit to DM (Commit Engine).
    // Input: M = 16 time steps, N = 8 f32 elements per packet (32 bytes).
    // After cast: N = 8 bf16 elements padded to 16 (32 bytes).
    // The sequencer writes across P = 256 slices.
    input.cast::<bf16, m![N # 16]>().commit(0)
}
```

## Adapter

The adapter transforms stream packets before writing to DM via [truncating](#truncating).

The [main context](../computing-tensors/index.md#execution-contexts) and [sub-context](../computing-tensors/index.md#execution-contexts) adapters both support truncating.
The sub-context is typically used for prefetching to TRF/VRF.


### Truncating

Truncating reduces packet size by keeping only the leading elements.
The input packet is always a full 32-byte flit.
The `commit_in_size` parameter controls how many bytes are actually written to DM: 8, 16, 24, or 32 bytes (where 32 bytes means no reduction).
This operation discards trailing elements or satisfies downstream alignment constraints.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![M = 4, K = 2, W = 8, N = 16, J = 64];

fn i8_padding_truncation<'l, const T: Tu>(
    input: CastTensor<'l, T, i8, m![1], m![1], m![1], m![M, K], m![W # 32]>,
) -> DmTensor<i8, m![1], m![1], m![1], m![M, K, W]> {
    // Input: 8 i8 elements padded to 32 (32 bytes per packet).
    // Truncation removes padding: only the 8 leading elements are written to DM.
    // commit_in_size = 8 elements × 1 byte = 8 bytes.
    input.commit(0)
}

fn f32_non_padding_truncation<'l, const T: Tu>(
    input: AccumulationTensor<'l, T, f32, m![1], m![1], m![1], m![M, K], m![W]>,
) -> DmTensor<f32, m![1], m![1], m![1], m![M, K, W = 4]> {
    // Input: 8 f32 elements (32 bytes per packet).
    // Truncation: only the first 4 elements are written to DM.
    // commit_in_size = 4 elements × 4 bytes = 16 bytes.
    input.commit(0)
}

fn bf16_truncation_with_transpose<'l, const T: Tu>(
    input: CastTensor<'l, T, bf16, m![1], m![1], m![1], m![M, K], m![N]>,
) -> DmTensor<bf16, m![1], m![1], m![1], m![K, M, N = 8]> {
    // Input: 16 bf16 elements (32 bytes per packet).
    // Truncation: only the leading 8 elements are written to DM.
    // commit_in_size = 8 elements × 2 bytes = 16 bytes.
    // Time is transposed: m![M, K] → m![K, M].
    input.commit(0)
}

fn i4_no_truncation_with_transpose<'l, const T: Tu>(
    input: CastTensor<'l, T, i4, m![1], m![1], m![1], m![M, K], m![J]>,
) -> DmTensor<i4, m![1], m![1], m![1], m![K, M, J]> {
    // Input: 64 i4 elements (32 bytes per packet).
    // No truncation: the full 32-byte packet is written to DM.
    // commit_in_size = 64 elements × 0.5 bytes = 32 bytes.
    // Time is transposed: m![M, K] → m![K, M].
    input.commit(0)
}
```

> [!NOTE]
> The `commit_in_size` value is automatically derived by the compiler from the output tensor mapping.
> It is not manually specified by the user.

<!--### Type Casting

> **TODO** (jeongmin.park): Is this section still relevant? If type casting here is legacy, should it be removed or kept for completeness/backward compatibility? When is it still used?

Type casting converts `f32` data to `bf16` format just before writing to DM.
This reduces storage requirements and optionally applies a `ReLU` activation function during conversion.

> [!NOTE]
> The current compiler pipeline performs most type conversions in the Cast Engine upstream.
> The Commit Engine's type casting and ReLU features are largely legacy functionality.

#### Standard Conversion (`f32` to `bf16`)

Standard conversion reduces precision from `f32` to `bf16` while preserving all values, including negative ones.

> **TODO** (pedro.lobo): Current API does not support legacy type casting.
> Remove `ignore` if we come up with an API for it, or delete the example if casting should only be handled in the Cast Engine.

```rust,ignore
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![N = 4, C = 3, H = 4, W = 8];

fn cast_commit<'l, const T: Tu>(
    input: AccumulationTensor<'l, T, f32, m![1], m![1], m![1], m![N, C, H], m![W]>,
) -> DmTensor<bf16, m![1], m![1], m![1], m![N, C, H, W # 16]> {
    // Cast f32 to bf16 (values preserved), then commit to DM.
    // W = 8 f32 elements (32 bytes) → 8 bf16 elements padded to 16 (32 bytes).
    input.commit_cast(0)
}
```

#### Conversion with ReLU (`f32` to `bf16`)

Conversion with ReLU applies the `ReLU` activation function during type conversion.
Negative values are clamped to zero, while non-negative values are converted to `bf16` unchanged.

> **TODO** (pedro.lobo): Current API does not support legacy type casting.
> Remove `ignore` if we come up with an API for it, or delete the example if casting should only be handled in the Cast Engine.

```rust,ignore
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![N = 4, C = 3, H = 4, W = 8];

fn cast_relu_commit<'l, const T: Tu>(
    input: AccumulationTensor<'l, T, f32, m![1], m![1], m![1], m![N, C, H], m![W]>,
) -> DmTensor<bf16, m![1], m![1], m![1], m![N, C, H, W # 16]> {
    // Cast f32 to bf16 with ReLU: negative values clamped to 0.0.
    // e.g., [-5.0, -0.1, 0.0, 3.7] → [0.0, 0.0, 0.0, 3.7]
    input.commit_cast_relu(0)
}
```-->

<!--### Chunking

Chunking splits a single input packet into multiple smaller writes.
This occurs when the output DM layout has stride discontinuities within `Packet`, such as padding between rows.
This disallows the packet from being written in a single contiguous operation.
Chunking is implicitly performed by the [Sequencer](#sequencer) and is not manually specified by the user.

The commit write size (`commit_in_size`) is determined by:

$$
\texttt{commit\_in\_size} = \gcd(\texttt{contiguous\_sram\_access\_size}, \texttt{ packet\_size})
$$

where `contiguous_sram_access_size` is the byte size of the innermost contiguous region in the output DM tensor, and `packet_size` is the input packet size in bytes.
When `commit_in_size < packet_size`, the packet is chunked into `packet_size / commit_in_size` writes.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![N = 3, R = 4, C = 8];

fn chunking_commit<'l, const T: Tu>(
    input: AccumulationTensor<'l, T, i8, m![1], m![1], m![1], m![N], m![R, C]>,
) -> DmTensor<i8, m![1], m![1], m![1], m![N, R, C # 16]> {
    // Input: R × C = 4 × 8 = 32 i8 elements = 32-byte packet.
    // Output: m![N, R, C # 16] — each C row is 8 useful bytes, padded to 16 in DM.
    //
    // Compiler-generated configuration: [
    //   N -> 3 : 64,   (64 == 4 * 16, contiguous)
    //   R -> 4 : 16,   (16 != 8 * 1,  NOT contiguous — padding gap)
    //   C -> 8 : 1
    // ] : 8
    // contiguous_sram_access_size = 8
    // commit_in_size = gcd(contiguous_sram_access_size, packet_size)
    //                = gcd(8, 32) = 8
    //
    // The 32-byte packet is split into 4 × 8-byte writes along the R axis:
    // - Write 0: packet[ 0.. 8] → DM offset  0 (row R=0)
    // - Write 1: packet[ 8..16] → DM offset 16 (row R=1, stride 16)
    // - Write 2: packet[16..24] → DM offset 32 (row R=2)
    // - Write 3: packet[24..32] → DM offset 48 (row R=3)
    // Each write places 8 non-padded elements into a 16-element-padded row.
    input.commit(0)
}
```

Without padding (`Element = m![N, R, C]`), the layout is fully contiguous: `contiguous_sram_access_size = N × R × C = 96`, `gcd(96, 32) = 32`, and the entire packet is written in a single 32-byte operation with no chunking.

> [!NOTE]
> Chunking is automatically performed by the Commit Engine's sequencer.
> The user does not manually specify the sequencer configuration.-->


## Commit Sequencer

The commit sequencer writes streams to DM across slices.
Each slice within an aggregation executes its own sequencer.
This mirrors how fetch sequencers pull data into Tensor Units.

The `commit_size` value determines how many bytes are written per sequencer step.
It is analogous to the Fetch Engine's [`fetch_size`](./fetch-engine.md#constraints) and is also derived from `contiguous_sram_access_size`:

$$
\texttt{commit\_size} = \gcd(\texttt{contiguous\_sram\_access\_size\_bytes},\ \texttt{commit\_in\_size})
$$

- When `commit_size == commit_in_size`, each time step produces a single DM write.
- When `commit_size < commit_in_size`, the packet is split into `commit_in_size / commit_size` writes per time step.

The main context supports a `commit_size` of 8, 16, 24, or 32 bytes (see [main context](../computing-tensors/index.md#execution-contexts)).
The sub-context supports a `commit_size` of 8 bytes only (see [sub-context](../computing-tensors/index.md#execution-contexts)).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![M = 4, K = 2, W = 8, N = 16];

// Compiler-generated configuration: [
//   M -> 4 : 16,  (16 == 2 * 8,  contiguous)
//   K -> 2 : 8,   (8  == 8 * 1,  contiguous)
//   W -> 8 : 1    (packet dimension, contiguous)
// ] : 8
// contiguous_sram_access_size = (8 * 2 * 4) elements × 1 byte = 64 bytes
// commit_in_size = 8 bytes (8 valid i8 elements out of 32-byte flit)
// commit_size = gcd(64, 8) = 8
fn no_transpose<'l, const T: Tu>(
    input: CastTensor<'l, T, i8, m![1], m![1], m![1], m![M, K], m![W # 32]>,
) -> DmTensor<i8, m![1], m![1], m![1], m![M, K, W]> {
    input.commit(0)
}

// Compiler-generated configuration: [
//   M -> 4 : 8,   (8  != 2 * 32, NOT contiguous)
//   K -> 2 : 32,  (32 != 8 * 1,  NOT contiguous)
//   W -> 8 : 1    (packet dimension, contiguous)
// ] : 32
// contiguous_sram_access_size = 8 elements × 4 bytes = 32 bytes
// commit_in_size = 32 bytes
// commit_size = gcd(32, 32) = 32
fn transpose<'l, const T: Tu>(
    input: AccumulationTensor<'l, T, f32, m![1], m![1], m![1], m![M, K], m![W]>,
) -> DmTensor<f32, m![1], m![1], m![1], m![K, M, W]> {
    input.commit(0)
}

// Compiler-generated configuration: [
//   M -> 4 : 8,   (8  != 2 * 32, NOT contiguous)
//   K -> 2 : 32,  (32 != 8 * 1,  NOT contiguous)
//   N -> 8 : 1    (truncated packet dimension, contiguous)
// ] : 16
// contiguous_sram_access_size = 8 elements × 2 bytes = 16 bytes
// commit_in_size = 16 bytes (8 bf16 elements; truncation from 16 elements to 8)
// commit_size = gcd(16, 16) = 16
fn transpose_with_truncation<'l, const T: Tu>(
    input: CastTensor<'l, T, bf16, m![1], m![1], m![1], m![M, K], m![N]>,
) -> DmTensor<bf16, m![1], m![1], m![1], m![K, M, N = 8]> {
    input.commit(0)
}

// Compiler-generated configuration: [
//   K -> 2 : 64,  (64 == 4 * 16, contiguous)
//   M -> 4 : 16,  (16 != 8 * 1,  NOT contiguous)
//   W -> 8 : 1    (packet dimension, contiguous)
// ] : 8
// contiguous_sram_access_size = 8 elements × 1 byte = 8 bytes
// commit_in_size = 32 bytes
// commit_size = gcd(8, 32) = 8
//
// The 32-byte packet is split into 4 × 8-byte writes along the M axis:
// - Write 0: packet[ 0.. 8] → DM offset  0
// - Write 1: packet[ 8..16] → DM offset 16
// - Write 2: packet[16..24] → DM offset 32
// - Write 3: packet[24..32] → DM offset 48
fn padding_chunking<'l, const T: Tu>(
    input: CastTensor<'l, T, i8, m![1], m![1], m![1], m![K], m![M, W]>,
) -> DmTensor<i8, m![1], m![1], m![1], m![K, M, W # 16]> {
    input.commit(0)
}
```

### Slice Bitmap

The slice bitmap enables selective commits to specific slices.
A `256`-bit mask controls which slices receive commit data, with each bit corresponding to one slice.

For example:
- `bitmap = 00000000...01` enables commit only to slice `0`
- `bitmap = 11111111...10` enables commit to all slices except slice `0`

This feature supports workflows that compute on specific slices and commit results only to those slices.


### Hardware Constraint

The commit sequencer must adhere to the same limits as fetch sequencers.
See [fetch sequencer constraints](./fetch-engine.md#constraints) for details.


## Sub-Context Operations

The sub-context Commit Engine provides specialized capabilities beyond the main context, though it supports fewer adapter stages.

<!-- > **TODO** (jeongmin.park): Please document this section. -->

- **Valid Count Packing**: This operation selectively commits only valid tensor elements based on a runtime count, excluding padding or invalid data from the output buffer. When computation produces variable-length results (for example, filtering operations or dynamic sequence lengths), valid count packing ensures that only meaningful elements are written to DM, preventing wasted memory and simplifying downstream processing. The hardware uses a count parameter to determine how many leading elements from each packet should be committed, discarding the remainder.

<!-- > **TODO**: Add detailed specification including count parameter format, interaction with slice bitmap, and performance characteristics of variable-length commits. -->

- **Generate Mode**: Writes a single `32`-bit value to a specified address via an `ITOS` (immediate-to-SRAM) command, bypassing the Tensor Unit execution pipeline.

## Constraints

- The input packet size must be 32 bytes.
- The `commit_in_size` must be 8, 16, 24, or 32 bytes.
The `commit_size` must be 8, 16, 24, or 32 bytes for the [main context](../computing-tensors/index.md#execution-contexts) and 8 bytes only for the [sub-context](../computing-tensors/index.md#execution-contexts). Note that the user only specifies the `Element` mapping. These constraints are internal to the compiler.
<!--- Type casting is available in the [main context](./computing-tensors.md#execution-contexts) only. The only supported type conversion is `f32` → `bf16`, with an optional `ReLU` activation. No other type conversions are supported.-->
- The two contexts support different capabilities:

| Stage               | Main context                    | Sub context |
|---------------------|---------------------------------|-------------|
| Truncating          | Yes                             | Yes         |
| Valid Count Packing | No                              | Yes         |
| Generate Mode       | No                              | Yes         |
<!--| Chunking            | Yes                             | Yes         |-->
<!--| Type Casting        | f32 → bf16 (with optional ReLU) | No          |-->

- Sub-context commits can only follow `fetch`.
These cannot be preceded by [Cast Engine](../computing-tensors/cast-engine.md) or [Transpose Engine](../computing-tensors/transpose-engine.md) operations.
- The commit sequencer shares the same limits as the fetch sequencer (see [fetch sequencer constraints](./fetch-engine.md#constraints)).
Additionally, all sequencer strides must be multiples of 8 bytes.


## Performance

Commit Engine performance directly affects overall computation throughput since DM writes must complete before subsequent operations can access the data.

### Write Bandwidth

The Commit Engine achieves maximum write bandwidth when:
- **Slice Interleaving**: Distributing writes across all active slices (or the subset specified by the slice bitmap) avoids bottlenecks on individual slices.
The RNGD chip has 64 slices per PE. The 256-bit bitmap accommodates up to 4 PEs (4 × 64 = 256).
- **Sequential Addresses**: Writing to sequential DM addresses within each slice enables parallel bank access (128 B/cycle per DMN, 256 B/cycle with DMN interleaving).
- **Aligned Packet Sizes**: Using 8-byte aligned packet sizes (8, 16, 24, 32 bytes) avoids partial bank writes.

For detailed memory performance characteristics, see [Memory Performance](./memory-performance.md).

### Adapter Stage Costs

Each adapter stage adds minimal latency:
- **Truncating**: Nearly zero cost (simple data width reduction)
<!--- **Chunking**: Increases time proportionally (splitting one 32-byte packet into four 8-byte packets takes 4× longer)-->
<!--- **Type Casting**: 1-2 cycle latency for `f32` to `bf16` conversion-->

### Bank Starvation Prevention

The Commit Engine shares DM bank access with the [Fetch Engine](./fetch-engine.md) and [DMA Engine](./dma-engine.md).
To prevent bank starvation and catastrophic NoC timeouts, ensure commit patterns avoid `64`+ consecutive accesses to the same bank.
The compiler automatically enforces this constraint by treating violating operations as if they occupy DMA context, preventing concurrent DMA operations.

See [DM Bank Starvation](./memory-performance.md#bank-starvation) for details.


