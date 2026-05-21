# Commit Engine

The Commit Engine writes Tensor Unit stream packets to DM in two stages:
- **[Commit Adapter](#commit-adapter)**: Applies optional per-element operations (truncating, type casting, and sub-context-only valid count packing and generate mode) that may change the mathematical tensor.
- **[Commit Sequencer](#commit-sequencer)**: A [mathematical tensor move](../mapping-tensors/tensor-semantics.md#mathematical-tensor-move) that writes packets to each slice's DM partition.

See [Optimizations](#optimizations) for write throughput considerations.

## Interface

A `TuTensor` carries `Chip`, `Cluster`, `Slice`, `Time`, and `Packet` dimensions at the end of the Tensor Unit pipeline.
Its `Time` reflects the temporal unrolling of the computation, and `Packet` is the element layout in the output stream.

`.commit()` converts the stream to a `DmTensor` in DM.
The Commit Engine mirrors the [Fetch Engine's](./fetch-engine.md) structure, but operates in reverse.

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/commit.rs:commit_impl}}
```

`.commit()` preserves the `Chip`, `Cluster`, and `Slice` dimensions unchanged, because each slice independently writes to its own DM partition.
The output `Element` mapping replaces `Time` and `Packet`, defining how the stream is laid out in DM.

`Element` configures both the Commit Sequencer and the Commit Adapter, and can reorder `Time` axes relative to the input stream, performing a transpose during the commit.
For performance implications of the `Element` mapping, see [Optimizations](#optimizations).

The following example commits a cast accumulation result to DM as `bf16`.
The output `DmTensor` stores 16 time steps × 8 `bf16` elements across 256 slices.
Here `D = bf16` and `Element = m![M, N # 16]`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, M = 16, N = 8];

fn cast_commit<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1], m![P], m![M], m![N]>,
) -> DmTensor<bf16, m![1], m![1], m![P], m![M, N # 16]> {
    // Cast f32 to bf16 (Cast Engine), then commit to DM (Commit Engine).
    // Input: M = 16 time steps, N = 8 f32 elements per packet (32 bytes).
    // After cast: N = 8 bf16 elements padded to 16 (32 bytes).
    // The sequencer writes across P = 256 slices.
    input.cast::<bf16, m![N # 16]>().commit(0)
}
```

## Commit Adapter

The Commit Adapter applies optional per-element operations on stream packets before they reach the [Commit Sequencer](#commit-sequencer).
Capabilities differ between the [main and sub contexts](../computing-tensors/index.md#scheduling-execution-contexts):

| Operation | Main | Sub |
| --- | --- | --- |
| [Truncating](#truncating) | ✅ | ✅ |
| [Type Casting](#type-casting) | ✅ | ❌ |
| [Valid Count Packing](#valid-count-packing) | ❌ | ✅ |
| [Generate Mode](#generate-mode) | ❌ | ✅ |


### Truncating


Stream packets in the Tensor Unit pipeline are always 32-byte *flits* (see [Collect Engine](../computing-tensors/collect-engine.md)), but a flit may carry fewer valid elements than its capacity, with trailing elements filled by padding.
Writing the full flit verbatim would clobber DM bytes beyond the valid region with the flit's padding values.

Truncating solves this by writing only the leading `valid_size` elements of each flit to DM, discarding the trailing padding.
The compiler derives `valid_size` from the output tensor mapping.
Users do not set it directly.
`D[valid_size]` must be 8, 16, 24, or 32 bytes (where 32 means no truncation).
Truncating adds nearly zero latency.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 4, K = 2, W = 8, N = 16, J = 64];

fn i8_padding_truncation<'l, const T: Tu>(
    input: CastTensor<'l, T, i8, m![1], m![1], m![1], m![M, K], m![W # 32]>,
) -> DmTensor<i8, m![1], m![1], m![1], m![M, K, W]> {
    // 8 valid i8 out of 32 padded; valid_size = 8.
    input.commit(0)
}

fn f32_non_padding_truncation<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1], m![1], m![M, K], m![W]>,
) -> DmTensor<f32, m![1], m![1], m![1], m![M, K, W = 4]> {
    // 4 valid f32 out of 8; valid_size = 4.
    input.commit(0)
}

fn bf16_truncation_with_transpose<'l, const T: Tu>(
    input: CastTensor<'l, T, bf16, m![1], m![1], m![1], m![M, K], m![N]>,
) -> DmTensor<bf16, m![1], m![1], m![1], m![K, M, N = 8]> {
    // 8 valid bf16 out of 16; valid_size = 8.
    input.commit(0)
}

fn i4_no_truncation_with_transpose<'l, const T: Tu>(
    input: CastTensor<'l, T, i4, m![1], m![1], m![1], m![M, K], m![J]>,
) -> DmTensor<i4, m![1], m![1], m![1], m![K, M, J]> {
    // No truncation; valid_size = 64.
    input.commit(0)
}
```

### Type Casting

Type casting converts `f32` data to `bf16` format just before writing to DM.
This reduces storage requirements and optionally applies a `ReLU` activation function during conversion.

The [Cast Engine](../computing-tensors/cast-engine.md) handles most type conversions in the Tensor Unit pipeline.
Commit Engine type casting exists for one specific case: running main-context contraction in parallel with sub-context Vector Engine work.
The Cast Engine sits on top of the Vector Engine and so occupies it during a conversion.
If the main-context performed its `f32` → `bf16` conversion through the Cast Engine, the Vector Engine would be busy and the sub-context could not run in parallel.
Routing the conversion through the Commit Engine instead leaves the Vector Engine free for the sub-context.
Sub-context itself does not support type casting (consistent with the [Commit Adapter](#commit-adapter) main/sub matrix above).


#### Standard conversion (`f32` to `bf16`)

Standard conversion reduces precision from `f32` to `bf16` while preserving all values, including negative ones.

```rust,ignore
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![N = 4, C = 3, H = 4, W = 8];

fn cast_commit<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1], m![1], m![N, C, H], m![W]>,
) -> DmTensor<bf16, m![1], m![1], m![1], m![N, C, H, W # 16]> {
    // Cast f32 to bf16 (values preserved), then commit to DM.
    // W = 8 f32 elements (32 bytes) → 8 bf16 elements padded to 16 (32 bytes).
    input.commit_cast(0)
}
```

#### Conversion with ReLU (`f32` to `bf16`)

Conversion with ReLU applies the `ReLU` activation function during type conversion.
Negative values are clamped to zero, while non-negative values are converted to `bf16` unchanged.

```rust,ignore
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![N = 4, C = 3, H = 4, W = 8];

fn cast_relu_commit<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1], m![1], m![N, C, H], m![W]>,
) -> DmTensor<bf16, m![1], m![1], m![1], m![N, C, H, W # 16]> {
    // Cast f32 to bf16 with ReLU: negative values clamped to 0.0.
    // e.g., [-5.0, -0.1, 0.0, 3.7] → [0.0, 0.0, 0.0, 3.7]
    input.commit_cast_relu(0)
}
```

### Valid Count Packing

Valid Count Packing selectively commits only valid tensor elements based on a runtime count, excluding padding or invalid data from the output buffer.
When computation produces variable-length results (for example, filtering operations or dynamic sequence lengths), valid count packing ensures that only meaningful elements are written to DM, preventing wasted memory and simplifying downstream processing.
The hardware uses a count parameter to determine how many leading elements from each packet should be committed, discarding the remainder.


### Generate Mode

Generate Mode writes a single `32`-bit value to a specified address via an `ITOS` (immediate-to-SRAM) command, bypassing the Tensor Unit execution pipeline.

## Commit Sequencer

`Chip`, `Cluster`, and `Slice` are the hardware spatial parallelism dimensions.
A Commit Sequencer runs independently in every slice, each writing to its own local DM partition.

### Constraints

- **Hardware dimensions**: `Chip::SIZE`, `Cluster::SIZE`, and `Slice::SIZE` must match the hardware configuration (see [Sequencer](./sequencer.md#architecture)).
- **Address alignment**: All sequencer strides must be multiples of 8 bytes.
- **Write unit alignment**: `D[valid_size]` must be 8, 16, 24, or 32 bytes (see [Truncating](#truncating)).

### Multi-Write Packet

Writing a packet may require multiple hardware writes because packet axes may not be contiguous in DM.
The per-write element count `write_size = gcd(valid_size, access_size)` is derived by the compiler, where `valid_size` comes from the [Commit Adapter](#commit-adapter) and `access_size` from the [Sequencer Architecture](./sequencer.md#access-size).
In the [sub-context](../computing-tensors/index.md#scheduling-execution-contexts), `D[write_size]` is fixed at 8 bytes.
The total cycle count is `Time::SIZE * (valid_size / write_size)`.
The division is always exact: in the main-context, `valid_size == write_size`, so each packet commits in a single cycle.
In the sub-context, `write_size` is fixed at 8 bytes and `valid_size` is one of 8, 16, 24, or 32 bytes (from the truncating constraint), so `valid_size / write_size` is always 1, 2, 3, or 4.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 4, K = 2, W = 8, N = 16];

// Compiler-generated configuration: [
//   M -> 4 : 16,  (16 == 2 * 8,  contiguous)
//   K -> 2 : 8,   (8  == 8 * 1,  contiguous)
//   W -> 8 : 1    (packet dimension, contiguous)
// ] : 8
// access_size = 64; valid_size = 8; write_size = gcd(64, 8) = 8; writes per packet = 1
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
// access_size = 8; valid_size = 8; write_size = gcd(8, 8) = 8; writes per packet = 1
fn transpose<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1], m![1], m![M, K], m![W]>,
) -> DmTensor<f32, m![1], m![1], m![1], m![K, M, W]> {
    input.commit(0)
}

// Compiler-generated configuration: [
//   M -> 4 : 8,   (8  != 2 * 32, NOT contiguous)
//   K -> 2 : 32,  (32 != 8 * 1,  NOT contiguous)
//   N -> 8 : 1    (truncated packet dimension, contiguous)
// ] : 16
// access_size = 8; valid_size = 8 (truncated from 16); write_size = gcd(8, 8) = 8; writes per packet = 1
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
// access_size = 8; valid_size = 32; write_size = gcd(8, 32) = 8; writes per packet = 4
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

The slice bitmap is a 256-bit mask covering one full cluster (one bit per slice, 256 slices per cluster) that gates which slices receive commit data.
For example, `bitmap = 00000000...01` enables commit only to slice `0`, and `bitmap = 11111111...10` enables commit to all slices except slice `0`.


### Optimizations

Three factors determine Commit Sequencer throughput.

- **Sequential Addresses**: Writing to sequential DM addresses within each slice enables parallel bank access (128 B/cycle per DMN, 256 B/cycle with DMN interleaving).
  Patterns that hit the same bank 64+ times consecutively trigger [DM Bank Starvation](./memory-performance.md#bank-starvation).
- **Spatial parallelism**: Distributing writes across all active slices maximizes throughput.
- **Aligned writes** (invariant): Partial bank writes never occur, because both the write address and the write unit are always 8-byte aligned.
  Sequencer strides are multiples of 8 bytes (see [Constraints](#constraints)) and `D[valid_size]` is one of 8, 16, 24, or 32 bytes (see [Truncating](#truncating)).

