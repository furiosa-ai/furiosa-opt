# Commit Engine

The Commit Engine writes a Tensor Unit stream packet to DM, the inverse of the [Fetch Engine](./fetch-engine.md).
It is the Commit Sequencer: a [mathematical tensor move](../mapping-tensors/tensor-semantics.md#mathematical-tensor-move) that runs independently in every slice, each writing to its own local DM partition.

## Interface

A `TuTensor` carries `Chip`, `Cluster`, `Slice`, `Time`, and `Packet` dimensions at the end of the Tensor Unit pipeline.
Its `Time` reflects the temporal unrolling of the computation, and `Packet` is the element layout in the output stream.

`.commit()` writes the stream to a `DmTensor` in DM.

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/commit.rs:commit_impl}}
```

`.commit()` preserves the `Chip`, `Cluster`, and `Slice` dimensions unchanged, because each slice independently writes to its own DM partition.
The output `Element` mapping replaces `Time` and `Packet`, defining how the stream is laid out in DM.
`Element` configures both the Commit Sequencer and the [Commit Adapter](../computing-tensors/commit-adapter.md), and can reorder `Time` axes relative to the input stream, performing a transpose during the commit.
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
    input: ContractTensor<'l, T, f32, m![1], m![1 # 2], m![P], m![M], m![N]>,
) -> DmTensor<bf16, m![1], m![1 # 2], m![P], m![M, N # 16]> {
    // Cast f32 to bf16 (Cast Engine), then commit to DM (Commit Engine).
    // Input: M = 16 time steps, N = 8 f32 elements per packet (32 bytes).
    // After cast: N = 8 bf16 elements padded to 16 (32 bytes).
    // After trim: N # 16 trimmed into N = 8.
    // The sequencer writes across P = 256 slices.
    input.cast::<bf16, m![N # 16]>().commit_trim::<m![N]>().commit()
}
#
# let mut ctx = Context::acquire();
#
# let c: ContractTensor<'_, _, f32, m![1], m![1 # 2], m![P], m![M], m![N]> = ContractTensor::new(&mut ctx.main, Tensor::zero());
# let _o = cast_commit(c);
```

## Constraints

- **Hardware dimensions**: `Chip::SIZE`, `Cluster::SIZE`, and `Slice::SIZE` must match the hardware configuration (see [Sequencer](./sequencer.md#architecture)).
- **Address alignment**: All sequencer strides must be multiples of 8 bytes.
- **Write unit alignment**: `D[valid_size]` must be 8, 16, 24, or 32 bytes (see the [Commit Adapter's Trimming](../computing-tensors/commit-adapter.md#trimming) stage).

## Multi-Write Packet

Writing a packet may require multiple hardware writes because packet axes may not be contiguous in DM.
The per-write element count `write_size = gcd(valid_size, access_size)` is derived by the compiler, where `valid_size` comes from the [Commit Adapter](../computing-tensors/commit-adapter.md) and `access_size` from the [Sequencer Architecture](./sequencer.md#access-size).
In the [sub-context](../computing-tensors/index.md#execution-context), `D[write_size]` is fixed at 8 bytes.
The total cycle count is `Time::SIZE * (valid_size / write_size)`.
The division is always exact: in the main-context, `valid_size == write_size`, so each packet commits in a single cycle.
In the sub-context, `write_size` is fixed at 8 bytes and `valid_size` is one of 8, 16, 24, or 32 bytes (from the trimming constraint), so `valid_size / write_size` is always 1, 2, 3, or 4.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 4, K = 2, W = 8, N = 16, L = 32];

// Compiler-generated configuration: [
//   M -> 4 : 64,  (64 == 2 * 32,  contiguous)
//   K -> 2 : 32,   (32  == 32 * 1,  contiguous)
//   M -> 32 : 1    (packet dimension, contiguous)
// ] : 8
// access_size = 64; valid_size = 8; write_size = gcd(64, 8) = 8; writes per packet = 1
fn no_transpose<'l, const T: Tu>(
    input: CastTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![L]>,
) -> DmTensor<i8, m![1], m![1 # 2], m![1 # 256], m![M, K, L]> {
    input.commit_trim::<m![L]>().commit()
}

// Compiler-generated configuration: [
//   M -> 4 : 8,   (8  != 2 * 32, NOT contiguous)
//   K -> 2 : 32,  (32 != 8 * 1,  NOT contiguous)
//   W -> 8 : 1    (packet dimension, contiguous)
// ] : 32
// access_size = 8; valid_size = 8; write_size = gcd(8, 8) = 8; writes per packet = 1
fn transpose<'l, const T: Tu>(
    input: ContractTensor<'l, T, f32, m![1], m![1 # 2], m![1 # 256], m![M, K], m![W]>,
) -> DmTensor<f32, m![1], m![1 # 2], m![1 # 256], m![K, M, W]> {
    input.commit_trim::<m![W]>().commit()
}

// Compiler-generated configuration: [
//   M -> 4 : 8,   (8  != 2 * 32, NOT contiguous)
//   K -> 2 : 32,  (32 != 8 * 1,  NOT contiguous)
//   N -> 8 : 1    (trimmed packet dimension, contiguous)
// ] : 16
// access_size = 8; valid_size = 8 (trimmed from 16); write_size = gcd(8, 8) = 8; writes per packet = 1
fn transpose_with_trimming<'l, const T: Tu>(
    input: CastTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![N # 32]>,
) -> DmTensor<i8, m![1], m![1 # 2], m![1 # 256], m![K, M, N]> {
    input.commit_trim::<m![N]>().commit()
}

#
# let mut ctx = Context::acquire();
# let a: CastTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![L]> = CastTensor::new(&mut ctx.main, Tensor::zero());
# let _o = no_transpose(a);
# let b: ContractTensor<'_, _, f32, m![1], m![1 # 2], m![1 # 256], m![M, K], m![W]> = ContractTensor::new(&mut ctx.main, Tensor::zero());
# let _o = transpose(b);
# let c: CastTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![M, K], m![N # 32]> = CastTensor::new(&mut ctx.main, Tensor::zero());
# let _o = transpose_with_trimming(c);
```

## Slice Bitmap

The slice bitmap is a 256-bit mask covering one full cluster (one bit per slice, 256 slices per cluster) that gates which slices receive commit data.
For example, `bitmap = 00000000...01` enables commit only to slice `0`, and `bitmap = 11111111...10` enables commit to all slices except slice `0`.


## Optimizations

Three factors determine Commit Sequencer throughput.

- **Sequential Addresses**: Writing to sequential DM addresses within each slice enables parallel bank access (128 B/cycle per DMN, 256 B/cycle with DMN interleaving).
  Patterns that hit the same bank 64+ times consecutively trigger [DM Bank Starvation](./memory-performance.md#bank-starvation).
- **Spatial parallelism**: Distributing writes across all active slices maximizes throughput.
- **Aligned writes** (invariant): Partial bank writes never occur, because both the write address and the write unit are always 8-byte aligned.
  Sequencer strides are multiples of 8 bytes (see [Constraints](#constraints)), and the [Commit Adapter's Trimming](../computing-tensors/commit-adapter.md#trimming) stage holds `D[valid_size]` to a multiple of 8 bytes.
