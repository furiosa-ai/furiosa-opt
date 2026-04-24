# Vector Engine

The Vector Engine applies element-wise operations: activations such as GELU and SiLU, normalizations such as softmax and layer norm, and binary operations. It is used both after the Contraction Engine (to post-process `f32`/`i32` accumulator results) and independently for element-wise kernels that skip contraction entirely.

The Vector Engine operates exclusively on `i32` and `f32` data types. Data moves in 32-byte units called *flits*, each containing eight 32-bit values. This 32-bit restriction exists because lower-precision data is widened before or during computation: `bf16` products accumulate in `f32`, and `i8` products accumulate in `i32`.

The Vector Engine sits between the Contraction Engine and the Cast Engine in the [Tensor Unit pipeline](../index.md):

```text
Fetch -> Switch -> Collect -> Contraction -> Vector -> Cast -> Transpose -> Commit
                    |                       ^
                    +-----------------------+
                     (skip contraction)
```

Data enters the Vector Engine as either:
- From the **Collect Engine** when the Contraction Engine is skipped
- From the **Contraction Engine** when it produces the input

## Interface

```rust,ignore
{{#include ../../../../furiosa-visa-std/src/stream_tensor.rs:collect_vector_init}}

{{#include ../../../../furiosa-visa-std/src/vector_engine/tensor/vector_tensor.rs:vector_intra_slice_branch}}

{{#include ../../../../furiosa-visa-std/src/vector_engine/tensor/vector_tensor.rs:vector_intra_slice_unzip}}

{{#include ../../../../furiosa-visa-std/src/vector_engine/tensor/vector_tensor.rs:init_inter_slice_reduce_i32}}
```

The same `vector_init()` entry point is available regardless of whether the input comes from the Collect Engine (when contraction is skipped) or the Contraction Engine (for post-contraction processing).
After `vector_init()`, choose the first block by calling either `vector_intra_slice_branch(...)`, `vector_intra_slice_unzip(...)`, or `vector_inter_slice_reduce(...)`.
For detailed stage-by-stage API coverage, see [Intra-Slice Block](./intra-slice-block.md) and [Inter-Slice Block](./inter-slice-block.md).

## Quick Reference

| Block | How to Reach It | Use It For | Output |
|-------|------------------|------------|--------|
| [Intra-Slice Block](./intra-slice-block.md) | Start with `vector_init()`, then call `vector_intra_slice_branch()` | Elementwise ops, binary ops, intra-slice reduce | Chain stages, then `vector_final()` |
| [Inter-Slice Block](./inter-slice-block.md) | Either call `vector_init() -> vector_inter_slice_reduce()` first, or switch from an eligible intra-slice tensor with `vector_inter_slice_reduce()` | Reduction across the 256 slices in a cluster | `vector_inter_slice_reduce()`, then optional intra-slice work or `vector_final()` |
| [Two-group intra-slice mode](./intra-slice-block.md#two-group-mode) | Start with `vector_init()`, then call `vector_intra_slice_unzip()` | Process two interleaved groups before combining them | `_zip` to merge, then `vector_final()` |

## Examples

### ReLU Activation

Applying ReLU activation (`max(x, 0)`) after matrix multiplication:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![M = 128, N = 256, K = 64];

fn relu<'l, const T: Tu>(
    input: AccumulationTensor<'l, T, f32, m![1], m![1], m![K], m![M], m![N]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![1], m![K], m![M], m![N]> {
    input
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_clip(ClipBinaryOpF32::Max, 0.0f32)
        .vector_final()
}
```

### Inter-Slice Reduce

Reducing a tensor across slices:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![R = 4, A = 512];

fn inter_slice_reduce<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![1 # 2], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
}
```

## Ordering

| Order | Flow | Typical Use |
|-------|------|-------------|
| `IntraFirst` | Intra-Slice Block -> optional Inter-Slice Block | Post-process each slice, then reduce across slices |
| `InterFirst` | Inter-Slice Block -> optional Intra-Slice Block | Reduce first, then apply elementwise post-processing |

The examples above show one concrete `IntraFirst` path and one concrete `InterFirst` path.

## Constraints

When using `i8` or `bf16` input without the Contraction Engine, widening must still fit within one 32-byte flit. This limits how much data the Fetch Engine can supply per flit after type conversion. See [Fetch Engine: Type Casting Constraints](../../moving-tensors/fetch-engine.md#type-casting-constraints).
