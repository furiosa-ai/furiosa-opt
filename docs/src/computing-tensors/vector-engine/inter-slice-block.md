# Inter-Slice Block

The Inter-Slice Block performs inter-slice reduction, aggregating partial results across the 256 slices within a cluster.
It preserves `Chip`, `Cluster`, and `Packet`, and rewrites `Slice` and `Time` to `SliceOut` and `TimeOut`.

## Interface

### `i32` Interface

```rust,ignore
{{#include ../../../../furiosa-visa-std/src/vector_engine/tensor/vector_tensor.rs:init_inter_slice_reduce_i32}}
```

### `f32` Interface

```rust,ignore
{{#include ../../../../furiosa-visa-std/src/vector_engine/tensor/vector_tensor.rs:init_inter_slice_reduce_f32}}
```

You can reach this block in two ways:

- Run inter-slice first: `vector_init() -> vector_inter_slice_reduce::<SliceOut, TimeOut>(op)`
- Run intra-slice first, then switch: call `vector_inter_slice_reduce()` directly on the current intra-slice tensor instead of calling `vector_init()` again.

In the `IntraFirst` path, `vector_inter_slice_reduce()` is available only from `Way8` intra-slice stages that can transition to inter-slice reduction: `Branch`, `Logic`, `Fxp`, `FxpToFp`, `Widen`, `FpToFxp`, and `Clip`.
It is not available from `Way4` stages such as `Narrow`, `Fp`, `IntraSliceReduce`, or `FpDiv`.

## Quick Reference

| Current state | Method | Result |
|---------------|--------|--------|
| Fresh VE input after `vector_init()` | `vector_inter_slice_reduce::<SliceOut, TimeOut>(op)` | Enters inter-slice reduction directly (`InterFirst`) |
| Eligible intra-slice tensor | `vector_inter_slice_reduce::<SliceOut, TimeOut>(op)` | Transitions from intra-slice to inter-slice reduction (`IntraFirst`) |
| Tensor after `vector_inter_slice_reduce()` | `vector_intra_slice_branch(BranchMode)` | Switches to intra-slice work after inter-slice reduction |

## Operations

### Integer Operations (`InterSliceReduceOpI32`)

| Operation | Description |
|-----------|-------------|
| `Add` | Wrapping addition |
| `AddSat` | Saturating addition |
| `Max` | Maximum value |
| `Min` | Minimum value |

### Floating-Point Operations (`InterSliceReduceOpF32`)

| Operation | Description |
|-----------|-------------|
| `Add` | Floating-point addition |
| `Max` | Maximum value |
| `Min` | Minimum value |
| `Mul` | Floating-point multiplication |

## Output Mapping Rule

After inter-slice reduction removes a slice factor `R`, the output mapping typically follows one of three rules:

| Rule | Output mapping | Reference |
|------|----------------|-----------|
| Broadcast | `Slice = m![A, R], Time = m![C] -> SliceOut = m![A, X], TimeOut = m![C]` | [Broadcast Into a New Slice Axis](#broadcast-into-a-new-slice-axis) |
| Dummy | `Slice = m![A, R], Time = m![C] -> SliceOut = m![A, 1 # n], TimeOut = m![C]` | [Dummy Replacement](#dummy-replacement) |
| Promotion | `Slice = m![A, R], Time = m![C] -> SliceOut = m![A, C], TimeOut = m![1]` | [Promotion from `Time` into `SliceOut`](#promotion-from-time-into-sliceout) |

`Chip`, `Cluster`, and `Packet` pass through unchanged.
Only `Slice` and `Time` are rewritten into `SliceOut` and `TimeOut`.

## Examples

### Dummy Replacement

Replace the reduced slice factor with a dummy factor in `SliceOut`:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![R = 4, A = 512];

fn inter_slice_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![1 # 2], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
}
```

`R` occupies part of the Slice dimension. After reduction, `R` is eliminated and the remaining `A / 8` positions are padded from `R`(=4) slots to `1 # 4`.

### Broadcast Into a New Slice Axis

Introduce a new axis in `SliceOut`, and broadcast the reduced value over that axis:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![W = 64, R = 4, X = 4, P = 8];

fn broadcast_into_x<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![1 # 2], m![W, R], m![1], m![P]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![1 # 2], m![W, X], m![1], m![P]> {
    input
        .vector_init()
        .vector_inter_slice_reduce::<m![W, X], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
}
```

Here, `R` is reduced away. `X` is a new axis that appears only in `SliceOut`,
so the reduced value is broadcast over the `X` positions in the output.

### Promotion from `Time` into `SliceOut`

If `Time` already contains an axis that should occupy the freed slice space, promote that axis into `SliceOut`.
The promoted axis does not have to be the outermost axis in `Time`:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![W = 32, R = 4, T0 = 2, T2 = 4, T1 = 2, P = 8];

fn axis_promotion<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![1 # 2], m![W, R], m![T0, T2, T1], m![P]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![1 # 2], m![W, T2], m![T0, T1], m![P]> {
    // Before: Slice = m![W, R], Time = m![T0, T2, T1], Packet = m![P]
    // After:  Slice = m![W, T2], Time = m![T0, T1], Packet = m![P]
    // R is reduced away, and T2 is promoted from the middle of Time into Slice.
    input
        .vector_init()
        .vector_inter_slice_reduce::<m![W, T2], m![T0, T1]>(InterSliceReduceOpF32::Add)
        .vector_final()
}
```

### Inter-Slice Reduce with `AddSat`, Then Intra-Slice

Reducing an `i32` tensor across slices, then applying an elementwise add:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![R = 4, A = 512];

fn reduce_then_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![1 # 2], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::AddFxp, 100)
        .vector_final()
}
```

### Intra-Slice Then Inter-Slice Reduce with `AddSat`

Applying an intra-slice operation first, then reducing the resulting `i32` tensor across slices:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![R = 4, A = 512];

fn add_then_reduce<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![1 # 2], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::AddFxp, 100)
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
}
```

## Constraints

| Constraint | Detail |
|------------|--------|
| Data types | `i32` and `f32` only |
| Scope | Reduction happens within one 256-slice cluster |
| Packet mapping | `Packet` does not change across inter-slice reduction |

## Performance

Inter-slice reduce is best understood as a ring-like global reduction across the participating slices. For documentation purposes, the most useful high-level estimate is:

| Quantity | Rough rule of thumb |
|----------|---------------------|
| First reduced output | on the order of one ring traversal for the reduction group |
| Total time | input streaming time + that ring-sized tail |
| Main tuning knob | reduction ratio, that is, how many slices participate in one inter-slice contraction group |

If you want a quick mental model, let `r` be the reduction ratio or route-group size:
- first output appears after roughly `O(r)` cycles
- larger `r` means more noticeable inter-slice tail latency
- if upstream already produces flits slowly, that upstream rate dominates and the inter-slice cost is partly hidden

This is intentionally a high-level approximation. The practical mental model is simple: stream partial results in, then pay about one ring traversal before the reduced result settles.

### Interaction With Other Pipelines

- **Contraction -> Inter-Slice**: if contraction takes longer to produce partial sums, contraction can dominate and inter-slice may not be the bottleneck.
- **Intra-Slice -> Inter-Slice**: intra-slice work can reduce the number of packets that reach inter-slice, or simply take longer itself. In those cases, inter-slice is less visible because there is less data to reduce, or because the front half already dominates.
- **Large ring / large reduction ratio**: when many slices participate, inter-slice tail latency grows and can become the bottleneck.
- **Small tensors**: even when total data volume is small, the fixed ring-style tail can still matter because it is amortized over fewer packets.

For an end-to-end contraction example that includes inter-slice reduction, see [Reducer](../contraction-engine/reducer.md).
