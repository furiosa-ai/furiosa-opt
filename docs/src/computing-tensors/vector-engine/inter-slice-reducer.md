# Inter-Slice Reducer

The Inter-Slice Reducer reduces a tensor across the 256 slices in a cluster.
It preserves `Chip`, `Cluster`, and `Packet`, and rewrites `Slice` and `Time` into `OutSlice` and `OutTime`.
The output tensor is always `Way8` regardless of the input mode.

## Interface

The reducer can be entered right after `vector_init()` (the `InterFirst` path, shown below) or from a [compatible intra-slice stage](./intra-slice-chain.md#transitioning-to-the-inter-slice-reducer) (the `IntraFirst` path, with the same `vector_inter_slice_reduce()` method called on the intra-slice tensor).
The signatures shown below are the `VectorInitTensor` variants. The same methods also exist on intra-slice tensors at the stages that support the transition, so the call site looks identical.

The inter-slice reducer provides separate APIs for `i32` and `f32`.

### `i32` Operations

```rust,ignore
{{#include ../../../../furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:vector_init_inter_slice_reduce_i32_impl}}
```

`InterSliceReduceOpI32` operations:

| Operation | Description |
|-----------|-------------|
| `Add` | Wrapping addition |
| `AddSat` | Saturating addition |
| `Max` | Maximum value |
| `Min` | Minimum value |

### `f32` Operations

```rust,ignore
{{#include ../../../../furiosa-opt-std/src/engine/vector/tensor/vector_tensor.rs:vector_init_inter_slice_reduce_f32_impl}}
```

`InterSliceReduceOpF32` operations:

| Operation | Description |
|-----------|-------------|
| `Add` | Floating-point addition |
| `Max` | Maximum value |
| `Min` | Minimum value |
| `Mul` | Floating-point multiplication |

## Constraints

The supported `Slice → OutSlice` and `Time → OutTime` shapes follow four rules:

1. **Reduce from innermost.** The reduced portion of `Slice` must be the innermost factors, contiguous, with stride 1 through the reduction ratio `r`.
2. **Replacement on the reduced axis.** Each reduced factor's slot in `OutSlice` is filled by one of: a dummy (`1 # n`), a broadcast over a fresh dimension, or a promotion from `Time`.
3. **Replacement kinds mix freely.** Dummy, broadcast, and promotion slots can appear together in `OutSlice` in any order.
4. **Promotion from `Time` to `OutSlice` reorders.** The `Time → OutTime` portion preserves the relative order of surviving factors, but the `Time → OutSlice` promotion path does not preserve order: a promoted factor's position in `OutSlice` is independent of its position in `Time`.

## Examples

The math in the examples below uses einsum notation.
A dimension that appears on the input side but not on the output side is reduced (summed), and a dimension that appears on the output side but not on the input side is broadcast.

### Dummy Replacement

This pass sums input across `R` and places the result in a dummy slot.
The einsum form is `AR -> A`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, B = 2, R = 4];

// When R is reduced and no other dimension fills its slot, the output keeps the slot as a 1 # n dummy.
// One position holds the reduced value, and the remaining n - 1 are padding.
// `# n` denotes dimension multiplicity (see the Mapping Expressions doc).
fn inter_slice_add<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![B], m![A / 8, R], m![1], m![A % 8]>,
) -> VectorFinalTensor<'l, T, i32, m![1], m![B], m![A / 8, 1 # 4], m![1], m![A % 8]> {
    input
        .vector_init()
        // sum across R, the freed R-slot becomes the 1 # 4 dummy in OutSlice
        .vector_inter_slice_reduce::<m![A / 8, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
}
```

```text
Slice = [A / 8, R]  ->  [A / 8, 1 # 4]
Time  = [1]         ->  [1]
```

### Broadcast Into a New Slice Dimension

This pass reduces `R` and broadcasts the result over a fresh dimension `X`.
The einsum form is `PRW -> PWX`, with the fresh `X` on the output side broadcasting.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 2, P = 8, R = 4, W = 64, X = 4];

// A fresh non-reduce dimension X takes the slot that R leaves behind.
// The reduced value broadcasts across every position of X.
fn broadcast_into_x<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![W, R], m![1], m![P]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![W, X], m![1], m![P]> {
    input
        .vector_init()
        // sum across R, broadcast result over X (fresh OutSlice dimension)
        .vector_inter_slice_reduce::<m![W, X], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
}
```

```text
Slice = [W, R]  ->  [W, X]
Time  = [1]     ->  [1]
```

### Promotion from `Time` into `OutSlice`

This pass reduces `R` and promotes the `Time` dimension `V` into OutSlice.
The einsum form is `PRSUVW -> PSUVW`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 2, P = 8, R = 4, S = 2, U = 2, V = 4, W = 64];

// A dimension from Time (here V) is promoted into OutSlice to fill R's slot.
// The promoted dimension does not need to be outermost in Time.
fn axis_promotion<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![B], m![W, R], m![S, V, U], m![P]>,
) -> VectorFinalTensor<'l, T, f32, m![1], m![B], m![W, V], m![S, U], m![P]> {
    input
        .vector_init()
        // sum across R, V moves from Time to OutSlice
        .vector_inter_slice_reduce::<m![W, V], m![S, U]>(InterSliceReduceOpF32::Add)
        .vector_final()
}
```

```text
Slice = [W, R]     ->  [W, V]
Time  = [S, V, U]  ->  [S, U]
```

For examples that combine the reducer with the intra-slice chain in either order, see the [Vector Engine Examples](./index.md#examples).

## Performance

The reduction ratio `r` (the number of slices in one reduction group) is the main tuning knob.
Inter-slice reduce latency is `O(r)` cycles, approximately one ring traversal of the reduction group.
Total time equals the input streaming time plus that ring-sized tail.

In practice, upstream work (contraction producing partial sums or intra-slice work before `vector_inter_slice_reduce()`) often dominates and hides this ring tail, so the reducer isn't the bottleneck.
The reducer becomes visible at large `r` (longer tails) and on small tensors that can't amortize the fixed tail across many packets.
