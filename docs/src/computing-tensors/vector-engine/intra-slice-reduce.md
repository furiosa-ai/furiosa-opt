# Intra-Slice Reduce

The `IntraSliceReduce` stage in the [Intra-Slice Chain](./intra-slice-chain.md) reduces dimensions that live in the `Time` and `Packet` of each slice (`Chip`, `Cluster`, and `Slice` pass through unchanged).
The [Inter-Slice Reducer](./inter-slice-reducer.md) covers the complementary case of reducing across the 256 slices of a cluster.

<a id="interface"></a>

## Examples

The reduce call's key parameters are below.

- **`REDUCE_LABEL`**: The axis to reduce.
  Reduction eliminates every factor in `Time` and `Packet` carrying this axis, so they must not appear in the output shape (`OutTime`, `OutPacket`).
  For example, if `R` is split as `R / 4` in `Time` and `R % 4` in `Packet`, specifying `REDUCE_LABEL = R` eliminates both.
- **`op`**: The reduce operation. `IntraSliceReduceOpI32` provides `AddSat`, `Max`, `Min`; `IntraSliceReduceOpF32` provides `Add`, `Max`, `Min`.
- **`OutTime`, `OutPacket`**: The output `Time` and `Packet` shape after reduction.
  These match the input `Time` and `Packet` with every `REDUCE_LABEL` factor removed.

The examples below exercise each parameter combination.

### Reduction in `Time`

`R` exists only in `Time`, so the stage accumulates across time steps.
It computes \\(output[a] = \sum_{r \in R} input[a, r]\\) with saturating addition.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, R = 16];

// R in Time → temporal accumulation. Packet is non-reduce.
fn reduce_time<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![A / 2], m![R], m![A % 2 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![A / 2], m![1], m![A % 2 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![A % 2 # 4]>()       // 8-way → 4-way
        // R eliminated from Time
        .vector_intra_slice_reduce::<R, m![1], m![A % 2 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
```

### Reduction in `Packet`

`R` exists only in `Packet`, so the hardware runs a 4-way tree reduce within each flit and skips temporal accumulation.
It computes \\(output[a] = \sum_{r \in R} input[a, r]\\).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 512, R = 4];

// R in Packet → tree reduce within flit.
fn reduce_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![A / 2], m![A % 2], m![R # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![A / 2], m![A % 2], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![R]>()             // 8-way → 4-way
        // R eliminated from Packet
        .vector_intra_slice_reduce::<R, m![A % 2], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
}
```

### Reduction in Both

`R` splits across `Packet` and `Time`.
`R % 4` in `Packet` tree-reduces within each flit, and `R / 4` in `Time` accumulates across time steps.
It computes \\(output[a] = \max_{r \in R} input[a, r]\\).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, R = 16];

// R % 4 in Packet → spatial tree reduce
// R / 4 in Time → temporal accumulation
fn reduce_time_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![A], m![R / 4], m![R % 4 # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![A], m![1], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![R % 4]>()            // 8-way → 4-way
        // R eliminated from both Time and Packet
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Max,
        )
}
```

### Per-Slice Reduction

`R` has portions in `Slice`, `Time`, and `Packet`, but the intra-slice reducer only collapses the `Time` and `Packet` portions, so the `Slice` portion of `R` stays in the output.
Here `R = 13` is padded to 32 to fit the layout (`R # 32`), and `R` is then split across 4 slices (8 `R`-positions per slice).
Only positions 0-12 hold real elements, so the slice straddling that boundary (positions 8-15: 5 real followed by 3 pad) is a boundary slice, and slices past it are fully padding.
The [VCG](#padding-strategy) drives the per-slice reduction count so each slice reduces only its real elements (see [Valid Count Generator](./vcg.md) for the exact mapping).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 13];

// R split across all three: Slice (groups of 8), Time (pairs within group), Packet (4 elements).
fn reduce_slice_time_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![R # 32 / 8], m![R # 32 / 4 % 2], m![R # 32 % 4 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![R # 32 / 8], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![R # 32 % 4]>()       // 8-way → 4-way
        // R eliminated from Time and Packet (accumulated within each slice)
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
```

## Architecture

The stage runs separate machinery for the `Time` and `Packet` axes.

### Reduction in `Time`

The stage applies the [temporal accumulator](../contraction-engine/time-reducer.md#constraints) model with slot capacity 8 (so `InnerTime::SIZE ≤ 8`).

For `reduce_time` above, `Time = m![R]` and `OutTime = m![1]`, so `R` is the outermost reduce dimension and `InnerTime = m![1]` (`InnerTime::SIZE = 1`). A single slot accumulates all `R` values into the output.

If `InnerTime::SIZE` exceeds 8, the API rejects the call. For example:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# axes![A = 6, B = 8, R = 16];
fn invalid_too_many_slots<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![A / 3], m![R, A % 3, B % 4], m![B / 4 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![A / 3], m![A % 3, B % 4], m![B / 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![B / 4]>()
        // Time      = m![R, A % 3, B % 4]
        // OutTime   = m![A % 3, B % 4]
        // InnerTime = m![A % 3, B % 4], InnerTime::SIZE = 3 × 4 = 12 > 8
        .vector_intra_slice_reduce::<R, m![A % 3, B % 4], m![B / 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // Rejected: 12 accumulator slots required, but only 8 are available.
}
```

### Reduction in `Packet`

The 4 Packet elements per flit pick one of two paths:

- `OutPacket = Packet`: elements pass through unreduced. 4 outputs per cycle, each accumulated independently over `Time`.
- `OutPacket = m![1 # 4]`: elements collapse to a single value via a 2-level tree `op(op(a, b), op(c, d))`. 1 output per cycle plus 3 padding positions.

<a id="padding-strategy"></a>

When a reduce axis is padded to fit hardware dimensions, the padded positions contain arbitrary data that the reduction must exclude.
Two strategies handle padding exclusion.

- **VCG (Valid Count Generator)**: Preferred when its axis placement is supported.
  The compiler configures the VCG automatically from the mapping, and the VCG tags each flit with a `valid_count` so pad elements are excluded automatically.
  Not all axis placements across Slice, Time, and Packet are supported.
  See [Valid Count Generator](./vcg.md) for details.

- **Identity-element padding**: Fill pad positions with the identity element of the reduce operation before data reaches the [Intra-Slice Chain](./intra-slice-chain.md).
  The [Fetch Engine's masking](../../moving-tensors/fetch-engine.md#masking) writes the identity value into pad positions during fetch:

  | Operation | Identity Element |
  |-----------|-----------------|
  | `AddSat` / `Add` | `0` / `0.0` |
  | `Max` | `i32::MIN` / `f32::NEG_INFINITY` |
  | `Min` | `i32::MAX` / `f32::INFINITY` |

  This strategy applies only when no non-invertible transformation precedes the reduce operation.
  For example, with `exp(x) + exp(y) + ...` (sum of exponentials), no value `p` satisfies `exp(p) = 0` (the additive identity), so identity padding does not apply.


## Performance

Throughput stays at one flit per cycle since the tree reduce is fully pipelined within the Intra-Slice Chain, adding no extra per-flit cost.

Latency adds a first-output delay of `n` flit cycles where `n` is the number of time steps in the reduce axis, because the stage must accumulate all input flits for a reduction group before emitting the result.
In a multi-engine pipeline, this accumulation delay stalls downstream engines waiting for the first flit.

