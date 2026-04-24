# Intra-Slice Reduce

The Intra-Slice Reduce is a reduction operation performed by the `IntraSliceReduce` stage within the [Intra-Slice Block](./intra-slice-block.md).
At the hardware level, this corresponds to the reduction unit in the 4-way path.
It reduces axes **within a single slice**, contrasting with the [Inter-Slice Block](./inter-slice-block.md) which reduces **across** the 256 slices of a cluster (inter-slice reduce).
This document covers the blocking-mode case where the accumulator result is not stored to an intermediate buffer.
Non-blocking mode (where accumulator results are stored to an intermediate buffer) is not covered here.

## Interface

```rust,ignore
{{#include ../../../../furiosa-visa-std/src/vector_engine/tensor/vector_tensor.rs:intra_slice_reduce_i32}}

{{#include ../../../../furiosa-visa-std/src/vector_engine/tensor/vector_tensor.rs:intra_slice_reduce_f32}}
```

Parameters:
- **`REDUCE_LABEL`**: Which axis to reduce, specified as an `Ident` value (e.g., `Ident::R`).
  Each `axes![]` declaration creates a named label (`Ident`); all factors derived from the same declaration share that label.
  All factors in Time and Packet carrying this label are eliminated by the reduction, so they must not appear in the output shape (`OTime`, `OPacket`).
  For example, if `R` is split as `R / 4` in Time and `R % 4` in Packet, specifying `REDUCE_LABEL = Ident::R` eliminates both.
- **`op`**: The reduce operation (`IntraSliceReduceOpI32` for `i32`, `IntraSliceReduceOpF32` for `f32`).
- **`OTime`, `OPacket`**: The output Time and Packet shape after reduction.
  These must be exactly the input Time and Packet with all `REDUCE_LABEL` factors removed.

The `Chip`, `Cluster`, and `Slice` dimensions pass through unchanged from input to output.

## Mechanism

### Conceptual Operation

The `IntraSliceReduce` stage sits inside the Intra-Slice Block pipeline, after `Narrow` and before `Widen`.
It accepts a 4-way input (after Buffering Split divides the 8-way flit), performs a 2-level tree reduce to produce a single value, and accumulates the result into an accumulator slot.

```text
4-way input from Narrow stage
    ┌───┬───┬───┬───┐
    │ a │ b │ c │ d │
    └─┬─┴─┬─┴─┬─┴─┬─┘
      │   │   │   │
      └─┬─┘   └─┬─┘          Level 1: pairwise reduce
    op(a,b)   op(c,d)
        └───┬───┘             Level 2: pairwise reduce
    op(op(a,b),op(c,d))
            │
            ▼
     ┌─────────────┐
     │ Accumulator │         Accumulate across time steps
     │ (8 slots)   │
     └─────────────┘
```

The accumulator holds partial results across multiple input flits, implementing temporal reduction over the Time axis.
Up to 8 accumulator slots are available, each serving as a buffer that accumulates partial reduce results across time steps.

Padding exclusion is handled by the [Valid Count Generator (VCG)](./vcg-interface.md), which tags each flit with the count of valid elements so the `IntraSliceReduce` stage can skip pad data.

### Architectural Parameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| Tree input width | 4-way | Fixed; `Narrow` produces the 4-way input |
| Tree depth | 2 | Two levels of pairwise reduction |
| Accumulator slots | 8 | Independent reduction accumulators |
| Accumulator type | Temporal | Accumulates across time steps within each slot |

### Supported Operations

#### Integer Operations (`IntraSliceReduceOpI32`)

| Operation | Description | Identity Element |
|-----------|-------------|-----------------|
| `AddSat` | Saturating addition | `0` |
| `Max` | Maximum value | `i32::MIN` |
| `Min` | Minimum value | `i32::MAX` |

#### Floating-Point Operations (`IntraSliceReduceOpF32`)

| Operation | Description | Identity Element |
|-----------|-------------|-----------------|
| `Add` | Floating-point addition | `0.0` |
| `Max` | Maximum value | `f32::NEG_INFINITY` |
| `Min` | Minimum value | `f32::INFINITY` |

## Examples

### Reduce in Time (i32 Saturating Add)

R exists only in Time, accumulated across time steps.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# axes![A = 8, R = 16];
// Slice = m![A / 2], Time = m![R], Packet = m![A % 2 # 8] (8-way)
// R in Time → temporal accumulation. Packet is non-reduce.
fn reduce_time<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![A / 2], m![R], m![A % 2 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![A / 2], m![1], m![A % 2 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![A % 2 # 4]>()       // 8-way → 4-way
        .vector_intra_slice_reduce::<R, m![1], m![A % 2 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // Output: Slice = m![A / 2], Time = m![1], Packet = m![A % 2 # 4]
    // R eliminated from Time.
}
```

### Reduce in Packet Only (f32 Add)

R exists only in Packet, so the hardware performs a 4-way tree reduce within each flit with no temporal accumulation.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# axes![A = 8, R = 4];
// Slice = m![A / 2], Time = m![A % 2], Packet = m![R # 8] (8-way)
// R in Packet → tree reduce within flit. VCG tags valid_count = |R|.
fn reduce_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![A / 2], m![A % 2], m![R # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![A / 2], m![A % 2], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![R]>()             // 8-way → 4-way
        .vector_intra_slice_reduce::<R, m![A % 2], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // Output: Slice = m![A / 2], Time = m![A % 2], Packet = m![1 # 4]
    // R eliminated from Packet.
}
```

### Reduce Split Across Time and Packet (f32 Max)

R is split: `R % 4` in Packet is tree-reduced within each flit, then accumulated across `R / 4` time steps.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# axes![A = 8, R = 16];
// Slice = m![A / 2], Time = m![R / 4], Packet = m![R % 4 # 8] (8-way)
// R % 4 in Packet → spatial tree reduce
// R / 4 in Time → temporal accumulation
fn reduce_time_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![A / 2], m![R / 4], m![R % 4 # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![A / 2], m![1], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![R % 4]>()            // 8-way → 4-way
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Max,
        )
    // Output: Slice = m![A / 2], Time = m![1], Packet = m![1 # 4]
    // Both R portions eliminated.
}
```

### Reduce Axis Spanning Slice, Time, and Packet (i32 Min)

R spans all three dimensions.
The VCG handles per-slice valid count variation for boundary slices.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# axes![R = 13];
// Slice = m![R # 32 / 8], Time = m![R # 32 / 4 % 2], Packet = m![R # 32 % 4 # 8] (8-way)
// R split across all three: Slice (groups of 8), Time (pairs within group), Packet (4 elements).
// Boundary slices may have fewer valid time steps, and the VCG handles this.
fn reduce_slice_time_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![R # 32 / 8], m![R # 32 / 4 % 2], m![R # 32 % 4 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![R # 32 / 8], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![R # 32 % 4]>()       // 8-way → 4-way
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
    // Output: Slice = m![R # 32 / 8], Time = m![1], Packet = m![1 # 4]
    // R eliminated from Time and Packet (accumulated within each slice).
}
```

## Constraints

### Accumulator Slot Limit

The `IntraSliceReduce` stage has 8 accumulator slots.
Each non-reduce (NR) position inside the outermost reduce factor occupies a separate slot, so the product of inner NR axis sizes must be ≤ 8.

Consider `Time = m![R / 2, A % 2, R % 2]` where `R` is the reduce label and `A` is non-reduce.
The NR factor `A % 2` sits between the outer reduce `R / 2` and inner reduce `R % 2`.
Each value of `A % 2` needs its own accumulator slot to maintain an independent reduction:

```text
Time = m![R / 2, A % 2, R % 2]
        ~~~~~~  ~~~~~~  ~~~~~~
        outer R   NR    inner R

Flit sequence (R=4, A%2 has values 0,1):

  flit #0: R/2=0, A%2=0, R%2=0  ──→ ┌─────────────────┐
  flit #1: R/2=0, A%2=0, R%2=1  ──→ │ Slot 0 (A%2=0)  │  accumulates R for A%2=0
  flit #4: R/2=1, A%2=0, R%2=0  ──→ │                 │
  flit #5: R/2=1, A%2=0, R%2=1  ──→ └─────────────────┘

  flit #2: R/2=0, A%2=1, R%2=0  ──→ ┌─────────────────┐
  flit #3: R/2=0, A%2=1, R%2=1  ──→ │ Slot 1 (A%2=1)  │  accumulates R for A%2=1
  flit #6: R/2=1, A%2=1, R%2=0  ──→ │                 │
  flit #7: R/2=1, A%2=1, R%2=1  ──→ └─────────────────┘

  2 NR positions → 2 slots used (≤ 8 ✓)
```

With multiple NR factors, slot usage multiplies:

```text
Valid:   Time = m![R, A % 2, B % 4]  →  2 × 4 = 8 slots  (≤ 8 ✓)
Invalid: Time = m![R, A % 3, B % 4]  →  3 × 4 = 12 slots (> 8 ✗)
```

If the NR product exceeds 8, the mapping must be restructured.

### Invalid: Accumulator Slot Limit Exceeded (i32 AddSat)

NR factors between reduce factors occupy accumulator slots.
Here `A % 3` (3 values) and `B % 4` (4 values) sit inside the reduce axis, requiring 3 × 4 = 12 slots, exceeding the 8-slot limit.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# axes![A = 6, B = 8, R = 16];
// Time = m![R, A % 3, B % 4] -> NR product = 3 × 4 = 12 > 8
fn invalid_too_many_slots<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![A / 3], m![R, A % 3, B % 4], m![B / 4 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![A / 3], m![A % 3, B % 4], m![B / 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![B / 4]>()
        .vector_intra_slice_reduce::<R, m![A % 3, B % 4], m![B / 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // Rejected: 12 accumulator slots required, but only 8 are available.
}
```

### Padding Strategy

When a reduce axis is padded to fit hardware dimensions, the padded positions contain arbitrary data that must be excluded from the reduction result.
Three strategies are available:

| Situation | Strategy |
|-----------|----------|
| Mapping supported by VCG (see [Valid Count Generator's Interface](./vcg-interface.md)) | VCG (automatic, no extra setup) |
| Unsupported VCG placement, simple reduce op (Add, Max, Min) | Identity-element padding via Fetch Engine's `pad_value` |
| Unsupported VCG placement, composed op (e.g., `exp` + `Add`) | Restructure the mapping, or use other methods |

**1. VCG (Valid Count Generator).**
The [VCG](./vcg-interface.md) tags each flit with a `valid_count` so the `IntraSliceReduce` stage excludes pad elements automatically.
Not all axis placements across Slice, Time, and Packet are supported; see [Valid Count Generator's Interface](./vcg-interface.md) for details.

**2. Identity-element padding.**
Fill pad positions with the identity element of the reduce operation before data reaches the [Intra-Slice Block](./intra-slice-block.md).
The [Fetch Engine's padding adapter](../../moving-tensors/fetch-engine.md) can set a `pad_value` during fetch.

| Operation | Identity Element |
|-----------|-----------------|
| `AddSat` / `Add` | `0` / `0.0` |
| `Max` | `i32::MIN` / `f32::NEG_INFINITY` |
| `Min` | `i32::MAX` / `f32::INFINITY` |

This does not work when the reduce operation is **composed** with a preceding non-invertible transformation.
For example, `exp(x) + exp(y) + ...` (sum of exponentials): there is no value `p` such that `exp(p) = 0` (the additive identity), so padding with any value produces an incorrect contribution.

**3. Other methods.**
NAN masking via ExecutionId and per-slice SFR override (`stosfr`/`itosfr`) are additional options, but they are not covered in this page.

## Performance

### Throughput

The 2-level tree reduce is fully pipelined within the Intra-Slice Block pipeline.
Each input flit passes through the tree in one pipeline stage, adding no extra per-flit throughput cost.

### Latency

The reduce must accumulate all input flits for a reduction group before emitting the result.
If the reduce axis spans `n` time steps, the first output flit is delayed by `n` flit cycles beyond the normal pipeline latency.
In a multi-engine pipeline, this accumulation delay can stall downstream engines waiting for the first flit.

This page covers blocking mode only. Non-blocking mode, ExecutionId-based NAN masking, and per-slice SFR override are outside its scope.
