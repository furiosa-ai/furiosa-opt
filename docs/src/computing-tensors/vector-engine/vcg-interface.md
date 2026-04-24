# Valid Count Generator's Interface

## Overview

Recall that when a reduce axis is padded, the extra positions contain arbitrary data that must be excluded from the reduction (see [Padding Strategy](./intra-slice-reduce.md#padding-strategy)).
The Valid Count Generator (VCG) solves this at the hardware level: it tags each 8-element packet with a `valid_count` (abbreviated `vc`), telling the [Intra-Slice Reduce stage](./intra-slice-reduce.md) how many elements are real data.

The [intra-slice reduce API](./intra-slice-reduce.md#interface) takes a `REDUCE_LABEL` that identifies the axis to reduce (e.g., `Ident::R`).
When that axis is padded, the VCG automatically determines which elements are real data and which are padding, based on how the axis is distributed across Slice, Time, and Packet.

The VCG requires the reduce axis to be structured in a specific way (padded, split, and distributed across Slice, Time, and Packet).
The distribution rules are described in [How `R` Should Be Distributed](#how-r-should-be-distributed), followed by concrete examples for each placement.
The VCG is configured automatically by the compiler; no manual setup is needed.
For the underlying hardware mechanism, see [Valid Count Generator's Implementation](./vcg-implementation.md).

## Quick Reference

If you are checking whether a mapping is supported, start with this table and the [examples](#examples-time-reduce-mode) below.

| Placement | Mode | Example | Supported |
|-----------|------|---------|-----------|
| [Slice + Time](#slice--time) | Time Reduce | `Slice = m![X, R # 24 / 3]`, `Time = m![R # 24 % 3]` | Yes |
| [Time only](#time-only) | Time Reduce | `Time = m![R # 16]` | Yes |
| [Slice + Time (transposed, supported)](#slice--time-transposed-supported) | Time Reduce | `Slice = m![X, R # 8 % 4]`, `Time = m![R # 8 / 4]` | Yes |
| [Slice + Time (transposed, not supported)](#slice--time-transposed-not-supported) | Time Reduce | `Slice = m![X, R # 20 % 4]`, `Time = m![R # 20 / 4]` | No |
| [Non-outer/inner ordering](#non-outerinner-ordering-not-supported) | Time Reduce | `Slice = m![X, R # 16 / 2 % 4, R # 16 / 8]` | No |
| [Packet only](#packet-only) | Packet Reduce | `Packet = m![R # 8]` | Yes |
| [Time + Packet](#time--packet) | Packet Reduce | `Time = m![R # 24 / 8]`, `Packet = m![R # 24 % 8]` | Yes |
| [Time + Packet (Packet not innermost)](#time--packet-packet-not-innermost-not-supported) | Packet Reduce | `Time = m![R # 24 % 8]`, `Packet = m![R # 24 / 8 # 8]` | No |
| [Time + Packet (mixed Packet axes)](#time--packet-mixed-packet-axes-not-supported) | Packet Reduce | `Packet = m![R # 24 % 4, A]` | No |
| [Slice + Packet](#slice--packet-not-supported) | Packet Reduce | `Slice = m![R # 2048 / 8]`, `Packet = m![R # 2048 % 8]` | No |

## Two Modes

Depending on whether `R` appears in Packet, the VCG operates in one of two exclusive modes:

**Packet Reduce Mode**: `R` appears in Packet, for example `Packet = m![R # 24 % 8]`.
The VCG assigns a per-packet `valid_count` from 0 to 8, so the `IntraSliceReduce` stage knows how many packet elements are real data.

**Time Reduce Mode**: `R` does not appear in Packet, only in Slice and/or Time.
The VCG makes a binary valid-or-invalid decision per flit.

## How `R` Should Be Distributed

All examples in this document assume the same high-level pattern: first pad the reduce axis, then split it, then place the resulting sub-expressions across Slice, Time, and Packet.

At the top level, `R # p` is split into an outer and an inner part:

```text
R # p -> R # p / k (outer), R # p % k (inner)
```

Each part is then assigned to one of the three hardware dimensions.
Within a dimension, a part can be split again recursively.

The VCG rule depends on where `R` appears:

- **Slice**: when `R` appears multiple times in Slice, the stride order must increase from inner to outer. This keeps each slice's `R` range contiguous.
- **Time**: multiple `R` sub-expressions can appear in any order, and non-reduce axes may sit between them.
- **Packet**: Packet must still be padded to `# 8`. `R` may appear at most once in Packet, and it must be the innermost `% k` part occupying the packet prefix.

<details>
<summary>Example: why Slice stride order matters</summary>

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![R = 13, X = 32];
// R # 16, split into 2 * 4 * 2:
//   Slice = m![R # 16 / 8, X, R # 16 / 2 % 4]
//   Time  = m![R # 16 % 2]
//
// Slice strides for R:
//   R # 16 / 2 % 4  -> stride = 2  (inner)
//   R # 16 / 8      -> stride = 8  (outer)
//   2 < 8, so each slice receives a contiguous R interval.
fn example_stride_ordering<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![R # 16 / 8, X, R # 16 / 2 % 4], m![R # 16 % 2], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![R # 16 / 8, X, R # 16 / 2 % 4], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
```

</details>

## Examples: Time Reduce Mode

`R` does **not** appear in Packet.
The VCG decides per-flit: valid or invalid.

### Slice + Time

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 4, R = 17, X = 32];

// The most common pattern. Outer part of R → Slice, inner part → Time.
//
// R = 17, padded to 24 = 8 * 3.
// R # 24 is split into:
//   R # 24 / 3  (size 8, outer) → Slice
//   R # 24 % 3  (size 3, inner) → Time
// This follows the standard outer→Slice, inner→Time pattern.
//
// Slice = m![X, R # 24 / 3], Time = m![R # 24 % 3], Packet = m![A # 8]
// |Slice| = X(32) * 8 = 256.
// Time Reduce Mode: R does not appear in Packet. VCG gates flits by slice and time.
//   Boundary slice = floor(17 / 3) = 5. Valid time steps in boundary = 17 mod 3 = 2.
//   For each X group, R-slices 0-4: all 3 time steps valid.
//   R-slice 5: 2 of 3 valid (boundary). R-slices 6-7: all invalid.
fn reduce_slice_time<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![X, R # 24 / 3], m![R # 24 % 3], m![A # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![X, R # 24 / 3], m![1], m![A], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![A]>()
        .vector_intra_slice_reduce::<R, m![1], m![A]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // Output: Slice = m![X, R # 24 / 3], Time = m![1], Packet = m![A]
    // R eliminated from Time. Boundary slice (R-slice #5) accumulated only their valid steps.
}
```

<details>
<summary>Valid count trace for R = 17 (within one X group)</summary>

| R-slice | Group | t=0 | t=1 | t=2 |
|---------|-------|-----|-----|-----|
| 0 | all-valid | 0 | 1 | 2 |
| 1 | all-valid | 3 | 4 | 5 |
| 2 | all-valid | 6 | 7 | 8 |
| 3 | all-valid | 9 | 10 | 11 |
| 4 | all-valid | 12 | 13 | 14 |
| 5 | boundary | 15 | 16 | **.** |
| 6 | all-invalid | **.** | **.** | **.** |
| 7 | all-invalid | **.** | **.** | **.** |

This pattern repeats identically for each of the 32 X groups (256 total slices).

</details>


### Slice + Time (`R` Split Into Multiple Sub-Expressions in Slice)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![R = 13, X = 32];
// R can appear as multiple sub-expressions in Slice.
//
// R # 16 is first split into outer (/ 2) and inner (% 2) for Slice vs Time:
//   R # 16 / 2  → Slice portion (size 8)
//   R # 16 % 2  → Time portion (size 2)
//
// The Slice portion (R # 16 / 2, size 8) is further split into two sub-expressions:
//   R # 16 / 8      stride = 8, size = 2  (outer)
//   R # 16 / 2 % 4  stride = 2, size = 4  (inner)
//
// Slice = m![R # 16 / 8, X, R # 16 / 2 % 4], Time = m![R # 16 % 2], Packet = m![1 # 8]
// Slice product = 2 * 32 * 4 = 256.
//
// Slice ordering check (inner to outer, ascending stride):
//   R # 16 / 2 % 4  stride = 2, size = 4  (inner)
//   R # 16 / 8       stride = 8, size = 2  (outer)
//   inner_stride(2) * size(4) = 8 = outer_stride ✓
//
// This gives each slice contiguous R indices (within one X group):
//   S0: 0,1  S1: 2,3  S2: 4,5  S3: 6,7  S4: 8,9  S5: 10,11  S6: 12,13  S7: 14,15
// Time Reduce Mode:
//   Boundary slice = floor(13 / 2) = 6. Valid time steps = 13 mod 2 = 1.
//   S0-S5: all-valid, S6: boundary (1 of 2), S7: all-invalid.
fn reduce_multi_level<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![R # 16 / 8, X, R # 16 / 2 % 4], m![R # 16 % 2], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![R # 16 / 8, X, R # 16 / 2 % 4], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
```

### Slice + Time (Transposed)

The reverse of Slice + Time: the inner part of `R` goes to Slice, the outer part to Time.
In transposed mode, the slice ID represents the inner index and the time step the outer index.
Slices beyond the boundary still have valid data at early time steps.

Given `R # p` split as `Slice = m![..., R # p % slice_size]`, `Time = m![R # p / slice_size, ...]` (where `slice_size` is the size of `R`'s portion in Slice), this mode is supported only when:

$$\text{time\_size} = p / \text{slice\_size} = \lceil |R| / \text{slice\_size} \rceil$$

- \\(|R|\\): original axis size (before padding)
- \\(\text{slice\_size}\\): the size of `R`'s portion in Slice (`R # p % slice_size`)
- \\(\text{time\_size}\\): the size of `R`'s portion in Time (`R # p / slice_size`), which equals \\(p / \text{slice\_size}\\)

### Slice + Time (Transposed, Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![R = 5, X = 64];
// Transposed: inner part of R → Slice, outer part → Time.
// (The reverse of standard Slice + Time.)
//
// R = 5, padded to 8 = 4 * 2.
// R # 8 is split into:
//   R # 8 % 4  (size 4, inner) → Slice  (transposed: inner goes to Slice)
//   R # 8 / 4  (size 2, outer) → Time   (transposed: outer goes to Time)
//
// Slice = m![X, R # 8 % 4], Time = m![R # 8 / 4], Packet = m![1 # 8]
// |Slice| = 64 * 4 = 256.
//
// time_size(2) == ceil(5 / 4) = 2 ✓
//
// Time Reduce Mode:
//   Boundary slice = |R| mod slice_size = 5 mod 4 = 1.
//   Valid time steps in boundary = floor(|R| / slice_size) = floor(5/4) = 1.
//   R-slice 0   (< boundary): all 2 time steps valid.
//   R-slice 1   (= boundary): 1 of 2 time steps valid.
//   R-slices 2-3 (> boundary): also 1 of 2 time steps valid (transposed behavior).
fn reduce_transposed<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![X, R # 8 % 4], m![R # 8 / 4], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![X, R # 8 % 4], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // Output: Slice = m![X, R # 8 % 4], Time = m![1], Packet = m![1 # 4]
}
```

### Slice + Time (Transposed, Not Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![R = 14, X = 64];
// NOT supported: time_size is over-allocated.
//
// Slice = m![X, R # 20 % 4], Time = m![R # 20 / 4], Packet = m![1 # 8]
// Slice = 64 * 4 = 256.
//
// R = 14, padded to 20 = 4 (S) * 5 (time_size).
// time_size(5) != ceil(14 / 4) = 4 ✗
//
// R-slice 0 should have 4 valid time steps (indices 0, 4, 8, 12, all < 14).
// But the VCG classifies R-slice 0 as "all-valid" = 5 time steps. WRONG.
//
// a possible Fix: pad to 16 instead, so time_size = ceil(14/4) = 4:
//   Slice = m![X, R # 16 % 4], Time = m![R # 16 / 4], Packet = m![1 # 8]
fn reduce_transposed_wrong<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![X, R # 20 % 4], m![R # 20 / 4], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![X, R # 20 % 4], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // ✗ VCG will over-count valid time steps. Use R # 16 instead of R # 20.
}
```

### Time Only

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, R = 12, X = 64];
// R exists only in Time. VCG gates excess time steps as invalid.
// All slices see the same pattern.
//
// Slice = m![X, A / 2], Time = m![R # 16], Packet = m![A % 2 # 8]
// Slice = 64 * 4 = 256.
//
// R = 12, padded to 16. Time Reduce Mode: time steps 0-11 valid, 12-15 invalid.
fn reduce_time_only<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![X, A / 2], m![R # 16], m![A % 2 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![X, A / 2], m![1], m![A % 2 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![A % 2 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![A % 2 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // Output: Slice = m![X, A / 2], Time = m![1], Packet = m![A % 2 # 4]
    // R eliminated from Time. Time steps 12-15 were gated off.
}
```

### Non-Outer/Inner Ordering (Not Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![R = 13, X = 32];
// NOT supported: reordered sub-expressions in Slice break monotonic validity.
// R's sub-expressions must form a clean outer/inner relationship across dimensions.
// If reordered, the VCG cannot express the resulting validity pattern.
//
// Slice = m![X, R # 16 / 2 % 4, R # 16 / 8], Time = m![R # 16 % 2]
// Slice = 32 * 4 * 2 = 256.
//
// Striped R indices per R-slice group:
//   S0: 0,1  S1: 8,9  S2: 2,3  S3: 10,11  S4: 4,5  S5: 12,13  S6: 6,7  S7: 14,15
//   Validity: valid, valid, valid, valid, valid, partial, valid, invalid
//
// Non-monotonic (S6 valid after S5 partial). The VCG cannot express this.
//
// Fix: use standard ordering instead:
//   Slice = m![X, R # 16 / 8, R # 16 / 2 % 4], Time = m![R # 16 % 2]
fn reduce_wrong_ordering<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![X, R # 16 / 2 % 4, R # 16 / 8], m![R # 16 % 2], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![X, R # 16 / 2 % 4, R # 16 / 8], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
    // ✗ Non-monotonic slice validity. VCG cannot express this pattern.
}
```

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![R = 13, X = 64];
// NOT supported: Time-Slice-Time interleave.
// Slice = m![X, R # 16 / 2 % 4], Time = m![R # 16 / 8, R # 16 % 2]
// Slice = 64 * 4 = 256.
//
// The interleave causes different R-slices to need different valid time step counts:
//   S0: R indices 0,1,8,9   -> 4/4 valid
//   S1: R indices 2,3,10,11 -> 4/4 valid
//   S2: R indices 4,5,12,13 -> 3/4 valid
//   S3: R indices 6,7,14,15 -> 2/4 valid
//
// The VCG has a single threshold, so it cannot express per-slice values.
//
// Fix: standard Slice outer, Time inner:
//   Slice = m![X, R # 16 / 4], Time = m![R # 16 % 4]
fn reduce_wrong_interleave<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![X, R # 16 / 2 % 4], m![R # 16 / 8, R # 16 % 2], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![X, R # 16 / 2 % 4], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
    // ✗ Per-slice V values needed. VCG cannot express this pattern.
}
```

### Time: Flexible Ordering

When `R` is split into multiple sub-expressions within Time, they can appear in any order and non-reduce axes can sit between them (unlike Slice, where ordering is strict).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 4, R = 45, X = 2, Y = 256];
// R's Time portion can be split into multiple sub-expressions (order does not matter).
//
// R # 48 is split into outer (/ 8) and inner (% 8) for Time vs Packet:
//   R # 48 / 8  → Time portion (size 6)
//   R # 48 % 8  → Packet portion (size 8)
//
// The Time portion (R # 48 / 8, size 6) is further split:
//   R # 48 / 8 / 2  (size 3)
//   R # 48 / 8 % 2  (size 2)
// These can appear in any order with non-reduce axes between them.
//
// Slice = m![Y], Time = m![R # 48 / 8 % 2, X, R # 48 / 8 / 2], Packet = m![A # 8]
// |Slice| = 256.
//
// The VCG combines both R sub-expressions' positions to determine validity.
// (Contrast with Slice, where ordering is strict.)
fn reduce_time_split<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![Y], m![R # 48 / 8 % 2, X, R # 48 / 8 % 2], m![A # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![Y], m![X], m![A], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![A]>()
        .vector_intra_slice_reduce::<R, m![X], m![A]>(
            IntraSliceReduceOpF32::Add,
        )
    // Output: Slice = m![Y], Time = m![X], Packet = m![A]
    // Both R sub-expressions eliminated from Time; X remains.
}
```

## Examples: Packet Reduce Mode

`R` appears in Packet.
The VCG assigns a per-packet `valid_count` (0-8) that varies by time step.

### Packet Only

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, R = 3, X = 64];
// R exists only in Packet. Every packet gets the same constant valid_count = |R|.
//
// Slice = m![X, A / 2], Time = m![1], Packet = m![R # 8]
// Slice = 64 * 4 = 256.
//
// R = 3, padded to 8. Packet Reduce Mode: every packet has vc = 3.
// No slice or time variation needed.
fn reduce_packet_only<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![1], m![R # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![1], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![R]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // Output: Slice = m![X, A / 2], Time = m![1], Packet = m![1 # 4]
    // R eliminated from Packet. All 3 of 8 elements were counted as valid.
}
```

### Time + Packet

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, R = 19, X = 64];
// R spans both Time and Packet.
// VCG produces full packets first, then a partial packet at the tail.
//
// Slice = m![X, A / 2], Time = m![R # 24 / 8], Packet = m![R # 24 % 8]
// Slice = 64 * 4 = 256.
//
// R = 19, padded to 24 = 3 (Time) * 8 (Packet).
// Packet Reduce Mode: R fills all 8 Packet positions.
//   t=0: vc = 8 (all valid)
//   t=1: vc = 8 (all valid)
//   t=2: vc = 3 (first 3 valid, last 5 are padding)
// All slices see the same [8, 8, 3] pattern.
fn reduce_time_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![R # 24 / 8], m![R # 24 % 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![1], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![R # 24 % 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // Output: Slice = m![X, A / 2], Time = m![1], Packet = m![1 # 4]
    // R eliminated from both Time and Packet.
}
```

### Time + Packet (Packet Not Innermost, Not Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, R = 19, X = 64];
// NOT supported: R's portion in Packet must be the innermost (lowest) sub-expression.
//
// R = 19, padded to 24 = 3 * 8.
// R # 24 is split into:
//   R # 24 % 8  (size 8, inner), should go to Packet
//   R # 24 / 8  (size 3, outer), should go to Time
// But here they are swapped: the OUTER part (R # 24 / 8) goes to Packet,
// and the INNER part (R # 24 % 8) goes to Time.
//
// Slice = m![X, A / 2], Time = m![R # 24 % 8], Packet = m![R # 24 / 8 # 8]
// |Slice| = 64 * 4 = 256.
//
// The VCG's prefix-based valid count assumes R in Packet is the innermost index.
// When the outer part is in Packet instead, the valid count pattern no longer
// forms a simple decreasing sequence; it would need non-contiguous validity.
//
// Fix: put the inner part in Packet and the outer part in Time:
//   Time = m![R # 24 / 8], Packet = m![R # 24 % 8]
fn reduce_wrong_packet_outer<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![R # 24 % 8], m![R # 24 / 8 # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![1], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![R # 24 / 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // ✗ Outer part in Packet violates innermost requirement.
}
```

### Time + Packet (R fills fewer than 8 positions)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, R = 7, X = 64];
// When R fills fewer than 8 Packet positions, the remaining must be padding, not another axis.
// The valid count is capped at the size of R in Packet.
//
// Slice = m![X, A / 2], Time = m![R # 8 / 4], Packet = m![R # 8 % 4 # 8]
// Slice = 64 * 4 = 256.
//
// R = 7, padded to 8 = 2 * 4. R fills 4 Packet positions, padded to 8-way.
// Packet Reduce Mode: valid count capped at 4 (the size of R in Packet).
//   t=0: vc = 4 (positions 0-3 valid, 4-7 are padding)
//   t=1: vc = 3 (positions 0-2 valid)
// All slices see the same [4, 3] pattern.
//
// Supported: R solely occupies the prefix; positions 4-7 are padding, not another axis.
fn reduce_time_packet_partial<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![R # 8 / 4], m![R # 8 % 4 # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![1], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![R # 8 % 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // Output: Slice = m![X, A / 2], Time = m![1], Packet = m![1 # 4]
}
```

### Time + Packet (Mixed Packet Axes, Not Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 2, R = 19, X = 256];
// NOT supported. R must be the sole occupant of the Packet prefix.
// If another axis shares the Packet, the prefix-based count marks that axis's data as padding.
//
// Slice = m![X], Time = m![R # 24 / 4], Packet = m![R # 24 % 4, A # 8]
// Slice = 256.
//
// R fills positions 0-3, A fills positions 4-5 (padded to 8).
// Packet Reduce Mode: valid count applies to the whole packet as a prefix.
//   vc = 3 means "positions 0-2 valid", but A's real data at positions 4-5
//   is ALWAYS treated as invalid, regardless of A's actual size.
//   The reduce result silently loses A's contributions.
//
// Fix: put A outside Packet, or pad R to fill all 8 positions:
//   Time = m![R # 24 / 8], Packet = m![R # 24 % 8]
fn reduce_wrong_mixed_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![X], m![R # 24 / 4], m![R # 24 % 4, A # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![X], m![1], m![A # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![R # 24 % 4, A]>()
        .vector_intra_slice_reduce::<R, m![1], m![A # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // ✗ A's data at positions 4-5 silently excluded by prefix-based vc.
}
```

### Time + Packet: Perfectly Aligned

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, R = 24, X = 64];
// When |R| is exactly divisible by the size of R in Packet,
// every packet is full and the VCG is not needed.
//
// Slice = m![X, A / 2], Time = m![R / 8], Packet = m![R % 8]
// Slice = 64 * 4 = 256.
//
// R = 24, no padding needed! 24 = 3 * 8.
// Every element is real data. All vc = 8.
fn reduce_time_packet_aligned<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![R / 8], m![R % 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![1], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![R % 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // Output: Slice = m![X, A / 2], Time = m![1], Packet = m![1 # 4]
}
```

### Slice + Packet (Not Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![R = 2045];
// NOT supported. The VCG produces the same valid_count for all slices at a given time step.
// When R spans Slice and Packet, different slices need different counts.
//
// Slice = m![R # 2048 / 8], Time = m![1], Packet = m![R # 2048 % 8]
// Slice = 2048 / 8 = 256.
//
// R = 2045, padded to 2048 = 256 (Slice) * 8 (Packet).
// Packet Reduce Mode:
//   Slices 0-254 need vc = 8 (full). Slice 255 needs vc = 5 (2045 mod 8 = 5).
//   But vc is the same for all slices at the same time step.
//   The VCG cannot produce vc = 8 for some slices and vc = 5 for others.
//
// Fix: add R to Time so R spans Slice + Time instead:
//   Slice = m![R # 2048 / 8], Time = m![R # 2048 % 8], Packet = m![A # 8]
// Another possible fix: if R's size was 2048, no padding is introduced, which does not need VCG at all:
//   Slice = m![R: 2048 / 8], Time: m![1], Packet: m![R % 8]
fn reduce_wrong_slice_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![R # 2048 / 8], m![1], m![R # 2048 % 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![R # 2048 / 8], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_trim_way4::<m![R # 2048 % 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // ✗ Slice-varying vc needed. VCG cannot express this pattern.
}
```
