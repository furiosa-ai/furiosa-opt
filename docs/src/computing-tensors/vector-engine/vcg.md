# Valid Count Generator

The Valid Count Generator (VCG) handles padding exclusion for intra-slice reduce by tagging each 8-element packet with a `valid_count` (abbreviated `vc`), telling the [Intra-Slice Reduce stage](./intra-slice-reduce.md) how many elements are real data.
When a reduce axis is padded, the extra positions contain arbitrary data that the reduction must exclude (see [Padding Strategy](./intra-slice-reduce.md#padding-strategy)).

The [intra-slice reduce API](./intra-slice-reduce.md#interface) takes a `REDUCE_LABEL` that identifies the axis to reduce (e.g., `Ident::R`).
When that axis is padded, the VCG automatically determines which elements are real data and which are padding, based on how the axis distributes across Slice, Time, and Packet.

Throughout this page, capitalized `Slice`, `Time`, and `Packet` refer to the mapping dimensions, and lowercase `slice` and `time step` refer to individual runtime instances within a cluster.

The compiler configures the VCG automatically, so no manual setup is needed.
The VCG requires the reduce axis to be padded, split, and distributed across Slice, Time, and Packet according to specific rules.
[Distribution Rules for `R`](#distribution-rules-for-r) lists the rules, followed by concrete examples for each placement.
For the underlying hardware mechanism, see the [Architecture](#architecture) section below.

## Interface

### Two Modes

The VCG operates in one of two scenarios, derived from where `R` sits in the mapping rather than selected by a hardware mode bit.
The hardware machinery is uniform (see [Validity Decision](#validity-decision)), so "mode" here is shorthand for the two qualitatively different valid-count patterns that the placement produces.

- **Packet Reduce Mode**: `R` appears in Packet, for example `Packet = m![R # 24 % 8]`.
  The VCG assigns a per-packet `valid_count` from 0 to 8, telling the `IntraSliceReduce` stage how many packet elements are real data.
- **Time Reduce Mode**: `R` does not appear in Packet, only in Slice and/or Time.
  The VCG makes a binary valid-or-invalid decision per flit.

Sketching the two output patterns side by side:

```text
Packet Reduce Mode: per-flit count (0..8)        Time Reduce Mode: per-flit gate (valid or not)
  flit 0: [████████]  vc = 8 (full)                flit 0: ✓  (whole flit valid)
  flit 1: [████████]  vc = 8                       flit 1: ✓
  flit 2: [███     ]  vc = 3 (partial)             flit 2: ✗  (whole flit gated off)
```

A *boundary slice* is the single slice that straddles the valid/padding edge of `R`, so only part of its time steps hold real data.
The formal definition lives in [Gate Dims](#gate-dims-per-flit-binary-validity).

### Distribution Rules for `R`

To use the VCG, pad the reduce axis to a hardware-aligned size, split it into sub-expressions, and distribute those sub-expressions across Slice, Time, and Packet according to the rules below.

At the top level, `R # p` splits into an outer and an inner part:

```text
R # p -> R # p / k (outer), R # p % k (inner)
```

Each part assigns to one of the three hardware dimensions.
Within a dimension, a part splits again recursively.

The rule depends on where `R` appears.
Each bullet states the rule first, then the reason that ties it back to the hardware (see [Why Limitations Arise](#why-limitations-arise) for the full derivation).
"Stride" below refers to the mapping-expression stride (see [Stride and Modulo](../../mapping-tensors/mapping-expressions.md#stride-and-modulo)).

- **Slice**: when `R` appears multiple times in Slice, the stride order must increase from inner to outer.
  This keeps each slice's `R` range contiguous, so all of `R`'s Slice sub-expressions can be tracked by one [gate dim](#gate-dims-per-flit-binary-validity) with a single threshold.
- **Time**: multiple `R` sub-expressions may appear in any order, and non-reduce axes may sit between them.
  The gate dim sums the counter contributions of all Time sub-expressions to recover `R`'s index, so the sub-expression ordering does not affect the threshold check.
- **Packet**: Packet must still pad to `# 8`. `R` appears at most once in Packet, and it must be the innermost `% k` part occupying the packet prefix.
  The packet count marks a contiguous prefix valid (see [Packet Dim](#packet-dim-packet-level-valid-count)), so `R` must own the prefix.
  Otherwise the count cannot reach `R`'s positions without also marking interleaved non-`R` positions as valid.

<details>
<summary>Example: why Slice stride order matters</summary>

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
```

</details>

## Examples

The table below maps each `R` placement to its example and shows whether the VCG supports it.
The Intended Mode column shows which mode the placement maps to.
Placements that put `R` only in Slice are absent: intra-slice reduce eliminates `Time` and `Packet` factors only, so a Slice-only `R` cannot be reduced (see [`REDUCE_LABEL`](./intra-slice-reduce.md#interface)).

| Placement | Intended Mode | Example | Supported |
|-----------|---------------|---------|-----------|
| [Slice + Time (standard)](#slice--time-standard) | Time Reduce | `Slice = m![X, R # 24 / 3]`, `Time = m![R # 24 % 3]` | Yes |
| [Time only](#time-only) | Time Reduce | `Time = m![R # 16]` | Yes |
| [Slice + Time (transposed, supported)](#transposed-supported) | Time Reduce | `Slice = m![X, R # 8 % 4]`, `Time = m![R # 8 / 4]` | Yes |
| [Slice + Time (transposed, not supported)](#transposed-not-supported) | Time Reduce | `Slice = m![X, R # 20 % 4]`, `Time = m![R # 20 / 4]` | No |
| [Non-outer/inner ordering](#non-outerinner-ordering-not-supported) | Time Reduce | `Slice = m![X, R # 16 / 2 % 4, R # 16 / 8]` | No |
| [Packet only](#packet-only) | Packet Reduce | `Packet = m![R # 8]` | Yes |
| [Time + Packet](#time--packet) | Packet Reduce | `Time = m![R # 24 / 8]`, `Packet = m![R # 24 % 8]` | Yes |
| [Time + Packet (Packet not innermost)](#time--packet-packet-not-innermost-not-supported) | Packet Reduce | `Time = m![R # 24 % 8]`, `Packet = m![R # 24 / 8 # 8]` | No |
| [Time + Packet (mixed Packet axes)](#time--packet-mixed-packet-axes-not-supported) | Packet Reduce | `Packet = m![R # 24 % 4, A]` | No |
| [Slice + Packet](#slice--packet-not-supported) | Packet Reduce | `Slice = m![R # 2048 / 8]`, `Packet = m![R # 2048 % 8]` | No |

### Time Reduce Mode

In Time Reduce Mode, `R` does not appear in Packet and the VCG makes a binary valid-or-invalid decision per flit.

#### Slice + Time (Standard)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![A]>()
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


#### Slice + Time (`R` Split Into Multiple Sub-Expressions in Slice)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
// Slice ordering check (ascending stride from inner to outer, see
// [Distribution Rules for R](#distribution-rules-for-r)):
//   R # 16 / 2 % 4  stride = 2  (inner)
//   R # 16 / 8       stride = 8  (outer)
//   2 < 8 ✓
//
// Per-slice R indices (within one X group):
//   S0: 0,1  S1: 2,3  S2: 4,5  S3: 6,7  S4: 8,9  S5: 10,11  S6: 12,13  S7: 14,15
// Time Reduce Mode:
//   Boundary slice = floor(13 / 2) = 6. Valid time steps = 13 mod 2 = 1.
//   S0-S5: all-valid, S6: boundary (1 of 2), S7: all-invalid.
fn reduce_multi_level<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![R # 16 / 8, X, R # 16 / 2 % 4], m![R # 16 % 2], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![R # 16 / 8, X, R # 16 / 2 % 4], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
```

#### Slice + Time (Transposed)

Transposed ordering reverses the standard Slice + Time mapping.
The inner part of `R` goes to Slice and the outer part to Time, so the slice ID represents the inner index and the time step represents the outer index.
Slices beyond the boundary still hold valid data at early time steps.

Use this layout when the mapping forces `R` into the inner Slice position, for example when a non-reduce axis already occupies the outer Slice factor and cannot be moved without an extra transpose.
On the hardware side, this placement is what triggers the [`P_gd = 1` transposed gate mode](#transposed-mode).

The Time factor must cover exactly the rounded-up number of slice-sized chunks needed to span `|R|`.
Over-allocating the Time factor would leave the gate's "below" group claiming more valid time steps than the data has.
Formally, given `R # p` split as `Slice = m![..., R # p % slice_size]`, `Time = m![R # p / slice_size, ...]` (where `slice_size` is the size of `R`'s portion in Slice), this mode is supported only when:

<a id="transposed-time-size-constraint"></a>

$$\text{time\_size} = p / \text{slice\_size} = \lceil |R| / \text{slice\_size} \rceil$$

- \\(|R|\\): original axis size (before padding)
- \\(\text{slice\_size}\\): the size of `R`'s portion in Slice (`R # p % slice_size`)
- \\(\text{time\_size}\\): the size of `R`'s portion in Time (`R # p / slice_size`), which equals \\(p / \text{slice\_size}\\)

The hardware reason is that transposed mode's "below" group always claims `time_size` all-valid steps (see [Transposed mode](#transposed-mode)), so `time_size` must match the count of slice-sized chunks `|R|` spans, no more.

The two examples below show the supported and unsupported variants.

##### Transposed (Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // Output: Slice = m![X, R # 8 % 4], Time = m![1], Packet = m![1 # 4]
}
```

##### Transposed (Not Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // ✗ VCG will over-count valid time steps. Use R # 16 instead of R # 20.
}
```

#### Time Only

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![A % 2 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![A % 2 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // Output: Slice = m![X, A / 2], Time = m![1], Packet = m![A % 2 # 4]
    // R eliminated from Time. Time steps 12-15 were gated off.
}
```

#### Time: Flexible Ordering

Multiple `R` sub-expressions in Time may appear in any order, and non-reduce axes may sit between them.
This flexibility contrasts with Slice, where ordering must be ascending.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![Y], m![R # 48 / 8 % 2, X, R # 48 / 8 / 2], m![A # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![Y], m![X], m![A], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![A]>()
        .vector_intra_slice_reduce::<R, m![X], m![A]>(
            IntraSliceReduceOpF32::Add,
        )
    // Output: Slice = m![Y], Time = m![X], Packet = m![A]
    // Both R sub-expressions eliminated from Time; X remains.
}
```

#### Non-Outer/Inner Ordering (Not Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
    // ✗ Non-monotonic slice validity. VCG cannot express this pattern.
}
```

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![1 # 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
    // ✗ Per-slice V values needed. VCG cannot express this pattern.
}
```

### Packet Reduce Mode

In Packet Reduce Mode, `R` appears in Packet and the VCG assigns a per-packet `valid_count` (0-8) that varies by time step.

#### Packet Only

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![R]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // Output: Slice = m![X, A / 2], Time = m![1], Packet = m![1 # 4]
    // R eliminated from Packet. All 3 of 8 elements were counted as valid.
}
```

#### Time + Packet

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![R # 24 % 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // Output: Slice = m![X, A / 2], Time = m![1], Packet = m![1 # 4]
    // R eliminated from both Time and Packet.
}
```

#### Time + Packet (Packet Not Innermost, Not Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, R = 19, X = 64];
// NOT supported: the OUTER part of R is in Packet. Outer part in Packet
// breaks prefix contiguity. See the Packet rule in
// [Distribution Rules for R](#distribution-rules-for-r) for why the
// innermost % k must own the Packet prefix.
//
// R = 19, padded to 24 = 3 * 8. Sub-expressions are swapped:
//   R # 24 / 8 (outer) → Packet (wrong)
//   R # 24 % 8 (inner) → Time   (wrong)
//
// Slice = m![X, A / 2], Time = m![R # 24 % 8], Packet = m![R # 24 / 8 # 8]
// |Slice| = 64 * 4 = 256.
//
// Fix: put the inner part in Packet and the outer part in Time:
//   Time = m![R # 24 / 8], Packet = m![R # 24 % 8]
fn reduce_wrong_packet_outer<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![R # 24 % 8], m![R # 24 / 8 # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![1], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_trim::<m![R # 24 / 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // ✗ Outer part in Packet violates innermost requirement.
}
```

#### Time + Packet (R Fills Fewer Than 8 Positions)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![R # 8 % 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // Output: Slice = m![X, A / 2], Time = m![1], Packet = m![1 # 4]
}
```

#### Time + Packet (Mixed Packet Axes, Not Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![R # 24 % 4, A]>()
        .vector_intra_slice_reduce::<R, m![1], m![A # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // ✗ A's data at positions 4-5 silently excluded by prefix-based vc.
}
```

#### Time + Packet (Perfectly Aligned)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
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
        .vector_narrow_trim::<m![R % 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
    // Output: Slice = m![X, A / 2], Time = m![1], Packet = m![1 # 4]
}
```

#### Slice + Packet (Not Supported)

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 2045];
// NOT supported. Worked example of the Slice + Packet conflict.
// For the formula-based reason, see "Slice + Packet (Not Supported)" under Implementation.
//
// Slice = m![R # 2048 / 8], Time = m![1], Packet = m![R # 2048 % 8]
// Slice = 2048 / 8 = 256.
//
// R = 2045, padded to 2048 = 256 (Slice) * 8 (Packet).
// Slices 0-254 need vc = 8 (full); slice 255 needs vc = 5. The VCG cannot
// produce two partial counts at the same time step.
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
        .vector_narrow_trim::<m![R # 2048 % 4]>()
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
    // ✗ Slice-varying vc needed. VCG cannot express this pattern.
}
```

## Architecture

The following sections explain the hardware mechanics behind the rules and examples above. Readers who only need the API can stop here.


The Valid Count Generator (VCG) hardware computes `vc(s, t)`, the number of valid (non-padding) elements in the flit at slice `s` and time step `t`.
This document describes what the hardware can express, independent of mapping expressions or tensor shapes.
The [Intra-Slice Reduce stage](./intra-slice-reduce.md) consumes VCG tags to exclude padding from reductions.
For how mapping expressions control VCG behavior (supported placements, constraints, and examples), see the [Distribution Rules](#distribution-rules-for-r) and [Examples](#examples) sections above.

Throughout this page, capitalized `Slice`, `Time`, and `Packet` name mapping dimensions, while lowercase `slice` and `time step` name individual runtime instances (one slice is one of the 256 hardware slices, one time step is one flit in that slice's sequence).

### Data Model

Data flows into the Vector Engine as a stream of flits, each containing 8 elements.
The VCG operates at the Vector Engine's input, tagging each 8-way flit with a valid count.
For the 4-way halving downstream, see [Downstream 4-Way Operations](#downstream-4-way-operations).

Two coordinates identify a flit.
A *slice* corresponds to the Slice dimension in the mapping, and a *time step* indexes sequential flits within that slice.

| Coordinate | Range | Meaning |
|------------|-------|---------|
| `s` (slice number) | `[0, num_slices)` | Which slice processes this flit |
| `t` (time step) | `[0, num_flits)` | Sequential position within a slice |

The VCG assigns a *valid count* (abbreviated `vc` in formulas and diagrams) to each flit:

$$\text{vc}(s, t) \in \{0, 1, \ldots, 8\}$$

Element `p` (with `p` in `[0, 8)`) within flit `(s, t)` is valid if and only if `p < vc(s, t)`.
Valid elements always form a *contiguous prefix*, a fundamental hardware constraint.
The VCG cannot express "elements 0, 1, 3 are valid but 2 is not."

### Valid Count Formula

The VCG computes `vc(s, t)` in three stages.
The Sequencer decomposes time index `t` into counters, Original Dimensions maps counters to per-dimension indices, and the Validity Decision combines a packet count with binary gates.

$$t \overset{\text{Sequencer}}{\longrightarrow} (c_0, c_1, \ldots, c_{k-1}) \overset{\text{Original Dims}}{\longrightarrow} \text{idx}(t) \overset{\text{Validity}}{\longrightarrow} \text{vc}(s, t)$$

1. **[Sequencer](#sequencer)**: The flat time index `t` is decomposed into counter values \\((c_0, c_1, \ldots, c_{k-1})\\) via mixed-radix decomposition.
2. **[Original Dimensions](#original-dimensions)**: Each counter is assigned to one of 4 dimensions (packet dim or gate dim 0-2).
   Per-dimension indices are computed as \\(\text{idx}(t) = \sum_{i} c_i \cdot \sigma _ i\\), where \\(\sigma_i\\) is the counter's stride.
3. **[Validity Decision](#validity-decision)**:
   - [Packet Dim](#packet-dim-packet-level-valid-count): produces a packet-level valid count \\(\text{packet\_vc}(t) = \min(\text{stride}_p,\; \max(0,\; V_p - \text{idx}_p(t)))\\).
   - [Gate Dims](#gate-dims-per-flit-binary-validity): each produces a binary gate \\(\text{gate}_d(s, t) \in \{0, 1\}\\) based on slice classification (below/boundary/above a threshold) and the per-dim index.

The final valid count combines these components:

$$\text{vc}(s, t) = \text{packet\_vc}(t) \times \text{gate}_0(s, t) \times \text{gate}_1(s, t) \times \text{gate}_2(s, t)$$

```text
vc(s,t) = packet_vc(t)  ×  gate_0(s,t)  ×  gate_1(s,t)  ×  gate_2(s,t)
          ───────────      ───────────      ───────────      ───────────
          packet dim       gate dim 0       gate dim 1       gate dim 2
          (count 0-8)      (gate 0/1)       (gate 0/1)       (gate 0/1)
```

- If **all** gates are open (= 1), the flit gets `packet_vc(t)` valid elements.
- If **any** gate is closed (= 0), `vc = 0` (entire flit is invalid, regardless of packet dim's count).

### VCG Configuration

VCG configuration is organized around two concepts.
Counters drive the sequencer, and original dimensions decide validity.
Counters produce a flit sequence, and each dim uses its assigned counters to compute an index and decide validity.

The table below lists the configurable parameters, and later sections explain each in detail.

| Field | Scope | Description |
|-------|-------|-------------|
| Counter limits \\(L_0 \ldots L_7\\) | per counter (up to 8) | Sequencer counter limits |
| Original dim assignment | per counter | Which dim (packet / gate 0-2) each counter belongs to |
| stride \\(\sigma_i\\) | per counter | stride for index computation |
| \\(\text{mask}\_{gd}\\) | per gate dim | Slice-id bitmask |
| \\(\text{match}\_{gd}\\) | per gate dim | Threshold for slice classification |
| \\(V_p\\) / \\(V\_{gd}\\) | packet dim / per gate dim | Valid count / threshold |
| \\(P\_{gd}\\) | per gate dim | Standard (0) vs transposed (1) |

Unassigned counters and disabled gate dims (\\(\text{mask}\_{gd} = 0, \text{match}\_{gd} = 1\\)) effectively pass through as "all valid."

### Sequencer

The sequencer interprets the flat time index `t` as a multi-dimensional counter.

#### Counter Structure

Up to 8 nested counters iterate together to produce the flit sequence:

$$t \to (c_0, c_1, \ldots, c_{k-1})$$

where \\(c_0\\) is the fastest (innermost) and \\(c_{k-1}\\) is the slowest (outermost).

Each counter \\(c_i\\) carries a **limit** \\(L_i\\) (cycling through \\(0, 1, \ldots, L_i - 1\\)) and a **stride** \\(\sigma_i\\) that scales the counter's contribution to the dimension index (see [Original Dimensions](#original-dimensions)).
The total flits per slice equal \\(L_0 \times L_1 \times \cdots \times L_{k-1}\\).

<details>
<summary>Example: 3 counters with limits [3, 2, 2]</summary>

This produces 3 * 2 * 2 = 12 flits per slice.
The counters cycle as:

```text
t=0:  (c_0=0, c_1=0, c_2=0)
t=1:  (c_0=1, c_1=0, c_2=0)
t=2:  (c_0=2, c_1=0, c_2=0)
t=3:  (c_0=0, c_1=1, c_2=0)   <- c_0 wraps, c_1 increments
t=4:  (c_0=1, c_1=1, c_2=0)
t=5:  (c_0=2, c_1=1, c_2=0)
t=6:  (c_0=0, c_1=0, c_2=1)   <- c_1 wraps, c_2 increments
t=7:  (c_0=1, c_1=0, c_2=1)
t=8:  (c_0=2, c_1=0, c_2=1)
t=9:  (c_0=0, c_1=1, c_2=1)
t=10: (c_0=1, c_1=1, c_2=1)
t=11: (c_0=2, c_1=1, c_2=1)
```

`c_0` changes every flit, `c_1` every 3 flits, `c_2` every 6 flits, just like digits in a mixed-radix number.

</details>

The sequencer produces counter values, which the next step maps to [original dimensions](#original-dimensions) and then feeds the [validity decision](#validity-decision).

#### Original Dimensions

Each counter assigns to one of 4 **original dimensions** (packet dim or gate dim 0-2), or stays unassigned.

Let \\(D_d\\) be the set of counters assigned to original dimension `d`.
Each counter contributes to its assigned dimension's index by multiplying its current value by its stride.
Summing all contributions gives the current position within that dimension's data:

$$\text{idx}_d(t) = \sum _ {i \in D_d} c_i(t) \cdot \sigma_i$$

This index tracks the position within that dimension's original data range.
Multiple counters may assign to the same dim, with their contributions summed.

<details>
<summary>Example: Counters mapped to original dimensions</summary>

Suppose 3 counters are configured as follows:

| Counter | Limit | stride | Assigned to |
|---------|-------|--------|-------------|
| c_0 | 3 | 8 | packet dim (W axis) |
| c_1 | 2 | 1 | gate dim 0 (C axis) |
| c_2 | 2 | 1 | gate dim 1 (H axis) |

At time step t=4, which gives (c_0=1, c_1=1, c_2=0):
- `idx_p = 1 * 8 = 8`, position 8 along W
- `idx_g0 = 1 * 1 = 1`, position 1 along C
- `idx_g1 = 0 * 1 = 0`, position 0 along H

Each dim uses its index independently to decide validity.

</details>

### Validity Decision

The validity decision combines the packet-dim count and up to three binary gates into the final `vc(s, t)` via multiplication.
The diagram below traces the pipeline from time index to final valid count, with each stage explained in the subsections that follow.

```text
t (flat time index)
│
├─ mixed-radix decomposition (see Sequencer above)
▼
(c_0, c_1, ..., c_{k-1})              ← counter values
│
├─ each counter assigned to a dim, multiplied by stride σ_i
│  (see Original Dimensions above)
▼
idx_p(t), idx_g0(t), idx_g1(t), idx_g2(t)  ← per-dim indices
│
├─ packet dim: packet_vc = min(stride_p, max(0, V_p - idx_p))
├─ gate dim 0: gate_0 = f(masked_id(s), idx_g0, match_g0, V_g0)
├─ gate dim 1: gate_1 = f(masked_id(s), idx_g1, match_g1, V_g1)
├─ gate dim 2: gate_2 = f(masked_id(s), idx_g2, match_g2, V_g2)
│
▼
vc(s,t) = packet_vc(t) × gate_0(s,t) × gate_1(s,t) × gate_2(s,t)
```

Packet dim and gate dims make qualitatively different judgments:

- **Packet dim** answers: "how many elements in this flit are valid?" (a count, 0-8)
- **Gate dims** each answer: "is this flit valid at all?" (a binary gate, yes or no)

Gate dims act as gates, so packet dim's count takes effect only when all three report "valid".
If any gate reports "invalid", the entire flit gets valid count = 0.

#### Packet Dim (Packet-Level Valid Count)

Packet dim tracks how many of the 8 elements in the current flit are real data versus padding.
As the sequencer steps through `R`, the count starts full, stays full for the bulk of the data, then drops to a partial count at the tail, producing a repeating sawtooth as the pattern restarts.
The formula below makes that pattern precise.

Two parameters control the packet-level valid count:

- \\(V_p\\): the **original valid count** for packet dim, the unpadded size of the data along this dimension.
- \\(\text{stride}_p\\): the **stride of the innermost counter assigned to packet dim**, representing how many flit elements belong to the axis tracked by packet dim.

The per-packet valid count is:

$$\text{packet\_vc}(t) = \min(\text{stride}_p, \max(0, V_p - \text{idx}_p(t)))$$

```text
packet_vc(t) = min( stride_p,         max(0, V_p - idx_p(t) ))
                    ─────────              ─────────────────
                    HW width cap           remaining valid data
```

When the axis fills all 8 flit positions, \\(\text{stride}_p = 8\\) and the formula reduces to \\(\min(8, \ldots)\\).
When the axis occupies only \\(k < 8\\) positions (with the remaining positions padded), \\(\text{stride}_p = k\\) caps the valid count so that only the axis's portion of the flit counts as valid.

Hardware constraints:

- The innermost Packet counter must always assign to packet dim.
- Other counters may also assign to packet dim (e.g., a Time counter for the same axis).
- When no axis assigns to packet dim, `packet_vc` is always 8 (full flit) or 0 (empty flit), making packet dim behave as a binary gate.

As the sequencer advances, \\(\text{idx}_p\\) increases and `packet_vc` decreases, producing a repeating *sawtooth* pattern:

```text
Example 1: V_p = 19, stride_p = 8, counter stride=8, limit=3  (axis fills full 8-way)

flit 0: idx_p =  0 -> packet_vc = min(8, 19 -  0) = 8  (full)
flit 1: idx_p =  8 -> packet_vc = min(8, 19 -  8) = 8  (full)
flit 2: idx_p = 16 -> packet_vc = min(8, 19 - 16) = 3  (partial)

Example 2: V_p = 11, stride_p = 4, counter stride=4, limit=3  (axis fills 4 of 8 positions)

flit 0: idx_p =  0 -> packet_vc = min(4, 11 -  0) = 4  (full within stride)
flit 1: idx_p =  4 -> packet_vc = min(4, 11 -  4) = 4  (full within stride)
flit 2: idx_p =  8 -> packet_vc = min(4, 11 -  8) = 3  (partial)
```

In Example 2, positions 4-7 in each flit are padding, and the \\(\text{stride}_p = 4\\) cap automatically excludes them.

One key property is that `packet_vc` depends only on the sequencer state `t`, not on the slice `s`.
All slices receive the same packet valid count at the same time step.

<details>
<summary>Example: Why packet_vc is slice-independent</summary>

If \\(V_p = 19\\), \\(\text{stride}_p = 8\\), and the packet counter cycles [0, 8, 16], then:
- At t=2 (idx_p=16): packet_vc = 3 for every slice.
- Slice 0 gets vc=3, slice 5 gets vc=3, slice 15 gets vc=3, all the same.

This is because packet dim's formula \\(\min(\text{stride}_p, V_p - \text{idx}_p)\\) has no `s` term.
Gate dims can still make certain slices' final vc = 0 (by reporting invalid),
but they cannot change the packet_vc value itself.

</details>

#### Gate Dims (Per-Flit Binary Validity)

Each gate dim classifies slices into groups by extracting a subset of the slice-id bits via a bitmask and comparing the result against a threshold.
The bitmask \\(\text{mask}\_{gd}\\) selects which bits of the slice-id this gate dim tracks:

$$\text{masked\_id} (s) = s \mathbin{\\&} \text{mask}\_{gd}$$

```text
Example: 16 slices (4-bit slice_id), mask_g0 = 0b1100

slice_id (4 bits):   [ b3  b2  b1  b0 ]
mask_g0 = 0b1100:    [  1   1   0   0 ]
                      ─────────────────
masked_id:           [ b3  b2   0   0 ]  → extracts the upper 2 bits
```

Slices fall into three groups based on comparing \\(\text{masked\_id}\\) with \\(\text{match}\_{gd}\\):

| Group | Condition | Meaning |
|-------|-----------|---------|
| **Below** | \\(\text{masked\_id}(s) < \text{match}\_{gd}\\) | All time steps valid |
| **Boundary** | \\(\text{masked\_id}(s) = \text{match}\_{gd}\\) | Valid when \\(\text{idx}\_{gd}(t) < V\_{gd}\\) |
| **Above** | \\(\text{masked\_id}(s) > \text{match}\_{gd}\\) | Depends on mode (see below) |

The \\(P\_{gd}\\) flag selects between two modes that differ only in the "above" group:

##### Standard mode (\\(P\_{gd} = 0\\)) {#standard-mode}

$$\text{gate}_d(s, t) = \begin{cases} 1 & \text{masked\_id}(s) < \text{match}\_{gd} \\\\ [\text{idx}\_{gd}(t) < V\_{gd}] & \text{masked\_id}(s) = \text{match}\_{gd} \\\\ 0 & \text{masked\_id}(s) > \text{match}\_{gd} \end{cases}$$

Above-threshold slices are **entirely invalid**.
This is the common case, since the Slice factor lays out in ascending order and slices beyond the boundary contain no valid data.

##### Transposed mode (\\(P\_{gd} = 1\\)) {#transposed-mode}

$$\text{gate}_d(s, t) = \begin{cases} 1 & \text{masked\_id}(s) < \text{match}\_{gd} \\\\ [\text{idx}\_{gd}(t) < V\_{gd}] & \text{masked\_id}(s) \ge \text{match}\_{gd} \end{cases}$$

Above-threshold slices get the **same** \\(V\_{gd}\\) check as the boundary, so they are not entirely invalid.
This handles the transposed case where the slice ID encodes the *inner* index.
Slices beyond the boundary still contain valid data at early time steps (the outer index is small enough), and they run out of valid data at the same point as the boundary slice.

To disable a gate dim (make it always valid), set \\(\text{mask}\_{gd} = 0, \text{match}\_{gd} = 1\\).
Then \\(\text{masked\_id} = 0 < 1\\) for all slices, placing every slice in the "below" group.

<details>
<summary>Example: Standard mode, H=5 split into Ho=4 (slice) × Hi=2 (time)</summary>

H=5 is split into `Ho × Hi = 4 × 2` (padded from 5 to 8).
`Ho` is the Slice factor (encoded in slice-id bits), `Hi` is the Time factor (sequencer counter).
Axis index = `Ho * 2 + Hi`.
Valid when index < 5.

Gate dim 0 config: `mask=0b1100` (extracts 2 bits for Ho), `match=2`, `V_g0=1`, standard mode.

16 slices, where `masked_id = (slice_id & 0b1100) >> 2` gives Ho:

| Ho | masked_id | Group | Hi=0 | Hi=1 |
|----|-----------|-------|------|------|
| 0 | 0 | below (< 2) | valid | valid |
| 1 | 1 | below (< 2) | valid | valid |
| 2 | 2 | boundary (= 2) | valid (idx=0 < 1) | invalid (idx=1 >= 1) |
| 3 | 3 | above (> 2) | invalid | invalid |

Ho=0,1: both time steps valid (index 0-3, all < 5).
Ho=2: only first time step (index 4 < 5), second invalid (index 5 >= 5).
Ho=3: fully invalid (index 6, 7 >= 5).

</details>

<details>
<summary>Example: Transposed mode, H=5 split into Ho=4 (slice, inner) × Hi=2 (time, outer)</summary>

H=5 is split into `Ho × Hi = 4 × 2` (padded from 5 to 8), but transposed: Ho is the inner factor, Hi is the outer factor.
Axis index = `Hi * 4 + Ho`.
Valid when index < 5.

Gate dim 0 config: `match=1` (= 5 mod 4), `V_g0=1` (= floor(5/4)), transposed mode.

| Ho | masked_id | Group | Hi=0 | Hi=1 |
|----|-----------|-------|------|------|
| 0 | 0 | below (< 1) | valid | valid |
| 1 | 1 | boundary (= 1) | valid (idx=0 < 1) | invalid (idx=1 >= 1) |
| 2 | 2 | above (> 1) | valid (idx=0 < 1) | invalid (idx=1 >= 1) |
| 3 | 3 | above (> 1) | valid (idx=0 < 1) | invalid (idx=1 >= 1) |

Verify: Ho=0, Hi=0: 0 < 5, Hi=1: 4 < 5, so 2 steps.
Ho=1, Hi=0: 1 < 5, Hi=1: 5 >= 5, so 1 step.
Ho=2, Hi=0: 2 < 5, Hi=1: 6 >= 5, so 1 step.
Ho=3, Hi=0: 3 < 5, Hi=1: 7 >= 5, so 1 step.

Unlike standard mode, the "above" group (Ho=2,3) still gets V_g0=1 valid time steps, not zero.

</details>

<details>
<summary>Example: Full VCG computation for [H=5, C=5, W=19], step-by-step build-up</summary>

This example builds up from one axis to three, so each dimension's contribution is clear.

Original shape `[H, C, W] = [5, 5, 19]`.
Each axis is split into a slice part (slice_id) and a time part (sequencer):

```text
H = 5  ->  Ho(slice) * Hi(time) = 4 * 2    (padded from 5 to 8)
C = 5  ->  Co(slice) * Ci(time) = 4 * 2    (padded from 5 to 8)
W = 19 ->  Wi(packet)            = 3 * 8    (padded from 19 to 24)
```

##### Step 1: W=19 only (packet dim, no gates)

Ignore H and C for now.
Disable gate dims 0 and 1.
Every slice processes 3 flits (Wi limit=3), and packet dim produces the sawtooth:

```text
packet_vc:  8, 8, 3
              ^     ^
            full   19 - 16 = 3 (partial)
```

Since there are no gates, **every slice gets this exact same pattern**:

```text
All slices, all flits:
flit 0: ████████  (vc=8)
flit 1: ████████  (vc=8)
flit 2: ███       (vc=3)
```

##### Step 2: Add C=5 (packet dim + gate dim 0)

Now enable the C-axis gate (gate dim 0).
C=5 is split into Co(slice, 4 values) * Ci(time, limit 2).
The C-gate uses: `mask=0b0011` (extracts Co from slice_id), `match=2`, `V_g0=1`, standard mode.

Each slice now runs 6 flits: Ci (limit 2) * Wi (limit 3).
The C-gate classifies slices by their Co value:

| Co | Group | Effect |
|----|-------|--------|
| 0 | below (< 2) | gate open: all 6 flits get packet dim's pattern |
| 1 | below (< 2) | gate open: same |
| 2 | boundary (= 2) | gate open for Ci=0, closed for Ci=1 |
| 3 | above (> 2) | gate closed: all 6 flits get vc=0 |

Result per slice (6 flits = 2 Ci groups * 3 Wi flits):

```text
Co=0:  [8,8,3, 8,8,3]    <- both Ci steps valid
Co=1:  [8,8,3, 8,8,3]    <- same
Co=2:  [8,8,3, 0,0,0]    <- Ci=0 valid, Ci=1 gated off
Co=3:  [0,0,0, 0,0,0]    <- entirely gated off
```

Notice the gate's effect.
Some slices go entirely to zero, and the boundary slice loses its second half.
But within the valid flits, the `[8,8,3]` pattern from packet dim is unchanged.

##### Step 3: Add H=5 (full 3-axis, packet dim + gate dim 0 + gate dim 1)

Now enable the H-axis gate (gate dim 1).
H=5 is split into Ho(slice, 4 values) * Hi(time, limit 2).
The H-gate uses: `mask=0b1100` (extracts Ho from slice_id), `match=0b1000`, `V_g1=1`, standard mode.

Slice ID encodes both slice factors: `slice_id = Ho * 4 + Co`, giving 16 slices.
Each slice now runs 12 flits: Hi (limit 2) * Ci (limit 2) * Wi (limit 3).

| Dim | Axis | What it tracks | VCG config |
|-----|------|----------------|------------|
| packet | W=19 | element count in packet | V_p=19, stride_p=8, counter stride=8, limit=3 |
| gate 0 | C=5 | gate: is Co within valid range? | mask=0b0011, match=2, V_g0=1, standard |
| gate 1 | H=5 | gate: is Ho within valid range? | mask=0b1100, match=0b1000, V_g1=1, standard |

The H-gate classifies slices by Ho, same logic as C-gate by Co:

| Ho | Group | Effect |
|----|-------|--------|
| 0 | below | H-gate open |
| 1 | below | H-gate open |
| 2 | boundary | H-gate open for Hi=0, closed for Hi=1 |
| 3 | above | H-gate closed |

The **final vc** for each flit is `packet_vc(t) * C_gate(s,t) * H_gate(s,t)`.
Both gates must be open for packet dim's count to survive.

The complete heatmap (16 slices * 12 flits).
Columns are slices grouped by Ho.
Rows are flits grouped by (Hi, Ci).
Right-side annotations show which gates are active for each row.

A way to read this without getting lost is to scan the right-side `H:` / `C:` annotations first to predict which row × column blocks should be all-zero (any gate `x`) versus carry data.
Then read the cells to confirm that data-carrying blocks repeat the same `[8, 8, 3]` packet-dim sawtooth.

```text
                     Ho=0       |Ho=1       |Ho=2       |Ho=3
                Co:  0  1  2  3 | 0  1  2  3| 0  1  2  3| 0  1  2  3
  H-gate:            v  v  v  v | v  v  v  v| >  >  >  >| x  x  x  x
  C-gate:            v  v  >  x | v  v  >  x| v  v  >  x| v  v  >  x
--------------------------------------------------------------------------------
 t= 0  Hi=0,Ci=0  W  8  8  8  0 | 8  8  8  0| 8  8  8  0| 0  0  0  0  H:v C:v
 t= 1             |  8  8  8  0 | 8  8  8  0| 8  8  8  0| 0  0  0  0
 t= 2             |  3  3  3  0 | 3  3  3  0| 3  3  3  0| 0  0  0  0
                                |           |           |
 t= 3  Hi=0,Ci=1  W  8  8  0  0 | 8  8  0  0| 8  8  0  0| 0  0  0  0  H:v C:>
 t= 4             |  8  8  0  0 | 8  8  0  0| 8  8  0  0| 0  0  0  0
 t= 5             |  3  3  0  0 | 3  3  0  0| 3  3  0  0| 0  0  0  0
                                |           |           |
 t= 6  Hi=1,Ci=0  W  8  8  8  0 | 8  8  8  0| 0  0  0  0| 0  0  0  0  H:> C:v
 t= 7             |  8  8  8  0 | 8  8  8  0| 0  0  0  0| 0  0  0  0
 t= 8             |  3  3  3  0 | 3  3  3  0| 0  0  0  0| 0  0  0  0
                                |           |           |
 t= 9  Hi=1,Ci=1  W  8  8  0  0 | 8  8  0  0| 0  0  0  0| 0  0  0  0  H:> C:>
 t=10             |  8  8  0  0 | 8  8  0  0| 0  0  0  0| 0  0  0  0
 t=11             |  3  3  0  0 | 3  3  0  0| 0  0  0  0| 0  0  0  0

v = open (below threshold)   > = boundary (partial)   x = closed (above)
```

Reading the patterns:

- **Ho=3 columns** (rightmost 4): all 0.
  H-gate `x` (above threshold, always closed).
- **Co=3 columns** (every 4th): all 0.
  C-gate `x`.
- **Co=2 columns** (H:v C:`>`): C-gate is boundary, so only rows with Ci=0 pass.
  Compare Co=1 vs Co=2 to see the gate's effect.
- **Ho=2 columns** (H:`>` C:v): H-gate is boundary, so only rows with Hi=0 pass.
  Compare Ho=1 vs Ho=2.
- **Ho=2 * Co=2** (both `>`): only (Hi=0, Ci=0) rows pass, the intersection of both boundaries.
- **Within valid cells**: the `[8, 8, 3]` sawtooth from packet dim always appears, the same regardless of slice.

</details>

### What Patterns Are Expressible

The [valid count formula](#valid-count-formula) is a product of four independent terms, and that multiplicative structure determines which `vc(s, t)` functions the hardware produces.

| Placement | Dim | Supported? | Key constraint |
|-----------|-----|------------|----------------|
| Packet only | packet | ✅ | none |
| Time only | gate | ✅ | none |
| Slice only | gate | ✅ | none |
| Slice + Time (standard) | gate | ✅ | none |
| Slice + Time (transposed) | gate | ✅ | `time_count = ⌈n / slice_count⌉` |
| Packet + Time | packet | ✅ | none |
| Slice + Packet | packet + gate | ❌ | `packet_vc(t)` cannot vary by slice |
| Slice + Time + Packet | packet + gate | ❌ | same as Slice + Packet |

The sections below explain why each boundary exists.

#### Why Limitations Arise

Each limitation traces back to a specific part of the formula.

- **Packet dim cannot see slice-id**: The [packet dim formula](#packet-dim-packet-level-valid-count) `packet_vc(t) = min(stride_p, max(0, V_p − idx_p(t)))` depends only on `t`.
  If two slices need different partial counts at the same time step, packet dim cannot produce both:

  ```text
  Suppose we need:  vc(s=0, t=0) = 8,  vc(s=1, t=0) = 3
                                         ───────────────
                                         packet dim would need to output
                                         both 8 and 3 at t=0, impossible
  ```

- **Each gate classifies slices by a single threshold after masking**: [Gate dims](#gate-dims-per-flit-binary-validity) first apply a bitmask to the slice-id (`masked_id = slice_id & mask`), then compare against one value `match`.
  This produces three contiguous groups (below, boundary, above).
  A gate cannot express "slices 0, 3, 7 are valid but 1, 2 are not".
  It represents only contiguous ranges of `masked_id`.
  The mask selects which bits of the slice-id to inspect, letting one gate track a specific axis even when the slice-id encodes multiple axes.

- **At most 4 independent checks**: one packet count (packet dim) plus three binary gates (gate dims) gives 4 orthogonal dimensions total.

#### Single-Axis Scenarios

A padded axis (original size `n`, padded to `n' > n`) occupies some combination of three positions, namely **Packet** ([packet dim](#packet-dim-packet-level-valid-count)), **Time** ([sequencer](#sequencer)), and **Slice** ([gate dims](#gate-dims-per-flit-binary-validity) via slice-id bits).

##### Single Position (Packet, Time, or Slice)

An axis in a single position maps directly to one VCG component, and all three single-position placements are always supported.

- **Packet only** → packet dim handles the sawtooth (see [Packet Dim examples](#packet-dim-packet-level-valid-count)).
- **Time only** → [gate dim](#gate-dims-per-flit-binary-validity) with `mask=0, match=0` (all slices are boundary, binary validity by time step).
- **Slice only** → [gate dim](#gate-dims-per-flit-binary-validity) with appropriate mask/match (all time steps within a valid slice pass, invalid slices are fully gated).

##### Slice + Time

One axis splits between slice-id bits and sequencer counters.
Slice + Time is the typical use of [gate dims](#gate-dims-per-flit-binary-validity), making it the most common VCG configuration.

In **standard** ordering (slice outer, time inner), axis index = `Ho × time_count + Hi`.

<details>
<summary>Example: H=14, Ho=8 (slice) × Hi=3 (time), standard mode</summary>

Gate dim config: `match = ⌊14/3⌋ = 4`, `V = 14 mod 3 = 2`.

```text
Ho  masked_id  group       Hi=0         Hi=1         Hi=2
──  ─────────  ─────────   ──────────   ──────────   ──────────
0   0          below       idx=0 < 2 ✅  idx=1 < 2 ✅  always ✅
1   1          below       always ✅     always ✅     always ✅
2   2          below       always ✅     always ✅     always ✅
3   3          below       always ✅     always ✅     always ✅
4   4          boundary    idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
5   5          above       ❌            ❌            ❌
6   6          above       ❌            ❌            ❌
7   7          above       ❌            ❌            ❌
```

"Below" is genuinely all-valid because `Ho × 3 + Hi < 4 × 3 = 12 ≤ 14` for all `Hi ∈ [0,3)`.
Standard mode tolerates over-allocated `time_count`, and the "below" interpretation remains correct.

</details>

In **transposed** ordering (time outer, slice inner), the gate dim uses [transposed mode](#transposed-mode) and the worked example below illustrates the resulting validity pattern.

<details>
<summary>Example: H=19, Ho=8 (slice, inner) × Hi=3 (time, outer), transposed mode</summary>

Gate dim config: `match = 19 mod 8 = 3`, `V = ⌊19/8⌋ = 2`, transposed mode.

```text
Ho  masked_id  group       Hi=0         Hi=1         Hi=2
──  ─────────  ─────────   ──────────   ──────────   ──────────
0   0          below       always ✅     always ✅     always ✅
1   1          below       always ✅     always ✅     always ✅
2   2          below       always ✅     always ✅     always ✅
3   3          boundary    idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
4   4          above       idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
5   5          above       idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
6   6          above       idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
7   7          above       idx=0 < 2 ✅  idx=1 < 2 ✅  idx=2 < 2 ❌
```

Verify against real data (axis index = `Hi × 8 + Ho`, valid when < 19):
- Ho=0, Hi=2: `2×8 + 0 = 16 < 19` ✅, "below" gives all-valid = 3 steps, need `V+1 = 3` steps ✅
- Ho=3, Hi=2: `2×8 + 3 = 19 ≥ 19` ❌, boundary gives 2 steps ✅
- Ho=7, Hi=1: `1×8 + 7 = 15 < 19` ✅, "above" gives V=2 steps, actual need is 2 steps ✅

The "below" group gets `time_count` valid steps from the HW "all-valid" interpretation, so `time_count` must equal `V + 1 = ⌈n / slice_count⌉`.
For the user-facing statement of this constraint, see [transposed time_size constraint](#transposed-time-size-constraint).

</details>

##### Packet + Time

When an axis spans both Packet and Time, both factors assign to **packet dim**, and multiple counters then contribute to `idx_p(t)` as described in [Original Dimensions](#original-dimensions).

<details>
<summary>Example: n=50, two counters on packet dim, contiguous (`stride_outer = 8 × 3 = 24`)</summary>

```text
c_inner (limit=3, stride=8):  packet counter
c_outer (limit=3, stride=24): time counter     (24 = 8 × 3 ✅ contiguous)

idx_p = c_outer × 24 + c_inner × 8

          c_inner=0    c_inner=1    c_inner=2
          ─────────    ─────────    ─────────
c_outer=0  idx_p=0→8   idx_p=8→8    idx_p=16→8
c_outer=1  idx_p=24→8  idx_p=32→8   idx_p=40→8
c_outer=2  idx_p=48→2  idx_p=56→0   idx_p=64→0
                  ↑
                  min(8, 50-48)=2
```

Packet dim handles both the within-flit and across-flit boundaries.

</details>

##### Slice + Packet (Not Supported)

One axis splits between **slice** (gate dim) and **packet** (packet dim).
This directly violates the **slice-independent packet count** constraint (see the [Packet Dim](#packet-dim-packet-level-valid-count) key property).

`packet_vc(t)` depends only on `t`, but the boundary slice (the one where only some of `R` is valid) needs a different partial count than all-valid slices.
The gate multiplies by 0 or 1, so it fully closes a flit but **cannot change the partial count**.

<details>
<summary>Example: n=10, stride_p=8, slice_count=2</summary>

```text
What we need:
  Ho=0: elements 0-7,   all valid     →  vc = 8
  Ho=1: elements 8-15,  first 2 valid →  vc = 2
                                           ─
                                           partial count, different from 8

Attempt 1: set `V_p = 10`:
  packet_vc = min(8, 10-0) = 8         for ALL slices
  Ho=0:  vc = 8 × 1 = 8  ✅
  Ho=1:  vc = 8 × 1 = 8  ❌  (need 2, not 8)
         vc = 8 × 0 = 0  ❌  (gate can close to 0, not to 2)

Attempt 2: set `V_p = 2`:
  packet_vc = min(8, 2-0) = 2          for ALL slices
  Ho=0:  vc = 2 × 1 = 2  ❌  (need 8, not 2)

No single V_p works.  Packet dim produces one value, and the gate can only multiply by 0 or 1.
```

</details>

Two degenerate cases avoid this conflict.
When `n % stride_p = 0`, every packet is either fully valid or fully invalid, so packet dim produces no partial counts and a gate alone handles validity.
This reduces to **Slice only**, not a true Slice + Packet scenario.
Similarly, `n <= stride_p` means a single flit covers the entire axis, reducing to **Packet only**.

When the VCG cannot express the required pattern, [Padding Strategy](./intra-slice-reduce.md#padding-strategy) provides alternatives.

##### Slice + Time + Packet (Not Supported)

The axis spans all three positions.
The Slice + Packet conflict carries over, since the boundary slice still needs a different partial count than all-valid slices, and `packet_vc(t)` still cannot vary by slice.

The same degenerate exception applies, where `n % stride_p = 0` eliminates partial counts and reduces this case to **Slice + Time** (packet dim unused).

#### Multiple Axes

Each padded axis that needs validity tracking consumes one [original dimension](#original-dimensions) slot.
The reduce axis `R` always needs a slot, and any non-reduce axis that is padded also needs one so the VCG can gate out its padding before the reduction sees it.
Intra-slice reduce takes a single `REDUCE_LABEL`, so multi-axis here means one reduce axis plus extra padded non-reduce axes, not multiple reductions in one Tensor Unit invocation.

| Resource | Capacity | Notes |
|----------|----------|-------|
| **[Packet Dim](#packet-dim-packet-level-valid-count)** (packet count) | 1 slot | Innermost counter determines `stride_p` |
| **[Gate Dims](#gate-dims-per-flit-binary-validity)** (binary gates) | 3 slots | One gate per padded axis |
| **Unpadded axes** | free | No dim needed (`mask=0, match=1`) |

When the packet axis is fully aligned (`n % stride_p = 0`), `packet_vc` is constant and packet dim is effectively unused, so it can be repurposed as a gate for another axis.

#### Summary

A valid count function `vc(s, t)` is VCG-expressible if and only if:

1. **Prefix property**: Valid elements form a contiguous prefix `[0, vc)` within each flit.
2. **Slice-independent packet count**: `packet_vc(t)` is the same across all slices at the same `t`.
   Slices can be gated to `vc = 0`, but cannot receive a different partial count.
3. **Monotonic slice ordering**: Each gate dim classifies slices by a single threshold on `masked_id`.
4. **At most 4 orthogonal dimensions**: 1 packet count plus 3 binary gates.

For mapping-level code examples of each placement, see [Examples](#examples).
For unsupported cases, see [Padding Strategy](./intra-slice-reduce.md#padding-strategy).

### Downstream 4-Way Operations

The VCG assigns valid counts per 8-way flit, but the intra-slice pipeline operates on 4-way halves after a `Narrow` stage.

| Operation | Input | Output | Valid Count Transformation |
|-----------|-------|--------|---------------------------|
| **split_way4** | 8-way flit (vc = v) | two 4-way flits | `vc_low = min(v, 4)`, `vc_high = max(v - 4, 0)` |
| **trim_way4** | 8-way flit (vc = v) | one 4-way flit | `vc = v` (requires v <= 4) |
| **concat_way8** | two 4-way flits | 8-way flit | `vc = vc_low + vc_high` |
| **pad_way8** | 4-way flit | 8-way flit | `vc` unchanged |

Split and concat preserve the prefix property.
For trim_way4, the mapping must statically guarantee `v <= 4`.
If the upper 4 elements could be valid, trimming them would lose data.
