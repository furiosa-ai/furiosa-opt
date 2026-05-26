# Valid Count Generator

The [Intra-Slice Reduce](./intra-slice-reduce.md) stage reduces the axis identified by `REDUCE_LABEL` (e.g., `R`) across its `Time` and `Packet` factors, leaving any `Slice` factor in the output.
That axis often needs padding to fit hardware dimensions, and the extra padded positions contain arbitrary data that the reduction must exclude.

The Valid Count Generator (VCG) solves this.
The user places `R` as sub-expressions across `Slice`, `Time`, and/or `Packet` in the mapping.
The compiler then configures the VCG to tag each 8-element flit with a `valid_size` count of how many elements are real data.
Each sub-expression in `Time` or `Slice` maps to a sequencer counter assigned to a time filter.
Each sub-expression in `Packet` drives the packet clipper.

Throughout this page, capitalized `Slice`, `Time`, `Packet` refer to mapping dimensions; lowercase `slice` and `time step` refer to runtime instances.

The mapping must express `R` in a specific form for the VCG to work.
`R` is padded to a hardware-aligned size, written `R # PADDED_SIZE` when discussed in general.
Concrete examples use the actual padded value (e.g., `R # 16`, `R # 48`).
Each sub-expression is then a factor of `R # PADDED_SIZE` of the form `R # PADDED_SIZE / n % m` (stride `n`, modulo `m`).
See [Stride and Modulo](../../mapping-tensors/mapping-expressions.md#stride-and-modulo) for `/ n` and `% m` semantics.
Each sub-expression is assigned to one hardware dimension.
One possible distribution is `R = 43` padded to `R # 48`, split across all three dimensions:

```text
Slice:  R # 48 / 8       (stride 8, 6 positions)
Time:   R # 48 / 2 % 4   (stride 2, 4 positions)
Packet: R # 48 % 2       (stride 1, 2 positions)
```

## Architecture

The VCG assigns a `valid_size(s, t) ∈ {0, 1, ..., 8}` to each flit, where `s` is the slice id (an integer encoding the flit's position across all `Slice` sub-expressions) and `t` is the time step.
The first `valid_size` elements of the flit are real data, the rest are padding.

```rust,ignore
struct VcgConfig {
    time_filters:   [TimeFilterConfig; 3], // for R's sub-expressions in Time and/or Slice
    packet_clipper: PacketClipperConfig,   // for R's sub-expressions in Packet
}

impl VcgConfig {
    fn valid_size(&self, s: u64, t: u64) -> u32 {
        if self.time_filters.iter().all(|tf| tf.valid(s, t)) {
            self.packet_clipper.valid_size(t)
        } else {
            0
        }
    }
}
```

When all timers report valid for flit `(s, t)`, the packet clipper decides how many elements in that flit are real data.
When any timer reports invalid, all elements in the flit are excluded regardless of what the packet clipper would say.
The two components, `TimeFilterConfig` and `PacketClipperConfig`, are explained in the sections below.

### Time Filter

For each slice `s`, a time filter determines whether each time step `t` carries valid `R` data.
The subsections below build up `fn valid()` step by step, starting from the simplest case and adding complexity.
When `R` has no sub-expressions in `Time` or `Slice`, the time filter is disabled by setting `slice_mask = 0` and `slice_thres = 1`. Then `s & 0 = 0 < 1` for every `s`, hitting the `Less` arm so `fn valid()` always returns `true`. The choice of `slice_thres = 1` is conventional; any positive value works since `0` is always less than it.

```rust,ignore
struct TimeFilterConfig {
    // How R's index is reconstructed from t.
    sequencer: Sequencer,

    // Slice classification (see `R in Slice and Time`, `R in Time and Slice`).
    slice_mask:  u32,
    slice_thres: u32,
    time_thres:  u32,
    mode:        TimeFilterMode, // SliceMajor | TimeMajor
}

impl TimeFilterConfig {
    /// Returns true if flit (s, t) carries valid R data.
    fn valid(&self, s: u64, t: u64) -> bool {
        let idx = self.sequencer.index(t);
        match ((s & self.slice_mask).cmp(&self.slice_thres), self.mode) {
            (Less,    _)          => true,
            (Greater, SliceMajor) => false,
            _                     => idx < self.time_thres as u64,
        }
    }
}
```

#### `R` as `Time`

In the simplest case, `R` occupies all of `Time` with no other axes.
Each time step `t` corresponds directly to one `R` index, and is valid when `t < R::SIZE`.

```rust,ignore
// The compiler emits roughly:
TimeFilterConfig {
    sequencer:   [R # PADDED_SIZE -> size PADDED_SIZE : stride 1],  // idx = t
    slice_mask:  0,           // no slice partitioning
    slice_thres: 0,           // 0 cmp 0 = Equal -> falls to `idx < time_thres`
    time_thres:  R::SIZE,     // valid when idx < R::SIZE
    mode:        SliceMajor,  // arbitrary; only the Equal arm is hit
}
```

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, R = 12, X = 64];

fn reduce_time_only<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![X, A / 4], m![R # 16], m![A % 4 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![X, A / 4], m![1], m![A % 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![A % 4]>()
        //   Slice     = m![X, A / 4]
        //   Time      = m![R # 16]     (steps < R::SIZE valid)
        //   Packet    = m![A % 4]
        //   OutTime   = m![1]          (R eliminated from Time)
        //   OutPacket = m![A % 4]
        .vector_intra_slice_reduce::<R, m![1], m![A % 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
```

In the example above, the sequencer iterates `R # 16` once with `size 16 : stride 1`, so `idx = t` for every time step.
With `time_thres = R::SIZE = 12`, the first 12 time steps (`t = 0..11`) are valid and the remaining 4 (`t = 12..15`) are filtered out.
The intra-slice reduce then folds exactly the 12 real `R` elements per slice.

#### `R` in `Time`

The VCG supports `Time` mappings where `R` shares space with other axes and appears as multiple sub-expressions, in any order.
The time filter uses a [Sequencer](../../moving-tensors/sequencer.md) to decompose `t` into per-sub-expression counters; summing `value × stride` over the `R`-assigned counters gives `idx`, which encodes `R`'s index for that time step.

The following example uses `R = 10` padded to `R # 12`, where `A` sits between `R`'s two sub-expressions in `Time`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 3, B = 4, R = 10, X = 64];

// R = 10, padded to R # 12, split as (size 3, stride 4) × (size 4, stride 1).
// time filter sums (R # 12 / 4 value) * 4 + (R # 12 % 4 value) * 1 to recover R index regardless of A.
fn reduce_time_reordered<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![X], m![R # 12 / 4, A, R # 12 % 4], m![B # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![X], m![A], m![B], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![B]>()
        //   Slice     = m![X]
        //   Time      = m![R # 12 / 4, A, R # 12 % 4]
        //   Packet    = m![B]
        //   OutTime   = m![A]    (R eliminated; A survives)
        //   OutPacket = m![B]
        .vector_intra_slice_reduce::<R, m![A], m![B]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
```

The compiler configures the time filter for this placement as follows:

```rust,ignore
TimeFilterConfig {
    sequencer:   [R # 12 / 4 -> size 3 : stride 4,   // assigned to time filter
                  A          -> size 3 : stride 0,   // not assigned (A's OutTime)
                  R # 12 % 4 -> size 4 : stride 1],  // assigned to time filter
    slice_mask:  0,           // no slice partitioning (R is only in Time)
    slice_thres: 0,           // every slice falls to the `idx < time_thres` arm
    time_thres:  R::SIZE,     // 10
    mode:        SliceMajor,  // arbitrary; only the Equal arm is hit
}
```

The `A` entry has stride 0, so it never contributes to `idx`, which means `A`'s position in `Time` has no effect on validity.
The remaining two entries reconstruct `R` as `(R # 12 / 4 value) × 4 + (R # 12 % 4 value)`.

For example:
- At `t = 13`, the sequencer state is `(R # 12 / 4, A, R # 12 % 4) = (1, 0, 1)`, giving `idx = 1 × 4 + 0 + 1 × 1 = 5`. `idx < 10`, so valid.
- At `t = 27`, the sequencer state is `(2, 0, 3)`, giving `idx = 2 × 4 + 0 + 3 × 1 = 11`. `idx ≥ 10`, so invalid (it is one of the two padded `R` positions).

#### `R` in `Slice` and `Time`

`SliceMajor` mode allows multiple `R` sub-expressions in both `Slice` and `Time`, with `Slice` sub-expressions more major (larger stride) than `Time` sub-expressions.
Within `Slice`, sub-expressions must appear in descending stride order (major before minor), and each must have a power-of-2 size and a power-of-2 stride so that its bits occupy a contiguous run of `slice_mask`.
Within `Time`, sub-expressions may appear in any order.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 11, X = 32];

fn reduce_slice_time_slicemajor<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![R # 16 / 8, X, R # 16 / 4 % 2], m![R # 16 % 2, R # 16 / 2 % 2], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![R # 16 / 8, X, R # 16 / 4 % 2], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![1 # 4]>()
        //   Slice     = m![R # 16 / 8, X, R # 16 / 4 % 2]   (major R sub-exprs, descending R-stride order required)
        //   Time      = m![R # 16 % 2, R # 16 / 2 % 2]      (minor R sub-exprs, any order OK)
        //   Packet    = m![1 # 4]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
```

The example places `R = 11` (padded to `R # 16`) across `Slice` and `Time` with the following sub-expressions.

| Dimension | Sub-expression | Stride |
|-----------|----------------|--------|
| `Slice` | `R # 16 / 8` | 8 |
| `Slice` | `R # 16 / 4 % 2` | 4 |
| `Time` | `R # 16 % 2` | 1 |
| `Time` | `R # 16 / 2 % 2` | 2 |

The example divides into 4 slice groups (one per `R` contribution from `Slice`: `0`, `4`, `8`, or `12`) and 4 time iterations per slice.
The layout has three regimes: 2 slices are fully valid, 1 slice is partial, and 1 slice is fully invalid.

| `R` contribution from `Slice` | `R` values across iterations | Valid time steps |
|-------------------------------|------------------------------|------------------|
| `0` | `0, 2, 1, 3` | 4 (all `< R::SIZE`) |
| `4` | `4, 6, 5, 7` | 4 (all `< R::SIZE`) |
| `8` | `8, 10, 9, 11` | 3 (`R = 11` invalid) |
| `12` | `12, 14, 13, 15` | 0 (all `≥ R::SIZE`) |

The compiler emits the time filter config below.

```rust,ignore
TimeFilterConfig {
    sequencer:   [R # 16 % 2     -> size 2 : stride 1,
                  R # 16 / 2 % 2 -> size 2 : stride 2],
    slice_mask:  0b1000001,  // bits 0 (R # 16 / 4 % 2) and 6 (R # 16 / 8) carry the R contribution
    slice_thres: 64,         // masked id encoding the boundary R contribution (= 8)
    time_thres:  3,          // = R::SIZE - boundary = 11 - 8
    mode:        SliceMajor,
}
```

Each field encodes one part of the validity decision:

- `sequencer` records each `Time` sub-expression with its `R` stride, so the reconstructed `idx` equals the `R` contribution from `Time` at each time step `t`. For this example `idx` takes values in `{0, 1, 2, 3}` as `t` ranges over `[0, 4)`.
- `slice_mask` extracts the bits of the slice id that carry the `R` contribution from `Slice`. For this example bit `0` carries `R # 16 / 4 % 2` and bit `6` carries `R # 16 / 8`, so `slice_mask = 0b1000001`.
- `slice_thres` is the bit pattern within `slice_mask` that encodes the partial slice's `R` contribution. From the table above, the partial slice has `R` contribution `8`, encoded by setting bit `6` (`R # 16 / 8 = 1`) and clearing bit `0` (`R # 16 / 4 % 2 = 0`). So `slice_thres = 64`.
- `time_thres` is `R::SIZE` minus the partial slice's `R` contribution. For this example `time_thres = 11 - 8 = 3`.

`fn valid` returns the correct validity for every flit. To verify, decompose the `R` index as `r = r_slice + idx`, where `r_slice` is the `R` contribution from `Slice` (encoded by `s & slice_mask`) and `idx` is the contribution from `Time`, and consider the three cases of the slice comparison.

- When `(s & slice_mask) < slice_thres`, every time step is valid: `r_slice` is at least one slice spacing below the partial slice's contribution, so `r = r_slice + idx < R::SIZE` for every `idx`.
- When `(s & slice_mask) = slice_thres`, a time step is valid iff `idx < time_thres`. This is the partial slice, and by definition of `time_thres` its `r_slice = R::SIZE - time_thres`. So `r = r_slice + idx < R::SIZE` exactly when `idx < time_thres`.
- When `(s & slice_mask) > slice_thres`, no time step is valid: `r_slice` is at least one slice spacing above the partial slice's contribution, so `r_slice ≥ R::SIZE`.

#### `R` in `Time` and `Slice`

`TimeMajor` mode is the dual of [`SliceMajor`](#r-in-slice-and-time): the major/minor roles are flipped, so `Time` sub-expressions carry the larger strides and `Slice` sub-expressions carry the smaller ones.
The within-`Slice` and within-`Time` ordering rules and the power-of-2 size/stride requirement on `Slice` sub-expressions carry over unchanged from `SliceMajor`.

`TimeMajor` adds one extra constraint on top of these inherited rules.
Recall that `PADDED_SIZE` decomposes as `slice_span × time_span`, where `slice_span` and `time_span` are the products of sizes of `R`'s sub-expressions in `Slice` and `Time` respectively.
`TimeMajor` requires `PADDED_SIZE - R::SIZE ≤ slice_span`, meaning at most `slice_span` `R` positions may be over-padded.
This constraint is essential, and placements that violate it are not supported by the VCG (see [`R` in `Time` and `Slice`, Over-padded](#r-in-time-and-slice-over-padded)).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 13, X = 32];

fn reduce_time_slice_timemajor<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![R # 16 / 2 % 2, X, R # 16 % 2], m![R # 16 / 4 % 2, R # 16 / 8], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![R # 16 / 2 % 2, X, R # 16 % 2], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![1 # 4]>()
        //   Slice     = m![R # 16 / 2 % 2, X, R # 16 % 2]
        //   Time      = m![R # 16 / 4 % 2, R # 16 / 8]
        //   Packet    = m![1 # 4]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
```

The example places `R = 13` (padded to `R # 16`) across `Slice` and `Time` with the following sub-expressions.

| Dimension | Sub-expression | Stride |
|-----------|----------------|--------|
| `Time` | `R # 16 / 4 % 2` | 4 |
| `Time` | `R # 16 / 8` | 8 |
| `Slice` | `R # 16 / 2 % 2` | 2 |
| `Slice` | `R # 16 % 2` | 1 |

These sub-expressions give `slice_span = 4`, `time_span = 4`, and over-padding `PADDED_SIZE - R::SIZE = 3`, which satisfies the constraint `3 ≤ 4`.
Each of the 4 slices, distinguished by its `R` contribution from `Slice` (`0`, `1`, `2`, or `3`), sweeps through 4 `R` values across the 4 time iterations.
The layout has two regimes: 1 slice is fully valid, and 3 are partial (each losing one iteration).

| `R` contribution from `Slice` | `R` values across iterations | Valid time steps |
|-------------------------------|------------------------------|------------------|
| `0` | `0, 8, 4, 12` | 4 (all `< R::SIZE`) |
| `1` | `1, 9, 5, 13` | 3 (`R = 13` invalid) |
| `2` | `2, 10, 6, 14` | 3 (`R = 14` invalid) |
| `3` | `3, 11, 7, 15` | 3 (`R = 15` invalid) |

The compiler emits the time filter config below.

```rust,ignore
TimeFilterConfig {
    sequencer:   [R # 16 / 4 % 2 -> size 2 : stride 1,    // 1 = 4 / slice_span
                  R # 16 / 8     -> size 2 : stride 2],   // 2 = 8 / slice_span
    slice_mask:  0b1000001,  // bits 0 (R # 16 % 2) and 6 (R # 16 / 2 % 2) carry the R contribution
    slice_thres: 1,          // masked id encoding r_slice = 1
    time_thres:  3,          // = time_span - 1 (partial slices drop the over-padded last iteration)
    mode:        TimeMajor,
}
```

The config differs from `SliceMajor` in three fields (`slice_mask` follows the same pattern):

- `sequencer`: every `Time` sub-expression stride is divided by `slice_span` first. For this example strides `4` and `8` become `1` and `2`.
- `slice_thres`: encodes the partial slice's `r_slice = slice_span - (PADDED_SIZE - R::SIZE)`. For this example the target value is `4 - 3 = 1`, encoded with bit `0` set (`R # 16 % 2 = 1`) and bit `6` clear (`R # 16 / 2 % 2 = 0`), giving `slice_thres = 1`.
- `time_thres`: always `time_span - 1`. For this example `time_thres = 3`.

`fn valid` returns the correct validity for every flit. To verify, decompose the `R` index as `r = idx * slice_span + r_slice`, where `r_slice` is the slice's `R` contribution decoded from `s & slice_mask`, and consider the two cases of the slice comparison.

- When `(s & slice_mask) < slice_thres`, every time step is valid: `r_slice < slice_span - (PADDED_SIZE - R::SIZE)`, so `r ≤ (time_span - 1) * slice_span + r_slice < R::SIZE` for every `idx`.
- When `(s & slice_mask) ≥ slice_thres`, a time step is valid iff `idx < time_thres = time_span - 1`. The slice's `r_slice ≥ slice_span - (PADDED_SIZE - R::SIZE)`. For `idx ≤ time_span - 2`, `r ≤ (time_span - 2) * slice_span + (slice_span - 1) < (time_span - 1) * slice_span ≤ R::SIZE`. For `idx = time_span - 1`, `r ≥ (time_span - 1) * slice_span + slice_span - (PADDED_SIZE - R::SIZE) = R::SIZE`.

Unlike `SliceMajor`, `TimeMajor` has no "all invalid" regime. The padding constraint `PADDED_SIZE - R::SIZE ≤ slice_span` caps over-padding tightly enough that no slice becomes fully invalid.

### Packet Clipper

For each flit, the packet clipper computes `valid_size(t)`, which depends only on `t` (not on the slice `s`), so all slices receive the same count at the same time step.
This slice-independence constrains which placements the VCG can express.
The subsections below build up `fn valid_size()` step by step.
When `R` has no sub-expression in `Packet`, the packet clipper is disabled by setting `axis_size = packet_span = 8` with an empty sequencer, making `fn valid_size()` always return `8` (the full flit).

```rust,ignore
struct PacketClipperConfig {
    sequencer:   Sequencer,
    axis_size:   u32, // R::SIZE
    packet_span: u32, // R positions per flit
}

impl PacketClipperConfig {
    /// Returns the valid element count for the flit at time step t.
    fn valid_size(&self, t: u64) -> u32 {
        let idx = self.sequencer.index(t);
        (self.axis_size - idx).clamp(0, self.packet_span)
    }
}
```

The packet clipper requires `Packet = m![R # PADDED_SIZE % packet_span # 8]`.
Any other axis sharing `Packet` with `R`, or `R` being split into multiple sub-expressions within `Packet`, breaks the contiguous-prefix property that `fn valid_size()` relies on (see [Inexpressible Patterns](#inexpressible-patterns)).


#### `R` as `Packet`

In the simplest case, `R` fits in a single flit (`R::SIZE ≤ 8`), so every flit has the same `valid_size = R::SIZE` regardless of time step or slice.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, R = 3, X = 64];

fn reduce_packet_only<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![1], m![R # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![1], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![R # 4]>()
        //   Slice     = m![X, A / 2]
        //   Time      = m![1]
        //   Packet    = m![R # 4]
        //   OutTime   = m![1]
        //   OutPacket = m![1 # 4]  (R eliminated from Packet)
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
}
```

The example above places `R = 3` (padded to `R # 8`) entirely in `Packet` with a single sub-expression.
The compiler configures the packet clipper as follows:

```rust,ignore
PacketClipperConfig {
    sequencer:   [],   // empty: a single flit holds all of R
    axis_size:   3,    // R::SIZE
    packet_span: 8,    // R fits in 8 flit positions
}
```

Every flit has `valid_size = clamp(3 - 0, 0, 8) = 3` (constant across time steps and slices).

#### `R` in `Time` and `Packet`

The VCG supports `R` with sub-expressions in both `Time` and `Packet`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, R = 10, X = 64];

fn reduce_time_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![R # 16 / 4], m![R # 16 % 4 # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![1], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![R # 16 % 4]>()
        //   Slice     = m![X, A / 2]
        //   Time      = m![R # 16 / 4]
        //   Packet    = m![R # 16 % 4 # 8]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
}
```

The example above places `R = 10` (padded to `R # 16`) across `Time` and `Packet`:

| Dimension | Sub-expression | Stride |
|-----------|----------------|--------|
| `Time` | `R # 16 / 4` | 4 |
| `Packet` | `R # 16 % 4 # 8` | 1 |

The compiler configures the packet clipper as follows:

```rust,ignore
PacketClipperConfig {
    sequencer:   [R # 16 / 4 -> size 4 : stride 4],
    axis_size:   10,   // R::SIZE
    packet_span: 4,    // 4 R positions per flit (the trailing 4 flit positions are always padding)
}
```

With this config, the sequencer reconstructs `idx(t)`, the `Time` contribution to the `R` index (see [`R` in `Time`](#r-in-time)).
After `Time` covers `idx(t)` of the `axis_size = 10` elements, `axis_size - idx(t)` remain.
The packet clipper fits as many of these into the flit as it can, capped at `packet_span = 4`.
So `fn valid_size(t) = clamp(10 - idx(t), 0, 4)`, with the trailing 4 flit positions always padding and the last flit carrying a partial count when `R::SIZE` is not a multiple of `packet_span`:

```text
  flit 0: idx =  0  →  clamp(10 -  0, 0, 4) = 4  [████    ]
  flit 1: idx =  4  →  clamp(10 -  4, 0, 4) = 4  [████    ]
  flit 2: idx =  8  →  clamp(10 -  8, 0, 4) = 2  [██      ]
  flit 3: idx = 12  →  clamp(10 - 12, 0, 4) = 0  [        ]
```


## Putting It All Together

The previous sections covered a single `R` sub-expression placed in one or two dimensions.
In practice, an Intra-Slice Reduce takes a single `REDUCE_LABEL` (`R`) plus extra padded non-reduce axes, and the VCG tracks all of them: each padded axis (whether `R` or another) occupies one time filter slot, and `R`'s `Packet` part occupies the packet clipper.
This example builds up from one axis to three, so each dimension's contribution is clear.

Original shape `[H, C, W] = [5, 5, 19]`.
Each axis is split into slice/time/packet parts depending on its placement:

| Axis | Padded | Slice          | Time            | Packet         |
|------|--------|----------------|-----------------|----------------|
| `H`  | `# 8`  | `H # 8 / 2` (size 4) | `H # 8 % 2` (size 2)  | -                    |
| `C`  | `# 8`  | `C # 8 / 2` (size 4) | `C # 8 % 2` (size 2)  | -                    |
| `W`  | `# 24` | -                    | `W # 24 / 8` (size 3) | `W # 24 % 8` (size 8) |

For brevity in the following steps, we use `Ho`/`Co`/`Wo` and `Hi`/`Ci`/`Wi` as shorthands: `*o` is the leftmost factor in the table row (slice for `H`/`C`, time for `W`), `*i` is the next one to its right.

### Step 1: `W=19` only (packet clipper, no time filters)

Ignore H and C for now.
Disable time filters 0 and 1 (`slice_mask=0, slice_thres=1`).
Every slice processes 3 flits (`Wo` size 3), and packet clipper produces the sawtooth:

```text
valid_size: 8, 8, 3
              ^     ^
            full   19 - 16 = 3 (partial)
```

Since there are no time filters, every slice gets this exact same pattern:

```text
All slices, all flits:
flit 0: ████████  (valid_size=8)
flit 1: ████████  (valid_size=8)
flit 2: ███       (valid_size=3)
```

### Step 2: Add `C=5` (packet clipper + time filter 0)

Now enable the `C`-axis timer (time filter 0).
`C=5` is split into `Co` (slice, size 4) × `Ci` (time, size 2).
The `C` time filter config: `slice_mask=0b0011` (extracts `Co` from `slice_id`), `slice_thres=2`, `time_thres=1`, `SliceMajor`.

Each slice now runs 6 flits: `Ci` size 2 × `Wo` size 3.
The `C` time filter classifies slices by their `Co` value:

| `Co` | Group | Effect |
|------|-------|--------|
| 0 | below (`< 2`) | all 6 flits pass packet clipper's pattern |
| 1 | below (`< 2`) | same |
| 2 | boundary (`= 2`) | valid for `Ci=0`, invalid for `Ci=1` |
| 3 | above (`> 2`) | all 6 flits have `valid_size = 0` |

Result per slice (6 flits = `Ci` size 2 × `Wo` size 3):

```text
Co=0:  [8,8,3, 8,8,3]   (both Ci steps valid)
Co=1:  [8,8,3, 8,8,3]   (same)
Co=2:  [8,8,3, 0,0,0]   (Ci=0 valid, Ci=1 invalid)
Co=3:  [0,0,0, 0,0,0]   (all invalid)
```

Notice the timer's effect.
Some slices go entirely to zero, and the boundary slice loses its second half.
But within the valid flits, the `[8,8,3]` pattern from packet clipper is unchanged.

### Step 3: Add `H=5` (packet clipper + time filter 0 + time filter 1)

Now enable the `H`-axis timer (time filter 1).
`H=5` is split into `Ho` (slice, size 4) × `Hi` (time, size 2).
The `H` time filter config: `slice_mask=0b1100` (extracts `Ho` from `slice_id`), `slice_thres=0b1000`, `time_thres=1`, `SliceMajor`.

The slice id encodes both slice factors as `slice_id = Ho * 4 + Co`, giving 16 slices.
Each slice now runs 12 flits: `Hi` size 2 × `Ci` size 2 × `Wo` size 3.

| Component | Axis | Tracks | Config |
|-----|------|----------------|------------|
| packet clipper | `W=19` | per-flit `R` count | `axis_size=19`, `packet_span=8`, sequencer = `[Wo -> size 3 : stride 8]` |
| time filter 0 | `C=5` | per-slice `Co` validity | `slice_mask=0b0011`, `slice_thres=2`, `time_thres=1`, `SliceMajor` |
| time filter 1 | `H=5` | per-slice `Ho` validity | `slice_mask=0b1100`, `slice_thres=0b1000`, `time_thres=1`, `SliceMajor` |

The `H` time filter classifies slices by `Ho`, same logic as `C` time filter by `Co`:

| `Ho` | Group | Effect |
|------|-------|--------|
| 0 | below | open |
| 1 | below | open |
| 2 | boundary | open for `Hi=0`, closed for `Hi=1` |
| 3 | above | closed |

The final `valid_size` is the packet clipper's count when both time filters return `true`, or `0` if either returns `false`.

The complete heatmap below has 16 slices (columns, grouped by `Ho`) by 12 flits (rows, grouped by `(Hi, Ci)`).
Right-side annotations (`H:`, `C:`) label which timers are active for each row: `v` = valid, `>` = boundary, `x` = invalid.
Scan these annotations first to predict which row × column blocks should be all-zero (any timer `x`) versus carry data, then read the cells to confirm the `[8, 8, 3]` packet-dim sawtooth.


```text
                     Ho=0       |Ho=1       |Ho=2       |Ho=3
                Co:  0  1  2  3 | 0  1  2  3| 0  1  2  3| 0  1  2  3
     H time filter: v  v  v  v  | v  v  v  v| >  >  >  >| x  x  x  x
     C time filter: v  v  >  x  | v  v  >  x| v  v  >  x| v  v  >  x
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

Legend: `v` = below (all valid), `>` = boundary (partial), `x` = above (all invalid)
```

- `Ho=3` columns (rightmost 4): all 0 (`H` time filter `x`, always closed).
- `Co=3` columns (every 4th): all 0 (`C` time filter `x`).
- `Co=2` columns (`H:v C:>`): `C` time filter is boundary, so only rows with `Ci=0` pass. Compare `Co=1` vs `Co=2`.
- `Ho=2` columns (`H:> C:v`): `H` time filter is boundary, so only rows with `Hi=0` pass. Compare `Ho=1` vs `Ho=2`.
- `Ho=2 × Co=2` (both `>`): only `(Hi=0, Ci=0)` rows pass, the intersection of both boundaries.
- Within valid cells, the `[8, 8, 3]` sawtooth from packet clipper always appears, the same regardless of slice.


## Inexpressible Patterns

The following placements cannot be expressed by the VCG.
Each subsection shows the placement and explains why.

### `R` in `Slice`, Out of Order

When `R` has multiple sub-expressions in `Slice`, each outer sub-expression must have a larger stride than the inner ones.
Reversing this order produces non-monotonic per-slice `R` index ranges that a single `slice_thres` cannot capture.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 13, X = 32];

// NOT supported: inner sub-expression (/ 2 % 4, stride 2) placed outside major (/ 8, stride 8) in Slice.
// Produces non-monotonic slice validity (S6 valid after S5 partial); VCG cannot express this.
fn reduce_wrong_ordering<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![X, R # 16 / 2 % 4, R # 16 / 8], m![R # 16 % 2], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![X, R # 16 / 2 % 4, R # 16 / 8], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![1 # 4]>()
        //   Slice     = m![X, R # 16 / 2 % 4, R # 16 / 8]
        //   Time      = m![R # 16 % 2]
        //   Packet    = m![1 # 8]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
```

### `R` in `Slice` and `Time`, Interleaved

When `R`'s sub-expressions are distributed such that a Slice factor falls between two Time factors, different slices end up needing different numbers of valid time steps.
A single `slice_thres` cannot express this per-slice variation.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 13, X = 64];

// NOT supported: Time-Slice-Time interleave.
// Different slices need different valid step counts (e.g., S2: 3/4, S3: 2/4); single threshold cannot express this.
fn reduce_wrong_interleave<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![X, R # 16 / 2 % 4], m![R # 16 / 8, R # 16 % 2], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![X, R # 16 / 2 % 4], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![1 # 4]>()
        //   Slice     = m![X, R # 16 / 2 % 4]
        //   Time      = m![R # 16 / 8, R # 16 % 2]
        //   Packet    = m![1 # 8]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::Min,
        )
}
```

### `R` in `Slice` and `Time`, Over-padded

`TimeMajor` mode requires `PADDED_SIZE - R::SIZE ≤ slice_span`. At most `slice_span` `R` positions are over-padded.
Padding beyond this point causes slices below `slice_thres` (which `fn valid()` always reports valid) to silently include padding in the reduction.

In the example below, `R = 14`, `Slice = m![X, R # 20 % 4]` (`slice_span = 4`), and `Time = m![A, R # 20 / 4]` with `A = 3` (so `time_span = 5` from `R # 20 / 4`, while `Time::SIZE = A × time_span = 15`).
The constraint is on `time_span`, not `Time::SIZE`: non-`R` axes in `Time` like `A` cycle without changing `R`'s index, so they do not affect the constraint.
The correct padding is `R # 16` (since `16 - 14 = 2 ≤ slice_span = 4`), giving `time_span = 4`.
Using `R # 20` over-pads `R`, making `time_span = 5` and adding an extra `Time` iteration that contains no real data.
For slices below `slice_thres` (slice contribution = 0), the sequencer-reconstructed `idx` reaches `4 × 4 = 16` when `R # 20 / 4 = 4`, giving `R` index 16 ≥ `R::SIZE`: padding, should be invalid.
But `fn valid()` sees `s & self.slice_mask = 0 < slice_thres = 2` and returns `true` regardless of `idx`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 3, R = 14, X = 64];

fn reduce_time_major_wrong<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![X, R # 20 % 4], m![A, R # 20 / 4], m![1 # 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![X, R # 20 % 4], m![A], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![1 # 4]>()
        //   Slice     = m![X, R # 20 % 4]
        //   Time      = m![A, R # 20 / 4]   (A is non-R; time_span = 5 from R # 20 / 4)
        //   Packet    = m![1 # 8]
        //   OutTime   = m![A]              (R eliminated; A survives)
        //   OutPacket = m![1 # 4]
        // NOT supported: R # 20 over-pads (20 - 14 = 6 > slice_span = 4).
        // time_span = 5 > 4. Below-group slices include time steps where R # 20 / 4 = 4 (R = 16 padding).
        .vector_intra_slice_reduce::<R, m![A], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
```

### `R` in `Packet`, Complex

The packet clipper requires `Packet = m![R # PADDED_SIZE % packet_span # 8]`.
Other forms break the contiguous-prefix property that `fn valid_size()` relies on.

The first example places `R`'s major part in `Packet` (form `R # 24 / 8` instead of `R # 24 % 8`), so the prefix mixes positions from different `R`-strides rather than holding `R`'s next contiguous run.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, R = 19, X = 64];

fn reduce_wrong_packet_outer<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![R # 24 % 8], m![R # 24 / 8 # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![X, A / 2], m![1], m![1 # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![R # 24 / 4]>()
        //   Slice     = m![X, A / 2]
        //   Time      = m![R # 24 % 8]
        //   Packet    = m![R # 24 / 8 # 8]   (NOT supported: major R in Packet)
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpF32::Add,
        )
}
```

The second example has `R` sharing `Packet` with another axis `A`, so `A`'s elements occupy positions that the prefix-based count treats as padding.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2, R = 19, X = 256];

fn reduce_wrong_mixed_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, f32, m![1], m![1], m![X], m![R # 24 / 4], m![R # 24 % 4, A # 8], f32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, f32, m![1], m![1], m![X], m![1], m![A # 4], f32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![R # 24 % 4, A]>()
        //   Slice     = m![X]
        //   Time      = m![R # 24 / 4]
        //   Packet    = m![R # 24 % 4, A # 8]   (NOT supported: A shares Packet with R)
        //   OutTime   = m![1]          (R eliminated; A silently excluded by prefix valid_size)
        //   OutPacket = m![A # 4]
        .vector_intra_slice_reduce::<R, m![1], m![A # 4]>(
            IntraSliceReduceOpF32::Add,
        )
}
```

### `R` in `Slice` and `Packet`

`R` splits between `Slice` and `Packet`.
To see why this is inexpressible, consider `R = 2045` (padded to `R # 2048`) split as `Slice = m![R # 2048 / 8]` (256 slices) and `Packet = m![R # 2048 % 8]` (8-element flits).
At the only time step `t = 0`, `fn valid_size` computes `clamp(2045 - 0, 0, 8) = 8` for every slice.
But slice 255's flit holds `R` indices 2040–2047, of which only five are real data:

At `t = 0`, `slice = 255`:

| flit position | 0    | 1    | 2    | 3    | 4    | 5     | 6     | 7     |
|---------------|------|------|------|------|------|-------|-------|-------|
| `R` index     | 2040 | 2041 | 2042 | 2043 | 2044 | 2045  | 2046  | 2047  |
| valid?        | yes  | yes  | yes  | yes  | yes  | [pad] | [pad] | [pad] |

Slices 0–254 legitimately need `valid_size = 8`, but slice 255 needs `valid_size = 5`.
`fn valid_size(t)` can only return one value for a given `t`, so no single configuration works.

The degenerate sub-case where `R::SIZE % packet_span = 0` (every packet is full or empty) reduces to Slice only and is supported.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![R = 2045];

// NOT supported: R = 2045 split across Slice (/ 8, 256 slices) and Packet (% 8).
// Slices 0-254 need valid_size = 8; slice 255 needs valid_size = 5. fn valid_size(t) cannot vary by slice.
fn reduce_wrong_slice_packet<'l, const T: Tu>(
    input: VectorBranchTensor<'l, T, i32, m![1], m![1], m![R # 2048 / 8], m![1], m![R # 2048 % 8], i32, NoTensor, { stage::VeOrder::IntraFirst }>,
) -> VectorIntraSliceReduceTensor<'l, T, i32, m![1], m![1], m![R # 2048 / 8], m![1], m![1 # 4], i32, NoTensor, { stage::VeOrder::IntraFirst }>
{
    input
        .vector_narrow_clip::<m![R # 2048 % 4]>()
        //   Slice     = m![R # 2048 / 8]
        //   Time      = m![1]
        //   Packet    = m![R # 2048 % 8]
        //   OutTime   = m![1]          (R eliminated)
        //   OutPacket = m![1 # 4]
        .vector_intra_slice_reduce::<R, m![1], m![1 # 4]>(
            IntraSliceReduceOpI32::AddSat,
        )
}
```

## Constraints

| Component | Capacity |
|-----------|----------|
| Packet clippers | 1 instance |
| Time filters | 3 instances |
| Sequencer entries per time filter / packet clipper | 8 (see [Sequencer](../../moving-tensors/sequencer.md)) |

Each padded axis that needs validity tracking occupies one time filter or the packet clipper.
At most 4 axes can be tracked in one invocation (1 packet clipper + 3 time filters).
Unpadded axes need no slot (`slice_mask=0, slice_thres=1` disables the time filter, making it always return `true`).

Intra-slice reduce takes a single `REDUCE_LABEL`, so "multi-axis" means one reduce axis `R` plus extra padded non-reduce axes, not multiple simultaneous reductions.


## Downstream 4-Way Operations


The VCG produces `valid_size` per 8-way flit before any narrowing.
A downstream `Narrow` stage splits each 8-way flit into 4-way halves, and the way the narrow is applied determines how each `valid_size` is split between the halves.

| Operation | Input | Output | Valid Count Transformation |
|-----------|-------|--------|---------------------------|
| **split_way4** | 8-way flit (`valid_size = v`) | two 4-way flits | low: `min(v, 4)`, high: `max(v - 4, 0)` |
| **trim_way4** | 8-way flit (`valid_size = v`) | one 4-way flit | `min(v, 4)` |
| **concat_way8** | two 4-way flits (`v_low`, `v_high`) | 8-way flit | `v_low + v_high` |
| **pad_way8** | 4-way flit | 8-way flit | unchanged |

Split and concat preserve the prefix property.
For trim_way4, the mapping must statically guarantee `v <= 4`.
If the upper 4 elements could be valid, trimming them would lose data.
