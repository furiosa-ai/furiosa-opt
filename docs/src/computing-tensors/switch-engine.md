# Switch Engine

While every other Tensor Unit engine runs per-slice on its own DM partition, the Switch Engine moves data across slices through a 256-slice ring network: broadcasting one slice's value to a group, swapping values between slices, or permuting which slice holds which value.

## Interface

`FetchTensor::switch()` produces a `SwitchTensor`, preserving `Chip`, `Cluster`, `Packet`, and the underlying data values.
Only the `Slice` and `Time` mappings change to reflect the selected configuration.

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/switch.rs:switch_impl}}
```

The kernel writer picks `OutSlice`, `OutTime`, and a `SwitchConfig` argument that selects the configuration and its parameters.
`SwitchConfig` is one of the predefined variants ([`Broadcast01`](#broadcast01), [`Broadcast1`](#broadcast1), [`Transpose`](#transpose), [`InterTranspose`](#intertranspose), [`TransposedBroadcast1`](#transposedbroadcast1)) for common patterns, or a [`CustomBroadcast`](#custom-configurations) for more general patterns.
The numeric suffix in each variant name lists the slice sub-dimensions that move out of `Slice`.
For example, `Broadcast01` broadcasts both `slice0` and `slice1`, while `Broadcast1` broadcasts only `slice1`.

The compiler verifies that `OutSlice` and `OutTime` match the configuration's required dimension structure (each per-config section below shows the required structure with input/output diagrams), and compilation fails when they do not.
Every configuration also requires `InSlice::SIZE == OutSlice::SIZE`: the switch preserves the total slice count.

## Regular Configurations

Each regular configuration partitions the 256 slices on a chip into parallel **sub-rings** of `ring_size` slices each.
`ring_size` is derived from the configuration's parameters (typically `slice1 × slice0`) and determines both the partitioning granularity and the cycle cost.
See [Architecture](#architecture) below for the ring topology and per-router decision logic, and [Performance](#performance) for the cycle-factor breakdown.

Regular configurations cover six common patterns.
Arbitrary `Slice` sub-dimension permutations not expressible by one of these patterns require [`CustomBroadcast`](#custom-configurations) instead.

Configurations that introduce broadcast axes (`X` or `Y` in `Broadcast01`, `Broadcast1`, `TransposedBroadcast1`) require those axes to be new: they must not already appear in input `Slice` or input `Time`.

| Configuration | Use case | `ring_size` |
|---|---|---|
| [`Forwarding`](#forwarding) | pass each slice's data through unchanged (no inter-slice exchange) | `1` |
| [`Broadcast01`](#broadcast01) | broadcast both inner `Slice` sub-dimensions (`slice1` and `slice0`) to every slice in a sub-ring | `slice1 × slice0` |
| [`Broadcast1`](#broadcast1) | broadcast `slice1` while keeping `slice0` in `Slice` | `slice1 × slice0` |
| [`Transpose`](#transpose) | swap `slice1` and `slice0` within `Slice` | `slice1 × slice0` |
| [`InterTranspose`](#intertranspose) | swap a `Slice` sub-dimension (`slice1`) with a `Time` sub-dimension (`time1`) | `slice1 × slice0` |
| [`TransposedBroadcast1`](#transposedbroadcast1) | broadcast `slice0` to `Time` while shifting `slice1` to innermost `Slice` (equivalent to `Transpose` then `Broadcast1`) | `slice1 × slice0` |

### Forwarding

`Forwarding` leaves the `Slice` and `Time` mappings unchanged: every router outputs its own slice's input directly, with no cross-slice movement.

`SwitchConfig` has no `Forwarding` variant.
When no inter-slice exchange is needed, skip `.switch()` and call [`.collect()`](./collect-engine.md) directly on the `FetchTensor`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 64, C = 32];

fn forwarding<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1], m![A], m![B], m![C # 32]>,
) -> CollectTensor<'l, T, f32, m![1], m![1], m![A], m![B], m![C # 32]> {
    input.collect()
}
```

### Broadcast01

`Broadcast01` broadcasts each slice's data to every slice in a sub-ring, moving both inner `Slice` sub-dimensions (`slice0` and `slice1`) into `Time`.

The input dimension structure (outermost to innermost, left to right):

```text
┌──────────────────────────┬───────────────┐
│          Slice           │      Time     │
├────────┬────────┬────────┼───────┬───────┤
│ slice2 │ slice1 │ slice0 │ time1 │ time0 │
└────────┴────────┴────────┴───────┴───────┘
```

After switching, `slice1` and `slice0` move from `Slice` into `Time` to broadcast across the sub-ring.
Two new broadcast dimensions, labeled `X` and `Y`, fill their vacated `Slice` positions:

```text
┌──────────────────────┬─────────────────────────────────┐
│        Slice         │              Time               │
├────────┬──────┬──────┼───────┬────────┬───────┬────────┤
│ slice2 │  X   │  Y   │ time1 │ slice1 │ time0 │ slice0 │
└────────┴──────┴──────┴───────┴────────┴───────┴────────┘
```

The sub-ring spans `ring_size = slice1 × slice0` slices, one for each `(slice1, slice0)` combination at fixed `slice2`.
Every slice sends its packet around the sub-ring, and every slice receives all `ring_size` packets, so each output slice ends up holding the full broadcast group's data, indexed along the new `X` and `Y` axes in the output `Slice`.

#### Example

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 64, C = 63, D = 8, X = 2, Y = 2];

fn broadcast01<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![D / 2], m![D % 2], m![A], m![B], m![C # 64]>,
) -> SwitchTensor<'l, T, f32, m![D / 2], m![D % 2], m![A / 4, X, Y], m![B / 4, A / 2 % 2, B % 4, A % 2], m![C # 64]> {
    input.switch::<m![A / 4, X, Y], m![B / 4, A / 2 % 2, B % 4, A % 2]>(
        SwitchConfig::Broadcast01 {
            slice1: 2,
            slice0: 2,
            time0: 4
        }
    )
}
```

With `slice1 = 2` (size of broadcast `X`), `slice0 = 2` (size of broadcast `Y`), and `time0 = 4`, the compiler derives `slice2 = 64`, `time1 = 16`, and `ring_size = 4` (64 sub-rings span 256 slices).

Sub-dimensions resolve to `slice2 = A / 4`, `slice1 = A / 2 % 2`, `slice0 = A % 2`, `time1 = B / 4`, `time0 = B % 4`, giving `OutSlice = m![A / 4, X, Y]` and `OutTime = m![B / 4, A / 2 % 2, B % 4, A % 2]`.

[Cycle estimate](#performance): `ring_size × Time::SIZE × flits_per_packet = 4 × 64 × 8 = 2048`, where `Time::SIZE = 64` and `flits_per_packet = sizeof(f32) × Packet::SIZE / 32 = 4 × 64 / 32 = 8`.

### Broadcast1

The input dimension structure (outermost to innermost, left to right):

```text
┌──────────────────────────┬────────┐
│           Slice          │  Time  │
├────────┬────────┬────────┼────────┤
│ slice2 │ slice1 │ slice0 │ time0  │
└────────┴────────┴────────┴────────┘
```

After switching, `slice1` moves from `Slice` into `Time` to broadcast across the sub-ring while `slice0` stays in `Slice`.
A new broadcast dimension labeled `X` fills `slice1`'s vacated `Slice` position:

```text
┌────────────────────────┬────────────────┐
│          Slice         │      Time      │
├────────┬──────┬────────┼───────┬────────┤
│ slice2 │  X   │ slice0 │ time0 │ slice1 │
└────────┴──────┴────────┴───────┴────────┘
```

The sub-ring spans `ring_size = slice1 × slice0` slices, the same physical extent as `Broadcast01`'s.
But broadcasting happens only along `slice1`: each output slice receives the `slice1` packets from sources at the same `slice0` position, sequenced along the innermost `Time`.
The new `X` axis in the output `Slice` (sized `slice1`) replicates this collected data, while `slice0` itself is preserved at its original position.

#### Example

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 64, C = 63, X = 4];

fn broadcast1<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1], m![A], m![B], m![C # 64]>,
) -> SwitchTensor<'l, T, i8, m![1], m![1], m![A / 32, X, A % 8], m![B, A / 8 % 4], m![C # 64]> {
    input.switch::<m![A / 32, X, A % 8], m![B, A / 8 % 4]>(
        SwitchConfig::Broadcast1 {
            slice1: 4,
            slice0: 8,
        }
    )
}
```

With `slice1 = 4` (size of broadcast `X`) and `slice0 = 8`, the compiler derives `slice2 = 8` and `ring_size = 32` (8 sub-rings span 256 slices).

Sub-dimensions resolve to `slice2 = A / 32`, `slice1 = A / 8 % 4`, `slice0 = A % 8`, `time0 = B`, giving `OutSlice = m![A / 32, X, A % 8]` and `OutTime = m![B, A / 8 % 4]`.

[Cycle estimate](#performance): `ring_size × Time::SIZE × flits_per_packet = 32 × 64 × 2 = 4096`, where `Time::SIZE = 64` and `flits_per_packet = sizeof(i8) × Packet::SIZE / 32 = 1 × 64 / 32 = 2`.

### Transpose

`Transpose` swaps `slice1` and `slice0` within the innermost part of `Slice`.

The input and output `Slice` orderings:

```text
┌──────────────────────────┐         ┌──────────────────────────┐
│           Slice          │         │           Slice          │
├────────┬────────┬────────┤   ──►   ├────────┬────────┬────────┤
│ slice2 │ slice1 │ slice0 │         │ slice2 │ slice0 │ slice1 │
└────────┴────────┴────────┘         └────────┴────────┴────────┘
```

Each sub-ring spans `slice0 × slice1` slices and circulates data so every slice in the sub-ring ends up holding the value previously held by its swap partner.

`Transpose` requires input `Time` and output `Time` to match (after normalization).

#### Example

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 64, C = 63];

fn transpose<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1], m![A], m![B], m![C # 64]>,
) -> SwitchTensor<'l, T, i8, m![1], m![1], m![A / 64, A % 2, A / 2 % 32], m![B], m![C # 64]> {
    input.switch::<m![A / 64, A % 2, A / 2 % 32], m![B]>(SwitchConfig::Transpose {
        slice1: 32,
        slice0: 2,
    })
}
```

With `slice1 = 32` and `slice0 = 2`, the compiler derives `slice2 = 4` and `ring_size = 64` (4 sub-rings span 256 slices).

Sub-dimensions resolve to `slice2 = A / 64`, `slice1 = A / 2 % 32`, `slice0 = A % 2`, `time0 = B`, giving `OutSlice = m![A / 64, A % 2, A / 2 % 32]` and `OutTime = m![B]` (slice0 and slice1 swap; `Time` is unchanged).

[Cycle estimate](#performance): `ring_size × Time::SIZE × flits_per_packet = 64 × 64 × 2 = 8192`, where `Time::SIZE = 64` and `flits_per_packet = sizeof(i8) × Packet::SIZE / 32 = 1 × 64 / 32 = 2`.

### InterTranspose

`InterTranspose` swaps a dimension between `Slice` and `Time`: `slice1` moves into `Time` and `time1` moves into `Slice` (regular `Transpose` stays within `Slice`).

The input dimension structure (outermost to innermost, left to right):

```text
┌──────────────────────────┬───────────────────────┐
│           Slice          │         Time          │
├────────┬────────┬────────┼───────┬───────┬───────┤
│ slice2 │ slice1 │ slice0 │ time2 │ time1 │ time0 │
└────────┴────────┴────────┴───────┴───────┴───────┘
```

After switching, `slice1` and `time1` swap positions across the `Slice`/`Time` boundary:

```text
┌─────────────────────────┬────────────────────────┐
│          Slice          │          Time          │
├────────┬───────┬────────┼───────┬───────┬────────┤
│ slice2 │ time1 │ slice0 │ time2 │ time0 │ slice1 │
└────────┴───────┴────────┴───────┴───────┴────────┘
```

Each sub-ring spans `slice1 × slice0` slices and circulates data over `time1` time steps so each slice's value previously indexed by `slice1` ends up indexed by `time1`, and vice versa.

`InterTranspose` enforces three sizing constraints:

- `InSlice` spans all 256 slices: `slice2 × slice1 × slice0 == 256`.
- The swapped dimensions match in size: `time1.SIZE == slice1`.
- `InTime::SIZE` is divisible by `slice1 × time0` so the `time2` decomposition is integral.

#### Example

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 8, C = 32];

fn inter_transpose<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1], m![A], m![B], m![C # 32]>,
) -> SwitchTensor<'l, T, i8, m![1], m![1], m![A / 32, B / 2 % 2, A % 16], m![B / 4, B % 2, A / 16 % 2], m![C # 32]> {
    input.switch::<m![A / 32, B / 2 % 2, A % 16], m![B / 4, B % 2, A / 16 % 2]>(
        SwitchConfig::InterTranspose {
            slice1: 2,
            slice0: 16,
            time0: 2,
        })
}
```

With `slice1 = 2`, `slice0 = 16`, and `time0 = 2`, the compiler derives `slice2 = 8`, `time2 = 2`, and `ring_size = 32` (8 sub-rings span 256 slices).

Sub-dimensions resolve to `slice2 = A / 32`, `slice1 = A / 16 % 2`, `slice0 = A % 16`, `time2 = B / 4`, `time1 = B / 2 % 2`, `time0 = B % 2`, giving `OutSlice = m![A / 32, B / 2 % 2, A % 16]` and `OutTime = m![B / 4, B % 2, A / 16 % 2]` (slice1 and time1 swap between `Slice` and `Time`).

[Cycle estimate](#performance): `ring_size × Time::SIZE × flits_per_packet = 32 × 8 × 1 = 256`, where `Time::SIZE = 8` and `flits_per_packet = sizeof(i8) × Packet::SIZE / 32 = 1 × 32 / 32 = 1`.

### TransposedBroadcast1

`TransposedBroadcast1` broadcasts `slice0` to the innermost `Time` position and shifts `slice1` to the innermost `Slice` position, equivalent to applying `Transpose` followed by `Broadcast1`.
An input tensor structured as:

```text
┌──────────────────────────┬────────┐
│           Slice          │  Time  │
├────────┬────────┬────────┼────────┤
│ slice2 │ slice1 │ slice0 │ time0  │
└────────┴────────┴────────┴────────┘
```

becomes the output below, where `slice0` moves to the innermost `Time` position to broadcast across the sub-ring, `slice1` shifts to the innermost `Slice` position, and a broadcast dimension fills `slice1`'s vacated middle slot:

```text
┌────────────────────────┬────────────────┐
│          Slice         │      Time      │
├────────┬──────┬────────┼───────┬────────┤
│ slice2 │  Y   │ slice1 │ time0 │ slice0 │
└────────┴──────┴────────┴───────┴────────┘
```

Each sub-ring spans `slice0 × slice1` slices and circulates data so every slice ends up with its swap partner's value, broadcast across the `slice0` positions at the innermost `Time` sub-dimension.

#### Example

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 16, C = 32, Y = 8];

fn transposed_broadcast1<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1], m![A], m![B], m![C # 32]>,
) -> SwitchTensor<'l, T, i8, m![1], m![1], m![A / 64, Y, A / 8 % 8], m![B, A % 8], m![C # 32]> {
    input.switch::<m![A / 64, Y, A / 8 % 8], m![B, A % 8]>(
        SwitchConfig::TransposedBroadcast1 {
            slice1: 8,
            slice0: 8,
        }
    )
}
```

With `slice1 = 8` and `slice0 = 8` (size of broadcast `Y`), the compiler derives `slice2 = 4` and `ring_size = 64` (4 sub-rings span 256 slices).

Sub-dimensions resolve to `slice2 = A / 64`, `slice1 = A / 8 % 8`, `slice0 = A % 8`, `time0 = B`, giving `OutSlice = m![A / 64, Y, A / 8 % 8]` and `OutTime = m![B, A % 8]`.

[Cycle estimate](#performance): `ring_size × Time::SIZE × flits_per_packet = 64 × 16 × 1 = 1024`, where `Time::SIZE = 16` and `flits_per_packet = sizeof(i8) × Packet::SIZE / 32 = 1 × 32 / 32 = 1`.

## Architecture

To execute the configurations introduced above, the Switch Engine arranges all 256 slices on a chip into a single physical ring (one router per slice), partitioned into `256 / ring_size` parallel sub-rings of `ring_size` slices each (one such sub-ring is shown below).
For regular configurations the compiler derives `ring_size` from the configuration's parameters (`slice1`, `slice0`, `time0`); for `CustomBroadcast` the kernel writer sets `ring_size` directly.

```text
   ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐    ┌────────────────────┐
   │      Router 0      │◀──▶│      Router 1      │◀──▶│        ...         │◀──▶│ Router ring_size-1 │
   └────────────────────┘    └────────────────────┘    └────────────────────┘    └────────────────────┘
             ▲                                                                                          ▲
             └──────────────────────────────────────────────────────────────────────────────────────────┘
                                            wrap-around (links are bidirectional)
```

The Switch Engine is configured by a *snoop bitmap*: 256 entries (one per slice), each naming the source slices whose data should arrive at that output slice.
Regular configurations come with built-in bitmap generators.
[`CustomBroadcast`](#custom-configurations) instead lets the compiler synthesize an arbitrary bitmap from the kernel writer's input/output mappings.


Based on its bitmap entry, every router decides per incoming packet which combination of three actions to take:

- **Output**: deliver the packet to the local slice's downstream pipeline, when the packet's source slice is selected for delivery here.
- **Forward right**: pass the packet to the right neighbor's router.
- **Forward left**: pass the packet to the left neighbor's router.

In each sub-ring, the leftmost router sends its own data rightward, and the rightmost router sends its own data leftward.
When `ring_size > 2`, the intermediate routers output incoming left-neighbor data and forward it rightward.
Every router also outputs any data that arrives from a neighbor.

The trace below illustrates per-router execution in one 2-slice sub-ring (`ring_size = 2`), running a 2-slice broadcast pattern.
The remaining 127 sub-rings behave identically and are omitted.
Each link has a 1-cycle traversal latency, so the leftmost router initiates at cycle 0 while the rightmost begins at cycle 1, once the first packet from its left neighbor arrives.
With `axes![A = 256, B = 2, C = 32]`, `Slice = m![A]`, `Time = m![B]`, and `Packet = m![C]`, the shown sub-ring contains slices 0 and 1: the leftmost slice holds packets `[0, 1]` and the rightmost slice holds `[2, 3]`.
Each cell reads as `<packet>: from <source>, to <action>`, where source is `input`/`left`/`right` and action is one or more of `output`/`right`/`left`.

| cycle | Leftmost slice                    | Rightmost slice                                    | Output Data                                                 |
| ----- | --------------------------------- | -------------------------------------------------- | ----------------------------------------------------------- |
| 0     | 0: from input, to (output, right) |                                                    | `Leftmost: [0]`<br>`Rightmost: []`                          |
| 1     | 1: from input, to (output, right) | 0: from left, to output<br> 2: from input, to left | `Leftmost: [0, 1]`<br>`Rightmost: [0]`                      |
| 2     | 2: from right, to (output, right) | 1: from left, to output<br> 3: from input, to left | `Leftmost: [0, 1, 2]`<br>`Rightmost: [0, 1]`                |
| 3     | 3: from right, to (output, right) | 2: from left, to output                            | `Leftmost: [0, 1, 2, 3]`<br>`Rightmost: [0, 1, 2]`          |
| 4     |                                   | 3: from left, to output                            | `Leftmost: [0, 1, 2, 3]`<br>`Rightmost: [0, 1, 2, 3]`       |

After the trace, both slices in the sub-ring hold all four packets, completing the broadcast.

The bitmap encodes transformations through the shape of its entries:

- **Broadcast shape**: multiple output slices receiving the same source data have identical bitmap entries.
- **`Slice`-to-`Time` shape**: one output slice listing several source slices means that output collects data from all of them across consecutive time steps.

For instance, the bitmap that reproduces the [`Broadcast01` example](#broadcast01) above looks like this:

| Bitmap Index | `(A / 4, A % 4)`                           | `A`                        | Ring Group                 |
| ------------ | ------------------------------------------ | -------------------------- | -------------------------- |
| 0            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 1            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 2            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 3            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 4            | `(1, 0)`, `(1, 1)`, `(1, 2)`, `(1, 3)`     | `4`, `5`, `6`, `7`         | `4`, `5`, `6`, `7`         |
| …            | …                                          | …                          | …                          |
| 255          | `(63, 0)`, `(63, 1)`, `(63, 2)`, `(63, 3)` | `252`, `253`, `254`, `255` | `252`, `253`, `254`, `255` |

Rows 0-3 share identical entries because slices `{0, 1, 2, 3}` all receive data from input slices `{0, 1, 2, 3}` (the broadcast shape), and each row lists all four sources because `slice1` and `slice0` collapse from `Slice` into `Time` (the `Slice`-to-`Time` shape).
This pattern repeats every 4 rows for the 64 sub-rings.
`SwitchConfig::Broadcast01` generates this bitmap automatically.

## Performance

A switch operation takes roughly `ring_size × Time::SIZE × flits_per_packet` cycles.
All sub-rings advance in parallel, so this per-ring cycle count is also the chip-wide latency.
The three factors are:

- `ring_size`: cycles for one flit to traverse one sub-ring.
  A larger `ring_size` reaches more slices per ring at a higher per-ring cost, while a smaller `ring_size` partitions the cluster into more parallel rings at a lower per-ring cost.
- `Time::SIZE`: number of time steps in the input tensor.
  The sub-ring traversal repeats once per time step.
- `flits_per_packet`: flits per packet, equal to size of `D[Packet::SIZE]` divided by 32 bytes.
  The traversal also repeats once per flit of the packet.

## Custom Configurations

Custom configurations handle movement patterns no regular configuration expresses, such as arbitrary dimension permutations or partial dimension extractions.
This flexibility comes with [configuration overhead](#configuration-overhead) and the [constraints](#constraints) listed at the end of the section.


The `SwitchConfig::CustomBroadcast` variant carries a single field:

```rust,ignore
/// Routes data across slices using a custom snoop bitmap.
/// The bitmap is computed by the compiler from the input shape and
/// topology parameters.
CustomBroadcast {
    /// Ring group size for the custom routing.
    ring_size: usize,
},
```

Where regular configurations supply built-in generators for the snoop bitmap, `CustomBroadcast` lets the compiler synthesize the bitmap directly from the kernel writer's input/output mappings together with `ring_size`.

### Supported Transformation Patterns

Custom bitmaps cover two patterns regular configurations cannot:

- **Free transpose with broadcast**: arbitrary permutation and broadcast of partitioning dimensions, beyond the fixed forms in `Transpose` or `TransposedBroadcast1`.
- **Partial dimension extraction**: only a subset of a dimension's values moves to `Time` during broadcasting, whereas regular configurations like `Broadcast01` always move the whole dimension.

The examples below illustrate these patterns.

### Configuration Overhead

Writing a custom snoop bitmap streams configuration data into the Switch Engine's Special Function Registers (SFRs), and this SFR write occupies both the DMA Engine and the sub-context for the duration.
While the bitmap is loading, the DMA and sub contexts cannot run any other operation, so the cost manifests as reduced scheduling parallelism rather than a fixed-cycle stall.

### Example 1: Arbitrary Permutation

This example reverses the four innermost slice sub-dimensions (`A / 4, A % 4, B / 4, B % 4`) into `[3, 2, 1, 0]`, a pattern no regular configuration expresses.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 16, B = 16, C = 8, D = 8, E = 8];

fn arbitrary_permutation<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1], m![A, B], m![C], m![D, E]>,
) -> SwitchTensor<'l, T, f32, m![1], m![1], m![B % 4, B / 4, A % 4, A / 4], m![C], m![D, E]> {
    input.switch::<m![B % 4, B / 4, A % 4, A / 4], m![C]>(
        SwitchConfig::CustomBroadcast { ring_size: 256 }
    )
}
```

The output `Slice = m![B % 4, B / 4, A % 4, A / 4]` permutes the input slice shape `[0, 1, 2, 3]` into `[3, 2, 1, 0]`, which no regular configuration covers but a custom bitmap does.

| Bitmap Index | `(B % 4, B / 4, A % 4, A / 4)` | `(A, B)`   | Ring Group |
| ------------ | ------------------------------ | ---------- | ---------- |
| 0            | `(0, 0, 0, 0)`                 | `(0, 0)`   | `0`        |
| 1            | `(0, 0, 0, 1)`                 | `(4, 0)`   | `64`       |
| 2            | `(0, 0, 0, 2)`                 | `(8, 0)`   | `128`      |
| 3            | `(0, 0, 0, 3)`                 | `(12, 0)`  | `192`      |
| 4            | `(0, 0, 1, 0)`                 | `(0, 1)`   | `1`        |
| 5            | `(0, 0, 1, 1)`                 | `(4, 1)`   | `65`       |
| …            | …                              | …          | …          |
| 255          | `(3, 3, 3, 3)`                 | `(15, 15)` | `255`      |

Plugging into the [cycle formula](#performance), `cycles ≈ ring_size × Time::SIZE × flits_per_packet = 256 × 8 × 8 = 16384`:

- `ring_size = 256`
- `Time::SIZE = C::SIZE = 8`
- `flits_per_packet = sizeof(f32) × Packet::SIZE / 32 = 4 × 64 / 32 = 8` (`Packet = m![D, E]`, `D::SIZE × E::SIZE = 64`)

The maximum `ring_size = 256` is necessary because the permutation creates dependencies across all slices with no repeating structure, so input and output slices can be arbitrarily far apart in the ring index, and any smaller sub-ring would fail to cover at least one such pair.

### Example 2: Multi-dimension Broadcast

Unlike Example 1's pure permutation, this example moves two non-contiguous dimensions (`A % 2` and `B % 2`) from `Slice` to `Time`, broadcasting at their original positions.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 16, B = 16, C = 8, D = 8, E = 8, X = 2, Y = 2];

fn multi_axis_broadcast<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1], m![A, B], m![C], m![D, E]>,
) -> SwitchTensor<'l, T, f32, m![1], m![1], m![A / 2, X, B / 2, Y], m![C, A % 2, B % 2], m![D, E]> {
    input.switch::<m![A / 2, X, B / 2, Y], m![C, A % 2, B % 2]>(
        SwitchConfig::CustomBroadcast { ring_size: 32 }
    )
}
```

The output moves `A % 2` and `B % 2` from `Slice` to `Time`, broadcasting at their original positions via the broadcast dimensions `X` and `Y`.
`Broadcast01` supports a similar form but requires the broadcast dimensions (`slice0`, `slice1`) to be contiguous in the input slice, so it cannot express non-contiguous dimensions moving from `Slice` to `Time`.
A custom bitmap expresses it instead.

| Bitmap Index | `(A / 2, A % 2, B / 2, B % 2)`                                 | `(A, B)`                                       | Ring Group                 |
| ------------ | -------------------------------------------------------------- | ---------------------------------------------- | -------------------------- |
| 0            | `(0, 0, 0, 0)`, `(0, 0, 0, 1)`, `(0, 1, 0, 0)`, `(0, 1, 0, 1)` | `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)`         | `0`, `1`, `16`, `17`       |
| 1            | `(0, 0, 0, 0)`, `(0, 0, 0, 1)`, `(0, 1, 0, 0)`, `(0, 1, 0, 1)` | `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)`         | `0`, `1`, `16`, `17`       |
| 2            | `(0, 0, 1, 0)`, `(0, 0, 1, 1)`, `(0, 1, 1, 0)`, `(0, 1, 1, 1)` | `(0, 2)`, `(0, 3)`, `(1, 2)`, `(1, 3)`         | `2`, `3`, `18`, `19`       |
| 3            | `(0, 0, 1, 0)`, `(0, 0, 1, 1)`, `(0, 1, 1, 0)`, `(0, 1, 1, 1)` | `(0, 2)`, `(0, 3)`, `(1, 2)`, `(1, 3)`         | `2`, `3`, `18`, `19`       |
| …            | …                                                              | …                                              | …                          |
| 255          | `(7, 0, 7, 0)`, `(7, 0, 7, 1)`, `(7, 1, 7, 0)`, `(7, 1, 7, 1)` | `(14, 14)`, `(14, 15)`, `(15, 14)`, `(15, 15)` | `238`, `239`, `254`, `255` |

Plugging into the [cycle formula](#performance), `cycles ≈ ring_size × Time::SIZE × flits_per_packet = 32 × 8 × 8 = 2048`:

- `ring_size = 32` (the outermost `A / 2` partition needs no inter-sub-ring exchange, so only the innermost 32 slices within each sub-ring communicate)
- `Time::SIZE = 8`
- `flits_per_packet = sizeof(f32) × Packet::SIZE / 32 = 4 × 64 / 32 = 8` (`Packet = m![D, E]`, `D::SIZE × E::SIZE = 64`)

### Example 3: Partial Axis Extraction (Slicing)

Unlike Examples 1 and 2, which include every value of the moved dimensions, here only 3 of 4 values in `B % 4` move from `Slice` to `Time`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 16, B = 16, C = 8, D = 8, E = 8, X = 4];

fn partial_axis_extraction<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1], m![A, B], m![C], m![D, E]>,
) -> SwitchTensor<'l, T, f32, m![1], m![1], m![A, B / 4, X], m![C, B % 4 = 3], m![D, E]> {
    input.switch::<m![A, B / 4, X], m![C, B % 4 = 3]>(
        SwitchConfig::CustomBroadcast { ring_size: 4 }
    )
}
```

The output moves `B % 4` from `Slice` to `Time` with broadcast at the original position, and the fourth value (`B % 4 = 3`) is discarded so only the first 3 are extracted.
`Broadcast1` supports a similar form but always moves the entire dimension, so it cannot express a subset.
A custom bitmap expresses it instead.

| Bitmap Index | `(A, B / 4, B % 4 = 3)`                  | `(A, B)`                           | Ring Group          |
| ------------ | ---------------------------------------- | ---------------------------------- | ------------------- |
| 0            | `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`    | `(0, 0)`, `(0, 1)`, `(0, 2)`       | `0`, `1`, `2`       |
| 1            | `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`    | `(0, 0)`, `(0, 1)`, `(0, 2)`       | `0`, `1`, `2`       |
| 2            | `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`    | `(0, 0)`, `(0, 1)`, `(0, 2)`       | `0`, `1`, `2`       |
| 3            | (unused, discarded by partial extraction) | (none)                            | (none)              |
| 4            | `(0, 1, 0)`, `(0, 1, 1)`, `(0, 1, 2)`    | `(0, 4)`, `(0, 5)`, `(0, 6)`       | `4`, `5`, `6`       |
| …            | …                                        | …                                  | …                   |
| 255          | `(15, 3, 0)`, `(15, 3, 1)`, `(15, 3, 2)` | `(15, 12)`, `(15, 13)`, `(15, 14)` | `252`, `253`, `254` |

Plugging into the [cycle formula](#performance), `cycles ≈ ring_size × Time::SIZE × flits_per_packet = 4 × 8 × 8 = 256`:

- `ring_size = 4` (the outer `A, B / 4` partition needs no inter-sub-ring exchange, so only the innermost 4 slices within each sub-ring communicate)
- `Time::SIZE = 8`
- `flits_per_packet = sizeof(f32) × Packet::SIZE / 32 = 4 × 64 / 32 = 8` (`Packet = m![D, E]`, `D::SIZE × E::SIZE = 64`)

The bitmap shows partial extraction directly: `bitmap[0] = {0, 1, 2}` means output slice 0 receives from only 3 input slices, whereas `{0, 1, 2, 3}` would receive the entire `B` axis (all 4 values).

### Constraints

Custom configurations come with six constraints that bound this flexibility.

#### Broadcast axes must be new

Same rule as the [Regular Configurations](#regular-configurations) intro: each broadcast axis introduced in the output `Slice` must not appear in the input `Slice` or input `Time`.

For instance, with `axes![A = 256, B = 64, C = 32]` and input `Slice = m![A], Time = m![B], Packet = m![C # 32]`, an output `Slice = m![A / 4, B / 32, A % 4]` violates this constraint because `B` already appears in the input `Time`.

#### Each broadcast axis used exactly once

Each broadcast axis must appear exactly once in the output `Slice`.
Repeating the same axis at two output positions has no defined meaning for the routing bitmap.

For instance, output `Slice = m![A / 4, X, X]` (where `X` is a new axis used twice) violates this constraint.

#### Broadcast axes must not be padded

Broadcast axes in the output `Slice` must not carry padding (no `Axis # N` form).
Padding on a broadcast axis would leave routing destinations undefined for the padded positions.

For instance, output `Slice = m![A / 4, X # 4, Y]` violates this constraint because `X` is a broadcast axis with padding.

#### Order preservation

Axes moving from `Slice` to `Time` must preserve their relative order from the input slice dimension, and the verifier in `SwitchConfig::CustomBroadcast` panics at kernel compile time when they do not.
Each router has minimal buffering (one packet) and must immediately decide to output locally or forward, with no opportunity to buffer multiple packets and reorder them.

For instance, with `axes![A = 16, B = 16, C = 8, D = 8, E = 8]` and `dtype = i8`, mapping input `Slice = m![A, B], Time = m![C], Packet = m![D, E]` to output `Slice = m![A, B / 4, 4], Time = m![C, B % 2, B / 2], Packet = m![D, E]` violates this constraint.
Here `B % 2` and `B / 2` appear in reversed order relative to their input slice arrangement.
Output `Time = m![C, B / 2, B % 2]` would be valid, since `B / 2, B % 2` match the input order.

#### Innermost time position

Axes moving from `Slice` to `Time` must occupy the innermost positions of the output `Time`.
Data from other slices arrives last within each packet in the pipeline, so `Slice`-to-`Time` sub-dimensions naturally land at the innermost time dimensions.
Placing them elsewhere would demand buffering and reordering full time sequences, which the hardware cannot do.

For instance, with the same `axes!` and `dtype` as above, mapping input `Slice = m![A, B], Time = m![C], Packet = m![D, E]` to output `Slice = m![A / 2, 2, B / 2, 2], Time = m![A % 2, C, B % 2], Packet = m![D, E]` violates this constraint.
Here `A % 2` and `B % 2` preserve their relative order correctly, but `C` sits between them.

> [!NOTE]
> `Broadcast01` works around this constraint via the `time0` parameter, but custom configurations lack that mechanism and must follow the constraint strictly.

#### Ring size

The `ring_size` parameter must be a power of 2.
The compiler also derives the expected `ring_size` from the input/output mappings (the outermost non-direct-cast boundary) and rejects any user-supplied value that does not match.

