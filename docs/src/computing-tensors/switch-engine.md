# Switch Engine

<!-- > **TODO** (youseok.yang): Add hardware diagram showing the network topology upfront. -->
<!-- > Slice-level operations (`switch()`, `reduce()`, etc.) use physical interconnect networks. -->
<!-- > A visual diagram makes this immediately obvious — "slices talk through a network, not direct wires." -->

The Fetch Engine produces a `FetchTensor` where each slice holds its own portion of data.
The Switch Engine then redistributes data across slices so each slice receives exactly what it needs for computation.
Data flows through a ring network of 256 interconnected slices; each slice's router decides per packet whether to output locally or forward to a neighbor.

This data redistribution overlaps with computation, enabling the Contraction Engine to receive data in the exact pattern it needs while continuously executing operations.
This page covers the interface, routing architecture (Forwarding, Broadcast01, Broadcast1, Transpose, InterTranspose, and Custom Topologies), hardware constraints, and performance characteristics.

## Interface

```rust,ignore
{{#include ../../../furiosa-visa-std/src/stream_tensor.rs:switch_impl}}
```

The transformation preserves the tensor's mathematical representation while redistributing data across slices.
The `Chip` and `Cluster` dimensions pass through unchanged; only `Slice` and `Time` are permuted.
The packet passes through the switch engine unchanged.
After switching, call [`collect()`](./collect-engine.md) to normalize the packet to 32-byte flits.

## Architecture

This section explains how routers make decisions to route data, then shows regular topologies with predictable data flow, and finally covers custom topologies that enable arbitrary patterns.
The Switch Engine only supports specific slice and temporal dimension transformations determined by the switching topology.

### Router Decision Process

Understanding how routers make forwarding decisions is essential before exploring specific topologies.

Each slice has a router that decides whether to send its input packet to an adjacent slice or to output.

Each packet has a source slice number attached, and each slice configures a *snoop bitmap* (a bitmask specifying which source slices' packets to accept and output) to control which data it receives.

Each slice's router can make the following three routing decisions:

1. **Input routing**: The router decides whether its input packet goes to output, rightward to the next slice, or leftward to the previous slice.
2. **Right-neighbor routing**: Data arriving from the right neighbor can be forwarded to output, rightward, or leftward.
3. **Left-neighbor routing**: Data arriving from the left neighbor can be forwarded to output, leftward, or rightward.

Using these settings, data moving in a counter-clockwise ring pattern can be configured to reach the desired slice.

**Common router configurations** for counter-clockwise ring communication:

- **Root node**: Outputs input data and data from the right slice, sends to right slice.
- **Middle node**: Outputs data from the left slice, forwards to right.
- **Leaf node**: Forwards input data to left, outputs data from the left slice.

To understand how the switching mechanism routes data through the ring network, consider a minimal example with 2 slices and 2 input packets per slice.

This example shows how data flows through the ring over time, with each slice deciding whether to output data locally or forward it to neighbors.

Given:
- `axes![A = 2, B = 2, C = 64]`
- `Slice: m![A]`
- `Time: m![B]`
- `Packet: m![C]`
- Input packets per slice: `slice0 = [0, 1]`, `slice1 = [2, 3]`

| i(cycle) | slice#0                           | slice#1                                            | Output Data                            |
| -------- | --------------------------------- | -------------------------------------------------- | -------------------------------------- |
| 0        | 0: from input, to (output, right) |                                                    | `0: [0]`<br>`1: []`                    |
| 1        | 1: from input, to (output, right) | 0: from left, to output<br> 2: from input, to left | `0: [0, 1]`<br>`1: [0]`                |
| 2        | 2: from right, to (output, right) | 1: from left, to output<br> 3: from input, to left | `0: [0, 1, 2]`<br>`1: [0, 1]`          |
| 3        | 3: from right, to (output, right) | 2: from left, to output                            | `0: [0, 1, 2, 3]`<br>`1: [0, 1, 2]`    |
| 4        |                                   | 3: from left, to output                            | `0: [0, 1, 2, 3]`<br>`1: [0, 1, 2, 3]` |

As a result, a tensor with the following mapping expression is output:
- `Slice: m![A / 2, 2]`
- `Time: m![B / 2, B % 2, A % 2]`
- `Packet: m![C]`

The hardware provides pre-defined regular topologies (like `Broadcast01` with `slice0 = 2`, `slice1 = 1`) that configure the routers to achieve such patterns efficiently.

### Forwarding

Forwarding passes data through the switching network unchanged, preserving the `Slice` and `Time` dimension mapping.
Each slice's router simply passes its input data directly to output; no inter-slice communication occurs.

To use forwarding, skip the `.switch()` call entirely and invoke [`collect()`](./collect-engine.md) directly on the `FetchTensor`:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 256, B = 64, C = 32];

fn forwarding<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1], m![A], m![B], m![C]>,
) -> CollectTensor<'l, T, f32, m![1], m![1], m![A], m![B], m![C]> {
    input.collect()
}
```

The ring network operates at the following minimum cost when forwarding:

$$
\text{#cycles} = \text{ring\_size} \times \text{input\_time} \times \text{cycles\_per\_packet}
$$

`ring_size` is 1 since no inter-slice communication is needed, making this the most efficient topology when no actual switching is required.

### Broadcast01

Broadcast01 replicates data across slices along two inner `Slice` sub-dimensions (called `slice0` and `slice1` in the layout diagram below), enabling parallel computation on the same data across multiple processing elements.

This topology is essential for operations like matrix-vector multiplication where a vector needs to be broadcast to all rows of a matrix distributed across slices.

This topology is parameterized by `slice1`, `slice0`, and `time0`.
The compiler infers `slice2 = InSlice::SIZE / (slice1 * slice0)` and `time1 = InTime::SIZE / time0`.
The following table shows the input axis structure (outermost to innermost, left to right):

```text
+--------------------------+---------------+
|          Slice           |      Time     |
+--------+--------+--------+-------+-------+
| slice2 | slice1 | slice0 | time1 | time0 |
+--------+--------+--------+-------+-------+
```

After switching, `slice1` and `slice0` move from `Slice` into `Time`, broadcasting those dimensions across the ring group while tiling `slice2`:

```text
+----------------------+---------------------------------+
|        Slice         |              Time               |
+--------+------+------+-------+--------+-------+--------+
| slice2 | tile | tile | time1 | slice1 | time0 | slice0 |
+--------+------+------+-------+--------+-------+--------+
```

Moving `slice1` and `slice0` from `Slice` to `Time` creates `time1 × time0` independent ring groups, each of size `slice1 × slice0`, where slices within each ring group exchange data to achieve the broadcast pattern.
The broadcast dimensions (`slice1`, `slice0`) are placed at the innermost positions of the output `Time` dimension (just outside `Packet`).

This broadcast topology takes data that was spatially distributed across slices (the `Slice` axis) and broadcasts it over time.
Instead of different slices having different data, all slices in the same ring group receive the same data sequentially through time.

#### Example

Consider the following configuration:

- `axes![A = 256, B = 64, C = 63, D = 8]`
- `dtype = i8`
- `In`:
  - `Chip: m![D / 2]`
  - `Cluster: m![D % 2]`
  - `Slice: m![A]`
  - `Time: m![B]`
  - `Packet: m![C # 64]`
- `Out`:
  - `Chip: m![D / 2]`
  - `Cluster: m![D % 2]`
  - `Slice: m![A / 4, 4]`
  - `Time: m![B / 4, A / 2 % 2, B % 4, A % 2]`
  - `Packet: m![C # 64]`

This configuration sets `slice1 = 2`, `slice0 = 2`, `time0 = 4` in the `Broadcast01` topology.
The compiler infers `slice2 = 256 / (2 * 2) = 64` and `time1 = 64 / 4 = 16`.
Notice that `slice2 * slice1 * slice0 = 256 = Slice::SIZE`, and `time1 * time0 = 64 = (old)Time::SIZE`.

The difference between the input and output mappings is that the `A % 4` axis moved from `Slice` to `Time`, while `slice2` is tiled.
This divides the 256 slices into 16 groups of size `ring_size = slice0 * slice1 = 4`.
The axis movement between `Slice` and `Time` enables this broadcast behavior: when an axis moves from `Slice` to `Time`, it creates dependencies where slices in a particular ring group receive data from the other slices in the same ring group.
The `slice1` and `slice0` broadcast axes each move to `Time` as `A / 2 % 2` and `A % 2`, respectively.

This particular configuration is equivalent to the following custom snoop bitmap, which maps the slice identified by the bitmap index to its corresponding ring group.
The broadcast pattern is evident: rows 0-3 have identical entries as the slices they represent (`{0, 1, 2, 3}`) receive data from the same input slices (`{0, 1, 2, 3}`).

| Bitmap Index | `(A / 4, A % 4)`                           | `A`                        | Ring Group                 |
| ------------ | ------------------------------------------ | -------------------------- | -------------------------- |
| 0            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 1            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 2            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 3            | `(0, 0)`, `(0, 1)`, `(0, 2)`, `(0, 3)`     | `0`, `1`, `2`, `3`         | `0`, `1`, `2`, `3`         |
| 4            | `(1, 0)`, `(1, 1)`, `(1, 2)`, `(1, 3)`     | `4`, `5`, `6`, `7`         | `4`, `5`, `6`, `7`         |
| …            | …                                          | …                          | …                          |
| 255          | `(63, 0)`, `(63, 1)`, `(63, 2)`, `(63, 3)` | `252`, `253`, `254`, `255` | `252`, `253`, `254`, `255` |

Since this matches exactly the pre-defined `Broadcast01` form, it is an input/output format that can be processed by the Switch Engine.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 256, B = 64, C = 63, D = 8, X = 4];

fn broadcast01<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![D / 2], m![D % 2], m![A], m![B], m![C # 64]>,
) -> SwitchTensor<'l, T, f32, m![D / 2], m![D % 2], m![A / 4, X], m![B / 4, A / 2 % 2, B % 4, A % 2], m![C # 64]> {
    // X is a newly introduced axis for broadcast semantics.
    // Input: each slice has its own portion of data (256 slices, 64 time steps, 64 byte packets)
    // Output: all slices receive broadcast data from their 4-slice ring group
    // Packet passes through unchanged; call .collect() afterwards to normalize to flits.
    input.switch::<m![A / 4, X], m![B / 4, A / 2 % 2, B % 4, A % 2]>(
        SwitchConfig::Broadcast01 {
            slice1: 2,
            slice0: 2,
            time0: 4
        }
    )
}
```

#### Cycle Estimation

The Switch Engine's cycle estimation follows the formula:

$$
\text{#cycles} = \text{ring_size} \times \text{input_time} \times \text{cycles_per_packet}
$$

$$
= (\texttt{slice0} \times \texttt{slice1}) \times \texttt{B::SIZE} \times \frac{\texttt{(C # 64)::SIZE}}{32}
$$

$$
= (2 \times 2) \times 64 \times \frac{64}{32} = 512 \text{ cycles}
$$

#### Ring Structure

The `ring_size` of 4 means that inter-slice data movement occurs in groups of 4 slices, with data dependencies existing only within each ring.

When we group all 256 slices into rings of size 4, we get 64 independent rings that operate in parallel.

Within each ring, exchanging data takes time proportional to `ring_size`, and each packet represents the minimum unit of data exchange.

Regular topologies can be expressed as tensor mapping expressions.
For example, with:
- `axes![A = 64, B = 64]`
- `Slice = m![A]`
- `Time = m![B / 2]`
- `Packet = m![B % 32]`

If configured with `Broadcast01` (`slice0 = 8`, `slice1 = 8`, `time0 = 2`), the tensor mapping expression corresponding to the output is:

- `axes![A = 64, B = 64]`
- `Slice = m![A / 64, 64]`
- `Time = m![B / 4, A / 8 % 8, B / 2 % 2, A % 8]`
- `Packet = m![B % 32]`

### Broadcast1

Broadcast1 replicates data across slices along `Slice` dimension 1, enabling parallel computation where a single dimension needs to be broadcast while preserving another dimension in the slice.
This topology is simpler than Broadcast01 as it only broadcasts along one `Slice` dimension.

This topology is parameterized by `slice1` and `slice0`.
The compiler infers `slice2 = InSlice::SIZE / (slice1 * slice0)`.
An input tensor structured as follows:

```text
+--------------------------+--------+
|           Slice          |  Time  |
+--------+--------+--------+--------+
| slice2 | slice1 | slice0 | time0  |
+--------+--------+--------+--------+
```

is transformed into the following output tensor, where only the `slice1` axis moves from `Slice` to `Time`, broadcasting this dimension across the slice's ring group, while preserving `slice0` in `Slice` dimension and tiling `slice2`.

```text
+------------------------+----------------+
|          Slice         |      Time      |
+--------+------+--------+-------+--------+
| slice2 | tile | slice0 | time0 | slice1 |
+--------+------+--------+-------+--------+
```

#### Example

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 256, B = 64, C = 63, X = 4];

fn broadcast1<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1], m![A], m![B], m![C # 64]>,
) -> SwitchTensor<'l, T, i8, m![1], m![1], m![A / 32, X, A % 8], m![B, A / 8 % 4], m![C # 64]> {
    // X is a newly introduced axis for broadcast semantics.
    // Packet passes through unchanged; call .collect() afterwards to normalize to flits.
    input.switch::<m![A / 32, X, A % 8], m![B, A / 8 % 4]>(
        SwitchConfig::Broadcast1 {
            slice1: 4,
            slice0: 8,
        }
    )
}
```

### Transpose

Transpose permutes axes within the innermost part of the `Slice` dimension.
This topology is parameterized by `slice1` and `slice0`.
An input tensor with the slice dimension structured as `[slice2, slice1, slice0]` is transformed so that the output slice becomes `[slice2, slice0, slice1]`:

```text
+--------------------------+         +--------------------------+
|           Slice          |         |           Slice          |
+--------+--------+--------+   -->   +--------+--------+--------+
| slice2 | slice1 | slice0 |         | slice2 | slice0 | slice1 |
+--------+--------+--------+         +--------+--------+--------+
```

#### Example

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 256, B = 64, C = 63];

// Transpose with slice1 = 32, slice0 = 2.
// Input Slice:  m![A]: [slice2 = 4, slice1 = 32, slice0 = 2]
// Output Slice: m![A / 64, A % 2, A / 2 % 32]
fn transpose<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1], m![A], m![B], m![C # 64]>,
) -> SwitchTensor<'l, T, i8, m![1], m![1], m![A / 64, A % 2, A / 2 % 32], m![B], m![C # 64]> {
    input.switch::<m![A / 64, A % 2, A / 2 % 32], m![B]>(SwitchConfig::Transpose {
        slice1: 32,
        slice0: 2,
    })
}
```

The output slice `m![A / 64, A % 2, A / 2 % 32]` decomposes the original axis `A` into three parts: `A / 64` extracts `slice2` (stride 64, size 4), `A % 2` extracts `slice0` (stride 1, size 2), and `A / 2 % 32` extracts `slice1` (stride 2, size 32).
Compared to the input slice ordering (`[slice2, slice1, slice0]`), `slice1` and `slice0` are swapped.

### InterTranspose

While regular Transpose permutes axes within `Slice` only, InterTranspose swaps between the `Slice` and `Time` dimensions and transposes in the `Time` dimension.

This topology is parameterized by `slice1` (the size of the dimension being swapped), `slice0`, and `time0`.
The compiler derives `slice2` and `time2` from the input `Slice` and `Time` mappings.
Since `time1` must have the same size as `slice1` for `OutSlice::SIZE` to be 256, this effectively swaps equally-sized chunks between the `Slice` and `Time` dimensions:

```text
Input:
+--------------------------+-----------------------+
|           Slice          |         Time          |
+--------+--------+--------+-------+-------+-------+
| slice2 | slice1 | slice0 | time2 | time1 | time0 |
+--------+--------+--------+-------+-------+-------+

Output:
+--------------------------+------------------------+
|          Slice           |         Time           |
+--------+--------+--------+-------+-------+--------+
| slice2 | time1  | slice0 | time2 | time0 | slice1 |
+--------+--------+--------+-------+-------+--------+
```

The `slice2` and `slice0` axes remain unchanged in `Slice`, while `time1` in `Slice` comes from the `Time` axis.
The output `Time` dimension contains `slice1` from the original `Slice` dimension.

#### Example

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, B = 32, C = 256];

// InterTranspose with slice1 = 2, slice0 = 16, time0 = 2.
// The compiler derives: slice2 = 8, time2 = 2.
// Input Slice:  m![C] = [slice2 = 8, slice1 = 2, slice0 = 16]
// Input Time:   m![A] = [time2 = 2, time1 = 2, time0 = 2]
// Output Slice: m![C / 32, A / 2 % 2, C % 16]
// Output Time:  m![A / 4, A % 2, C / 16 % 2]
fn inter_transpose<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1], m![C], m![A], m![B # 32]>,
) -> SwitchTensor<'l, T, i8, m![1], m![1], m![C / 32, A / 2 % 2, C % 16], m![A / 4, A % 2, C / 16 % 2], m![B # 32]> {
    input.switch::<m![C / 32, A / 2 % 2, C % 16], m![A / 4, A % 2, C / 16 % 2]>(
        SwitchConfig::InterTranspose {
            slice1: 2,
            slice0: 16,
            time0: 2,
        })
}
```

The output `Slice` (`m![C / 32, A / 2 % 2, C % 16]`) decomposes into:
1. `C / 32` extracts `slice2` (from input `Slice`)
2. `A / 2 % 2` extracts `time1` (from input `Time`)
3. `C % 16` extracts `slice0` (from input `Slice`)

The output `Time` (`m![A / 4, A % 2, C / 16 % 2]`) contains:
1. `A / 4` extracts `time2` (from input `Time`)
2. `A % 2` extracts `time0` (from input `Time`)
3. `C / 16 % 2` extracts `slice1` (from input `Slice`)

### Custom Topologies

Regular topologies cover the most common data movement patterns efficiently, but some tensor operations require arbitrary permutations or partial axis extractions that don't fit these predefined patterns.

Custom topologies solve this problem by allowing you to program exactly which input slices map to which output slices using a bitmap, giving you complete flexibility for complex transformations.

#### Configuration Overhead

The tradeoff for this flexibility is configuration overhead: using a custom topology requires preempting DMA and sub-context operations to write the bitmap to the hardware's Special Function Registers (SFRs).

This setup cost makes custom topologies most appropriate when the computation benefits outweigh the initialization overhead.

#### Supported Transformation Patterns

Custom bitmaps support two key transformation patterns that regular topologies cannot express.

First, they enable **free transpose with broadcast**, allowing arbitrary permutation and broadcast of partitioning axes—regular topologies only support specific forms like `Transpose` or `TransposedDim1Broadcast`, but custom bitmaps let you freely mix axes while broadcasting.

Second, they support **partial axis extraction**, where only a portion of an axis moves to `Time` during broadcasting—regular topologies like `Broadcast01` always move the entire broadcast axis, but custom bitmaps can select subsets.

#### Example 1: Arbitrary Permutation

This example demonstrates arbitrary slice dimension permutations that regular topologies cannot express, enabling flexible data reordering for specialized computation patterns.

Arguments:
- 1 cluster (256 slices): `Chip`/`Cluster` context applies to a single cluster.
- `axes![A = 16, B = 16, C = 8, D = 8, E = 8]`
- `dtype = i8`
- `In`:
  - `Slice: m![A, B]`
  - `Time: m![C]`
  - `Packet: m![D, E]`
- `Out`:
  - `Slice: m![B % 4, B / 4, A % 4, A / 4]`
  - `Time: m![C]`
  - `Packet: m![D, E]`

The difference between `In` and `Out` is that permutation occurred in the slice shape.

The form of permutation is `[0, 1, 2, 3]` to `[3, 2, 1, 0]`.

There is no regular topology corresponding to such free permutation, but it is a form that can be simply expressed with custom bitmap.

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

The cycle calculation follows the standard formula:

$$
\text{cycle} = \text{ring_size} \times \text{input_time} \times \text{cycles_per_packet}.
$$

$$
= 256 \times \texttt{C::SIZE} \times \frac{\texttt{m![D, E]::SIZE}}{32} = 4096
$$

The ring size must be a power of 2, and in this case we need the maximum value of `256`, as this particular permutation creates dependencies across all slices with no repeating structure.
For example, data from input slice `196` must reach output slice `3`, which means we need a ring large enough to cover all such cross-slice dependencies.

This high cycle count reflects the cost of the arbitrary permutation—contrast this with regular topologies that achieve much lower cycle counts through structured parallelism.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 16, B = 16, C = 8, D = 8, E = 8];

fn arbitrary_permutation<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1], m![A, B], m![C], m![D, E]>,
) -> SwitchTensor<'l, T, f32, m![1], m![1], m![B % 4, B / 4, A % 4, A / 4], m![C], m![D, E]> {
    input.switch::<m![B % 4, B / 4, A % 4, A / 4], m![C]>(
        SwitchConfig::CustomBroadcast { ring_size: 256 }
    )
}
```

#### Example 2: Multi-Axis Broadcast

This example shows broadcasting across multiple non-contiguous axes within the `Slice` dimension, useful for complex tensor operations that require replication along several independent dimensions simultaneously.

Arguments:
- 1 cluster (256 slices): `Chip`/`Cluster` context applies to a single cluster.
- `axes![A = 16, B = 16, C = 8, D = 8, E = 8]`
- `dtype = i8`
- `In`:
  - `Slice: m![A, B]`
  - `Time: m![C]`
  - `Packet: m![D, E]`
- `Out`:
  - `Slice: m![A / 2, 2, B / 2, 2]`
  - `Time: m![C, A % 2, B % 2]`
  - `Packet: m![D, E]`

The difference between `In` and `Out` is that the two axes `A % 2` and `B % 2` moved from `Slice` to `Time`, and broadcast occurred at the original position.

Among regular topologies, `Dim0`/`Dim1Broadcast` supports a similar form, but cases where axes corresponding to `Slice` to `Time` are separated within the slice cannot be expressed.

<!-- > **TODO**: The sentence above is unclear. Rewrite to explain precisely what limitation of `Dim0`/`Dim1Broadcast` this example is demonstrating. -->

However, this is a form that can be simply expressed with custom bitmap.

| Bitmap Index | `(A / 2, A % 2, B / 2, B % 2)`                                 | `(A, B)`                                       | Ring Group                 |
| ------------ | -------------------------------------------------------------- | ---------------------------------------------- | -------------------------- |
| 0            | `(0, 0, 0, 0)`, `(0, 0, 0, 1)`, `(0, 1, 0, 0)`, `(0, 1, 0, 1)` | `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)`         | `0`, `1`, `16`, `17`       |
| 1            | `(0, 0, 0, 0)`, `(0, 0, 0, 1)`, `(0, 1, 0, 0)`, `(0, 1, 0, 1)` | `(0, 0)`, `(0, 1)`, `(1, 0)`, `(1, 1)`         | `0`, `1`, `16`, `17`       |
| 2            | `(0, 0, 1, 0)`, `(0, 0, 1, 1)`, `(0, 1, 1, 0)`, `(0, 1, 1, 1)` | `(0, 2)`, `(0, 3)`, `(1, 2)`, `(1, 3)`         | `2`, `3`, `18`, `19`       |
| 3            | `(0, 0, 1, 0)`, `(0, 0, 1, 1)`, `(0, 1, 1, 0)`, `(0, 1, 1, 1)` | `(0, 2)`, `(0, 3)`, `(1, 2)`, `(1, 3)`         | `2`, `3`, `18`, `19`       |
| …            | …                                                              | …                                              | …                          |
| 255          | `(7, 0, 7, 0)`, `(7, 0, 7, 1)`, `(7, 1, 7, 0)`, `(7, 1, 7, 1)` | `(14, 14)`, `(14, 15)`, `(15, 14)`, `(15, 15)` | `238`, `239`, `254`, `255` |

The cycle calculation gives us:

$$
\text{#cycles} = \text{ring_size} \times \text{input_time} \times \text{cycles_per_packet}
$$

$$
= 32 \times 8 \times 2 = 512 \text{ cycles}
$$

The `ring_size` of `32` is smaller than the full 256 slices because the outermost `A / 2` part of the slice dimension doesn't require data exchange—only the remaining 32 slices within each `A / 2` group need to exchange data with each other.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 16, B = 16, C = 8, D = 8, E = 8, X = 2, Y = 2];

fn multi_axis_broadcast<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1], m![A, B], m![C], m![D, E]>,
) -> SwitchTensor<'l, T, f32, m![1], m![1], m![A / 2, X, B / 2, Y], m![C, A % 2, B % 2], m![D, E]> {
    input.switch::<m![A / 2, X, B / 2, Y], m![C, A % 2, B % 2]>(
        SwitchConfig::CustomBroadcast { ring_size: 32 }
    )
}
```

#### Understanding the Bitmap Pattern

Two key patterns appear in the bitmap that reveal how the transformation works.

First, broadcast manifests as identical bitmaps: `bitmap[0]` and `bitmap[1]` are completely identical because output slices 0 and 1 both receive the same source data, implementing the broadcast operation.

Second, the `Slice` to `Time` movement appears as one output slice receiving from multiple input slices: `bitmap[0] = {0, 1, 16, 17}` shows that output slice 0 collects data from four different input slices.

#### Example 3: Partial Axis Extraction (Slicing)

This example demonstrates extracting only a subset of an axis during the `Slice` to `Time` transformation, enabling selective data distribution for operations that don't require the full axis range.

Arguments:
- 1 cluster (256 slices): `Chip`/`Cluster` context applies to a single cluster.
- `axes![A = 16, B = 16, C = 8, D = 8, E = 8]`
- `dtype = i8`
- `In`:
  - `Slice: m![A, B]`
  - `Time: m![C]`
  - `Packet: m![D, E]`
- `Out`:
  - `Slice: m![A, B / 4, 4]`
  - `Time: m![C, B % 4 = 3]`
  - `Packet: m![D, E]`

The difference between `In` and `Out` is that the `B % 4` axis moved from `Slice` to `Time`, and broadcast occurred at the original position.

The somewhat unusual point is that `B % 4` did not move completely intact to `Time`, but was partially sliced (3 out of total 4).

Among regular topologies, `Dim0`/`Dim1Broadcast` supports a similar form, but the form where axes corresponding to `Slice` to `Time` are sliced cannot be expressed.

However, this is a form that can be simply expressed with a custom bitmap.

| Bitmap Index | `(A, B / 4, B % 4 = 3)`                  | `(A, B)`                           | Ring Group          |
| ------------ | ---------------------------------------- | ---------------------------------- | ------------------- |
| 0            | `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`    | `(0, 0)`, `(0, 1)`, `(0, 2)`       | `0`, `1`, `2`       |
| 1            | `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`    | `(0, 0)`, `(0, 1)`, `(0, 2)`       | `0`, `1`, `2`       |
| 2            | `(0, 0, 0)`, `(0, 0, 1)`, `(0, 0, 2)`    | `(0, 0)`, `(0, 1)`, `(0, 2)`       | `0`, `1`, `2`       |
| 4            | `(0, 1, 0)`, `(0, 1, 1)`, `(0, 1, 2)`    | `(0, 4)`, `(0, 5)`, `(0, 6)`       | `4`, `5`, `6`       |
| …            | …                                        | …                                  | …                   |
| 255          | `(15, 3, 0)`, `(15, 3, 1)`, `(15, 3, 2)` | `(15, 12)`, `(15, 13)`, `(15, 14)` | `252`, `253`, `254` |

The cycle calculation follows the formula:

$$
\text{#cycles} = \text{ring_size} \times \text{input_time} \times \text{cycles_per_packet}
$$

$$
= 4 \times 8 \times 2 = 128 \text{ cycles}
$$

The small `ring_size` of `4` reflects that the `A, (B / 4)` outermost portion of the slice doesn't exchange data. Only the innermost 4 slices within each group need to communicate.

The bitmap reveals how partial axis extraction works: `bitmap[0] = {0, 1, 2}` shows that output slice 0 receives from only 3 input slices.

If the bitmap were `{0, 1, 2, 3}`, it would represent receiving the entire `B` axis (all 4 values).

By including only `{0, 1, 2}`, the bitmap implements slicing—extracting 3 out of 4 values from the `B` axis dimension.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 16, B = 16, C = 8, D = 8, E = 8, X = 4];

fn partial_axis_extraction<'l, const T: Tu>(
    input: FetchTensor<'l, T, f32, m![1], m![1], m![A, B], m![C], m![D, E]>,
) -> SwitchTensor<'l, T, f32, m![1], m![1], m![A, B / 4, X], m![C, B % 4 = 3], m![D, E]> {
    input.switch::<m![A, B / 4, X], m![C, B % 4 = 3]>(
        SwitchConfig::CustomBroadcast { ring_size: 4 }
    )
}
```

#### Constraint 1: Order Preservation

Hardware limitations require that axes moving from `Slice` to `Time` must preserve their relative order from the input `Slice` dimension.

This constraint exists because the routing network can efficiently forward data in the original axis order, but reordering axes during the transfer would require additional buffering that the hardware doesn't provide.

The following example shows an unsupported transformation that violates this constraint.

Arguments:
- 1 cluster (256 slices): `Chip`/`Cluster` context applies to a single cluster.
- `axes![A = 16, B = 16, C = 8, D = 8, E = 8]`
- `dtype = i8`
- `In`:
  - `Slice: m![A, B]`
  - `Time: m![C]`
  - `Packet: m![D, E]`
- `Out`:
  - `Slice: m![A, B / 4, 4]`
  - `Time: m![C, B % 2, B / 2]`
  - `Packet: m![D, E]`

In this example, the `B % 2` and `B / 2` axes appear in reversed order compared to their arrangement in the input slice dimension.

While the slice bitmap could theoretically represent this pattern, the hardware cannot execute it because it lacks the buffering needed to reorder axes during transfer.

If the output slice were instead `m![A, B / 4, 4]` with time `m![C, B / 2, B % 2]` and packet `m![D, E]`, then the transformation would be valid.

In this corrected version, the `B / 2, B % 2` axes maintain their original order from the input slice, satisfying the order preservation constraint.

#### Constraint 2: Innermost Time Position

The hardware requires axes moving from `Slice` to `Time` to appear at the innermost positions of the output time dimension.

Axes moving from `Slice` to `Time` are delivered last per packet, so they become the innermost `Time` dimensions in the output stream.

The following example shows an unsupported transformation that violates this constraint.

Arguments:
- 1 cluster (256 slices): `Chip`/`Cluster` context applies to a single cluster.
- `axes![A = 16, B = 16, C = 8, D = 8, E = 8]`
- `dtype = i8`
- `In`:
  - `Slice: m![A, B]`
  - `Time: m![C]`
  - `Packet: m![D, E]`
- `Out`:
  - `Slice: m![A / 2, 2, B / 2, 2]`
  - `Time: m![A % 2, C, B % 2]`
  - `Packet: m![D, E]`

In this example, the `A % 2` and `B % 2` axes that move from `Slice` to `Time` preserve their relative order correctly.

However, the transformation is still invalid because these axes are not positioned at the innermost part of the output time dimension—the `C` axis appears between `A % 2` and `B % 2`, violating the innermost position requirement.

Note that `Broadcast01` topology can sometimes work around this constraint using the `time0` parameter, which provides additional flexibility in axis positioning.

Custom topologies lack this `time0` mechanism, so they must strictly place all `Slice` to `Time` axes at the innermost time positions.

## Constraints

Understanding switching constraints prevents compilation errors and ensures correct data movement patterns.

### Why Switching Constraints Exist

The Switch Engine constraints reflect fundamental hardware design decisions about the ring network topology and router capabilities.

**Ring network topology fundamentally limits flexibility.**
The hardware implements a physical ring connecting 256 slices in a fixed order.
Data flows counter-clockwise through this ring, with each router deciding whether to output locally or forward to neighbors.
This topology is highly efficient for regular patterns (like broadcasting) where all slices follow similar routing rules.
However, it cannot efficiently express arbitrary permutations that would require complex routing tables or multiple ring passes.
The hardware provides only 256 router configuration entries—one per slice—rather than a full crossbar switch that could connect any slice to any other.

**Buffering constraints drive the order preservation rule.**
Each slice router has minimal buffering (essentially one packet), which enables high throughput but prevents reordering.
When data arrives from the ring network, the router must immediately decide: output locally or forward?
It cannot buffer multiple packets and reorder them.
Therefore, axes moving from the `Slice` to `Time` dimension must maintain their original order—the hardware simply forwards data in arrival order without reordering capabilities.

**Pipeline structure requires innermost time position.**
Data from other slices arrives last within each packet, so `Slice`-to-`Time` axes naturally become the innermost time dimensions.
Placing these axes anywhere else would require the hardware to buffer and reorder complete time sequences, which would require prohibitive amounts of SRAM and complex control logic.

### Regular Topology Constraints

Regular topologies impose specific structural requirements:

- **Topology pattern matching**: Input/output mapping expressions must match the predefined topology pattern.
  Violating this causes a compilation error.
  Example: `Broadcast01` requires specific axis ordering (`slice2`, `slice1`, `slice0`, `time1`) that cannot be arbitrarily reordered.

- **Full cluster operation**: `InSlice::SIZE` = `OutSlice::SIZE` = 256.
  Partial cluster operations are not supported; violating this causes a compilation error.

### Custom Topology Constraints

Custom topologies provide flexibility but impose two critical constraints:

**1. Order Preservation**: Axes moving from `Slice` to `Time` must preserve their relative order from the input slice dimension (see [Buffering constraints](#why-switching-constraints-exist) above).
Violating this causes a compilation error or incorrect data routing.

```text
// Input:  Slice: m![A, B]
// INVALID: B % 2 and B / 2 are reversed
// Output: Time: m![C, B % 2, B / 2]
// Valid:   Time: m![C, B / 2, B % 2]
```

**2. Innermost Time Position**: Axes moving from `Slice` to `Time` must appear at the innermost positions of the output `Time` dimension (see [Pipeline structure](#why-switching-constraints-exist) above).
Violating this causes a compilation error or incorrect data ordering.

```text
// Input:  Slice: m![A, B]
// INVALID: C appears between moved axes
// Output: Time: m![A % 2, C, B % 2]
// Valid:   Time: m![C, A % 2, B % 2]
```

> [!Note]
> The `Broadcast01` topology can sometimes work around the innermost position constraint using the `time0` parameter. Custom topologies lack this mechanism and must strictly follow the constraint.

## Performance

The Switch Engine performance directly affects computation throughput since data redistribution overlaps with tensor operations.

### Cycle Estimation

Switching operations follow the formula:

$$
\text{cycle} = \text{ring\_size} \times \text{input\_time} \times \text{cycle\_per\_packet}
$$

Where:
- `ring_size`: Number of slices in each independent ring (e.g., `slice0 × slice1` for `Broadcast01`)
- `input_time`: Size of the input time dimension
- `cycles_per_packet`: `packet_size / 32` (number of 32-byte flits per packet)

For example, with `ring_size = 4`, `input_time = 64`, and 64-byte packets (2 flits):

$$
\text{#cycles} = \text{ring_size} \times \text{input_time} \times \text{cycles_per_packet}
$$

$$
= 4 \times 64 \times 2 = 512 \text{ cycles}
$$

### Parallelism Across Rings

When 256 slices are grouped into rings (e.g., 64 rings of size 4), all rings operate **independently and in parallel**.

This parallelism is critical for high throughput: although each ring takes `ring_size × input_time × cycles_per_packet` cycles to complete, all rings finish simultaneously.

### Custom Topology Overhead

Custom topologies provide arbitrary permutation flexibility but incur **configuration overhead**:
- Requires preempting DMA and sub-context operations
- Must write the bitmap to Special Function Registers (SFRs)
- Setup cost makes custom topologies most appropriate when computation benefits outweigh initialization overhead

### Communication Cost

Communication cost in the ring network scales with ring size and data volume:
- **Regular topologies**: Optimized for common patterns, minimal overhead
- **Custom topologies**: Flexible but potentially higher setup cost
- **Ring topology characteristic**: Data movement cost increases proportionally with ring size, unlike other dimensions where stride differences have minimal impact

<!-- > **TODO**: Communication cost specification needed. This is necessary for programming while being aware of HW's communication cost. It is also a main reason for distinguishing Tiles in GPUs. Clearly knowing that data movement characteristics differ is an important aspect in low level programming. -->
<!-- > **Open Question**: How should communication cost be modeled in the tensor mapping expressions? For example, in the Head, whether the class-stride differs by 1 or 100, the cost of data movement is virtually the same. However, in Partitioning, since ring topology is used, the data movement cost tends to increase proportionally to the class-stride. This asymmetry needs to be captured in the cost model. -->



