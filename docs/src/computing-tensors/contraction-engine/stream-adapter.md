# Stream Adapter

The Stream Adapter is part of the [Aligner](./aligner.md) stage.
It transforms activation data from the [Collect Engine](../collect-engine.md) into the computation mapping required by the [Reducer](./reducer.md).
It collects incoming flits into properly sized packets and broadcasts them across Rows, enabling data reuse across output channels.
This operation is the data-side counterpart to the [TRF Sequencer](./trf-sequencer.md), which prepares weight data on the other side.

## Interface

The Stream Adapter is configured through the `align` method on `CollectTensor` (see [TRF Sequencer — Interface](./trf-sequencer.md#interface) for the full API).
The `Time` and `Packet` type parameters determine how the Stream Adapter reshapes the input:

```rust,ignore
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
impl<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet>
    CollectTensor<'l, { T }, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Aligns this input stream with a TRF tensor for contraction.
    /// Configures both the Stream Adapter (data path) and TRF Sequencer (weight path)
    /// to produce a matching computation mapping.
    pub fn align<OutTime: M, OutPacket: M, Row: M, TrfElement: M>(
        self,
        trf_tensor: &TrfTensor<D, Chip, Cluster, Slice, Row, TrfElement>,
    ) -> AlignedPair<'l, { T }, D, Chip, Cluster, Slice, Row, OutTime, OutPacket> {
        // Hardware implementation: configures Stream Adapter and TRF Sequencer
    }
}
```

The typical data flow is: `switch() → collect() → align(&trf) → contract() → accumulate()` for activations (main context).
The `Chip`, `Cluster`, and `Slice` dimensions pass through unchanged.

## Architecture

### Conceptual Operation

The Stream Adapter transforms the collect tensor mapping into the computation mapping:

```text
Collect mapping:     [Chip, Cluster, Slice, Time, Packet]
                         ↓ Stream Adapter (collect + broadcast)
Computation mapping: [Chip, Cluster, Slice, Row, OutTime, OutPacket]
```

This transformation involves three operations:

1. **Collect**: Buffer `collect_flits` incoming 32-byte flits from the innermost `Time` axis into `Packet`, creating the `OutTime` and `OutPacket` mappings.
2. **Rows broadcast**: Broadcast the collected `OutPacket` to 1, 2, 4, or 8 Rows (determined by the computation mapping).
3. **Time broadcast**: Repeat the same activation data across tiling axes in `OutTime`.

For advanced operations (transpose, shift-and-reuse for convolutions), see [Advanced Operations](./stream-adapter-advanced.md).

### Flit Buffer

The Flit Buffer buffers incoming flits so the Reducer receives data in properly sized units.

The Collect Engine sends data in 32-byte flits.
The `collect_flits` parameter controls how many consecutive flits are collected into one `OutPacket`:

| `collect_flits` | Data per Packet | Zero padding | MAC utilization | Use case |
|-----------------|-----------------|--------------|-----------------|----------|
| 1 | 32 bytes | 32 bytes | Half | Small data where a single flit covers the `Packet` axis |
| 2 (default) | 64 bytes | None | Full | Standard — full `mac_width` utilization |
| 3 | 96 bytes | N/A | Full | Shift-reuse with padding (see [Advanced](./stream-adapter-advanced.md)) |

`OutPacket` is always 64 bytes (`mac_width`).
The `collect_flits` parameter determines how much of that 64 bytes is actual data versus zero padding.

When `collect_flits = 2`, the innermost `Time` axis is consumed into `Packet`.
For example, if the collect mapping has `Time: [..., L = 2]` and `Packet: [K = 16]`, collecting `L = 2` produces `Packet = [L = 2, K = 16]` = 32 `bf16` elements = 64 bytes of data, filling the entire `mac_width`.

When `collect_flits = 1`, no `Time` axis is consumed.
The original `Packet` (32 bytes) occupies the first half, and the remaining 32 bytes are zero-padded.
Only half the MACs produce meaningful results — the zero-padded half always multiplies by zero.

The Flit Buffer has 96-byte physical capacity: up to 3 single-channel flits (32 bytes each) or 1 dual-channel flit (64 bytes).

### Rows Broadcast

After collection, the Stream Adapter broadcasts the same `OutPacket` data to multiple Rows.
The number of Rows receiving the broadcast is determined by the computation mapping: 1, 2, 4, or 8.

This is in contrast to the TRF Sequencer, where each Row reads different weight data from its own TRF partition.
The Reducer then multiplies each Row's shared activation data against its unique weights.

```text
                 ┌─── Row 0: Packet (same data)
Stream Adapter ──┼─── Row 1: Packet (same data)
  (rows=4)       ├─── Row 2: Packet (same data)
                 └─── Row 3: Packet (same data)
```

### Time Broadcast

When the computation mapping includes `Time` axes that have no corresponding axes in the activation data, the Stream Adapter tiles the input data.

For example, if the TRF data has a `T = 5` axis that the activation data lacks, the Stream Adapter tiles the input `Packet` 5 times.

```text
                  ┌─── T = 0: Packet (same data)
                  ├─── T = 1: Packet (same data)
Time broadcast ───┼─── T = 2: Packet (same data)
  (T = 5)         ├─── T = 3: Packet (same data)
                  └─── T = 4: Packet (same data)
```

Tiling axes are placed at the **innermost** positions of `OutTime`.
Multiple tiling axes can be used.

### Specifications

| Parameter | Values | Description |
|-----------|--------|-------------|
| `collect_flits` | 1, 2, 3 | Number of 32-byte flits collected per `OutPacket` |
| Flit Buffer capacity | 96 bytes | Physical buffer limit (3 × 32-byte flits) |
| `OutPacket` size | Always 64B | = `mac_width`; zero-padded when `collect_flits = 1` |
| `Rows` | 1, 2, 4, 8 | Number of Rows receiving the broadcast (from computation mapping) |
| Tiling axes | Any size, stride = 0 | `Time` axes that broadcast activation data without re-fetching |

## Performance

For `collect_flits = 1` or `2`, the Stream Adapter is effectively a pass-through with no overhead.
The `collect_flits = 3` case (shift-reuse) introduces additional latency; see [Advanced Operations](./stream-adapter-advanced.md).

## Examples

### `collect_flits = 2` (Flit Collection)

This example collects `L = 2` flits from the innermost `Time` axis into `Packet`, producing a 64B `OutPacket` (computation packet):

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![M = 32, N = 8, K = 16, L = 2, O = 2];

fn align<'l, const T: Tu>(
    input: CollectTensor<'l, { T }, bf16, m![1], m![1], m![1], m![M, O, L], m![K]>,
    trf: &TrfTensor<bf16, m![1], m![1], m![1], m![N], m![O, K]>,
) -> AlignedPair<'l, { T }, bf16, m![1], m![1], m![1], m![N], m![M, O], m![L, K]> {
    // Collect mapping: [Time: [M=32, O=2, L=2], Packet: [K=16]]
    //
    // Stream Adapter (collect_flits = 2):
    //   Flit Collection:
    //     Collects L = 2 flits from innermost Time into Packet:
    //       Time = [M = 32, O = 2], Packet = [L = 2, K = 16] = 32 bf16 = 64B
    //   Broadcasts Packet to Rows (N = 8).
    //
    // Computation mapping:
    //   [Time: [M = 32, O = 2] | Row: [N = 8] | Packet: [L = 2, K = 16]]
    input.align::<m![M, O], m![L, K], _, _>(trf)
}
```

### `collect_flits = 1` (No Collection)

When the `Packet` axis already covers the contraction dimension and no additional flits need to be collected:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![M = 32, N = 8, K = 16];

fn align<'l, const T: Tu>(
    input: CollectTensor<'l, { T }, bf16, m![1], m![1], m![1], m![M], m![K]>,
    trf: &TrfTensor<bf16, m![1], m![1], m![1], m![N], m![K]>,
) -> AlignedPair<'l, { T }, bf16, m![1], m![1], m![1], m![N], m![M], m![K # 32]> {
    // Switch mapping: [Time: [M = 32], Packet: [K = 16]]
    //
    // Stream Adapter (collect_flits = 1):
    //   Flit Collection:
    //     No Time axis collected — data = [K = 16] = 16 bf16 = 32B.
    //     Packet = [K = 16 # 32] = 64B (32B data + 32B zero padding).
    //   Broadcasts Packet to Rows (N = 8).
    //   Half MAC utilization — zero-padded half always multiplies by zero.
    //
    // Computation mapping:
    //   [Time: [M = 32] | Row: [N = 8] | Packet: [K = 16 # 32]] (64 bytes)
    input.align::<m![M], m![K # 32], _, _>(trf)
}
```

### Time Broadcast

When the TRF has axes not present in the input data, the Stream Adapter tiles the activation across `Time`:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![M = 32, N = 8, K = 16, T = 5];

fn align<'l, const T: Tu>(
    input: CollectTensor<'l, { T }, bf16, m![1], m![1], m![1], m![M], m![K]>,
    trf: &TrfTensor<bf16, m![1], m![1], m![1], m![N], m![T, K]>,
) -> AlignedPair<'l, { T }, bf16, m![1], m![1], m![1], m![N], m![M, T], m![K # 32]> {
    // Collect mapping: [Time: [M = 32], Packet: [K = 16]]
    //
    // Stream Adapter (collect_flits = 1):
    //   Flit Collection:
    //     No Time axis collected — Packet = [K = 16 # 32] (32B data + 32B zero padding).
    //   Rows Broadcast: N = 8.
    //   Time Broadcast: T = 5 - activation tiled 5 times per M position.
    //
    // Computation mapping:
    //   [Row: [N = 8], Time: [M = 32, T = 5], Packet: [K = 16 # 32]]
    input.align::<m![M, T], m![K # 32], _, _>(trf)
}
```
