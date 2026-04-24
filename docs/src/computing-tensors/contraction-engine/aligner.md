# Aligner

The Aligner stage prepares both operands for the [Reducer](./reducer.md) by transforming them into a matching **computation mapping**.
The *computation mapping* is the common tensor layout (`[Chip, Cluster, Slice, Row, Time, Packet]`) that both the Stream Adapter and TRF Sequencer must produce so the Reducer can pair them element-by-element.
It is positioned within the Contraction Engine data flow as follows:

```text
fetch() -> switch() -> collect() -> align(trf) -> contract() -> accumulate()
```

The Aligner consists of two parallel paths:

| Path | Component | Source | Role |
|------|-----------|--------|------|
| Data  | [Stream Adapter](./stream-adapter.md) | [Collect Engine](../collect-engine.md) (Stream data from DM) | Collect flits, broadcast to Rows |
| Weight | [TRF Sequencer](./trf-sequencer.md) | TRF (weight data) | Broadcast and transform weight data |

## Overview

```text
                    ┌───────────────────────────────────────────────────┐
                    │                      Aligner                      │
                    │                                                   │
                    │                           ┌─────────────────────┐ │
  Switching ──────► │   Stream Adapter ────────►│                     │ │
  Engine            │                           │ Computation mapping │───► Reducer
                    │                           |                     | │
  TRF ────────────► │   TRF Sequencer  ────────►│                     │ │
                    │                           └─────────────────────┘ │
                    │                                                   │
                    └───────────────────────────────────────────────────┘
```

The computation mapping consists of the following dimensions:
- `Chip`: No change from Stream Adapter/TRF Sequencer input
- `Cluster`: No change from Stream Adapter/TRF Sequencer input
- `Slice`: No change from Stream Adapter/TRF Sequencer input
- `Row`: Maps to the 8 Rows in the Reducer
- `Time`: The temporal dimension for sequential processing
- `Packet`: Data packet dimension

The key difference between the two paths is:
- **Stream Adapter**: Always populates Rows via broadcasting, and supports basic flit collection and data feeding for convolutions.
- **TRF Sequencer**: Leverages a sequencer to enable more complex data transformations.

## Example: Batched MatMul

A batched matrix multiplication demonstrates how the Stream Adapter and TRF Sequencer align data and weights into a matching computation mapping (each detailed in the [Stream Adapter](./stream-adapter.md) and [TRF Sequencer](./trf-sequencer.md) sub-sections).
The code below does three things:

1. **Flit Collection** (Stream Adapter, `collect_flits = 2`):
   `L = 2` flits are collected from the innermost `Time` axis into the `Packet` dimension, forming a 64B packet.
   The collected data is broadcast to Rows (1, 2, 4, or 8 rows depending on the computation mapping).
2. **Packet Broadcast** (TRF Sequencer, `reg_read_size = 32B`):
   The TRF Sequencer reads 32B (`K = 16` `bf16`) contiguously each cycle and broadcasts twice to fill the 64 bytes, matching the Stream Adapter's Packet.
3. **Time Permute** (TRF Sequencer):
   The order of axes in TRF Element `[O = 2, M = 32, K = 16]` does not match `Time: [M = 32, O = 2]`.
   The sequencer reorders this by placing `O` in Entry 0 (inner loop) with stride 1024, while `M` uses Entry 1 (outer loop) with stride 32.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![M = 32, N = 8, K = 16, L = 2, O = 2];

/// Stores weights into TRF (sub context).
fn store_weights<'l, const T: Tu>(
    weights: CollectTensor<'l, T, bf16, m![1], m![1], m![1], m![N, O, M], m![K]>,
) -> TrfTensor<bf16, m![1], m![1], m![1], m![N], m![O, M, K]> {
    // TRF mapping: [
    //   Row: [N = 8]: 8 output channels mapped to 8 Rows
    //   Element: [O = 2, M = 32, K = 16]: each Row stores 2×32×16 = 1024 bf16 elements
    // ]
    weights.to_trf(TrfAddress::FirstHalf)
}

/// Aligns data and weights, then contracts (main context).
fn matmul<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1], m![1], m![M, O, L], m![K]>,
    trf: &TrfTensor<bf16, m![1], m![1], m![1], m![N], m![O, M, K]>,
) -> AccumulationTensor<'l, T, f32, m![1], m![1], m![1], m![M, O, L], m![N]> {
    // Collect mapping: [Time: [M = 32, O = 2, L = 2], Packet: [K = 16]]
    // TRF mapping:     [Row: [N = 8], Element: [O = 2, M = 32, K = 16]]
    //
    // Stream Adapter (collect_flits = 2):
    //   Flit Collection:
    //     Collects L = 2 flits from innermost Time into Packet.
    //     After collection, the computation mapping dimensions become:
    //       Time = [M = 32, O = 2], Packet = [L = 2, K = 16] = 32 bf16 = 64B
    //   Broadcasts Packet to Rows (N = 8).
    //
    // TRF Sequencer (reg_read_size = 32B):
    //   Packet Broadcast:
    //     reg_read_size read: reads K = 16 bf16 = 32B contiguously from TRF,
    //     then broadcasts 2× to fill the 64B — matching Packet = [L = 2, K = 16].
    //   Time Permute:
    //     TRF Element outer of reg_read_size(K) is [O = 2(outer), M = 32(inner)],
    //     but Time is [M = 32(outer), O = 2(inner)] — M, O are reordered via sequencer.
    //
    // Compiler-generated TRF Sequencer configuration:
    //   Entry 0: { size: 2, stride: 1024 }  — O (inner loop, stride = K×M×sizeof(bf16))
    //   Entry 1: { size: 32, stride: 32 }  — M (outer loop, stride = K×sizeof(bf16))
    //
    // Computation mapping: [Time: [M = 32, O = 2], Row: [N = 8], Packet: [L = 2, K = 16]]
    // Output mapping: [Time: [M = 32, O = 2, L = 2], Packet: [N = 8]]
    //   (K is contracted, column major)
    input.align::<m![M, O], m![L, K], _, _>(trf)
         .contract::<m![L]>()
         .accumulate::<m![M, O, L], m![N]>(AccumulationKind::Interleaved)
}
```

For details on each component, see the sub-sections:
- [Stream Adapter](./stream-adapter.md) — Flit collection, Rows broadcast
  - [Advanced Operations](./stream-adapter-advanced.md) — Transpose, Shift (for convolutions)
- [TRF Sequencer](./trf-sequencer.md) — SRAM-to-TRF, weight broadcasting
