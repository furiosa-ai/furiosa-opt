# Time Reducer

The Time Reducer accumulates the [Packet Reducer](./packet-reducer.md)'s `[Lane, Packet]` output across `Time` into `OutTime`, with *temporal accumulators*.

## Interface

`.contract_time::<OutTime>()` invokes the Time Reducer.
`OutTime` names the `Time` dimensions that survive (the rest are summed away).

```rust,ignore
{{#include ../../../../furiosa-opt-std/src/engine/contraction/time.rs:contract_time_def}}
```

For example, the kernel below reduces a 2D tensor along `B` (surviving only `A`).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048, B = 8];

/// Reduces along B; A survives.
fn reduce_b<'l, const T: Tu>(
    // Streaming operand: Slice = m![A / 8] (256 outer A chunks across slices).
    // Time = m![B / 4, A % 8]; Packet = m![B % 4].
    // B splits across Packet (B % 4) and Time (B / 4): each cycle produces a partial sum.
    input: CollectTensor<'l, T, bf16, m![1], m![1], m![A / 8], m![B / 4, A % 8], m![B % 4]>,
    // TRF operand: single-lane weight per slice.
    trf: &TrfTensor<bf16, m![1], m![1], m![A / 8], m![1], m![B]>,
    // Output: one f32 per (slice, A % 8) cell.
) -> ContractTensor<'l, T, f32, m![1], m![1], m![A / 8], m![A % 8], m![1]> {
    input
         // Outer: Lane = m![1], OutTime = m![B / 4, A % 8], OutPacket = m![B % 4].
         .contract_outer::<m![B / 4, A % 8], m![B % 4], _, _>(trf)
         // Packet Reducer: OutPacket = m![1]. Collapses B % 4 spatially.
         .contract_packet::<m![1]>()
         // Time Reducer: OutTime = m![A % 8]. Accumulator receives
         // Time::SIZE = (B / 4) × (A % 8) = 2 × 8 = 16 flits; B / 4 outer
         // chunks accumulate into 8 slots indexed by A % 8.
         .contract_time::<m![A % 8]>()
         // Lane Folder: Lane folds into OutPacket. Sequential mode (Lane = m![1]).
         .contract_lane::<m![A % 8], m![1]>(LaneMode::Sequential)
}
```

## Architecture

The Time Reducer receives the Packet Reducer's per-cycle `[Lane, Packet]` output.
The hardware caps `Lane::SIZE ≤ 8` (spatially parallel lanes) and `Packet::SIZE ≤ 32` upstream.

Each cycle the Time Reducer folds the `[Lane, Packet]` spatial grid across `Time` into `OutTime`.
`OutTime` must be a subset of `Time` with the relative order of surviving dimensions preserved (enforced by `verify_contract_time`).
Dimensions in `Time` absent from `OutTime` are summed away, and the outermost such dimension iterates over flits.

Let `InnerTime` denote the inner non-reduce dimensions of `Time` (the dimensions inner to the outermost reduce dimension that survive in `OutTime`).
In `reduce_b` above, `Time = m![B / 4, A % 8]` and `OutTime = m![A % 8]`, so `B / 4` is the outermost reduce dimension (iterates over `Time::SIZE = 2 × 8 = 16` flits) and `InnerTime = m![A % 8]`.

Accumulation requires `InnerTime::SIZE` slots of `[Lane, Packet]`, one per `InnerTime` tuple value. Flits with the same tuple accumulate into the same slot.
For `reduce_b`, 8 slots accumulate across the `B / 4 = 2` iterations, and after flit 15 the buffer contains the final reduced results and hands them off to the [Lane Folder](./lane-folder.md):

```text
Time = m![B / 4, A % 8]
          ~~~~~  ~~~~~
        outer R  inner non-R (A % 8)

Flit sequence (B / 4 has values 0,1; A % 8 has values 0..7):

  flit #0:  B/4=0, A%8=0  ──→ ┌─────────────────┐
  flit #8:  B/4=1, A%8=0  ──→ │ slot 0 (A%8=0)  │  accumulates B for A%8=0
                              └─────────────────┘

  flit #1:  B/4=0, A%8=1  ──→ ┌─────────────────┐
  flit #9:  B/4=1, A%8=1  ──→ │ slot 1 (A%8=1)  │  accumulates B for A%8=1
                              └─────────────────┘
   ⋮

  flit #7:  B/4=0, A%8=7  ──→ ┌─────────────────┐
  flit #15: B/4=1, A%8=7  ──→ │ slot 7 (A%8=7)  │  accumulates B for A%8=7
                              └─────────────────┘

  8 non-reduce positions → 8 slots used
```

## Constraints

The mapping fits the buffer when `InnerTime::SIZE` does not exceed the slot capacity (the number of slots the buffer holds).

The slot capacity follows from the buffer's 1,024 cells and the downstream [Lane Folder](./lane-folder.md)'s `LaneMode`.
Each slot is a `[Lane, Packet]` chunk whose shape the `LaneMode` decides, so the number of slots is 1,024 divided by the cells per chunk:

| `LaneMode` | Chunk shape | Cells per chunk | Slot capacity |
|------------|----------------------|--------------------|----------------------|
| `Interleaved` | `[Lane # 8, Packet]` | `8 × Packet::SIZE` | `128 / Packet::SIZE` |
| `Sequential` | `[Lane, Packet # 32]` | `Lane::SIZE × 32` | `32 / Lane::SIZE` |

For `reduce_b` under the downstream `.contract_lane(LaneMode::Sequential)` with `Lane::SIZE = 1`, the slot capacity is 32 and `InnerTime::SIZE = 8` fits comfortably.
If `InnerTime::SIZE` exceeded the slot capacity, you would restructure `Time` (e.g., split `B` further) or switch `LaneMode` to trade throughput for slot headroom.

## Performance

Throughput is one packet per cycle on the input side.
The effective output rate is `1 / N` of the input after reducing `N` inputs into one output.

Latency for a `Time`-dimension reduction of size `N` is approximately `N` cycles.
