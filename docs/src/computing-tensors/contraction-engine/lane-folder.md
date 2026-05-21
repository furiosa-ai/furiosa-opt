# Lane Folder

The Lane Folder is the Contraction Engine's final stage.
It eliminates the `Lane` dimension by relocating its 8 values into either `OutPacket` (Interleaved) or `OutTime` (Sequential).
No values are summed: the stage folds `Lane` into another axis rather than reducing it.

## Interface

`.contract_lane(mode)` invokes the Lane Folder.
The stage drains the upstream Time Reducer's buffer through an 8-element-wide output bus one cycle at a time, and `LaneMode` selects what each cycle's flit carries.

```rust,ignore
{{#include ../../../../furiosa-opt-std/src/engine/contraction/lane.rs:contract_lane_def}}
```

The minimal examples below take a `ContractTimeTensor` (the output of the upstream Time Reducer) and call only `.contract_lane(...)`, so each example shows the Lane Folder in isolation.
The input `Packet` carries the size that survived the [Packet Reducer](./packet-reducer.md), one of `{1, 2, 4, 8, 16, 32}` elements per lane.

### Interleaved

The `Lane` dimension folds into `OutPacket`: each cycle reads one column position across all 8 lanes (one value per lane, 8 values per flit), with `Lane` materialized as the innermost `OutPacket`.

```text
OutTime   = [Time, Packet]
OutPacket = [Lane # 8]
```

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![N = 8, M = 4, P = 4];

/// Lane folds into OutPacket.
fn lane_interleaved<'l, const T: Tu>(
    // Input from upstream Time Reducer: Lane = m![N], Time = m![M], Packet = m![P].
    input: ContractTimeTensor<'l, T, f32, m![1], m![1], m![1], m![N], m![M], m![P]>,
    // Output: OutTime = m![M, P] = [Time, Packet], OutPacket = m![N] = [Lane].
) -> ContractTensor<'l, T, f32, m![1], m![1], m![1], m![M, P], m![N]> {
    input.contract_lane::<m![M, P], m![N]>(LaneMode::Interleaved)
}
```

### Sequential

The `Lane` dimension folds into `OutTime`: each cycle reads 8 column positions from one lane's `Packet` (8 values per flit), with `Lane` iterating across successive cycles.
Since each cycle is 8 elements wide, `Packet` is first padded up to a multiple of 8 (the bus width), then split into `PadPacket / 8` cycles per lane and `PadPacket % 8` elements per cycle.

```text
PadPacket = Packet # align_up(Packet::SIZE, 8)   (pad Packet up to the next multiple of 8)
OutTime   = [Time, Lane, PadPacket / 8]
OutPacket = [PadPacket % 8]
```

For `Packet::SIZE < 32`, `[PadPacket / 8]::SIZE = ceil(Packet::SIZE / 8)` is the number of cycles per packet (e.g., 1 cycle for `Packet::SIZE = 4`, 2 cycles for `Packet::SIZE = 16`).

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![N = 8, M = 4, P = 12];

/// Lane folds into OutTime.
fn lane_sequential<'l, const T: Tu>(
    // Input from upstream Time Reducer: Lane = m![N], Time = m![M], Packet = m![P].
    input: ContractTimeTensor<'l, T, f32, m![1], m![1], m![1], m![N], m![M], m![P]>,
    // Output: OutTime = m![M, N, P / 8] = [Time, Lane, Packet / 8], OutPacket = m![P % 8] = [Packet % 8].
) -> ContractTensor<'l, T, f32, m![1], m![1], m![1], m![M, N, P / 8], m![P % 8]> {
    input.contract_lane::<m![M, N, P / 8], m![P % 8]>(LaneMode::Sequential)
}
```

## Constraints

The Lane Folder has no constraints of its own.
The `LaneMode` selected here determines the slot capacity bound that the upstream Time Reducer enforces (see [Time Reducer Constraints](./time-reducer.md#constraints)).


## Performance

In Interleaved mode, throughput drops by `Lane::SIZE / 8` when `Lane < 8` (inactive lanes leave bus positions empty).

In Sequential mode, when `Packet::SIZE < 8` (e.g., `Packet::SIZE = 4` after the Packet Reducer collapses half of an `bf16` packet), each cycle carries exactly `Packet::SIZE` elements rather than the full 8-element bus width: the output is narrower per cycle, but there is no padding and no wasted bus slots.

Latency is negligible: the Lane Folder reshapes per-lane outputs and does not add cycles beyond the buffer's drain time.
