# Outer

The Outer stage broadcasts the two operands into a matching shape and multiplies them elementwise.

"Outer" comes from the **outer product** of linear algebra.
For vectors `u` (length `n`) and `v` (length `m`), `u v^T` is the `n × m` matrix where `(u v^T)[i, j] = u[i] × v[j]`.
That matrix is produced by broadcasting `u` along the column axis (length `m`), broadcasting `v` along the row axis (length `n`), and multiplying elementwise.
The Outer stage's three sub-stages are the hardware embodiment of this exact semantics, run in series:

- The [Stream Adapter](#stream-adapter) handles (and broadcasts) the streaming operand from the [Collect Engine](../collect-engine.md).
- The [TRF Sequencer](#trf-sequencer) handles (and broadcasts) the TRF operand from TRF SRAM.
- The [Multiplier](#multiplier) widens the operand types (`i4` / `i8` to `i32`, `f8` / `bf16` to `f32`) and multiplies the two aligned operands elementwise.

The output is a single multiplied tensor in the joint mapping `[Chip, Cluster, Slice, Lane, Time, Packet]`, ready for the [Packet Reducer](./packet-reducer.md) to reduce-add.

## Interface

`.contract_outer(&trf)` on `CollectTensor` invokes the Outer stage.

```rust,ignore
{{#include ../../../../furiosa-opt-std/src/engine/contraction/outer/mod.rs:contract_outer_def}}
```

After their respective adapters (Stream Adapter for the streaming path, TRF Sequencer for the TRF path), both paths feed `Lane` / `Time` / `Packet` of matching shape, and the Multiplier multiplies them elementwise at aligned positions.

The streaming operand's `Time` / `Packet` map to the output's `OutTime` / `OutPacket`:
`OutPacket` absorbs the innermost size-1-or-2 factor of `Time` via [Packing](#packing).
`OutTime` retains the remaining factors of `Time`, with broadcast factors added at its innermost positions via [Broadcast](#broadcast).
Broadcast factors come from the TRF operand's `Lane` / `Element` (where the streaming operand replicates against the TRF mapping) and from any purely-output axes that appear in `OutTime` / `OutPacket` but in neither the input nor the TRF (e.g. einsum `AB, BC -> ABCD` where `D` is broadcast).

The `TrfTensor` has shape `[Chip, Cluster, Slice, Lane, Element]`, with `Chip` / `Cluster` / `Slice` / `Lane` spatially parallel: `Chip` / `Cluster` / `Slice` pass through to the output, and `Lane` partitions per-lane data across 1–8 hardware lanes. `Element` (the per-lane layout set by [`.to_trf()`](../register-files.md#tensor-register-file)) is reshaped by the TRF Sequencer to fill `OutTime` / `OutPacket`.

## Stream Adapter

The Stream Adapter transforms the streaming `Time` / `Packet` into the computation shape (`Lane` / `OutTime` / `OutPacket`) via two operations: [Packing](#packing) and [Broadcast](#broadcast).
The compiler derives the three free variables (`PackSize`, `LaneBroadcast`, `TimeBroadcast`) from the user-supplied `OutTime` / `OutPacket`, the TRF operand, and any purely-output broadcast axes, giving the mapping:

```text
Lane      = LaneBroadcast
OutTime   = [Time / PackSize, TimeBroadcast]
OutPacket = [Time % PackSize, Packet] # (64 / D::SIZE)
```

### Packing

The Collect Engine produces 32 B flits, and the Outer stage emits packets of `PackSize × 32` B (32 or 64 on RNGD).
Packing combines `PackSize ∈ {1, 2}` consecutive flits into one packet:

```text
PackTime   = [Time / PackSize]
PackPacket = [Time % PackSize, Packet] # (PackSize × 32 / D::SIZE)
```

`PackSize` is set by matching `OutPacket` against the input `Packet`: `PackSize = 2` if `OutPacket` absorbs the innermost size-2 factor of `Time`, otherwise `PackSize = 1`.
Equivalently, `PackSize = OutPacket::SIZE * D::SIZE / 32`, so the user picks `OutPacket` (32 B or 64 B) and Packing's collect-flit count follows.

Hardware always operates on 64 B packets internally; when `PackSize = 1`, the unused 32 B half holds zeros that do not propagate into the logical `OutPacket` type. Downstream stages (Packet Reducer, Lane Folder) therefore see only the `PackSize × 32` B payload, avoiding dummy cycles, see the [Lane Folder Sequential note](./lane-folder.md#sequential).

### Broadcast

After packing, the Stream Adapter broadcasts the data spatially via `LaneBroadcast` (the TRF's `Lane` mapping, ∈ {1, 2, 4, 8}) and temporally via `TimeBroadcast`.
`TimeBroadcast` covers factors of TRF `Element` not in the input `Time`, and also any purely-output axes in `OutTime` that appear in neither the input nor the TRF: the same broadcast machinery replicates the packet across both.
Each destination receives the same `OutPacket`:

```text
Lane      = LaneBroadcast
OutTime   = [PackTime, TimeBroadcast]
OutPacket = PackPacket
```

`TimeBroadcast` factors occupy the *innermost* positions of `OutTime`: the same `OutPacket` is re-sent across those factors before iterating any outer `PackTime` factor.

### Examples

The example below exercises both operations: Packing absorbs the innermost size-2 factor `L` of `Time` into `Packet` (`PackSize = 2`), Lane Broadcast distributes the resulting packet to `N = 8` lanes, and Time Broadcast tiles the streaming data across a TRF-only `B = 5` axis.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 32, N = 8, K = 16, L = 2, B = 5];

fn stream_adapter_example<'l, const T: Tu>(
    input: CollectTensor<'l, { T }, bf16, m![1], m![1], m![1], m![M, L], m![K]>,
    trf: &TrfTensor<bf16, m![1], m![1], m![1], m![N], m![B, L, K]>,
) -> ContractOuterTensor<'l, { T }, bf16, m![1], m![1], m![1], m![N], m![M, B], m![L, K]> {
    // Packing (PackSize = 2):
    //   L = 2 (innermost Time) absorbed into Packet.
    //   PackTime = [M = 32], PackPacket = [L = 2, K = 16] = 32 bf16 = 64B.
    // Lane Broadcast: same packet to all N = 8 lanes.
    // Time Broadcast: B = 5 (TRF-only) added at innermost OutTime.
    //   OutTime = [M, B = 5], OutPacket = [L = 2, K = 16].
    input.contract_outer::<m![M, B], m![L, K], _, _>(trf)
}
```

### Constraints

- `OutPacket::SIZE * D::SIZE ∈ {32, 64}` bytes (on RNGD): 32 for `PackSize = 1`, 64 for `PackSize = 2`. The user picks this size and Packing's collect-flit count follows.
- `PackSize ∈ {1, 2}` (see [Packing](#packing)).
- `Lane::SIZE ∈ {1, 2, 4, 8}`.

### Performance

`PackSize` sets MAC utilization.
`PackSize = 2` fills the full 64 B and uses all MACs.
`PackSize = 1` fills only 32 B, so the zero-padded half always multiplies by zero and effective throughput halves.

`PackSize = 2` takes 2 cycles per Packet (two 32 B flits combine into one 64 B Packet), but this is not a pipeline bottleneck: upstream supplies one 32 B flit every cycle at the full fetch rate, so the Stream Adapter consumes flits as fast as they arrive and emits a Packet every 2 cycles to match downstream consumption.

Time Broadcasting amortizes fetches.
Broadcast factors reuse the same streaming packet across cycles without re-fetching, which eliminates bandwidth cost for those factors.

Fetch bandwidth (up to 32 B/cycle per fetch) bounds the Stream Adapter overall.
Interleave fetch patterns across slices to maximize utilization.

## TRF Sequencer

The TRF Sequencer reads a `TrfTensor` and reshapes its `Element` into `OutTime` / `OutPacket` for the [Packet Reducer](./packet-reducer.md).
See [Register Files](../register-files.md) for the TRF storage layout (lanes, banks, rows, double-buffering, cache).

The mapping:

```text
OutTime   = (sequencing over [Element / ReadSize] with broadcasts)
OutPacket = [PacketBroadcast, Element % ReadSize]
```

`OutPacket` is filled each cycle by one TRF read: one full `OutPacket` (640 bits per lane across both banks, or 320 bits when only one bank is read) is generated every cycle.
See [To Contraction Engine](../register-files.md#to-contraction-engine) for the per-slice totals across 8 lanes (in bytes).
The read pulls the innermost contiguous portion of `Element` and replicates it to fill the 64 B `OutPacket`.
The compiler picks the largest `ReadSize` such that `Element % ReadSize == OutPacket % ReadSize` and `ReadSize * D::SIZE ≤ 64` bytes: a wider `ReadSize` spans both TRF banks per lane, a narrower one uses just one bank.

`OutTime` is filled across cycles by sequencing `Element / ReadSize` (plus optional broadcasts).
The TRF Sequencer uses the same nested-loop configuration as all other [sequencers](../../moving-tensors/sequencer.md).


### Examples

In this example, `ReadSize` covers all of `Element` in one 64 B read, so `Element / ReadSize` is trivial and the sequencer iterates only broadcasts:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 32, N = 8, K = 32];

fn trf_sequencer_full_read<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1], m![1], m![M, K / 16], m![K % 16]>,
    trf: &TrfTensor<bf16, m![1], m![1], m![1], m![N], m![K]>,
) -> ContractOuterTensor<'l, T, bf16, m![1], m![1], m![1], m![N], m![M], m![K]> {
    // Element         = K
    // ReadSize        = 32
    // PacketBroadcast = 1
    // OutTime         = M      (sequencing over [K / 32] (= 1), broadcast M)
    // OutPacket       = K      (= [1, K % 32])
    input.contract_outer::<m![M], m![K], _, _>(trf)
}
```

In this example, `ReadSize` covers only part of `Element`, so `Element / ReadSize` is non-trivial and the sequencer iterates the outer `Element` factor alongside a broadcast:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![M = 32, N = 8, K = 16, L = 2, O = 2];

fn trf_sequencer_partial_read<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1], m![1], m![O, M, L], m![K]>,
    trf: &TrfTensor<bf16, m![1], m![1], m![1], m![N], m![O, K]>,
) -> ContractOuterTensor<'l, T, bf16, m![1], m![1], m![1], m![N], m![O, M], m![L, K]> {
    // Element         = [O, K]
    // ReadSize        = 16
    // PacketBroadcast = L
    // OutTime         = [O, M]    (sequencing over [O, K] / 16 (= O), broadcast M)
    // OutPacket       = [L, K]    (= [L, [O, K] % 16])
    input.contract_outer::<m![O, M], m![L, K], _, _>(trf)
}
```

### Constraints

- **Hardware dimensions**: `Chip::SIZE`, `Cluster::SIZE`, and `Slice::SIZE` must match the hardware configuration (see [Sequencer](../../moving-tensors/sequencer.md#configuration)).
- **Address alignment**: when `Element % ReadSize` covers all 64 B, the read spans both TRF banks per lane, so the sequencer's base address and all strides must align to 64 B.

### Architecture

Across cycles, the sequencer iterates the outer factors of `Element` (i.e. `Element / ReadSize`) using the same nested-loop configuration as all other [sequencers](../../moving-tensors/sequencer.md), so a single `TrfTensor` walks `Element / ReadSize` cycles before exhausting its content.
`PacketBroadcast` factors replicate the same row within a single cycle, filling the 64 B `OutPacket` past the natural `ReadSize` without consuming additional TRF read bandwidth.

### Performance

Throughput is one full `OutPacket` per lane per cycle: 640 bits per lane when both banks are read, 320 bits per lane when only one bank is read.
See [Register Files: To Contraction Engine](../register-files.md#to-contraction-engine) for the per-slice byte totals.
The TRF read cache and bank alternation (see [Register Files: Double Buffering](../register-files.md#double-buffering)) keep the concurrent sub-context store unblocked across broadcast reuse and narrow reads.

## Multiplier

The Multiplier consumes the two aligned operands from the Stream Adapter and TRF Sequencer, widens each input element to the contraction output type (`i4`/`i8` -> `i32`, `f8`/`bf16` -> `f32`) to keep the downstream accumulator from overflowing, and multiplies them elementwise.
Its output, a single tensor in the joint mapping `[Chip, Cluster, Slice, Lane, Time, Packet]`, becomes the input to the [Packet Reducer](./packet-reducer.md).
Each `Time` cycle, every `Lane` produces a full `packet` of products in parallel.
