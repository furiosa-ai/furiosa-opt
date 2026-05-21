# Packet Reducer

The Packet Reducer reduce-adds the innermost contracted axes within a single Packet, with one reduction tree per lane.

## Interface

`.contract_packet()` invokes the Packet Reducer.
Each lane receives a 32 B or 64 B Packet of `i4`, `i8`, `f8`, or `bf16` elements, inherited from the [Outer](./outer.md) stage's `OutPacket`.
Formally, it computes \\(\text{output}[i] = \sum_{j} \text{input}[i, j]\\), where `i` ranges over the surviving (output) axes and `j` ranges over the contracted axes inside `Packet`.

```rust,ignore
{{#include ../../../../furiosa-opt-std/src/engine/contraction/packet.rs:contract_packet_def}}
```

The kernel below uses all 8 lanes in parallel: tree depth 5 sums over the 32 `bf16` elements of `B`, producing one `f32` per `A` position.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 32, B = 32, C = 8];

fn matmul<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1], m![1], m![A, B / 16], m![B % 16]>,
    trf: &TrfTensor<bf16, m![1], m![1], m![1], m![C], m![B]>,
) -> ContractTensor<'l, T, f32, m![1], m![1], m![1], m![A], m![C]> {
    //
    // Spatial reduction: tree depth 5 reduces 32 bf16 elements along B → f32
    // Output (Interleaved): Time = [A], Packet = [C]
    input.contract_outer::<m![A], m![B], _, _>(&trf)
         .contract_packet::<m![1]>()
         .contract_time::<m![A]>()
         .contract_lane::<m![A], m![C]>(LaneMode::Interleaved)
}
```

## Architecture

```text
ReducePacket = Packet / 2^d        for 0 ≤ d ≤ log2(Packet::SIZE)
OutPacket    = ReducePacket        if ReducePacket::SIZE ≤ 32
               ReducePacket = 32   otherwise
```

The Packet Reducer first runs an independent [reduction tree](https://en.wikipedia.org/wiki/Reduction_operator) per lane on its input Packet.
At depth 0, the tree leaves hold the input Packet's elements, and each subsequent depth sums pairs, halving the element count.
The maximum tree depth is `log2(Packet::SIZE)`, so 7 for `i4` (`Packet::SIZE = 128`), 6 for `i8` / `f8` (64), 5 for `bf16` (32).
Given the user's `OutPacket`, the compiler derives the tree depth `d`, and the tree consumes the innermost `2^d` elements to produce `ReducePacket`.

The Packet Reducer then clips `ReducePacket` to `OutPacket`, whose size is capped at 32 elements because the downstream Time Reducer's per-lane accumulator only has 32 columns.
When `ReducePacket::SIZE > 32`, the outer dummy is sliced and only the innermost 32 elements survive.
For example, `i4` arrives as a 128-element Packet, so `d ∈ {0, 1}` produce a 128- or 64-element `ReducePacket`, both clipped to `OutPacket::SIZE = 32`.



## Performance

Latency depends on tree depth: 7 cycles for `i4`, 6 for `i8`/`f8`, 5 for `bf16`.
Wider element types reduce depth because fewer elements fit in each Packet.
The adder tree is fully pipelined, so depth adds first-output latency but does not reduce steady-state throughput: one Packet enters and one reduced output emerges every cycle once the pipeline is filled.

When `Lane < 8`, the inactive lanes' reduction trees are idle, so per-cycle throughput drops proportionally to `Lane::SIZE / 8`.
