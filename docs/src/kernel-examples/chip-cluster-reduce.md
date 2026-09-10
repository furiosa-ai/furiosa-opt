# Case Study: Chip and Cluster Reduction

Chip and cluster reduction combines partial values stored along different `Chip` or `Cluster` dimensions.
The examples implement ReduceScatter, AllGather, their composition, and direct AllReduce for both dimensions.
A separate chip example implements Butterfly AllReduce.

## Shape Transformations

The notation `[dimension], [local]` separates the dimension shape from the shape stored locally at each dimension index.
Let `R` be a dimension to reduce, `S` a local axis to distribute or gather, and `P...` and `Q...` the local shapes before and after `S`.
For ReduceScatter and AllGather, `R` and `S` must have the same size `N`.
`Broadcast(N)` means that all `N` indices along the dimension hold the same value.
The table describes the logical result and preserves the order of local axes that are not removed.

| Operation | Shape transformation |
|-----------|----------------------|
| ReduceScatter | `[R], [P..., S, Q...] -> [S], [P..., Q...]` |
| AllGather | `[S], [P..., Q...] -> [Broadcast(N)], [P..., S, Q...]` |
| AllReduce | `[R], [P...] -> [Broadcast(N)], [P...]` |

ReduceScatter reduces `R` and distributes the resulting `S` axis: dimension index `s` retains coordinate `s` of `S`.
AllGather reverses that distribution by collecting every `S` coordinate at every dimension index.
AllReduce preserves the local shape while reducing `R` and replicating the result.
Direct cyclic reduction, ReduceScatter followed by AllGather, and butterfly exchanges have the same AllReduce shape transformation.
These transformations apply to both chip and cluster dimensions.

### Floating-Point Reduction Order

`Broadcast(N)` is a logical placement guarantee.
Independently computed floating-point replicas may differ because cyclic reductions use orders such as `0 + 1 + 2 + 3` and `1 + 2 + 3 + 0`.
AllGather after ReduceScatter computes each shard once and gathers it, so those replicas are identical.
The examples use `i32`, where rounding order does not apply.

## Redistribution Primitives

The examples use the deferred SRAM [redistribution plan](../moving-tensors/dma-engine.md#redistribution-operations).
Shuffle and slice stages before `to_dm` fuse into one DMA.
Direct `asymmetric_chip_slice` and `asymmetric_cluster_slice` calls instead use the sub-context without moving data along either dimension.

### Synchronization

The compiler derives synchronization from the final redistribution plan when `to_dm` materializes it.
A source chip change inserts `ChipSync`, and a source cluster change inserts `ClusterSync`.
A fused plan that changes both dimensions gets both synchronization protocols.
Local `DmTensor` slicing adds neither.

The compiler emits a destination-readiness sync before remote DMA access and a completion sync afterward.
A self-to-self chip route emits no inter-chip send or receive.

| Example | Redistribution calls | Automatically inserted synchronization |
|---------|----------------------|----------------------------------------|
| Chip ReduceScatter | 4 `chip_shuffle().chip_slice().to_dm()` chains | 3 `ChipSync` protocols; round 0 is self-to-self, while rounds 1-3 communicate between chips |
| Chip AllGather | 1 dimension-to-element `to_dm()` | 1 `ChipSync` protocol |
| Chip AllGather after ReduceScatter | ReduceScatter plus AllGather | 4 `ChipSync` protocols |
| Chip AllReduce | 3 `chip_shuffle().to_dm()` chains; the local value is already present | 3 `ChipSync` protocols |
| Chip Butterfly AllReduce | 2 `chip_shuffle().to_dm()` chains | 2 `ChipSync` protocols |
| Cluster ReduceScatter | 2 local slices and 1 `cluster_swap().to_dm()` chain | 1 `ClusterSync` protocol |
| Cluster AllGather | 1 dimension-to-element `to_dm()` | 1 `ClusterSync` protocol |
| Cluster AllGather after ReduceScatter | ReduceScatter plus AllGather | 2 `ClusterSync` protocols |
| Cluster AllReduce | 1 `cluster_swap().to_dm()` chain | 1 `ClusterSync` protocol |

These counts describe the explicit redistribution calls in the examples.
HBM transfers and local Vector Engine work have their own dependencies but do not add chip or cluster redistribution syncs.

## Chip Examples

The examples use four-chip `A` and `B` axes and a 256-byte `[C, D]` payload.
Chip and shard indices range from 0 through 3.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/chip_reduce.rs:axes}}
```

The reduction examples share this Vector Engine addition:

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/chip_reduce.rs:add_pair}}
```

### ReduceScatter

The input places `A` across chips and keeps `[B, C, D]` local.
Each `chip_shuffle().chip_slice().to_dm()` chain selects one `B` coordinate and routes it to a target chip.
The three Vector Engine chains then sum the four routed `[C, D]` values.
Here, `in[p, s]` is source chip `p`'s value for shard `s`, and `out[s] = sum_p in[p, s]`.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/chip_reduce.rs:reduce_scatter}}
```

| Target chip | Round 0 | Round 1 | Round 2 | Round 3 | Result |
|-------------|---------|---------|---------|---------|--------|
| 0 | `in[0,0]` | `in[1,0]` | `in[2,0]` | `in[3,0]` | `out[0]` |
| 1 | `in[1,1]` | `in[2,1]` | `in[3,1]` | `in[0,1]` | `out[1]` |
| 2 | `in[2,2]` | `in[3,2]` | `in[0,2]` | `in[1,2]` | `out[2]` |
| 3 | `in[3,3]` | `in[0,3]` | `in[1,3]` | `in[2,3]` | `out[3]` |

The result remains sharded: target chip `s` owns `out[s]`.

### AllGather

The input gives source chip `s` one `[C, D]` shard, denoted by `in[s]`.
One `to_dm()` moves the chip dimension into the local element shape and broadcasts that shape to every chip.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/chip_reduce.rs:all_gather}}
```

| Target chip | Local element after `to_dm()` |
|-------------|-------------------------------|
| 0 | `[in[0], in[1], in[2], in[3]]` |
| 1 | `[in[0], in[1], in[2], in[3]]` |
| 2 | `[in[0], in[1], in[2], in[3]]` |
| 3 | `[in[0], in[1], in[2], in[3]]` |

Every target receives every shard.

### AllGather after ReduceScatter

This example feeds the ReduceScatter result directly into AllGather.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/chip_reduce.rs:all_gather_after_reduce_scatter}}
```

Its final placement is the AllGather state shown above.

### AllReduce

The input gives chip `p` one partial `in[p]` with shape `[C, D]`.
The original tensor supplies the identity rotation, and three chip shuffles supply the remaining cyclic rotations.
Adding all four rotations leaves `out[target] = sum_i in[i]` on every target chip.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/chip_reduce.rs:all_reduce}}
```

| Target chip | Local | Rotation 1 | Rotation 2 | Rotation 3 | Result |
|-------------|-------|------------|------------|------------|--------|
| 0 | `in[0]` | `in[1]` | `in[2]` | `in[3]` | `out[0] = sum_i in[i]` |
| 1 | `in[1]` | `in[2]` | `in[3]` | `in[0]` | `out[1] = sum_i in[i]` |
| 2 | `in[2]` | `in[3]` | `in[0]` | `in[1]` | `out[2] = sum_i in[i]` |
| 3 | `in[3]` | `in[0]` | `in[1]` | `in[2]` | `out[3] = sum_i in[i]` |

### Butterfly AllReduce

The input gives chip `p` one partial `in[p]` with shape `[C, D]`.
Two shuffle-and-add rounds leave `sum_i in[i]` on every chip.
Here, `i` ranges over all source chips.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/chip_reduce.rs:butterfly_all_reduce}}
```

| Target chip | Initial | First peer | First sum | Second peer sum | Final |
|-------------|---------|------------|-----------|-----------------|-------|
| 0 | `in[0]` | `in[1]` | `in[1]+in[0]` | `in[3]+in[2]` | `out[0] = sum_i in[i]` |
| 1 | `in[1]` | `in[0]` | `in[0]+in[1]` | `in[2]+in[3]` | `out[1] = sum_i in[i]` |
| 2 | `in[2]` | `in[3]` | `in[3]+in[2]` | `in[1]+in[0]` | `out[2] = sum_i in[i]` |
| 3 | `in[3]` | `in[2]` | `in[2]+in[3]` | `in[0]+in[1]` | `out[3] = sum_i in[i]` |

## Cluster Examples

The cluster examples use two-element `A` and `B` axes with the same 256-byte `[C, D]` payload.
Cluster and shard indices range from 0 through 1.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/cluster_reduce.rs:axes}}
```

The cluster examples use the corresponding two-cluster addition:

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/cluster_reduce.rs:add_pair}}
```

### ReduceScatter

The first local `cluster_slice` selects `B` coordinates `[0, 1]`, one value per cluster.
The second selects `[1, 0]`, and `cluster_swap().to_dm()` moves those values to the opposite clusters.
The Vector Engine chain then adds the local and moved `[C, D]` values.
Here, `in[c, s]` is source cluster `c`'s value for shard `s`, and `out[s] = sum_c in[c, s]`.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/cluster_reduce.rs:reduce_scatter}}
```

| Target cluster | Local | Remote before swap | Remote after swap | Result |
|----------------|-------|--------------------|-------------------|--------|
| 0 | `in[0,0]` | `in[0,1]` | `in[1,0]` | `out[0]` |
| 1 | `in[1,1]` | `in[1,0]` | `in[0,1]` | `out[1]` |

### AllGather

The input gives source cluster `s` one `[C, D]` shard, denoted by `in[s]`.
One `to_dm()` moves the cluster dimension into the local element shape and broadcasts that shape to both clusters.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/cluster_reduce.rs:all_gather}}
```

| Target cluster | Local element after `to_dm()` |
|----------------|-------------------------------|
| 0 | `[in[0], in[1]]` |
| 1 | `[in[0], in[1]]` |

### AllGather after ReduceScatter

This example feeds the cluster ReduceScatter result directly into AllGather.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/cluster_reduce.rs:all_gather_after_reduce_scatter}}
```

### AllReduce

The input gives cluster `c` one partial `in[c]` with shape `[C, D]`.
One cluster swap exchanges the two partials, and one addition leaves the complete reduction on both clusters.

```rust,ignore
{{#include ../../../furiosa-opt-examples/src/cluster_reduce.rs:all_reduce}}
```

| Target cluster | Local | Swapped peer | Result |
|----------------|-------|--------------|--------|
| 0 | `in[0]` | `in[1]` | `out[0] = sum_i in[i]` |
| 1 | `in[1]` | `in[0]` | `out[1] = sum_i in[i]` |

## DMA Layout Requirements

See [Redistribution Operations](../moving-tensors/dma-engine.md#redistribution-operations) for the alignment, live-index, and 256-byte contiguous-run requirements.
These examples use `m![B, C, D]`, whose inner `m![C, D]` run is exactly 256 bytes of `i32` data.
