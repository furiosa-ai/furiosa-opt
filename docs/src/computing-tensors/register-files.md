# Register Files

The [Collect Engine](./collect-engine.md) streams into the [Contraction Engine](./contraction-engine/index.md) and the [Vector Engine](./vector-engine/index.md).
Both engines also take input from a register file (one per slice): the Tensor Register File (TRF) feeds the Contraction Engine, and the Vector Register File (VRF) feeds the Vector Engine.
These register files must be populated before their consumer engine runs.

## Tensor Register File

### Interface

A `TrfTensor` is a tensor stored in the TRF:

```rust,ignore
{{#include ../../../furiosa-opt-std/src/tensor/memory.rs:trf_tensor_def}}
```

`Chip` / `Cluster` / `Slice` pass through from the source.
`Lane` indexes the spatial parallelism (1, 2, 4, or 8 active lanes).
`Element` holds the per-lane layout.

#### From Collect Engine

`.to_trf::<Lane, Element>()` on `CollectTensor` produces a `TrfTensor`.
Where in the register file it lands is the compiler's to decide, so the call names no region:

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/collect.rs:collect_to_trf}}
```

`.to_trf` reshapes the streaming `Time` / `Packet` into `Lane` / `Element`:

```text
Lane    = Time / FlitsPerLane
Element = [Time % FlitsPerLane, Packet]
```

for some `FlitsPerLane` that the compiler derives from `Lane` and `Time`, so each lane is filled by `FlitsPerLane` consecutive flits.

The following simple case occupies the full TRF:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 32];

fn load_trf<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![1], m![B]>,
) -> TrfTensor<i8, m![1], m![1 # 2], m![1 # 256], m![1], m![B]> {
    input.to_trf()
}
#
# let mut ctx = Context::acquire();
#
# let c: CollectTensor<'_, _, i8, m![1], m![1 # 2], m![1 # 256], m![1], m![B]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = load_trf(c);
```

For example, in a matmul kernel `Lane` holds output channels and `Element` holds the contracted axis.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![V = 32, M = 32, N = 8, K = 32];

type Chip    = m![1];
type Cluster = m![V / 16];
type Slice   = m![V % 16 # 256];
type Lane    = m![N];

/// Stores matmul weights into TRF for consumption by `bmatmul` in
/// [Contraction Engine: Example: Batched MatMul](./contraction-engine/index.md#example-batched-matmul).
fn store_bmatmul_trf<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, Chip, Cluster, Slice, m![N, K / 16], m![K % 16]>,
) -> TrfTensor<bf16, Chip, Cluster, Slice, Lane, m![K]> {
    input.to_trf()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, bf16, Chip, Cluster, Slice, m![N, K / 16], m![K % 16]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = store_bmatmul_trf(c);
```

#### From Data Memory

For completely contiguous input access (no gaps or reordering), TRF supports a *short command* (StoTRF), a compact hardware instruction that loads data from Data Memory directly into the TRF, bypassing the full Fetch → Switch → Collect → `to_trf()` pipeline.
The shortcut trades arbitrary-layout support for lower setup overhead.


#### To Contraction Engine

Each read covers 8 lanes × (1 or 2) banks × 1 row × 320 bits per bank, producing 320 bits per lane per cycle for a narrow read (one bank) or 640 bits per lane per cycle for a wide read (both banks): all active lanes access one or both banks of the same row in parallel (one bank for narrow reads, both for wide reads).
Per slice that totals 320 bytes/cycle (narrow) or 640 bytes/cycle (wide) across all 8 lanes.
See [TRF Sequencer](./contraction-engine/outer.md#trf-sequencer) for how the sequencer iterates these reads across rows and broadcasts.

### Architecture

The TRF is a banked SRAM with the structure 8 lanes × 2 banks × 128 rows × 320 bits = 80 KB per slice.
The 8 lanes operate in parallel, with 1, 2, 4, or 8 active per access.

How many elements pack into a single 320-bit row depends on the data type:

| Type | Element size stored | Elements per row |
|------|---------------------|------------------|
| `i4` → `i5` (hardware behaviour; not wired, so an `i4` weight stays `i4` today) | 5 bits | 64 |
| `i8` → `i9` | 9 bits (approx; rounds up to fit 320-bit row) | 32 |
| `i8` / `f8` | 8 bits | 32 (40 bytes per row) |
| `bf16` | 16 bits | 16 (32 bytes per row) |

Only integers promote on store: `i4 → i5` (5 bits) and `i8 → i9` (9 bits) leave room for the fetch adapter's optional zero-point subtraction, which widens an integer weight by one bit.
`i8` / `f8` and `bf16` stay at their native widths (8 and 16 bits respectively); the 320-bit row holds extra slack relative to a flat 8- or 16-bit packing so the same physical row width serves all types.


With fewer than 8 active lanes, each active lane sees more rows (as if the row count grew).
Halving the active count doubles the rows per active lane (e.g., 4 active → 256 rows per bank, 1 active → 1024).

### Double Buffering

The TRF enables double-buffering by splitting each bank into two halves: the [TRF Sequencer](./contraction-engine/outer.md#trf-sequencer) loads from one half while a store fills the other, and the two can be flipped between iterations.
Three address modes name the region, and the scheduler picks one at store time: `Full` uses all 128 rows per bank, `FirstHalf` uses rows 0–63, and `SecondHalf` uses rows 64–127.
The half modes cap per-slice capacity at 40 KB.

See [Scheduling and Tuning: Double Buffering](../scheduling/tuning.md#double-buffering) for the tuning pattern that uses these halves across the main and sub contexts.

Both halves share the same banks, so reads and writes contend at the bank level even though they target different rows.
When both target the same bank in a cycle, the read takes priority because the contraction pipeline needs the data this cycle while the store can wait.

The TRF mitigates this contention with a read cache and bank alternation.

TRF reads have heavy reuse: the same data is typically broadcast across many cycles, so a direct-mapped read cache (8 lanes × 2 banks × 4 rows × 320 bits = 2.5 KB) sits in front of the banks and absorbs repeated reads.
The cache also relieves contention with the concurrent store.
On a hit the read skips the bank, so the store can use it that cycle.
On a miss the cache refills from the bank, occupying it for that cycle.

For narrow reads (≤ 32 bytes), bank alternation adds a second mitigation.
Reads use only one bank, so they can alternate at 32-byte granularity across the two banks.
Reads and writes then end up on different banks on successive cycles, avoiding contention even on cache misses.
Wide reads (64 bytes) occupy both banks every cycle, so each cache miss blocks the concurrent store; narrow reads preserve half-bandwidth alternation even when the cache misses.


## Vector Register File

The VRF is written either from the [Collect Engine](./collect-engine.md) or directly from Data Memory, and read by the [Vector Engine](./vector-engine/index.md).

### Interface

A `VrfTensor` is a tensor stored in the VRF:

```rust,ignore
{{#include ../../../furiosa-opt-std/src/tensor/memory.rs:vrf_tensor_def}}
```

`Chip` / `Cluster` / `Slice` pass through from the source.
`Element` holds the per-(slice) layout.

#### From Collect Engine

`.to_vrf::<Element2>()` on `CollectTensor` stores the flits into the VRF and produces a `VrfTensor`.
Where in the register file they land is the compiler's to decide, so the call names no address:

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/collect.rs:collect_to_vrf}}
```

`.to_vrf` flattens the streaming `Time` / `Packet` into `Element2`:

```text
Element2 = [Time, Packet]
```

The user picks `Element2`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 64];

fn store_vrf<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![B / 8], m![B % 8]>,
) -> VrfTensor<i32, m![1], m![1 # 2], m![1 # 256], m![B]> {
    input.to_vrf()
}
# 
# let mut ctx = Context::acquire();
# 
# let c: CollectTensor<'_, _, i32, m![1], m![1 # 2], m![1 # 256], m![B / 8], m![B % 8]> = CollectTensor::new(&mut ctx.main, Tensor::zero());
# let _o = store_vrf(c);
```

#### From Data Memory

For completely contiguous input access (no gaps or reordering), VRF supports a *short command* (StoVRF), a compact hardware instruction that loads data from Data Memory directly into the VRF, bypassing the full Fetch → Switch → Collect → `to_vrf()` pipeline.
The shortcut trades arbitrary-layout support for lower setup overhead.


### Architecture

