# DMA Engine

The DMA Engine moves tensors directly between memory tiers without engaging the Tensor Unit pipeline.
Each transfer pairs two coordinated stages:
- **[Read Sequencer](#architecture)**: Reads from the source tier.
- **[Write Sequencer](#architecture)**: Writes to the destination tier, possibly with a layout transformation.

A DMA transfer is a [mathematical tensor move](../mapping-tensors/tensor-semantics.md#mathematical-tensor-move): the output holds the same mathematical tensor as the input even when the layouts differ.
Tensor DMA spans cross-DMN, cross-cluster, and cross-chip transfers, with chip IDs globally agreed across the system.


See [Optimizations](#optimizations) for transfer throughput considerations.

## Interface

A DMA transfer takes a tensor in one memory tier and produces a tensor in another (or the same) tier.
The kernel writer calls `.to_dm()`, `.to_hbm()`, or related methods on the source tensor, passing in a `DmaContext`:
- `Context::tdma`: Tensor DMA context for on-chip transfers (HBM ↔ HBM, HBM ↔ DM, DM ↔ DM).
- `Context::pdma`: PCIe DMA context for host ↔ HBM transfers (see [PCIe DMA](#pcie-dma)).

```rust,ignore
{{#include ../../../furiosa-opt-std/src/tensor/memory.rs:dma_impl}}
```

The compiler derives the read and write sequencer configurations from the source and destination tensor types.
The kernel writer specifies the destination type's `Cluster`, `Slice`, and `Element` (for DM tensors) or `Element` (for HBM tensors), which encode the layout transformation.

The example below transposes a tensor from `[A, B, C]` to `[C, A, B]` using two HBM-to-HBM transfers:

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 16, C = 32];

fn transpose_simple(
    ctx: &mut Context,
    input: &HbmTensor<f32, m![1], m![A, B, C]>,
) -> HbmTensor<f32, m![1], m![C, A, B]> {
    // Step 1: [A, B, C] → [A, C, B]
    let intermediate: HbmTensor<f32, m![1], m![A, C, B]> = input.to_hbm(&mut ctx.tdma, 0);

    // Step 2: [A, C, B] → [C, A, B]
    intermediate.to_hbm(&mut ctx.tdma, 0x1000)
}
```

A transfer that crosses tiers also takes a layout transformation through the destination type's mapping.
For an HBM-to-DM transfer, the destination DM tensor adds `Cluster` and `Slice` axes that distribute the tensor across hardware partitions.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 2048];

fn hbm_to_dm(
    ctx: &mut Context,
    input: &HbmTensor<i8, m![1], m![A]>,
) -> DmTensor<i8, m![1], m![1 # 2], m![A / 8], m![A % 8]> {
    input.to_dm::<m![1 # 2], m![A / 8], m![A % 8]>(&mut ctx.tdma, 0)
}
```

Here the 2,048-element vector is distributed as 256 elements per slice (`Slice = m![A / 8]`) with 8 elements per slice (`Element = m![A % 8]`), spread across 2 clusters.

## Architecture

Each RNGD chip holds 8 DMA Engines, one per pair of DMNs, running up to 8 independent transfers in parallel.
A single DMA Engine runs paired read and write sequencers, and a tensor move spreads across multiple engines through an aggregate.
The subsections below describe its static structure, sequencer representation, dynamic behavior, compiler derivation, and aggregate operations.

> [!NOTE]
> The Tensor Unit (via [Fetch](./fetch-engine.md) and [Commit](./commit-engine.md) Engines) is often more efficient than DMA for SRAM-to-SRAM transfers, since DMA may underutilize SRAM slice bandwidth.
> HBM bandwidth, however, is typically the bottleneck in practice, making this gap less critical for HBM ↔ DM transfers.

### Static Structure

`Chip`, `Cluster`, and `Slice` are the hardware spatial parallelism dimensions.
The 8 DMA Engines per chip transfer between different memory components in parallel (e.g., engine #0 handles HBM ↔ DM while engine #1 handles DM ↔ DM).

Each DMA Engine runs paired read and write sequencers in lockstep.
The read sequencer traverses source addresses, the write sequencer traverses destination addresses.
Both share the same loop count but use different strides and base addresses, since the layout transformation reorders how the same logical elements appear in source vs. destination memory.
The compiler represents the pair compactly as a single sequencer with paired strides per loop entry, exploiting the matched read and write counts.

The compiler distributes a single tensor move across the available DMA Engines, partitioning the work along chip, cluster, and slice dimensions and assigning each partition to a DMA Engine.
Any DMA Engine handles any transfer.
By default the compiler picks the source DM's local DMA Engine, since local DMN access is faster than cross-DMN access.
The kernel writer can also specify an engine explicitly.

### Sequencer Representation


The compiler represents each DMA Engine's work as a `DmaSequencer` paired with source and destination addressing:

```rust,ignore
struct DmaSequencer {
    entries: Vec<DmaEntry>,
    stride0: u16,        // 1..=4096, per-iteration packet size in bytes
    source_base: usize,
    dest_base: usize,
}

struct DmaEntry {
    axis: AxisName,
    size: usize,
    source_stride: isize,
    dest_stride: isize,
}
```

Each entry specifies a loop with a shared `size` but separate `source_stride` and `dest_stride`, since the layout transformation reorders the same logical elements between source and destination memory.
The innermost-loop stride `stride0` (1 to 4,096 bytes) sets the per-iteration packet size.
A full DMA command bundles the sequencer with the engine's location and media:

```rust,ignore
struct DmaDescriptor {
    sequencer: DmaSequencer,
    source_media: Media,
    dest_media: Media,
}

struct DmnIndex {
    chip: ChipIndex,
    cluster_in_chip: ClusterInChipIndex,
    slice_in_cluster: SliceInClusterIndex,
}

enum Media {
    Hbm(ChipIndex),
    Dm(DmnIndex),
    Spm(DmnIndex),
}

enum Dtype {
    I4, I8, F8E4M3, F8E5M2, I16, Bf16, F16, I32, F32,
}
```

A homogeneous aggregate uses one descriptor template parameterized across all participating DMA Engines.
A heterogeneous aggregate uses a `HashMap<DmnIndex, DmaDescriptor>` that pairs each DMN with its specific descriptor.
DM tensor specifications must include chip, cluster, and slice in the mapping expression to identify the exact memory location.

### Dynamic Behavior

Each loop in the `DmaSequencer` advances a counter in row-major order, deriving the read and write addresses from the paired strides.
For the sequencer below with `base = (0, 256³)`:

```text
[
  A -> 256 : (65,536, 256),
  B -> 256 : (256, 65,536),
  C -> 256 : (1, 1),
] : 256
```

| iteration `i`              | counters      | read addr | write addr                            |
|----------------------------|---------------|-----------|---------------------------------------|
| 0                          | `(0, 0, 0)`   | 0         | `write_base`                          |
| 1                          | `(0, 0, 1)`   | 1         | `1 + write_base`                      |
| ...                        | ...           | ...       | ...                                   |
| 255                        | `(0, 0, 255)` | 255       | `255 + write_base`                    |
| 256                        | `(0, 1, 0)`   | 256       | `65,536 + write_base`                 |
| `i = a·256² + b·256 + c`   | `(a, b, c)`   | `i`       | `256·a + 256²·b + c + write_base`     |

With `stride0 = 256`, the hardware reads and writes 256 bytes per iteration, so iteration 0 processes all values for `(A, B, C) = (0, 0, 0..255)` as a single packet.
The transfer completes in approximately 500 cycles of startup latency plus 256 × 256 cycles of data transfer.

### Compiler Derivation

Given source and destination tensor mappings (`In`, `Out`) and a stream shape (`Stream`), the compiler derives the read and write sequencers:
- **Read sequencer**: Projects `In` onto the stream shape, producing per-loop strides into the source tier.
- **Write sequencer**: Projects `Out` onto the stream shape, producing per-loop strides into the destination tier.
- **Unified sequencer**: Merges the two so each entry pairs the read and write strides.
- **Packet size**: Infers `stride0` from the consecutive read/write volume.
  When both read and write access 256 consecutive bytes, the optimal `stride0` is 256.

For the layout transformation `m![A, B, C] → m![B, A, C]` over `axes![A=256, B=256, C=256]` with `Stream = m![A, B, C]`, the compiler uses the index relation `m![A, B, C]::map(i) = i![A: i / 65,536, B: (i % 65,536) / 256, C: i % 256]` (see [Mapping Expressions](../mapping-tensors/mapping-expressions.md) for the notation) to derive:

```text
read_sequencer  = [
  A -> 256 : 65,536,
  B -> 256 : 256,
  C -> 256 : 1,
] : 256, HBM @ 0
write_sequencer = [
  A -> 256 : 256,
  B -> 256 : 65,536,
  C -> 256 : 1,
] : 256, HBM @ 256³
```

These combine into a single `DmaSequencer` with paired strides per entry.
Each side of the unified `DmaSequencer` (read or write) can be displayed as a single-stride sequencer for that direction, omitting the paired-stride bracket for clarity.

### Aggregate Operations

The aggregate takes one of two forms based on whether tensor shapes divide evenly across DMNs.

When shapes divide evenly, all participating engines run a *homogeneous* aggregate.
Every DMA Engine uses the same parametric stream environment (`Stream = { chip, cluster, slice, time, packet }`), differing only by base address.

When shapes do not divide evenly, the compiler falls back to a *heterogeneous* aggregate.
Each DMN gets its own stream environment via `StreamFn(chip, cluster, slice)`, and boundary DMNs split their work across multiple DMA commands to avoid writing past the valid region.
The input and output mapping environments (`In` and `Out`) remain structurally identical to the homogeneous case, so the overall logical tensor move is well-defined.

Two correctness invariants apply across both forms:
- **Same media types**: All participating DMA Engines must use the same source and destination media.
- **Single unified mapping**: One input and one output tensor mapping govern the overall transfer.

Each command incurs its own startup latency, so prefer tensor shapes that divide evenly across DMNs to keep the aggregate homogeneous.

## Constraints

The DMA Engine enforces hardware-level alignment and packet-size rules.
Violations cause correctness errors or hardware exceptions, not just performance degradation.

- **Address alignment**:

  | Tier | Read | Write |
  |------|------|-------|
  | HBM | 1 byte | 1 byte |
  | DM (SRAM) | 1 byte | 8 bytes |

  HBM ↔ DM transfers additionally require 8-byte alignment for the read address, write address, and packet size, regardless of the table above.
  The asymmetric DM rule reflects asymmetric SRAM hardware.
  Read ports use byte-select logic to extract arbitrary byte ranges, but write ports operate on full 8-byte bank-width units.
  Misaligned DM writes therefore trigger a Read-Modify-Write operation that triples the write time and blocks other operations on the affected bank.

  The compiler enforces these constraints as hardware invariants.

- **Packet size**: The maximum packet size is 4,096 bytes, set by the AXI protocol constraint that transactions cannot exceed 256 beats × 16-byte data width.

## Optimizations

Three factors determine DMA throughput: memory bandwidth, channel and DMN interleaving, and startup latency with packet splitting.

### Memory Bandwidth

Each tier has a peak bandwidth that bounds achievable throughput, and the actual rate is limited by the slowest component on the streaming path.

| Tier | Peak bandwidth |
|------|----------------|
| HBM | 1.5 TB/s per chip (32 channels × 48 GB/s per channel at 0.75 GHz) |
| DM | 256 B/cycle per cluster (with DMN interleaving, 128 B/cycle per DMN) |
| SPM | 128 B/cycle per cluster (same-chip only, not yet exposed in the API) |
| PCIe | 30 B/cycle for both reads and writes (see [PCIe DMA](#pcie-dma)) |

Each DMA Engine moves up to 256 B/cycle on its own.
HBM bandwidth is shared across all engines transferring HBM data, so an aggregate saturating HBM is bounded by the 1.5 TB/s HBM peak rather than by any per-engine sum.

Same-cluster DM-to-DM transfers serialize their reads and writes, since both phases contend for the same DM bank access.
Cross-tier transfers like HBM ↔ DM pipeline the read and write phases.

> [!NOTE]
> The Tensor Unit (via [Fetch](./fetch-engine.md) and [Commit](./commit-engine.md) Engines) is often more efficient than DMA for SRAM-to-SRAM transfers, since DMA may underutilize SRAM slice bandwidth.
> HBM bandwidth, however, is typically the bottleneck in practice, making this gap less critical for HBM ↔ DM transfers.

### Channel and DMN Interleaving

Sustaining peak bandwidth requires interleaving access patterns across the underlying memory partitions.

HBM channel selection uses address bits 9 to 28, and address bit 8 is the stack bit.
Access patterns must toggle all of these bits to spread requests across all 32 channels.
Missing the stack bit (address bit 8) alone halves effective bandwidth by routing all requests to only 16 of the 32 channels.
Access patterns that hit the same HBM bank repeatedly (toggling row-address bits 21+ on consecutive accesses) trigger row-conflict penalties of approximately 40 cycles per access, degrading bandwidth by an order of magnitude.
FR-FCFS memory scheduling recovers some throughput, but the fundamental cost remains severe.

DM bandwidth requires alternating between both DMNs (each 128 B/cycle), so a single-DMN access pattern halves DM bandwidth.

### Startup Latency and Packet Splitting

Each DMA command incurs approximately 500 cycles of fixed startup latency before data transfer begins.
Combining multiple transfers into fewer commands amortizes this cost, while heterogeneous aggregates split into per-DMN commands and pay the latency on each command.

Within a single command, the hardware splits each packet into 256-byte units, so an n-byte packet becomes `ceil(n / 256)` AXI requests.
A 4,095-byte packet therefore costs 16 requests, while a 4,099-byte packet (a prime length, awkwardly placed past the 4,096-byte limit) requires splitting into multiple commands.
The innermost-loop stride (`stride0`) determines packet alignment: when `stride0` is 256-byte aligned, the cycle count is `ceil(stride0 / 256)`.
When `stride0` is not 256-byte aligned, HBM writes additionally pay a Read-Modify-Write penalty for the partial 256-byte block.
HBM reads incur only the `ceil` overhead, and DM operations are largely unaffected by this kind of misalignment.

DMA has the lowest DM bank-access priority, so 64+ consecutive same-bank accesses from the Fetch or Commit Engine can starve it and trigger a NoC timeout.
See [DM Bank Starvation](./memory-performance.md#bank-starvation) for details.

## Detailed Examples

The examples below give concrete sequencer configurations and cycle estimates for representative transfer patterns.
Examples 1 to 3 cover well-tuned single-engine cases for each tier pair.
Examples 4 and 5 contrast pathological access patterns that lose 10x or more.
Example 6 illustrates heterogeneous segmentation when shapes do not divide evenly across DMNs.

The sequencer configurations use two address-stride symbols:
- `slice_stride`: The virtual address span of one in-slice DM partition (4 MB).
- `DMN_stride`: The address span between two DMNs within the same cluster.

### Example 1: HBM ↔ HBM Layout Transformation

Arguments:
- `axes![A = 8, B = 8, C = 256]`
- `dtype = i8`
- Source: HBM at offset `0`, mapping `m![A, B, C]`
- Destination: HBM at offset `16,384`, mapping `m![B, A, C]`
- Stream: time `m![A, B]`, packet `m![C]`

Generated sequencers:

```text
read = [
  A -> 8   : 2,048,
  B -> 8   : 256,
  C -> 256 : 1,
] : 256, HBM @ 0

write = [
  A -> 8   : 256,
  B -> 8   : 2,048,
  C -> 256 : 1,
] : 256, HBM @ 16,384
```

The non-innermost strides (256 and 2,048) toggle HBM address bits 8 (stack) and 11 (channel), spreading every request across distinct HBM channels for parallel execution.
A single 256-byte transfer takes 4 cycles per channel at 0.75 GHz, but parallel channel distribution sustains near-peak bandwidth.
Total time: approximately 64 read requests + 64 write requests at 1 GHz, plus 500 cycles startup ≈ 628 cycles.

When 4 DMA Engines share HBM ↔ HBM traffic, each gets approximately 0.1875 TB/s out of the 0.75 TB/s read bandwidth.
Even with `stride0 = 256`, no single engine completes one request per cycle under that share.

### Example 2: HBM → DM with Full Bandwidth

This cross-tier transfer pipelines reads and writes by interleaving across HBM channels and both DMNs.

Arguments:
- `axes![A = 256, B = 256, C = 256]`
- `dtype = i8`
- Source: HBM at chip 0, mapping `m![B, A, C]`
- Destination: DM at chip 0, cluster 0, slice 0. Slice mapping `m![A / 4]`, element mapping `m![A % 4, B, C]`
- Stream: time `m![B, A % 4, A / 4 % 32, A / 128]`, packet `m![C]`

Generated sequencers:

```text
read = [
  B      -> 256 : 65,536,
  A%4    -> 4   : 256,
  A/4%32 -> 32  : 1,024,
  A/128  -> 2   : 32,768,
  C      -> 256 : 1,
] : 256, HBM @ 0

write = [
  B      -> 256 : 256,
  A%4    -> 4   : 65,536,
  A/4%32 -> 32  : slice_stride,
  A/128  -> 2   : DMN_stride,
  C      -> 256 : 1,
] : 256, DM @ 0
```

On the HBM side, the 32,768-stride on `A/128=2` interleaves access across channels, and the hardware command queue keeps all 65,536 requests (256 × 4 × 32 × 2) flowing.
On the DM side, `slice_stride` and `DMN_stride` interleave consecutive 256-byte writes across the two DMNs, keeping both at one request per cycle.
Reads and writes pipeline across tiers, so total time is approximately max(65,536 read cycles, 65,536 write cycles) + 500 startup ≈ 66,036 cycles.

### Example 3: DM → DM Within One Cluster

Same-cluster DM-to-DM transfers serialize reads and writes, since both contend for the same DM bank access.

Arguments:
- `axes![A = 256, B = 256, C = 256]`
- `dtype = i8`
- Source: DM at chip 0, cluster 0, slice 0, element offset `0`. Slice mapping `m![A / 4]`, element mapping `m![A % 4, B, C]`
- Destination: DM at chip 0, cluster 0, slice 0, element offset `4·256·256`. Slice mapping `m![A / 4]`, element mapping `m![B, A % 4, C]`
- Stream: time `m![B, A % 4, A / 4 % 32, A / 128]`, packet `m![C]`

Generated sequencers:

```text
read = [
  B      -> 256 : 1,
  A%4    -> 4   : 65,536,
  A/4%32 -> 32  : slice_stride,
  A/128  -> 2   : DMN_stride,
  C      -> 256 : 1,
] : 256, DM @ 0

write = [
  B      -> 256 : 1,024,
  A%4    -> 4   : 256,
  A/4%32 -> 32  : slice_stride,
  A/128  -> 2   : DMN_stride,
  C      -> 256 : 1,
] : 256, DM @ (4·256·256)
```

DMN and slice interleaving give each phase the full 256 B/cycle, but the two phases serialize.
Total time: approximately 131,072 cycles (65,536 reads + 65,536 writes) + 500 startup.

> [!NOTE]
> Choose `C` to be a multiple of 256 when possible.
> For `C = 256n + r` with `0 < r < 256`, the cycle count grows by a factor of `n+1` because each access splits into more requests, even though the total data volume changes only slightly.

### Example 4: HBM Bank-Conflict Pathology

DM interleaving is healthy here, but a pathological HBM access pattern costs roughly 10x the well-tuned cycle count.

Arguments:
- 1 chip (8 DMNs)
- `axes![A = 64, B = 2,048, C = 1,024]`
- `dtype = i8`
- Source: HBM, with cluster mapping `m![B / 1024]`, slice mapping `m![B / 256 % 4, A]`, element mapping `m![B % 256, C]`
- Destination: DM, with slice mapping `m![A / 4]`, element mapping `m![B, A % 4, C]`
- Stream: cluster `m![B / 1024]`, slice `m![B / 256 % 4]`, time `m![B % 256, C / 256, A % 32, A / 32]`, packet `m![C % 256]`

Generated sequencers per `(cluster_i, dmn_j)`:

```text
read = [
  B%256 -> 256 : 1,024,
  C/256 -> 4   : 256,
  A%32  -> 32  : 2²¹,
  A/32  -> 2   : 2²⁶,
  C%256 -> 256 : 1,
] : 256, HBM @ (i·2²⁰ + j·2¹⁸)

write = [
  B%256 -> 256 : 1,024,
  C/256 -> 4   : 256,
  A%32  -> 32  : slice_stride,
  A/32  -> 2   : DMN_stride,
  C%256 -> 256 : 1,
] : 256, DM @ (cluster_i, dmn_j, 0)
```

The strides on `A%32` and `A/32` toggle HBM address bits 21 and 26, which select the row within an HBM bank.
Consecutive accesses within each channel therefore close one row and open the next on nearly every request, paying approximately 40 cycles per access.
Channel interleaving via `C / 256 = 4` (stride 256) does spread requests across all 32 channels, but cannot hide the row-conflict cost within each channel.

Performance breakdown:
- HBM reads: 524,288 total requests across 32 channels = 16,384 per channel × ~40 cycles ≈ 655,360 cycles.
- DM writes: 65,536 requests per DMN at one per cycle, hidden under the read latency.

Total time: approximately 655,360 cycles + 500 startup ≈ 655,860 cycles.
FR-FCFS scheduling recovers some throughput, but the order-of-magnitude penalty remains.

### Example 5: Missing Stack Bit Pathology

The access pattern fails to interleave across HBM's stack dimension (address bit 8), routing all traffic to half the channels and halving effective bandwidth.

Arguments:
- 1 chip (8 DMNs)
- `axes![A = 8, B = 64, C = 8, D = 512]`
- `dtype = i8`
- Source: HBM, mapping `m![A, B, C, D]`
- Destination: DM, with cluster mapping `m![A / 4]`, slice mapping `m![A % 4, B]`, element mapping `m![C, D % 256]`
- Stream: cluster `m![A / 4]`, slice `m![A % 4]`, time `m![C, B % 32, B / 32]`, packet `m![D % 256]`

Generated sequencers per `(cluster_i, dmn_j)`:

```text
read = [
  C     -> 8   : 512,
  B%32  -> 32  : 4,096,
  B/32  -> 2   : 131,072,
  D%256 -> 256 : 1,
] : 256, HBM @ (i·2²⁰ + j·2¹⁸)

write = [
  C     -> 8   : 256,
  B%32  -> 32  : slice_stride,
  B/32  -> 2   : DMN_stride,
  D%256 -> 256 : 1,
] : 256, DM @ (cluster_i, dmn_j, 0)
```

The `C` stride of 512 never toggles HBM address bit 8 (the stack bit), so the eight DMNs concentrate on 16 of the 32 HBM channels.

Performance breakdown:
- HBM reads (bottleneck): 4,096 total requests across 16 channels = 256 per channel × ~5.3 cycles per request at 1 GHz ≈ 1,357 cycles.
- DM writes: 512 requests per DMN, pipelined under the reads.

Total time: approximately 1,357 cycles + 500 startup ≈ 1,857 cycles.
Restoring stack-bit interleaving across all 32 channels would halve the HBM cycle count.

### Example 6: Heterogeneous DMN Segmentation

When tensor shapes do not divide evenly across DMNs, the compiler segments the boundary DMN's work into multiple commands, each paying its own startup latency.

Arguments:
- 4 chips
- `axes![A = 15, B = 32, C = 256, D = 8]`
- `dtype = i8`
- Source: DM, with (writing `A' = A + 1#`) chip mapping `m![D / 2]`, cluster mapping `m![D % 2]`, slice mapping `m![A' / 4, A' / 2 % 2, B]`, element mapping `m![A' % 2, C]`
- Destination: HBM, with chip mapping `m![D / 2]`, element mapping `m![D % 2, B, A, C]`
- Stream (per-DMN, expressed as `StreamFn(chip_i, cluster_j, slice_k)`):

```text
StreamFn(chip_i, cluster_j, slice_k) = let A' = A + 1# in
  { chip: m![(D / 2) @ i = 1], cluster: m![(D % 2) @ j = 1],
    slice: m![(A' / 4) @ k = 1],
    time: (k == 0,1,2): m![A' % 2, B, A' / 2 % 2, C]
          (k == 3, exec #0): m![A' % 2, B, A' / 2 = 1, C]
          (k == 3, exec #1): m![A' = 1, B, A' / 2 % 2 @ 1, C],
    packet: m![C] }
```

The dimension `A = 15` does not divide across 4 DMNs (15 = 3·4 + 3), so DMNs 0 to 2 each handle 4 elements while DMN 3 handles only 3.
A single descriptor on DMN 3 would write a fourth element past the valid region, so the compiler segments DMN 3's work into two commands that together cover exactly 3 elements.

Performance breakdown:
- DMNs 0 to 2 (one command each): ~256 cycles + 500 startup ≈ 756 cycles.
- DMN 3 (two commands): ~192 data cycles + 1,000 startup (500 each) ≈ 1,192 cycles.

Total time: approximately 1,192 cycles, gated by DMN 3.
Choose tensor shapes that divide evenly across DMNs to avoid this segmentation cost.

## Shuffle Operations

Shuffle operations redistribute a tensor across clusters or chips according to a per-partition source pattern.
The methods chain off the source tensor, matching the `to_dm` / `to_hbm` convention: `dm_cluster_shuffle` and `dm_chip_shuffle` live on `DmTensorView`, while `hbm_cluster_shuffle` and `hbm_chip_shuffle` live on `HbmTensor`.
The shuffle pattern specifies, for each destination cluster or chip, which source cluster or chip provides its data.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 256, B = 4096];

fn cluster_shuffle(
    ctx: &mut Context,
    input: &DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]>,
) -> DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> {
    // Shuffle pattern [1, 0]: cluster 0 ↔ cluster 1
    input.view().dm_cluster_shuffle::<2>(&mut ctx.tdma, &[1, 0])
}
```

Inter-chip shuffles use the system-wide global chip IDs.
`hbm_chip_shuffle` is generic over the DMA context (`tdma` or `pdma`) because the cross-chip operation is HBM ↔ HBM, and HBM ↔ HBM is the one DMA pair that both Tensor DMA and PCIe DMA support.
The other shuffle methods, and DMA pairs like HBM ↔ DM and DM ↔ DM in general, are not context-generic.

## Scatter and Gather

Scatter and gather move tensor elements at addresses computed from an index tensor rather than at fixed strides.

`DmTensor::dma_scatter` writes DM values to HBM at indices given by an index tensor.
`HbmTensor::dma_gather` reads HBM rows into DM at indices given by an index tensor.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![K = 512, D = 128, C = 612];

fn scatter_minimal(
    ctx: &mut Context,
    data: &HbmTensor<bf16, m![1], m![K, D]>,
    index: &HbmTensor<i32, m![1], m![K]>,
    output: &mut HbmTensor<bf16, m![1], m![C, D]>,
) {
    let data_dm: DmTensor<bf16, m![1], m![1 # 2], m![K / 2], m![K % 2, D]> =
        data.to_dm(&mut ctx.tdma, 0x0);

    data_dm.dma_scatter::<m![K], _, _>(index, output, true);
}

fn gather_minimal(
    table: &HbmTensor<bf16, m![1], m![K, D]>,
    index: &HbmTensor<i32, m![1], m![C]>,
) -> DmTensor<bf16, m![1], m![1 # 2], m![C / 2], m![C % 2, D]> {
    table.dma_gather::<m![1 # 2], m![C / 2], m![C % 2, D], _>(index, 0x0, true)
}
```

`scaled` matches `dma_scatter`'s convention.
`true` treats index values as byte-offsets along the gather axis (divided internally to recover the row position).
`false` treats index values as raw row positions.

## PCIe DMA

PCIe DMA (`Context::pdma`) moves tensors between host system memory and device HBM.
It is a separate physical engine from the on-chip Tensor DMA.
PCIe DMA handles only host ↔ HBM, while Tensor DMA handles all on-chip transfers.

The kernel writer calls `.to_hbm()` on a `HostTensor` (host → device) or `.to_host()` on an `HbmTensor` (device → host).
Both are async operations.

```rust,ignore
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use rand::{rngs::SmallRng, SeedableRng};
axes![A = 8, B = 512];

async fn upload_and_download(ctx: &mut Context) {
    let mut rng = SmallRng::seed_from_u64(0);
    let host: HostTensor<i8, m![A, B]> = HostTensor::rand(&mut rng);

    // Host → HBM at address 0x1000
    let hbm: HbmTensor<i8, m![A], m![B]> = host.to_hbm(&mut ctx.pdma, 0x1000).await;

    // HBM → host (back to system memory)
    let _round_tripped: HostTensor<i8, m![A, B]> = hbm.to_host(&mut ctx.pdma).await;
}
```

`HostTensor` carries only an `Element` mapping (host memory has no chip/cluster/slice partitioning), while the destination `HbmTensor` adds the `Chip` axis to distribute across chips.
The destination element layout in HBM may differ from the host layout, since `to_hbm` accepts new `Chip` and `Element` type parameters.

PCIe DMA bandwidth is 30 B/cycle, an order of magnitude slower than on-chip Tensor DMA (256 B/cycle).
Algorithms should minimize host ↔ device traffic, uploading data once and reusing it across many on-chip operations.
