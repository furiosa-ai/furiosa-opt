# Memory Performance

Memory performance is critical to kernel throughput.
The Fetch, Commit, and DMA Engines each expose API choices (such as `Packet` size and access ordering) that map directly to performance outcomes.
This page documents the hardware specifications and constraint rationale that connect those choices to measured throughput.

Each memory type has a peak bandwidth per chip:

| Memory | Peak Bandwidth |
|--------|---------------|
| DM | 2 TB/s per chip |
| SPM | 2 TB/s per chip |
| HBM | 1.5 TB/s per chip |

Reaching these peaks requires specific access patterns.
The following table lists rules whose violation degrades throughput, and the sections below explain each memory type and factor in detail:

| Memory | Issue | Rule | Penalty |
|--------|-------|------|---------|
| DM | Bank starvation | < 64 consecutive same-bank accesses | NoC timeout → hardware reset |
| DM | DMN interleaving | Alternate across 2 DMNs per cluster | 50% bandwidth loss |
| DM | Slice interleaving | Spread across 32 slices per DMN | Command queue contention |
| HBM | Alignment | 256-byte aligned access | Unaligned read: 2× penalty; unaligned write: ~50× penalty (RMW) |
| HBM | Bank conflicts | Avoid row switches within same bank | 30–40× degradation |
| HBM | Channel interleaving | Spread across 32 channels | Reduced parallelism |


## Data Memory (DM)

Data Memory (DM) holds 256MB per chip, organized hierarchically into clusters, Data Memory Networks (DMNs), slices, and banks.
The following table summarizes the geometry:

| Unit | Count |
|------|-------|
| Clusters | 2 / Chip |
| Data Memory Networks (DMNs) | 8 / Cluster |
| Slices | 32 / DMN |
| Banks | 16 / Slice |
| Rows | 4096 / Bank |
| Bytes | 8 / Row |

Clusters can exchange data through the Switch Engine.
See [the dedicated section](../computing-tensors/switch-engine.md) for details.
The subsections below explain how this structure determines bandwidth and the bank access constraint.

### Bank Structure in a Slice

Each slice provides 512KB of SRAM with a dedicated address space.
The memory is organized into 16 parallel banks, each with an 8-byte data width, enabling a total data access rate of 128 B/cycle.
Access to any individual bank is serialized, but the address space distributes 128 consecutive bytes across all 16 banks (8 bytes per bank) for parallel access.
The following bit mapping defines this distribution:

| Bit # | Component        |
|-------|------------------|
| `0–2`   | Byte             |
| `3–6`   | Bank             |
| `7–18`  | Row              |

Consecutive addresses map to different banks, enabling parallel access during sequential scans.

### DMN and Slice Interleaving

Each DMN provides only 128 B/cycle bandwidth (its 32 slices share data paths).
Since the standard 256-byte transfer unit requires two cycles per DMN, pipeline accesses across both DMNs to maintain continuous throughput:

  | cycle | DMN #0  | DMN #1 |
  |-------|---------|--------|
  |   0   | read #0 (1/2) |    (idle)   |
  |   1   | read #0 (2/2) | read #1 (1/2) |
  |   2   | read #2 (1/2) | read #1 (2/2) |
  |   3   | read #2 (2/2) | read #3 (1/2) |
  |  ...  |     ...      |     ...      |
  |  2n-1 | read #2n-2 (2/2)| read #2n-1 (1/2)|
  |  2n   |      (idle)       | read #2n-1 (2/2)|

> [!NOTE]
> While command queues theoretically allow some burst access without interleaving, always interleave across DMNs when generating DMA streams, as this is the most natural approach.

Slices are shared by the DMA, Fetch, and Commit Engines, so spreading requests across the 32 slices within each DMN reduces contention.
Data Memory Routers connect those slices in a ring topology within each DMN: slice0_in → slice31_out, slice32_in → slice63_out.
Each Data Memory Slice has a 2-entry command queue for pending DMA requests.
Distributing requests across M slices reduces required throughput per slice to 1/M, even when priority delays individual slices.
DMN interleaving every n cycles achieves saturated 256 B/cycle.

### Bank Starvation

Bank starvation occurs when the DMA Engine is indefinitely blocked waiting for a DM bank held by higher-priority engines.
The 64-access rule prevents this.
Violating this rule causes a Network-on-Chip (NoC) timeout and a full cluster reset, losing all computation state.

Each DM bank is a shared resource.
When high-priority engines continuously access it, lower-priority requesters are indefinitely blocked, a form of priority inversion.
The DM controller prioritizes requests in this order:

- Main-context Fetch Engine
- Main-context Commit Engine
- Sub-context Fetch Engine
- Sub-context Commit Engine
- DMA Engine

DMA has the lowest priority among all memory engines because computation engines must get first access to data during normal operation.
However, this creates a dangerous scenario when high-priority engines continuously access the same bank: the DMA Engine's request sits in the queue, unable to make progress, while the higher-priority engines monopolize that bank.
Tensor DMA communicates with DRAM and DMN through a NoC hub where each port (DMA, DRAM, DMN) must acknowledge requests within 4,096 cycles.
After 4,096 cycles without a response, the NoC protocol declares the transaction dead and enters an exception state as a safety mechanism to detect deadlocks and indefinitely hung transactions.
When the timeout triggers, the hardware lacks a graceful recovery mechanism.
The only recovery is a full cluster domain reset, losing all computation state and requiring complete reinitialization.

The 64-access rule prevents this catastrophe: the Fetch and Commit Engines must not access the same bank for 64 or more consecutive operations while DMA is active.
Why 64?
The constraint is `(TDMA_IO_BYTE / DMN_IO_BYTE) * Max_Consecutive_Access * DMN_SIZE < 4096` (with `TDMA_IO_BYTE = 256`, `DMN_IO_BYTE = 128`, `DMN_SIZE = 32`), which yields `Max_Consecutive_Access < 64`.
This ensures DMA requests complete before the NoC timeout even in the worst case.


For example, suppose the DMA Engine issues a request to bank 0 (along with 15 other banks), but the main-context's Fetch Engine continuously requests bank 0.
The DMA request stalls, and if this exceeds 4,096 cycles, a NoC timeout forces a hardware reset.

- **Scheduling model**: The scheduler uses context occupancy information: if operation A occupies a context (e.g., main-context), the next operation B using that context waits until A completes.
  Understanding which contexts operations occupy enables predicting parallel execution.
- **Compiler scheduling behavior**: When Tensor Unit operations would violate the 64-access limit, the compiler schedules them as if they occupy DMA, preventing concurrent DMA operations.
  This sacrifices the TCP architecture's inherent main/sub/DMA context parallelism where data preparation and computation occur in parallel, but avoids catastrophic hardware resets.
  Treat this as a hard constraint: never use patterns with 64+ consecutive same-bank accesses.
- **The 64-access limit details**:
  - The limit is cumulative: total accesses from all engines to the same bank must stay below 64, since even interleaved accesses across commands accumulate toward this total.
    For example, main: 30, sub: 20, DMA: 1 totals 51 (safe), but main: 30, sub: 35, DMA: 1 totals 66 (triggers starvation).
  - The compiler keeps each individual command below 64 consecutive same-bank accesses, but cannot prevent the total from reaching 64 when multiple commands run concurrently.
  - In practice, sub-context rarely accesses the same bank consecutively (`StoTrf`, `StoVrf` operations typically use sequential addresses and tiling prevents same-bank access).
  - Sub-context operations that would exceed the limit are also not scheduled concurrently with DMA.
- **Main/sub-context contention**: Main-context can starve sub-context, but this is less severe:
  - Unlike DMA starvation, sub-context starvation does not cause NoC timeout or hardware reset and only increases processing time.
  - Collision probability is lower: DMA Engine occupies 16 banks at once, while sub fetch/commit engines occupy only one bank.
  - Starvation does not occur between fetch and commit engines within the same context due to pipeline back-pressure.
  - **Performance impact example**: If main-context exec command continuously accesses a specific bank while sub-context stos command is scheduled, sub-context processing is delayed.
    Worst case: total time = main-context time + sub-context time.
    Ideal case: main and sub access different banks, achieving total time = max(main-context time, sub-context time).

See [Schedule Viewer](../introduction.md#schedule-viewer) for a scheduling visualization utility that shows which operations run in parallel and verifies actual context assignments.


## Scratchpad Memory (SPM)


DM and SPM are both on-chip SRAM.
They are distinguished by *intended use* (and corresponding compiler allocation policy) rather than by any documented latency or capacity difference.

**DM (Data Memory)** is the main working memory for tensor data flowing through the Tensor Unit pipeline.
The DMA Engine populates DM from HBM (or other tiers), the Fetch Engine streams data from DM into the Tensor Unit, and the Commit Engine writes the pipeline's results back to DM.
DM allocation follows general-purpose policies driven by the program's tensor lifetimes.

**SPM (Scratchpad Memory)** is a compiler-managed staging tier.
The compiler explicitly chooses what lives in SPM, reserving it for small, frequently reused values that should not have to be refetched from DM on every access:

- scalar constants and configuration data,
- activation function lookup tables,
- small per-DMN working sets that are read many times.

SPM is most useful for per-DMN state that would otherwise force repeated DM reads through the Fetch or DMA Engines.

DM and SPM are both distinct from the per-slice **register files** that feed the Tensor Unit's compute engines directly (see [Computing Tensors](../computing-tensors/index.md)):

- **TRF (Tensor Register File)** is a per-slice register file populated via [`.to_trf()`](../computing-tensors/collect-engine.md#to-trf); the Contraction Engine reads it each cycle.
- **VRF (Vector Register File)** is a per-slice register file populated via [`.to_vrf()`](../computing-tensors/collect-engine.md#to-vrf); the Vector Engine reads it each cycle.

The pipeline's data flow is therefore: DMA populates **DM** from HBM → Fetch streams data from DM into the Tensor Unit → Collect writes the stream into **TRF / VRF** → Contraction / Vector read directly from TRF / VRF → Commit writes results back into DM.
**SPM** sits to the side as a compiler-controlled staging area for the small, high-locality data the kernel needs but does not want to refetch.

Each DMN contains SPM with a bandwidth of 128 B/cycle, and because each DMN has dedicated SPM there are no inter-DMN contention issues.


## High-Bandwidth Memory (HBM)

HBM holds 48GB per chip and delivers 1.5 TB/s aggregate bandwidth, but reaching that peak requires 256-byte-aligned access and channel interleaving across all 32 channels.
Misaligned writes and bank conflicts can degrade throughput by 30–50×.
The following table summarizes the HBM geometry:

| Unit | Count |
|------|-------|
| Stacks | 2 / Chip |
| Channels | 16 / Stack |
| Slices | 3 / Channel |
| Bank Groups | 4 / Slice |
| Banks | 4 / Bank Group |
| Rows | 16K / Bank |
| Bytes | 2K / Row |


### Peak Bandwidth

Saturating a single DMA Engine (256GB/s capacity) requires interleaving accesses across multiple channels.
Peak HBM bandwidth reaches 1.5TB/s per chip through parallel operation of stacks and channels.
The channel controller transfers 64B/cycle at 0.75GHz,[^HBM] yielding 48GB/s per channel (0.75GHz x 64B/cycle) or 1.5TB/s per chip (48GB/s x 32 channels).
The fundamental transfer unit is 256 bytes, requiring 4 clock cycles per channel.

[^HBM]: Although the channel controller operates at a frequency of 0.75GHz, it performs eight bursts per cycle, leading to an effective frequency of 0.75×8=6GHz.

Peak bandwidth is sensitive to access patterns.
Misalignment, bank conflicts, and resource sharing can each severely degrade throughput.
Each channel controller has a 64-entry command queue that interleaves accesses to minimize penalties, but pathological cases can still cause severe degradation.
The following sections describe causes of performance degradation and how to avoid them.


### Address Space in a Chip

The HBM address space uses a non-linear bit mapping optimized for parallel sequential access.
This design maximizes parallelism and minimizes overhead:

| Bit # | Main Component   | Additional Components |
|-------|------------------|--------|
| 0–7   | Byte             |        |
| 8     | Stack            |        |
| 9–12  | Channel |        |
| 13    | Bank Group | Channel |
| 14–16 | Byte | Channel           |
| 17–18 | Bank | Channel           |
| 19    | Bank Group | Channel     |
| 20    | Slice | Channel           |
| 21–33 | Row | Channel (21–28)         |
| 34    | Slice     | Row |
| 35    | Row              | |

The bit assignment for each component corresponds to the physical memory geometry.
For instance, the byte component occupies 11 bits (bits 0-7, 14-16) to represent 2K (2^11) bytes per row.
Three exceptions exist:

- **Slice representation**: Two bits (20 and 34) represent slice, even though there are only three slices.
- **Contiguous address space**: Bit 34 is influenced by the row component to ensure bits 34 and 35 are never both 1, guaranteeing a contiguous 48GB address space.
- **Channel XOR mapping**: The channel component equals the XOR of bits 9-12 and 13-28 (e.g., the channel's first bit equals the XOR of bits 9, 13, 21, and 25).

This bit ordering ensures that sequential accesses are spread across stacks, channels, bank groups, and banks simultaneously, keeping multiple memory resources busy in parallel.


### Misaligned Access

Misaligned access degrades HBM performance substantially.
Reads crossing a 256-byte boundary require two transfers (2x penalty), and unaligned writes require a Read-Modify-Write (RMW) operation (roughly 50x penalty).
The 256-byte minimum access unit is defined by bits 0-7 (the eight LSBs), so data that crosses this boundary incurs these penalties.

- **Unaligned Read**: Read requests crossing a 256-byte boundary require two NoC transfers, effectively halving bandwidth.
- **Unaligned or Partial Write**: An unaligned write arises because DMA packets are internally segmented into 256-byte transactions.
  When a packet's size is not 256-byte aligned (e.g., a 2,800-byte packet splits into ten 256-byte requests plus one 240-byte request), the final "leftover" transaction requires an RMW operation.
  RMW reads the entire 256-byte unit, updates the requested bytes, then writes the entire unit back.
  RMW can slow writes by roughly 50× compared to aligned writes.


### Bank Conflict

HBM banks hold one open row at a time.
Switching to a different row within the same bank requires closing the current row and opening the new one.
This adds 40-50 ns (60-75 cycles at 1.5 GHz) of latency, which is 30-40x slower than accessing an already-open row.
This penalty occurs whenever consecutive accesses target different rows within the same bank.
All rows start closed, so the first access to any row always pays the open-row cost.

Channel interleaving mitigates bank conflicts.
Interleaving accesses across all 32 channels distributes load and reduces conflicts.
Bits 8-12 (the next five LSBs) represent independent stacks and channels.
Placing these at low addresses prevents interference between adjacent accesses, which is vital for parallelizing contiguous operations.
Non-contiguous operations often benefit from natural channel interleaving because the channel component spans bits 9-28.
However, the stack component corresponds only to bit 8, so the programmer must explicitly ensure accesses alternate between the two stacks to achieve full stack interleaving.

The controller hides row-switch latency through command interleaving.
Within each channel, the controller automatically interleaves commands across banks, enabling useful transfers while other banks perform row switches.
The controller manages bank states using its command queue.
It employs FR-FCFS (First Ready-First Come First Served) scheduling, prioritizing commands targeting already-open rows.

Despite this sophisticated scheduling, access patterns that continuously switch rows within the same bank still degrade performance significantly.
Compilers and programmers should estimate row-switch costs when generating code.


### Column-to-Column Delay

`tCCD` (Column-to-Column Delay) is the minimum time between consecutive read or write commands on the same channel, which determines the maximum command issue rate.
In most access patterns, bank conflicts or channel interleaving dominate before `tCCD` becomes the bottleneck.
Vendor specifications set `tCCD` values based on analog constraints for accessing `DRAM` stack layers and shared resources.

The `tCCD` value depends on which memory resources consecutive commands target:

| Command Relation | `tCCD` (cycles @ `1.5GHz`) | Relative Performance | Reason for Penalty |
|-----------|----------------------|------------|--------|
| Same Slice, Different Bank Group | `2` | `1` | Ideal interleaving of bank groups |
| Different Slice | `3` | `2/3` | Data path switching |
| Same Slice, Same Bank Group | `4` | `1/2` | Shared I/O buffer among four banks |

The optimal case is interleaving between different bank groups within the same slice (`tCCD = 2` cycles at `1.5GHz`), allowing a new `64B` command to be issued every cycle at `0.75GHz`, achieving back-to-back transmission and full channel speed.
Any `tCCD` greater than `2` reduces the command rate and channel utilization.

Compared to bank conflicts, `tCCD` degradation is less severe because the worst-case patterns either coincide with bank conflicts (making `tCCD` the secondary effect) or are masked by channel interleaving:

- Different Slice (`tCCD = 3`): Slice ID corresponds to bit `20`, and bit `21` corresponds to the row.
  Interleaving across slices therefore likely causes bank conflicts simultaneously.
- Same Slice, Same Bank Group (`tCCD = 4`): This pattern interleaves bits `8-35` except bits `13`, `19`, `20`, and `34`.
  Bits `29-35` relate to bank conflicts.
  Bits `8-28` relate to channel interleaving.
