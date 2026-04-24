# Memory Performance

Memory performance fundamentally determines TCP program efficiency.
This page is the primary actionable reference for kernel writers: it documents hardware specifications, explains why each constraint exists, and maps API choices to their performance consequences.
It covers DM first (the tier kernel writers interact with most), then SPM and HBM.

In practice, most performance problems trace back to two root causes, each with multiple specific manifestations:

1. **Unnecessary hops**: routing data through an intermediate memory tier (e.g., through SPM when a direct HBM→DM transfer suffices) adds latency and bandwidth pressure.
2. **Low throughput**: a `Packet` that is smaller than necessary or non-contiguous in memory causes more sequencer iterations and strided access patterns.
   The following table details the hardware constraints that determine how `Packet` choices affect each memory type:

| Memory | Issue | Rule | Penalty |
|--------|-------|------|---------|
| DM | Bank starvation | < 64 consecutive same-bank accesses | NoC timeout → hardware reset |
| DM | DMN interleaving | Alternate across 2 DMNs per cluster | 50% bandwidth loss |
| DM | Slice interleaving | Spread across 32 slices per DMN | Command queue contention |
| HBM | Alignment | 256-byte aligned access | Unaligned read: 2× penalty; unaligned write: ~50× penalty (RMW) |
| HBM | Bank conflicts | Avoid row switches within same bank | 30–40× degradation |
| HBM | Channel interleaving | Spread across 32 channels | Reduced parallelism |


## Data Memory (DM)

Data Memory is the primary SRAM for tensor computations.
A single RNGD chip contains 256MB of DM, structured to maximize parallel access and bandwidth.
The following table summarizes the DM geometry:

| Unit | Count |
|------|-------|
| Clusters | 2 / Chip |
| Data Memory Networks (DMNs) | 8 / Cluster |
| Slices | 32 / DMN |
| Banks | 16 / Slice |
| Rows | 4096 / Bank |
| Bytes | 8 / Row |

The SRAM hierarchy consists of clusters, DMNs, and slices.
A single chip contains two clusters, each with eight Data Memory Networks.
Each DMN contains 32 slices, totaling 256 slices per cluster.
Clusters can exchange data through the Switch Engine; see [the dedicated section](../computing-tensors/switch-engine.md) for details.

### Address Space in a Slice

Each slice provides 512KB of SRAM with a dedicated address space.
The memory is organized into 16 parallel banks, each with an 8-byte data width, enabling a total data access rate of 128 B/cycle.
Access to any individual bank is serialized, but the address space distributes 128 consecutive bytes across all 16 banks (8 bytes per bank) for parallel access.
The following bit mapping defines this distribution:

| Bit # | Component        |
|-------|------------------|
| `0–2`   | Byte             |
| `3–6`   | Bank             |
| `7–18`  | Row              |

This bit mapping optimizes various access patterns, particularly during sequential access.
Distributing consecutive bytes across banks enables parallel access and maximizes bandwidth utilization.

### Optimizing DMA Performance

Achieving full DMA bandwidth requires following three guidelines: interleaving across DMNs, interleaving across slices, and preventing bank starvation.

**1. Interleave across DMNs.**

DMN interleaving is essential because each DMN provides only 128 B/cycle bandwidth.
Since the standard 256-byte transfer unit requires two cycles per DMN, you should pipeline accesses across both DMNs to maintain continuous throughput:

| cycle | DMN #0  | DMN #1 |
|-------|---------|--------|
|   0   | read #0 (1/2) |    (idle)   |
|   1   | read #0 (2/2) | read #1 (1/2) |
|   2   | read #2 (1/2) | read #1 (2/2) |
|   3   | read #2 (2/2) | read #3 (1/2) |
|  ...  |     ...      |     ...      |
|  2n-1 | read #2n-2 (2/2)| read #2n-1 (1/2)|
|  2n   |      (idle)       | read #2n-1 (2/2)|

**2. Interleave across slices.**

Slice interleaving improves efficiency by distributing DMA requests across the 32 slices within each DMN.
Slices are shared resources used by the DMA, Fetch, and Commit Engines, so spreading requests helps manage contention.
Each slice has two command queue entries to hold pending DMA requests.

**3. Prevent bank starvation.** Unlike the first two issues, bank starvation may force a complete chip reset, not just a slowdown.

### Bank Starvation

The key constraint is the **64-access rule**: Fetch and Commit engines must not access the same DM bank for more than 64 consecutive operations while DMA is active. Violating this causes a NoC timeout and full cluster reset.

The fundamental issue is priority inversion in a shared resource system: a low-priority requester is indefinitely blocked by high-priority ones accessing the same resource.
The DM controller prioritizes requests in this order:

- Main-context Fetch Engine
- Main-context Commit Engine
- Sub-context Fetch Engine
- Sub-context Commit Engine
- DMA Engine

DMA has the lowest priority among all memory engines, which makes sense during normal operation since computation engines should get first access to data.
However, this creates a dangerous scenario when high-priority engines continuously access the same bank: the DMA Engine's request sits in the queue, unable to make progress, while the higher-priority engines monopolize that bank.
After 4,096 cycles without a response, the NoC (Network on Chip) protocol declares the transaction dead and enters an exception state.
This 4,096-cycle limit exists as a safety mechanism to detect deadlocks and hung transactions in the NoC protocol; without this timeout, a stuck transaction could hang the entire system indefinitely.
When the timeout triggers, the hardware lacks a graceful recovery mechanism, and the only recovery is a full cluster domain reset, losing all computation state and requiring complete reinitialization.

The **64-access rule** prevents this catastrophe: the Fetch and Commit Engines must not access the same bank for more than 64 consecutive operations while DMA is active.
Why 64? The math comes from NoC bandwidth: `(256 B/cycle DMA ÷ 128 B/cycle per DMN) × 64 accesses × 32 slices < 4096 cycles`—this ensures DMA requests complete before the timeout even in the worst case.

For example, suppose the DMA Engine issues a request to bank 0 (along with 15 other banks), but the main-context's Fetch Engine continuously requests bank 0.
The DMA request stalls, and if this exceeds 4,096 cycles, a NoC timeout forces a hardware reset.

**Compiler scheduling behavior:** When Tensor Unit operations would violate the 64-access limit, the compiler schedules them as if they occupy DMA, preventing concurrent DMA operations.
This sacrifices the TCP architecture's inherent main/sub/DMA context parallelism where data preparation and computation occur in parallel, but avoids catastrophic hardware resets.
Treat this as a hard constraint: never use patterns with 64+ consecutive same-bank accesses.

Main-context starving sub-context is less severe because it does not trigger NoC timeouts and only increases processing time.
Additionally, the Tensor Unit's internal pipeline naturally generates back-pressure between Fetch and Commit Engines, preventing internal starvation.

**Scheduling model:** The scheduler uses context occupancy information: if operation A occupies a context (e.g., main context), the next operation B using that context waits until A completes.
Understanding which contexts operations occupy enables predicting parallel execution.
A scheduling visualization utility would help verify actual schedules.

**The 64-access limit details:**
- The limit is **cumulative across all concurrent commands**: if the total number of consecutive accesses from all commands (main-context fetch/commit + sub-context fetch/commit + DMA) to the same bank exceeds 64 cycles, DMA starvation occurs.
- Even if individual commands interleave accesses to the same bank, their combined access count still accumulates toward the 64-access limit, which can cause DMA Engine starvation.
- The compiler controls only single commands accessing the same bank consecutively; multiple commands interleaving the same bank are not controlled.
- In practice, sub-context rarely accesses the same bank consecutively (`StoTrf`, `StoVrf` operations typically use sequential addresses and tiling prevents same-bank access).
- Sub-context operations that would exceed the limit are also not scheduled concurrently with DMA.

> [!NOTE]
> **Cumulative Bank Access Constraint**
>
> Even if each individual command accesses a bank fewer than 64 times, the **TOTAL across all concurrent main/sub/DMA commands to the SAME bank must be less than 64**.
>
> The compiler prevents individual commands from exceeding this limit, but it **cannot prevent accumulation from multiple concurrent operations**. For example:
> - Main-context Fetch: 30 consecutive bank accesses
> - Sub-context Fetch: 20 consecutive bank accesses
> - DMA: 1 concurrent request to the same bank
> - **Total: 51 accesses** → Safe (below 64)
>
> But if either main or sub reaches 35+ accesses, the combined total exceeds 64 and triggers starvation. This is why the compiler sacrifices main/sub/DMA parallelism (scheduling them sequentially instead) when cumulative access patterns would exceed 64 to the same bank.

**Main/sub-context contention:** Main-context can starve sub-context, but this is less severe:
- Unlike DMA starvation, sub-context starvation does not cause NoC timeout or hardware reset and only increases processing time.
- Collision probability is lower: DMA Engine occupies 16 banks at once, while sub fetch/commit engines occupy only one bank.
- Starvation does not occur between fetch and commit engines within the same context due to pipeline back-pressure.

**Performance impact example:** If main-context exec command continuously accesses a specific bank while sub-context stos command is scheduled, sub-context processing is delayed.
Worst case: total time = main-context time + sub-context time.
Ideal case: main and sub access different banks, achieving total time = max(main-context time, sub-context time).

### Technical Details: Banks and Command Queues

**Bank access:** At 128 B/cycle DMN access, 16 banks are accessed simultaneously.
Banks are shared resources among Fetch Engine, Commit Engine, and DMA Engine.
Access to any individual bank is serialized.

**DMN bandwidth:** Within the Data Memory Network, Data Memory Slices share data paths, so DMA Engine transfers achieve 128 B/cycle per DMN.

**Command queues:** Each Data Memory Slice has a 2-entry command queue for pending DMA read/write requests.
Since this is limited, spreading DMA requests across multiple slices is ideal: distributing across M slices reduces required throughput to 1/M even if request processing slows due to priority.
DMN interleaving every n cycles achieves saturated 256 B/cycle.

> [!NOTE]
> While command queues theoretically allow some burst access without interleaving, we strongly recommend always interleaving across DMNs when generating DMA streams, as this is the most natural approach.

**The 4096-cycle limit derivation:** The formula is: `(TDMA_IO_BYTE / DMN_IO_BYTE) * Max_Consecutive_Access * DMN_SIZE < 4096`.
With `TDMA_IO_BYTE=256`, `DMN_IO_BYTE=128`, `DMN_SIZE=32`, this yields `Max_Consecutive_Access < 64`.

**DMN NoC architecture:** Tensor DMA connects to DRAM and DMN through a NoC acting as a hub.
Each port (DMA port, DRAM port, DMN ports) receives requests and must send responses.
Transactions are considered hung if response takes more than 4096 cycles after request.
When a DMN port request doesn't receive a response within 4096 cycles, the NoC treats it as an error and enters an exception state, requiring a cluster domain reset.

**Data Memory Network topology:** Data Memory Routers are DMN components connected in a ring topology forming the Data Memory Network.
The path is: slice0_in → slice31_out, slice32_in → slice63_out.

<!-- > **TODO**: Should we document the relationship between DMN and DMA terminology more clearly? The terms are used somewhat interchangeably in different contexts. -->


<!-- > **TODO**: Add a scheduling visualization utility reference. Programmers need to verify how their operations are actually scheduled—seeing which operations run in parallel would help optimize context usage. -->


## Scratchpad Memory (SPM)

> [!Note]
> This section is a work in progress; hardware-specific details (capacity, addressing, bank structure) are pending.

Scratchpad Memory provides additional fast storage within each DMN for temporary data and intermediate results.
Each DMN contains SPM with a bandwidth of 128 B/cycle, offering high-speed access for frequently reused values such as constants, lookup tables, or small working sets that don't require the full capacity of SRAM.

<!-- > **TODO** (jeongmin.park): Does SPM share the same 128 B/cycle DMN bandwidth as DM, or is it an independent path? Clarify whether SPM and DM compete for bandwidth. -->

SPM serves as a middle tier in the memory hierarchy between the ultra-fast VRF (Vector Register File) and the larger SRAM.
Its primary use cases include storing scalar constants, small weight matrices, activation function lookup tables, and configuration data that needs rapid access without consuming scarce VRF capacity.
The compiler automatically selects SPM for data that exhibits high temporal locality but modest capacity requirements.

The key distinction from SRAM is explicit software management: the compiler explicitly allocates data to SPM when beneficial, whereas SRAM allocation follows more general-purpose policies.
SPM's 128 B/cycle bandwidth per DMN enables high-throughput access for small tensors, and because each DMN has dedicated SPM, there are no inter-DMN contention issues.
SPM is particularly valuable for per-DMN state that would otherwise require repeated SRAM fetches.

<!-- > **TODO** (jeongmin.park): Please document the SPM (Scratchpad Memory) section with complete specifications. Details needed: exact capacity per DMN, addressing scheme (how software references SPM addresses), specific access patterns and latency characteristics, bank structure (if any), constraints on data types and alignment, and interaction with other memory systems. The above description provides the conceptual overview, but hardware-specific details are essential for programming. -->


## High-Bandwidth Memory (HBM)

HBM provides high-capacity off-chip storage with substantial bandwidth for large tensor operations.
A single RNGD chip contains 48GB of HBM.
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


### Address Space in a Chip

The HBM address space uses a non-linear bit mapping optimized for parallel sequential access.
This design maximizes parallelism and minimizes overhead rather than directly mapping to physical geometry:

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

1. **Slice representation:** Two bits (20 and 34) represent slice, even though there are only three slices.
2. **Contiguous address space:** Bit 34 is influenced by the row component to ensure bits 34 and 35 are never both 1, guaranteeing a contiguous 48GB address space.
3. **Channel XOR mapping:** The channel component equals the XOR of bits 9-12 and 13-28 (e.g., the channel's first bit equals the XOR of bits 9, 13, 21, and 25).

This unconventional bit order enhances performance by enabling parallelism across different memory resources.


### Peak Bandwidth

Peak HBM bandwidth reaches 1.5TB/s per chip through parallel operation of stacks and channels.
The channel controller transfers 64B/cycle at 0.75GHz,[^HBM] yielding 48GB/s per channel (0.75GHz x 64B/cycle) or 1.5TB/s per chip (48GB/s x 32 channels).
The fundamental transfer unit is 256 bytes, requiring 4 clock cycles per channel.
Saturating a single DMA Engine (256GB/s capacity) requires interleaving accesses across multiple channels.

[^HBM]: Although the channel controller operates at a frequency of 0.75GHz, it performs eight bursts per cycle, leading to an effective frequency of 0.75×8=6GHz.

Achieving peak bandwidth requires careful attention to access patterns.
Channel throughput is highly sensitive to misalignment, bank conflicts, and resource sharing.
Each channel controller has a 64-entry command queue that interleaves accesses to minimize penalties, but pathological cases can still cause severe degradation.
The following sections describe causes of performance degradation and how to avoid them.


### Misaligned Access

Misaligned access significantly degrades HBM performance.
Bits 0-7 (the eight LSBs) represent the 256-byte minimum access unit within a memory row.
Accessing data that crosses this boundary incurs substantial penalties.

**Unaligned Read:** Read requests crossing a 256-byte boundary require two NoC transfers, effectively halving bandwidth.

**Unaligned or Partial Write:** DMA packets are internally segmented into 256-byte transactions.
When a packet's size is not 256-byte aligned (e.g., a 2,800-byte packet splits into ten 256-byte requests plus one 240-byte request), the final "leftover" transaction requires a Read-Modify-Write (RMW) operation.
RMW reads the entire 256-byte unit, updates the requested bytes, then writes the entire unit back.
RMW can slow writes by roughly 50× compared to aligned writes.


### Bank Conflict

Bank conflicts cause severe performance degradation of 30–40× compared to accessing an already-open row.
They occur when consecutive accesses target different rows within the same bank.
Only one row per bank can be open at a time; all rows start closed.
Once open, a row's 256-byte words can be accessed quickly, but switching rows requires closing the current row and opening a new one, adding 40–50 ns (60–75 cycles at 1.5 GHz) of latency.

**Channel interleaving mitigates bank conflicts.**
Interleaving accesses across all 32 channels distributes load and reduces conflicts.
Bits 8-12 (the next five LSBs) represent independent stacks and channels; placing these at low addresses prevents interference between adjacent accesses, which is vital for parallelizing contiguous operations.
Non-contiguous operations often benefit from natural channel interleaving because the channel component spans bits 9-28.
However, the stack component only corresponds to bit 8, so interleaving this bit requires explicit attention.

**The controller hides row-switch latency through command interleaving.**
Within each channel, the controller automatically interleaves commands across banks, enabling useful transfers while other banks perform row switches.
The controller manages bank states using its command queue and employs FR-FCFS (First Ready-First Come First Served) scheduling, prioritizing commands targeting already-open rows.

Despite this sophisticated scheduling, access patterns that continuously switch rows within the same bank still degrade performance significantly.
Compilers and programmers should estimate row-switch costs when generating code.


### Column-to-Column Delay

Column-to-Column Delay (`tCCD`) rarely affects performance significantly, so skip this section on first reading.

`tCCD` is the minimum time between consecutive read or write commands on the same channel.
It determines the maximum command issue rate, directly affecting channel throughput.
Vendor specifications set `tCCD` values based on analog constraints for accessing `DRAM` stack layers and shared resources.

The `tCCD` value depends on which memory resources consecutive commands target:

| Command Relation | `tCCD` (cycles @ `1.5GHz`) | Relative Performance | Reason for Penalty |
|-----------|----------------------|------------|--------|
| Same Slice, Different Bank Group | `2` | `1` | Ideal interleaving of bank groups |
| Different Slice | `3` | `2/3` | Data path switching |
| Same Slice, Same Bank Group | `4` | `1/2` | Shared I/O buffer among four banks |

The optimal case is interleaving between different bank groups within the same slice (`tCCD = 2` cycles at `1.5GHz`), allowing a new `64B` command to be issued every cycle at `0.75GHz`, achieving back-to-back transmission and full channel speed.
Any `tCCD` greater than `2` reduces the command rate and channel utilization.

Pathological tCCD patterns cause less severe degradation than bank conflicts for two reasons: either they often coincide with bank conflicts anyway, or channel interleaving masks their impact:

- **Different Slice (`tCCD = 3`):** Slice ID corresponds to bit `20`, and bit `21` corresponds to the row. Interleaving across slices therefore likely causes bank conflicts simultaneously.
- **Same Slice, Same Bank Group (`tCCD = 4`):** This pattern interleaves bits `8-35` except bits `13`, `19`, `20`, and `34`. Bits `29-35` relate to bank conflicts; bits `8-28` relate to channel interleaving.
