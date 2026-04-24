# DMA Engine

The DMA Engine moves tensors directly between memory locations without involving the Tensor Unit.
It supports all combinations of HBM, SPM, and DM transfers while optionally transforming memory layouts.

As a kernel writer, you control the source and destination memory tiers and any layout transformation expressed as mapping expressions.
Prefer direct transfers between tiers: routing data through an intermediate tier (e.g., HBM→SPM→DM when HBM→DM suffices) adds unnecessary latency and bandwidth pressure.
The compiler derives the read/write sequencer configuration.

This page covers the interface, worked examples, architecture, and performance characteristics.

## Interface

```rust,ignore
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
/// Moves a tensor from one memory location to another using DMA.
/// Supports layout transformations during transfer.
fn dma<D: Scalar, InMedia, OutMedia, InMapping, OutMapping, StreamMapping>(
    input: &Tensor<D, InMedia, InMapping>,
    output: &mut Tensor<D, OutMedia, OutMapping>,
    stream: StreamMapping,
) {
    // Hardware implementation:
    // - Read sequencer fetches from source memory
    // - Write sequencer stores to destination memory
    // - Stream mapping coordinates the transfer
}
```

The operation signature follows this pattern:
```rust,ignore
{{#include ../../../furiosa-visa-std/src/memory_tensor.rs:dma_impl}}
```

Transfer capabilities:
- All nine source-destination pairs between DM, SPM, and HBM (including same-tier copies)
- Cross-DMN, cross-cluster, and cross-chip transfers
- Inter-chip transfers via PCIe at 30 bytes/cycle

See also: [Memory Performance](./memory-performance.md), [Sequencer](./sequencer.md).

## Examples

### Layout Transformation

Consider transposing a tensor's layout while moving it from HBM to DM:

```rust,ignore
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![N = 4, C = 3, H = 8, W = 8];

// Tensor in HBM with NCHW layout
let hbm: HbmTensor<i8, m![1], m![N, C, H, W]> = /* ... */;

// DMA Engine moves to DM with NHWC layout
let dm: DmTensor<i8, m![1], m![1], m![1], m![N, H, W, C]> =
    dma_engine(&hbm);
```

The DMA Engine reads from HBM using one access pattern and writes to DM using a different pattern, transforming the layout during transfer.
For parameter definitions, see the [Architecture](#architecture) section below.

## Architecture

The DMA Engine coordinates paired read and write sequencers for flexible tensor movement.
Each RNGD chip contains eight DMA Engines, one per pair of DMNs, so up to eight independent tensor transfers can proceed simultaneously.

### Single-Engine Operation

A single DMA Engine operation transforms a tensor by reading it from one memory location and writing it to another with a potentially different layout.

### Parameters

The DMA operation requires several parameters to specify the source tensor, destination tensor, and how data flows between them:

- `shape`: The tensor's logical shape (declared via `axes![...]`)
- `dtype`: Element datatype (e.g., `i8`, `bf16`)
- `media_in`, `media_out`: Source and destination media types (DM/SPM/HBM)
- `b_in`, `b_out`: Base memory addresses for input/output tensors (when media is HBM, `b = { element: b_element }`)
- `In`, `Out`: Mapping environments that specify how logical tensor indices map to physical memory locations
- `Stream`: Intermediate stream mapping environment that coordinates the read and write sequencers

The operation executes using two coordinated sequencers:
The read sequencer applies `read(shape, dtype, b_in, In, Stream)` to fetch data from the source, while the write sequencer applies `write(shape, dtype, b_out, Out, Stream)` to store data at the destination.
These sequencers work together through the shared `Stream` environment to ensure data flows correctly from source to destination.

### Alignment Constraints

These constraints reflect the physical organization of memory hardware and the AXI bus protocol.
**The 8-byte DM write alignment** stems from SRAM bank structure: each bank has an 8-byte data width, and the bank controller can only write complete 8-byte units.
Misaligned writes require a read-modify-write operation, tripling the time and blocking other operations on that bank.
**The 1-byte read alignment** reflects asymmetric hardware capabilities: SRAM read ports can extract arbitrary byte ranges using byte-select logic, but write ports cannot.
**HBM-to-DM 8-byte alignment** combines both constraints: unaligned HBM reads incur severe performance penalties (potentially halving bandwidth), so the hardware enforces alignment for this critical path.
**The 4096-byte packet limit** comes from the AXI bus protocol: AXI transactions cannot exceed 256 beats, and with 16-byte data width this yields 4096 bytes maximum.
Violating these constraints causes correctness errors or hardware exceptions, not just performance degradation.
The compiler enforces these rules because they are hardware invariants, not optimization hints.

### Structural Requirements

The mapping environments must follow specific structural requirements depending on the media types involved:

- `Stream` must have a specific form:
  ```text
  // Stream = { time: Time, packet: Packet }
  ```

- `In/out` must have a specific form depending on the media `media_in/out`:
  ```text
  // In =
  //   if media_in in {HBM, SPM}: { element: ElementIn }
  //   if media_in in {DM}: { slice: SliceIn, element: ElementIn }
  //
  // Out =
  //   if media_out in {HBM, SPM}: { element: ElementOut }
  //   if media_out in {DM}: { slice: SliceOut, element: ElementOut }
  ```
  This specifies the respective memory space.

- `b_in/out` must have a specific form depending on the media `media_in/out`:
  ```text
  // b_in =
  //   if media_in in {HBM, SPM}: { chip: b_chip_in, element: b_element_in }
  //   if media_in in {DM}: { chip: b_chip_in, cluster: b_cluster_in, slice: b_sliceIn, element: b_element_in }
  //
  // b_out =
  //   if media_out in {HBM, SPM}: { chip: b_chip_out, element: b_element_out }
  //   if media_out in {DM}: { chip: b_chip_out, cluster: b_cluster_out, slice: b_sliceOut, element: b_element_out }
  ```
  This specifies addresses in the respective memory space.

- RNGD imposes the following hardware constraints on DMA Engine sequencers (see [sequencer constraints](./sequencer.md#constraints) for details):
  + Alignment requirements for addresses and packet size (`Packet::SIZE`):

    |       | HBM | DM (SRAM) |
    |-------|-----|------|
    | Read address | `1B` | `1B` |
    | Write address | `1B` | `8B` |
    | packet size | `1B` | `8B` |

    In addition, HBM-to-DM DMA transfers require an `8`-byte alignment for the read address, write address, and packet size, regardless of the values shown in the table above.

  + The packet size must be less than or equal to `4096` bytes (AXI protocol constraint).

### Example: Basic HBM-to-HBM Layout Transformation

This example demonstrates how a DMA operation transforms a tensor's memory layout through a simple HBM-to-HBM transfer that rearranges tensor dimensions.

Consider a DMA operation with the following arguments:

```text
axes![N = 4, C = 3, H = 8, W = 8];
// dtype = i8
// media_in = media_out = HBM
// b_in = { chip: 0, element: 1024 }, b_out = { chip: 0, element: 2048 }
// In = { element: m![N, C, H, W] }, Out = { element: m![H, C, N, W] }
// Stream = { time: m![H, C, N], packet: m![W] }
```

The compiler generates the following sequencer configurations from these arguments:

<!-- > **TODO** (jeongmin.park): The axes above declare `N=4`, but the configs below say `N=3`. Also verify the write sequencer: the `H` stride (192) and `C` size (8) look inconsistent with `axes![N=4, C=3, H=8, W=8]`. Fix. -->

- Read sequencer configuration: `[H=8:8, C=3:64, N=3:192, W=8:1]:8  HBM/D@1024`
- Write sequencer configuration: `[H=8:192, C=8:32, N=3:8, W=8:1]:8  HBM/D@2048`

The hardware traverses memory locations according to these sequencer configurations.
The following pseudocode models this behavior conceptually:
```rust
fn dma_sequencer() {
    let packet_size = 8; // packet size divides last consecutive read/write sequencer configuration entry

    for h in 0..8 {
        for c in 0..3 {
            for n in 0..4 {
                for w_packet in 0..1 {
                    // packet size is 8, so W=8 is accessed as a single chunk
                    let read_index = h * 8 + c * 64 + n * 192 + w_packet * 1;
                    let stream = Mem[read_index..(read_index + packet_size)];
                    let writ_index = h * 96 + c * 32 + n * 8 + w_packet * 1;
                    Mem[writ_index..(writ_index + packet_size)] = stream;
                }
            }
        }
    }
}
```

This example illustrates how the stream environment (`Stream`) mediates between different input and output layouts (`In` and `Out`), transforming the tensor's organization in memory while moving it.

### Performance

Optimal DMA performance requires attention to startup overhead, alignment, and packet size:

**Startup overhead:** Each DMA operation incurs approximately 500 cycles of initial overhead. Combining multiple transfers into fewer operations improves efficiency.

**Alignment:** While the constraints above specify minimum requirements, using larger alignment factors (particularly 256-byte alignment) yields better throughput. For detailed guidance, refer to the [memory performance section](./memory-performance.md).

**Packet size and internal DMA requests:** DMA automatically splits packets into 256-byte units internally: an n-byte packet becomes ceil(n / 256) DMA requests. Examples:
- If the innermost entry is `x=4095:1`, packet size 4095 results in 16 DMA requests.
- If the innermost entry is `x=4099:1`, since 4099 is prime, a single DmaCommand processes 1 byte at a time (very inefficient). Split into two DmaCommands (e.g., 4096/3 portions) instead, though each additional DmaCommand adds ~500 cycles of initial latency.

### Homogeneous Aggregate Operation

Multiple DMA Engines work together in parallel to improve throughput for large tensor moves.
The homogeneous aggregate operation distributes a single logical tensor move across DMA Engines in multiple DMNs, with all DMNs using identical stream environments to coordinate their work.
With four chips, up to 32 DMA Engines execute portions of a single tensor move concurrently.

The operation has the following form:

```text
// dma(shape, dtype, media_in, media_out, b_in, b_out, In, Stream, Out)
```

Each participating DMN executes its own DMA Engine to handle a portion of the overall transfer, together implementing the following single logical tensor move:

```text
// <shape, In, media_in / dtype @ { element: b_in }> --id--> <shape, Out, media_out / dtype @ { element: b_out }>
```

Parallel execution across multiple DMNs requires extending the mapping environments beyond the single-DMN case to include chip, cluster, and slice dimensions:

```text
// In =
//   if media_in in {HBM, SPM}: { chip: ChipIn, element: ElementIn }
//   if media_in in {DM}: { chip: ChipIn, cluster: ClusterIn, slice: SliceIn,
//                          element: ElementIn }
//
// Out =
//   if media_out in {HBM, SPM}: { chip: ChipOut, element: ElementOut }
//   if media_out in {DM}: { chip: ChipOut, cluster: ClusterOut, slice: SliceOut,
//                           element: ElementOut }
```

The key characteristic of homogeneous operations is that all DMNs share the same parametric stream environment:

```text
// Stream = { chip: ChipStream, cluster: ClusterStream, slice: SliceStream,
//              time: Time, packet: Packet }
```

### Heterogeneous Aggregate Operation

The heterogeneous aggregate operation provides flexibility for different DMNs to process data differently during a parallel transfer.
This variant allows each DMN to use a distinct stream environment while coordinating to perform a single logical tensor move.

Two constraints maintain correctness with this added flexibility:
- All participating DMA Engines must use the same input and output media types
- A single, unified input and output tensor mapping expression must govern the overall transfer

The heterogeneous aggregate DMA operation is defined as:

```text
// dma(shape, dtype, media_in, media_out, b_in, b_out, In, StreamFn, Out)
```

Each DMN executes its own DMA Engine to implement the following single logical tensor move:

```text
// <shape, In, media_in / dtype @ { element: b_in }> --id--> <shape, Out, media_out / dtype @ { element: b_out }>
```

The stream environment specification distinguishes this operation.
Instead of a single parametric stream environment shared by all DMNs, the heterogeneous operation uses `StreamFn`, a function mapping each DMN's location to its own unique stream environment.
For a DMN at chip `i`, cluster `j`, and slice index `k`, the function `StreamFn(i, j, k)` returns that DMN's specific stream mapping of the form `{ time: Time, packet: Packet }`.
The input and output mapping environments (`In` and `Out`) remain structurally identical to the homogeneous case, ensuring a well-defined overall logical tensor move.

### DMA Command Syntax

Two syntactic forms express DMA operations, depending on whether each DMN needs its own descriptor or can share a common pattern.

#### Heterogeneous Syntax (Full Flexibility)

The heterogeneous syntax specifies a complete DMA descriptor for each DMN individually, including potentially different source and destination media:

```bnf
<DMACommand> ::= HashMap(<DmnIndex>, <DmaDescriptor>)
<DmaDescriptor> ::= (<DmaSequencer>, <source_media: Media>, <dest_media: Media>)
<DmaSequencer> ::= (<limit: integer>, <source_stride: integer>, <dest_stride: integer>)*,
                   (<source_base: integer>, <dest_base: integer>), <stride0: 1~4096>
<Media> ::= "HBM"(<ChipIndex>) | "DM"(DmnIndex) | "SPM"(DmnIndex)
<DmnIndex> ::= (<ChipIndex>, <ClusterInChipIndex>, <SliceInClusterIndex>)
<ChipIndex> ::= 0 | 1 | 2 | 3 (when using 4 chips)
<ClusterInChipIndex> ::= 0 | 1
<SliceInClusterIndex> ::= 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7
```

Note: While a DMA operation logically uses separate read and write sequencers, the compiler represents them compactly as a single DmaSequencer with paired strides and bases per entry (one for source, one for destination).

#### Homogeneous Syntax (Common Case)

For the common case where all DMNs follow a regular pattern, the homogeneous syntax offers a concise representation:

```bnf
<DMACommand> ::= ( <source: Tensor>, <dest: Tensor>, HashMap(<DmnIndex>, <StreamShape>) )
<Tensor> ::= ( <Shape>, <Memory Mapping Expression>, <Media>, <addr: integer>, <Dtype> )
<Shape>, <Memory Mapping Expression>: defined before
<Media> ::= "HBM" | "DM" | "SPM"
<Dtype> ::= i4 | i8 | f8e4m3 | f8e5m2 | i16 | fp16 | bf16 | i32 | f32
<StreamShape> ::= <Memory Mapping Expression>
<DmnIndex> ::= (<ChipIndex>, <ClusterInChipIndex>, <SliceInClusterIndex>)
```

**Key usage notes:**
- DM tensor specifications must include chip, cluster, and slice dimensions in the Memory Mapping Expression to identify the exact memory location
- Each DMN's `StreamShape` includes inter-DMN mapping information (e.g., `chip: A!4, chip #2` means the stream shape uses `A@2!1` to specify reading from a particular chip)
- Stream shapes are often inferred: if only source and destination tensors are provided, the compiler derives appropriate stream shapes. Alternatively, specify a single stream shape with chip/cluster/slice dimensions, from which per-DMN stream shapes are automatically derived

**Example of heterogeneous mapping:**

```text
axes![A = 4, B = 256, C = 256, D = 256];
// source: [Chip: [A % 4], Dram: [B % 256 * C % 256 * D % 256]], HBM @ 0
// dest: [Chip: [A % 4], Cluster: [B / 128], Partitioning: [C % 256], InSlice: [B % 128 * D % 256]], DM @ 0
```

StreamShape for (Chip_i, Cluster_j, Slice_k):
```text
// [A @ i % 1 * (B / 128) @ j % 1 * (C / 64) @ k % 1 * C % 32 * B % 128 * C / 32 % 2 (DMN) * D % 256]
// DMA Sequencer = [C=32:(32 * 256, slice_stride), B=128:(256 * 256, 256),
//                  C/32=2:(32 * 256, 32) * slice_stride, D=256:(1, 1)],
//                  base: (Chip, Cluster, Slice, HBM/InSlice) = ((i, i), (j, j), (k, k), (0, 0))
// (slice_stride = 4MB = virtual address space of in_slice DM)
```

### Implementation Details

This section explains how the compiler generates DMA operations and how the hardware executes them.

**Compiler generates aggregate operations by default:** The compiler treats tensor-to-tensor moves (T → T') as atomic units and automatically distributes work across available DMNs in parallel, similar to Fetch/Commit Sequencers. Aggregate operations are the primary abstraction programmers interact with, which explains why this documentation emphasizes them rather than single-DMN DMA.

**Sequencer representation is compact:** Although DMA operations logically use separate read and write sequencers, the compiler represents them efficiently as a single structure. Each entry in this unified sequencer contains shared loop limits but separate strides (one for read, one for write) and separate base addresses (one for source, one for destination). This compact representation exploits the fact that read and write amounts must always match.

**DMA Engine assignment is flexible:**
- Any DMA Engine among the 8 can handle any transfer, but using the DMN's own DMA Engine is more efficient (not quantitatively measured).
- The compiler typically uses the source DM DMN's DMA Engine, but any DMA Engine works.
- The 8 DMA Engines can transfer between different memory components in parallel (e.g., DMA #0: HBM ↔ DM, DMA #1: DM ↔ DM).
- The compiler only allows moving from one tensor (HBM/DM/SPM) to another tensor (HBM/DM/SPM).
- For inter-chip transfers, all chip IDs are globally agreed upon across the system
- Programmers can leave DMA Engine selection unspecified and let the compiler choose, though explicit specification is also supported

**SRAM access patterns for optimal bandwidth:** SRAM memory bandwidth depends critically on DMN (Data Memory Network) interleaving. For detailed SRAM performance characteristics and interleaving patterns, see the [Data Memory section](./memory-performance.md#data-memory-dm). The key principle: interleave across both DMNs to achieve full 256 B/cycle bandwidth.

**Bandwidth trade-offs:** DMA provides flexibility for arbitrary tensor moves but may underutilize SRAM slice bandwidth compared to the Tensor Unit. However, HBM bandwidth is often the bottleneck in practice, making this less critical. For SRAM-to-SRAM transfers, the Tensor Unit is often more efficient, except when the Switch Engine operates at size 256 (which may be slower than DMA).

### Tensor Memory Mapping

The compiler automatically derives the correspondence between source and destination memory indices from the mapping environments.
Given tensor memory mappings `(e, e')`, the compiler computes how each flat memory index relates to the logical tensor dimensions:
```text
// S, e' |- i ~ { i_A = (i % 65536) / 256, i_B = i / 65536, i_C = i % 256 }
```

A simple layout transformation that reorders dimensions:
```text
axes![A = 256, B = 256, C = 256];
// e_1 = A * B * C, e_2 = B * A * C
// DMA: <S, e_1, HBM@0> =id=> <S, e_2, HBM@256^3>
```

### DMA Sequencer Internals

This section explains how DMA sequencers execute at the hardware level.

A DMA Descriptor represents a single execution unit that the hardware can process.
Each DMN's DMA Engine can accept multiple DmaDescriptors, which it executes in sequence (or potentially in parallel when resources permit).
The sequencer within each descriptor determines the exact order in which memory addresses are accessed.

**Startup overhead detail:** As mentioned in the performance considerations above, each DMA Descriptor incurs approximately 500 cycles of initial latency before data transfer begins.

**Example sequencer execution:**
```text
// DmaSequencer = [A=256:(65536, 256), B=256:(256, 65536), C=256:(1, 1)], base=(0, 256^3)
```

Reading one data element per cycle, each cycle performs:

| i | ti | read addr | write addr |
|---|---|---|---|
| 0 | `{ A: 0, B: 0, C: 0 }` | 0 | write_base (=256³) |
| 1 | `{ A: 0, B: 0, C: 1 }` | 1 | 1 + write_base |
| ... | ... | ... | ... |
| 255 | `{ A: 0, B: 0, C: 255 }` | 255 | 255 + write_base |
| 256 | `{ A: 0, B: 1, C: 0 }` | 256 | 65536 + write_base |
| `i = a*256² + b*256 + c` | `{ A: a, B: b, C: c }` | `i` | `256*a + 256²*b + c + write_base` |

The DmaSequencer compactly represents this address mapping table.
With `stride0 = 256`, the hardware reads and writes 256 bytes per cycle: cycle 0 processes all values for `(A, B, C) = (0, 0, 0..255)` as a single packet.

A complete descriptor example:
```text
// DmaSequencer = [A=256:(65536, 256), B=256:(256, 65536), C=256:(1, 1)],
//                base=(0, 256^3), stride0 = 256
// media_source = HBM, media_dest = HBM, DmnIndex = (0, 0, 0)
```
This descriptor activates the DMA Engine on Chip 0, Cluster 0, DMN 0, moving data from HBM starting at address 0 to HBM starting at address 256³.
The transfer completes in approximately 500 cycles (initial latency) + 256 × 256 cycles (data transfer).

**How the compiler derives sequencers:** Given source and destination tensor shapes along with a stream shape, the compiler derives the DMA sequencer configuration:
```text
// stream_shape = [A * B * C]
// => read_sequencer = [A=256:65536, B=256:256, C=256:1], base=0
// => write_sequencer = [A=256:256, B=256:65536, C=256:1], base=256^3
```

The derivation process follows these steps:
- The read sequencer is derived by projecting the source tensor mapping onto the stream shape
- The write sequencer is derived by projecting the destination tensor mapping onto the stream shape
- These are combined into a unified DMA sequencer with paired strides and bases
- The packet size (stride0) is inferred from the consecutive read/write volume: if both read and write access 256 consecutive bytes, the optimal stride0 is 256 bytes

When stride0 is not 256-byte aligned, the cycle count formula is `ceil(stride0 / 256)`. However, **HBM write operations** incur additional penalties beyond the ceil calculation. The unaligned write requires a Read-Modify-Write (RMW) operation for the partial 256-byte block, slowing the operation significantly (see the Misaligned Access section in ./memory-performance.md for details). For **HBM read operations**, the penalty is limited to the ceil overhead. For **SRAM operations**, alignment has minimal impact.

### Memory Bandwidth Limits

Memory bandwidth limits are crucial for achieving optimal DMA performance.
A single DMA Engine can theoretically move up to 256 bytes per clock cycle, but the actual transfer rate is constrained by the slowest component in the data path: the source memory, the destination memory, or the PCIe interconnect for inter-chip transfers.

For detailed characteristics and optimization strategies for each memory type, see:
- [Data Memory (DM) performance](./memory-performance.md#data-memory-dm)
- [High-Bandwidth Memory (HBM) performance](./memory-performance.md#high-bandwidth-memory-hbm)
- [Scratchpad Memory (SPM) performance](./memory-performance.md#scratchpad-memory-spm)

**Key bandwidth constraints:**
- HBM: 1.5 TB/s combined read + write per chip (0.75 TB/s read + 0.75 TB/s write)
- DM: 256 B/cycle per cluster with proper DMN interleaving (128 B/cycle per DMN)
- SPM: 128 B/cycle per cluster
- PCIe DMA Engine: 30 bytes/cycle for inter-chip transfers

## Detailed Examples

The following examples illustrate DMA Engine behavior across various configurations, from simple single-engine transfers to complex multi-DMN operations with performance considerations.


### Example 1: Single DMA Engine HBM to HBM

This example demonstrates a basic HBM-to-HBM transfer using a single DMA Engine to rearrange tensor dimensions.
The operation achieves good performance through effective channel interleaving, distributing memory accesses across different HBM channels to enable parallel processing.

**Operation arguments:**

```text
axes![A = 8, B = 8, C = 256];
// dtype = i8
// media_in = media_out = HBM
// b_in = { chip: 0, element: 0 }, b_out = { chip: 0, element: 16384 }
// In = { element: m![A, B, C] }
// Out = { element: m![B, A, C] }
// Stream = { time: m![A, B], packet: m![C] }
```

**Generated sequencer configurations:**
- Read sequencer configuration: `[A=8:2048, B=8:256, C=256:1]:256  HBM/D@0`
- Write sequencer configuration: `[A=8:256, B=8:2048, C=256:1]:256  HBM/D@16384`

**Why this achieves good performance:**
Channel interleaving enables efficient parallel processing.
The strides in the non-innermost sequencer entries (256 and 2048) toggle HBM address bits 8 and 11, which correspond to the stack and channel selection bits.
This access pattern ensures that every read and write request targets a different HBM channel, with multiple memory operations proceeding in parallel.

Although each 256-byte transfer takes 4 cycles at 0.75GHz clock speed, the parallel distribution across channels enables efficient execution.
At 1GHz, the total time is approximately 128 cycles (64 read requests + 64 write requests) plus approximately 500 cycles of initial latency.


| read #i | ti                            |  read addr   |  write addr  |
|---------|-------------------------------|--------------|--------------|
|    0    | `{ A: 0, B: 0, C: 0 }` |      0       |      write_base(=16384)       |
|    1    | `{ A: 0, B: 0, C: 1 }` |      1       |      1 + write_base   |
|    2    | `{ A: 0, B: 0, C: 2 }` |      2       |      2 + write_base   |
|   ...   | ... |
|   255   | `{ A: 0, B: 0, C: 255 }` |  255     |    255 + write_base   |
|   256   | `{ A: 0, B: 1, C: 0 }` |    256     |    2048 + write_base |
|   257   | `{ A: 0, B: 1, C: 1 }` |    257     |    2048 + 1  + write_base|
|   ...   | ... |
| `i = a * 2048 + b * 256 + c` | `{ A: a, B: b, C: c }` | `i` | 256 * a + 2048 * b + c + write_base |
|   ...   | ... |

**Bandwidth sharing note:** HBM bandwidth is 1.5 TB/s (read + write combined), and each DMA Engine has 256 GB/s bandwidth. For DRAM ↔ DRAM operations, read bandwidth is 0.75 TB/s. If 4 DMA Engines perform DRAM ↔ DRAM operations, each gets ~0.1875 TB/s. Even with stride0=256, each engine reads 256B per request but cannot complete one request per cycle due to this bandwidth constraint.

### Example 2: Single DMA Engine HBM to DM

This example demonstrates an HBM-to-DM transfer that achieves optimal bandwidth by carefully interleaving both HBM channels and DM DMNs.
Both memory systems require specific access patterns to reach their full bandwidth potential.

**Operation arguments:**

```text
axes![A = 256, B = 256, C = 256];
// dtype = i8
// media_in = HBM
// media_out = DM
// b_in = { chip: 0, element: 0 }
// b_out = { chip: 0, cluster: 0, slice: 0, element: 0 }
// In = { element: m![B, A, C] }
// Out = { slice: m![A / 4], element: m![A % 4, B, C] }
// Stream = { time: m![B, A % 4, A / 4 % 32, A / 128], packet: m![C] }
```

**Generated sequencer configurations:**
- Read sequencer configuration: `[B=256:65536, A%4=4:256, A/4%32=32:1024, A/128=2:32768, C=256:1]:256  HBM/D@0`
- Write sequencer configuration: `[B=256:256, A%4=4:65536, A/4%32=32:slice_stride, A/128=2:DMN_stride, C=256:1]:256  DM/D@0`

**Performance analysis:**
Both HBM and DM achieve full bandwidth through careful interleaving in their respective access patterns.

**HBM side:** The stride of 32768 for the `A/128=2` loop interleaves memory accesses effectively.
For the innermost 2 iterations, this interleaves at the byte level; for outer iterations, it interleaves across HBM channels.
The hardware command queue processes all 65536 requests (256 * 4 * 32 * 2) efficiently, utilizing full HBM bandwidth.

**DM side:** DMN and slice interleaving work together to maximize throughput.
Each of the two DMNs provides 128 bytes/cycle bandwidth, so a 256-byte write normally requires 2 cycles on a single DMN.
However, interleaving consecutive requests across both DMNs (achieved through the DMN_stride and slice_stride) enables the two DMNs to operate in parallel, processing one 256-byte request per cycle.
All 65536 write requests therefore complete at one request per cycle.

**Total execution time:** Approximately 65536 cycles (read) + 65536 cycles (write) + 500 cycles (initial latency). Since reads and writes overlap in the pipeline, the actual time is closer to max(65536, 65536) + 500 ≈ 66036 cycles.


### Example 3: Single DMA Engine DM to DM

This example shows a DM-to-DM transfer within a single cluster, where both reads and writes access the same DM.
This scenario requires careful DMN interleaving for both operations to avoid contention and achieve maximum bandwidth.

**Operation arguments:**

```text
axes![A = 256, B = 256, C = 256];
// dtype = i8
// media_in = DM
// media_out = DM
// b_in = { chip: 0, cluster: 0, slice: 0, element: 0 }
// b_out = { chip: 0, cluster: 0, slice: 0, element: 4 * 256 * 256 }
// In = { slice: m![A / 4], element: m![A % 4, B, C] }
// Out = { slice: m![A / 4], element: m![B, A % 4, C] }
// Stream = { time: m![B, A % 4, A / 4 % 32, A / 128], packet: m![C] }
```

**Generated sequencer configurations:**
- Read sequencer configuration: `[B=256:1, A%4=4:65536, A/4%32=32:slice_stride, A/128=2:DMN_stride, C=256:1]:256  DM/D@0`
- Write sequencer configuration: `[B=256:1024, A%4=4:256, A/4%32=32:slice_stride, A/128=2:DMN_stride, C=256:1]:256  DM/D@(4 * 256 * 256)`

**Performance analysis:**
DMN and slice interleaving enable full bandwidth for both read and write operations.
Each 256-byte access is structured to interleave across the two DMNs, while the outer loops interleave across different DM slices.
Each DMN provides 128 bytes/cycle bandwidth, so a single 256-byte access normally requires 2 cycles on one DMN.
However, alternating requests between both DMNs enables parallel operation to achieve full 256 B/cycle bandwidth.

**Request execution:**
- Total read requests: 65536 (256 * 4 * 32 * 2)
- Total write requests: 65536
- At saturation with proper interleaving, one request completes per cycle

**Total execution time:** Approximately 131072 cycles (since reads and writes must proceed sequentially for DM-to-DM within the same cluster) + 500 cycles (initial latency).

**Note on packet size alignment:** The choice of C=256 is important for performance. If C were between 1-255, the cycle count remains similar because the number of DMA requests determines execution time. However, if the packet size is 256n+r (where 0 ≤ r < 256), the cycle count increases by a factor of (n+1) due to more requests. Aligning packet sizes to 256-byte boundaries maximizes data transferred per request.

### Example 4: Homogeneous DMA Engine, HBM to DM (Pathological: Bank Conflict)

This example demonstrates performance degradation from poorly designed memory access patterns: severe HBM bank conflicts.
The issue arises when the stream shape causes consecutive accesses to trigger row switches within HBM banks, preventing efficient parallel execution and resulting in approximately 10x slower performance compared to well-optimized access patterns.

**Operation arguments:**

```text
// 1 chip (8 DMNs): chip-related mapping is not needed
axes![A = 64, B = 2048, C = 1024];
// dtype = i8
// media_in = HBM
// media_out = DM
// b_in = 0
// b_out = 0
// In = { cluster: m![B / 1024], slice: m![B / 256 % 4, A], element: m![B % 256, C] }
// Out = { slice: m![A / 4], element: m![B, A % 4, C] }
// Stream = { cluster: m![B / 1024], slice: m![B / 256 % 4],
//              time: m![B % 256, C / 256, A % 32, A / 32], packet: m![C % 256] }
```

**Generated sequencer configurations:**
- Read sequencer configuration at `(cluster_i, dmn_j)`: `[B%256=256:1024, C/256=4:256, A%32=32:2^21, A/32=2:2^26, C%256=256:1]:256  HBM/D@(i * (1024 * 1024) + j * (256 * 1024))`
    - The base address offset `i * (1024 * 1024) + j * (256 * 1024)` is derived from the DMN location `(B/1024, B/256%4) = (i, j)`
- Write sequencer configuration at `(cluster_i, dmn_j)`: `[B%256=256:1024, C/256=4:256, A%32=32:slice_stride, A/32=2:DMN_stride, C%256=256:1]:256  DM/D@(cluster_i, dmn_j, 0)`

**Why this performs poorly: row-level bank conflicts**
The stream shape structure optimized for DM's DMN/slice interleaving creates a pathological access pattern for HBM.
The innermost interleaving dimensions (A%32 and A/32) correspond to HBM address bits 21 and 26, which control row addressing within banks.
Consecutive memory accesses trigger row switches within the same bank on nearly every request.

Channel interleaving still occurs (the C dimension's stride of 256 enables stack interleaving across all 32 channels), but this parallelism cannot compensate for the row conflict penalty within each channel.
Each access within a channel must wait for the previous row to close and the new row to open, dramatically increasing latency.

**Performance breakdown:**

HBM reads (the bottleneck):
- Per DMN: 65536 data requests (256 * 4 * 32 * 2)
- Across 8 DMNs: 524288 total requests, distributed evenly across 32 HBM channels
- Each channel handles: 16384 requests
- Each request incurs approximately 40 cycles due to bank conflicts (a conservative estimate; actual penalty depends on tCCD and FR-FCFS scheduling)
- Total HBM time: approximately 655360 cycles (16384 * 40)

DM writes (not the bottleneck):
- DMN interleaving works correctly, achieving full 256 B/cycle bandwidth
- 65536 requests per DMN, processing at one request per cycle

**Total execution time:** Approximately 655360 cycles + 500 cycles (initial latency) ≈ 655860 cycles.

**Critical lesson:** Careful access pattern design is essential for performance. Avoid bank conflicts through proper stream shape construction. Note that this estimate is conservative; actual performance may be somewhat better due to FR-FCFS (First Ready-First Come First Served) memory scheduling, which can mitigate some conflicts, but the fundamental problem remains severe.

### Example 5: Homogeneous DMA Engine HBM to DM (Pathological: Missing Stack Interleaving)

This example demonstrates another common pitfall: failing to interleave across HBM's stack dimension (address bit 8).
When this bit is not toggled by the access pattern, only 16 of the 32 available HBM channels are utilized, cutting effective bandwidth in half.

**Operation arguments:**

```text
// 1 chip (8 DMNs)
axes![A = 8, B = 64, C = 8, D = 512];
// dtype = i8
// media_in = HBM
// media_out = DM
// b_in = 0
// b_out = 0
// In = { element: m![A, B, C, D] }
// Out = { cluster: m![A / 4], slice: m![A % 4, B], element: m![C, D % 256] }
// Stream = { cluster: m![A / 4], slice: m![A % 4], time: m![C, B % 32, B / 32], packet: m![D % 256] }
```

**Generated sequencer configurations:**
- Read sequencer configuration at `(cluster_i, dmn_j)`: `[C=8:512, B%32=32:4096, B/32=2:131072, D%256=256:1]:256  DM/D@(i * 2^20 + j * 2^18)`
    - The base address offset `i * 2^20 + j * 2^18` is derived from the DMN location `(A/4, A%4) = (i, j)`
- Write sequencer configuration at `(cluster_i, dmn_j)`: `[C=8:256, B%32=32:slice_stride, B/32=2:DMN_stride, D%256=256:1]:256  DM/D@(cluster_i, dmn_j, 0)`

**Why this performs poorly: missing stack bit interleaving**
The stream shape does not exercise HBM address bit 8, which controls the stack dimension.
In the HBM access pattern, the C axis has a stride of 512, so bit 8 is never toggled during the innermost loops.
This occurs in operations like tensor splits where dimension structure changes between input and output (notice that the input tensor mapping includes D/256 but the output/stream does not).

HBM channel selection uses address bits 9-28, while the stack bit is bit 8.
Without bit 8 interleaving, memory requests distribute across only 16 of the 32 available channels, immediately halving achievable bandwidth.

**Performance breakdown:**

HBM reads (the bottleneck):
- Per DMN: 512 data requests (8 * 32 * 2)
- Across 8 DMNs: 4096 total requests, distributed across only 16 channels
- Each channel handles: 256 requests
- Each channel's bandwidth: 256B per 4 cycles at 0.75GHz, or approximately 5.3 cycles per request at 1GHz
- Total HBM time: approximately 1357 cycles (256 * 5.3)

DM writes (not the bottleneck):
- DMN interleaving achieves full 256 B/cycle bandwidth
- 512 requests per DMN (8 * 32 * 2), processing at one request per cycle
- DM writes overlap with HBM reads in the pipeline, so their latency is hidden

**Total execution time:** Approximately 1357 cycles + 500 cycles (initial latency) ≈ 1857 cycles.

**Critical lesson:** Achieving full HBM bandwidth (1.5TB/s) and DMA Engine bandwidth (2TB/s) requires memory access patterns that interleave across all 32 channels by toggling all relevant address bits including the stack bit (bit 8). Missing even one dimension of interleaving significantly degrades performance.

### Example 6: Heterogeneous DMA Engine with Segmentation

This example demonstrates a heterogeneous DMA operation where the tensor shape does not divide evenly across all DMNs.
Some DMNs must use different stream environments than others, and in extreme cases, a DMN may need to segment its work into multiple DMA commands to avoid writing to incorrect memory locations.
This illustrates both the flexibility and complexity of heterogeneous DMA operations.

**Operation arguments:**

```text
// 4 chips
axes![A = 15, B = 32, C = 256, D = 8];
// dtype = i8
// media_in = DM
// media_out = HBM
// b_in = 0
// b_out = 0
// In = let A' = A + 1# in
//        { chip: m![D / 2], cluster: m![D % 2], slice: m![A' / 4, A' / 2 % 2, B],
//          element: m![A' % 2, C] }
// Out = { chip: m![D / 2], element: m![D % 2, B, A, C] }
// StreamFn(chip_i, cluster_j, slice_k) = let A' = A + 1# in
//        { chip: m![(D / 2) @ i = 1],
//          cluster: m![(D % 2) @ j = 1],
//          slice: m![(A' / 4) @ k = 1],
//          time: (k == 0,1,2): m![A' % 2, B, A' / 2 % 2, C]
//                (k == 3, exec #0): m![A' % 2, B, A' / 2 = 1, C]
//                (k == 3, exec #1): m![A' = 1, B, A' / 2 % 2 @ 1, C],
//          packet: m![C] }
```

The compiler generates the following sequencer configurations:
- Read sequencer configuration at `(chip_i, cluster_j, dmn_k)`:
    - k = 0, 1, 2:
    `[A'%2=2:256, B=32:slice_stride, A'/2%2=2:DMN_stride, C=256:1]:256  DM/D@(chip_i, cluster_j, dmn_k, 0)`
    - k = 3:
        - execution #0
            `[A'%2=2:256, B=32:slice_stride, A'/2=1:DMN_stride, C=256:1]:256  DM/D@(chip_i, cluster_j, dmn_3, 0)`
        - execution #1
            `[A'%2=2:256, B=32:slice_stride, A'/2=1:DMN_stride, C=256:1]:256  DM/D@(chip_i, cluster_j, dmn_3, 0)`
- Write sequencer configuration at `(chip_i, cluster_j, dmn_k)`:
    - k = 0, 1, 2:
    `[A'%2=2:256, B=32:15 * 256, A'/2%2=2:512, C=256:1]:256  HBM/D@(0 + i * 2 * (15 * 32 * 256) + j * (15 * 32 * 256) + k * (4 * 256))`
    - k = 3:
        - execution #0
            `[A'%2=2:256, B=32:15 * 256, A'/2=1:512, C=256:1]:256  HBM/D@(0 + i * 2 * (15 * 32 * 256) + j * (15 * 32 * 256) + 3 * (4 * 256))`
        - execution #1
            `[A'%2=1:256, B=32:15 * 256, A'/2=1:512, C=256:1]:256  HBM/D@(0 + i * 2 * (15 * 32 * 256) + j * (15 * 32 * 256) + 3 * (4 * 256) + 512)`
            - 512: offset by `A'/2%2@1`


**Why DMN #3 requires segmentation:**
The tensor dimension A=15 does not divide evenly across 4 DMNs (15 = 3*4 + 3).
DMNs #0, #1, and #2 each process exactly 4 elements of the A dimension.
DMN #3 must process the remaining 3 elements (A=12, 13, 14) but its sequencer would naturally try to process 4 elements.
If DMN #3 used the same single-command pattern as the other DMNs, it would write one extra element, corrupting memory in the region designated for B * (A + 1#) * C.

The compiler segments DMN #3's work into two commands to avoid this:
- Execution #0 handles part of the valid range
- Execution #1 handles the remainder, ensuring the total is exactly 3 elements rather than 4

**Performance comparison:**

DMNs #0, #1, #2 (single command each):
- DM reads: 128 cycles for 2 * 32 * 2 packets of 256B each
- HBM writes: 128 cycles with proper channel interleaving
- Total: approximately 256 cycles (reads and writes overlap) + 500 cycles (initial latency) = 756 cycles

DMN #3 (two commands):
- Execution #0: 64 DM read cycles + 64 HBM write cycles + 500 cycles initial latency
  - Note: reads from one DMN only, but slice interleaving still applies
- Execution #1: 32 DM read cycles + 32 HBM write cycles + 500 cycles initial latency
- Total: 192 data cycles + 1000 cycles (initial latency for two commands) = 1192 cycles

**Overall execution time:** The heterogeneous operation completes when the slowest DMN finishes. DMN #3 determines the total time: approximately 1192 cycles.

**Key insight:** Command segmentation incurs additional startup overhead (500 cycles per command). Choose tensor shapes that divide evenly across DMNs when possible, avoiding the need for heterogeneous stream environments and command segmentation.


## Performance

DMA Engine performance depends on memory types, access patterns, and parallelism strategies.

### Memory-Specific Bandwidth

Transfer bandwidth varies by memory type and configuration:

**Data Memory (DM/SRAM)**:
- Peak bandwidth: 256 B/cycle (with proper DMN interleaving)
- Requires interleaving across both DMNs (128 B/cycle each)
- Bank conflicts and starvation can severely degrade performance
- See [Memory Performance](./memory-performance.md) for DM optimization details

**High-Bandwidth Memory (HBM)**:
- Peak bandwidth: 1.5 TB/s per chip (48 GB/s per channel × 32 channels)
- Channel interleaving is essential for high bandwidth
- Misaligned access and bank conflicts cause severe degradation
- See [HBM Performance](./memory-performance.md#high-bandwidth-memory-hbm) for optimization strategies

**Scratchpad Memory (SPM)**:
- Bandwidth: 128 B/cycle per cluster
- Restricted to same-chip transfers

### Startup Latency

Each DMA command incurs approximately **500 cycles** of startup latency before data transfer begins.
This fixed cost is amortized over large transfers but becomes significant for small tensors.

Command segmentation (as shown in Example 6) doubles startup latency by requiring two separate commands, emphasizing the importance of tensor shapes that divide evenly across DMNs.

### Parallelism Strategies

Multiple DMA Engines can operate simultaneously:
- **8 DMA Engines per chip** (one per pair of DMNs, 8 DMNs per cluster)
- Parallel DMA operations on independent data enable high aggregate bandwidth
- Local DMN memory access is faster than cross-DMN access (not quantitatively measured)

### Alignment Constraints

Strict alignment requirements affect performance:
- **DM writes**: 8-byte alignment required for addresses and packet sizes
- **HBM operations**: 1-byte alignment for reads/writes, but HBM-to-DM transfers require 8-byte alignment
- **Maximum packet size**: 4096 bytes (AXI protocol constraint)
- Misaligned access in HBM can halve bandwidth or trigger expensive Read-Modify-Write operations

### Bank Starvation Prevention

DMA Engine shares DM bank access with Fetch and Commit Engines.
DMA has the **lowest priority** among these engines, making it vulnerable to bank starvation.
If a DMA request blocks for more than 4,096 cycles, a NoC timeout occurs, requiring a hardware reset.

The compiler prevents this by ensuring operations with 64+ consecutive same-bank accesses are not scheduled concurrently with DMA.
See [Bank Starvation](./memory-performance.md#bank-starvation) for details.

### Inter-Chip Transfers

PCIe-based inter-chip transfers have limited bandwidth:
- **30 B/cycle** for both reads and writes
- Significantly slower than on-chip transfers
- Consider minimizing cross-chip data movement in algorithm design
