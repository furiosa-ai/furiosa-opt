# TRF Sequencer

The TRF Sequencer is part of the [Aligner](./aligner.md) stage.
It reads weight data from the Tensor Register File (TRF) and reshapes it to match the computation mapping required by the [Reducer](./reducer.md).
It broadcasts stored weights across the temporal (via sequencer) and spatial (via `reg_read_size`) dimensions, enabling weight reuse without additional memory usage.
This operation is the weight-side counterpart to the [Stream Adapter](./stream-adapter.md), which prepares activation data on the input side.

## Interface

```rust,ignore
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
/// TRF address modes for partitioning the register file.
enum TrfAddress {
    FirstHalf,  // First half of TRF
    SecondHalf, // Second half of TRF
    Full,       // Entire TRF
}

impl<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet>
    CollectTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Stores tensor data from the Collect Engine into TRF.
    /// The outermost axes of the input become the Row dimension,
    /// and the remaining inner axes become the Element dimension.
    /// The resulting layout is [Chip, Cluster, Slice, Row, Element].
    pub fn to_trf<Row: M, Element: M>(
        self,
        address: TrfAddress,
    ) -> TrfTensor<D, Chip, Cluster, Slice, Row, Element> {
        // Hardware implementation: writes data to TRF via SRAM-to-TRF
    }

    /// Aligns this input stream with a TRF tensor for contraction.
    /// Configures the TRF Sequencer to reshape the TRF tensor
    /// to match the computation mapping.
    pub fn align<OutTime: M, OutPacket: M, Row: M, TrfElement: M>(
        self,
        trf_tensor: &TrfTensor<D, Chip, Cluster, Slice, Row, TrfElement>,
    ) -> AlignedPair<'l, T, D, Chip, Cluster, Slice, Row, OutTime, OutPacket> {
        // Hardware implementation: configures Stream Adapter and TRF Sequencer
    }
}
```

The typical data flow is: `collect() → to_trf()` for weights (sub context), then `collect() → align(&trf) → contract() → accumulate()` for activations (main context).
The `Chip`, `Cluster`, and `Slice` dimensions pass through unchanged.

## Architecture

### Conceptual Operation

The TRF Sequencer transforms the TRF tensor mapping into the computation mapping:

```text
TRF mapping:         [Chip, Cluster, Slice, Row, Element]
                          ↓ TRF Sequencer
Computation mapping: [Chip, Cluster, Slice, Row, OutTime, OutPacket]
```

This transformation involves four operations:

1. **Spatial Read**: Fill in `OutPacket` (which is 64 bytes), with the mechanism involving `reg_read_size`.
2. **Row Partitioning**: Each Row reads from its own TRF region.
3. **Temporal Broadcasting**: Axes with stride 0 are broadcast across `Time`, reusing the same weight data each cycle.
4. **Time Reordering**: The `Time` axes are reordered via a nested-loop sequencer configuration.

```text
       SRAM
        │
        │ SRAM-to-TRF (short command or tensor unit path)
        ▼
┌──────────────┐   TRF mapping: [Chip, Cluster, Slice, Row, Element]
│     TRF      │
└──────┬───────┘
       │ TRF Sequencer (nested-loop config)
       ▼
┌──────────────┐   Computation mapping: [Chip, Cluster, Slice, Row, OutTime, OutPacket]
│   Reducer    │◄── Stream Adapter (activation data)
└──────────────┘
```

### TRF Read Mechanism

Every cycle, the [Reducer](./reducer.md) consumes exactly `mac_width` (64 bytes) of data. This 64-byte window is composed of two parts:

```text
┌──────────── mac_width (64 bytes) ────────────┐
│   broadcast    │  reg_read_size (contiguous) |
│ ← (repeated) → │    ← inner (from TRF) →     |
└────────────────┴─────────────────────────────┘
```

- `reg_read_size`: The number of contiguous bytes read from TRF each cycle. Must be a power of two: 1, 2, 4, 8, 16, 32, or 64 bytes.
- `broadcast`: The portion of `mac_width` not covered by `reg_read_size` is filled by repeating the read data. For example, if `reg_read_size = 8`, the 8 bytes are broadcast 8× to fill 64 bytes.

The **inner part** (within `reg_read_size`) is always read contiguously each cycle. The sequencer does not control this region.
The **outer part** (beyond `mac_width`) is controlled by the sequencer entries' `(size, stride)` pairs, which specify the iteration order over the remaining dimensions.

> [!NOTE]
> `reg_read_size` is **not a user-specified parameter**.
> The compiler determines it by comparing the innermost axes of the TRF mapping and the computation mapping: the contiguous portion that is common to both (within 64 bytes) becomes `reg_read_size`.
> This means neither the TRF mapping alone nor the computation mapping alone determines `reg_read_size` — it is derived from their intersection.

> [!NOTE]
> **64-byte alignment constraint**: When `reg_read_size = 64` bytes (i.e., equal to `mac_width`), the base address and all sequencer strides must be aligned to 64 bytes.
> A 64-byte read spans both bank columns (32 bytes each).
> If the address is not 64-byte aligned, the read would cross a bank column boundary, which is not supported by the hardware.

### SRAM-to-TRF (StoTRF)

Data is loaded into TRF after the Collect Engine.
If the sequencer is configured for completely contiguous access (no gaps or reordering), the load can be executed as a **short command** (a compact hardware instruction that bypasses the full tensor unit pipeline).

If this condition is not met, the load goes through the full **tensor unit path** (SRAM → Fetch → Switch → Collect → `to_trf()`), which supports arbitrary layouts via the fetch engine but has higher setup overhead.

### TRF Memory Layout

The TRF is a banked SRAM organized as 8 bank rows × 2 bank columns.
Each bank row corresponds to a Row.
Each bank contains 128 rows (Full mode) or 64 rows (Half mode), and each row holds 32 bytes (`320b`) of data:

```text
  bank row 7 ──────────────────────────────────────────┐
  (= Row 7)  bank col 0           bank col 1      │
       :       ┌─ 320b ─┐           ┌─ 320b ─┐       ╱
       :      ╱         ╱|         ╱         ╱|      ╱
  bank row 1 ╱─────────╱ |        ╱─────────╱ |    ╱  bank row
  bank row 0╱         ╱  |       ╱         ╱  |   ╱   (= Row)
            │         │  │       │         │  │
            │ 128 rows│ ╱        │ 128 rows│ ╱
            │         │╱         │         │╱
            └─────────┘          └─────────┘
```

Each bank row corresponds to a Row and can be accessed independently in parallel.
The 2 bank columns within each bank row share the same row address space.

Each element in the TRF is addressed via a bit-field index:

```text
┌───────────┬──────────────┬──────────┬──────────┐
│ bank row  │ row in bank  │ bank col │  offset  │
│  (3 bit)  │  (7b / 6b)   │  (1 bit) │  (6 bit) │
└───────────┴──────────────┴──────────┴──────────┘
```

| Field | Bits | Description |
|-------|------|-------------|
| bank row | 3 | Selects Row (0–7). Each bank row corresponds directly to a Row and can be accessed independently, enabling parallel reads. When `rows < 8`, unused bits extend the row address space |
| row in bank | 7 (Full) / 6 (Half) | Selects row within a bank: 128 rows (Full) or 64 rows (Half). `FirstHalf` uses the lower 64 rows (rows 0–63) and `SecondHalf` uses the upper 64 rows (rows 64–127), so Half mode needs only 6 bits |
| bank col | 1 | Selects bank column (2 columns per bank) |
| offset | 6 | Element offset within a row in 5-bit granularity (64 positions × 5 bits = 320 bits per row) |

The `reg_types` value determines how many 5-bit slots each element occupies:

| `reg_types` | Element Width | Slots per Element | Elements per Row |
|-------------|---------------|-------------------|------------------|
| 0 | 5-bit (`i4` extended) | 1 | 64 |
| 1 | 10-bit (`i4`→`i8`) | 2 | 32 |
| 2 | 10-bit (`i8`→`f8`) | 2 | 32 |
| 3 | 20-bit (`bf16`) | 4 | 16 |

When `rows < 8`, the unused bank row bits effectively increase the per-Row capacity. For example, with `rows = 4`, one extra bit extends `row in bank` from 7 to 8 bits, doubling the rows available per Row.

### Specifications

#### TRF Address Modes

| Mode | Region | Capacity |
|------|--------|----------|
| `FirstHalf` | First half of TRF | `register_file_size / 2` |
| `SecondHalf` | Second half of TRF | `register_file_size / 2` |
| `Full` | Entire TRF | `register_file_size` |

These address modes partition the TRF so that tensors stored in different regions do not interfere with each other.
`Full` dedicates the entire TRF to a single tensor.
`FirstHalf` and `SecondHalf` isolate up to two tensors, allowing them to coexist in TRF simultaneously (for example, one half can be read by the Sequencer while the other is written by SRAM-to-TRF, enabling double buffering).
The two halves can be flipped between iterations.

#### Rows

Each Row maps directly to a bank row in TRF. Since bank rows are physically independent, all Rows can read in parallel without contention.

| `rows` | Description |
|------------|-------------|
| 1 | Single Row (1 bank row used) |
| 2 | 2 Rows (2 bank rows used) |
| 4 | 4 Rows (4 bank rows used) |
| 8 | 8 Rows (all bank rows used) |

Each Row reads the same sequencer pattern from a different TRF offset (different bank row, same `row_in_bank`/`bank_col`/`offset`).

#### Sequencer Configuration

The TRF Sequencer uses the same nested-loop configuration as all other sequencers (see [Sequencer](../../moving-tensors/sequencer.md)):

| Parameter | Range | Description |
|-----------|-------|-------------|
| Entries | 1–8 | Each entry is a `(size, stride)` pair |
| `size` per entry | 1–65,536 | Iteration count for this dimension |
| `stride` per entry | signed 32-bit | Address increment per iteration |

The sequencer entries control iteration over the **outer part** — the dimensions beyond `mac_width`. The inner part (within `reg_read_size`) is read contiguously each cycle and is not represented in the sequencer entries. See [TRF Read Mechanism](#trf-read-mechanism) for how the inner and outer parts relate.

Axes with `stride = 0` are broadcast: the same data is repeated for each iteration of that dimension.

## Performance

### TRF Cache

#### Purpose

The TRF bank columns are shared between read (main context sequencer) and write (sub context StoTRF). A read cache sits between the TRF banks and the Reducer so that cache hits serve reads without occupying a bank — freeing the bank for concurrent StoTRF writes.

#### Structure

The cache is **direct-mapped**:

| Parameter | Value |
|-----------|-------|
| Rows per Row | 4 rows × 2 bank columns = 8 entries |
| Entry size | 32 bytes |
| Rows | 8 |
| Total capacity | 8 × 4 × 2 × 32 = 2,048 bytes |

#### Operation

1. **First read** (cache miss) — data is fetched from the TRF bank and loaded into the cache. The bank is occupied for this cycle.
2. **Subsequent reads to the same address** (cache hit) — data is served from the cache. The bank is **not** occupied, allowing StoTRF writes to proceed simultaneously.

#### Bank Conflict Priority

When a cache miss and an StoTRF write target the same bank column in the same cycle, **read has higher priority** than write. This means frequent cache misses can stall concurrent StoTRF operations.

#### Impact of `reg_read_size`

| `reg_read_size` | Bank columns used per cycle | Cache miss impact |
|-----------------|------------------------------|-------------------|
| ≤ 32 bytes | 1 column | A miss on the same bank column as a concurrent StoTRF write will still cause a conflict. However, if the innermost sequencer entry interleaves reads at 32-byte granularity across the two bank columns, then the sequencer and StoTRF alternate columns on successive cycles — avoiding degradation even during misses. |
| 64 bytes | Both columns | A miss occupies both columns simultaneously, blocking StoTRF for that cycle. Write throughput degrades in proportion to the miss rate. |

## Examples

### Basic Weight Broadcasting (MatMul)

This example shows a matrix multiplication where weights are stored in TRF and broadcast across the `M` (output row) dimension:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![M = 32, N = 8, K = 32];

/// Stores weights into TRF (sub context).
fn store_weights<'l, const T: Tu>(
    weights: CollectTensor<'l, T, bf16, m![1], m![1], m![1], m![N, K / 16], m![K % 16]>,
) -> TrfTensor<bf16, m![1], m![1], m![1], m![N], m![K]> {
    // TRF mapping: [
    //   Row: [N = 8]: output channels mapped to 8 Rows
    //   Element: [K = 32]: 32 weight elements stored per Row
    // ]
    weights.to_trf(TrfAddress::Full)
}

/// Performs matmul contraction (main context).
fn matmul<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1], m![1], m![M, K / 16], m![K % 16]>,
    trf: &TrfTensor<bf16, m![1], m![1], m![1], m![N], m![K]>,
) -> AccumulationTensor<'l, T, f32, m![1], m![1], m![1], m![M], m![N]> {
    // TRF mapping: [
    //   Row: [N = 8],
    //   Element: [K = 32]
    // ]
    // Computation mapping: [
    //   Time: [M = 32, K / 16 = 2],
    //   Row: [N = 8],
    //   Packet: [K % 16 = 16]
    // ]
    //
    // reg_read_size: K = 32 bf16 elements = 64 bytes = mac_width
    //   → reg_read_size = 64B (no broadcast, full mac_width read each cycle)
    //   → 64B alignment required: base and strides must be 64-byte aligned
    //
    // Compiler-generated TRF Sequencer configuration:
    //   Entry 0: { size: 32, stride: 0 }  — M (broadcast, not in TRF)

    // 1. M = 32 is broadcast (stride 0): weights reused for each M iteration
    // 2. N = 8 maps to Row: each Row reads from its TRF partition
    // 3. K axis is contracted, and remaining Time: [M], Row: [N]
    //    By using column major, outputs Time: [M], Packet: [N]
    input.align::<m![M], m![K], _, _>(trf)
         .contract::<m![1]>()
         .accumulate::<m![M], m![N]>(AccumulationKind::Interleaved)
}
```

### Small `reg_read_size`

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![M = 32, N = 8, K = 16, L = 2, O = 2];

/// Stores weights into TRF (sub context).
fn store_weights<'l, const T: Tu>(
    weights: CollectTensor<'l, T, bf16, m![1], m![1], m![1], m![N, O], m![K]>,
) -> TrfTensor<bf16, m![1], m![1], m![1], m![N], m![O, K]> {
    // TRF mapping: [
    //   Row: [N = 8]: output channels mapped to 8 Rows
    //   Element: [O = 2, K = 16]: 16 weight elements stored per Row
    // ]
    weights.to_trf(TrfAddress::FirstHalf)
}

/// Performs matmul contraction (main context).
fn matmul<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1], m![1], m![O, M, L], m![K]>,
    trf: &TrfTensor<bf16, m![1], m![1], m![1], m![N], m![O, K]>,
) -> AccumulationTensor<'l, T, f32, m![1], m![1], m![1], m![O, M, L], m![N]> {
    // TRF mapping: [
    //   Row: [N = 8],
    //   Element: [O = 2, K = 16]
    // ]
    // Computation mapping: [
    //   Time: [O = 2, M = 32],
    //   Row: [N = 8],
    //   Packet: [L = 2, K = 16]
    // ]
    //
    // reg_read_size: K=16 bf16 elements = 32 bytes (= mac_width/2)
    //   → reg_read_size = 32B (outer size 2 is broadcast)
    //
    // Compiler-generated TRF Sequencer configuration:
    //   Entry 0: { size: 32, stride: 0 }  — M (broadcast, not in TRF)
    //   Entry 1: { size: 2, stride: 32 }  — O (direct from O)

    // 1. M = 32 is broadcast (stride 0): weights reused for each M iteration
    // 2. N = 8 maps to Row: each Row reads from its TRF partition
    // 3. K axis is contracted, and remaining Time: [O, M], Row: [N], Packet: [L]
    //    By using column major, outputs Time: [O, M, L], Packet: [N]
    input.align::<m![O, M], m![L, K], _, _>(trf)
         .contract::<m![L]>()
         .accumulate::<m![O, M, L], m![N]>(AccumulationKind::Interleaved)
}
```

### TODO: Read (Main) and Write (Sub) to TRF at the same time