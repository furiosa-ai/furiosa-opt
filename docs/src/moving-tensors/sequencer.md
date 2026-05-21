# Sequencer

A sequencer reads a memory buffer as a packet stream and writes a packet stream back to memory.
The Fetch and Commit Engines each use one sequencer to address DM.
The DMA Engine chains a read sequencer and a write sequencer to move data among DM, SPM, and HBM without intermediate buffers.

## Interface

`BufTensor` and `StreamTensor` are pedagogical pseudo types that capture the buffer-stream pattern in isolation, so this page can explain sequencer mechanics without dragging in each engine's full type machinery.
The real engine APIs use different types (`DmTensor`, `HbmTensor`, `TuTensor`, …), but every concrete pair maps onto the same `BufTensor` → `StreamTensor` shape illustrated below.

`BufTensor` holds data in some memory mapping `Buf`, and any concrete buffer tensor (DM, SPM, HBM) plays this role.

```rust,ignore
{{#include ../../../furiosa-opt-std/src/tensor/pseudo.rs:buf_tensor_def}}
```

`StreamTensor` is a tensor in flight.
The lifetime `'l` ties the stream to its source buffer so a stream cannot outlive its data.
`Time` is the temporal mapping (iteration over time) and `Packet` is the spatial mapping (contents of a single packet).

```rust,ignore
{{#include ../../../furiosa-opt-std/src/tensor/pseudo.rs:stream_tensor_def}}
```

`read` converts a `BufTensor` into a `StreamTensor` and `write` reverses it, both preserving values.
Each engine's full API adds spatial dimensions (`Chip`, `Cluster`, `Slice`) on top, covered in the engine-specific pages.

```rust,ignore
{{#include ../../../furiosa-opt-std/src/tensor/pseudo.rs:buf_tensor_read_write}}
```

For any `BufTensor`, many valid `Time` and `Packet` combinations exist, each producing a different `StreamTensor`.
Among valid choices, larger `Packet` sizes improve bandwidth utilization, and [Memory Performance](./memory-performance.md) covers the trade-offs in detail.


## Examples

The following examples show common read and write patterns using the core API above.
[Architecture](#architecture) below explains how the compiler derives each pattern's hardware configuration.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{BufTensor, StreamTensor};
axes![A = 8, B = 512, N = 4, C = 3, H = 8, W = 8, T = 4, P = 4];

/// Strided access: read 8×512 tensor as 128 packets of 32 elements.
/// Time = m![A, B / 32] produces 8 * 16 = 128 time steps.
/// Packet = m![B % 32] delivers 32 consecutive elements per packet.
fn strided_read<'l>(
    buf: &'l BufTensor<bf16, m![A, B]>,
) -> StreamTensor<'l, bf16, m![A, B / 32], m![B % 32]> {
    buf.read()  // Automatic type inference
}

/// Strided write: write 128 packets of 32 elements back to 8×512 tensor.
fn strided_write(
    buf: &mut BufTensor<bf16, m![A, B]>,
    stream: StreamTensor<bf16, m![A, B / 32], m![B % 32]>,
) {
    buf.write(stream)
}

/// Axis reordering read: change traversal from [N, C, H, W] to [W, H, C, N].
/// Time = m![W, H, C, N] iterates in reversed axis order.
/// Packet = m![1] delivers single-element packets.
fn axis_reordering_read<'l>(
    buf: &'l BufTensor<bf16, m![N, C, H, W]>,
) -> StreamTensor<'l, bf16, m![W, H, C, N], m![1]> {
    buf.read()
}

/// Axis reordering write: write [W, H, C, N] stream back to [N, C, H, W] buffer.
fn axis_reordering_write(
    buf: &mut BufTensor<bf16, m![N, C, H, W]>,
    stream: StreamTensor<bf16, m![W, H, C, N], m![1]>,
) {
    buf.write(stream)
}

/// Tiling read: break axes into sub-blocks for cache efficiency.
/// Time = m![A % 2, B % 4, A / 2, B / 4] tiles A into 2 × 4, B into 4 × 128 blocks.
/// Packet = m![C # 32] pads C to 32 elements per packet.
fn tiling_read<'l>(
    buf: &'l BufTensor<i8, m![A, B, C # 8]>,
) -> StreamTensor<'l, i8, m![A % 2, B % 4, A / 2, B / 4], m![C # 32]> {
    buf.read()
}

/// Tiling write: write tiled stream back to buffer.
fn tiling_write(
    buf: &mut BufTensor<i8, m![A, B, C # 8]>,
    stream: StreamTensor<i8, m![A % 2, B % 4, A / 2, B / 4], m![C # 32]>,
) {
    buf.write(stream)
}

/// Broadcasting read: replicate elements absent from `Buf`.
/// Time = m![T, A] broadcasts T temporally (same data repeated T times).
/// Packet = m![P] broadcasts P spatially (same element fills packet).
fn broadcasting_read<'l>(
    buf: &'l BufTensor<i8, m![A]>,
) -> StreamTensor<'l, i8, m![T, A], m![P]> {
    buf.read()
}

/// Broadcasting write: write broadcast stream back to buffer.
fn broadcasting_write(
    buf: &mut BufTensor<i8, m![A]>,
    stream: StreamTensor<i8, m![T, A], m![P]>,
) {
    buf.write(stream)
}
```

## Architecture

Each sequencer call is compiled from its input and output tensor mappings into a nested-loop configuration that the sequencer hardware executes.
Each configuration takes the form `[size_0 : stride_0, size_1 : stride_1, ...] : packet_size`, where subscript 0 is the outermost loop:

```rust
struct Config {
    /// Each entry defines a nested loop level.
    entries: Vec<Entry>,
    /// Number of elements per packet.
    packet_size: usize,
}

struct Entry {
    /// Number of iterations for this loop level.
    size: usize,
    /// Memory address distance (in elements) to skip after each iteration.
    stride: isize,
}
```

Each entry encodes one dimension of tensor traversal.
The `size` field determines how many times this loop iterates, while the `stride` field determines the memory offset between consecutive iterations.

### Access Size

`access_size = gcd(Packet::SIZE, contiguous_run)` is the number of elements per hardware access.

Here, `contiguous_run` is the element count of the innermost physically contiguous run of `Config` entries.
A larger `access_size` means fewer accesses per packet.
The following shows how `access_size` is computed from a `Config`:

```rust
# struct Config { entries: Vec<Entry>, packet_size: usize }
# struct Entry { size: usize, stride: isize }
# fn gcd(mut a: usize, mut b: usize) -> usize { while b != 0 { (a, b) = (b, a % b); } a }
impl Config {
    fn contiguous_run(&self) -> usize {
        // Walk pairs from innermost outward; stop at the first non-contiguous pair.
        // Two adjacent entries (n_outer : s_outer) and (n_inner : s_inner)
        // are contiguous when s_outer == n_inner * s_inner.
        let mut contiguous_run = self.entries.last().map_or(1, |e| e.size);
        for w in self.entries.windows(2).rev() {
            if w[0].stride == w[1].size as isize * w[1].stride {
                contiguous_run *= w[0].size;
            } else {
                break;
            }
        }
        contiguous_run
    }

    fn access_size(&self) -> usize {
        gcd(self.packet_size, self.contiguous_run())
    }
}
```

In most cases the packet layout is fully contiguous in DM and `access_size == Packet::SIZE`.
See [Non-Contiguous Packets](#non-contiguous-packets) for a case where `access_size < Packet::SIZE`.

### How It Works

The `Config` for `m![N, C, H, W]` → `m![W, H, C, N]` has one entry per axis in the stream, each with a stride equal to that axis's span in the source buffer.
Since `Packet = m![1]`, `Packet::SIZE = access_size = 1` and the sequencer issues one DM access per loop iteration.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{BufTensor, StreamTensor};
# struct Config {
#     entries: Vec<Entry>,
#     packet_size: usize,
# }
# struct Entry {
#     size: usize,
#     stride: isize,
# }
axes![N = 4, C = 3, H = 8, W = 8];

fn read_nchw_whcn(buf: &BufTensor<bf16, m![N, C, H, W]>) ->
                     StreamTensor<bf16, m![W, H, C, N], m![1]> {
    // Compiler-generated configuration: [8 : 1, 8 : 8, 3 : 64, 4 : 192] : 1
    let config = Config {
        entries: vec![
            Entry { size: 8, stride: 1 },   // W
            Entry { size: 8, stride: 8 },   // H
            Entry { size: 3, stride: 64 },  // C
            Entry { size: 4, stride: 192 }, // N
        ],
        packet_size: 1,
    };

    // The hardware executes the configuration as nested loops:
    for w in 0..8 {
        for h in 0..8 {
            for c in 0..3 {
                for n in 0..4 {
                    // Read each address
                    let addr = 1 * w + 8 * h + 64 * c + 192 * n;
                    // yield buf[addr];
                }
            }
        }
    }

    buf.read()
}

fn write_whcn_nchw(buf: &mut BufTensor<bf16, m![N, C, H, W]>,
                  stream: StreamTensor<bf16, m![W, H, C, N], m![1]>) {
    // The compiler generates an identical config for writing
    // The hardware executes the configuration as nested loops:
    for w in 0..8 {
        for h in 0..8 {
            for c in 0..3 {
                for n in 0..4 {
                    // Write to each address
                    let addr = 1 * w + 8 * h + 64 * c + 192 * n;
                    // buf[addr] = stream.next();
                }
            }
        }
    }
}
```




## Configurations

The following patterns cover most configurations a kernel writer is likely to encounter.

### Transposing Axes

Axes may be transposed so that the stream visits them in a different order than the buffer, and the compiler computes the strides needed to traverse memory in that order.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{BufTensor, StreamTensor};
axes![A = 8, B = 8, C = 8];

fn read_rearranging<'l>(
    buf: &'l BufTensor<i8, m![A, B, C # 32]>,  // Buf
) -> StreamTensor<'l, i8, m![B, A], m![C # 16]> {  // Time, Packet
    buf.read()
}
```

The compiler generates configuration entries by processing the combined mapping `m![B, A, C # 16]` term by term, transforming `Buf` along the way.
For each term, the entry size equals the term size, and the stride equals the volume that term occupies within the current `Buf`.
After processing a term, `Buf` is updated to reflect that the axis has been consumed:

| Term | Entry | Stride Source | `Buf` After |
|------|-------|---------------|---------------|
| `B` | `8 : 32` | `m![C # 32]::SIZE` | `m![A, 1 # 8, C # 32]` |
| `A` | `8 : 256` | `m![1 # 8, C # 32]::SIZE` | `m![1 # 64, C # 32]` |
| `C # 16` | `16 : 1` | contiguous (`Packet` dimension) | `1 # 2048` |

`Packet::SIZE = access_size = 16`.
The innermost entry `16 : 1` is contiguous, so the hardware transfers the full packet in one access.


### Splitting Axes

Tiling breaks a logical axis into sub-blocks for cache efficiency or to match tensor unit buffer sizes, and the compiler achieves this by splitting the axis into multiple entries.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{BufTensor, StreamTensor};
axes![A = 8, B = 8, C = 4];

fn read_splitting<'l>(
    buf: &'l BufTensor<i8, m![A, B, C # 8]>,  // Buf
) -> StreamTensor<'l, i8, m![A % 2, B % 4, A / 2, B / 4], m![C # 32]> {  // Time, Packet
    buf.read()
}
```

Expressions like `A % 2` and `A / 2` split axis `A` into separate entries.
The compiler processes `m![A % 2, B % 4, A / 2, B / 4, C # 32]` term by term:

| Term | Entry | Stride Source | `Buf` After |
|------|-------|---------------|---------------|
| `A % 2` | `2 : 64` | `m![B, C # 8]::SIZE` | `m![A / 2, 1 # 2, B, C # 8]` |
| `B % 4` | `4 : 8` | `m![C # 8]::SIZE` | `m![A / 2, 1 # 2, B / 4, 1 # 4, C # 8]` |
| `A / 2` | `4 : 128` | `m![1 # 2, B / 4, 1 # 4, C # 8]::SIZE` | `m![1 # 8, B / 4, 1 # 4, C # 8]` |
| `B / 4` | `2 : 32` | `m![1 # 4, C # 8]::SIZE` | `m![1 # 64, C # 8]` |
| `C # 32` | `32 : 1` | contiguous (`Packet` dimension) | `1 # 512` |

`Packet::SIZE = access_size = 32`.


### Slicing Axes

Slicing reads only a partial range of indices from the memory layout, a condition that arises when an indexed view selects a subset of the original tensor.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{BufTensor, StreamTensor};
axes![A = 16, B = 8, C = 8];

fn read_slicing<'l>(
    buf: &'l BufTensor<i8, m![A, B, C]>,  // Buf
) -> StreamTensor<'l, i8, m![A / 4, A % 4 = 3, B / 4, B % 4 = 2], m![C]> {  // Time, Packet
    buf.read()
}
```

The `= 3` notation limits `A % 4` to only 3 iterations instead of 4, restricting the hardware to a sub-region of the tensor.
The compiler processes `m![A / 4, A % 4 = 3, B / 4, B % 4 = 2, C]` term by term:

| Term | Entry | Stride Source | `Buf` After |
|------|-------|---------------|---------------|
| `A / 4` | `4 : 256` | `m![A % 4, B, C]::SIZE` | `m![1 # 4, A % 4, B, C]` |
| `A % 4 = 3` | `3 : 64` | `m![B, C]::SIZE` (sliced to 3) | `m![1 # 16, B, C]` |
| `B / 4` | `2 : 32` | `m![B % 4, C]::SIZE` | `m![1 # 32, B % 4, C]` |
| `B % 4 = 2` | `2 : 8` | `m![C]::SIZE` (sliced to 2) | `m![1 # 128, C]` |
| `C` | `8 : 1` | contiguous (`Packet` dimension) | `1 # 1024` |

`Packet::SIZE = access_size = 8`.


### Broadcasting Axes

Broadcasting replicates an element across multiple packets or time steps when the stream visits axes that `Buf` does not carry.
Any axis (or partial-axis fragment like `N / 512`) present in `Time` or `Packet` but absent from `Buf` becomes a broadcast entry, shown as `: 0` in the stride table (the hardware revisits the same address on each iteration).

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{BufTensor, StreamTensor};
axes![A = 16, T = 4, P = 4];

fn read_broadcasting<'l>(
    buf: &'l BufTensor<i8, m![A]>,  // Buf
) -> StreamTensor<'l, i8, m![T, A], m![P]> {  // Time, Packet
    buf.read()
}
```

The compiler processes `m![T, A, P]` term by term:

| Term | Entry | Stride Source | `Buf` After |
|------|-------|---------------|---------------|
| `T` | `4 : 0` | not in `Buf` (broadcast) | `m![A]` |
| `A` | `16 : 1` | `A` in `m![A]` | `1 # 16` |
| `P` | `4 : 0` | not in `Buf` (broadcast) | `1 # 16` |

`Packet::SIZE = access_size = 4`.
`P` is broadcast, so the same element is replicated across the packet (spatial broadcast).
`T` is broadcast, so the same data is repeated across time steps (temporal broadcast).

The same rule applies when `Time` or `Packet` references a fragment of an axis that `Buf` does not carry.
For example, a buffer of `m![N % 512]` read as `StreamTensor<m![N / 512], m![N % 512]>` broadcasts on the `N / 512` time entry: the buffer's 512 elements are reused across each of the `N / 512` outer iterations.


### Merging Entries

The hardware supports at most 8 entries per configuration, so when a transformation produces more, the compiler merges adjacent entries to satisfy that limit.
Adjacent entries `(n1 : s1)` and `(n2 : s2)` merge into `(n1 * n2 : s2)` when physically contiguous: `s1 == n2 * s2`.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{BufTensor, StreamTensor};
axes![N = 8, C = 8, H = 8, W = 32];

fn read_merging<'l>(
    buf: &'l BufTensor<i8, m![N, C, H, W]>,  // Buf
) -> StreamTensor<'l, i8, m![W / 16, H % 2, H / 2, C / 2, C % 2, N / 2, N % 2, W / 8 % 2], m![W % 8]> {  // Time, Packet
    buf.read()
}
```

The compiler processes `m![W / 16, H % 2, H / 2, C / 2, C % 2, N / 2, N % 2, W / 8 % 2, W % 8]` term by term, producing 9 initial entries:

| Term | Entry | Stride Source |
|------|-------|---------------|
| `W / 16` | `2 : 16` | `m![W % 16]::SIZE` |
| `H % 2` | `2 : 32` | `m![W]::SIZE` |
| `H / 2` | `4 : 64` | `m![H % 2, W]::SIZE` |
| `C / 2` | `4 : 512` | `m![C % 2, H, W]::SIZE` |
| `C % 2` | `2 : 256` | `m![H, W]::SIZE` |
| `N / 2` | `4 : 4096` | `m![N % 2, C, H, W]::SIZE` |
| `N % 2` | `2 : 2048` | `m![C, H, W]::SIZE` |
| `W / 8 % 2` | `2 : 8` | `m![W % 8]::SIZE` |
| `W % 8` | `8 : 1` | contiguous (packet dimension) |

Since 9 entries exceed the hardware limit of 8, the compiler merges contiguous pairs where `s1 == n2 * s2`.
The entries for `H % 2 -> (2 : 32)` and `H / 2 -> (4 : 64)` are not merged because they are not physically contiguous (\\(s_1 \neq n_2 \times s_2 \iff 32 \neq 4 \times 64\\)).
The final configuration has 6 entries.
The last merge crosses the time/packet boundary: `W/8%2 (2:8)` and `W%8 (8:1)` merge into `W%16 (16:1)`.

| Term | Entry | Merged Entries |
|------|-------|----------------|
| `W / 16` | `2 : 16` |  |
| `H % 2` | `2 : 32` |  |
| `H / 2` | `4 : 64` |  |
| `C` | `8 : 256` | `C / 2 (4 : 512)`,<br>`C % 2 (2 : 256)` |
| `N` | `8 : 2048` | `N / 2 (4 : 4096)`,<br>`N % 2 (2 : 2048)` |
| `W % 16` | `16 : 1` | `W / 8 % 2 (2 : 8)`,<br>`W % 8 (8 : 1)` |

`Packet::SIZE = access_size = 8`.


### Non-Contiguous Packets

When the DM layout has stride discontinuities within the packet span, `access_size < Packet::SIZE` and the hardware issues one access per contiguous sub-block rather than one per packet.
The example below writes a packet of 32 elements (`m![A, B]`) to a buffer where each B row is padded to 16 slots in DM.
A's stride is 16 rather than 8, so the packet span is not contiguous and the hardware issues 4 accesses instead of 1:

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{BufTensor, StreamTensor};
axes![A = 4, B = 8];

fn write_padded(
    buf: &mut BufTensor<i8, m![A, B # 16]>,
    stream: StreamTensor<i8, m![1], m![A, B]>,
) {
    // Compiler-generated configuration: [
    //   A -> 4 : 16,   (16 != 8 * 1, NOT contiguous — padding gap after each B row)
    //   B -> 8 : 1,    (packet sub-block, contiguous)
    // ] : 32
    buf.write(stream)
}
```

`Packet::SIZE = 32`, `contiguous_run = 8`, `access_size = 8`.

Non-contiguous strides also arise when `Packet` contains non-adjacent axes from the source layout.
The example below reads the same `m![N, C, H, W]` buffer with two different `Packet` choices.
Placing only the innermost axis `W` in `Packet` gives `access_size = Packet::SIZE = 8`, one access per packet.
Placing `m![N, H, W]` in `Packet` skips `C`, so N's stride in source (96) does not equal H×W (32): `contiguous_run = 32`, `access_size = 32`, and the hardware issues 4 accesses per packet instead of 1.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{BufTensor, StreamTensor};
axes![N = 4, C = 3, H = 4, W = 8];

// Compiler-generated configuration: [
//   N -> 4 : 96,   (96 == 3 × 32, contiguous)
//   C -> 3 : 32,   (32 == 4 × 8,  contiguous)
//   H -> 4 : 8,    (8  == 8 × 1,  contiguous)
//   W -> 8 : 1,    (packet dimension)
// ] : 8
// contiguous_run = 8 (W); ×4 (H): 8==8×1 ✓; ×3 (C): 32==4×8 ✓; ×4 (N): 96==3×32 ✓; all axes contiguous
// access_size = gcd(packet_size, contiguous_run) = packet_size = 8
fn read_contiguous<'l>(
    buf: &'l BufTensor<i8, m![N, C, H, W]>,
) -> StreamTensor<'l, i8, m![N, C, H], m![W]> {
    buf.read()
}

// Compiler-generated configuration: [
//   C -> 3 : 32,   (time dimension)
//   N -> 4 : 96,   (96 != 4 × 8 = 32, NOT contiguous — C axis interspersed)
//   H -> 4 : 8,    (8  == 8 × 1, contiguous)
//   W -> 8 : 1,    (packet dimension)
// ] : 128
// contiguous_run = 8 (W); ×4 (H): 8==8×1 ✓ → 32; ×4 (N): 96!=4×8 ✗ stop → 32
// access_size = gcd(128, 32) = 32; hardware issues 128/32 = 4 accesses per packet
fn read_non_contiguous<'l>(
    buf: &'l BufTensor<i8, m![N, C, H, W]>,
) -> StreamTensor<'l, i8, m![C], m![N, H, W]> {
    buf.read()
}
```

## Constraints

In RNGD, exceeding any of the following hardware limits causes a compilation error:

- **Entry limit**: Maximum 8 entries, so the compiler merges adjacent entries where possible (see [Merging Entries](#merging-entries) in Configurations).
- **Iteration limit**: `size <= 65,536` per entry.
- **Packet size**: Must be 1, 2, 4, 8, 16, or 32 bytes.
- **Packet fetch**: The innermost entry `n : s` must satisfy one of:
  - Contiguous access (adjacent elements): `(s == 0 || s == 1) && n % packet_size == 0`
  - Discrete access (single-element packets): `packet_size == 1`

If merging fails or limits are exceeded, redesign the tensor mapping or split the operation across multiple sequencer calls.


### Compatible Axis Decompositions

Each axis named in both `Buf` and the stream must use the same decomposition.
The compiler walks the stream term by term and consumes axes from `Buf` (see [Architecture](#architecture)).
When `Buf` splits an axis one way and the stream splits it another with no common refinement, no traversal order works and the configuration is rejected, even when both sides have the same total element count.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use furiosa_opt_std::pseudo::{BufTensor, StreamTensor};
axes![A = 15];

fn read_incompatible<'l>(
    buf: &'l BufTensor<i8, m![A % 5, A / 5]>,  // Buf
) -> StreamTensor<'l, i8, m![1], m![A % 3, A / 3]> {  // Time, Packet
    buf.read() // Compilation error: incompatible decomposition
}
```

`Buf` decomposes `A` as `5 × 3` while the stream decomposes it as `3 × 5`.
Since `gcd(5, 3) = 1`, neither decomposition refines the other: the compiler cannot consume `A % 3` from a `Buf` that has already committed to a 5-block split.


## Indirect Access


All entries above use fixed strides: the memory offset between iterations is constant.
`IndirectLoop` extends this by allowing variable offsets per iteration, enabling gather operations with data-dependent access patterns.

The standard pattern `(limit, stride)` becomes `(limit, [offset0, offset1, ...])`, where each iteration uses a different offset from the provided sequence.
This supports operations like embedding lookups where indices are determined at runtime.
