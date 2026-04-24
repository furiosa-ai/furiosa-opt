# Sequencer

The Fetch and Commit Engines use sequencers to address DM; this page explains how sequencers work, including their configuration constraints and the failure cases that arise when those constraints are exceeded.
Sequencers convert between tensors in memory and *packet* streams: reading converts a memory buffer into a stream of packets, and writing performs the reverse.
Each packet is a fixed-size chunk delivered each clock cycle; its size is set by the `Packet` mapping dimension.
The DMA Engine chains a read and a write sequencer to move data between HBM/SPM and DM without intermediate buffers.

As a kernel writer, you control the `Time` and `Packet` type parameters, which determine iteration count and packet size; the compiler derives the register configuration and strides.
For performance implications of `Packet` choices, see [Memory Performance](./memory-performance.md).

## Interface

To explain sequencer concepts, we use simplified types that capture the essential structure.
The actual API is introduced in later sections.
The `read` and `write` methods preserve tensor values while transforming between memory layout and stream format.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::marker::PhantomData;
/// A tensor in a linear buffer with mapping `Buf`.
struct BufTensor<D: Scalar, Buf: M> {
    /* ... */
#   _marker: PhantomData<(D, Buf)>,
}

/// A tensor in motion.
/// - `'l`: Lifetime tied to the source buffer, ensuring the stream cannot outlive its data.
/// - `Time`: Temporal mapping (iteration over time).
/// - `Packet`: Spatial mapping (contents of a single packet).
struct StreamTensor<'l, D: Scalar, Time: M, Packet: M> {
    /* ... */
#   _marker: PhantomData<&'l (D, Time, Packet)>,
}

impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
    /// Reads a tensor from a linear buffer into a stream.
    fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> {
        // hardware implementation
#       unimplemented!()
    }

    /// Writes a stream of packets back into a linear buffer.
    fn write<'l, Time: M, Packet: M>(&mut self, stream: StreamTensor<'l, D, Time, Packet>) {
        // hardware implementation
#       unimplemented!()
    }
}
```

## Examples

(The [Configuration](#configuration) section below explains how the compiler derives these configurations from tensor mappings.)

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::marker::PhantomData;
# struct BufTensor<D: Scalar, Buf: M>(PhantomData<(D, Buf)>);
# struct StreamTensor<'l, D: Scalar, Time: M, Packet: M>(PhantomData<&'l (D, Time, Packet)>);
# impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
#     fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> { unimplemented!() }
#     fn write<'l, Time: M, Packet: M>(&mut self, stream: StreamTensor<'l, D, Time, Packet>) { let _ = stream; }
# }
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

/// Broadcasting read: replicate elements using stride 0.
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

## Configuration

The compiler translates input and output tensor mappings into nested-loop configurations that the sequencer hardware executes.
Each configuration has the form `[size_0 : stride_0, size_1 : stride_1, ...] : packet_size`, where each entry's subscript corresponds to its position in the loop nest (0 = outermost), represented by the following Rust type:

```rust
struct Config {
    /// Each entry defines a nested loop level.
    entries: Vec<Entry>,
    /// Number of elements per hardware fetch.
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
Together, entries form nested loops that traverse memory.

### Example: `[N, C, H, W]` ↔ `[W, H, C, N]`

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::marker::PhantomData;
# struct BufTensor<D: Scalar, Buf: M>(PhantomData<(D, Buf)>);
# struct StreamTensor<'l, D: Scalar, Time: M, Packet: M>(PhantomData<&'l (D, Time, Packet)>);
# impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
#     fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> { unimplemented!() }
#     fn write<'l, Time: M, Packet: M>(&mut self, stream: StreamTensor<'l, D, Time, Packet>) { let _ = stream; }
# }
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

The `Time` dimension represents logical iteration steps, not physical clock cycles.
The `Packet` dimension represents logical unit of data processed per `Time`.
The hardware computes `fetch_size` to determine the minimum number of fetch cycles required (see [Fetch Engine](./fetch-engine.md#fetch-sequencer) for the constraints of `fetch_size`).



## Configuration Examples

The compiler automatically derives configurations required to traverse memory.
The following examples illustrate common patterns.


### Rearranging Axes

Rearranging axes changes the traversal order of the tensor.
When `Time` specifies a different axis order than `Buf`, the compiler computes strides to traverse memory in the requested order.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::marker::PhantomData;
# struct BufTensor<D: Scalar, Buf: M>(PhantomData<(D, Buf)>);
# struct StreamTensor<'l, D: Scalar, Time: M, Packet: M>(PhantomData<&'l (D, Time, Packet)>);
# impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
#     fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> { unimplemented!() }
# }
axes![A = 8, B = 8, C = 8];

fn read_rearranging<'l>(
    buf: &'l BufTensor<i8, m![A, B, C # 32]>,  // Buf
) -> StreamTensor<'l, i8, m![B, A], m![C # 16]> {  // Time, Packet
    // Compiler-generated configuration: [
    //   B      ->  8 : 32,
    //   A      ->  8 : 256,
    //   C # 16 -> 16 : 1,
    // ] : 16
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

The maximum `fetch_size` is 16.
The innermost entry `16 : 1` has stride 1, making elements contiguous within the packet.


### Splitting Axes

Splitting axes enables tiling by breaking logical axes into multiple entries.
This is useful for cache efficiency or matching tensor unit buffer sizes.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::marker::PhantomData;
# struct BufTensor<D: Scalar, Buf: M>(PhantomData<(D, Buf)>);
# struct StreamTensor<'l, D: Scalar, Time: M, Packet: M>(PhantomData<&'l (D, Time, Packet)>);
# impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
#     fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> { unimplemented!() }
# }
axes![A = 8, B = 8, C = 4];

fn read_splitting<'l>(
    buf: &'l BufTensor<i8, m![A, B, C # 8]>,  // Buf
) -> StreamTensor<'l, i8, m![A % 2, B % 4, A / 2, B / 4], m![C # 32]> {  // Time, Packet
    // The compiler generates: [
    //   A % 2  ->  2 : 64,
    //   B % 4  ->  4 : 8,
    //   A / 2  ->  4 : 128,
    //   B / 4  ->  2 : 32,
    //   C # 32 -> 32 : 1,
    // ] : 32
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

The maximum `fetch_size` is 32.


### Slicing Axes

Slicing reads only a partial range of indices from the memory layout.
This arises from indexed views that select subsets of the original tensor.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::marker::PhantomData;
# struct BufTensor<D: Scalar, Buf: M>(PhantomData<(D, Buf)>);
# struct StreamTensor<'l, D: Scalar, Time: M, Packet: M>(PhantomData<&'l (D, Time, Packet)>);
# impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
#     fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> { unimplemented!() }
# }
axes![A = 16, B = 8, C = 8];

fn read_slicing<'l>(
    buf: &'l BufTensor<i8, m![A, B, C]>,  // Buf
) -> StreamTensor<'l, i8, m![A / 4, A % 4 = 3, B / 4, B % 4 = 2], m![C]> {  // Time, Packet
    // Compiler-generated configuration: [
    //   A / 4     -> 4 : 256,
    //   A % 4 = 3 -> 3 : 64,
    //   B / 4     -> 2 : 32,
    //   B % 4 = 2 -> 2 : 8,
    //   C         -> 8 : 1,
    // ] : 8
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

The maximum `fetch_size` is 8.


### Broadcasting Axes

Broadcasting replicates tensor elements across multiple packets or time steps.
Stride 0 causes the hardware to repeatedly read from the same memory location.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::marker::PhantomData;
# struct BufTensor<D: Scalar, Buf: M>(PhantomData<(D, Buf)>);
# struct StreamTensor<'l, D: Scalar, Time: M, Packet: M>(PhantomData<&'l (D, Time, Packet)>);
# impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
#     fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> { unimplemented!() }
# }
axes![A = 16, T = 4, P = 4];

fn read_broadcasting<'l>(
    buf: &'l BufTensor<i8, m![A]>,  // Buf
) -> StreamTensor<'l, i8, m![T, A], m![P]> {  // Time, Packet
    // Compiler-generated configuration: [
    //   T ->  4 : 0,   // temporal broadcast
    //   A -> 16 : 1,
    //   P ->  4 : 0,   // spatial broadcast
    // ] : 4
    buf.read()
}
```

Axes not present in `Buf` get stride 0.
The compiler processes `m![T, A, P]` term by term:

| Term | Entry | Stride Source | `Buf` After |
|------|-------|---------------|---------------|
| `T` | `4 : 0` | not in `Buf` (broadcast) | `m![A]` |
| `A` | `16 : 1` | `A` in `m![A]` | `1 # 16` |
| `P` | `4 : 0` | not in `Buf` (broadcast) | `1 # 16` |

The maximum `fetch_size` is 4.
Since `P` has stride 0, the same element is replicated across the packet (spatial broadcast).
Since `T` has stride 0, the same data is repeated across time steps (temporal broadcast).


### Merging Entries

When a transformation produces more than 8 entries, the compiler merges adjacent entries to meet hardware limits.
Adjacent entries `(n1 : s1)` and `(n2 : s2)` merge into `(n1 * n2 : s2)` when physically contiguous: `s1 == n2 * s2`.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::marker::PhantomData;
# struct BufTensor<D: Scalar, Buf: M>(PhantomData<(D, Buf)>);
# struct StreamTensor<'l, D: Scalar, Time: M, Packet: M>(PhantomData<&'l (D, Time, Packet)>);
# impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
#     fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> { unimplemented!() }
# }
axes![N = 8, C = 8, H = 8, W = 32];

fn read_merging<'l>(
    buf: &'l BufTensor<i8, m![N, C, H, W]>,  // Buf
) -> StreamTensor<'l, i8, m![W / 16, H % 2, H / 2, C / 2, C % 2, N / 2, N % 2, W / 8 % 2], m![W % 8]> {  // Time, Packet
    // Initial 9 entries:
    //   W / 16     -> 2 : 16,
    //   H % 2      -> 2 : 32,
    //   H / 2      -> 4 : 64,
    //   C / 2      -> 4 : 512,
    //   C % 2      -> 2 : 256,
    //   N / 2      -> 4 : 4096,
    //   N % 2      -> 2 : 2048,
    //   W / 8 % 2  -> 2 : 8,
    //   W % 8      -> 8 : 1,

    // After merging to 6 entries:
    //   W / 16     -> 2 : 16,
    //   H % 2      -> 2 : 32,
    //   H / 2      -> 4 : 64,
    //   C          -> 8 : 256,    // merged C / 2 and C % 2
    //   N          -> 8 : 2048,   // merged N / 2 and N % 2
    //   W % 16     -> 16 : 1,     // merged W / 8 % 2 and W % 8
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
The last merge combines a temporal entry with the packet entry, increasing the packet size from 8 to 16.

| Term | Entry | Merged Entries |
|------|-------|----------------|
| `W / 16` | `2 : 16` |  |
| `H % 2` | `2 : 32` |  |
| `H / 2` | `4 : 64` |  |
| `C` | `8 : 256` | `C / 2 (4 : 512)`,<br>`C % 2 (2 : 256)` |
| `N` | `8 : 2048` | `N / 2 (4 : 4096)`,<br>`N % 2 (2 : 2048)` |
| `W % 16` | `16 : 1` | `W / 8 % 2 (2 : 8)`,<br>`W % 8 (8 : 1)` |

## Configuration Failures

In TCP, violating any of the following limits causes a compilation error:

- **Entry limit**: Maximum 8 entries; the compiler merges adjacent entries where possible (see [Merging Entries](#merging-entries) in Configuration Examples).
- **Iteration limit**: `size <= 65,536` per entry.
- **Packet size**: Must be 1, 2, 4, 8, 16, or 32 bytes.
- **Packet fetch**: The innermost entry `n : s` must satisfy one of:
  - Contiguous access (adjacent elements): `(s == 0 || s == 1) && n % packet_size == 0`
  - Discrete access (single-element packets): `packet_size == 1`

If merging fails or limits are exceeded, redesign the tensor mapping or split the operation across multiple sequencer calls.
The following examples illustrate common failure cases.


### Insufficient Input

The temporal mapping `Time` attempts to iterate over indices that do not exist in `Buf`.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::marker::PhantomData;
# struct BufTensor<D: Scalar, Buf: M>(PhantomData<(D, Buf)>);
# struct StreamTensor<'l, D: Scalar, Time: M, Packet: M>(PhantomData<&'l (D, Time, Packet)>);
# impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
#     fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> { unimplemented!() }
# }
axes![N = 2048];

fn read_insufficient<'l>(
    buf: &'l BufTensor<i8, m![N % 512]>,  // Buf
) -> StreamTensor<'l, i8, m![N / 512], m![N % 512]> {  // Time, Packet
    buf.read() // Compilation error: insufficient input
}
```

`Time` requires `N / 512`, but `Buf` only contains `N % 512`.
The buffer does not have the indices that the temporal mapping tries to iterate over.


### Incompatible Shapes

The buffer and stream mappings have the same total size but different mathematical structures that no single sequencer configuration can reconcile.

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::marker::PhantomData;
# struct BufTensor<D: Scalar, Buf: M>(PhantomData<(D, Buf)>);
# struct StreamTensor<'l, D: Scalar, Time: M, Packet: M>(PhantomData<&'l (D, Time, Packet)>);
# impl<D: Scalar, Buf: M> BufTensor<D, Buf> {
#     fn read<'l, Time: M, Packet: M>(&'l self) -> StreamTensor<'l, D, Time, Packet> { unimplemented!() }
# }
axes![A = 15];

fn read_incompatible<'l>(
    buf: &'l BufTensor<i8, m![A % 5, A / 5]>,  // Buf
) -> StreamTensor<'l, i8, m![1], m![A % 3, A / 3]> {  // Time, Packet
    buf.read() // Compilation error: incompatible shapes
}
```

Both `Buf` and `Packet` represent 15 elements, but their internal index mappings differ.
The buffer uses a base-5 decomposition (`A % 5, A / 5`) while the packet uses a base-3 decomposition (`A % 3, A / 3`).
These are mathematically incompatible: there is no way to traverse memory in one pattern to produce the other.

The compiler detects this using a "factorizable" concept: it attempts to decompose axes with the same label into `(inner, intersection, outer)` components to find a common representation.
When no such factorization exists, the configuration is rejected.


## Indirect Access

<!-- > **TODO** (jeongmin.park): Detailed content needed for indirect access patterns and implementation. -->

Standard sequencer entries use fixed strides: the memory offset between iterations is constant.
`IndirectLoop` extends this by allowing variable offsets per iteration, enabling gather operations with data-dependent access patterns.

The standard pattern `(limit, stride)` becomes `(limit, [offset0, offset1, ...])`, where each iteration uses a different offset from the provided sequence.
This supports operations like embedding lookups where indices are determined at runtime.
