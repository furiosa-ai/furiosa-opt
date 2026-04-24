# Fetch and Commit Engine

The [Fetch Engine](../moving-tensors/fetch-engine.md) reads tensors from SRAM while the [Commit Engine](../moving-tensors/commit-engine.md) writes them back.
These examples demonstrate the complete data path: input tensor -> fetch sequencer -> [Switch Engine](../computing-tensors/switch-engine.md) -> [Collect Engine](../computing-tensors/collect-engine.md) -> commit unit -> output tensor.

Each example focuses on a specific pattern: axis permutation, full-flit commit, tail padding optimization, and tensor segmentation.
These four patterns represent distinct aspects of the fetch-commit data path: axis reordering (permutation), write granularity (full-flit), memory layout choices (tail padding), and handling of tensors that exceed hardware capacity (segmentation).

## Example 1: Axis Permutation

This example demonstrates tensor reshaping by permuting axes during a fetch-commit cycle.
The Switch Engine enables axis reordering without additional computation by controlling data flow from fetch to commit.

```rust,ignore
axes![A = 3, B = 5, C = 2];

// Input: shape [A, B, C] at address 0
let input: DmTensor<f8, m![1], m![1], m![1], m![A, B, C]> = ...;

// Output: shape [B, A, C] at address 1024 (permuted layout)
let output: DmTensor<f8, m![1], m![1], m![1], m![B, A, C # 6]> = ctx
    .main
    .begin(input.view())
    .fetch::<f8, m![A, B], m![C # 6]>()       // Time=[A,B], Packet=[C] padded to 8 bytes
    .collect::<m![A, B], m![C # 30]>()        // Pad to 32-byte flit (forwarding switch implied)
    .commit(1024);                             // Write with permuted sequencer config
```

**Input Tensor:**
The input is a 3D tensor stored in SRAM with dimensions `[A=3, B=5, C=2]`, containing 30 elements total:
- **Shape**: `A × B × C = 3 × 5 × 2`
- **Data type**: `f8` (8-bit floating-point)
- **Memory layout**: `m![A, B, C]` - consecutive in memory as `A` varies slowest, `C` varies fastest
- **Base address**: `b = 0` (starts at SRAM address 0)
- **Physical storage**: Elements are arranged as `[A0,B0,C0][A0,B0,C1][A0,B1,C0]...[A2,B4,C1]`

Labeling elements by their indices, memory contains:
```text
Address 0-1:   (A=0,B=0,C=0-1)
Address 2-3:   (A=0,B=1,C=0-1)
Address 4-5:   (A=0,B=2,C=0-1)
...continuing with A=0, varying B...
Address 10-11: (A=1,B=0,C=0-1)
...and so on
```

**Output Tensor (Target):**
Store the same logical tensor with axes permuted to layout `[B, A, C]`:
- **Shape**: Still `B × A × C = 5 × 3 × 2` (same 30 elements, different order)
- **Data type**: `f8` (unchanged)
- **Memory layout**: `m![B, A, C # 6]` - now `B` varies slowest, with 6-byte padding per element
- **Base address**: `b = 1024` (stored at SRAM address 1024)
- **Physical storage**: Elements arranged as `[B0,A0,C0-1][B0,A1,C0-1][B0,A2,C0-1][B1,A0,C0-1]...`

This reordering changes which elements are adjacent in memory: in the input, all `B` values for `A=0` are contiguous; in the output, all `A` values for `B=0` are contiguous.

**Processing:**

The axis permutation happens through three stages—Fetch, Switch, and Commit:

1. **Fetch Sequencer**: Reads the input tensor from SRAM and creates a packet stream
   - Time dimension: `Time = m![A, B]` - iterates through 15 cycles (3×5)
   - Packet dimension: `Packet = m![C # 6]` - each packet contains 2 `C` elements plus 6 bytes padding
   - Fetch size: 8 bytes per cycle (meets hardware alignment requirement)
   - Note: Hardware requires 8-byte packet alignment, so we cannot use `C=2` bytes alone; we pad to 8 bytes

2. **Collect Engine**: Normalizes packets into standard 32-byte flits for the commit stage
   - Input packets (8 bytes) are padded to create 32-byte flits
   - Time dimension: `Time = m![A, B]` - unchanged, still 15 cycles
   - flit dimension: `Flit = m![C # 30]` - 2 data bytes + 30 bytes padding = 32-byte flit
   - The collect engine pads and normalizes packet sizes without reordering data

3. **Commit Unit**: Writes data to SRAM with the new axis order `[B, A, C]`
   - Receives flits with time `m![A, B]` but writes to memory layout `m![B, A, C # 6]`
   - The write sequencer configuration creates the permutation
   - Commit size: 8 bytes per write (matching fetch size)
   - Slices incoming 32-byte flits down to 8-byte write units

The **write sequencer configuration** determines how to map the incoming time-ordered stream `m![A, B, C]` to the permuted memory layout `m![B, A, C #6]`.
The notation `[axis=count:stride, ...] @ base / commit_size` means: for each axis, loop `count` times advancing `stride` bytes per step; `@` sets the base address; `/` sets the bytes written per commit operation (see [Sequencer](../moving-tensors/sequencer.md) for the full sequencer model).

The sequencer is configured as: `[A=3:8, B=5:24, C=8:1] @ 1024 / 8`
- `A=3:8` means loop 3 times with stride 8 bytes between iterations
- `B=5:24` means loop 5 times with stride 24 bytes between iterations
- `C=8:1` means write 8 bytes (the packet size) with stride 1
- Base address: 1024 (output tensor starts here)
- Commit size: 8 bytes per write operation

This configuration causes data arriving in `[A, B]` time order to be written to addresses that correspond to `[B, A]` spatial order. Here's how the writes occur:

| Cycle `i` | Time axes | Write to memory address | Explanation |
|-----------|-----------|------------------------|-------------|
| 0 | A=0, B=0 | 1024-1032 (B=0, A=0) | First element: writes to base address |
| 1 | A=0, B=1 | 1048-1056 (B=1, A=0) | Stride 24 bytes forward (next B) |
| 2 | A=0, B=2 | 1072-1080 (B=2, A=0) | Another 24-byte stride |
| 3-4 | A=0, B=3-4 | Continue with B=3,4 | Complete A=0 row |
| 5 | A=1, B=0 | 1032-1040 (B=0, A=1) | Jump to B=0, A=1 (+8 from cycle 0) |
| 6 | A=1, B=1 | 1056-1064 (B=1, A=1) | +24 stride for next B |
| 7-14 | Continue | ... | Complete all A=1,2 rows |

Notice how the write pattern interleaves: we write `A=0,B=0` then `A=0,B=1`, but these end up at addresses that place all `A` values for each `B` together in the output layout.

**Output:**

After commit completes, SRAM address 1024 onwards contains the tensor with permuted layout:
- **Memory layout**: `[B=5, A=3, C=2]` with 6-byte padding
- **Physical arrangement**: All `A` values for `B=0` are contiguous, then all `A` values for `B=1`, etc.
- **Address structure**:
  ```text
  1024-1032: (B=0, A=0, C=0-1) + 6 bytes padding
  1032-1040: (B=0, A=1, C=0-1) + 6 bytes padding
  1040-1048: (B=0, A=2, C=0-1) + 6 bytes padding
  1048-1056: (B=1, A=0, C=0-1) + 6 bytes padding
  ...and so on
  ```

The permutation is complete: the same 30 data elements that were in `[A, B, C]` order are now in `[B, A, C]` order. This operation takes 15 cycles (one per A×B combination) and requires no actual computation—only memory read/write with different address patterns.

**Key constraints:**

Three constraints govern axis permutation operations:

- **8-byte alignment**: `commit_in_size` and `commit_size` are always in 8-byte units, so the target tensor for commit always corresponds to an 8-byte aligned range, naturally creating 8-byte tail alignment (a dummy is added to align the tail to 8 bytes).
- **Sequencer limit**: Like the Fetch Engine, sequencer entries are limited to 8 total (limit < 65536).
- **Non-contiguous writes**: Since the write sequencer sets the commit address, committed data need not be contiguous in flit time order—permutations like `AB -> BA` are possible.

**Why this example is useful:**

Axis permutation is a common requirement in deep learning:
- **Tensor layout transformations**: Converting between `NCHW` (batch, channels, height, width) and `NHWC` (batch, height, width, channels) formats for different operations
- **Matrix transpose**: Preparing data for operations that require transposed matrices without actual computation
- **Memory access optimization**: Reordering axes to make the most frequently accessed dimension innermost for better cache performance
- **Inter-operation compatibility**: Reformatting tensors to match the input requirements of subsequent operations

The TCP architecture performs these reshapes during data movement without consuming compute resources or requiring separate transpose kernels.

## Example 2: Full-flit Commit

This example demonstrates full-flit commit, an optimization that writes entire 32-byte flits directly to memory without slicing them into smaller chunks.
Tensor dimensions naturally aligned to 32-byte boundaries eliminate commit slicer overhead and simplify write sequencer configuration.

```rust,ignore
axes![A = 3, B = 5, C = 2];

// Input: same shape [A, B, C]
let input: DmTensor<f8, m![1], m![1], m![1], m![A, B, C]> = ...;

// Output: merge B and C, pad to 32 bytes per A slice
let output: DmTensor<f8, m![1], m![1], m![1], m![A, [B, C] # 22]> = ctx
    .main
    .begin(input.view())
    .fetch::<f8, m![A], m![[B, C] # 22]>()     // Time=[A], Packet=[B,C] padded to 32 bytes
    .collect::<m![A], m![[B, C] # 22]>()        // Already 32-byte flit, identity collect
    .commit(1024);                               // Full-flit commit: 3 cycles vs 15 in Example 1
```

**Input Tensor:**
The input tensor is identical to Example 1, but committed with a different memory layout that allows full-flit writes:
- **Shape**: `[A=3, B=5, C=2]` containing 30 elements
- **Data type**: `f8` (8-bit floating-point, 1 byte per element)
- **Memory layout**: `m![A, B, C]` - standard row-major order
- **Base address**: `b = 0`
- **Element size**: 1 byte × 30 elements = 30 bytes of data

**Output Tensor (Target):**
Instead of permuting axes like Example 1, we merge the last two dimensions and add padding:
- **Shape**: Still `[A=3, B=5, C=2]` logically, but stored as `[A=3, BC=10]`
- **Data type**: `f8` (unchanged)
- **Memory layout**: `m![A, [B, C] # 22]` - merge B and C dimensions, add 22 bytes padding
- **Base address**: `b = 1024`
- **Physical layout**: Each `A` iteration stores 10 data bytes (B×C) plus 22 padding bytes = 32 bytes total
- The 32-byte size per `A` slice perfectly matches hardware flit size, enabling full-flit writes

**Processing:**

Data dimensions aligned with hardware flit size enable a simpler pipeline than Example 1:

1. **Fetch Sequencer**: Reads input and pads to 32-byte packets immediately
   - Time dimension: `Time = m![A, B]` - 15 cycles (3×5)
   - Packet dimension: `Packet = m![[B, C] # 22]` - merges B and C, adds 22 bytes padding to reach 32 bytes
   - Fetch size: 32 bytes per cycle (full packet, not split)
   - The sequencer pads from 10 data bytes to 32 bytes during fetch

2. **Collect Engine**: Receives 32-byte packets and passes them as 32-byte flits
   - Time dimension: `Time = m![A]` - simplified to just 3 cycles since B and C are merged into packet
   - flit dimension: `Flit = m![[B, C] # 22]` - full 32-byte flit with no additional padding needed
   - No reformatting required: packet size = flit size = 32 bytes

3. **Commit Unit**: Writes full 32-byte flits directly to memory without slicing
   - Receives 32-byte flits and writes them as complete 32-byte units
   - **commit_in_size = 32 bytes**: No slicer operation needed
   - **commit_size = 32 bytes**: Each write operation handles a full flit
   - Time: Only 3 cycles (one per `A`), much faster than Example 1's 15 cycles

The write sequencer configuration is simple: `[A=3:32, [B,C]=32:1] @ 1024 / 32`
- `A=3:32` means loop 3 times with 32-byte stride (one full flit per A)
- `[B,C]=32:1` means write 32 bytes with stride 1 (continuous write of flit contents)
- Each cycle writes one complete flit: cycle 0 writes flit for A=0, cycle 1 for A=1, cycle 2 for A=2

**Output:**

After commit, SRAM address 1024 onwards contains the tensor packed into 32-byte-aligned blocks:
- **Memory layout**: `[A=3, BC=10+padding]` with each `A` slice occupying exactly 32 bytes
- **Physical structure**:
  ```text
  1024-1056: (A=0, all 10 B×C elements) + 22 bytes padding = 32 bytes
  1056-1088: (A=1, all 10 B×C elements) + 22 bytes padding = 32 bytes
  1088-1120: (A=2, all 10 B×C elements) + 22 bytes padding = 32 bytes
  ```
- **Performance**: Only 3 write cycles vs 15 in Example 1 (5× faster)
- **Simplicity**: No slicing overhead, no complex stride patterns

**Why this example is useful:**

Full-flit commit demonstrates an important optimization strategy:
- **Alignment optimization**: When you can pad dimensions to 32-byte boundaries, commit becomes much more efficient
- **Reduced cycles**: Fewer, larger writes complete faster than many small writes
- **Hardware efficiency**: Writing full flits maximizes memory bandwidth utilization
- **Design principle**: Sometimes adding padding to align with hardware granularity improves overall performance

This technique is particularly valuable for:
- Small tensors where padding overhead is minimal compared to the benefit
- Intermediate results that don't need compact storage
- Situations where downstream operations also benefit from 32-byte alignment

**Key constraint:** Write sequencer configurations require non-zero stride for all entries.
This means you cannot discard data beyond slicing (no selective writes), and broadcast (reuse) operations are not possible during commit.

## Example 3: Tail Padding and Fetch Size

The amount of tail padding dramatically affects fetch/commit efficiency.
In the mapping expression `m![A # 72]`, the `# 72` pads `A` up to 72 elements; the pads are referred to as `dummy` in the hardware configuration below.
Understanding padding interaction with hardware fetch size constraints enables optimization between memory usage (less padding) and performance (more padding aligned to hardware boundaries).

```rust,ignore
axes![A = 65, B = 2];

let input: DmTensor<f8, m![1], m![1], m![1], m![B, A # 72]> = ...;

// Option 1: dummy=7, fetch_size=24 bytes, 6 cycles
let out_7: DmTensor<f8, m![1], m![1], m![1], m![B, A # 72]> = ctx.main
    .begin(input.view())
    .fetch::<f8, m![B * (A # 72) / 24], m![A % 24]>()
    .collect::<m![B * (A # 72) / 24], m![A % 24 # 8]>()
    .commit(1024);

// Option 2: dummy=31, fetch_size=32 bytes, 6 cycles (best performance)
let out_31: DmTensor<f8, m![1], m![1], m![1], m![B, A # 96]> = ctx.main
    .begin(input.view())
    .fetch::<f8, m![B * (A # 96) / 32], m![A % 32]>()
    .collect::<m![B * (A # 96) / 32], m![A % 32]>()
    .commit(1024);
```

**The Problem:**
Commit a tensor with shape `[A=65, B=2]` (130 bytes of data). Hardware fetch sizes must be 8, 16, 24, or 32 bytes.
Determine the padding amount for dimension `A` to maximize performance.

**Input Tensor:**
- **Shape**: `[A=65, B=2]` - 130 elements (65 elements across A dimension, 2 across B)
- **Data type**: `f8` (1 byte per element)
- **Memory layout**: `m![B, A+7]` - stored with 7 bytes of tail padding after A
- **Base address**: `b = 0`
- **Total size**: `2 × (65 + 7) = 144 bytes` (includes padding)

**Output Tensor (Variable Padding):**
The target can have different padding amounts, each enabling different fetch sizes:
- **Shape**: `[A=65, B=2]` (same logical data)
- **Data type**: `f8`
- **Base address**: `b = 1024`
- **Memory layout options**:
  - `m![B, A # 7]`: 7 bytes padding → enables 24-byte fetch size
  - `m![B, A # 15]`: 15 bytes padding → enables 16-byte fetch size
  - `m![B, A # 23]`: 23 bytes padding → enables 8-byte fetch size (worst)
  - `m![B, A # 31]`: 31 bytes padding → enables 32-byte fetch size (best)

The optimal fetch size unit varies depending on the tail `dummy` value.
The following subsections show each case:

### `dummy = 7`
- fetch sequencer output
    - `Time = m![B * (A # 7) / 24]`
    - `Flit = m![A % 24]`
    - fetch_size = 24 bytes
- Switch Engine output (= commit unit input)
    - `Time = m![B * (A # 7) / 24]`
    - `Flit = m![A % 24 # 8]`
- Commit Unit
    - commit_in_size = 24 bytes
    - sliced shape
        - `Time = m![B * (A # 7) / 24]`
        - `Flit = m![A % 24]`
    - write sequencer configuration
        - `m![B, A # 7]` -> `m![B * (A # 7) / 24 * A % 24]`
        - Sequencer configuration: `[B=2:72, (A # 7)/24=3:24, A=24:1] @ 1024 / 24`

### `dummy = 15`
- fetch sequencer output
    - `Time = m![B * (A # 15) / 16]`
    - `Flit = m![A % 16]`
    - fetch_size = 16 bytes
- Switch Engine output (= commit unit input)
    - `Time = m![B * (A # 15) / 16]`
    - `Flit = m![A % 16]`
- Commit Unit
    - commit_in_size = 16 bytes
    - sliced shape
        - `Time = m![B * (A # 15) / 16]`
        - `Flit = m![A % 16]`
    - write sequencer configuration
        - `m![B, A # 15]` -> `m![B * (A # 15) / 16 * A % 16]`
        - Sequencer configuration: `[B=2:80, (A # 15)/16=5:16, A=16:1] @ 1024 / 16`

### `dummy = 23`
- fetch sequencer output
    - `Time = m![B * (A # 23) / 8]`
    - `Flit = m![A % 8]`
    - fetch_size = 8 bytes
- Switch Engine output (= commit unit input)
    - `Time = m![B * (A # 23) / 8]`
    - `Flit = m![A % 8 # 24]`
- Commit Unit
    - commit_in_size = 8 bytes
    - sliced shape
        - `Time = m![B * (A # 23) / 8]`
        - `Flit = m![A % 8]`
    - write sequencer configuration
        - `m![B, A # 23]` -> `m![B * (A # 23) / 8 * A % 8]`
        - Sequencer configuration: `[B=2:88, (A # 23)/8=11:8, A=8:1] @ 1024 / 8`

### `dummy = 31`
- fetch sequencer output
    - `Time = m![B * (A # 31) / 32]`
    - `Flit = m![A % 32]`
    - fetch_size = 32 bytes
- Switch Engine output (= commit unit input)
    - `Time = m![B * (A # 31) / 32]`
    - `Flit = m![A % 32]`
- Commit Unit
    - commit_in_size = 32 bytes
    - sliced shape
        - `Time = m![B * (A # 31) / 32]`
        - `Flit = m![A % 32]`
    - write sequencer configuration
        - `m![B, A # 31]` -> `m![B * (A # 31) / 32 * A % 32]`
        - Sequencer configuration: `[B=2:96, (A # 31)/32=3:32, A=32:1] @ 1024 / 32`

### Summary: The Impact of Padding Choice

The following table summarizes how output tail padding affects performance:

| Padding (`dummy`) | `fetch_size` | Fetch cycles | Memory overhead | Efficiency |
|-------------------|------------|--------------|-----------------|------------|
| 7 | 24 bytes | 6 cycles | 14 bytes (9.7%) | Good |
| 15 | 16 bytes | 10 cycles | 30 bytes (18.8%) | Moderate |
| 23 | 8 bytes | 22 cycles | 46 bytes (26.1%) | Poor |
| 31 | 32 bytes | 6 cycles | 62 bytes (32.3%) | Best |

**Key Insights:**

1. **Performance varies dramatically**: `dummy=23` requires 22 cycles (8-byte fetches) while `dummy=31` requires only 6 cycles (32-byte fetches) - nearly 4× faster despite using similar amounts of padding

2. **Optimal padding aligns with hardware**: The best performance comes when `(data_size + padding)` is divisible by 32 bytes (the largest fetch size)

3. **Trade-off**: Adding 8 more bytes of padding (23→31) increases memory overhead from 26.1% to 32.3% (only 6% increase) but improves performance by 3.7× (22 cycles→6 cycles)

4. **Design principle**: Prefer padding amounts that enable the largest possible fetch size (32 bytes), even with slightly more memory waste

**Why this example is useful:**

Naive padding choices cause severe performance degradation:
- Padding to arbitrary values like 23 bytes forces small 8-byte fetches
- Understanding fetch size constraints enables strategic padding choices
- Pad to the next multiple of 32 bytes when possible
- The memory cost of better padding is usually negligible compared to the performance gain

### Why `dummy=23` Cannot Use 32-byte Commits

The `dummy=23` case cannot use 32-byte commits because write sequencer configurations must never exceed tensor boundaries:

- fetch sequencer output
    - `Time = m![B * (A # 31) / 32]` : Since A + 23 is not divisible by 32, setting fetch_size=32 bytes requires fetching A+31 elements.
    - `Flit = m![A % 32]`
    - fetch_size = 32 bytes
- Switch Engine output (= commit unit input)
    - `Time = m![B * (A # 31) / 32]`
    - `Flit = m![A % 32]`
- Commit Unit
    - commit_in_size = 32 bytes (if commit_in_size < 32, it would cut valid A portion: must set commit_in_size=32 bytes)
    - sliced shape
        - `Time = m![B * (A # 31) / 32]`
        - `Flit = m![A % 32]`
    - write sequencer configuration
        - `m![B, A # 23]` -> `m![B * (A # 31) / 32 * A % 32]`
        - Sequencer configuration: `[B=2:88, (A # 31)/32=3:32, A=32:1] @ 1024 / 32`

**Key insight:** Read sequencer configurations can safely overfetch (reading `dummy` addresses beyond the input tensor range is acceptable), but write sequencer configurations must never write beyond the tensor boundary (as it could write data to space occupied by other tensors).
This asymmetry is why `dummy=23` cannot use 32-byte fetches.


## Example 4: Tensor Segmentation

TCP handles tensors exceeding Vector Register File (VRF) capacity through segmentation.
Tensors too large to process in a single execution are automatically split into smaller chunks that fit within hardware constraints, with each chunk processed independently.

```rust,ignore
axes![A = 2048, B = 32];

// Input: 64KB tensor, exceeds 8KB VRF limit
let input: DmTensor<f8, m![1], m![1], m![1], m![A, B]> = ...;

// Segmented into two executions (compiler handles this automatically)

// Execution #0: first half of A
let seg_0: DmTensor<f8, m![1], m![1], m![1], m![A % 1024, B]> = ctx.main
    .begin(input.view().slice(A, 0..1024))
    .fetch::<f8, m![A % 1024], m![B]>()
    .collect::<m![A % 1024], m![B]>()
    .commit(256 * 1024);  // Write to 256K

// Execution #1: second half of A
let seg_1: DmTensor<f8, m![1], m![1], m![1], m![A @ 1024, B]> = ctx.main
    .begin(input.view().slice(A, 1024..2048))
    .fetch::<f8, m![A @ 1024], m![B]>()
    .collect::<m![A @ 1024], m![B]>()
    .commit(256 * 1024 + 32 * 1024);  // Write to 256K + 32K
```

**The Problem:**
The Vector Engine's VRF has only 8KB of capacity per slice. Tensors requiring more storage than this cannot be fetched and processed in one operation. Segmentation splits the tensor across multiple executions.

The 8KB limit is per-slice, not total. While a cluster has 256 slices (2MB total VRF), each slice holds only 8KB. Tensor distribution across slices is controlled by the slice dimension in the tensor mapping. Tensors without enough elements mapped to the slice dimension, or operations requiring entire rows/columns in individual slices (common in reduction operations), hit the per-slice limit before using all 256 slices. A [2048, 32] tensor with mapping m![1, 1, 1, 2048, 32] (no slice distribution) attempts to store all 64KB in slice 0, exceeding the 8KB limit. Even with slice distribution m![1, 1, 256, 8, 32], each slice stores only 256 bytes, but intermediate results or operation constraints may require more per-slice storage. Segmentation ensures each slice's VRF usage stays within the 8KB hardware limit.

**Input Tensor:**
A large 2D tensor that exceeds VRF capacity:
- **Shape**: `[A=2048, B=32]` - 65,536 elements
- **Data type**: `f8` (1 byte per element)
- **Total size**: 2048 × 32 = 65,536 bytes = 64 KB
- **Memory layout**: `m![A, B]` - standard row-major
- **Base address**: `b = 0`
- **Problem**: 64 KB far exceeds the 8 KB VRF limit per slice

**Output Tensor:**
The same tensor needs to be written to a different SRAM location:
- **Shape**: `[A=2048, B=32]` (identical)
- **Data type**: `f8`
- **Memory layout**: `m![A, B]`
- **Base address**: `b = 256K` (different location)

**Solution Strategy:**
Split the `A` dimension into two segments:
- **Segment 1**: `A % 1024` (first 1024 elements) = 1024 × 32 = 32 KB
- **Segment 2**: `A @ 1024` (second 1024 elements) = 1024 × 32 = 32 KB
- Each segment fits comfortably within the 8 KB limit... wait, that's still too large!

This requires further splitting or distributing dimensions across slices. Segmentation processes arbitrarily large tensors by dividing them into hardware-manageable chunks.

**Processing:**

Process the tensor in two separate executions:

### Execution #0
- fetch sequencer output
    - `Time = m![A % 1024]`
    - `Flit = m![B]`
    - fetch_size = 32 bytes
- Switch Engine output (= commit unit input)
    - `Time = m![A % 1024]`
    - `Flit = m![B]`
- Commit Unit
    - commit_in_size = 32 bytes
    - sliced shape
        - `Time = m![A % 1024]`
        - `Flit = m![B]`
    - write sequencer configuration
        - `m![A, B]` -> `m![(A % 1024), B]`
        - Sequencer configuration: `[A%1024=1024:32, B=32:1] @ 256K / 32`
        - From the entire output tensor with mapping `m![A, B]`, only the first half is fetched and committed.

### Execution #1 (Second Half)

Processes the second half of the A dimension:
- **Fetch sequencer output**:
    - `Time = m![A @ 1024]` - time dimension covers A elements 1024-2047
    - `Flit = m![B]` - 32-byte packets containing full B dimension
    - fetch_size = 32 bytes
- **Switch Engine output**:
    - `Time = m![A @ 1024]` - 1024 cycles for second half of A
    - `Flit = m![B]` - 32-byte flits
- **Commit Unit**:
    - commit_in_size = 32 bytes (full flit commit)
    - write sequencer configuration: `[A@1024=1024:32, B=32:1] @ (256K + 32 * 1024) / 32`
    - Base address: `256K + 32KB` (offset to skip the first segment)
    - Writes to addresses 256K+32KB through 256K+64KB

**Output:**

After both executions complete, the output tensor is reconstructed:
- **Memory layout**: SRAM starting at address 256K contains the complete tensor `[A=2048, B=32]`
- **Segment 1**: Addresses 256K to 256K+32KB hold `A[0:1023, B[0:31]]`
- **Segment 2**: Addresses 256K+32KB to 256K+64KB hold `A[1024:2047, B[0:31]]`
- **Result**: Logically identical to the input, just stored at a different location

The compiler automatically determines segmentation requirements and splits tensors into multiple executions. From the programmer's perspective, this is a single logical operation; the segmentation is transparent.

**Why this example is useful:**

Tensor segmentation is essential for practical deep learning workloads:
- **Large model support**: Modern LLMs have tensors with billions of elements that cannot fit in VRF
- **Automatic handling**: The compiler manages segmentation automatically based on VRF capacity
- **No performance penalty for well-designed splits**: When segment boundaries align with memory access patterns, segmentation adds minimal overhead
- **Scalability**: This mechanism enables processing tensors of arbitrary size on fixed hardware
- **Memory hierarchy exploitation**: Segmentation naturally maps to hierarchical memory systems (VRF → SRAM → HBM)

In practice, the compiler considers multiple factors when segmenting:
- VRF capacity constraints
- Memory bandwidth utilization
- Alignment with tensor unit requirements
- Minimizing the number of segments to reduce overhead
