# Stream Adapter: Advanced Operations

> **⚠ Draft — this chapter is under revision and not yet ready for review.**

> **TODO(jeongmin.park)**: No page in the book shows a convolution kernel that uses the features documented here.
> Add a "Convolution" kernel example to the Kernel Examples chapter that walks through a 2D convolution (e.g., 3×3 filter) showing how `initial_shift`, `shift_stride`, and `pop_dim` map to the sliding window pattern.

These operations extend the [Stream Adapter](./stream-adapter.md) for convolution workloads, enabling 3-flit collection, transpose, and shift-and-reuse patterns.

If you are only working with einsum (matrix multiplication) workloads, you can simply skip this chapter.

## Constraints

### Constraint Specifications

| Constraint | Limit | Cause |
|------------|-------|-------|
| Flit Buffer capacity | `feed_flits` ∈ {1, 2, 3} | 96-byte physical buffer (register file storage) |
| Transpose scope | Within a single 32-byte flit | Fixed-function permutation network |
| Shift buffer range | i4: [-15, 16], i8: [-7, 8], bf16: [-3, 4] | Limited register chain in Stream Shift Unit |
| Mapping alignment | Stream Adapter output must match TRF Sequencer computation mapping | Reducer has no buffering or reordering capability |

> **TODO**: Verify exact limits for shift buffer capacity.

### Design Rationale

**96-byte Flit Buffer**: The buffer must support single-cycle access for the downstream Reducer, requiring register file storage rather than standard SRAM.
Register files consume significantly more area per bit, making large buffers prohibitively expensive.
96 bytes (3 flits) balances useful convolution patterns with silicon cost.

**Single-flit Transpose**: Extending transpose across multiple flits would require either a much larger permutation network (exponentially more complex) or multi-cycle buffering (adding latency).
The single-flit restriction keeps the operation fast for the common case of ensuring the contraction axis is innermost.

**Shift buffer limits**: The Stream Shift Unit implements sliding windows by physically shifting data through a register chain.
The current limits (15 for i8, 7 for bf16) support common convolution filter sizes (3×3, 5×5, 7×7) while keeping hardware cost reasonable.

**TRF Sequencer alignment**: The Reducer is a fixed-function multiply-accumulate array that expects precisely aligned input streams.
It cannot buffer, reorder, or adapt to mismatched mappings.
Misaligned mappings cause incorrect computation, not graceful degradation.

## Flit Buffer: feed_flits 3

For shift-reuse with padding, the Flit Buffer supports a third mode:

| Parameter | Value | Description |
|-----------|-------|-------------|
| `feed_flits: 3` | 96 bytes | For shift-reuse with padding |

The `feed_flits: 3` case is detailed in the [Shift](#shift-stream-shift-unit) section below.

## Transpose

Transpose reorders axes within a flit when the contraction axis is not innermost.
The Reducer reduces adjacent pairs starting from the innermost axis, so the axis being contracted must be innermost.

Transpose swaps axes within a 32-byte flit.

### Supported Transposes

The supported transposes depend on data type (total volume is always 32 bytes):

| Data Type | Supported Transposes |
|-----------|---------------------|
| int4      | `[4][16] → [16][4]` |
| i8/fp8    | `[2][16] → [16][2]`, `[4][8] → [8][4]` |
| bf16      | `[2][8] → [8][2]` |

> These types are those that can be computed in the Contraction Engine; i32/f32 etc. cannot perform local adder tree operations.

### Example 1: Transpose within flit

```text
axes![A = 3, B = 2, C = 2, D = 8];

// Input: time -> [1], num_flits -> [3_a], flit -> [2_b × 2_c × 8_d], i8
//
// Possible transpose outputs:
//   1. [2][16] → [16][2]: flit -> [2_c × 8_d × 2_b]
//   2. [4][8]  → [8][4]:  flit -> [8_d × 2_b × 2_c]
//
// From the Stream Shift Unit onward, num_flits × flit = flits:
//   1. flits -> [3_a × 2_c × 8_d × 2_b]
//   2. flits -> [3_a × 8_d × 2_b × 2_c]
```

### Example 2: Transpose in computation mapping

```text
axes![P = 64, A = 2, B = 2, C = 16];

// Configuration: feed_flits = 2, datatype = bf16, transpose_flit(32B) = true
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, c_1=16, b_1=2] ] (16)
// Computation mapping:       [ P: [P_1=64] | H: [a_1=2] | Row: [] | T: [b_1=2, c_1=16] ] (16)
```

**Key observations:**
- `c_1=16, b_1=2` becomes `b_1=2, c_1=16`, moving `b` to the innermost position for Reducer reduction
- With axes properly ordered, the Stream Adapter can apply shift-and-reuse for sliding window operations

## Shift (Stream Shift Unit)

The Stream Shift Unit enables **shift-and-reuse** for sliding window operations like convolutions.
Instead of fetching overlapping windows multiple times, it fetches data once and shifts it to produce multiple windows.

Three parameters control shifting:
- **`initial_shift`**: Starting offset when data is first loaded
- **`shift_stride`**: How much to shift for each iteration
- **`pop_dim`**: Which dimension triggers fetching new data

### initial_shift

The `initial_shift` parameter sets the starting offset when data enters the shift buffer.
Negative values shift toward upper bits; positive values shift toward lower bits.

#### Valid Ranges

| Data Type | Range | Distinct Values |
|-----------|-------|-----------------|
| i4  | -15 to 16 | 32 |
| i8  | -7 to 8   | 16 |
| bf16 | -3 to 4   | 8  |

The ranges are asymmetric (one more positive than negative) because they encode exactly 2^N distinct positions centered around zero.
The range sizes (32, 16, 8) correspond to half the number of elements in a 32-byte flit for each data type, matching the Stream Shift Unit's buffer capacity.

After applying `initial_shift`, output is sliced to 64 bytes for the Reducer. Padding is filled with zeros.

#### Example: Negative initial_shift

```text
axes![A = 96, B = 3];

// Input: time -> [3_b], flits -> [a], i8
// initial_shift = -7
//
// After initial shift: time -> [3_b], Row -> [7_pad + a.slice(57)]
//   - Left portion is zero-padded
//   - After shifting by 7, only the lower 64 bytes are sliced and output
//   - For feed_flits = 1 or 2, portions beyond the actual fetched region are zero-padded
```

#### Example: Positive initial_shift

```text
axes![A = 96, B = 3];

// Input: time -> [3_b], flit -> [a], i8
// initial_shift = 8
//
// After initial shift: time -> [3_b], Row -> [a.offset(8).slice(64)]
//   - 8-element shift + 64-byte slicing
//   - For feed_flits ≠ 3, the upper 8 bytes are zero-padded
```

#### Per-index initial_shift

The `initial_shift_dim` parameter allows different shifts per index:

| `initial_shift_dim` | Behavior |
|---------------------|----------|
| 8 | Single shift value for all indices |
| 0..7 | Use `initial_shift_elements[i]` based on index value |

```text
axes![A = 96, B = 3];

// Input: time -> [3_b], flits -> [a], i8
// seq_limits: [3, 1, 1, 1, 1, 1, 1, 1] (flits not shown in index)
// initial_shift_dim = 0 (b-axis)
// initial_shift_elements = [-7, 8, 0]
//
// at b = 0: Row -> [7_pad + a.slice(57)]
// at b = 1: Row -> [a.offset(8).slice(64)]
// at b = 2: Row -> [a.slice(64)]
```

The initial_shift is applied once when flits are popped and is not applied during reuse/shift.

#### Example: Negative initial_shift (computation mapping)

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4)
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, [[c_1=(1,31)]+1]=32] ] (16)
// Computation mapping:       [ P: [P_1=64] | H: [a_1=2] | Row: [] | T: [c_1=32] ] (16)
```

**Key observations:**
- With `init_shift: -1`, data is shifted left by 1 element, adding 1 zero-padding element at the start
- `[[c_1=(1,31)]+1]`: first element is padding, 31 elements are original data

#### Example: Positive initial_shift (computation mapping)

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = 1, init_shift_range = (-3, 4)
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, [1+[c_1=31]]=32] ] (16)
// Computation mapping:       [ P: [P_1=64] | H: [a_1=2] | Row: [] | T: [[[c_1=31]+1]=32] ] (16)
```

**Key observations:**
- With `init_shift: 1`, data is shifted right by 1 element, dropping the first element and adding 1 zero-padding at the end
- Switch Engine mapping `[1+[c_1=31]]`: first element is included
- Computation mapping `[[c_1=31]+1]`: 31 data elements followed by 1 padding element

#### Example: Per-index initial_shift using indirect vectors

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift_tag = a_1, init_shifts = [-1, 1], init_shift_range = (-3, 4)
//   indirect_vecs: [I0 = (c_1=32)[1, -1]]
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2 @ I0_1, c_1=32] ] (16)
// Computation mapping:       [ P: [P_1=64] | H: [a_1=2] | Row: [] | T: [[[c_1=31]+1]=32] ] (16)
```

**Key observations:**
- With `init_shift_tag: a_1` and `init_shifts: [-1, 1]`, the shift varies per index
- At `a_1=0`: shift by -1 (left); at `a_1=1`: shift by 1 (right)
- The indirect vector `[1, -1]` controls element selection from `c_1=32` for each `a_1` index

#### Example: initial_shift with interleaved sliding window

```text
axes![P = 64, A = 2, B = 31, C = 3];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4), shift_stride = 1
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,30):1)+(c_1=3:1)]=32] ] (16)
// Computation mapping:       [ P: [P_1=64] | H: [a_1=2, c_1=3] | Row: [] | T: [[[b_1=31]+1]=32] ] (16)
```

**Key observations:**
- `$[(b_1=(1,30):1)+(c_1=3:1)]` is an interleaved pattern: `b_1` starts at offset 1 with stride 1, `c_1` has 3 iterations with stride 1
- Initial shift of -1 adds 1 padding element at the start, producing `[[b_1=31]+1]`

### shift_stride and pop_dim

The `shift_stride` parameter controls reuse: each iteration along `shift_dim` applies an additional shift of `shift_stride` elements.

The `pop_dim` parameter marks when to fetch new data: when the index at `pop_dim` increments, fresh data is loaded and `initial_shift` is reapplied.

Dimensions are ordered: **tile → shift_dim → pop_dim** (inner to outer).
Indices below `pop_dim` that are not `shift_dim` produce tiled (broadcast) outputs.

#### Valid `shift_stride` Ranges

| Data Type | Range |
|-----------|-------|
| i4  | 0 to 31 |
| i8  | 0 to 15 |
| bf16 | 0 to 7  |

#### Example: shift_stride with pop_dim

```text
axes![A = 96, B = 3];

// initial_shift = -1, shift_dim = 1, shift_stride = 3, pop_dim = 2
// seq_limits: [2, 3, 3, 1, 1, 1, 1, 1]
// Input: time -> [b = 3], flits -> [a], i8
//
// flits #0 (indexer: [0, 0, 0]): Apply initial shift.
//   [1_pad + a], slice to 64 → Row -> [(1_pad + a) % 63]
// flits #1 (indexer: [1, 0, 0]): Same as #0 (dim0 is not shift_dim → tiling)
// flits #2 (indexer: [0, 1, 0]): shift_dim=1, apply shift_stride=3 from #0 state.
//   [a @ 2], slice to 64 → Row -> [(a @ 2) % 64]
// flits #3 (indexer: [1, 1, 0]): Same as #2 (dim0 is not shift_dim → tiling)
// flits #4 (indexer: [0, 2, 0]): shift_dim=1, apply shift_stride=3 from #2 state.
//   [a @ 5], slice to 64 → Row -> [(a @ 5) % 64]
// flits #5 (indexer: [1, 2, 0]): Same as #4 (dim0 is not shift_dim → tiling)
// flits #6 (indexer: [0, 0, 1]): pop_dim=2, fetch new flits and apply initial shift.
//   [1_pad + a], slice to 64 → Row -> [(1_pad + a) % 63]
//
// Output mapping: time -> [b × (1_pad + a) / f=3:3 × Broadcast=2], Row -> [(1_pad + a) / w=64:1]
```

### Shift Examples

**Shift with stride 1:**

```text
axes![P = 64, A = 2, B = 31, C = 3];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4), shift_stride = 1
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,30):1)+(c_1=3:1)]=32] ] (16)
// Computation mapping:       [ P: [P_1=64] | H: [a_1=2, c_1=3] | Row: [] | T: [[[b_1=31]+1]=32] ] (16)
```

**Shift with stride 2:**

```text
axes![P = 64, A = 2, B = 16, C = 4];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -2, init_shift_range = (-3, 4), shift_stride = 2
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,15):2)+(c_1=4:1)]=32] ] (16)
// Computation mapping:       [ P: [P_1=64] | H: [a_1=2, c_2=2] | Row: [] | T: [b_1=16, c_1=2] ] (16)
```

**Data reuse without shift (tiling):**

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, c_1=32] ] (16)
// Computation mapping:       [ P: [P_1=64] | H: [a_1=2, #t_1=5] | Row: [] | T: [c_1=32] ] (16)
```

**Key observations:**
- The `#t_1=5` broadcasts the same data 5 times without shifting

**pop_dim with shift dimension:**

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, c_1=32] ] (16)
// Computation mapping:       [ P: [P_1=64] | H: [a_1=2, #s_1=3, #t_1=5] | Row: [] | T: [c_1=32] ] (16)
```

**Key observations:**
- `#s_1=3`: shift dimension — data is shifted 3 times (shift-and-reuse)
- `#t_1=5`: pop_dim — broadcasts (tiles) the result 5 times
- New data is fetched only when moving beyond the outermost dimension

**pop_dim with sliding window and tiling:**

```text
axes![P = 64, A = 2, B = 31, C = 3];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4), shift_stride = 1
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,30):1)+(c_1=3:1)]=32] ] (16)
// Computation mapping:       [ P: [P_1=64] | H: [a_1=2, c_1=3, #t_1=5] | Row: [] | T: [[[b_1=31]+1]=32] ] (16)
```

**Key observations:**
- `c_1=3` acts as the shift dimension with stride 1, generating 3 shifted windows per data buffer
- `#t_1=5` tiles (broadcasts) each of the 3 windows 5 times
- New data is fetched only after all `c_1` and `#t_1` iterations complete


## Performance

**Transpose latency**: Transpose operations within flits add 1–2 cycles of latency. This overhead is unavoidable when the contraction axis is not innermost.

**Shift complexity**: The Stream Shift Unit enables convolution weight reuse but adds complexity:
- Initial shift configuration overhead (setup cycles)
- Stride and `pop_dim` parameters affect buffer management
- Incorrect shift configuration can cause incorrect results, not just performance degradation

**Data reuse benefits**: Properly configured shift operations dramatically reduce memory bandwidth requirements for convolutions by reusing input data across multiple output positions.
Without shift, each output would require separate input fetches.
