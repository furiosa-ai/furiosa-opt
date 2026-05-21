# 2D Convolution


2D convolution is the einsum `$(H + Fh)$(W + Fw)K, FhFwKC -> HWC` with spatial output axes `H`, `W`, output channel `C`, and contraction axes `Fh`, `Fw`, `K`.
The `$(W + Fw)` sliding maps to [Stream Adapter shift-reuse](#shift-stream-shift-unit) (documented below).

The 4 variants in [Variants](#variants) differ in how the Stream Adapter shifts the sliding window (the conv-specific machinery).
Choice of which axis to put in `Time` follows the same trade-off as in the [matmul mapping discussion](./index.md#example-batched-matmul) and is not repeated.

## Variants

### Filter-Stride 1

For stride-1 convolution, the Fetch Engine handles `$(H+Fh)` sliding before data reaches the Stream Adapter.
The Stream Adapter then processes the `$(W+Fw)` dimension via shift-reuse to produce `Fw, W` sliding in the computation.
The example below uses shift-stride 1 with two shifts.

```text
// Configuration: input_type = bf16, trf_type = bf16
//        Input mapping: [ H: [H=30, Fh=3, K=32, $(W=30 + Fw=3)=32] ] (1)
//          TRF mapping: [ Lane: [C=8] | H: [K=32, C=24, Fh=3, Fw=3] ] (1)
//  Contraction mapping: [ H: [H=30, C/8=3, Fh=3, K=32, Fw=3] | Lane: [C=8] | T: [W=30+2#] ] (1)
// Accumulation mapping: [ H: [H=30, C=32] | T: [W=30+2#] ] (1)
```

### Filter-Stride 2

For stride-2 convolution, a shift-stride of 2 (one shift per output position) extracts strided windows.
The input expression `$(W:2=15 + Fw=4)=32` factors into `Fw/2=2, (W=15, Fw=2)` by extracting a size-2 axis with stride `:2` as an outer product: `$(W:2=15 + (Fw/2:2=2, Fw=2))=32` becomes `Fw/2=2, $(W:2=15, Fw=2)`.

```text
// Configuration: input_type = bf16, trf_type = bf16
//        Input mapping: [ H: [H=15, Fh=4, K=32, $(W:2=15 + Fw=4)=32] ] (1)
//          TRF mapping: [ Lane: [C=8] | H: [K=32, C=24, Fh=4, Fw=4] ] (1)
//  Contraction mapping: [ H: [H=15, C/8=3, Fh=4, K=32, Fw/2=2] | Lane: [C=8] | T: [W=15+1#, Fw=2] ] (1)
// Accumulation mapping: [ H: [H=15, C=32] | T: [W=15+1#] ] (1)
```

The previous example underutilizes MACs because the shift buffer is not full.
Setting `feed_flits` to 3 (from the default of 2) fills more flits in the shift buffer and achieves full MAC utilization.
The transformation `$(W:2=16 + Fw=4)=34` then produces `Fw/2=2, (W=16, Fw=2)`.

```text
// Configuration: feed_flits = 3, input_type = bf16, trf_type = bf16
//        Input mapping: [ H: [H=16, Fh=4, K=32, $(W:2=16 + Fw=4)=34] ] (1)
//          TRF mapping: [ Lane: [C=8] | H: [K=32, C=24, Fh=4, Fw=4] ] (1)
//  Contraction mapping: [ H: [H=16, C/8=3, Fh=4, K=32, Fw/2=2] | Lane: [C=8] | T: [W=16, Fw=2] ] (1)
// Accumulation mapping: [ H: [H=16, C=32] | T: [W=16] ] (1)
```

### Dilation 2

For dilation-2 convolution, the filter samples input positions separated by a stride of 2.
A shift-stride of 2 with 2 shifts extracts these dilated filter positions.

The transformation `$(W=27 + Fw:2=3)=32` produces `Fw=3, W=27`, extracting a size-3 axis with stride `:2` from a linear combination as an outer product: `$(W=27 + Fw:2=3)=32` becomes `Fw=3, $(W=27)`.

```text
// Configuration: input_type = bf16, trf_type = bf16
//        Input mapping: [ H: [H=27, Fh=3, K=32, $(W=27 + Fw:2=3)=32] ] (1)
//          TRF mapping: [ Lane: [C=8] | H: [K=32, C=24, Fh=3, Fw=3] ] (1)
//  Contraction mapping: [ H: [H=27, C/8=3, Fh=3, K=32, Fw=3] | Lane: [C=8] | T: [W=27+5#] ] (1)
// Accumulation mapping: [ H: [H=27, C=32] | T: [W=27+5#] ] (1)
```

### Filter-Stride 2, Dilation 2

Combining stride-2 and dilation-2 requires shift operations similar to dilation 2 alone.

The transformation `$(W:2=14 + Fw:2=3)=31 + 1#` produces `Fw=3, W=14, 1+1#`, extracting a size-3 axis with stride `:2` from a linear combination as an outer product: `$(W:2=14 + Fw:2=3)=31` becomes `Fw=3, $(W:2=14)`.

The TRF must carry zeros in dummy slots so that `1+1#` contracted with `1+1z` yields 1, rather than the arbitrary `1#` that would result from contracting `1+1#` with `1+1#`.
The notation `1z` is like `1#` (dummy padding) but filled with zeros instead of arbitrary values.

```text
// Configuration: input_type = bf16, trf_type = bf16
//        Input mapping: [ H: [H=14, Fh=3, K=32, $(W:2=14 + Fw:2=3)=31+1#] ] (1)
//          TRF mapping: [ Lane: [C=8] | H: [K=32, C=24, Fh=3, Fw=3, 1+1z] ] (1)
//  Contraction mapping: [ H: [H=14, C/8=3, Fh=3, K=32, Fw=3] | Lane: [C=8] | T: [W=14+2#, 1+1z] ] (1)
// Accumulation mapping: [ H: [H=14, C=32] | T: [W=14+2#] ] (1)
```

## Stream Adapter Machinery for Convolutions

Convolution workloads need sliding-window data reuse to avoid refetching the same input elements.
The [Stream Adapter](./outer.md#stream-adapter) provides this reuse through three extensions: 3-flit collection, transpose, and shift-and-reuse.

Skip this section if you only work with einsum (matrix multiplication) workloads.

### Flit Buffer: feed_flits 3

`feed_flits: 3` extends the Flit Buffer beyond the default 2-flit capacity to fill all 96 bytes with three consecutive 32 B flits.
The third flit gives the [Stream Shift Unit](#shift-stream-shift-unit) enough buffered data to shift the window without refetching, at the cost of one extra flit's worth of buffering per Packet.

| Parameter | Value | Description |
|-----------|-------|-------------|
| `feed_flits: 3` | 96 bytes | Reserves all 3 flits for shift-reuse |

See [Shift](#shift-stream-shift-unit) for how the Stream Shift Unit consumes the extra flit.

### Transpose

The Packet Reducer reduces adjacent pairs starting from the innermost axis, so the contracted axis must be innermost.
Transpose reorders axes within a 32 B flit when the incoming data has a different axis order.

#### Supported Transposes

The supported transposes depend on data type (total volume is always 32 B).

| Data Type | Supported Transposes |
|-----------|---------------------|
| int4      | `[4][16] → [16][4]` |
| i8/fp8    | `[2][16] → [16][2]`, `[4][8] → [8][4]` |
| bf16      | `[2][8] → [8][2]` |

> These types are the ones the Contraction Engine can compute.
> Types like `i32` and `f32` cannot use the Packet Reducer's reduction tree.

#### Example 1: Transpose within flit

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

#### Example 2: Transpose in Contraction Mapping

```text
axes![P = 64, A = 2, B = 2, C = 16];

// Configuration: feed_flits = 2, datatype = bf16, transpose_flit(32B) = true
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, c_1=16, b_1=2] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2] | Lane: [] | T: [b_1=2, c_1=16] ] (16)
```

- `c_1=16, b_1=2` becomes `b_1=2, c_1=16`, moving `b` to the innermost position for Packet Reducer reduction.
- With axes properly ordered, the Stream Adapter applies shift-and-reuse for sliding window operations.

### Shift (Stream Shift Unit)

The Stream Shift Unit performs shift-and-reuse for sliding-window operations like convolutions.
Rather than fetching overlapping windows multiple times, it fetches data once and shifts it to produce multiple windows.


Three parameters control shifting.

- **`initial_shift`**: starting offset when data is first loaded.
- **`shift_stride`**: amount to shift per iteration along the shift dimension.
- **`pop_dim`**: dimension that triggers fetching new data.

#### initial_shift

The `initial_shift` parameter sets the starting offset when data enters the shift buffer.
The buffer holds elements in order from low address to high.
Negative `initial_shift` shifts data toward later positions (higher addresses, upper bits), padding the early positions with zeros.
Positive `initial_shift` shifts toward earlier positions (lower addresses, lower bits), padding the late positions.

##### Valid ranges

| Data Type | Range | Distinct Values |
|-----------|-------|-----------------|
| i4  | -15 to 16 | 32 |
| i8  | -7 to 8   | 16 |
| bf16 | -3 to 4   | 8  |

The distinct-value counts (32, 16, 8) correspond to half the number of elements in a 32 B flit, matching the Stream Shift Unit's buffer capacity.
The ranges are asymmetric (one more positive than negative) because they encode a power-of-two count of distinct positions with zero not at the exact midpoint.

After applying `initial_shift`, the output slices to 64 B for the Packet Reducer, with zeros filling any padding.

##### Example: negative initial_shift

```text
axes![A = 96, B = 3];

// Input: time -> [3_b], flits -> [a], i8
// initial_shift = -7
//
// After initial shift: time -> [3_b], Lane -> [7_pad + a.slice(57)]
//   - Left portion is zero-padded
//   - After shifting by 7, only the lower 64 B are sliced and output
//   - For feed_flits = 1 or 2, portions beyond the actual fetched region are zero-padded
```

##### Example: positive initial_shift

```text
axes![A = 96, B = 3];

// Input: time -> [3_b], flit -> [a], i8
// initial_shift = 8
//
// After initial shift: time -> [3_b], Lane -> [a.offset(8).slice(64)]
//   - 8-element shift + 64 B slicing
//   - For feed_flits ≠ 3, the upper 8 bytes are zero-padded
```

##### Per-index initial_shift

The `initial_shift_dim` parameter selects a different shift per index.

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
// at b = 0: Lane -> [7_pad + a.slice(57)]
// at b = 1: Lane -> [a.offset(8).slice(64)]
// at b = 2: Lane -> [a.slice(64)]
```

The `initial_shift` applies once when flits pop, and not during reuse/shift.

##### Example: negative initial_shift (contraction mapping)

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4)
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, [[c_1=(1,31)]+1]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2] | Lane: [] | T: [c_1=32] ] (16)
```

- With `init_shift: -1`, data shifts left by 1 element, adding 1 zero-padding element at the start.
- `[[c_1=(1,31)]+1]`: first element is padding, 31 elements are original data.

##### Example: positive initial_shift (contraction mapping)

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = 1, init_shift_range = (-3, 4)
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, [1+[c_1=31]]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2] | Lane: [] | T: [[[c_1=31]+1]=32] ] (16)
```

- With `init_shift: 1`, data shifts right by 1 element, dropping the first element and adding 1 zero-padding at the end.
- Switch Engine mapping `[1+[c_1=31]]`: first element is included.
- Contraction mapping `[[c_1=31]+1]`: 31 data elements followed by 1 padding element.

##### Example: per-index initial_shift using indirect vectors

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift_tag = a_1, init_shifts = [-1, 1], init_shift_range = (-3, 4)
//   indirect_vecs: [I0 = (c_1=32)[1, -1]]
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2 @ I0_1, c_1=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2] | Lane: [] | T: [[[c_1=31]+1]=32] ] (16)
```

- With `init_shift_tag: a_1` and `init_shifts: [-1, 1]`, the shift varies per index.
- At `a_1=0`: shift by -1 (left). At `a_1=1`: shift by 1 (right).
- The indirect vector `[1, -1]` controls element selection from `c_1=32` for each `a_1` index.

##### Example: initial_shift with interleaved sliding window

```text
axes![P = 64, A = 2, B = 31, C = 3];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4), shift_stride = 1
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,30):1)+(c_1=3:1)]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, c_1=3] | Lane: [] | T: [[[b_1=31]+1]=32] ] (16)
```

- `$[(b_1=(1,30):1)+(c_1=3:1)]` is an interleaved pattern: `b_1` starts at offset 1 with stride 1, and `c_1` has 3 iterations with stride 1.
- Initial shift of -1 adds 1 padding element at the start, producing `[[b_1=31]+1]`.

#### shift_stride and pop_dim

The `pop_dim` parameter marks when to fetch new data: when the index at `pop_dim` increments, fresh data loads and `initial_shift` reapplies.
The `shift_dim` is the dimension along which reuse occurs: each iteration along `shift_dim` applies an additional shift of `shift_stride` elements.

Indices below `pop_dim` that are not `shift_dim` produce tiled (broadcast) outputs, giving the dimension ordering `tile → shift_dim → pop_dim` (inner to outer).

##### Valid `shift_stride` ranges

| Data Type | Range |
|-----------|-------|
| i4  | 0 to 31 |
| i8  | 0 to 15 |
| bf16 | 0 to 7  |

##### Example: shift_stride with pop_dim

```text
axes![A = 96, B = 3];

// initial_shift = -1, shift_dim = 1, shift_stride = 3, pop_dim = 2
// seq_limits: [2, 3, 3, 1, 1, 1, 1, 1]
// Input: time -> [b = 3], flits -> [a], i8
//
// flits #0 (indexer: [0, 0, 0]): Apply initial shift.
//   [1_pad + a], slice to 64 → Lane -> [(1_pad + a) % 63]
// flits #1 (indexer: [1, 0, 0]): Same as #0 (dim0 is not shift_dim → tiling)
// flits #2 (indexer: [0, 1, 0]): shift_dim=1, apply shift_stride=3 from #0 state.
//   [a @ 2], slice to 64 → Lane -> [(a @ 2) % 64]
// flits #3 (indexer: [1, 1, 0]): Same as #2 (dim0 is not shift_dim → tiling)
// flits #4 (indexer: [0, 2, 0]): shift_dim=1, apply shift_stride=3 from #2 state.
//   [a @ 5], slice to 64 → Lane -> [(a @ 5) % 64]
// flits #5 (indexer: [1, 2, 0]): Same as #4 (dim0 is not shift_dim → tiling)
// flits #6 (indexer: [0, 0, 1]): pop_dim=2, fetch new flits and apply initial shift.
//   [1_pad + a], slice to 64 → Lane -> [(1_pad + a) % 63]
//
// Output mapping: time -> [b × (1_pad + a) / f=3:3 × Broadcast=2], Lane -> [(1_pad + a) / w=64:1]
```

#### Shift Examples

##### Shift with stride 1

```text
axes![P = 64, A = 2, B = 31, C = 3];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4), shift_stride = 1
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,30):1)+(c_1=3:1)]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, c_1=3] | Lane: [] | T: [[[b_1=31]+1]=32] ] (16)
```

##### Shift with stride 2

```text
axes![P = 64, A = 2, B = 16, C = 4];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -2, init_shift_range = (-3, 4), shift_stride = 2
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,15):2)+(c_1=4:1)]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, c_2=2] | Lane: [] | T: [b_1=16, c_1=2] ] (16)
```

##### Data reuse without shift (tiling)

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, c_1=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, #t_1=5] | Lane: [] | T: [c_1=32] ] (16)
```

- `#t_1=5` broadcasts the same data 5 times without shifting.

##### pop_dim with shift dimension

```text
axes![P = 64, A = 2, C = 32];

// Configuration: feed_flits = 2, datatype = bf16
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, c_1=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, #s_1=3, #t_1=5] | Lane: [] | T: [c_1=32] ] (16)
```

- `#s_1=3`: shift dimension, where data shifts 3 times (shift-and-reuse).
- `#t_1=5`: pop_dim, broadcasting (tiling) the result 5 times.
- New data fetches only when moving beyond the outermost dimension.

##### pop_dim with sliding window and tiling

```text
axes![P = 64, A = 2, B = 31, C = 3];

// Configuration: feed_flits = 2, datatype = bf16
//   init_shift = -1, init_shift_range = (-3, 4), shift_stride = 1
// Switch Engine output: [ P: [P_1=64] | H: [a_1=2, $[(b_1=(1,30):1)+(c_1=3:1)]=32] ] (16)
// Contraction mapping:       [ P: [P_1=64] | H: [a_1=2, c_1=3, #t_1=5] | Lane: [] | T: [[[b_1=31]+1]=32] ] (16)
```

- `c_1=3` acts as the shift dimension with stride 1, generating 3 shifted windows per data buffer.
- `#t_1=5` tiles (broadcasts) each of the 3 windows 5 times.
- New data fetches only after all `c_1` and `#t_1` iterations complete.

### Constraints

The hardware limits each advanced operation, and the table below summarizes the limits with their physical causes.

| Constraint | Limit | Cause |
|------------|-------|-------|
| Flit Buffer capacity | `feed_flits` ∈ {1, 2, 3} | 96-byte physical buffer (register file storage) |
| Transpose scope | Within a single 32 B flit | Fixed-function permutation network |
| Shift buffer range | i4: [-15, 16], i8: [-7, 8], bf16: [-3, 4] | Limited register chain in Stream Shift Unit |
| Mapping alignment | Stream Adapter output must match TRF Sequencer contraction mapping | Packet Reducer has no buffering or reordering capability |


#### Design Rationale

- **96-byte Flit Buffer**: single-cycle access for the downstream Packet Reducer requires register file storage rather than standard SRAM.
  Register files consume significantly more area per bit, making large buffers prohibitively expensive.
  96 bytes (3 flits) balances useful convolution patterns with silicon cost.
- **Single-flit Transpose**: restricting transpose to a single flit keeps the operation fast for the common case (ensuring the contraction axis is innermost).
  Extending it across multiple flits would require either a much larger permutation network or multi-cycle buffering.
- **Shift buffer limits**: the Stream Shift Unit implements sliding windows by physically shifting data through a register chain.
  The current limits (15 for `i8`, 7 for `bf16`) support common convolution filter sizes (3×3, 5×5, 7×7) while keeping hardware cost reasonable.


- **TRF Sequencer alignment**: the Outer stage's elementwise multiplication and the Packet Reducer's tree form a fixed-function multiply-accumulate array that expects precisely aligned input streams.
  It cannot buffer, reorder, or adapt to mismatched mappings, so misaligned mappings cause incorrect computation, not graceful degradation.

### Performance

- **Transpose latency**: transpose within a flit adds 1-2 cycles of latency.
  This overhead is unavoidable when the contraction axis is not innermost.
- **Shift configuration**: incorrect shift configuration causes incorrect results, not just performance degradation.
  Other considerations:
  - Initial shift configuration overhead (setup cycles).
  - Stride and `pop_dim` parameters affect buffer management.
- **Data reuse benefits**: properly configured shift operations dramatically reduce memory bandwidth for convolutions by reusing input data across multiple output positions.
  Without shift, each output requires separate input fetches.
