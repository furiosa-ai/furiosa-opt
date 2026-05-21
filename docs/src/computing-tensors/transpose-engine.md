# Transpose Engine

The Transpose Engine swaps the `Time` and `Packet` dimensions, while leaving the `Chip`, `Cluster`, and `Slice` dimensions unchanged.

## Interface

`CollectTensor` and `VectorFinalTensor` both expose `.transpose()`.
The `VectorFinalTensor` entry point feeds the Transpose Engine directly from the Vector Engine output.

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/transpose.rs:transpose_impl}}
```

The kernel writer chooses `OutTime` and `OutPacket` (the output dimension layouts), and the compiler verifies the result against the hardware constraints listed under [Parameters](#parameters).

The example below transposes a fully-utilized 8×8 `i8` matrix.
It is reused as the running example throughout the rest of this page.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, C = 8, D = 8, E = 8];

fn basic_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1], m![P], m![C, D], m![E # 32]>,
) -> TransposeTensor<'l, T, i8, m![1], m![1], m![P], m![C, E], m![D # 32]> {
    input.transpose()
}
```

## Architecture

The four transpose stages below are illustrated using the running example from [Interface](#interface).

### Parameters

`valid_size` is the number of valid elements the Transpose Engine reads per cycle from its 32-byte input bus, and every input flit arrives in `bit-width × valid_size` form, with any remaining bytes of the 32-byte flit treated as padding and discarded by the Unpack stage.
Data reaches the Transpose Engine via `CollectTensor::transpose()` (after `Fetch → [Switch →] Collect → [Cast →] Transpose`) or `VectorFinalTensor::transpose()` (directly from the [Vector Engine](./vector-engine/index.md)).
The [Contraction Engine](./contraction-engine/index.md) emits `32b × 8` only, while the [Vector Engine](./vector-engine/index.md) and [Fetch Engine](../moving-tensors/fetch-engine.md) emit any combination from the table below.

`in_cols`, `in_rows`, and `out_rows` are fixed by the kernel writer's `OutTime` and `OutPacket` choices.
All four are constrained by the element size:

| Element size | `valid_size` | Max `in_rows` | Valid `in_cols` |
|--------------|--------------|---------------|-----------------|
| 4-bit        | 16           | 16            | 16, 32          |
| 8-bit        | 8            | 8             | 8, 16, 32       |
| 16-bit       | 8            | 4             | 8, 16, 32       |
| 32-bit       | 8            | 2             | 8, 16, 32       |

For the running example (`i8`, so `valid_size = 8`), the compiler derives:

| Parameter   | Value | Notes      |
|-------------|-------|------------|
| `in_cols`   | 8     | `E::SIZE`  |
| `in_rows`   | 8     | `D::SIZE`  |
| `out_rows`  | 8     | `E::SIZE`  |

```text
                 in_cols                  in_rows # F
           ┌─────────────────┐         ┌──────────────────┐
           │ 12 13 14 15 ... │         │ 3  7  11 15  ... │
 in_rows   │ 8  9  10 11 ... │  ────►  │ 2  6  10 14  ... │  out_rows
           │ 4  5  6  7  ... │         │ 1  5  9  13  ... │
           │ 0  1  2  3  ... │         │ 0  4  8  12  ... │
           └─────────────────┘         └──────────────────┘
                data_in                      data_out
```

### Unpack

Each 32-byte input packet carries `valid_size` valid elements, and the Unpack stage discards the rest as padding.
The packets from each time step combine into a row of width `in_cols` (a multiple of `valid_size`).
Across `in_rows` time steps, these rows stack into the `[in_rows × in_cols]` input matrix.

In the running example: `[D, E # 32]` → `[D, E]`.

### Transpose

The matrix is transposed: `[in_rows × in_cols]` → `[in_cols × in_rows]`.

In the running example: `[D, E]` → `[E, D]`.

### Trim

When some input packets carry fewer valid elements than `valid_size`, the transposed matrix has padded rows.
The Trim stage drops those rows, producing `[out_rows × in_rows]` where `out_rows ≤ in_cols`.

In the running example: `[E, D]` → `[E, D]` (the input is fully utilized, so no rows are trimmed).
See the [Small Matrix](#small-matrix) example for a case where Trim actually discards rows.

### Align

The transposed rows are `in_rows` elements wide, but DM packets must be 32 bytes.
The Align stage pads each row to a 32-byte flit, producing shape `[out_rows × (in_rows # F)]` where `F` is chosen so that `D[F]` is 32 bytes.

In the running example: `[E, D]` → `[E, D # 32]`.

### Latency

> [!NOTE]
> Read [Performance](#performance) first for the formulas.

For the running example, `in_cols = 8 ≤ 16` selects double buffering.
With `in_flits = 8`, `out_rows = 8`, and `n = 8`, the total latency is `8 + 7 × max(8, 8) + 8 = 72` cycles.

## Examples

### Small Matrix

This example demonstrates the Trim stage discarding padded rows when `out_rows < in_cols`:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 64, A = 4, B = 2];

fn small_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1], m![P], m![A], m![B # 32]>,
) -> TransposeTensor<'l, T, i8, m![1], m![1], m![P], m![B], m![A # 32]> {
    input.transpose()
}
```

Parameters:

| Parameter   | Value | Notes                       |
|-------------|-------|-----------------------------|
| `in_cols`   | 8     | `B::SIZE = 2`, padded to 8  |
| `in_rows`   | 4     | `A::SIZE`                   |
| `out_rows`  | 2     | `B::SIZE`                   |

Stages:
- **Unpack**: `[A, B # 32]` → `[A, B # 8]`.
- **Transpose**: `[A, B # 8]` → `[B # 8, A]`.
- **Trim**: `[B # 8, A]` → `[B, A]` (6 padded rows trimmed).
- **Align**: `[B, A]` → `[B, A # 32]`.

Latency: `in_cols = 8 ≤ 16` selects double buffering.
With `in_flits = 4`, `out_rows = 2`, and `n = 1`, the total is `4 + 0 × max(4, 2) + 2 = 6` cycles.

### Large Column

This example forces single buffering by making `in_cols > 16`, which prevents input and output from overlapping and so increases total cycles:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, B = 2, C = 8, D = 4, E = 8];

fn large_col_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1], m![P], m![B, C, D], m![E # 32]>,
) -> TransposeTensor<'l, T, i8, m![1], m![1], m![P], m![B, D, E], m![C # 32]> {
    input.transpose()
}
```

Parameters:

| Parameter   | Value | Notes                |
|-------------|-------|----------------------|
| `in_cols`   | 32    | `D::SIZE × E::SIZE`  |
| `in_rows`   | 8     | `C::SIZE`            |
| `out_rows`  | 32    | `D::SIZE × E::SIZE`  |

Stages:
- **Unpack**: `[C, D, E # 32]` → `[C, D, E]`.
- **Transpose**: `[C, D, E]` → `[D, E, C]`.
- **Trim**: `[D, E, C]` → `[D, E, C]` (no rows trimmed).
- **Align**: `[D, E, C]` → `[D, E, C # 32]`.

Latency: `in_cols = 32 > 16` selects single buffering.
With `in_flits = 32`, `out_rows = 32`, and `n = 2` (B), the total is `2 × (32 + 32) = 128` cycles.

### 16-bit Data Type

This example uses `bf16`, where the wider element halves max `in_rows` (4 instead of 8) and shrinks the 32-byte output flit to 16 elements (instead of 32 for `i8`):

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, C = 8, D = 4, E = 8];

fn bf16_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1], m![P], m![C, D], m![E # 16]>,
) -> TransposeTensor<'l, T, bf16, m![1], m![1], m![P], m![C, E], m![D # 16]> {
    input.transpose()
}
```

Parameters:

| Parameter   | Value | Notes      |
|-------------|-------|------------|
| `in_cols`   | 8     | `E::SIZE`  |
| `in_rows`   | 4     | `D::SIZE`  |
| `out_rows`  | 8     | `E::SIZE`  |

Stages:
- **Unpack**: `[D, E # 16]` → `[D, E]`.
- **Transpose**: `[D, E]` → `[E, D]`.
- **Trim**: `[E, D]` → `[E, D]` (no rows trimmed).
- **Align**: `[E, D]` → `[E, D # 16]`.

Latency: `in_cols = 8 ≤ 16` selects double buffering.
With `in_flits = 4`, `out_rows = 8`, and `n = 8` (C), the total is `4 + 7 × max(4, 8) + 8 = 68` cycles.

### 4-bit Data Type

This example uses `i4`, where `valid_size = 16` doubles the per-cycle element count and max `in_rows` rises to 16 (16 × 4 bits = 8 bytes), while the 32-byte flit grows to 64 elements:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, B = 4, C = 16, E = 16];

fn i4_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, i4, m![1], m![1], m![P], m![B, C], m![E # 64]>,
) -> TransposeTensor<'l, T, i4, m![1], m![1], m![P], m![B, E], m![C # 64]> {
    input.transpose()
}
```

Parameters:

| Parameter   | Value | Notes      |
|-------------|-------|------------|
| `in_cols`   | 16    | `E::SIZE`  |
| `in_rows`   | 16    | `C::SIZE`  |
| `out_rows`  | 16    | `E::SIZE`  |

Stages:
- **Unpack**: `[C, E # 64]` → `[C, E]`.
- **Transpose**: `[C, E]` → `[E, C]`.
- **Trim**: `[E, C]` → `[E, C]` (no rows trimmed).
- **Align**: `[E, C]` → `[E, C # 64]`.

Latency: `in_cols = 16 ≤ 16` selects double buffering.
With `in_flits = 16`, `out_rows = 16`, and `n = 4` (B), the total is `16 + 3 × max(16, 16) + 16 = 80` cycles.

### 32-bit Data Type

This example uses `f32`, one of the two 32-bit formats the [Contraction Engine](./contraction-engine/index.md) emits (`32b × 8`, either `f32` or `i32`).
The wider element drops max `in_rows` to 2 (2 × 4 bytes = 8 bytes) and shrinks the 32-byte flit to 8 elements:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![P = 256, B = 4, D = 2, E = 8];

fn f32_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, f32, m![1], m![1], m![P], m![B, D], m![E # 8]>,
) -> TransposeTensor<'l, T, f32, m![1], m![1], m![P], m![B, E], m![D # 8]> {
    input.transpose()
}
```

Parameters:

| Parameter   | Value | Notes                       |
|-------------|-------|-----------------------------|
| `in_cols`   | 8     | `E::SIZE`                   |
| `in_rows`   | 2     | `D::SIZE`                   |
| `out_rows`  | 8     | `E::SIZE`                   |

Stages:
- **Unpack**: `[D, E # 8]` → `[D, E]`.
- **Transpose**: `[D, E]` → `[E, D]`.
- **Trim**: `[E, D]` → `[E, D]` (no rows trimmed).
- **Align**: `[E, D]` → `[E, D # 8]`.

Latency: `in_cols = 8 ≤ 16` selects double buffering.
With `in_flits = 2`, `out_rows = 8`, and `n = 4` (B), the total is `2 + 3 × max(2, 8) + 8 = 34` cycles.

## Performance

A burst runs `n = OutTime::SIZE / out_rows` transpose iterations.
Each iteration moves `in_flits = in_rows × (in_cols / valid_size)` input flits and `out_rows` output flits.

### Buffering Modes

The Transpose Engine has two internal buffers, each holding 16 columns.
The compiler picks double buffering when `in_cols ≤ 16` and single buffering otherwise: double buffering overlaps input and output to reduce total cycles, while single buffering serializes the two phases.

### Single Buffering Latency

The total burst latency is `n × (in_flits + out_rows)`.
Both buffers are used together, so input and output add together each iteration.

### Double Buffering Latency

The total burst latency is `in_flits + (n - 1) × max(in_flits, out_rows) + out_rows`, which breaks into three phases:

- **Input-only phase** (`in_flits` cycles): the first buffer fills.
- **Overlap phase** (`(n - 1) × max(in_flits, out_rows)` cycles): one buffer receives input while the other produces output simultaneously, so the slower side gates each iteration.
- **Output-only phase** (`out_rows` cycles): the last buffer drains.
