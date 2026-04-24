# Transpose Engine

When computation results are in a different memory layout than DM requires, the Transpose Engine reorders the data within flits before the [Commit Engine](../moving-tensors/commit-engine.md) writes to DM.
The Transpose Engine reorders data within a 2D matrix by swapping rows and columns.
It interprets input data as a `[in_rows, in_cols]` matrix, transposes it, and optionally slices padded elements to produce the desired output shape.

## Interface

```rust,ignore
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
impl<'l, const T: Tu, D, Chip, Cluster, Slice, Time, Packet>
    CollectTensor<'l, T, D, Chip, Cluster, Slice, Time, Packet>
{
    /// Transposes axes between the Time and Packet mappings.
    /// Swaps the innermost Time axes with the Packet axis, converting [A, B] layout to [B, A].
    pub fn transpose<OutTime: M, OutPacket: M>(
        self,
    ) -> TransposeTensor<'l, T, D, Chip, Cluster, Slice, OutTime, OutPacket>
    {
        // Hardware implementation: swaps rows and columns within [Time, Packet]
    }
}
```

The Transpose Engine operates on the `Time` and `Packet` dimensions only. The `Chip`, `Cluster`, and `Slice` dimensions pass through unchanged.

## Architecture

### Conceptual Operation

The Transpose Engine performs four stages:

1. **Unpack**: Each input packet is 32 bytes, but the transpose buffer only uses the first `elements_per_packet` elements (see [Internal Buffer Architecture](#internal-buffer-architecture)).
This stage discards extraneous padding from each packet, keeping only `elements_per_packet` elements.
There are `in_rows` time steps (each delivering `packets_per_col` packets), which assemble the `[in_rows × in_cols]` input matrix, where `in_cols = packets_per_col × elements_per_packet`.
2. **Transpose**: The matrix is transposed: `[in_rows × in_cols]` → `[in_cols × in_rows]`.
3. **Trim**: After transposing, padded elements within each input packet constitute entire rows.
This stage allows the removal of those padded rows, producing `[out_rows × in_rows]`, where `out_rows` <= `in_cols`.
4. **Align**: Each output row is `in_rows` elements wide.
This stage pads each row to 32 bytes (`output_alignment` elements), producing the final output packets of shape `[out_rows × (in_rows # output_alignment)]`.

```text
                 in_cols                 output_alignment
           ┌─────────────────┐         ┌──────────────────┐
           │ 12 13 14 15 ... │         │ 3  7  11 15  ... │
 in_rows   │ 8  9  10 11 ... │  ────►  │ 2  6  10 14  ... │  out_rows
           │ 4  5  6  7  ... │         │ 1  5  9  13  ... │
           │ 0  1  2  3  ... │         │ 0  4  8  12  ... │
           └─────────────────┘         └──────────────────┘
                data_in                      data_out
```

### Specifications

#### Internal Buffer Architecture

The Transpose Engine has two internal buffers, each with `num_buffer_cols = 16` columns. The input interface receives a fixed number of elements per cycle based on the data type:

| Data Type   | `elements_per_packet` |
| ----------- | ---------------------- |
| 4-bit       | 16                     |
| 8/16/32-bit | 8                      |

#### Input Bus Constraints

The input bus to the Transpose Engine is 32 bytes, but its usable capacity depends on the data type:

| Type  | Input Format |
| ----- | ------------ |
| 4-bit | 4b × 16      |
| 8-bit | 8b × 8       |
| 16-bit| 16b × 8      |
| 32-bit| 32b × 8      |

The Transpose Engine receives data from three possible sources:
- [**Contraction Engine**](./contraction-engine/index.md): Outputs 32b × 8
- [**Vector Engine**](./vector-engine/index.md): Outputs 4b × 16, 8b × 8, 16b × 8, or 32b × 8
- [**Fetch Engine**](../moving-tensors/fetch-engine.md): Outputs 4b x 16, 8b x 8, 16b x 8, or 32b x 8

#### Constraints

The following parameters are dependent on the data type:

| Data type | `elements_per_packet` | `output_alignment` | Max `in_rows` | Valid `in_cols`  |
|-----------|-----------------------|--------------------|---------------|------------------|
| 4-bit     | 16                    | 64                 | 16            | 16, 32           |
| 8-bit     | 8                     | 32                 | 8             | 8, 16, 32        |
| 16-bit    | 8                     | 16                 | 4             | 8, 16, 32        |
| 32-bit    | 8                     | 8                  | 2             | 8, 16, 32        |

The following are type-agnostic:

- Both the input and output packets must be 32 bytes.
- `out_rows` <= `in_cols` (determines the number of sliced rows in the **Trim** stage)

## Performance

### Double Buffering

The buffering mode is determined by comparing `in_cols` with `num_buffer_cols`.
Double buffering occurs when `in_cols <= num_buffer_cols`.
Otherwise, single buffering is used.

| `in_cols` | Condition        | Buffering Mode   |
| ------------- | ---------------- | ---------------- |
| 8             | 8 ≤ 16           | Double buffering |
| 16            | 16 ≤ 16          | Double buffering |
| 32            | 32 > 16          | Single buffering |

- **Double buffering**: One buffer receives input while the other produces output simultaneously
- **Single buffering**: Both buffers are used together, so input and output must alternate

### Cycle Calculation

**Variable definitions**:

$$
\texttt{input\_flits\_per\_iter} = \texttt{in\_rows} \times \frac{\texttt{in\_cols}}{\texttt{elements\_per\_packet}}
$$
$$
= \texttt{in\_rows} \times \texttt{packets\_per\_col}
$$
$$
\texttt{output\_flits\_per\_iter} = \texttt{out\_rows}
$$
$$
\texttt{n} = \frac{\texttt{OutTime::SIZE}}{\texttt{out\_rows}}
$$

**Cycles per iteration**:

- **Double buffering**: `max(input_flits_per_iter, output_flits_per_iter)`
  - Input and output happen simultaneously, so the slower one determines the cycle count
- **Single buffering**: `input_flits_per_iter` + `output_flits_per_iter`
  - Input and output alternate, so both are added

**Total cycles in a burst**:

- **Double buffering** (pipelined execution):
$$
\texttt{input\_flits\_per\_iter} + (n - 1) \times \texttt{cycles\_per\_iter} + \texttt{output\_flits\_per\_iter}
$$
  - `input_flits_per_iter`: Initial input-only phase (filling first buffer)
  - `(n - 1) * cycles_per_iter`: Middle phase where input and output overlap
  - `output_flits_per_iter`: Final output-only phase (draining last buffer)

- **Single buffering** (sequential execution):
$$
n \times \texttt{cycles\_per\_iter}
$$

## Examples

### Basic 8×8 Transpose

The simplest case transposes an 8×8 matrix across the `Packet` and `Time` dimensions:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![P = 256, C = 8, D = 8, E = 8];

fn basic_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1], m![P], m![C, D], m![E # 32]>,
) -> TransposeTensor<'l, T, i8, m![1], m![1], m![P], m![C, E], m![D # 32]> {
    // in_rows = 8 (D)
    // packets_per_col = 1,
    // elements_per_packet = 8 (i8),
    // in_cols = packets_per_col * elements_per_packet = 8 (E)
    // out_rows = 8 (E)
    // output_alignment = 32 (i8)

    // 1. Unpack: [in_rows x packets_per_col x packet]: [D, E # 32] →
    //            [in_rows x packets_per_col x elements_per_packet]: [D, E] =
    //            [in_rows x in_cols]
    // 2. Transpose: [in_rows x in_cols]: [D, E] →
    //               [in_cols x in_rows]: [E, D]
    // 3. Trim: [in_cols x in_rows]: [E, D] →
    //          [out_rows x in_rows]: [E, D] (no rows trimmed)
    // 4. Align: [out_rows x in_rows]: [E, D] →
    //           [out_rows x (in_rows # output_alignment)]: [E, D # 32]

    // cycle estimation: in_cols (8) ≤ num_buffer_cols (16), double buffering
    // input_flits_per_iter = 8, output_flits_per_iter = 8, n = 8 (C), cycles_per_iter = 8
    // cycles = input_flits_per_iter + (n - 1) * cycles_per_iter + output_flits_per_iter = 72
    input.transpose()
}
```

### Small Matrix Transpose

Transpose works with matrices smaller than the maximum size:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![P = 64, A = 4, B = 2];

fn small_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1], m![P], m![A], m![B # 32]>,
) -> TransposeTensor<'l, T, i8, m![1], m![1], m![P], m![B], m![A # 32]> {
    // in_rows = 4 (A),
    // packets_per_col = 1,
    // elements_per_packet = 8 (i8),
    // in_cols = packets_per_col * elements_per_packet = 1 * 8 = 8
    //   (B=2 data elements, padded internally to 8)
    // out_rows = 2 (B),
    // output_alignment = 32 (i8)

    // 1. Unpack: [in_rows x packets_per_col x packet]: [A, B # 32] →
    //            [in_rows x packets_per_col x elements_per_packet]: [A, B # 8] =
    //            [in_rows x in_cols]
    // 2. Transpose: [in_rows x in_cols]: [A, B # 8] →
    //               [in_cols x in_rows]: [B # 8, A]
    // 3. Trim: [in_cols x in_rows]: [B # 8, A] →
    //          [out_rows x in_rows]: [B, A] (6 rows trimmed)
    // 4. Align: [out_rows x in_rows]: [B, A] →
    //           [out_rows x (in_rows # output_alignment)]: [B, A # 32]
    // cycle estimation: in_cols (8) ≤ num_buffer_cols (16), double buffering
    // input_flits_per_iter = 4, output_flits_per_iter = 2, n = 1, cycles_per_iter = 4
    // cycles = input_flits_per_iter + (n - 1) * cycles_per_iter + output_flits_per_iter = 6
    input.transpose()
}
```

### Large Column Transpose (`in_cols` = 32, single buffering)

When `in_cols` exceeds `num_buffer_cols`, single buffering is used:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![P = 256, B = 2, C = 8, D = 4, E = 8];

fn large_col_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1], m![P], m![B, C, D], m![E # 32]>,
) -> TransposeTensor<'l, T, i8, m![1], m![1], m![P], m![B, D, E], m![C # 32]> {
    // in_rows = 8 (C),
    // packets_per_col = 4 (D),
    // elements_per_packet = 8 (i8),
    // in_cols = packets_per_col * elements_per_packet = 32 (D * E),
    // out_rows = 32 (D * E),
    // output_alignment = 32 (i8)

    // 1. Unpack: [in_rows x packets_per_col x packet]: [C, D, E # 32] →
    //            [in_rows x packets_per_col x elements_per_packet]: [C, D, E] =
    //            [in_rows x in_cols]
    // 2. Transpose: [in_rows x in_cols]: [C, D, E] →
    //               [in_cols x in_rows]: [D, E, C]
    // 3. Trim: [in_cols x in_rows]: [D, E, C] →
    //          [out_rows x in_rows]: [D, E, C] (no rows trimmed)
    // 4. Align: [out_rows x in_rows]: [D, E, C] →
    //           [out_rows x (in_rows # output_alignment)]: [D, E, C # 32]
    // cycle estimation: in_cols (32) > num_buffer_cols (16), single buffering
    // input_flits_per_iter = 8 * 4 = 32, output_flits_per_iter = 32, n = 2 (B), cycles_per_iter = 32 + 32 = 64
    // cycles = n * cycles_per_iter = 128
    input.transpose()
}
```

### 16-bit Data Type (bf16)

For 16-bit types, the maximum `in_rows` is reduced to 4:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![P = 256, C = 8, D = 4, E = 8];

fn bf16_transpose<'l, const T: Tu>(
    input: CollectTensor<'l, T, bf16, m![1], m![1], m![P], m![C, D], m![E # 16]>,
) -> TransposeTensor<'l, T, bf16, m![1], m![1], m![P], m![C, E], m![D # 16]> {
    // in_rows = 4 (D),
    // packets_per_col = 1,
    // elements_per_packet = 8 (bf16),
    // in_cols = 8 (E),
    // out_rows = 8 (E),
    // output_alignment = 16 (bf16)

    // 1. Unpack: [in_rows x packets_per_col x packet]: [D, E # 16] →
    //            [in_rows x in_cols]: [D, E]
    // 2. Transpose: [in_rows x in_cols]: [D, E] →
    //               [in_cols x in_rows]: [E, D]
    // 3. Trim: [in_cols x in_rows]: [E, D] →
    //          [out_rows x in_rows]: [E, D] (no rows trimmed)
    // 4. Align: [out_rows x in_rows]: [E, D] →
    //           [out_rows x (in_rows # output_alignment)]: [E, D # 16]

    // cycle estimation: in_cols (8) ≤ num_buffer_cols (16), double buffering
    // input_flits_per_iter = 4, output_flits_per_iter = 8, n = 8 (C), cycles_per_iter = 8
    // cycles = input_flits_per_iter + (n - 1) * cycles_per_iter + output_flits_per_iter = 68
    input.transpose()
}
```
