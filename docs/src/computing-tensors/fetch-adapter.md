# Fetch Adapter

The Fetch Adapter applies element-wise transformations (type casting, masking, table lookup, zero-point subtraction) to the packet stream emitted by the [Fetch Engine](../moving-tensors/fetch-engine.md), before the [Switch Engine](./switch-engine.md) routes it across slices.
The Fetch Engine itself does not run any of these transforms; they live here under Computing Tensors and are applied as separate stages between the Fetch Engine and the Switch Engine.
The kernel writer composes the per-stage methods directly on a `FetchTensor`, and each call advances to the next stage.

The adapter has three usable stages, each optional and invoked by calling its method on the stream in hardware pipeline order.
A `FetchTensor` may flow directly into the [Switch Engine](./switch-engine.md) or the [Collect Engine](./collect-engine.md) with no adapter call at all.

- [Table Lookup](#table-lookup) replaces values via a hardware lookup table.
- [Type Casting](#type-casting) converts the element type.
- [Zero-Point Subtraction](#zero-point-subtraction) subtracts a quantization zero point, widening an integer stream to the [Contraction Engine](./contraction-engine/index.md)'s staging type (`i4` to `i5`, `i8` to `i9`).


## Table Lookup

Table lookup provides hardware-accelerated lookup tables during the fetch stage.
Each value is treated as an index into a pre-configured table, and the corresponding table entry is output instead.
This is useful for operations that cannot be efficiently implemented with standard arithmetic, such as non-linear activation functions like Sigmoid and GeLU, or quantization schemes that use custom encoding tables.
This enables:

- **Non-linear activations**: Implements Sigmoid, GeLU, and other functions through pre-computed lookup tables.
- **Custom type casting**: Translates specialized encodings like `MXFP4` / `NVFP4` to standard formats using conversion tables.

Sigmoid and GeLU can also be expressed directly in the [Vector Engine](./vector-engine/index.md), so table lookup is one option among several for these activations rather than the only path.

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/fetch_adapter.rs:fetch_table_lookup_impl}}
```

The decode table is selected by the input scalar type (via the `TableLookup` trait), not by a runtime argument, exactly as `fetch_cast` selects its conversion by the input type.
The key width picks the table and the requested output type picks the entry encoding:

| Input | Output |
|-------|--------|
| `f4e2m1` | `f8e4m3`, `f8e5m2` |
| `f8e4m3` | `bf16` |
| `f8e5m2` | `bf16` |

Widen a decoded 8-bit float to `f32` with a following `fetch_cast`, and apply any NVFP4 / MXFP4 per-block scale downstream in the [Vector Engine](./vector-engine/index.md) or the [Contraction Engine](./contraction-engine/index.md) (it is intentionally not folded into the table, which stays a static 16-entry, block-independent decode).

The table for the `f4e2m1` decode is a compile-time constant baked into the fetch-sequencer configuration, so the `f4e2m1` weight stream is the only data input to the stage; there is no per-invocation table staging.
The 16-entry e2m1 decode is block-independent, which is why the per-block scale stays out of it: folding the scale in would need a distinct table per block and defeat the shared static table.

The hardware sequencer walks byte-aligned keys (1 or 2 bytes), never a raw 4-bit nibble, and a 4-bit key must use a *paired* table.
The `f4e2m1` decode therefore runs as a 256-entry byte-indexed table: each byte carries two nibbles and one lookup yields the pair of decoded `f8e4m3` values, decoding at two elements per key.
The `f4e2m1` scalar is modelled accordingly as a byte holding two nibbles, with `BITS = 4` so the fetch mapping accounts elements rather than bytes.

```rust,ignore
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8];

/// Decodes an e2m1 (NVFP4 / MXFP4) weight stream to f8e4m3 via the hardware table,
/// then widens to f32 with a following fetch_cast.
fn fetch_decode_e2m1<'l, const T: Tu>(
    input: BeginTensor<'l, T, f4e2m1, m![1], m![1], m![1], m![1], m![A]>,
) -> FetchCastTensor<'l, T, f32, m![1], m![1], m![1], m![1], m![A]> {
    input
        .fetch::<m![1], m![A]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
}
```



## Type Casting

`fetch_cast::<OutD>()` converts the element type from `D` to `OutD`, preserving the `Time` and `Packet` mapping.
Type casting adds 1 to 2 cycles of latency.
`fetch_cast` performs only the type conversion; the integer widenings that hold a zero-point offset (`i4` to `i5`, `i8` to `i9`) are a separate stage, [Zero-Point Subtraction](#zero-point-subtraction), so `fetch_cast` never produces an `i5`/`i9`.

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/fetch_adapter.rs:fetch_cast_impl}}
```

RNGD supports the conversions below and no others: every widening lands in the `i32` / `f32` compute width, and `f32` to `bf16` is the only narrowing.

| Input | Output |
|-------|--------|
| `i4` | `i32` |
| `i8` | `i32` |
| `i16` | `i32` |
| `f8e4m3` | `f32` |
| `f8e5m2` | `f32` |
| `bf16` | `f32` |
| `f32` | `bf16` |

In particular the Fetch Adapter has no cast from an 8-bit float to `bf16`, and none from `f32` to an 8-bit float.
To land an `f8e4m3` stream in `bf16`, decode it through the non-paired [Table Lookup](#table-lookup) table instead; to reach an 8-bit float, narrow with the [Cast Engine](./cast-engine.md) later in the pipeline.

The example below fetches an 8-element `i8` stream and casts it to `i32`.
The `Time` and `Packet` mapping is unchanged across the call.

```rust,ignore
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8];

/// Fetches with type casting: converts i8 storage to i32 for computation.
/// Input:   i8 [0, 1, 2, 3, 4, 5, 6, 7]
/// Output: i32 [0, 1, 2, 3, 4, 5, 6, 7]
fn fetch_with_type_cast<'l, const T: Tu>(
    input: BeginTensor<'l, T, i8, m![1], m![1], m![1], m![1], m![A]>,
) -> FetchCastTensor<'l, T, i32, m![1], m![1], m![1], m![1], m![A]> {
    input.fetch::<m![1], m![A]>().fetch_cast::<i32>()
}
#
# let mut ctx = Context::acquire();
# let x: BeginTensor<'_, _, i8, m![1], m![1], m![1], m![1], m![A]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_with_type_cast(x);
```

Type casting adds an additional limit on `read_size`.
The cast output per fetch must fit in a single 32-byte flit (see [Collect Engine](./collect-engine.md)).

- Valid:
  - `i4` -> `i32`, `read_size = 8 (4 bytes)`: produces 8 × 4 = 32 B
  - `i8` -> `i32`, `read_size = 8 (8 bytes)`: produces 8 × 4 = 32 B
- Invalid:
  - `i4` -> `i32`, `read_size = 16 (8 bytes)`: produces 16 × 4 = 64 B
  - `i8` -> `i32`, `read_size = 16 (16 bytes)`: produces 16 × 4 = 64 B

## Zero-Point Subtraction

`fetch_zero_point_sub::<OutD>(zero_point)` subtracts the quantization `zero_point` from each element and widens the stream to the [Contraction Engine](./contraction-engine/index.md)'s staging type: `i4` to `i5`, `i8` to `i9`.
It is the only stage that produces an `i5`/`i9`.

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/fetch_adapter.rs:fetch_zero_point_sub_impl}}
```

### Extra-Bit Rationale

Subtracting the zero point turns an unsigned-around-`zero_point` quantized value into a signed residual whose range no longer fits the input width.
For a symmetric-signed input the residual is a difference of two same-width values:

- `i4` residual: `[-8, 7] - [-8, 7] = [-15, 15]`, which needs `i5`'s `[-16, 15]`.
- `i8` residual: `[-128, 127] - [-128, 127] = [-255, 255]`, which needs `i9`'s `[-256, 255]`.

The subtraction therefore produces one more bit than it consumes.
The conversion checks this at runtime: a residual outside the `i5`/`i9` range (an out-of-range `zero_point` or input) is rejected rather than silently wrapped.

### Contraction-Only Staging

An `i5`/`i9` stream may flow through the [Switch Engine](./switch-engine.md) and [Collect Engine](./collect-engine.md), but from there its **only** legal consumer is `contract_outer`, which pairs it with a weight as [Operand types](./contraction-engine/outer.md#operand-types) lays out.
It cannot be committed to memory, stored to a register file (`to_trf`/`to_vrf`), transposed, or fed to any other engine.

This restriction is enforced at compile time, not by a runtime check: passing an `i5`/`i9` stream to any consumer other than `contract_outer` is a compile error.

