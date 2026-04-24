# Cast Engine

Storing full `f32`/`i32` results in DM would waste memory; the Cast Engine narrows them back to application-specified types (e.g., `bf16`) before the Commit Engine writes to DM.

## Interface

```rust,ignore
{{#include ../../../furiosa-visa-std/src/stream_tensor.rs:cast_impl}}
```

## Precision Lowering

Precision lowering downcasts `f32` or `i32` data into specific lower-precision formats:

| Input Type (`D1`) | Supported Output Types (`D2`) |
| --- | --- |
| `i32` | `i4`, `i8`, `i16` |
| `f32` | `f8e5m2`, `f8e4m3`, `f16`, `bf16` |

## Packet Transformation

The input packet must be exactly 32 bytes (one flit).
The [Collect Engine](./collect-engine.md) ensures this before data reaches the Cast Engine.

After casting each element to the output type, the result is padded back to 32 bytes.
Time passes through unchanged.

```text
Input:  Time = [T],  Packet = [P # (32 / sizeof(D1))],  dtype = D1
Output: Time = [T],  Packet = [P # (32 / sizeof(D2))],  dtype = D2
```

## Examples

### Single-flit packet

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![B = 4, A = 8];

fn cast_i32_to_i8<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1], m![1], m![B], m![A]>,
) -> CastTensor<'l, T, i8, m![1], m![1], m![1], m![B], m![A # 32]> {
    input.cast()
}
```

Before the cast, each flit is fully utilized: `A` = 8 elements x 4 bytes (`i32`) = 32 bytes.
After the cast, each element shrinks to 1 byte (`i8`), so `A` = 8 elements occupy only 8 bytes.
The `A # 32` padding fills the remaining 24 bytes to maintain the 32-byte flit alignment.
Time stays `m![B]` because it passes through unchanged.

### Padded input packet

When the input data doesn't fill the full flit, it arrives already padded from the Collect Engine.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 4];

fn cast_padded<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1], m![1], m![1], m![A # 8]>,
) -> CastTensor<'l, T, i8, m![1], m![1], m![1], m![1], m![A # 32]> {
    input.cast()
}
```

Input packet `A # 8` = 4 data elements padded to 8 elements at `i32` = 32 bytes (one flit).
After cast to `i8`, 4 data elements occupy 4 bytes, padded to 32: `m![A # 32]`.

This under-utilization may look wasteful, but the Cast Engine is a pass-through stage that is never the pipeline bottleneck.
The downstream [Commit Engine](../moving-tensors/commit-engine.md) can aggregate multiple under-utilized flits into dense DM writes anyway.
The net effect is the same: no bandwidth is wasted at the DM level.
