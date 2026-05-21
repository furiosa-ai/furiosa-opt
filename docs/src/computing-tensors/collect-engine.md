# Collect Engine

All downstream engines ([Contraction Engine](./contraction-engine/index.md), [Vector Engine](./vector-engine/index.md), [Cast Engine](./cast-engine.md), [Transpose Engine](./transpose-engine.md), and [Commit Engine](../moving-tensors/commit-engine.md)) consume exactly 32-byte *flits*.
The Collect Engine normalizes arbitrary-sized packets to one flit in two steps:
1. **Pad** the input packet up to the next 32-byte boundary.
   Skipped if the packet is already 32-byte aligned.
2. **Split** at the flit boundary: the inner 32 bytes become `Packet2`, and the outer flit count is absorbed into `Time2`.
   Skipped if the packet is already 32 bytes.

The resulting `CollectTensor` either flows down the pipeline to a downstream engine or is stored in the [Register Files](./register-files.md).

## Interface

`SwitchTensor` and `FetchTensor` both expose `.collect()` with the same semantics.
The `FetchTensor` entry point bypasses the Switch Engine when no slice distribution is needed.

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/collect.rs:collect_impl}}
```

## Examples

### Single-Flit Packet

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 32];

fn collect_identity<'l, const T: Tu>(
    input: SwitchTensor<'l, T, i8, m![1], m![1], m![1], m![A], m![B]>,
) -> CollectTensor<'l, T, i8, m![1], m![1], m![1], m![A], m![B # 32]> {
    // B=32 elements × 1 byte (i8) = 32 bytes = one flit.
    // Time and Packet pass through unchanged.
    input.collect()
}
```

When the input packet is already exactly 32 bytes, `collect` passes it through unchanged (`B = 32` elements × 1 byte for `i8` = 32 bytes).

```text
Before:   Time = m![A]
          Packet = m![B]
          ┌──────────────────────────┐
          │            B             │  32 bytes
          └──────────────────────────┘

After:    Time = m![A]
          Packet = m![B # 32]
          ┌──────────────────────────┐
          │          B # 32          │  32 bytes
          └──────────────────────────┘
```

### Sub-Flit Packet

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 16];

fn collect_padding<'l, const T: Tu>(
    input: SwitchTensor<'l, T, i8, m![1], m![1], m![1], m![A], m![B]>,
) -> CollectTensor<'l, T, i8, m![1], m![1], m![1], m![A], m![B # 32]> {
    // B=16 elements × 1 byte = 16 bytes < 32 bytes.
    // Padded to 32 bytes: Packet2 = m![B # 32].
    // Time unchanged since it fits in one flit.
    input.collect()
}
```

When the input packet is smaller than 32 bytes, `collect` pads to 32 bytes (`B = 16` elements × 1 byte for `i8` = 16 bytes).

```text
Before:   Time = m![A]
          Packet = m![B]
          ┌────────────┐
          │     B      │  16 bytes
          └────────────┘

After:    Time = m![A]
          Packet = m![B # 32]
          ┌────────────┬─────────────┐
          │     B      │     pad     │  32 bytes
          └────────────┴─────────────┘
```

### Multi-Flit Packet

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 32];

fn collect_multi_flit<'l, const T: Tu>(
    input: SwitchTensor<'l, T, bf16, m![1], m![1], m![1], m![A], m![B]>,
) -> CollectTensor<'l, T, bf16, m![1], m![1], m![1], m![A, B / 16], m![B % 16]> {
    // B=32 elements × 2 bytes (bf16) = 64 bytes = 2 flits.
    // Inner 16 elements = 32 bytes → Packet2 = m![B % 16].
    // Outer 2 flits → absorbed into Time2 = m![A, B / 16].
    input.collect()
}
```

When the input packet exceeds 32 bytes, `collect` splits into flits and absorbs the outer flit count into Time (`B = 32` elements × 2 bytes for `bf16` = 64 bytes, so `B / 16 = 2` flits).

```text
Before:   Time = m![A]
          Packet = m![B]
          ┌──────────────────────────┬──────────────────────────┐
          │       B / 16 == 0        │       B / 16 == 1        │  64 bytes
          └──────────────────────────┴──────────────────────────┘
                    32 bytes                   32 bytes

After:    Time = m![A, B / 16]
          Packet = m![B % 16]
          ┌──────────────────────────┐
          │          B % 16          │  32 bytes  × B/16 time steps
          └──────────────────────────┘
```

### Multi-Flit Packet With Padding

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 8, B = 51];

fn collect_multi_flit_padded<'l, const T: Tu>(
    input: SwitchTensor<'l, T, i8, m![1], m![1], m![1], m![A], m![B]>,
) -> CollectTensor<'l, T, i8, m![1], m![1], m![1], m![A, B # 64 / 32], m![B # 64 % 32]> {
    // B is not 32-byte aligned; first pad B to a multiple of 32 bytes.
    // B # 64=64 elements × 1 byte (i8) = 64 bytes = 2 flits.
    // Inner 32 elements = 32 bytes → Packet2 = m![B # 64 % 32].
    // Outer 2 flits → absorbed into Time2 = m![A, B # 64 / 32].
    input.collect()
}
```

When the input packet is not aligned to 32 bytes, it is first padded (`B = 51` elements × 1 byte for `i8` = 51 bytes, padded to 64).
Then, `collect` splits into flits and absorbs the outer flit count (`B # 64 / 32 = 2`) into Time.

```text
Before:   Time = m![A]
          Packet = m![B]
          ┌──────────────────────────┬───────────────┐
          │       B / 32 == 0        │  B / 32 == 1  │  51 bytes
          └──────────────────────────┴───────────────┘
                    32 bytes             19 bytes

Padded:   Time = m![A]
          Packet = m![B # 64]
          ┌──────────────────────────┬───────────────┬──────────┐
          │       B / 32 == 0        │  B / 32 == 1  │   pad    │  64 bytes
          └──────────────────────────┴───────────────┴──────────┘
                    32 bytes                   32 bytes

After:    Time = m![A, B # 64 / 32]
          Packet = m![B # 64 % 32]
          ┌──────────────────────────┐
          │       B # 64 % 32        │  32 bytes  × B # 64 / 32 time steps
          └──────────────────────────┘
```

## Register File Loading

After normalization, store the `CollectTensor` into the Tensor Register File via [`.to_trf()`](#to-trf) or the Vector Register File via [`.to_vrf()`](#to-vrf).
The "To TRF" / "To VRF" subsections below describe the store mechanism (`time_inner` derivation, `[time_inner, Packet]` sequenced into `Element`).

### To TRF

`.to_trf::<Row, Element>(address)` partitions the TRF along its row dimension.
The kernel writer chooses `Row` (the row layout in the TRF, with `Row::SIZE` in {1, 2, 4, 8}) and `Element` (the per-row element layout).
The compiler then finds a `time_inner` such that `Time` decomposes into `[Row, time_inner]` and `[time_inner, Packet]` is sequenced into `Element`, so each row of the TRF is filled by `time_inner` consecutive flits.

`address` is a `TrfAddress` that selects the TRF region:
- `Full`: the entire TRF.
- `FirstHalf` / `SecondHalf`: the TRF split into two halves, allowing two tensors to occupy it independently.

The compiler bounds the resulting tensor's total byte size by the chosen region's capacity.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 64];

fn load_trf<'l, const T: Tu>(
    input: CollectTensor<'l, T, i8, m![1], m![1 # 2], m![1 # 256], m![1], m![B]>,
) -> TrfTensor<i8, m![1], m![1 # 2], m![1 # 256], m![1], m![B]> {
    input.to_trf(TrfAddress::Full)
}
```

### To VRF

`.to_vrf::<Element>(address)` stores the flits into the VRF at a raw `Address` (no bounded-region selection).
The kernel writer chooses `Element`, the destination element layout in the VRF.
Unlike `.to_trf` (which accepts any `Scalar` element type), `.to_vrf` requires a `VeScalar` element type (i.e., `i32` or `f32`) because the Vector Engine downstream consumes these types only.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![B = 64];

fn load_vrf<'l, const T: Tu>(
    input: CollectTensor<'l, T, i32, m![1], m![1 # 2], m![1 # 256], m![B / 8], m![B % 8]>,
) -> VrfTensor<i32, m![1], m![1 # 2], m![1 # 256], m![B]> {
    input.to_vrf(0)
}
```

