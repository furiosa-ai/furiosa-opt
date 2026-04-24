# Collect Engine

The Collect Engine normalizes packets to exactly one *flit* (a 32-byte flow control unit that all downstream engines operate on). It follows the Switch Engine in the pipeline, or the Fetch Engine directly when forwarding is implied.

<!-- > **TODO** (jeongmin.park): This page covers packet normalization and pipeline position only. TRF and VRF loading via `.to_trf()` and `.to_vrf()` also happen at the Collect Engine and are not yet documented here. -->

## Interface

```rust,ignore
{{#include ../../../furiosa-visa-std/src/stream_tensor.rs:collect_impl}}
```

## Packet Normalization

The `collect()` method transforms an arbitrary-sized packet into exactly one flit (32 bytes):

1. **Pad** the input packet to the nearest 32-byte boundary (if not already aligned). Skipped if the packet is already 32-byte aligned.
2. **Split** at the flit boundary: the inner 32 bytes become `Packet2`, and the outer flit count is absorbed into `Time2`. Skipped if the padded packet is at most 32 bytes (i.e., fits in one flit).

### Packet = 32 bytes (identity)

`i8`, `B = 32`: packet = 32 elements × 1 byte = 32 bytes = one flit. Nothing changes.

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

### Packet < 32 bytes (pad to one flit)

`i8`, `B = 16`: packet = 16 elements × 1 byte = 16 bytes. Padded to 32 bytes.

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

### Packet > 32 bytes (split into flits)

`bf16`, `B = 32`: packet = 32 elements × 2 bytes = 64 bytes = 2 flits.
The outer flit count (2) is absorbed into Time.

```text
Before:   Time = m![A]
          Packet = m![B]
          ┌──────────────────────────┬──────────────────────────┐
          │       B/16 == 0          │       B/16 == 1          │  64 bytes
          └──────────────────────────┴──────────────────────────┘
                  32 bytes                    32 bytes

After:    Time = m![A, B/16]
          Packet = m![B % 16]
          ┌──────────────────────────┐
          │         B % 16           │  32 bytes  × B/16 time steps
          └──────────────────────────┘
```

Each flit is delivered in a separate time step, so Time grows from `m![A]` to `m![A, B/16]`.

## Pipeline Position

The collect step is mandatory in the Tensor Unit pipeline: all downstream engines (Contraction, TRF/VRF load, etc.) require exactly-32-byte flits, so every execution must pass through `fetch → [switch →] collect` to normalize packets before proceeding.

When no slice redistribution is needed, call `FetchTensor::collect()` directly — no `.switch()` call is required:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, B = 64];

fn direct_collect<'l, const T: Tu>(
    input: FetchTensor<'l, T, i8, m![1], m![1], m![1], m![A], m![B]>,
) -> CollectTensor<'l, T, i8, m![1], m![1], m![1], m![A, B / 32], m![B % 32]> {
    input.collect()
}
```

## Examples

### Single-flit packet (identity)

When the input packet is already exactly 32 bytes, collect passes it through unchanged.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
axes![A = 8, B = 32];

fn collect_identity<'l, const T: Tu>(
    input: SwitchTensor<'l, T, i8, m![1], m![1], m![1], m![A], m![B # 32]>,
) -> CollectTensor<'l, T, i8, m![1], m![1], m![1], m![A], m![B # 32]> {
    // B=32 elements × 1 byte (i8) = 32 bytes = one flit.
    // Time and Packet pass through unchanged.
    input.collect()
}
```

### Sub-flit packet (padding added)

When the input packet is smaller than 32 bytes, collect pads to 32 bytes.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
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

### Multi-flit packet (outer absorbed into Time)

When the input packet exceeds 32 bytes, collect splits into flits and absorbs the outer portion into Time.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
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
