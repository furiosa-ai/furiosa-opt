# Fetch Engine

The Fetch Engine reads a DM tensor and produces a packet stream for the Tensor Unit, with `OutTime` and `OutPacket` choosing the stream consumed by later stages.

Choose Fetch when data in DM must enter the Tensor Unit stream with a specific temporal and packet order.

## Interface

`BeginTensor` represents a tensor resident in DM, at the entry of the Tensor Unit pipeline.
Its `Time` is `m![1]` (no temporal iteration before the pipeline starts) and `Packet` is the element layout in DM.

`BeginTensor::fetch()` runs the sequencer and produces a `FetchTensor` packet stream.
The stream can feed the [Fetch Adapter](../computing-tensors/fetch-adapter.md), [Switch Engine](../computing-tensors/switch-engine.md), or [Collect Engine](../computing-tensors/collect-engine.md).
The `assert_eq!` calls enforce hardware constraints on `Cluster::SIZE`, `Slice::SIZE`, and packet alignment (see [Constraints](#constraints)).

```rust,ignore
{{#include ../../../furiosa-opt-std/src/engine/fetch.rs:fetch_impl}}
```

As introduced in [Mapping Tensors](../mapping-tensors/index.md), the `Chip`, `Cluster`, `Slice`, `Time`, `Packet` mapping distributes data across space and time.
`.fetch()` preserves the `Chip`, `Cluster`, and `Slice` dimensions unchanged from the input, because each slice independently reads its own DM partition.
Later the [Switch Engine](../computing-tensors/switch-engine.md) changes the `Slice` mapping by moving data across slices.

`fetch()` takes `OutTime` and `OutPacket` type parameters that configure the Fetch Sequencer.
`OutTime` sets the number of time steps in the output stream, and `OutPacket` sets the element layout within each packet.
For performance implications of `OutPacket` choices, see [Optimizations](#optimizations).

The following example fetches an `i8` matrix from DM as an `i8` packet stream.
The output `FetchTensor` streams 512 time steps, each a 32-element `i8` packet (32 bytes).
Here `OutTime = m![A]` and `OutPacket = m![B]`.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![CH = 4, CL = 2, S = 256, A = 512, B = 32];

fn fetch_matrix_example<'l, const T: Tu>(
    input: BeginTensor<'l, T, i8, m![CH], m![CL], m![S], m![1], m![A, B]>,
) -> FetchTensor<'l, T, i8, m![CH], m![CL], m![S], m![A], m![B]> {
    input.fetch::<m![A], m![B]>()
}
```

`Chip`, `Cluster`, and `Slice` are the hardware spatial parallelism dimensions.
A Fetch Sequencer runs independently in every slice, each operating on its own local DM partition.
In this example, `CH = 4`, `CL = 2`, and `S = 256` describe a 4-chip system with two clusters per chip and 256 slices per cluster.
Each slice runs the same sequencer over its own `A×B` sub-tensor.


## Constraints

- **Hardware dimensions**: `Chip::SIZE`, `Cluster::SIZE`, and `Slice::SIZE` must match the hardware configuration (see [Sequencer](./sequencer.md#architecture)).

## Multi-Read Packet

Preparing a packet may require multiple hardware reads because packet axes may not be contiguous in DM, and the hardware reads at most 32 bytes at once.
In the [main-context](../computing-tensors/index.md#execution-context), `read_size` is the largest divisor of `max_access_size` for which `D[read_size]` is 1, 2, 4, 8, 16, or 32 bytes.
See [Sequencer Architecture](./sequencer.md#access-size) for `max_access_size`.
In the [sub-context](../computing-tensors/index.md#execution-context), `read_size` is fixed at 8 bytes.
The compiler derives `read_size` from the input element type and any [Fetch Adapter](../computing-tensors/fetch-adapter.md) cast.
Users do not set it directly.
Multi-read occurs whenever `Packet::SIZE > read_size`.
For example, a 24-byte packet in the main-context forces `read_size = 8` and 3 reads per packet.
The total cycle count is `Time::SIZE * (Packet::SIZE / read_size)`.

The following examples fetch the same `i4` tensor of shape `m![N, C, H, W]` (with `N=4, C=3, H=4, W=16`) using four different `OutPacket` choices.
```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![N = 4, C = 3, H = 4, W = 16];

/// Sequencer config: [N = 4 : 192, C = 3 : 64, H = 4 : 16, W = 16 : 1].
/// max_access_size = 16; read_size = 16 (8 bytes); reads per packet = 1; cycles = 48
fn fetch_batch_1<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![N, C, H], m![W]> {
    input.fetch()
}

/// Sequencer config: [N = 4 : 192, C = 3 : 64, H / 2 = 2 : 32, H % 2 = 2 : 16, W = 16 : 1].
/// max_access_size = 32; read_size = 32 (16 bytes); reads per packet = 1; cycles = 24
fn fetch_batch_2<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![N, C, H / 2], m![H % 2, W]> {
    input.fetch()
}

/// Sequencer config: [N = 4 : 192, C = 3 : 64, H = 4 : 16, W = 16 : 1].
/// max_access_size = 64; read_size = 64 (32 bytes); reads per packet = 1; cycles = 12
fn fetch_batch_3<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![N, C], m![H, W]> {
    input.fetch()
}

/// Sequencer config: [N = 4 : 192, C = 3 : 64, H = 4 : 16, W = 16 : 1].
/// max_access_size = 192; read_size = 64 (32 bytes); reads per packet = 3; cycles = 12
fn fetch_batch_4<'l, const T: Tu>(
    input: BeginTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]>,
) -> FetchTensor<'l, T, i4, m![1], m![1 # 2], m![1 # 256], m![N], m![C, H, W]> {
    input.fetch()
}
#
# let mut ctx = Context::acquire();
#
# let b: BeginTensor<'_, _, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_batch_1(b);
#
# let b: BeginTensor<'_, _, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_batch_2(b);
#
# let b: BeginTensor<'_, _, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_batch_3(b);
#
# let b: BeginTensor<'_, _, i4, m![1], m![1 # 2], m![1 # 256], m![1], m![N, C, H, W]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_batch_4(b);
```

## Interleaving

Interleaving combines two tensors with identical mappings into a single sequencer operation, reducing overhead when both tensors are needed for the same computation.
An explicit `Time` axis encodes alternation between the two tensors.

In the following example, the main-context creates an interleaved tensor using `begin_interleaved()`.
The first temporal iteration fetches from `lhs`, the second from `rhs`, the third from `lhs` again, and so on.
At most two tensors can be interleaved in a single fetch operation.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 16, B = 32, I = 2];

/// Interleaves two input tensors into a single packet stream.
/// Useful for operations like 'input1 + input2' in the Vector Engine.
/// The interleaved BeginTensor is created via Tu.begin_interleaved().
/// The `I = 2` axis in Time encodes alternation between the two tensors.
fn fetch_interleaved<'l>(
    ctx: &'l mut Context,
    lhs: &'l DmTensor<i8, m![1], m![1 # 2], m![1 # 256], m![A, B]>,
    rhs: &'l DmTensor<i8, m![1], m![1 # 2], m![1 # 256], m![A, B]>,
) -> FetchTensor<'l, { Tu::Main }, i8, m![1], m![1 # 2], m![1 # 256], m![A, I], m![B]> {
    ctx.main.begin_interleaved::<I, _, _, _, _, _>(lhs.view(), rhs.view()).fetch()
}
#
# let mut ctx = Context::acquire();
#
# let lhs = DmTensor::new();
# let rhs = DmTensor::new();
# let _o = fetch_interleaved(&mut ctx, &lhs, &rhs);
```

## Optimizations

Three factors determine Fetch Sequencer throughput.

- **Input bandwidth**: `read_size` is limited by axis contiguity in DM and packet size.
  Non-adjacent axes reduce `max_access_size` and therefore `read_size` (see [Non-Contiguous Packets](./sequencer.md#non-contiguous-packets)).
  A packet smaller than the contiguous run also limits `read_size`.
  Padding to a larger power-of-two raises it (see [Packet padding](#example-packet-padding)).

  Repeated access to the same bank can starve lower-priority [Commit Engine](./commit-engine.md) and [DMA Engine](./dma-engine.md) operations.
The limit is 64 consecutive accesses; exceeding it can cause a NoC timeout.

  See [Memory Performance](./memory-performance.md) for details.
- **Output bandwidth**: the downstream [Collect Engine](../computing-tensors/collect-engine.md) converts Fetch's packets to 32-byte *flits*, so packet sizes that don't align to 32 bytes waste bandwidth.
  A 20-byte packet fills one flit with 12 bytes of zero-padding, wasting `12 / 32 = 37.5%`.
  A 40-byte packet spans two flits (64 bytes total) and zero-pads the final 24 bytes of the second flit, wasting `24 / 64 = 37.5%`.
- **Spatial parallelism**: Distributing fetches across slices maximizes throughput.

### Example: Packet padding

Padding `OutPacket` to a larger power-of-two element count can increase `read_size`.
The three examples below fetch the same 30-byte tensor in 15, 3, and 1 cycles by growing the packet from 2 to 16 to 32 bytes:

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
axes![A = 3, B = 5, C = 2];

/// Smallest packet: only C dimension padded to 8bytes. Takes 15 cycles.
fn fetch_packet_C<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![A, B], m![C # 8]> {
    input.fetch()
}

/// Medium packet: B and C dimensions padded to 16 bytes. Takes 3 cycles.
fn fetch_packet_BC<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![A], m![[B, C] # 16]> {
    input.fetch()
}

/// Largest packet: all dimensions padded to 32 bytes. Takes 1 cycle.
fn fetch_packet_ABC<'l, const T: Tu>(
    input: BeginTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]>,
) -> FetchTensor<'l, T, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![[A, B, C] # 32]> {
    input.fetch()
}

#
# let mut ctx = Context::acquire();
# let x: BeginTensor<'_, _, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_packet_C(x);
# let y: BeginTensor<'_, _, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_packet_BC(y);
# let z: BeginTensor<'_, _, f8e4m3, m![1], m![1 # 2], m![1 # 256], m![1], m![A, B, C]> = BeginTensor::new(&mut ctx.main, Tensor::zero());
# let _o = fetch_packet_ABC(z);
```

In these examples, padding reads beyond the actual data, but this is safe because padding values do not affect computation.
Different padding strategies produce different `FetchTensor` mappings, which may affect downstream components.

The same `A = 3`, `B = 5`, `C = 2` `f8e4m3` tensor can use these packet mappings:

| Function | Fetch `Time` | Fetch `Packet` | Fetch work | Collect padding |
|----------|--------------|----------------|------------|-----------------|
| `fetch_packet_c` | `A, B` | `C # 8` | 15 packets of 8 bytes | 24 bytes for each packet. |
| `fetch_packet_bc` | `A` | `[B, C] # 16` | 3 packets of 16 bytes | 16 bytes for each packet. |
| `fetch_packet_abc` | `1` | `[A, B, C] # 32` | 1 packet of 32 bytes | None. |

The 8-byte and 16-byte candidates require Collect to pad each packet to a 32-byte flit.
The 32-byte candidate reaches Collect as one full flit, but its extra physical capacity can increase DM usage.
The schedule comparison decides whether fewer fetches outweigh the padding and downstream layout cost for the fixed workload.
