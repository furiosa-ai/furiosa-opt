# Spatial and Temporal Dimensions

`HostTensor<D, E>` uses a single mapping to fully capture its layout.
Device tensors split their layout across multiple dedicated dimensions:

- **Spatial dimensions**: `Chip`, `Cluster`, and `Slice` distribute data across the hardware hierarchy. In stream tensors, `Packet` additionally sizes parallel delivery within each temporal iteration.
- **Temporal dimension**: `Time` sequences the delivery iterations in stream tensors.

## Spatial Dimensions

Each spatial level in the hardware hierarchy gets its own type parameter in the tensor type, enabling spatial parallelism.
All units at each level are assumed to share the same mapping.

```rust
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use std::marker::PhantomData;
// Assumed throughout this page.
axes![A = 8, B = 512];

// HBM tensors
struct HbmTensor<D: Scalar, Chip: M, Element: M> {
    /* ... */
    # _marker: PhantomData<(D, Chip, Element)>,
}

// SRAM tensors
// DM (Data Memory), TRF (Tensor Register File), and VRF (Vector Register File)
struct DmTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M> {
    /* ... */
    # _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}
struct TrfTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Lane: M, Element: M> {
    /* ... */
    # _marker: PhantomData<(D, Chip, Cluster, Slice, Lane, Element)>,
}
struct VrfTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M> {
    /* ... */
    # _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}
```

HBM tensors distribute data across chips for spatial parallelism: each chip processes its own portion of the data simultaneously.
For example, `HbmTensor<bf16, m![A], m![B]>` distributes `8 × 512 = 4096` elements across 8 chips with 512 elements per chip.
The `i`-th chip's `j`-th element stores tensor index `i![A: i, B: j]`.

SRAM tensor types add `Cluster` and `Slice` dimensions for finer-grained parallelism.
`TrfTensor` additionally has a `Lane` dimension that distributes TRF data across the 8 lanes per slice.
See [Contraction Engine](../computing-tensors/contraction-engine/index.md) for details.

Every storage tensor (`HostTensor`, `HbmTensor`, and the SRAM types) places its element data at a starting address.
For example, a `DmTensor<D, ..., Element>` at address `addr` occupies bytes `addr..(addr + Element::SIZE * size_of::<D>())`; `TrfTensor` and `VrfTensor` follow the same pattern.

### Constraints

- **`Chip`, `Cluster`, and `Slice` size**: they must exactly match the hardware counts:

  | Unit      | Count            | Constraint                  | Padding Example        |
  |-----------|------------------|-----------------------------|------------------------|
  | `Chip`    | System-dependent | `Chip::SIZE == NUM_CHIPS`   | `m![1 # NUM_CHIPS]`    |
  | `Cluster` | 2 / Chip         | `Cluster::SIZE == 2`        | `m![1 # 2]`            |
  | `Slice`   | 256 / Cluster    | `Slice::SIZE == 256`        | `m![X / N # 256]`      |

  Any dimension can be padded with `#` when the kernel uses fewer units than the hardware provides.
  For example, `type Cluster = m![1 # 2]` uses 1 active cluster and 1 padding-only cluster, satisfying the hardware's 2-cluster-per-chip requirement.

  > [!NOTE]
  > The runtime operates at chip granularity (`#[device(chip = N)]`), so partial chip or cluster usage is not yet supported.
  > This may be relaxed in future releases.

- **`Element` size**: `Element::SIZE * size_of::<D>()` must not exceed the per-unit SRAM capacity, which varies by tensor type:

  | Type        | Unit          | Constraint                                    |
  |-------------|---------------|-----------------------------------------------|
  | `DmTensor`  | 512KB / Slice | `Element::SIZE * size_of::<D>() <= 512KB`     |
  | `TrfTensor` | 8KB / Lane     | `Lane::SIZE <= 8`, `Element::SIZE * size_of::<D>() <= 8KB` |
  | `VrfTensor` | 8KB / Slice   | `Element::SIZE * size_of::<D>() <= 8KB`      |

- **`Element` alignment**: The starting address must be a multiple of `size_of::<D>()`, because misaligned writes require a read-modify-write cycle that can slow DM access by roughly 50×.

## Temporal Dimension

`TuTensor` represents tensor data flowing through the [Tensor Unit](../computing-tensors/index.md) as a stream.
It retains the same `Chip`, `Cluster`, and `Slice` dimensions as the SRAM types, and adds `Time` and `Packet` for streaming.
`Time` is the temporal dimension: it sequences the delivery iterations.
Unlike the spatial dimensions, `Time` has no hardware-imposed size limit and grows with the amount of data to process.
`Packet` is an additional spatial dimension that determines how many elements each slice receives per temporal iteration.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_opt_std;
# use furiosa_opt_std::prelude::*;
# use std::marker::ConstParamTy;
# use std::marker::PhantomData;
axes![N = 4, C = 64, H = 32, W = 32];

/// Pipeline stage.
/// `Vector` is intentionally absent: the Vector Engine uses a separate typestate
/// (`VectorBranchTensor` and friends) that tracks branch, ALU, and other Vector-specific state.
/// `Commit` is intentionally absent: once the Commit Engine writes results back to DM,
/// the data is at rest and the type becomes `DmTensor`, not `TuTensor`.
# #[derive(PartialEq, Eq, ConstParamTy)]
enum Position {
    Begin,       // After the start of the pipeline
    Fetch,       // After the Fetch Engine
    Switch,      // After the Switch Engine
    Collect,     // After the Collect Engine
    Contraction, // After the Contraction Engine
    Reduce,      // After the Reduce Engine
    Cast,        // After the Cast Engine
    Transpose,   // After the Transpose Engine
}

struct TuTensor<
    'l,                // Lifetime tied to the Tensor Unit context
    const P: Position,
    D: Scalar,
    Chip: M,
    Cluster: M,
    Slice: M,
    Time: M,
    Packet: M,
> {
    /* ... */
    #  _marker: PhantomData<&'l (D, Chip, Cluster, Slice, Time, Packet)>,
}

type T<'l> = TuTensor<
    'l,
    { Position::Fetch }, // Fetch Engine's output
    bf16,
    m![1],           // Chip: single chip
    m![1],           // Cluster: single cluster
    m![C / 2],       // Slice: distribute 64 channels across 32 slices
    m![N, H, W],     // Time: iterate over batch (N) and spatial (H, W) dimensions
    m![C % 2],       // Packet: 2 channels per cycle
>;
```

Type `T` streams a tensor with an aggregate shape of \\(\\{N=4, C=64, H=32, W=32\\}\\) across 32 slices (`Slice::SIZE = m![C / 2]::SIZE = 32`).
The `Time` dimension (`m![N, H, W]`) has size `4 * 32 * 32 = 4096`, which means there are 4,096 temporal iterations.
For each temporal iteration, the `Packet` dimension `m![C % 2]` delivers 2 channels to each slice.
Since 32 slices operate in parallel, each temporal iteration processes `32 * 2 = 64` channels total.
