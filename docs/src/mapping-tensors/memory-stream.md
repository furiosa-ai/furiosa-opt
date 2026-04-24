# Memory and Stream

The [Mapping Expressions](./mapping-expressions.md) page covered host tensors, which use a single flat buffer.
Device tensors extend host tensors in two directions: storage adds spatial dimensions (chips, clusters, slices) to match the hardware hierarchy, while data flowing through the Tensor Unit pipeline takes a streaming form with `Time` and `Packet` dimensions.
Both arise from the same hardware distinction between static storage and pipeline flow.


## HBM and SRAM

(See [Formal Definition](#formal-definition) at the end of this page for the precise buffer-to-tensor correspondence.)

Device memory has multiple levels, each with its own geometry.
Each level is represented as a separate type parameter, enabling spatial parallelism:

```rust
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
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
struct TrfTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Row: M, Element: M> {
    /* ... */
    # _marker: PhantomData<(D, Chip, Cluster, Slice, Row, Element)>,
}
struct VrfTensor<D: Scalar, Chip: M, Cluster: M, Slice: M, Element: M> {
    /* ... */
    # _marker: PhantomData<(D, Chip, Cluster, Slice, Element)>,
}
```

HBM tensors distribute data across chips for spatial parallelism (processing different data elements simultaneously on different hardware units).
For example, `HbmTensor<bf16, m![A], m![B]>` distributes `8 × 512 = 4096` elements across 8 chips with 512 elements per chip.
The `i`-th chip's `j`-th element stores tensor index `i![A = i, B = j]`.

SRAM tensor types add `Cluster` and `Slice` dimensions for finer-grained parallelism.
`TrfTensor` additionally has a `Row` dimension that distributes weight data across the 8 MAC rows per slice.
See [Contraction Engine](../computing-tensors/contraction-engine/index.md) for details.

These tensor types assume all units at each level share the same mapping.
The type parameters directly mirror the device structure, avoiding complex address calculations that would arise from flattening multi-dimensional storage into linear indices.

### Alignment Constraint

Alignment constraints apply to the `Element` dimension: the starting address must be a multiple of `size_of::<D>()`.
This ensures natural alignment for maximum throughput.
The `Chip`, `Cluster`, and `Slice` dimensions have no additional alignment constraints.

### Size Constraint

Each dimension must fit within hardware limits.
Each chip has 256MB of SRAM: 2 clusters × 256 slices × 512KB per slice.
An 8-chip system provides 2GB total SRAM capacity.

All device tensor types share the following spatial constraints:

| Unit    | Count            | Constraint                  | Padding Required       |
|---------|------------------|-----------------------------|------------------------|
| Chip    | System-dependent | `Chip::SIZE == NUM_CHIPS`   | `m![1 # NUM_CHIPS]`    |
| Cluster | 2 / Chip         | `Cluster::SIZE == 2`        | `m![1 # 2]`            |
| Slice   | 256 / Cluster    | `Slice::SIZE == 256`        | `m![X / N # 256]`      |

> [!NOTE]
> These exact-match constraints are a current limitation: the runtime operates at chip granularity (`#[device(chip = N)]`), so partial chip or cluster usage is not yet supported.
> Use the `#` padding operator to fill unused positions.
> This may be relaxed in future releases.

The `Element` dimension varies by tensor type:

| Type        | Unit         | Constraint                                    |
|-------------|--------------|-----------------------------------------------|
| `DmTensor`  | 512KB / Slice | `Element::SIZE * size_of::<D>() <= 512KB`     |
| `TrfTensor` | 8KB / Row    | `Row::SIZE <= 8`, `Element::SIZE * size_of::<D>() <= 8KB` |
| `VrfTensor` | 8KB / Slice  | `Element::SIZE * size_of::<D>() <= 8KB`      |

When a kernel uses fewer clusters than the hardware provides, the `Cluster` dimension is padded.
For example, a single-cluster kernel uses `type Cluster = m![1 # 2]`, meaning 1 logical cluster padded to the hardware's 2 clusters per chip.
A `DmTensor<D, ..., Element>` at address `addr` occupies `addr..(addr + Element::SIZE * size_of::<D>())`.

## Tensor Unit Stream

While tensor data is stored in DM in a compact, storage-optimized layout, the [Tensor Unit](../computing-tensors/index.md) receives tensor data as streams of elements delivered over time.
The `Packet` dimension determines how many elements are delivered to the Tensor Unit in a single cycle.
[Fetch Sequencers](../moving-tensors/sequencer.md) read DM data chunks and deliver a portion each clock cycle.

The `Time` dimension models this sequence of data delivery.
Unlike spatial dimensions that are constrained by hardware capacity, `Time` has no hardware-imposed size limit; it grows with the amount of data to process.

```rust
# #![feature(adt_const_params)]
# extern crate furiosa_visa_std;
# use furiosa_visa_std::prelude::*;
# use std::marker::ConstParamTy;
# use std::marker::PhantomData;
axes![N = 4, C = 64, H = 32, W = 32];

/// Pipeline stage.
/// `Vector` is intentionally absent: the Vector Engine uses a separate typestate
/// (`VectorBranchTensor` and friends) that tracks branch, ALU, and other Vector-specific state.
/// `Commit` is intentionally absent: once the Commit Engine writes results back to DM,
/// the data is at rest and the type becomes `DmTensor`, not `StreamTensor`.
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

struct StreamTensor<
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

type T<'l> = StreamTensor<
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
The `Time` dimension (`m![N, H, W]`) has size `4 * 32 * 32 = 4096`, which means there are 4096 temporal iterations or cycles.
Each cycle, the `Packet` dimension `m![C % 2]` delivers 2 channels to each slice.
Since 32 slices operate in parallel, each cycle processes `32 * 2 = 64` channels total.

## Formal Definition

The following formalizes the buffer-to-tensor correspondence described above for multi-dimensional storage.

For an HBM tensor holding tensor \\(T\\), the correspondence is:
- For every chip index `i`, element index `j`, and corresponding tensor indices `ti`, `tj`:
- if `Chip::map(i) = Some(ti)` and `Element::map(j) = Some(tj)`,
- then the `i`-th chip's `j`-th element stores the value of tensor \\(T\\) at index `ti ∪ tj` (the union of the two partial tensor indices).

The same principle extends to SRAM tensors: for a `DmTensor`, the correspondence additionally requires matching `Cluster::map` and `Slice::map` indices, with the tensor index being the union of all four partial indices.
`TrfTensor` further adds a `Row` index.
Stream tensors add `Time` and `Packet` dimensions: the `Time` dimension indexes which cycle delivers the data, and the `Packet` dimension indexes elements within a single cycle.
