# Mapping Tensors

A mapping is the source of truth for logical order and physical placement.
Validate its cells and dimensions before selecting movement or compute operations.
Every mapping expression maps a logical index to a physical location.
Read the result categories defined in [Position Cell Summary](#position-cell-summary) before selecting movement or compute operations.

## Place and Reduce Dimensions

The table places axes before movement or compute operations are selected.

| Dimension | Type | Defined in | Reduced in |
|-----------|------|------------|------------|
| `Chip` | Spatial | [HBM, DM](./spatial-temporal-dimensions.md#spatial-dimensions), [Stream](./spatial-temporal-dimensions.md#temporal-dimension) | [DMA](../moving-tensors/dma-engine.md) + [Vector](../computing-tensors/vector-engine/index.md) |
| `Cluster` | Spatial | [DM](./spatial-temporal-dimensions.md#spatial-dimensions), [Stream](./spatial-temporal-dimensions.md#temporal-dimension) | [DMA](../moving-tensors/dma-engine.md) + [Vector](../computing-tensors/vector-engine/index.md) |
| `Slice` | Spatial | [DM](./spatial-temporal-dimensions.md#spatial-dimensions), [Stream](./spatial-temporal-dimensions.md#temporal-dimension) | [Vector](../computing-tensors/vector-engine/index.md) |
| `Lane` | Spatial | [TRF](./spatial-temporal-dimensions.md#spatial-dimensions) | [Contraction](../computing-tensors/contraction-engine/index.md) |
| `Time` | Temporal | [Stream](./spatial-temporal-dimensions.md#temporal-dimension) | [Contraction](../computing-tensors/contraction-engine/index.md) |
| `Packet` | Spatial | [Stream](./spatial-temporal-dimensions.md#temporal-dimension) | [Contraction](../computing-tensors/contraction-engine/index.md) |

Chip and Cluster reductions use DMA redistribution followed by Vector Engine reduction.

## Mapping Order and Performance

A tensor has values but no intrinsic storage order.
A mapping chooses that order, and hardware accesses are most efficient when adjacent values occupy contiguous buffer positions.
The outermost mapping dimension is major and changes slowest.
The innermost is minor and changes fastest.
For a height-by-width tensor, `m![H, W]` makes rows contiguous, while `m![W, H]` makes columns contiguous.
Each order favors locality along a different axis: an H-major mapping makes a scan along W contiguous, while a W-major mapping makes a scan along H contiguous.
A tiled mapping such as `m![H / 2, W / 2, H % 2, W % 2]` places small two-dimensional neighborhoods contiguously, so neither H nor W is globally the sole contiguous direction.
This changes which access patterns map to nearby buffer positions, but does not inherently guarantee faster hardware access.
The additional mapping state needed to represent the split dimensions is its main extra cost.

For axes `H` (six rows) and `W` (eight columns), the same values can therefore have different physical access patterns:

| H\\W | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | a | b | c | d | e | f | g | h |
| 1 | i | j | k | l | m | n | o | p |
| 2 | · | · | · | · | · | · | · | · |
| 3 | · | · | · | · | · | · | · | · |
| 4 | · | · | · | · | · | · | · | · |
| 5 | · | · | · | · | · | · | · | · |

- **H-major, W-minor: `m![H, W]`** — a scan along W is contiguous.
  A scan along H touches one value per cache line.

  <table>
    <tr>
      <th colspan="8" align="center">H=0</th>
      <th colspan="8" align="center">H=1</th>
      <th align="center">...</th>
    </tr>
    <tr>
      <td>a</td><td>b</td><td>c</td><td>d</td><td>e</td><td>f</td><td>g</td><td>h</td>
      <td>i</td><td>j</td><td>k</td><td>l</td><td>m</td><td>n</td><td>o</td><td>p</td>
      <td>...</td>
    </tr>
  </table>

- **W-major, H-minor: `m![W, H]`** — a scan along H is contiguous.
  A scan along W touches one value per cache line.

  <table>
    <tr>
      <th colspan="6" align="center">W=0</th>
      <th colspan="6" align="center">W=1</th>
      <th colspan="6" align="center">W=2</th>
      <th align="center">...</th>
    </tr>
    <tr>
      <td>a</td><td>i</td><td>·</td><td>·</td><td>·</td><td>·</td>
      <td>b</td><td>j</td><td>·</td><td>·</td><td>·</td><td>·</td>
      <td>c</td><td>k</td><td>·</td><td>·</td><td>·</td><td>·</td>
      <td>...</td>
    </tr>
  </table>

- **2×2 tiles: `m![H / 2, W / 2, H % 2, W % 2]`** — H-major and W-major each sacrifice locality along one axis.
  Tiling places small neighborhoods contiguously along both axes, changing the locality distribution without inherently guaranteeing faster hardware access.
  The split dimensions require a non-trivial address formula and additional mapping state.

  <table>
    <tr>
      <th colspan="4" align="center">t(0,0)</th>
      <th colspan="4" align="center">t(0,1)</th>
      <th colspan="4" align="center">t(0,2)</th>
      <th align="center">...</th>
    </tr>
    <tr>
      <td>a</td><td>b</td><td>i</td><td>j</td>
      <td>c</td><td>d</td><td>k</td><td>l</td>
      <td>e</td><td>f</td><td>m</td><td>n</td>
      <td>...</td>
    </tr>
  </table>

The selected mapping also constrains later execution.
Changing a representation's order after allocation generally requires copying or transposing its data, so allocation-time choices affect subsequent operations.
Hardware geometry, alignment, and scheduling remain architectural constraints that compiler lowering must account for.
The mapping expresses the logical order rather than a raw address calculation.
Device-specific alignment checks apply, and misaligned accesses can require read-modify-write cycles with substantial performance cost (historically observed at roughly 50× for affected DM accesses).

The mapping is the complete technical description of this choice.
A physical representation is the concrete buffer, device placement, and stream decomposition produced from that mapping.
The same values can move between different physical representations without changing the logical tensor.

## Position Cell Summary

Every mapping expression maps a physical position to a `Cell`.
  Identify the result category before interpreting any coordinates.

```rust
# extern crate furiosa_opt_std;
# extern crate furiosa_mapping;
# use furiosa_opt_std::prelude::*;
axes![A = 4];
type E = m![A];

assert_eq!(E::map(0), i![A: 0]);
assert_eq!(E::map(4), Cell::OutOfBounds);
```

`Cell::Index` carries live `Index` terms.
`Cell::OutOfBounds` identifies a position outside the mapping.
`Cell::Padding` is introduced with padding in [Mapping Expressions](./mapping-expressions.md).

This section summarizes the result categories.
After identifying the `Cell`, use [Mapping Expressions](./mapping-expressions.md) for constructors and API syntax.

## Representation and Distribution

### Physical Representations

Storage and stream tensors split one mapping across hardware dimensions.
An HBM representation may distribute channels across chips.
A DM representation may repartition those channels across slices.
A stream representation may put the remaining dimensions in `Time` and `Packet`.
[Spatial and Temporal Dimensions](./spatial-temporal-dimensions.md) follows one NCHW tensor through those representations with a concrete coordinate trace.

This representation trace closes the chapter: choose a mapping, derive each physical representation, then verify that the value at a traced `Index` reaches the expected stream cell.
The formal tensor-value definition is in [Tensor Semantics](./tensor-semantics.md).
This chapter keeps the explanation operational.

For a minimal computation over a representation, see the [Vector Engine](../computing-tensors/vector-engine/index.md).
Elementwise operations belong with that execution API, while this chapter focuses on mapping and representation.

### Distribution Across Space and Time

Mapping dimensions are assigned explicitly to spatial hardware dimensions (`Chip`, `Cluster`, `Slice`, and `Packet`) or to the temporal `Time` loop.
These names describe where a mapping is consumed, not a second representation vocabulary.
See [Spatial and Temporal Dimensions](./spatial-temporal-dimensions.md) for the tensor declarations and constraints.

### Declarative Mapping Context

Declarative mappings state order in terms of logical axes instead of raw strides or offsets.
The H×W examples above use `m![H, W]` for H-major order, `m![W, H]` for W-major order, and `m![H / 2, W / 2, H % 2, W % 2]` for 2×2 tiles.
The first two tile dimensions identify a tile and the last two identify its position within that tile.

Mapping expressions can be normalized to a standard representation, which makes equivalent constructor forms easier to inspect.
The tested normalization properties and their limits are documented with [Equivalent Mappings](./mapping-expressions.md#equivalent-mappings).
This chapter does not claim a universal proof or unique representation.
