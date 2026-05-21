# Mapping Tensors

Tensors have no intrinsic order of elements.
A mapping assigns each tensor index to a buffer position, defining that order.

This chapter shows how layout choices affect memory access and explains TCP's declarative approach to tensor layout with mapping expressions.

## Layout and Performance

The mapping determines access efficiency: hardware reads memory in contiguous blocks, so elements stored far apart require more transfers.
This choice affects programs too: programmers choose which axis is *major* (outermost, changes slowest) and which is *minor* (innermost, changes fastest, stored contiguously).
Layout cannot change after allocation without copying and transposing data, so the choice at allocation time constrains all subsequent operations.

Consider a tensor with axes H (height, 6 rows) and W (width, 8 columns).
The same tensor admits different mappings, each with different performance characteristics.

| H\W | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 0 | a | b | c | d | e | f | g | h |
| 1 | i | j | k | l | m | n | o | p |
| 2 | · | · | · | · | · | · | · | · |
| 3 | · | · | · | · | · | · | · | · |
| 4 | · | · | · | · | · | · | · | · |
| 5 | · | · | · | · | · | · | · | · |

- **H Major, W Minor**: A scan along W is contiguous; a scan along H accesses one element per cache line.

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

- **W Major, H Minor**: A scan along H is contiguous; a scan along W accesses one element per cache line.

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

- **2×2 Tiles**: Either choice sacrifices locality along one axis: H-major stores H-adjacent elements far apart, and W-major stores W-adjacent elements far apart.
  To avoid this trade-off, tiling groups nearby H and W indices into 2D tiles, achieving good locality along both axes, at the cost of a non-trivial address formula.

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

  Decompositions also determine hardware execution structure.
  The outer dimension can become a hardware time loop, and the inner dimension becomes a parallel lane.
  TCP names these hardware dimensions `Time` (the sequential loop counter) and `Packet` (the parallel data lane width), used throughout this book.
  [Spatial and Temporal Dimensions](./spatial-temporal-dimensions.md) explains how decompositions map to them.


## The Declarative Approach

Choosing optimal layout combinations manually is complex: the programmer must account for hardware geometry, alignment constraints, and execution patterns simultaneously.
The compiler derives physical placement, alignment, and hardware scheduling.

Virtual ISA lets the programmer and compiler declare a mapping in terms of logical axes.
For the H×W tensor from [Layout and Performance](#layout-and-performance), the leftmost axis is major and the rightmost is minor:
- **H-major**: `m![H, W]`
- **W-major**: `m![W, H]`
- **2×2 Tiles**: `m![H / 2, W / 2, H % 2, W % 2]`, where the first two dimensions are tile indices and the last two are positions within the tile.

Declarative mappings offer two benefits:
- **Expressiveness**: Layout is stated in terms of logical axes (`m![H, W]`), not raw strides or offsets.
- **Correctness**: Mapping expressions are normalized to canonical form and verified symbolically, turning layout properties into compile-time invariants.

Mapping expressions describe a tensor at every stage of its life: the same tensor can be stored in HBM, loaded into DM with a different layout, and streamed through the pipeline as packets.
Each stage holds the same mathematical values in a different physical representation.
[Tensor Semantics](./tensor-semantics.md) formalizes this perspective and shows how it makes data movement composable with computation in the same pipeline.
