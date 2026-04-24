# Mapping Tensors

This chapter explains what mappings are, how to declare them in TCP's Virtual ISA, and how to choose them for performance.

## Layout and Performance

Tensors have no intrinsic order of elements. A mapping is a function from tensor indices to buffer positions, which defines the order in which elements are stored.
When storing a tensor in hardware, you need to decide how they will be mapped into the flat buffer.

The choice of mapping matters because hardware reads memory in contiguous blocks: elements stored far apart require more memory transfers.
For example, one can choose which axis is *major* (outermost, changes slowest) and which is *minor* (innermost, changes fastest, stored contiguously).
Changing a layout after allocation requires copying and transposing data, so the mapping chosen at allocation time constrains all subsequent operations to match that layout.

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

### H Major, W Minor

A scan along W is contiguous; a scan along H accesses one element per cache line.

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

### W Major, H Minor

A scan along H is contiguous; a scan along W accesses one element per cache line.

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

Either choice sacrifices **spatial locality** (the property that nearby elements are stored at nearby addresses) in one direction.
Tiling achieves good locality along both axes by grouping nearby H and W indices into 2D tiles.

### 2×2 Tiles

All elements within a tile are contiguous.

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

W-minor layout is fast along W but slow along H; H-minor layout is the reverse; tiling gives balanced locality in both directions at the cost of more complex address calculation.
The outer dimension of a decomposition can become a hardware time loop; the inner dimension, a parallel lane.
Choosing a decomposition is how programmers control which dimensions execute sequentially and which execute in parallel.
TCP names these hardware dimensions `Time` (the sequential loop counter) and `Packet` (the parallel data lane width), used throughout this book; [Memory and Stream](./memory-stream.md) explains how decompositions map to them.


## The Declarative Approach

Virtual ISA lets the programmer declare a mapping in terms of logical axes; the compiler derives physical placement, alignment, and hardware scheduling.
In the [above example](#layout-and-performance), the simplest form is `m![H, W]` for H-major and `m![W, H]` for W-major, where the leftmost axis is major and the rightmost is minor.
Decomposing axes further with `/` and `%` enables tiling, expressed as `m![H / 2, W / 2, H % 2, W % 2]`: the first two dimensions are the tile indices and the last two are positions within the tile.

Declarative mappings offer three benefits:
- **Expressiveness**: Layout is stated in terms of logical axes (`m![H, W]`), not raw memory strides or offsets.
- **Correctness**: The compiler normalizes mapping expressions to canonical form and verifies them symbolically, turning layout properties into compile-time invariants.
- **Portability**: The same expression targets CPUs, GPUs, and TCPs without rewrites; the compiler derives hardware-specific placement from the axis description.

Mapping expressions describe a tensor at every stage of its life, not only when it is at rest in memory.
The same tensor can be stored in HBM, loaded into DM with a different layout, and streamed through the pipeline as packets; each stage holds the same mathematical values under a different mapping.

This unified view treats data movement as preserving the mathematical tensor: moving a tensor between stages changes only its physical representation, not its values.
The [Tensor Functions](./tensor-functions.md) page formalizes this perspective and shows how it makes data movement composable with computation in the same pipeline.
