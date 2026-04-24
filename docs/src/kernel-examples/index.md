# Kernel Examples

The [introductory tutorial](./hello-tcp.md) briefly introduced temporal and spatial partitioning for large tensors in its Further Reading section.
The preceding chapters explained how mapping expressions distribute work across TCP's hardware hierarchy and how each component reduces partial results.
This chapter shows how to combine mapping, movement, computation, and scheduling into complete, working kernels.
The table below summarizes the available parallelism and reduction at each level:

| Dimension | Type | Defined in | Reduced in |
|-----------|------|------------|------------|
| `Chip` | Spatial | [HBM, SRAM](../mapping-tensors/memory-stream.md#hbm-and-sram), [Stream](../mapping-tensors/memory-stream.md#tensor-unit-stream) | [DMA](../moving-tensors/dma-engine.md) + [Vector](../computing-tensors/vector-engine/index.md) |
| `Cluster` | Spatial | [SRAM](../mapping-tensors/memory-stream.md#hbm-and-sram), [Stream](../mapping-tensors/memory-stream.md#tensor-unit-stream) | [DMA](../moving-tensors/dma-engine.md) + [Vector](../computing-tensors/vector-engine/index.md) |
| `Slice` | Spatial | [SRAM](../mapping-tensors/memory-stream.md#hbm-and-sram), [Stream](../mapping-tensors/memory-stream.md#tensor-unit-stream) | [Vector](../computing-tensors/vector-engine/index.md) |
| `Row` | Spatial | [TRF](../mapping-tensors/memory-stream.md#hbm-and-sram) | [Contraction](../computing-tensors/contraction-engine/index.md) |
| `Time` | Temporal | [Stream](../mapping-tensors/memory-stream.md#tensor-unit-stream) | [Contraction](../computing-tensors/contraction-engine/index.md) |
| `Packet` | Spatial | [Stream](../mapping-tensors/memory-stream.md#tensor-unit-stream) | [Contraction](../computing-tensors/contraction-engine/index.md) |

For cross-chip and cross-cluster reduction patterns (the `Chip` and `Cluster` rows above), see [Chip/Cluster Reduce](./chip-cluster-reduce.md), which demonstrates DMA broadcast followed by Vector Engine binary add.

<!-- > **TODO** (youseok.yang): -->
<!-- > - Add decision flowchart: "I need to do X" → "Use this example" -->
<!-- > - Add common anti-patterns section: what NOT to do and why -->

The examples progress from single-engine patterns to composed multi-engine patterns to full model implementations:

- [Tiling](./tiling.md) *(coming soon)*: Tile size selection, memory layout, and accumulation strategies.
- [Split Reduce](./split-reduce.md): Interleaved fetch for reducing across multiple tensor instances.
  Use when a reduction dimension exceeds what a single tile can accumulate.
- [Chip/Cluster Reduce](./chip-cluster-reduce.md): ReduceScatter and AllReduce across chips.
  Use when computation must be distributed across multiple chips or clusters.
- [Fetch and Commit Engine](./fetch-commit-engine.md): Axis permutation, full-flit commit, tail padding, and tensor segmentation.
  Use when data layout transformations are needed between memory and compute.
- GEMM with Double-Buffering *(coming soon)*: DMA load from HBM, sub-context TRF prefetch, main-context tiled contraction, cast, and commit.
  A short end-to-end example bridging single-engine patterns and full model implementations.
- [Transformer](./transformer.md): Llama 3 70B implementation with prefill and decode phases.
  A full model combining tiling, multi-chip reduce, and memory management.
- [Mixture of Experts](./mixture-of-experts.md): Branchless TopK routing and blockwise sparse computation.
  A full model demonstrating dynamic routing with sparse computation patterns.
