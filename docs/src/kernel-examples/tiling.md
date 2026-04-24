# Tiling

> [!Warning]
> This page is a work in progress. Content will be added in a future release.

Tiling breaks large tensors into smaller tiles that fit in on-chip memory.
When a tensor exceeds [VRF capacity](./hello-tcp.md) (8KB per slice) or [DM capacity](../moving-tensors/memory-performance.md), it must be processed in multiple iterations.

## When to Use Tiling

Tiling applies when:
- A tensor dimension exceeds what fits in a single hardware pass — compare the dimension size against the DM capacity table in [Memory Performance](../moving-tensors/memory-performance.md).
- Memory bandwidth needs to be optimized by reusing loaded data — check whether the same data is fetched more than once across operations.
- Computation needs to be distributed across time rather than space — use when the spatial dimensions are already fully distributed but a loop over tiles is needed.

## Basic Tiling Pattern

The basic pattern is: (1) choose a tile size that fits in VRF/DM, (2) loop over tiles in the outer dimensions, (3) fetch each tile from HBM to DM, (4) run the computation, and (5) accumulate partial results before writing back.
The tile size must satisfy alignment constraints (32-byte flits) and leave room for double-buffering if overlapping fetch with compute.

> [!Warning]
> Add a simple tiling example showing:
>- Original tensor shape exceeding VRF
>- Tile size calculation
>- Loop structure for processing tiles
>- Accumulation of partial results

```rust,ignore
// TODO: Example code
// axes![M = 8192, N = 8192, K = 2048];
//
// Tile sizes chosen to fit in VRF:
// type TileM = m![M / 32];  // 256 elements per tile
// type TileN = m![N / 32];  // 256 elements per tile
//
// Outer loop iterates over tiles
// Inner computation processes one tile
```

## Example: Tiled Matrix Multiplication

> [!Warning]
> Add complete GEMM example with tiling:
- Input matrices A[M, K] and B[K, N] where M, N, K exceed VRF capacity
- Tile along M and N dimensions
- Accumulate partial results across K tiles

### Memory Layout

> [!Warning]
> Describe how tiles are laid out in HBM and DM

### Tile Size Selection

> [!Warning]
> Explain constraints for choosing tile sizes:
> - VRF capacity (8KB per slice)
> - DM capacity
> - Alignment requirements (32-byte flits)
> - Trade-off between tile size and iteration count

### Accumulation Strategy

> [!Warning]
>  Explain how partial results are accumulated:
> - Accumulate in higher precision (f32) to avoid precision loss
> - Store intermediate results in DM or HBM depending on size
> - Final cast to output precision (bf16)

## Example: Tiled Attention

> [!Warning]
> Add attention example showing tiling for long sequences:
>- Query, Key, Value tensors with long sequence length
>- Tile along sequence dimension
>- FlashAttention-style tiling for memory efficiency

## Performance Considerations

> [!Warning]
> Add performance analysis:
> - Overhead of tile boundary handling
> - Memory bandwidth utilization
> - Optimal tile sizes for different tensor shapes
> - Interaction with hardware prefetching
