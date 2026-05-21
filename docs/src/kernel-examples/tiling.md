# Tiling

> [!WARNING]
> **WIP**: This chapter is currently in progress. Content will be added in a future release.

Tiling breaks large tensors into smaller tiles that fit in on-chip memory.
On-chip capacity is the governing constraint: when a tensor exceeds [VRF capacity](../quick-start.md#vector-register-file-vrf) (8KB per slice) or [DM capacity](../moving-tensors/memory-performance.md), it must be processed in multiple iterations.

## When to Use Tiling


Tiling applies when one or more of the following conditions hold (they often apply together):
- A tensor dimension exceeds what fits in a single hardware pass:
  compare the dimension size against the DM capacity table in [Memory Performance](../moving-tensors/memory-performance.md).
- Memory bandwidth needs to be optimized by reusing loaded data:
  check whether the same data is fetched more than once across operations.
- Computation needs to be distributed across time rather than space:
  use when the spatial dimensions are already fully distributed but a loop over tiles is needed.

## Basic Tiling Pattern

The basic tiling pattern covers a tensor that is too large for a single hardware pass but whose computation structure is otherwise straightforward.
Tile size drives the other decisions: it must fit in VRF/DM, satisfy alignment constraints (32-byte flits), and leave room for double-buffering if overlapping fetch with compute.
Given a valid tile size, the execution proceeds as: (1) loop over tiles in the outer dimensions, (2) fetch each tile from HBM to DM, (3) run the computation, and (4) accumulate partial results before writing back.


