//! Scatter/gather test kernels.
//!
//! `scatter_minimal` scatters `K = 512` keys into a `C = 612` cache (sparse: `C > K`);
//! the non-power-of-2, larger-than-`K` cache stresses non-aligned scatter coverage.
//!
//! `gather_minimal` gathers `G = 512` rows from the `K`-row table. Its gather count `G` is
//! separate from the scatter cache `C` so the values partition `G / 2 = 256` equals
//! `num_slices` (the sparse `C / 2 = 306` would round up past it and fail to lower).
//! `gather_unaligned` covers a non-power-of-2 count (`U = 768`, 3 rows/slice) that still meets
//! the 256-slice constraint.
//!
//! `gather_paged_kv` gathers over a 4-D DRAM pool whose gather-key label `NBlocks` is reused in
//! the gathered output. That reused key is what the visa->LIR indirect-key relabeling isolates by
//! subtracting the index's own labels.

use furiosa_opt_std::prelude::*;

axes![
    K = 512, // Scatter key
    D = 128, // Payload per key
    C = 612, // Cache length (non-power-of-2, > K: stresses unaligned coverage)
    G = 512, // Slice-aligned gather count (G / 2 = 256)
    U = 768, // Unaligned gather count (non-power-of-2, > K: 3 rows/slice, U / 3 = 256)
    CL = 2   // Real cluster partition (hardware has 2 clusters/chip): placed, not broadcast
];

// Multi-dim DRAM-pool gather axes. `NBlocks * PBlock = 256` fills the slices; the gather remaps
// the leading `NBlocks` rows to physical pool rows via a byte-offset index, carrying a
// `[KvHeads, HeadDim]` payload per gathered row.
axes![
    NBlocks = 16,  // gathered block count
    PBlock = 16,   // block size (lane-aligned); NBlocks * PBlock = 256
    KvHeads = 8,   // outer payload axis
    HeadDim = 128  // inner payload axis
];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Scatter values into cache at index positions.
#[device(chip = 1)]
pub fn scatter_minimal(
    ctx: &mut Context,
    data: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![K]>,
    output: &mut HbmTensor<bf16, Chip, m![C, D]>,
) {
    let data_dm: DmTensor<bf16, Chip, Cluster, m![K / 2], m![K % 2, D]> = data.to_dm(&mut ctx.tdma);

    data_dm.dma_scatter::<m![K], _, _>(index, output);
}

/// Gather `G` rows from `table` at positions given by `index` into a fresh HBM output.
///
/// Inverse of [`scatter_minimal`]. Produces a `DmTensor` via [`HbmTensor::dma_gather_scaled`], then
/// writes it back to HBM via [`DmTensor::to_hbm`], returned by value because `dma_gather_scaled` does
/// not write into an existing `&mut HbmTensor`. The `G / 2 = 256` values partition equals
/// `num_slices`, so it lowers end to end through VISA -> LIR -> EDF.
#[device(chip = 1)]
pub fn gather_minimal(
    ctx: &mut Context,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
) -> HbmTensor<bf16, Chip, m![G, D]> {
    let values_dm: DmTensor<bf16, Chip, Cluster, m![G / 2], m![G % 2, D]> = table.dma_gather_scaled(index);

    values_dm.to_hbm(&mut ctx.tdma)
}

/// Gather an unaligned count `U = 768` (`= 3 * num_slices`, non-power-of-2, `> K`) from the
/// `K`-row table. Unlike [`gather_minimal`]'s `2` rows/slice, the values partition here packs
/// `3` rows/slice (`U / 3 = 256 = num_slices`), so it still lowers to EDF while covering a
/// non-power-of-2 gather size that wraps the table.
#[device(chip = 1)]
pub fn gather_unaligned(
    ctx: &mut Context,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![U]>,
) -> HbmTensor<bf16, Chip, m![U, D]> {
    let values_dm: DmTensor<bf16, Chip, Cluster, m![U / 3], m![U % 3, D]> = table.dma_gather_scaled(index);

    values_dm.to_hbm(&mut ctx.tdma)
}

/// Gather over a multi-dimensional DRAM pool `[NBlocks, KvHeads, PBlock, HeadDim]`, remapping the
/// leading `NBlocks` rows to physical pool rows via a byte-offset index. The gathered output
/// regroups the two sub-axes `[NBlocks, PBlock]` onto the 256 slices with a `[KvHeads, HeadDim]`
/// payload per slice.
///
/// Unlike the 2-D `[K, D]` gathers above, the gather-key label `NBlocks` is reused in the gathered
/// output (the gathered rows stay ordered by the same `NBlocks` axis the index addresses). That
/// reuse hits the operator-verify wall (`index_axis.len() (0) != 1`) unless the visa->LIR
/// relabeling subtracts the index's own labels before deciding which table axis is genuinely kept,
/// see `LirBuilder::relabel_indirect_key_axis`.
#[device(chip = 1)]
pub fn gather_paged_kv(
    ctx: &mut Context,
    pool: &HbmTensor<bf16, Chip, m![NBlocks, KvHeads, PBlock, HeadDim]>,
    block_table: &HbmTensor<i32, Chip, m![NBlocks]>,
) -> HbmTensor<bf16, Chip, m![NBlocks, KvHeads, PBlock, HeadDim]> {
    // HBM write-back past the pool; only distinctness matters for a lowering probe.
    let gathered: DmTensor<bf16, Chip, Cluster, m![NBlocks, PBlock], m![KvHeads, HeadDim]> =
        pool.dma_gather_scaled(block_table);

    gathered.to_hbm(&mut ctx.tdma)
}

/// Unscaled twin of [`gather_minimal`]: the index holds RAW row positions instead of byte offsets.
/// The unscaled path reads the index straight off SPM, so the kernel first stages the DRAM index
/// on-chip with [`HbmTensor::to_dm`] and hands the resulting `DmTensor` to
/// [`HbmTensor::dma_gather_unscaled`]. This is the embed / paged-attention block-table idiom. Same
/// `G / 2 = 256` partitioning as [`gather_minimal`] so it lowers end to end through VISA -> LIR.
#[device(chip = 1)]
pub fn gather_aligned_unscaled(
    ctx: &mut Context,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
) -> HbmTensor<bf16, Chip, m![G, D]> {
    let index_dm: DmTensor<i32, Chip, Cluster, m![G / 2], m![G % 2]> = index.to_dm(&mut ctx.tdma);
    let values_dm: DmTensor<bf16, Chip, Cluster, m![G / 2], m![G % 2, D]> = table.dma_gather_unscaled(&index_dm);

    values_dm.to_hbm(&mut ctx.tdma)
}

/// Placed-cluster twin of [`gather_aligned_unscaled`]: the staged SPM index is distributed across
/// the two clusters as a REAL partition (`m![CL]`, a `LabelStride` cluster) instead of a broadcast
/// (`m![1 # 2]`). Each cluster gathers its own `G` rows through its own slice of the index, so the
/// index's cluster axis carries genuine per-cluster data rather than a replicated copy. Exercises
/// the unscaled gather lowering with a non-broadcast SPM index cluster.
#[device(chip = 1)]
pub fn gather_placed_unscaled(
    ctx: &mut Context,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![CL, G]>,
) -> HbmTensor<bf16, Chip, m![CL, G, D]> {
    let index_dm: DmTensor<i32, Chip, m![CL], m![G / 2], m![G % 2]> = index.to_dm(&mut ctx.tdma);
    let values_dm: DmTensor<bf16, Chip, m![CL], m![G / 2], m![G % 2, D]> = table.dma_gather_unscaled(&index_dm);

    values_dm.to_hbm(&mut ctx.tdma)
}
