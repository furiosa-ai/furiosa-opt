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
//!
//! [`negative::scatter_gather::invalid_gather_unscaled_index_over_spm`](crate::negative::scatter_gather::invalid_gather_unscaled_index_over_spm)
//! pins the per-PE SPM bound on an unscaled gather's index list, and the two kernels here are the
//! ways out. `gather_scaled_from_raw_index` takes a RAW row-position index and scales it on-device
//! into the byte offsets a scaled gather wants, the recommended shape for an index too large for
//! that list. `gather_unscaled_split_index` answers the same limit without leaving the unscaled
//! gather, by splitting the index into SPM-sized chunks and writing each chunk's gathered block
//! into its slot of one output.

use furiosa_opt_std::prelude::*;

axes![
    K = 512, // Scatter key
    D = 128, // Payload per key
    C = 612, // Cache length (non-power-of-2, > K: stresses unaligned coverage)
    G = 512, // Slice-aligned gather count (G / 2 = 256)
    U = 768, // Unaligned gather count (non-power-of-2, > K: 3 rows/slice, U / 3 = 256)
    CL = 2   // Real cluster partition (hardware has 2 clusters/chip): placed, not broadcast
];

// Raw-index scaled-gather axes, sized like an index the unscaled gather cannot take: `IdxRows *
// Indices = 2048` i32 entries is 8 KiB, twice the SPM one PE holds the unscaled index list in.
axes![
    Rows = 12,     // table rows the raw index selects (non-power-of-2, coprime to the index's 37)
    IdxRows = 16,  // index rows; `IdxRows * (Indices / 8) = 256` fills the hardware slices
    Indices = 128, // indices per index row
    Width = 8      // bf16 payload ELEMENTS per table row; the row's byte stride is `Width * 2 = 16`
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

/// Scaled gather fed by a RAW row-position index, scaled on-device: the DRAM index is staged to DM
/// (`to_dm`), multiplied by the table's row stride in bytes on the vector engine
/// (`FxpBinaryOp::MulInt`, an i32 multiply), written back to DRAM (`to_hbm`), and handed to
/// [`HbmTensor::dma_gather_scaled`].
///
/// The scale is a BYTE stride, not an element count: the `bf16` payload makes the two differ
/// (`ROW_BYTES = Width * 2 = 16`, against `Width = 8` elements), so scaling by the element count
/// would gather every other row's first half and the value oracle would diverge.
///
/// This is the recommended shape for a large index. The unscaled gather reads its index list off
/// ONE PE's SPM, so `IdxRows * Indices = 2048` i32 entries (8 KiB against a 4 KiB per-PE budget)
/// cannot be scheduled; the scaled gather reads its index from DRAM and has no such bound, and the
/// scaling the caller would otherwise do on the host is four vISA statements here.
#[device(chip = 1)]
pub fn gather_scaled_from_raw_index(
    ctx: &mut Context,
    table: &HbmTensor<bf16, Chip, m![Rows, Width]>,
    index: &HbmTensor<i32, Chip, m![IdxRows, Indices]>,
) -> HbmTensor<bf16, Chip, m![IdxRows, Indices, Width]> {
    type Slice = m![IdxRows, Indices / 8];
    type Packet = m![Indices % 8];
    const ROW_BYTES: i32 = (<m![Width]>::SIZE * size_of::<bf16>()) as i32;

    let raw: DmTensor<i32, Chip, Cluster, Slice, Packet> = index.to_dm(&mut ctx.tdma);
    let scaled: DmTensor<i32, Chip, Cluster, Slice, Packet> = ctx
        .main
        .begin(raw.view())
        .fetch::<m![1], Packet>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![Indices % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::MulInt, ROW_BYTES)
        .vector_final()
        .commit_trim::<Packet>()
        .commit();
    let offsets: HbmTensor<i32, Chip, m![IdxRows, Indices]> = scaled.to_hbm(&mut ctx.tdma);

    let values: DmTensor<bf16, Chip, Cluster, Slice, m![Indices % 8, Width]> = table.dma_gather_scaled(&offsets);

    values.to_hbm(&mut ctx.tdma)
}

/// The other way out of the SPM bound
/// [`crate::negative::scatter_gather::invalid_gather_unscaled_index_over_spm`] hits: keep the
/// index RAW and the gather UNSCALED, and split the INDEX instead. Each of the `Indices / 32 = 4`
/// chunks stages its own `IdxRows * (Indices % 32) = 512` i32 entries (2 KiB, inside the 4 KiB
/// per-PE budget); the four gathered blocks are reassembled by writing each into its
/// `view_mut().tile()` slot of one HBM output.
///
/// The chunk narrows each slice's PACKET (`Indices % 2`, 2 of the whole gather's 8) and keeps all
/// `IdxRows * (Indices % 32 / 2) = 256` slices, because a DM allocation must span the device's whole
/// slice extent -- chunking the index by rows (a 64-slice `IdxRows / 4` chunk) is rejected at MIR
/// ("slice extent 64 does not match the device config"). Trades
/// [`gather_scaled_from_raw_index`]'s vector-engine scaling pass plus its index round trip through
/// DRAM for four narrower staged DMAs, keeping the index on-chip.
///
/// The output is a kernel-local `HbmTensor::new` that is returned, the same idiom as
/// [`crate::tile::tile_chunked_output`]: a caller-allocated `&mut` output
/// parameter (the safe form [`scatter_minimal`] uses) does execute under VISA, but the per-chunk
/// tiled writeback then reaches LIR with NO registered output -- a tile write resolves against the
/// returned tensor after the loop closes, and a parameter has no such resolution.
#[device(chip = 1)]
pub fn gather_unscaled_split_index(
    ctx: &mut Context,
    table: &HbmTensor<bf16, Chip, m![Rows, Width]>,
    index: &HbmTensor<i32, Chip, m![IdxRows, Indices]>,
) -> HbmTensor<bf16, Chip, m![IdxRows, Indices, Width]> {
    type Slice = m![IdxRows, Indices % 32 / 2];
    type Packet = m![Indices % 2];

    let mut out = HbmTensor::<bf16, Chip, m![IdxRows, Indices, Width]>::new();
    for c in 0..<m![Indices / 32]>::SIZE {
        let chunk: DmTensor<i32, Chip, Cluster, Slice, Packet> = index
            .view()
            .tile::<m![Indices / 32], 1, m![IdxRows, 1 # 4, Indices % 32]>(c)
            .to_dm(&mut ctx.tdma);
        let values: DmTensor<bf16, Chip, Cluster, Slice, m![Indices % 2, Width]> = table.dma_gather_unscaled(&chunk);

        values.view().to_hbm_view(
            &mut ctx.tdma,
            out.view_mut()
                .tile::<m![Indices / 32], 1, m![IdxRows, 1 #{!} 4, Indices % 32, Width]>(c),
        );
    }

    out
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
