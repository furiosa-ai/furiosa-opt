//! Rejection fixture for the per-PE SPM budget an unscaled gather's index list must fit.

use furiosa_opt_std::prelude::*;

pub use crate::scatter_gather::{IdxRows, Indices, Rows, Width};

type Chip = m![1];
type Cluster = m![1 # 2];

/// The same gather as [`crate::scatter_gather::gather_scaled_from_raw_index`] written as an unscaled
/// gather, which cannot be compiled: the unscaled index list lives in ONE PE's SPM, and
/// `IdxRows * Indices = 2048` i32 entries is 8 KiB against a 4 KiB per-PE budget. Pinned as a
/// rejection so the diagnostic keeps naming the limit
/// (`OperatorVerifier::verify_indirect_tensor_dma`) instead of degenerating into a scheduler that
/// finds no SPM (`V1 failed for all operator schedule heuristics`).
///
/// The two ways out are [`crate::scatter_gather::gather_scaled_from_raw_index`], which scales the
/// raw index on-device and gathers from DRAM, and
/// [`crate::scatter_gather::gather_unscaled_split_index`], which stays unscaled and splits the index
/// into SPM-sized chunks.
#[device(chip = 1)]
pub fn invalid_gather_unscaled_index_over_spm(
    ctx: &mut Context,
    table: &HbmTensor<bf16, Chip, m![Rows, Width]>,
    index: &HbmTensor<i32, Chip, m![IdxRows, Indices]>,
) -> HbmTensor<bf16, Chip, m![IdxRows, Indices, Width]> {
    type Slice = m![IdxRows, Indices / 8];

    let index: DmTensor<i32, Chip, Cluster, Slice, m![Indices % 8]> = index.to_dm(&mut ctx.tdma);
    let values: DmTensor<bf16, Chip, Cluster, Slice, m![Indices % 8, Width]> = table.dma_gather_unscaled(&index);

    values.to_hbm(&mut ctx.tdma)
}
