//! Embedding lookup.
//!
//! Scales `input_ids` to byte offsets, then gathers rows from the embedding table.

use crate::transformer::axes::{
    H,             // = 896, hidden_size = 14 * 64
    S_decode as S, // = 128, sequence length
    W_vocab as W,  // = 151936, vocab_size
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

/// Embedding lookup: input_ids → DmTensor in SRAM.
///
/// Returns the gather result in SRAM. Caller is responsible for spilling
/// to DRAM for downstream normalization and output paths.
pub(super) fn embed(
    ctx: &mut Context,
    input_ids: &HbmTensor<i32, Chip, m![S]>,
    embedding_table: &HbmTensor<bf16, Chip, m![W, H]>,
) -> DmTensor<bf16, Chip, Cluster, m![H / 128 # 8, S / 4], m![S % 4, H % 128]> {
    // Load token ids to SRAM.
    let ids_dm = input_ids.to_dm::<Cluster, m![1 # 256], m![S]>(&mut ctx.tdma, 0x200);

    // Convert token ids into byte offsets for table gather.
    let ids_scaled: DmTensor<i32, Chip, Cluster, m![1 # 256], m![S]> = ctx
        .main
        .begin(ids_dm.view())
        .fetch::<i32, m![S / 8], m![S % 8]>()
        .collect::<m![S / 8], m![S % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::MulInt, 1792)
        .vector_final()
        .commit::<m![S]>(0x200);

    // Reshape indices for gather layout.
    let ids_reshaped: DmTensor<i32, Chip, Cluster, m![1 # 256], m![S % 4, S / 4]> = ctx
        .main
        .begin(ids_scaled.view())
        .fetch::<i32, m![S / 4], m![S % 4]>()
        .collect::<m![S / 4], m![S % 4 # 8]>()
        .commit::<m![S % 4, S / 4]>(0x0);

    // Move reshaped indices to HBM for gather input.
    let ids_hbm: HbmTensor<i32, Chip, m![S % 4, S / 4]> = ids_reshaped.to_hbm(&mut ctx.tdma, 0x10e36000);

    // Gather embedding rows for each token id.
    embedding_table.dma_gather(&ids_hbm, 0x0)
}
