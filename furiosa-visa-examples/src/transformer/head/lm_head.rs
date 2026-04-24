//! LM Head: matmul input × embedding_table^T → logits [128, 151936].
//!
//! PyTorch equivalent: `logits = hidden_states @ embedding_table^T`.
//! The embedding table is processed in chunks, while hidden states are staged in TRF.
//! Each chunk computes partial logits for all tokens and writes into the output tile.

use crate::transformer::axes::{
    C_lmhead as C, // = 8192, lm_head weight chunk
    H,             // = 896, hidden_size = 14 * 64
    I,             // = 2, interleave (ClipAdd dual-input)
    S_decode as S, // = 128, sequence length
    W_vocab as W,  // = 151936, vocab_size
    Y,             // = 4, broadcast
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

pub(super) fn lm_head(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, m![Y, H / 14], m![H % 14, S]>,
    embedding_table: &HbmTensor<bf16, Chip, m![W, H]>,
    out_logits: &mut HbmTensor<bf16, Chip, m![S, W]>,
) {
    // Retile the normalized input into a layout suitable for DPE loading.
    let input_it: DmTensor<bf16, Chip, Cluster, m![Y, H / 56, S / 32], m![H % 14, H / 14 % 4, S % 32]> = ctx
        .main
        .begin(input.view())
        .fetch::<bf16, m![H % 14, H / 14 % 4], m![S % 32]>()
        .switch::<m![Y, H / 56, S / 32], m![H % 14, H / 14 % 4]>(SwitchConfig::Transpose { slice1: 4, slice0: 16 })
        .collect::<m![H % 14, H / 14 % 4], m![S % 32]>()
        .commit(0x1e000);

    // Reorder packet dimensions so sequence stays contiguous for TRF staging.
    let input_tiled: DmTensor<bf16, Chip, Cluster, m![Y, H / 56, S / 32], m![S % 32, H % 56 # 64]> = ctx
        .main
        .begin(input_it.view())
        .fetch::<bf16, m![H % 14, H / 14 % 4], m![S % 32]>()
        .collect::<m![H % 14, H / 14 % 4], m![S % 32]>()
        .commit(0x1ee00);

    // Split sequence into two halves and stage both halves in TRF.
    let first_half = input_tiled.view().tile::<m![S / 16], 2, m![S % 16, H % 56 # 64]>(0);
    let input_trf_first: TrfTensor<
        bf16,
        Chip,
        Cluster,
        m![Y, H / 56, S / 32],
        m![S % 8],
        m![H / 8 % 7, H / 56 % 4, S / 8 % 2, S % 8 # 16],
    > = ctx
        .sub
        .begin(first_half)
        .fetch::<bf16, m![S % 8], m![H / 8 % 7, H % 8]>()
        .collect::<m![S % 8], m![H / 8 % 7, H / 56 % 4, S / 8 % 2, S % 8 # 16]>()
        .to_trf(TrfAddress::FirstHalf);
    let second_half = input_tiled.view().tile::<m![S / 16], 2, m![S % 16, H % 56 # 64]>(1);
    let input_trf_second: TrfTensor<
        bf16,
        Chip,
        Cluster,
        m![Y, H / 56, S / 32],
        m![S % 8],
        m![H / 8 % 7, H / 56 % 4, S / 8 % 2, S % 8 # 16],
    > = ctx
        .sub
        .begin(second_half)
        .fetch::<bf16, m![S % 8], m![H / 8 % 7, H % 8]>()
        .collect::<m![S % 8], m![H / 8 % 7, H / 56 % 4, S / 8 % 2, S % 8 # 16]>()
        .to_trf(TrfAddress::SecondHalf);

    // Process vocabulary chunks with double-buffered SRAM addresses.
    for chunk_idx in 0..19usize {
        let weight_chunk = embedding_table.view().tile::<m![W / 8192], 0, m![C, H]>(chunk_idx);
        let sram_dma = if chunk_idx % 2 == 0 { 0x20000 } else { 0x10000 };
        let sram_it = if chunk_idx % 2 == 0 { 0x30000 } else { 0x00000 };
        let sram_out = if chunk_idx % 2 == 0 { 0x3e000 } else { 0x10000 };

        // Load one embedding-table chunk.
        let weight_dm = weight_chunk.to_dm::<Cluster, m![C / 128, C / 32 % 4], m![C % 32, H]>(&mut ctx.tdma, sram_dma);

        // Reorder chunked weights for contraction alignment.
        let weight_it: DmTensor<bf16, Chip, Cluster, m![C / 128, H / 224], m![C % 128, H % 224]> = ctx
            .main
            .begin(weight_dm.view())
            .fetch::<bf16, m![C % 32, H / 224], m![H % 224]>()
            .switch::<m![C / 128, H / 224], m![C / 32 % 4, C % 32]>(SwitchConfig::InterTranspose {
                slice1: 4,
                slice0: 1,
                time0: 1,
            })
            .collect::<m![C / 32 % 4, C % 32], m![H % 224]>()
            .commit(sram_it);

        // Reshape weight Slice to match TRF Slice for alignment.
        let weight_it: DmTensor<bf16, Chip, Cluster, m![Y, H / 56, S / 32], m![C % 128, H % 224]> =
            unsafe { weight_it.reshape() };

        // First-half contraction for this chunk.
        let first: DmTensor<bf16, Chip, Cluster, m![C / 128, S / 32], m![S % 32, C % 128]> = ctx
            .main
            .begin(weight_it.view())
            .fetch::<bf16, m![C / 4 % 32, H / 56 % 4, H / 8 % 7, C / 2 % 2], m![C % 2, H % 8]>()
            .collect::<m![C / 4 % 32, H / 56 % 4, H / 8 % 7, C / 2 % 2], m![C % 2, H % 8]>()
            .align::<m![C / 4 % 32, H / 56 % 4, H / 8 % 7, S / 8 % 2, S / 32 % 4], m![C % 4, H % 8], _, _>(
                &input_trf_first,
            )
            .contract::<m![1]>()
            .accumulate::<m![C % 32, C / 32 % 4, S / 8 % 8], m![S % 8]>(AccumulationKind::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<m![C / 128, S / 32], m![S % 32, C % 128]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![S % 32]>()
            .commit(sram_out);

        // Second-half contraction for this chunk.
        let second: DmTensor<bf16, Chip, Cluster, m![C / 128, S / 32], m![S % 32, C % 128]> = ctx
            .main
            .begin(weight_it.view())
            .fetch::<bf16, m![C / 4 % 32, H / 56 % 4, H / 8 % 7, C / 2 % 2], m![C % 2, H % 8]>()
            .collect::<m![C / 4 % 32, H / 56 % 4, H / 8 % 7, C / 2 % 2], m![C % 2, H % 8]>()
            .align::<m![C / 4 % 32, H / 56 % 4, H / 8 % 7, S / 8 % 2, S / 32 % 4], m![C % 4, H % 8], _, _>(
                &input_trf_second,
            )
            .contract::<m![1]>()
            .accumulate::<m![C % 32, C / 32 % 4, S / 8 % 8], m![S % 8]>(AccumulationKind::Interleaved)
            .vector_init()
            .vector_inter_slice_reduce::<m![C / 128, S / 32], m![S % 32, C % 128]>(InterSliceReduceOpF32::Add)
            .vector_final()
            .cast::<bf16, m![S % 32]>()
            .commit(0x08000);

        // Accumulate both halves to form final logits for the chunk.
        let accumulated: DmTensor<bf16, Chip, Cluster, m![C / 128, S / 32], m![S % 32, C % 128]> = ctx
            .main
            .begin_interleaved::<I, _, _, _, _, _>(first.view(), second.view())
            .fetch::<f32, m![I, S % 32], m![C % 128]>()
            .collect::<m![I, S % 32], m![C % 128]>()
            .vector_init()
            .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
            .vector_clip_zip(ClipBinaryOpF32::Add)
            .vector_final()
            .cast::<bf16, m![C % 128]>()
            .commit(sram_out);
        // Write the chunk logits into the output vocabulary slice.
        let out_tile = out_logits.view_mut().tile::<m![W / 8192], 1, m![S, C]>(chunk_idx);
        accumulated.view().to_hbm_view(&mut ctx.tdma, out_tile);
    }
}
