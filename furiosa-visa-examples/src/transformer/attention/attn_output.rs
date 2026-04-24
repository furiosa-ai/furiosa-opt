//! Score × V matmul + output reshape.

use crate::transformer::axes::{
    D,              // = 64, head_dim
    G,              // = 7, gqa_ratio = 14 / 2
    H,              // = 896, hidden_size = 14 * 64
    I,              // = 2, interleave (ClipAdd dual-input)
    K,              // = 128, kv_proj_size = N * D
    N,              // = 2, num_kv_heads
    S_prefill as S, // = 1024, query sequence length (prefill)
    T,              // = 1024, key/value sequence length
    Z,              // = 2, broadcast
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

pub(super) fn attn_output(
    ctx: &mut Context,
    softmax_scores: &DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, T]>,
    attn_v: &HbmTensor<bf16, Chip, m![T, K]>,
    out_attn: &mut HbmTensor<bf16, Chip, m![S, H]>,
) {
    // Load value tensor from HBM to DM for the score x V matmul.
    let v_dm: DmTensor<bf16, Chip, Cluster, m![Z, T / 8 % 64, T / 512], m![T % 8, K]> =
        attn_v.to_dm(&mut ctx.tdma, 0x200);

    // Reorder value tiles to prepare V for TRF loading.
    let v_it: DmTensor<bf16, Chip, Cluster, m![Z, T / 8 % 64, K / 64], m![T / 512, T % 8, K % 64]> = ctx
        .main
        .begin(v_dm.view())
        .fetch::<bf16, m![T % 8, K / 64], m![K % 64]>()
        .switch::<m![Z, T / 8 % 64, K / 64], m![T % 8, T / 512]>(SwitchConfig::InterTranspose {
            slice1: 2,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![T / 512, T % 8], m![K % 64]>()
        .commit(0xa00);

    // Transpose V packet layout for TRF consumption.
    let v_transpose: DmTensor<bf16, Chip, Cluster, m![Z, T / 8 % 64, K / 64], m![T / 512, K % 64, T % 8]> = ctx
        .main
        .begin(v_it.view())
        .fetch::<bf16, m![T / 512, K / 64], m![T % 8, K % 64]>()
        .collect::<m![T / 512, K / 64], m![K % 64, T % 8]>()
        .commit(0x1200);

    // Copy the first T half of V.
    let mut v_half_0: DmTensor<bf16, Chip, Cluster, m![Z, T / 8 % 64, K / 64], m![K % 64, T % 8]> =
        unsafe { DmTensor::from_addr(0x1a00) };
    v_transpose
        .view()
        .tile::<m![T / 512], 2, m![K % 64, T % 8]>(0)
        .to_dm_view_pcopy(&mut ctx.sub, v_half_0.view_mut());

    // Copy the second T half of V.
    let mut v_half_1: DmTensor<bf16, Chip, Cluster, m![Z, T / 8 % 64, K / 64], m![K % 64, T % 8]> =
        unsafe { DmTensor::from_addr(0x1200) };
    v_transpose
        .view()
        .tile::<m![T / 512], 2, m![K % 64, T % 8]>(1)
        .to_dm_view_pcopy(&mut ctx.sub, v_half_1.view_mut());

    // Load first V half into TRF first half.
    let v_trf_first: TrfTensor<bf16, Chip, Cluster, m![S / 8, N], m![T % 8], m![K % 64]> = ctx
        .sub
        .begin(v_half_0.view())
        .fetch::<bf16, m![T % 8], m![K % 64]>()
        .switch::<m![S / 8, N], m![T % 8]>(SwitchConfig::Broadcast1 { slice1: 128, slice0: 2 })
        .collect::<m![T % 8], m![K % 64]>()
        .to_trf(TrfAddress::FirstHalf);

    // Load second V half into TRF second half.
    let v_trf_second: TrfTensor<bf16, Chip, Cluster, m![S / 8, N], m![T % 8], m![K % 64]> = ctx
        .sub
        .begin(v_half_1.view())
        .fetch::<bf16, m![T % 8], m![K % 64]>()
        .switch::<m![S / 8, N], m![T % 8]>(SwitchConfig::Broadcast1 { slice1: 128, slice0: 2 })
        .collect::<m![T % 8], m![K % 64]>()
        .to_trf(TrfAddress::SecondHalf);

    // Split the score T axis into two T halves for cascaded matmul.
    let scores_reshaped: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![T / 512, S % 8, G, T % 512]> = ctx
        .main
        .begin(softmax_scores.view())
        .fetch::<bf16, m![S % 8, G], m![T]>()
        .collect::<m![S % 8, G, T / 8], m![T % 8]>()
        .commit(0x28000);
    let mut scores_half_0: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, T % 512]> =
        unsafe { DmTensor::from_addr(0x28000) };
    scores_reshaped
        .view()
        .tile::<m![T / 512], 2, m![S % 8, G, T % 512]>(0)
        .to_dm_view_pcopy(&mut ctx.sub, scores_half_0.view_mut());
    let mut scores_half_1: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, T % 512]> =
        unsafe { DmTensor::from_addr(0x36000) };
    scores_reshaped
        .view()
        .tile::<m![T / 512], 2, m![S % 8, G, T % 512]>(1)
        .to_dm_view_pcopy(&mut ctx.sub, scores_half_1.view_mut());

    // Multiply first score half with first TRF-loaded V half.
    let _matmul_97: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, D]> = ctx
        .main
        .begin(scores_half_0.view())
        .fetch::<bf16, m![S % 8, G], m![T % 512]>()
        .collect::<m![S % 8, G, T / 8 % 64], m![T % 8]>()
        .align::<m![S % 8, G, T / 16 % 32], m![T % 16], _, _>(&v_trf_first)
        .contract::<m![1]>()
        .accumulate::<m![S % 8, G], m![T % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![S / 8, N], m![S % 8, G]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit(0x2200);

    // Accumulate first score half with second TRF-loaded V half.
    let matmul_98: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, D]> = ctx
        .main
        .begin(scores_half_0.view())
        .fetch::<bf16, m![S % 8, G], m![T % 512]>()
        .collect::<m![S % 8, G, T / 8 % 64], m![T % 8]>()
        .align::<m![S % 8, G, T / 16 % 32], m![T % 16], _, _>(&v_trf_second)
        .contract::<m![1]>()
        .accumulate::<m![S % 8, G], m![T % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![S / 8, N], m![S % 8, G]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit(0x2200);

    // Multiply second score half with first TRF-loaded V half.
    let _matmul_100: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, D]> = ctx
        .main
        .begin(scores_half_1.view())
        .fetch::<bf16, m![S % 8, G], m![T % 512]>()
        .collect::<m![S % 8, G, T / 8 % 64], m![T % 8]>()
        .align::<m![S % 8, G, T / 16 % 32], m![T % 16], _, _>(&v_trf_first)
        .contract::<m![1]>()
        .accumulate::<m![S % 8, G], m![T % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![S / 8, N], m![S % 8, G]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit(0x3e00);

    // Accumulate second score half with second TRF-loaded V half.
    let matmul_102: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, D]> = ctx
        .main
        .begin(scores_half_1.view())
        .fetch::<bf16, m![S % 8, G], m![T % 512]>()
        .collect::<m![S % 8, G, T / 8 % 64], m![T % 8]>()
        .align::<m![S % 8, G, T / 16 % 32], m![T % 16], _, _>(&v_trf_second)
        .contract::<m![1]>()
        .accumulate::<m![S % 8, G], m![T % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![S / 8, N], m![S % 8, G]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![D % 8 # 16]>()
        .commit(0x3e00);

    // Add both T-half matmul outputs to form the final attention output block.
    let matmul_out: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, D]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(matmul_98.view(), matmul_102.view())
        .fetch::<f32, m![I, S % 8, G], m![D]>()
        .collect::<m![I, S % 8, G], m![D]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![D]>()
        .commit(0x2200);

    // Reshape output from [N, G, D] back to hidden-size layout H.
    let output_reshaped: DmTensor<bf16, Chip, Cluster, m![S / 8, S / 4 % 2], m![S % 4, H]> = ctx
        .main
        .begin(matmul_out.view())
        .fetch::<bf16, m![S % 4, S / 4 % 2], m![H % 448]>()
        .switch::<m![S / 8, S / 4 % 2], m![S % 4, H / 448]>(SwitchConfig::InterTranspose {
            slice1: 2,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![S % 4, H / 448], m![H % 448]>()
        .commit(0x0);

    // Write reshaped attention output back to HBM.
    output_reshaped.view().to_hbm_view(&mut ctx.tdma, out_attn.view_mut());
}
