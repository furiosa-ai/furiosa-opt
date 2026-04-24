use crate::transformer::axes::{
    D,              // = 64, head_dim
    G,              // = 7, gqa_ratio = 14 / 2
    H,              // = 896, hidden_size = 14 * 64
    K,              // = 128, kv_proj_size = N * D
    N,              // = 2, num_kv_heads
    S_prefill as S, // = 1024, query sequence length (prefill)
    T,              // = 1024, key/value sequence length
    W_kbcast as W,  // = 32, broadcast (K replication across S positions)
    Y,              // = 4, broadcast
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

pub(super) fn attn_weight(
    ctx: &mut Context,
    attn_q: &HbmTensor<bf16, Chip, m![S, H]>,
    attn_k: &HbmTensor<bf16, Chip, m![T, K]>,
) -> DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, T]> {
    // Load key tensor from HBM to DM.
    let k_dm: DmTensor<bf16, Chip, Cluster, m![Y, T / 16 % 32, T / 512], m![T % 16, K]> =
        attn_k.to_dm(&mut ctx.tdma, 0x200);

    // Reorder key tiles to split K into head chunks for TRF loading.
    let k_it: DmTensor<bf16, Chip, Cluster, m![Y, T / 16 % 32, K / 64], m![T / 512, T % 16, K % 64]> = ctx
        .main
        .begin(k_dm.view())
        .fetch::<bf16, m![T % 16, K / 64], m![K % 64]>()
        .switch::<m![Y, T / 16 % 32, K / 64], m![T % 16, T / 512]>(SwitchConfig::InterTranspose {
            slice1: 2,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![T / 512, T % 16], m![K % 64]>()
        .commit(0x1200);

    // Copy the first T half for matmul.
    let mut k_half_0: DmTensor<bf16, Chip, Cluster, m![Y, T / 16 % 32, K / 64], m![T % 16, K % 64]> =
        unsafe { DmTensor::from_addr(0x1200) };
    k_it.view()
        .tile::<m![T / 512], 2, m![T % 16, K % 64]>(0)
        .to_dm_view_pcopy(&mut ctx.sub, k_half_0.view_mut());

    // Copy the second T half for matmul.
    let mut k_half_1: DmTensor<bf16, Chip, Cluster, m![Y, T / 16 % 32, K / 64], m![T % 16, K % 64]> =
        unsafe { DmTensor::from_addr(0x1a00) };
    k_it.view()
        .tile::<m![T / 512], 2, m![T % 16, K % 64]>(1)
        .to_dm_view_pcopy(&mut ctx.sub, k_half_1.view_mut());

    // Load first key half into TRF first half.
    let k_trf_0: TrfTensor<bf16, Chip, Cluster, m![S / 256, W, N], m![T % 8], m![K % 64]> = ctx
        .sub
        .begin(k_half_0.view())
        .fetch::<bf16, m![T % 8], m![T / 8 % 2, K % 64]>()
        .switch::<m![S / 256, W, N], m![T % 8]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 2 })
        .collect::<m![T % 8], m![K % 64]>()
        .to_trf(TrfAddress::FirstHalf);

    // Load first key half into TRF second half for cascade accumulation.
    let k_trf_0s: TrfTensor<bf16, Chip, Cluster, m![S / 256, W, N], m![T % 8], m![K % 64]> = ctx
        .sub
        .begin(k_half_0.view())
        .fetch::<bf16, m![T % 8], m![T / 8 % 2, K % 64]>()
        .switch::<m![S / 256, W, N], m![T % 8]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 2 })
        .collect::<m![T % 8], m![K % 64]>()
        .to_trf(TrfAddress::SecondHalf);

    // Load query tensor from HBM to DM.
    let q_dm: DmTensor<bf16, Chip, Cluster, m![S / 256, W, N], m![S % 8, G, D]> = attn_q.to_dm(&mut ctx.tdma, 0x2200);

    // Compute Q x K^T using TRF first half and write partial scores.
    ctx.main
        .begin(q_dm.view())
        .fetch::<bf16, m![S % 8, G], m![D]>()
        .collect::<m![S % 8, G], m![D]>()
        .align::<m![S % 8, G], m![D], _, _>(&k_trf_0)
        .contract::<m![1]>()
        .accumulate::<m![S % 8, G], m![T % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![S / 8, N], m![S % 8, G]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![T % 512]>()
        .commit::<m![S % 8, G, T % 512]>(0x8000);

    // Accumulate Q x K^T with TRF second half into the same score buffer.
    ctx.main
        .begin(q_dm.view())
        .fetch::<bf16, m![S % 8, G], m![D]>()
        .collect::<m![S % 8, G], m![D]>()
        .align::<m![S % 8, G], m![D], _, _>(&k_trf_0s)
        .contract::<m![1]>()
        .accumulate::<m![S % 8, G], m![T % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![S / 8, N], m![S % 8, G]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![T % 512]>()
        .commit::<m![S % 8, G, T % 512]>(0x8000);

    // Load second key half into TRF first half.
    let k_trf_1: TrfTensor<bf16, Chip, Cluster, m![S / 256, W, N], m![T % 8], m![K % 64]> = ctx
        .sub
        .begin(k_half_1.view())
        .fetch::<bf16, m![T % 8], m![T / 8 % 2, K % 64]>()
        .switch::<m![S / 256, W, N], m![T % 8]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 2 })
        .collect::<m![T % 8], m![K % 64]>()
        .to_trf(TrfAddress::FirstHalf);

    // Load second key half into TRF second half for cascade accumulation.
    let k_trf_1s: TrfTensor<bf16, Chip, Cluster, m![S / 256, W, N], m![T % 8], m![K % 64]> = ctx
        .sub
        .begin(k_half_1.view())
        .fetch::<bf16, m![T % 8], m![T / 8 % 2, K % 64]>()
        .switch::<m![S / 256, W, N], m![T % 8]>(SwitchConfig::Broadcast1 { slice1: 32, slice0: 2 })
        .collect::<m![T % 8], m![K % 64]>()
        .to_trf(TrfAddress::SecondHalf);

    // Compute Q x K^T for the second T half using TRF first half.
    ctx.main
        .begin(q_dm.view())
        .fetch::<bf16, m![S % 8, G], m![D]>()
        .collect::<m![S % 8, G], m![D]>()
        .align::<m![S % 8, G], m![D], _, _>(&k_trf_1)
        .contract::<m![1]>()
        .accumulate::<m![S % 8, G], m![T % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![S / 8, N], m![S % 8, G]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![T % 512]>()
        .commit::<m![S % 8, G, T % 512]>(0x16000);

    // Accumulate the second T-half scores with TRF second half.
    ctx.main
        .begin(q_dm.view())
        .fetch::<bf16, m![S % 8, G], m![D]>()
        .collect::<m![S % 8, G], m![D]>()
        .align::<m![S % 8, G], m![D], _, _>(&k_trf_1s)
        .contract::<m![1]>()
        .accumulate::<m![S % 8, G], m![T % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_inter_slice_reduce::<m![S / 8, N], m![S % 8, G]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![T % 512]>()
        .commit::<m![S % 8, G, T % 512]>(0x16000);

    // View both T-half score blocks as one concatenated tensor.
    let qk_concat: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![T / 512, S % 8, G, T % 512]> =
        unsafe { DmTensor::from_addr(0x8000) };

    // Merge the two T chunks into a contiguous T axis.
    let qk_scores: DmTensor<bf16, Chip, Cluster, m![S / 8, N], m![S % 8, G, T]> = ctx
        .main
        .begin(qk_concat.view())
        .fetch::<bf16, m![T / 512, S % 8, G], m![T % 512]>()
        .collect::<m![T / 512, S % 8, G], m![T % 512]>()
        .commit(0x28000);

    qk_scores
}
