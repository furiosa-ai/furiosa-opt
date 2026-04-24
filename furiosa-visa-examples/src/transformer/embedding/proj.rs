//! Q/K/V projection kernels used by transformer attention.
//!
//! This module implements the PyTorch linear projections:
//! `q = x @ Wq + bq`, `k = x @ Wk + bk`, and `v = x @ Wv + bv`.
//!
//! The kernels load projection weights from HBM, reshape them for DPE contraction,
//! run matmul with bias add, and write `[S, Q/K/V]` outputs back to HBM.
//! V and K use the same projection pattern, while Q uses a reversed placement
//! so the activation tile is loaded into TRF.

use crate::transformer::axes::{
    H,             // = 896, hidden_size = 14 * 64
    K,             // = 128, kv_proj_size = N * D
    Q,             // = 896, q_proj_size
    S_decode as S, // = 128, sequence length
    V,             // = 128, v_proj_size
    Y,             // = 4, broadcast
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

/// V projection: normalized input × V_weight → out_v.
pub(crate) fn v_proj(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32, H % 56]>,
    weight: &HbmTensor<bf16, Chip, m![V, H]>,
    out: &mut HbmTensor<bf16, Chip, m![S, V]>,
) {
    let weight_dm: DmTensor<bf16, Chip, Cluster, m![V / 8, H / 224, V / 2 % 4], m![V % 2, H % 224]> =
        weight.to_dm(&mut ctx.tdma, 0x400);

    // Reorder the weight tile so H blocks align with DPE contraction.
    let weight_it: DmTensor<bf16, Chip, Cluster, m![V / 8, H / 224, H / 56 % 4], m![V % 8, H % 56 # 64]> = ctx
        .main
        .begin(weight_dm.view())
        .fetch::<bf16, m![V % 2, H / 56 % 4], m![H % 56]>()
        .switch::<m![V / 8, H / 224, H / 56 % 4], m![V / 2 % 4, V % 2]>(SwitchConfig::InterTranspose {
            slice1: 4,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![V / 2 % 4, V % 2], m![H % 56 # 64]>()
        .commit(0x800);

    // Load reordered V weight into TRF for the DPE matmul.
    let weight_trf: TrfTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![V % 8], m![V / 8 % 4, H % 56 # 64]> = ctx
        .sub
        .begin(weight_it.view())
        .fetch::<bf16, m![V % 8], m![H % 56]>()
        .switch::<m![Y, S / 32, H / 56], m![V % 8]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
        .collect::<m![V % 8], m![V / 8 % 4, H % 56 # 64]>()
        .to_trf(TrfAddress::FirstHalf);

    // Load the pre-tiled V bias buffer and stage it in VRF for fused bias add.
    let v_bias_hbm: HbmTensor<f32, Chip, m![1 # 2, 1 # 2, S / 32, S / 4 % 8, S / 2 % 2, V % 32]> =
        unsafe { HbmTensor::from_addr(0xc000) };
    let v_bias_dm: DmTensor<f32, Chip, Cluster, m![1 # 2, 1 # 2, S / 32, S / 4 % 8, S / 2 % 2], m![V % 32]> =
        v_bias_hbm.to_dm(&mut ctx.tdma, 0x400);
    let v_bias_vrf: VrfTensor<f32, Chip, Cluster, m![1 # 2, 1 # 2, S / 32, S / 4 % 8, S / 2 % 2], m![V % 32]> = ctx
        .sub
        .begin(v_bias_dm.view())
        .fetch::<f32, m![1], m![V % 32]>()
        .collect::<m![1], m![V % 32]>()
        .to_vrf(0x0);

    // Compute V = input @ V_weight and apply bias in the VE path.
    let result: DmTensor<bf16, Chip, Cluster, m![1 # 2, 1 # 2, S / 32, S / 4 % 8, S / 2 % 2], m![S % 2, V % 32]> = ctx
        .main
        .begin(input.view())
        .fetch::<bf16, m![S / 4 % 8, S / 2 % 2, S % 2], m![H / 8 % 7, H % 8]>()
        .collect::<m![S / 4 % 8, S / 2 % 2, S % 2], m![H / 8 % 7, H % 8]>()
        .align::<m![S / 4 % 8, S / 2 % 2, S % 2], m![H / 8 % 7, H % 8], _, _>(&weight_trf)
        .contract::<m![1]>()
        .accumulate::<m![S / 4 % 8, V / 8 % 4, S % 4], m![V % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_clip(ClipBinaryOpF32::Add, &v_bias_vrf)
        .vector_inter_slice_reduce::<m![1 # 2, 1 # 2, S / 32, S / 4 % 8, S / 2 % 2], m![V / 8 % 4, S % 2]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .cast::<bf16, m![V % 8 # 16]>()
        .commit(0x600);

    let v_reassembled: DmTensor<bf16, Chip, Cluster, m![S / 2, Y], m![S % 2, V]> = ctx
        .main
        .begin(result.view())
        .fetch::<bf16, m![S % 2], m![V % 32]>()
        .switch::<m![S / 2, Y], m![S % 2]>(SwitchConfig::CustomBroadcast { ring_size: 256 })
        .collect::<m![S % 2], m![V % 32]>()
        .commit(0x800);

    v_reassembled.view().to_hbm_view(&mut ctx.tdma, out.view_mut());
}

/// Q projection: normalized input × Q_weight + bias.
///
/// The Q path stages activations in TRF and streams Q weights from SRAM.
pub(crate) fn q_proj(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32, H % 56]>,
    weight: &HbmTensor<bf16, Chip, m![Q, H]>,
    bias: &HbmTensor<bf16, Chip, m![Q]>,
) -> HbmTensor<bf16, Chip, m![S, Q]> {
    // Reshape Q bias from HBM layout into a VRF layout for fused add.

    let q_bias_dm: DmTensor<bf16, Chip, Cluster, m![Q / 32 % 2, Q / 4 % 8, 1 # 16], m![Q / 64, Q % 4]> =
        bias.to_dm(&mut ctx.tdma, 0x6600);

    let q_bias_te: DmTensor<bf16, Chip, Cluster, m![Q / 32 % 2, Q / 4 % 8, 1 # 16], m![Q % 4, Q / 64 # 16]> = ctx
        .main
        .begin(q_bias_dm.view())
        .fetch::<bf16, m![Q / 64], m![Q % 4]>()
        .collect::<m![Q / 64], m![Q % 4]>()
        .commit(0x500);

    let q_bias_it: DmTensor<bf16, Chip, Cluster, m![Q / 32 % 2, Q / 4 % 8, Q % 4, 1 # 4], m![1 # 4, Q / 64 # 16]> = ctx
        .main
        .begin(q_bias_te.view())
        .fetch::<bf16, m![Q % 4], m![Q / 64 # 16]>()
        .switch::<m![Q / 32 % 2, Q / 4 % 8, Q % 4, 1 # 4], m![1 # 4]>(SwitchConfig::InterTranspose {
            slice1: 4,
            slice0: 4,
            time0: 1,
        })
        .collect::<m![1 # 4], m![Q / 64 # 16]>()
        .commit(0x600);

    let q_bias_reassembled: DmTensor<bf16, Chip, Cluster, m![Q / 32 % 2, Q / 4 % 8, Q % 4, Y], m![Q / 64 # 16]> = ctx
        .main
        .begin(q_bias_it.view())
        .fetch::<bf16, m![1], m![Q / 64 # 16]>()
        .switch::<m![Q / 32 % 2, Q / 4 % 8, Q % 4, Y], m![1]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![1], m![Q / 64 # 16]>()
        .commit(0x500);

    // Reshape bias Slice to match the Q matmul pipeline's Slice.
    let q_bias_reassembled: DmTensor<bf16, Chip, Cluster, m![Q % 2, Q / 2 % 2, Y, H / 56], m![Q / 64 # 16]> =
        unsafe { q_bias_reassembled.reshape() };

    let q_bias_vrf: VrfTensor<f32, Chip, Cluster, m![Q % 2, Q / 2 % 2, Y, H / 56], m![Q / 64 # 16]> = ctx
        .sub
        .begin(q_bias_reassembled.view())
        .fetch::<f32, m![1], m![Q / 64 # 16]>()
        .collect::<m![1], m![Q / 64 # 16]>()
        .to_vrf(0x0);

    // Pad the input packet axis for TRF-friendly alignment.
    let input_padded: DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32, H % 56 # 64]> = ctx
        .main
        .begin(input.view())
        .fetch::<bf16, m![S % 32], m![H % 56]>()
        .collect::<m![S % 32], m![H % 56 # 64]>()
        .commit(0x2100);

    // Stage padded activations in TRF for Q projection.
    let input_trf: TrfTensor<
        bf16,
        Chip,
        Cluster,
        m![Q % 2, Q / 2 % 2, Y, H / 56],
        m![S % 8],
        m![H / 8 % 7, S / 8 % 4, S % 8 # 32],
    > = ctx
        .sub
        .begin(input_padded.view())
        .fetch::<bf16, m![S % 8], m![H % 56]>()
        .switch::<m![Q % 2, Q / 2 % 2, Y, H / 56], m![S % 8]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
        .collect::<m![S % 8], m![H / 8 % 7, S / 8 % 4, S % 8 # 32]>()
        .to_trf(TrfAddress::SecondHalf);

    let weight_dm: DmTensor<
        bf16,
        Chip,
        Cluster,
        m![Q / 4 % 16, H / 224, Q / 448, Q / 2 % 2],
        m![Q / 64, Q % 2, H % 224],
    > = weight.to_dm(&mut ctx.tdma, 0x3100);

    // Reorder Q weight blocks so H fragments align with contraction groups.
    let weight_it: DmTensor<
        bf16,
        Chip,
        Cluster,
        m![Q / 4 % 16, H / 224, H / 112 % 2, H / 56 % 2],
        m![Q / 64, Q % 2, Q / 448, Q / 2 % 2, H % 56 # 64],
    > = ctx
        .main
        .begin(weight_dm.view())
        .fetch::<bf16, m![Q % 2, Q / 448, Q / 2 % 2, H / 56 % 4], m![H % 56]>()
        .switch::<m![Q / 4 % 16, H / 224, H / 112 % 2, H / 56 % 2], m![Q / 64, Q % 2, Q / 448, Q / 2 % 2]>(
            SwitchConfig::InterTranspose {
                slice1: 4,
                slice0: 1,
                time0: 1,
            },
        )
        .collect::<m![Q / 64, Q % 2, Q / 448, Q / 2 % 2], m![H % 56 # 64]>()
        .commit(0x4a00);

    // Strip packet padding before DPE alignment.
    let weight_stripped: DmTensor<
        bf16,
        Chip,
        Cluster,
        m![Q / 4 % 16, H / 224, H / 112 % 2, H / 56 % 2],
        m![Q / 448, Q / 64, Q / 2 % 2, Q % 2, H % 56],
    > = ctx
        .main
        .begin(weight_it.view())
        .fetch::<bf16, m![Q / 448, Q / 64, Q / 2 % 2, Q % 2], m![H % 56]>()
        .collect::<m![Q / 448, Q / 64, Q / 2 % 2, Q % 2], m![H % 56]>()
        .commit(0x3100);

    // Reshape weight Slice to match TRF Slice for alignment.
    let weight_stripped: DmTensor<
        bf16,
        Chip,
        Cluster,
        m![Q % 2, Q / 2 % 2, Y, H / 56],
        m![Q / 448, Q / 64, Q / 2 % 2, Q % 2, H % 56],
    > = unsafe { weight_stripped.reshape() };

    // Compute Q = input @ Q_weight and apply Q bias in the VE path.
    let result: DmTensor<bf16, Chip, Cluster, m![Q % 2, Q / 2 % 32, S / 32], m![Q / 64, S % 32]> = ctx
        .main
        .begin(weight_stripped.view())
        .fetch::<bf16, m![Q / 64, Q / 448, Q / 2 % 2, Q % 2], m![H / 8 % 7, H % 8]>()
        .collect::<m![Q / 64, Q / 448, Q / 2 % 2, Q % 2], m![H / 8 % 7, H % 8]>()
        .align::<m![Q / 64, Q / 448, Q / 2 % 2, Q % 2], m![H / 8 % 7, H % 8], _, _>(&input_trf)
        .contract::<m![1]>()
        .accumulate::<m![Q / 64, S / 8 % 4, Q / 448], m![S % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_clip(ClipBinaryOpF32::Add, &q_bias_vrf)
        .vector_inter_slice_reduce::<m![Q % 2, Q / 2 % 32, S / 32], m![Q / 64, S % 32]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .cast::<bf16, m![S % 32]>()
        .commit(0x0A00);

    let result_d0b: DmTensor<bf16, Chip, Cluster, m![Q % 2, Q / 2 % 32, 1 # 4], m![Q / 64, S]> = ctx
        .main
        .begin(result.view())
        .fetch::<bf16, m![Q / 64], m![S % 32]>()
        .switch::<m![Q % 2, Q / 2 % 32, 1 # 4], m![Q / 64]>(SwitchConfig::Broadcast01 {
            slice1: 4,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![Q / 64], m![S % 32]>()
        .commit(0x2100);

    result_d0b.to_hbm(&mut ctx.tdma, 0x10e36000)
}

/// K projection: normalized input × K_weight → K in HBM.
pub(crate) fn k_proj(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32, H % 56]>,
    weight: &HbmTensor<bf16, Chip, m![K, H]>,
) -> HbmTensor<bf16, Chip, m![S, K]> {
    // Load the pre-tiled K bias buffer and stage it in VRF for fused bias add.
    let k_bias_hbm: HbmTensor<f32, Chip, m![1 # 2, 1 # 2, S / 32, S / 4 % 8, S / 2 % 2, K % 32]> =
        unsafe { HbmTensor::from_addr(0x4000) };
    let k_bias_dm: DmTensor<f32, Chip, Cluster, m![1 # 2, 1 # 2, S / 32, S / 4 % 8, S / 2 % 2], m![K % 32]> =
        k_bias_hbm.to_dm(&mut ctx.tdma, 0x400);
    let k_bias_vrf: VrfTensor<f32, Chip, Cluster, m![1 # 2, 1 # 2, S / 32, S / 4 % 8, S / 2 % 2], m![K % 32]> = ctx
        .sub
        .begin(k_bias_dm.view())
        .fetch::<f32, m![1], m![K % 32]>()
        .collect::<m![1], m![K % 32]>()
        .to_vrf(0x0);

    let weight_dm: DmTensor<bf16, Chip, Cluster, m![K / 8, H / 224, K / 2 % 4], m![K % 2, H % 224]> =
        weight.to_dm(&mut ctx.tdma, 0x0);

    // Reorder the weight tile so H blocks align with DPE contraction.
    let weight_it: DmTensor<bf16, Chip, Cluster, m![K / 8, H / 224, H / 56 % 4], m![K % 8, H % 56 # 64]> = ctx
        .main
        .begin(weight_dm.view())
        .fetch::<bf16, m![K % 2, H / 56 % 4], m![H % 56]>()
        .switch::<m![K / 8, H / 224, H / 56 % 4], m![K / 2 % 4, K % 2]>(SwitchConfig::InterTranspose {
            slice1: 4,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![K / 2 % 4, K % 2], m![H % 56 # 64]>()
        .commit(0x0E00);

    // Load reordered K weight into TRF for the DPE matmul.
    let weight_trf: TrfTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![K % 8], m![K / 8 % 4, H % 56 # 64]> = ctx
        .sub
        .begin(weight_it.view())
        .fetch::<bf16, m![K % 8], m![H % 56]>()
        .switch::<m![Y, S / 32, H / 56], m![K % 8]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 16 })
        .collect::<m![K % 8], m![K / 8 % 4, H % 56 # 64]>()
        .to_trf(TrfAddress::FirstHalf);

    // Compute K = input @ K_weight and apply bias in the VE path.
    let result: DmTensor<bf16, Chip, Cluster, m![1 # 2, 1 # 2, S / 32, S / 4 % 8, S / 2 % 2], m![S % 2, K % 32]> = ctx
        .main
        .begin(input.view())
        .fetch::<bf16, m![S / 4 % 8, S / 2 % 2, S % 2], m![H / 8 % 7, H % 8]>()
        .collect::<m![S / 4 % 8, S / 2 % 2, S % 2], m![H / 8 % 7, H % 8]>()
        .align::<m![S / 4 % 8, S / 2 % 2, S % 2], m![H / 8 % 7, H % 8], _, _>(&weight_trf)
        .contract::<m![1]>()
        .accumulate::<m![S / 4 % 8, K / 8 % 4, S % 4], m![K % 8]>(AccumulationKind::Interleaved)
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_clip(ClipBinaryOpF32::Add, &k_bias_vrf)
        .vector_inter_slice_reduce::<m![1 # 2, 1 # 2, S / 32, S / 4 % 8, S / 2 % 2], m![K / 8 % 4, S % 2]>(
            InterSliceReduceOpF32::Add,
        )
        .vector_final()
        .cast::<bf16, m![K % 8 # 16]>()
        .commit(0x0);

    // Reorder K output slices into the final `[S, K]` layout.
    let k_xpose: DmTensor<bf16, Chip, Cluster, m![S / 2, Y], m![S % 2, K % 32]> = ctx
        .main
        .begin(result.view())
        .fetch::<bf16, m![S % 2], m![K % 32]>()
        .switch::<m![S / 2, Y], m![S % 2]>(SwitchConfig::Transpose { slice1: 4, slice0: 64 })
        .collect::<m![S % 2], m![K % 32]>()
        .commit(0x100);

    // DMA K result → DRAM (for RoPE)
    k_xpose.to_hbm(&mut ctx.tdma, 0x8c800)
}
