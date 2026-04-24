//! Residual addition + RMS normalization.
//!
//! PyTorch: hidden = residual + x; hidden = RMSNorm(hidden)
//!   RMS Norm: y = x * (1 / sqrt(mean(x²) + eps)) * weight
//!
//! Three variants for different pipeline positions:
//! - `residual_norm`: post-attention, returns (normalized, residual_sum)
//! - `residual_norm_post`: post-MLP, writes residual sum to output
//! - `final_norm`: post-MLP last layer, no output write
//!
//! The residual sum is computed in VE, then RMS normalization runs as
//! variance -> reciprocal sqrt -> scale by per-channel weight.

use crate::transformer::axes::{
    H,             // = 896, hidden_size = 14 * 64
    I,             // = 2, interleave (ClipAdd dual-input)
    S_decode as S, // = 128, sequence length (tw128)
    W_norm as W,   // = 64, broadcast (decoder norm weight, S/2 replication)
    Y,             // = 4, broadcast (final_norm weight, 4PE replication)
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

/// Post-attention residual add + RMS norm. Returns (normalized, residual_sum).
pub(crate) fn residual_norm(
    ctx: &mut Context,
    hidden_states: &HbmTensor<bf16, Chip, m![S, H]>,
    o_proj_out: &HbmTensor<bf16, Chip, m![S, H]>,
    norm_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> (
    DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32, H % 56]>,
    HbmTensor<bf16, Chip, m![S, H]>,
) {
    let hidden_dm: DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> =
        hidden_states.to_dm(&mut ctx.tdma, 0x8000);
    let oproj_dm: DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> =
        o_proj_out.to_dm(&mut ctx.tdma, 0x18500);

    // Add residual and projection outputs elementwise.
    let residual_dm: DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(hidden_dm.view(), oproj_dm.view())
        .fetch::<f32, m![I, S % 2], m![H % 224]>()
        .collect::<m![I, S % 2, H / 8 % 28], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![H % 224]>()
        .commit(0x8000);

    // Keep the residual sum for the next residual connection.
    let residual_hbm = residual_dm.to_hbm(&mut ctx.tdma, 0x10e00000);

    let normalized = rms_norm_pipeline(ctx, &residual_dm, norm_weight);

    let norm_hbm: HbmTensor<bf16, Chip, m![S, H]> = normalized.to_hbm(&mut ctx.tdma, 0x10e10000);
    let retiled: DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32, H % 56]> =
        norm_hbm.to_dm(&mut ctx.tdma, 0x1300);
    (retiled, residual_hbm)
}

/// Post-MLP residual add + RMS norm. Writes residual sum to `out_hidden`.
pub(crate) fn residual_norm_post(
    ctx: &mut Context,
    residual: &HbmTensor<bf16, Chip, m![S, H]>,
    mlp_out: &HbmTensor<bf16, Chip, m![S, H]>,
    norm_weight: &HbmTensor<bf16, Chip, m![H]>,
    out_hidden: &mut HbmTensor<bf16, Chip, m![S, H]>,
) -> DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32, H % 56]> {
    // Load both inputs using the normalization-friendly tile shape.
    let residual_dm: DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> =
        residual.to_dm(&mut ctx.tdma, 0x8000);
    let mlp_dm: DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> =
        mlp_out.to_dm(&mut ctx.tdma, 0x18500);

    let residual_sum: DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(residual_dm.view(), mlp_dm.view())
        .fetch::<f32, m![I, S % 2], m![H % 224]>()
        .collect::<m![I, S % 2, H / 8 % 28], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![H % 224]>()
        .commit(0x8000);

    // Export residual sum for the next layer input.
    residual_sum.view().to_hbm_view(&mut ctx.tdma, out_hidden.view_mut());

    let rms_result = rms_norm_pipeline(ctx, &residual_sum, norm_weight);
    // Retile normalized output for the next block.
    let norm_hbm: HbmTensor<bf16, Chip, m![S, H]> = rms_result.to_hbm(&mut ctx.tdma, 0x10e10000);
    norm_hbm.to_dm(&mut ctx.tdma, 0x1300)
}

/// RMS norm VE pipeline shared by all variants.
fn rms_norm_pipeline(
    ctx: &mut Context,
    input: &DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]>,
    norm_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> {
    let weight_dm: DmTensor<bf16, Chip, Cluster, m![W, H / 224], m![H % 224]> =
        norm_weight.to_dm(&mut ctx.tdma, 0x9000);

    let weight_vrf: VrfTensor<f32, Chip, Cluster, m![S / 2, H / 224], m![H % 224]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<f32, m![1], m![H % 224]>()
        .switch::<m![S / 2, H / 224], m![1]>(SwitchConfig::Broadcast01 {
            slice1: 64,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![H / 8 % 28], m![H % 8]>()
        .to_vrf(0);

    // Copy input so variance and final scale can read independently.
    let mut input_copy: DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> =
        unsafe { DmTensor::from_addr(0x8400) };
    input.to_dm_pcopy(&mut ctx.sub, &mut input_copy);

    // Compute variance per token and add epsilon.
    let variance: DmTensor<f32, Chip, Cluster, m![S / 2, H / 224], m![S % 2]> = ctx
        .main
        .begin(input.view())
        .fetch::<f32, m![S % 2], m![H % 224]>()
        .collect::<m![S % 2, H / 8 % 28], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![S % 2, H / 8 % 28, H / 4 % 2], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![S % 2], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(
            FpDivBinaryOp::DivF.with_mode(BinaryArgMode::Mode01),
            ArrayVec::new([VeBranchOperand::always(VeRhs::constant(896.0f32))]),
        )
        .vector_pad_way8::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, 6.25e-8f32) // TODO: use filter to move Time -> Packet
        .vector_inter_slice_reduce::<m![S / 2, H / 224], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit(0x18500);

    // Compute reciprocal RMS and store in VRF.
    let inv_rms_vrf: VrfTensor<f32, Chip, Cluster, m![S / 2, H / 224], m![S % 2]> = ctx
        .main
        .begin(variance.view())
        .fetch::<f32, m![S % 2], m![1 # 8]>()
        .collect::<m![S % 2], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_trim_way4::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div(
            FpDivBinaryOp::DivF.with_mode(BinaryArgMode::Mode10),
            ArrayVec::new([VeBranchOperand::always(VeRhs::constant(1.0f32))]),
        )
        .vector_pad_way8::<m![1 # 8]>() // TODO: use filter to move Time -> Packet
        .vector_final()
        .to_vrf(0);

    // Apply RMS scaling and affine weight.
    let result: DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> = ctx
        .main
        .begin(input_copy.view())
        .fetch::<f32, m![S % 2, H / 8 % 28], m![H % 8]>()
        .collect::<m![S % 2, H / 8 % 28], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![S % 2, H / 8 % 28, H / 4 % 2], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_concat::<m![S % 2, H / 8 % 28], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit(0x8400);

    result
}
/// Final residual add + RMS norm (no out_hidden write — last layer before lm_head).
///
/// Unlike `residual_norm`/`residual_norm_post` which output H-contiguous tiling
/// (m![S%2, H%224]) for the attention/MLP consumers, `final_norm` outputs
/// S-contiguous tiling (m![H%14, S]) for the lm_head consumer.
pub(crate) fn final_norm(
    ctx: &mut Context,
    residual: &HbmTensor<bf16, Chip, m![S, H]>,
    mlp_out: &HbmTensor<bf16, Chip, m![S, H]>,
    norm_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![Y, H / 14], m![H % 14, S]> {
    let residual_dm: DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> =
        residual.to_dm(&mut ctx.tdma, 0x8000);
    let mlp_dm: DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> =
        mlp_out.to_dm(&mut ctx.tdma, 0x18500);
    let residual_sum: DmTensor<bf16, Chip, Cluster, m![S / 2, H / 224], m![S % 2, H % 224]> = ctx
        .main
        .begin_interleaved::<I, _, _, _, _, _>(residual_dm.view(), mlp_dm.view())
        .fetch::<f32, m![I, S % 2], m![H % 224]>()
        .collect::<m![I, S % 2], m![H % 224]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        .vector_clip_zip(ClipBinaryOpF32::Add)
        .vector_final()
        .cast::<bf16, m![H % 224]>()
        .commit(0x8000);
    // Use an HBM round-trip to retile from H-contiguous to S-contiguous layout.
    let residual_hbm: HbmTensor<bf16, Chip, m![S, H]> = residual_sum.to_hbm(&mut ctx.tdma, 0x10e10000);
    let retiled: DmTensor<bf16, Chip, Cluster, m![Y, H / 14], m![H % 14, S]> =
        residual_hbm.to_dm(&mut ctx.tdma, 0x1e000);

    // Compute variance per token and add epsilon.
    let variance: DmTensor<f32, Chip, Cluster, m![Y, H / 14], m![1]> = ctx
        .main
        .begin(retiled.view())
        .fetch::<f32, m![H % 14, S / 8 % 16], m![S % 8]>()
        .collect::<m![H % 14, S / 8 % 16], m![S % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![H % 14, S / 8 % 16, S / 4 % 2], m![S % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<S, m![H % 14], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(
            FpDivBinaryOp::DivF.with_mode(BinaryArgMode::Mode01),
            ArrayVec::new([VeBranchOperand::always(VeRhs::constant(896.0f32))]),
        )
        .vector_concat::<m![H % 14], m![1 # 8]>() // TODO: use filter to move Time -> Packet
        .vector_clip(ClipBinaryOpF32::Add, 1.5625e-8f32)
        .vector_inter_slice_reduce::<m![Y, H / 14], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit(0x1ee00);

    // Compute reciprocal RMS and store in VRF.
    let inv_rms_vrf: VrfTensor<f32, Chip, Cluster, m![Y, H / 14], m![1 # 8]> = ctx
        .main
        .begin(variance.view())
        .fetch::<f32, m![1], m![1 # 8]>()
        .collect::<m![1], m![1 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_trim_way4::<m![1 # 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div(
            FpDivBinaryOp::DivF.with_mode(BinaryArgMode::Mode10),
            ArrayVec::new([VeBranchOperand::always(VeRhs::constant(1.0f32))]),
        )
        .vector_pad_way8::<m![1 # 8]>()
        .vector_final()
        .to_vrf(0);

    let weight_dm: DmTensor<bf16, Chip, Cluster, m![Y, H / 14], m![H % 14]> = norm_weight.to_dm(&mut ctx.tdma, 0xc200);

    let weight_vrf: VrfTensor<f32, Chip, Cluster, m![Y, H / 14], m![H % 14]> = ctx
        .sub
        .begin(weight_dm.view())
        .fetch::<f32, m![1], m![H % 14]>()
        .switch::<m![Y, H / 14], m![1]>(SwitchConfig::Broadcast01 {
            slice1: 4,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![1], m![H % 14]>()
        .to_vrf(0);

    // Apply RMS scaling and affine weight.
    let result: DmTensor<bf16, Chip, Cluster, m![Y, H / 14], m![H % 14, S]> = ctx
        .main
        .begin(retiled.view())
        .fetch::<f32, m![H % 14, S / 8 % 16], m![S % 8]>()
        .collect::<m![H % 14, S / 8 % 16], m![S % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![H % 14, S / 8 % 16, S / 4 % 2], m![S % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_concat::<m![H % 14, S / 8 % 16], m![S % 8]>()
        .vector_final()
        .cast::<bf16, m![S % 8 # 16]>()
        .commit(0x1ee00);

    result
}
