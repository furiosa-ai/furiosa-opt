//! RMS normalization for embedding output.
//!
//! `y = x * (1 / sqrt(sum(x^2) / H + eps)) * weight`

use crate::transformer::axes::{
    H,             // = 896, hidden_size = 14 * 64
    S_decode as S, // = 128, sequence length
    X,             // = 16, broadcast
    Y,             // = 4, broadcast
    Z,             // = 2, broadcast
};
use furiosa_visa_std::prelude::*;

type Chip = m![1];
type Cluster = m![1 # 2]; // 1 logical cluster, padded to 2 (hardware has 2 clusters/chip)

/// RMS Norm: hidden_states [S, H] → normalized [S, H]
///
/// Returns a DmTensor in SRAM for downstream QKV projections.
pub(super) fn rms_norm(
    ctx: &mut Context,
    hidden_hbm: &HbmTensor<bf16, Chip, m![S, H]>,
    norm_weight: &HbmTensor<bf16, Chip, m![H]>,
) -> DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32, H % 56]> {
    // Reload hidden states into a tiled SRAM layout for RMSNorm.
    let hidden_dm_0: DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 224, S / 8 % 4], m![S % 8, H % 224]> =
        hidden_hbm.to_dm(&mut ctx.tdma, 0x500);

    // Reorder tiles so hidden channels map to 56-wide vector lanes.
    let hidden_dm_1: DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 224, S / 8 % 4], m![S % 32, H % 56 # 64]> = ctx
        .main
        .begin(hidden_dm_0.view())
        .fetch::<bf16, m![S % 8, H / 56 % 4], m![H % 56]>()
        .collect::<m![S % 32], m![H % 56 # 64]>()
        .commit(0x1300);

    // Strip padded lanes and keep contiguous 56-wide channel blocks.
    let hidden_dm: DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 224, S / 8 % 4], m![S % 32, H % 56]> = ctx
        .main
        .begin(hidden_dm_1.view())
        .fetch::<bf16, m![S % 32], m![H % 56]>()
        .collect::<m![S % 32], m![H % 56]>()
        .commit(0x500);

    // Load norm weights.
    let weight_dm_0: DmTensor<bf16, Chip, Cluster, m![X, H / 112, 1 # 2], m![H % 112]> =
        norm_weight.to_dm(&mut ctx.tdma, 0x0);

    // Reorder weight tiles into 56-wide channel groups.
    let weight_dm_1: DmTensor<bf16, Chip, Cluster, m![X, H / 112, H / 56 % 2], m![1 # 2, H % 56 # 64]> = ctx
        .main
        .begin(weight_dm_0.view())
        .fetch::<bf16, m![H / 56 % 2], m![H % 56]>()
        .switch::<m![X, H / 112, H / 56 % 2], m![Z]>(SwitchConfig::InterTranspose {
            slice1: 2,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![Z], m![H % 56 # 64]>()
        .commit(0x100);

    // Channel padding strip pass.
    let weight_dm_2: DmTensor<bf16, Chip, Cluster, m![X, H / 112, H / 56 % 2], m![1 # 2, H % 56]> = ctx
        .main
        .begin(weight_dm_1.view())
        .fetch::<bf16, m![Z], m![H % 56]>()
        .collect::<m![Z], m![H % 56]>()
        .commit(0x0);

    // Dummy-lane strip pass for compact per-channel weights.
    let weight_dm_3: DmTensor<bf16, Chip, Cluster, m![X, H / 112, H / 56 % 2], m![H % 56]> = ctx
        .main
        .begin(weight_dm_2.view())
        .fetch::<bf16, m![1], m![H % 56]>()
        .collect::<m![1], m![H % 56]>()
        .commit(0x100);

    // Reshape weight tiles to match the normalization pipeline's Slice decomposition.
    let weight_dm_3: DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![H % 56]> =
        unsafe { weight_dm_3.reshape() };

    // Load weights into VRF as f32 for vector multiply.
    let weight_vrf: VrfTensor<f32, Chip, Cluster, m![Y, S / 32, H / 56], m![H % 56 # 64]> = ctx
        .sub
        .begin(weight_dm_3.view())
        .fetch::<f32, m![H / 8 % 7 # 8], m![H % 8]>()
        .collect::<m![H / 8 % 7 # 8], m![H % 8]>()
        .to_vrf(0);

    // Duplicate hidden states so variance and final scaling can run independently.
    let mut hidden_copy: DmTensor<
        bf16,
        Chip,
        Cluster,
        m![Y, S / 32, H / 224, S / 8 % 4],
        m![S / 8 % 4, S % 8, H % 56],
    > = unsafe { DmTensor::from_addr(0x1300) };

    hidden_dm.to_dm_pcopy(&mut ctx.sub, &mut hidden_copy);

    // Compute per-token variance term in f32.
    let variance: DmTensor<f32, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32]> = ctx
        .main
        .begin(hidden_dm.view())
        .fetch::<f32, m![S % 32, H / 8 % 7], m![H % 8]>()
        .collect::<m![S % 32, H / 8 % 7], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![S % 32, H / 8 % 7, H / 4 % 2], m![H % 4]>()
        .vector_stash()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_intra_slice_reduce::<H, m![S % 32], m![1 # 4]>(IntraSliceReduceOpF32::Add)
        .vector_fp_div(
            FpDivBinaryOp::DivF.with_mode(BinaryArgMode::Mode01),
            ArrayVec::new([VeBranchOperand::always(VeRhs::constant(896.0f32))]),
        )
        .vector_pad_way8::<m![1 # 8]>()
        .vector_clip(ClipBinaryOpF32::Add, 6.25e-8f32) // TODO: use filter to move Time -> Packet
        .vector_inter_slice_reduce::<m![Y, S / 32, H / 56], m![1]>(InterSliceReduceOpF32::Add)
        .vector_final()
        .commit(0x2400);

    // Compute reciprocal RMS scale and keep it in VRF.
    let inv_rms_vrf: VrfTensor<f32, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32]> = ctx
        .main
        .begin(variance.view())
        .fetch::<f32, m![S / 8 % 4], m![S % 8]>()
        .collect::<m![S / 8 % 4], m![S % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![S / 8 % 4, S / 4 % 2], m![S % 4]>()
        .vector_fp_unary(FpUnaryOp::Sqrt)
        .vector_fp_div(
            FpDivBinaryOp::DivF.with_mode(BinaryArgMode::Mode10),
            ArrayVec::new([VeBranchOperand::always(VeRhs::constant(1.0f32))]),
        )
        .vector_concat::<m![S / 8 % 4], m![S % 8]>()
        .vector_final()
        .to_vrf(0);

    // Reshape hidden copy to match VRF Slice decomposition for vector operations.
    let hidden_copy: DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S / 8 % 4, S % 8, H % 56]> =
        unsafe { hidden_copy.reshape() };

    // Apply normalization: x * inv_rms * weight, then cast back to bf16.
    let result_dm_y: DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32, H % 56]> = ctx
        .main
        .begin(hidden_copy.view())
        .fetch::<f32, m![S % 32, H / 8 % 7], m![H % 8]>()
        .collect::<m![S % 32, H / 8 % 7], m![H % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_split::<m![S % 32, H / 8 % 7, H / 4 % 2], m![H % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), &inv_rms_vrf)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &weight_vrf)
        .vector_concat::<m![S % 32, H / 8 % 7], m![H % 8]>()
        .vector_final()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit(0x1300);

    let result_dm: DmTensor<bf16, Chip, Cluster, m![Y, S / 32, H / 56], m![S % 32, H % 56]> = ctx
        .main
        .begin(result_dm_y.view())
        .fetch::<bf16, m![S % 32], m![H % 56]>()
        .collect::<m![S % 32], m![H % 56]>()
        .commit(0x1300);

    result_dm
}
