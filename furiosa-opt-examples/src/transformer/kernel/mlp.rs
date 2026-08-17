use furiosa_opt_std::prelude::*;

use crate::transformer::Chip;
use crate::transformer::axes::{Dummy4, H, L};
use crate::transformer::ops::SliceP4;

type Cluster = m![1 # 2];

pub(crate) fn forward(
    ctx: &mut Context,
    input: DmTensor<bf16, Chip, Cluster, SliceP4, m![H]>,
    up_weight: &HbmTensor<bf16, Chip, m![L, H]>,
    gate_weight: &HbmTensor<bf16, Chip, m![L, H]>,
    down_weight: &HbmTensor<bf16, Chip, m![H, L]>,
) -> DmTensor<bf16, Chip, Cluster, SliceP4, m![H]> {
    type SliceUG = m![L / 24, 1 # 2];

    let x: DmTensor<bf16, Chip, Cluster, m![L / 48, 1 # 4], m![H]> = unsafe { input.reshape() };
    let x: DmTensor<bf16, Chip, Cluster, m![L / 48, Dummy4], m![H]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .switch::<m![L / 48, Dummy4], m![H / 16]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit();
    let x: DmTensor<bf16, Chip, Cluster, SliceUG, m![H]> = unsafe { x.reshape() };

    let up_weight: DmTensor<bf16, Chip, Cluster, SliceUG, m![L % 24, H]> = up_weight.to_dm(&mut ctx.tdma);
    let up_weight_trf: TrfTensor<bf16, Chip, Cluster, SliceUG, m![L % 8], m![L / 8 % 3, H]> = ctx
        .sub
        .begin(up_weight.view())
        .fetch::<m![L % 8, L / 8 % 3, H / 16], m![H % 16]>()
        .collect::<m![L % 8, L / 8 % 3, H / 16], m![H % 16]>()
        .to_trf();

    let up: DmTensor<bf16, Chip, Cluster, SliceUG, m![L % 24]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .contract_outer::<m![H / 32, L / 8 % 3], m![H % 32], _, _, _>(&up_weight_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L / 8 % 3]>()
        .contract_lane::<m![L / 8 % 3], m![L % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let gate_weight: DmTensor<bf16, Chip, Cluster, SliceUG, m![L % 24, H]> = gate_weight.to_dm(&mut ctx.tdma);
    let gate_weight_trf: TrfTensor<bf16, Chip, Cluster, SliceUG, m![L % 8], m![L / 8 % 3, H]> = ctx
        .sub
        .begin(gate_weight.view())
        .fetch::<m![L % 8, L / 8 % 3, H / 16], m![H % 16]>()
        .collect::<m![L % 8, L / 8 % 3, H / 16], m![H % 16]>()
        .to_trf();

    let gate: DmTensor<bf16, Chip, Cluster, SliceUG, m![L % 24]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .contract_outer::<m![H / 32, L / 8 % 3], m![H % 32], _, _, _>(&gate_weight_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![L / 8 % 3]>()
        .contract_lane::<m![L / 8 % 3], m![L % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let silu: DmTensor<f32, Chip, Cluster, SliceUG, m![L % 24]> = ctx
        .sub
        .begin(gate.view())
        .fetch::<m![L / 8 % 3], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 3], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 6], m![L % 4]>()
        .vector_stash()
        .vector_fp_unary(FpUnaryOp::Sigmoid)
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul0), Stash)
        .vector_widen_concat::<m![L / 8 % 3], m![L % 8]>()
        .vector_final()
        .commit_trim::<m![L % 8]>()
        .commit();

    let silu_vrf: VrfTensor<f32, Chip, Cluster, SliceUG, m![L % 24]> = ctx
        .sub
        .begin(silu.view())
        .fetch::<m![L / 8 % 3], m![L % 8]>()
        .collect::<m![L / 8 % 3], m![L % 8]>()
        .to_vrf();

    let x: DmTensor<bf16, Chip, Cluster, SliceUG, m![L % 24]> = ctx
        .main
        .begin(up.view())
        .fetch::<m![L / 8 % 3], m![L % 8]>()
        .fetch_cast::<f32>()
        .collect::<m![L / 8 % 3], m![L % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_narrow_split::<m![L / 4 % 6], m![L % 4]>()
        .vector_fp_binary(FpBinaryOp::MulF(FpMulAlu::Mul1), &silu_vrf)
        .vector_widen_concat::<m![L / 8 % 3], m![L % 8]>()
        .vector_final()
        .cast::<bf16, m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    type SliceD = m![H / 8, 1 # 2];
    let x: DmTensor<bf16, Chip, Cluster, SliceD, m![L]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![L / 8 % 3], m![L % 8 # 16]>()
        .switch::<SliceD, m![L / 8 % 3, L / 24]>(SwitchConfig::Broadcast1 { slice1: 128, slice0: 2 })
        .collect::<m![L / 8 % 3, L / 24], m![L % 8 # 16]>()
        .commit_trim::<m![L % 8]>()
        .commit();

    let down_weight: DmTensor<bf16, Chip, Cluster, SliceD, m![H % 8, L]> = down_weight.to_dm(&mut ctx.tdma);

    let weight_trf: TrfTensor<bf16, Chip, Cluster, SliceD, m![H % 8], m![L]> = ctx
        .sub
        .begin(down_weight.view())
        .fetch::<m![H % 8, L / 16], m![L % 16]>()
        .collect::<m![H % 8, L / 16], m![L % 16]>()
        .to_trf();

    let down: DmTensor<f32, Chip, Cluster, SliceD, m![H % 8]> = ctx
        .main
        .begin(x.view())
        .fetch::<m![L / 16], m![L % 16]>()
        .collect::<m![L / 16], m![L % 16]>()
        .contract_outer::<m![L / 32], m![L % 32], _, _, _>(&weight_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![1]>()
        .contract_lane::<m![1], m![H % 8]>(LaneMode::Interleaved)
        .commit_trim::<m![H % 8]>()
        .commit();

    ctx.main
        .begin(down.view())
        .fetch::<m![1], m![H % 8]>()
        .switch::<SliceP4, m![H / 8]>(SwitchConfig::Broadcast1 { slice1: 128, slice0: 2 })
        .collect::<m![H / 8], m![H % 8]>()
        .cast::<bf16, m![H % 8 # 16]>()
        .commit_trim::<m![H % 8]>()
        .commit()
}
