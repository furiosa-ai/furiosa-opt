use furiosa_opt_std::prelude::*;

use crate::transformer::Chip;
use crate::transformer::axes::{D, G, H, N, P, Q, T};
use crate::transformer::ops::{SliceN32, SliceP4};

type Cluster = m![1 # 2];

pub(crate) fn proj_q(
    ctx: &mut Context,
    input: DmTensorView<'_, bf16, Chip, Cluster, SliceP4, m![H]>,
    weight: &HbmTensor<bf16, Chip, m![Q, H]>,
) -> DmTensor<bf16, Chip, Cluster, SliceN32, m![G, D]> {
    type SliceQ = m![Q / 32, 1 # 4];

    let input: DmTensorView<'_, bf16, Chip, Cluster, SliceQ, m![H]> = unsafe { input.reshape() };

    let weight: DmTensor<bf16, Chip, Cluster, SliceQ, m![Q % 32, H]> = weight.to_dm(&mut ctx.tdma);
    let weight_trf: TrfTensor<bf16, Chip, Cluster, SliceQ, m![Q / 4 % 8], m![Q % 4, H]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![Q % 32, H / 16], m![H % 16]>()
        .collect::<m![Q % 32, H / 16], m![H % 16]>()
        .to_trf();

    let result: DmTensor<bf16, Chip, Cluster, SliceQ, m![Q % 32]> = ctx
        .main
        .begin(input)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .contract_outer::<m![H / 32, Q % 4], m![H % 32], _, _, _>(&weight_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Q % 4]>()
        .contract_lane::<m![Q % 4], m![Q / 4 % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![Q / 4 % 8 # 16]>()
        .transpose::<m![Q / 4 % 8], m![Q % 4 # 16]>()
        .commit_trim::<m![Q % 4]>()
        .commit();

    let result: DmTensor<bf16, Chip, Cluster, m![Q / 256, 1 # 32], m![Q % 256]> = ctx
        .main
        .begin(result.view())
        .fetch::<m![1], m![Q % 32]>()
        .switch::<m![Q / 256, 1 # 32], m![Q / 32 % 8]>(SwitchConfig::Broadcast1 { slice1: 8, slice0: 4 })
        .collect::<m![Q / 16 % 16], m![Q % 16]>()
        .commit_trim::<m![Q % 16]>()
        .commit();

    unsafe { result.reshape() }
}

/// Shared `[P]` projection (`x @ weight^T`) used by both the K and V heads.
fn proj_p(
    ctx: &mut Context,
    input: DmTensorView<'_, bf16, Chip, Cluster, SliceP4, m![H]>,
    weight: &HbmTensor<bf16, Chip, m![P, H]>,
) -> DmTensor<bf16, Chip, Cluster, SliceN32, m![D]> {
    type SliceP = m![P / 32, 1 # 8];

    let input: DmTensorView<'_, bf16, Chip, Cluster, SliceP, m![H]> = unsafe { input.reshape() };

    let weight: DmTensor<bf16, Chip, Cluster, SliceP, m![P % 32, H]> = weight.to_dm(&mut ctx.tdma);
    let weight_trf: TrfTensor<bf16, Chip, Cluster, SliceP, m![P % 8], m![P / 8 % 4, H]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![P % 8, P / 8 % 4, H / 16], m![H % 16]>()
        .collect::<m![P % 8, P / 8 % 4, H / 16], m![H % 16]>()
        .to_trf();

    let result: DmTensor<bf16, Chip, Cluster, SliceP, m![P % 32]> = ctx
        .main
        .begin(input)
        .fetch::<m![H / 16], m![H % 16]>()
        .collect::<m![H / 16], m![H % 16]>()
        .contract_outer::<m![H / 32, P / 8 % 4], m![H % 32], _, _, _>(&weight_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![P / 8 % 4]>()
        .contract_lane::<m![P / 8 % 4], m![P % 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![P % 8 # 16]>()
        .commit_trim::<m![P % 8]>()
        .commit();

    let result: DmTensor<bf16, Chip, Cluster, m![P / 128, 1 # 32], m![P % 128]> = ctx
        .main
        .begin(result.view())
        .fetch::<m![1], m![P % 32]>()
        .switch::<m![P / 128, 1 # 32], m![P / 32 % 4]>(SwitchConfig::Broadcast1 { slice1: 4, slice0: 8 })
        .collect::<m![P / 16 % 8], m![P % 16]>()
        .commit_trim::<m![P % 16]>()
        .commit();

    unsafe { result.reshape() }
}

pub(crate) fn proj_k(
    ctx: &mut Context,
    input: DmTensorView<'_, bf16, Chip, Cluster, SliceP4, m![H]>,
    weight: &HbmTensor<bf16, Chip, m![P, H]>,
) -> DmTensor<bf16, Chip, Cluster, SliceN32, m![D]> {
    proj_p(ctx, input, weight)
}

pub(crate) fn proj_v(
    ctx: &mut Context,
    input: DmTensorView<'_, bf16, Chip, Cluster, SliceP4, m![H]>,
    kv_offset: &HbmTensor<i32, Chip, m![1]>,
    weight: &HbmTensor<bf16, Chip, m![P, H]>,
    v_cache: &mut HbmTensor<bf16, Chip, m![T, N, D]>,
) {
    let result = proj_p(ctx, input, weight);
    result.dma_scatter::<m![1], _, _>(kv_offset, v_cache);
}

pub(crate) fn proj_o(
    ctx: &mut Context,
    input: DmTensor<bf16, Chip, Cluster, SliceP4, m![Q]>,
    weight: &HbmTensor<bf16, Chip, m![H, Q]>,
) -> DmTensor<bf16, Chip, Cluster, SliceP4, m![H]> {
    type SliceO = m![H / 16, 1 # 4];

    let input: DmTensor<bf16, Chip, Cluster, SliceO, m![Q]> = unsafe { input.reshape() };

    let weight: DmTensor<bf16, Chip, Cluster, SliceO, m![H % 16, Q]> = weight.to_dm(&mut ctx.tdma);
    let weight_trf: TrfTensor<bf16, Chip, Cluster, SliceO, m![H / 4 % 4], m![H % 4, Q]> = ctx
        .sub
        .begin(weight.view())
        .fetch::<m![H % 16, Q / 16], m![Q % 16]>()
        .collect::<m![H % 16, Q / 16], m![Q % 16]>()
        .to_trf();

    let result: DmTensor<bf16, Chip, Cluster, SliceO, m![H % 16]> = ctx
        .main
        .begin(input.view())
        .fetch::<m![Q / 16], m![Q % 16]>()
        .collect::<m![Q / 16], m![Q % 16]>()
        .contract_outer::<m![Q / 32, H % 4], m![Q % 32], _, _, _>(&weight_trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![H % 4]>()
        .contract_lane::<m![H % 4], m![H / 4 % 4 # 8]>(LaneMode::Interleaved)
        .cast::<bf16, m![H / 4 % 4 # 16]>()
        .transpose::<m![H / 4 % 4], m![H % 4 # 16]>()
        .commit_trim::<m![H % 4]>()
        .commit();

    ctx.main
        .begin(result.view())
        .fetch::<m![1], m![H % 16]>()
        .switch::<SliceP4, m![H / 16]>(SwitchConfig::Broadcast1 { slice1: 64, slice0: 4 })
        .collect::<m![H / 16], m![H % 16]>()
        .commit_trim::<m![H % 16]>()
        .commit()
}
