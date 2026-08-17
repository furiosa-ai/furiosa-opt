//! Segmented VRF addition running on a single PE.
//!
//! Same computation as `npu-opt-examples::unsupported::vrf_add` (which does not reach EDF), but with
//! a smaller `A` and a single-cluster layout, committing the result in tiled segments.

use furiosa_opt_std::prelude::*;

axes![A = 128, B = 512];

type Chip = m![1];
type Cluster = m![1];

fn vrf_add_kernel_segmented(
    ctx: &mut Context,
    lhs: HbmTensorView<'_, i32, Chip, m![A, B]>,
    rhs: HbmTensorView<'_, i32, Chip, m![B]>,
    mut out: DmTensor<i32, Chip, Cluster, m![A / 2 # 64], m![A % 2, B]>,
) -> DmTensor<i32, Chip, Cluster, m![A / 2 # 64], m![A % 2, B]> {
    // Load lhs into DM (mainstream data)
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 2 # 64], m![A % 2, B]>(&mut ctx.tdma);

    // Load rhs into DM
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 2 # 64], m![B]>(&mut ctx.tdma);

    // Prepare rhs data for VRF by committing to DM first
    let rhs_vrf: VrfTensor<i32, Chip, Cluster, m![A / 2 # 64], m![B]> = ctx
        .sub
        .begin(rhs_dm.view())
        .fetch::<m![1], m![B]>()
        .fetch_cast::<i32>()
        .collect::<m![B / 8], m![B % 8]>()
        .to_vrf();

    for i in 0..2 {
        let lhs_dm_view = lhs_dm.view().tile::<m![A % 2], 1, m![A % 2 = 1 # 2, B]>(i);
        let out_view = out.view_mut().tile::<m![A % 2], 1, m![A % 2 = 1 #{!} 2, B]>(i);

        // Perform addition: lhs_dm + rhs_vrf using vector engine
        ctx.main
            .begin(lhs_dm_view)
            .fetch::<m![A % 2 = 1], m![B]>()
            .fetch_cast::<i32>()
            .collect::<m![A % 2 = 1, B / 8], m![B % 8]>()
            .vector_init()
            .vector_intra_slice_tag(TagMode::Zero)
            .vector_fxp(FxpBinaryOp::AddFxp, &rhs_vrf)
            .vector_final()
            .commit_trim::<m![B % 8]>()
            .commit_view(out_view);
    }

    out
}

/// Add two tensors using VRF, committing the result in tiled segments (1 PE).
#[device(chip = 1, pe = 1)]
pub fn vrf_add_segmented(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A, B]>,
    rhs: &HbmTensor<i32, Chip, m![B]>,
) -> HbmTensor<i32, Chip, m![A, B]> {
    type ResultDmTensor = DmTensor<i32, Chip, Cluster, m![A / 2 # 64], m![A % 2, B]>;
    let result = ResultDmTensor::new();

    let result = vrf_add_kernel_segmented(ctx, lhs.view(), rhs.view(), result);

    // Write result back to HBM
    result.to_hbm(&mut ctx.tdma)
}
