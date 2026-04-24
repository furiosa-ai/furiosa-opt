//! Simple VRF addition test.
//! Tests adding data from mainstream with VRF tensor using vector engine.

use furiosa_visa_std::prelude::*;

axes![A = 1024, B = 512];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Simple test: mainstream_data + vrf_data using VRF
fn vrf_add_kernel(
    ctx: &mut Context,
    lhs: HbmTensorView<i32, Chip, m![A, B]>,
    rhs: HbmTensorView<i32, Chip, m![B]>,
    out: DmTensorViewMut<i32, Chip, Cluster, m![A / 8 # 256], m![B, A % 8]>,
) {
    // Load lhs into DM (mainstream data)
    let lhs_dm = lhs.to_dm::<Cluster, m![A / 8 # 256], m![B, A % 8]>(&mut ctx.tdma, 0);

    // Load rhs into DM
    let rhs_dm = rhs.to_dm::<Cluster, m![A / 8 # 256], m![B]>(&mut ctx.tdma, 512);

    // Prepare rhs data for VRF by committing to DM first
    let rhs_vrf: VrfTensor<i32, Chip, Cluster, m![A / 8 # 256], m![B]> = ctx
        .sub
        .begin(rhs_dm.view())
        .fetch::<i32, m![1], m![B]>()
        .collect::<m![B / 8], m![B % 8]>()
        .to_vrf(0);

    // Perform addition: lhs_dm + rhs_vrf using vector engine
    ctx.main
        .begin(lhs_dm.view())
        .fetch::<i32, m![B], m![A % 8]>()
        .collect::<m![B], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::AddFxp, &rhs_vrf)
        .vector_final()
        .commit_view(out);
}

/// Add two tensors using VRF
#[device(chip = 1)]
pub fn vrf_add(
    ctx: &mut Context,
    lhs: &HbmTensor<i32, Chip, m![A, B]>,
    rhs: &HbmTensor<i32, Chip, m![B]>,
) -> HbmTensor<i32, Chip, m![A, B]> {
    type ResultDmTensor = DmTensor<i32, Chip, Cluster, m![A / 8 # 256], m![B, A % 8]>;
    let mut result = unsafe { ResultDmTensor::from_addr(0) };

    vrf_add_kernel(ctx, lhs.view(), rhs.view(), result.view_mut());

    // Write result back to HBM
    result.to_hbm(&mut ctx.tdma, 0x3000)
}
