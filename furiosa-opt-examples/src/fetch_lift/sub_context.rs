use furiosa_opt_std::prelude::*;

use super::{A, Chip, Cluster, FourChips, H, PaddedSlice, Q, Slice, V};

/// Runs the cluster-lift example on the sub execution context.
#[device(chip = 1)]
pub fn fetch_sub_cluster_lift_axis(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A, H, V]>,
) -> HbmTensor<bf16, Chip, m![H, A, V]> {
    let dm: DmTensor<bf16, Chip, m![2], PaddedSlice, m![H, V]> =
        input.to_dm::<m![2], PaddedSlice, m![H, V]>(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, m![H], PaddedSlice, m![V]> = ctx
        .sub
        .begin(dm.view())
        .fetch::<m![H], m![V]>()
        .fetch_cluster_lift::<m![H], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![H, A, V]>(&mut ctx.tdma)
}

/// Runs the slice-lift example on the sub execution context.
#[device(chip = 1)]
pub fn fetch_sub_slice_lift_axis(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A, H, V]>,
) -> HbmTensor<bf16, Chip, m![A, H, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, Slice, m![H, V]> = input.to_dm::<Cluster, Slice, m![H, V]>(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![A, H], m![V]> = ctx
        .sub
        .begin(dm.view())
        .fetch::<m![H], m![V]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![A, H, V]>(&mut ctx.tdma)
}

/// Runs the chip-lift example on the sub execution context.
#[device(chip = 4)]
pub fn fetch_sub_chip_lift_axis(
    ctx: &mut Context,
    input: &HbmTensor<bf16, FourChips, m![A, Q, V]>,
) -> HbmTensor<bf16, m![Q], m![A, V]> {
    let dm: DmTensor<bf16, FourChips, Cluster, PaddedSlice, m![Q, V]> =
        input.to_dm::<Cluster, PaddedSlice, m![Q, V]>(&mut ctx.tdma);

    let result: DmTensor<bf16, m![Q], Cluster, PaddedSlice, m![V]> = ctx
        .sub
        .begin(dm.view())
        .fetch::<m![Q], m![V]>()
        .fetch_chip_lift::<m![Q], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![A, V]>(&mut ctx.tdma)
}

axes![Part = 32, Step = 4, Beat = 8];
type LiftAndSwitchSlice = m![Part, 2, 1 # 4];

/// Combines a slice lift with a custom-broadcast switch on the sub context.
#[device(chip = 1)]
pub fn fetch_sub_slice_lift_custom_switch(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![Part, H, Beat, V]>,
) -> HbmTensor<bf16, Chip, m![Part, H, Step, Beat, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, LiftAndSwitchSlice, m![H, Beat, V]> =
        input.to_dm::<Cluster, LiftAndSwitchSlice, m![H, Beat, V]>(&mut ctx.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![Part, H, Step], m![Beat, V]> = ctx
        .sub
        .begin(dm.view())
        .fetch::<m![H, Beat], m![V]>()
        .fetch_slice_lift::<m![Part, H, 1 # 4], m![Beat]>()
        .switch::<m![Part, H, Step], m![Beat]>(SwitchConfig::CustomBroadcast { ring_size: 4 })
        .collect::<m![Beat], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![Part, H, Step, Beat, V]>(&mut ctx.tdma)
}

axes![Win = 32];

/// Lifts a slice dimension from dynamically selected DM windows.
#[device(chip = 1)]
pub fn fetch_slice_lift_dynamic_view(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A, H, Win]>,
) -> HbmTensor<bf16, Chip, m![A, H, Win]> {
    let dm: DmTensor<bf16, Chip, Cluster, Slice, m![H, Win]> = input.to_dm::<Cluster, Slice, m![H, Win]>(&mut ctx.tdma);
    let mut result = DmTensor::<bf16, Chip, Cluster, m![A, H], m![Win]>::new();

    for window_index in 0..2 {
        let offset: usize = if window_index == 0 { 0 } else { 16 };
        let window = dm.view().tile::<m![Win], 16, m![H, Win = 16 # 32]>(offset);
        let out_window = result.view_mut().tile::<m![Win], 16, m![Win = 16 #{!} 32]>(offset);

        ctx.main
            .begin(window)
            .fetch::<m![H], m![Win = 16]>()
            .fetch_slice_lift::<m![A, H], m![1]>()
            .collect::<m![1], m![Win = 16]>()
            .commit_trim::<m![Win = 16]>()
            .commit_view(out_window);
    }

    result.to_hbm::<m![A, H, Win]>(&mut ctx.tdma)
}

/// Runs the dynamic-view slice lift on the sub execution context.
#[device(chip = 1)]
pub fn fetch_sub_slice_lift_dynamic_view(
    ctx: &mut Context,
    input: &HbmTensor<bf16, Chip, m![A, H, Win]>,
) -> HbmTensor<bf16, Chip, m![A, H, Win]> {
    let dm: DmTensor<bf16, Chip, Cluster, Slice, m![H, Win]> = input.to_dm::<Cluster, Slice, m![H, Win]>(&mut ctx.tdma);
    let mut result = DmTensor::<bf16, Chip, Cluster, m![A, H], m![Win]>::new();

    for window_index in 0..2 {
        let offset: usize = if window_index == 0 { 0 } else { 16 };
        let window = dm.view().tile::<m![Win], 16, m![H, Win = 16 # 32]>(offset);
        let out_window = result.view_mut().tile::<m![Win], 16, m![Win = 16 #{!} 32]>(offset);

        ctx.sub
            .begin(window)
            .fetch::<m![H], m![Win = 16]>()
            .fetch_slice_lift::<m![A, H], m![1]>()
            .collect::<m![1], m![Win = 16]>()
            .commit_trim::<m![Win = 16]>()
            .commit_view(out_window);
    }

    result.to_hbm::<m![A, H, Win]>(&mut ctx.tdma)
}

axes![Word = 8];

/// Stores a lifted sub-context fetch to VRF through the vector engine.
#[device(chip = 1)]
pub fn fetch_sub_slice_lift_to_vrf(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![A, H, Word]>,
    addend: &HbmTensor<i32, Chip, m![A, H, Word]>,
) -> HbmTensor<i32, Chip, m![A, H, Word]> {
    let dm: DmTensor<i32, Chip, Cluster, Slice, m![H, Word]> =
        input.to_dm::<Cluster, Slice, m![H, Word]>(&mut ctx.tdma);
    let addend_dm: DmTensor<i32, Chip, Cluster, m![A, H], m![Word]> =
        addend.to_dm::<Cluster, m![A, H], m![Word]>(&mut ctx.tdma);

    let operand: VrfTensor<i32, Chip, Cluster, m![A, H], m![Word]> = ctx
        .sub
        .begin(dm.view())
        .fetch::<m![H], m![Word]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .collect::<m![1], m![Word]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 0)
        .vector_final()
        .to_vrf();

    let result: DmTensor<i32, Chip, Cluster, m![A, H], m![Word]> = ctx
        .main
        .begin(addend_dm.view())
        .fetch::<m![1], m![Word]>()
        .collect::<m![1], m![Word]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, &operand)
        .vector_final()
        .commit_trim::<m![Word]>()
        .commit();

    result.to_hbm::<m![A, H, Word]>(&mut ctx.tdma)
}

/// Stores a lifted sub-context fetch directly to VRF.
#[device(chip = 1)]
pub fn fetch_sub_slice_lift_bare_to_vrf(
    ctx: &mut Context,
    input: &HbmTensor<i32, Chip, m![A, H, Word]>,
    addend: &HbmTensor<i32, Chip, m![A, H, Word]>,
) -> HbmTensor<i32, Chip, m![A, H, Word]> {
    let dm: DmTensor<i32, Chip, Cluster, Slice, m![H, Word]> =
        input.to_dm::<Cluster, Slice, m![H, Word]>(&mut ctx.tdma);
    let addend_dm: DmTensor<i32, Chip, Cluster, m![A, H], m![Word]> =
        addend.to_dm::<Cluster, m![A, H], m![Word]>(&mut ctx.tdma);

    let operand: VrfTensor<i32, Chip, Cluster, m![A, H], m![Word]> = ctx
        .sub
        .begin(dm.view())
        .fetch::<m![H], m![Word]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .collect::<m![1], m![Word]>()
        .to_vrf();

    let result: DmTensor<i32, Chip, Cluster, m![A, H], m![Word]> = ctx
        .main
        .begin(addend_dm.view())
        .fetch::<m![1], m![Word]>()
        .collect::<m![1], m![Word]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, &operand)
        .vector_final()
        .commit_trim::<m![Word]>()
        .commit();

    result.to_hbm::<m![A, H, Word]>(&mut ctx.tdma)
}
