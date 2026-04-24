//! Binary addition of two tensors with size 2048.
//! Uses interleaved ALC vector engine to perform element-wise addition.

use furiosa_visa_std::prelude::*;

axes![A = 2048, I = 2];

type Chip = m![1];
type Cluster = m![1 # 2];

/// Add two tensors element-wise using interleaved ALC vector engine.
///
/// This function performs element-wise addition of two tensors of shape [A=2048].
/// Data flow: Host -> HBM -> DM with layout Chip: m![1], Cluster: m![1#2],
/// Slice: [A/8], Element: [A%8]
fn binary_add_kernel(
    ctx: &mut Context,
    lhs: HbmTensorView<i8, Chip, m![A]>,
    rhs: HbmTensorView<i8, Chip, m![A]>,
    out: DmTensorViewMut<i32, Chip, Cluster, m![A / 8], m![A % 8]>,
) {
    // Load lhs from HBM to DM with the desired layout
    // Slice: [A/8], Element: [A%8]
    let lhs = lhs.to_dm::<Cluster, m![A / 8], m![A % 8]>(&mut ctx.tdma, 0);

    // Load rhs from HBM to DM with the desired layout
    // Slice: [A/8], Element: [A%8]
    let rhs = rhs.to_dm::<Cluster, m![A / 8], m![A % 8]>(&mut ctx.tdma, 32 * 1024);

    // Perform element-wise addition using VectorTensorPair API
    // lhs will have Group 0, rhs will have Group 1
    // I=2 axis is reduced by binary add in VE
    ctx.main
        .begin_interleaved::<I, _, _, _, _, _>(lhs.view(), rhs.view())
        .fetch::<i32, m![I], m![A % 8]>()
        .collect::<m![I], m![A % 8]>()
        // init_unzip: creates VectorTensorPair directly
        .vector_init()
        .vector_intra_slice_unzip::<I, m![1 # 2], m![1]>()
        // zip operation: group1 + group0 = rhs + lhs
        // After vector_clip_zip, result is implicitly filtered to Group 1 only
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        .vector_final()
        .commit_view(out)
}

/// Add two tensors of size 2048 element-wise.
#[device(chip = 1)]
pub fn binary_add_2048(
    ctx: &mut Context,
    lhs: &HbmTensor<i8, m![1], m![A]>,
    rhs: &HbmTensor<i8, m![1], m![A]>,
) -> HbmTensor<i32, m![1], m![A]> {
    // Allocate DM tensor for output with layout:
    // Chip: m![1], Cluster: m![1#2], Slice: [A/8], Element: [A%8]
    type ResultDmTensor = DmTensor<i32, Chip, Cluster, m![A / 8], m![A % 8]>;
    let mut result = unsafe { ResultDmTensor::from_addr(0) };

    // Perform binary addition in DM
    binary_add_kernel(ctx, lhs.view(), rhs.view(), result.view_mut());

    // Write result back to HBM
    result.to_hbm(&mut ctx.tdma, 0x3000)
}
