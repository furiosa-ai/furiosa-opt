//! Matrix multiplication with chip reduce: (1024x4) * (4x1024) -> (1024x1024)
//! This example demonstrates reduction over the chip dimension.

use furiosa_visa_std::prelude::*;

axes![A = 1024, B = 4, C = 1024, I = 2];

type Chip = m![B]; // Chip dimension = 4
type Cluster = m![1 # 2];

/// Multiply matrices: [A, B] * [B, C] -> [A, C]
/// where B=4 is mapped to Chip dimension and needs to be reduced
#[device(chip = 4)]
pub fn matmul_chip_reduce(
    ctx: &mut Context,
    lhs: &HbmTensor<i8, Chip, m![A]>,
    rhs: &HbmTensor<i8, Chip, m![C]>,
) -> HbmTensor<i8, m![C / 2 % 4], m![A, C / 8, C % 2]> {
    let lhs = lhs.to_dm::<Cluster, m![A / 4], m![A % 4]>(&mut ctx.tdma, 0);

    let rhs = rhs.to_dm::<Cluster, m![C / 4], m![C % 4]>(&mut ctx.tdma, 128 * 1024);

    // Load rhs into VRF
    let rhs_broadcasted: DmTensor<i8, Chip, Cluster, m![A / 4], m![C / 4, C % 4 # 8]> = ctx
        .main
        .begin(rhs.view())
        .fetch::<i8, m![1], m![C % 4 # 8]>()
        .switch::<m![A / 4], m![C / 4]>(SwitchConfig::Broadcast01 {
            slice1: 256,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![C / 4], m![C % 4 # 32]>()
        .commit(0);
    let rhs_vrf: VrfTensor<i32, Chip, Cluster, m![A / 4], m![C / 4, C % 4 # 8]> = ctx
        .sub
        .begin(rhs_broadcasted.view())
        .fetch::<i32, m![C / 4], m![C % 4 # 8]>()
        .collect::<m![C / 4], m![C % 4 # 8]>()
        .to_vrf(0);

    // Perform elementwise mul
    // The B dimension is in Chip (=4), which will be reduced later via reduce_over_chip
    // Result shape: [Chip=4, Cluster=2, Slice=A/4, Element=[C, A%4 # 8]]
    let mul_result: DmTensor<i32, Chip, Cluster, m![A / 4], m![C, A % 4 # 8]> = ctx
        .main
        .begin(lhs.view())
        .fetch::<i32, m![C / 4, C % 4], m![A % 4]>()
        .collect::<m![C / 4, C % 4], m![A % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_branch(BranchMode::Unconditional)
        .vector_fxp(FxpBinaryOp::MulInt, &rhs_vrf)
        .vector_final()
        .commit(0);

    // Now reduce over the Chip dimension using ReduceScatter pattern (for 4 chips)
    let reduced = reduce_over_chip(ctx, &mul_result);

    // Write back to HBM
    reduced.to_hbm(&mut ctx.tdma, 0x3000)
}

/// Reduce over chip axis using ReduceScatter pattern for Chip=4.
fn reduce_over_chip(
    ctx: &mut Context,
    tensor: &DmTensor<i32, Chip, Cluster, m![A / 4], m![C, A % 4 # 8]>,
) -> DmTensor<i8, m![C / 2 % 4], Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> {
    // ReduceScatter for Chip=4
    // Input:  Chip=[B=4], Cluster=2, Slice=[A/4], Element=[C, A%4 # 8]
    // Output: Chip=[C/2%4], Cluster=2, Slice=[A/4], Element=[C/8, C%2, A%4 # 8]
    //
    // parallel_copy_chip_slice will:
    // - Take Element's first dimension (C) and slice it into 4 parts
    // - C splits as: C = (C/8) × 4 × (C%2), each chip picks 1 of 4
    // - Each chip ends up with C/4 = (C/8 × C%2) elements

    // Step 1: Create T0 with Slice(0,1,2,3)
    let sliced0: DmTensor<i32, Chip, Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> = ctx
        .sub
        .parallel_copy_chip_slice::<4, m![C / 2 % 4], m![C / 8, 1 # 4, C % 2, A % 4 # 8], _, _, _, _, _, _>(
            tensor.view(),
            &[0, 1, 2, 3],
        );

    // Step 2: Create T1 with Slice(3,0,1,2) + ChipShuffle(1,2,3,0)
    let sliced1: DmTensor<i32, Chip, Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> = ctx
        .sub
        .parallel_copy_chip_slice::<4, m![C / 2 % 4], m![C / 8, 1 # 4, C % 2, A % 4 # 8], _, _, _, _, _, _>(
            tensor.view(),
            &[3, 0, 1, 2],
        );

    let shuffled1: DmTensor<i32, Chip, Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> = ctx
        .tdma
        .dm_chip_shuffle::<_, 4, _, _, _, _>(sliced1.view(), &[1, 2, 3, 0]);

    // Step 3: Create T2 with Slice(2,3,0,1) + ChipShuffle(2,3,0,1)
    let sliced2: DmTensor<i32, Chip, Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> = ctx
        .sub
        .parallel_copy_chip_slice::<4, m![C / 2 % 4], m![C / 8, 1 # 4, C % 2, A % 4 # 8], _, _, _, _, _, _>(
            tensor.view(),
            &[2, 3, 0, 1],
        );

    let shuffled2: DmTensor<i32, Chip, Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> = ctx
        .tdma
        .dm_chip_shuffle::<_, 4, _, _, _, _>(sliced2.view(), &[2, 3, 0, 1]);

    // Step 4: Create T3 with Slice(1,2,3,0) + ChipShuffle(3,0,1,2)
    let sliced3: DmTensor<i32, Chip, Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> = ctx
        .sub
        .parallel_copy_chip_slice::<4, m![C / 2 % 4], m![C / 8, 1 # 4, C % 2, A % 4 # 8], _, _, _, _, _, _>(
            tensor.view(),
            &[1, 2, 3, 0],
        );

    let shuffled3: DmTensor<i32, Chip, Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> = ctx
        .tdma
        .dm_chip_shuffle::<_, 4, _, _, _, _>(sliced3.view(), &[3, 0, 1, 2]);

    // Step 5: Add T0 + T1 + T2 + T3 using two-stage binary addition
    let mut sum01: DmTensor<i32, Chip, Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> =
        unsafe { DmTensor::from_addr(0) };
    let mut sum012: DmTensor<i32, Chip, Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> =
        unsafe { DmTensor::from_addr(0) };

    // Add T0 + T1
    ctx.main
        .begin_interleaved::<I, _, _, _, _, _>(sliced0.view(), shuffled1.view())
        .fetch::<i32, m![C / 8, C % 2, I], m![A % 4 # 8]>()
        .collect::<m![C / 8, C % 2, I], m![A % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C / 8, C % 2, 1 # 2], m![C / 8, C % 2]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        // After vector_clip_zip, result is implicitly filtered to Group 1 only
        .vector_final()
        .commit_view(sum01.view_mut());

    // Add (TO + T1) + T2
    ctx.main
        .begin_interleaved::<I, _, _, _, _, _>(sum01.view(), shuffled2.view())
        .fetch::<i32, m![C / 8, C % 2, I], m![A % 4 # 8]>()
        .collect::<m![C / 8, C % 2, I], m![A % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C / 8, C % 2, 1 # 2], m![C / 8, C % 2]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        // After vector_clip_zip, result is implicitly filtered to Group 1 only
        .vector_final()
        .commit_view(sum012.view_mut());

    // Stage 2: Add ((T0 + T1) + T2) + T3 to get final result
    let mut reduced: DmTensor<i8, Chip, Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> =
        unsafe { DmTensor::from_addr(0) };

    ctx.main
        .begin_interleaved::<I, _, _, _, _, _>(sum012.view(), shuffled3.view())
        .fetch::<i32, m![C / 8, C % 2, I], m![A % 4 # 8]>()
        .collect::<m![C / 8, C % 2, I], m![A % 4 # 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C / 8, C % 2, 1 # 2], m![C / 8, C % 2]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        // After vector_clip_zip, result is implicitly filtered to Group 1 only
        .vector_final()
        .cast::<i8, m![A % 4 # 32]>()
        .commit_view(reduced.view_mut());

    // Reshape to output format
    // Input:  Chip=[B=4], Cluster, Slice=[A/4], Element=[C/8, C%2, A%4#8]
    // Output: Chip=[C/2%4], Cluster, Slice=[A/4], Element=[C/8, C%2, A%4#8]
    let reshaped: DmTensor<i8, m![C / 2 % 4], Cluster, m![A / 4], m![C / 8, C % 2, A % 4 # 8]> =
        unsafe { reduced.reshape() };

    reshaped
}
