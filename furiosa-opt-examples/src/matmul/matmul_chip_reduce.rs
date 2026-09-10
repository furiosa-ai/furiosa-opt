//! Matrix multiplication with chip reduce: (2048x4) * (4x2048) -> (2048x2048)
//! This example demonstrates reduction over the chip dimension.

use furiosa_opt_std::prelude::*;

axes![A = 2048, B = 4, C = 2048, I = 2];

type Chip = m![B]; // Chip dimension = 4
type Cluster = m![1 # 2];

/// Multiply matrices: [A, B] * [B, C] -> [A, C]
/// where B=4 is mapped to Chip dimension and needs to be reduced
#[device(chip = 4)]
pub fn matmul_chip_reduce(
    device: &mut Device,
    lhs: &HbmTensor<i8, Chip, m![A]>,
    rhs: &HbmTensor<i8, Chip, m![C]>,
) -> HbmTensor<i8, m![C / 512], m![A, C % 512]> {
    let lhs = lhs.to_dm::<Cluster, m![A / 8], m![A % 8]>(&mut device.tdma);

    let rhs = rhs.to_dm::<Cluster, m![C / 8], m![C % 8]>(&mut device.tdma);

    // Load rhs into VRF
    let rhs_broadcasted: DmTensor<i8, Chip, Cluster, m![A / 8], m![C / 8, C % 8]> = device
        .main
        .begin(rhs.view())
        .fetch::<m![1], m![C % 8]>()
        .switch::<m![A / 8], m![C / 8]>(SwitchConfig::Broadcast01 {
            slice1: 256,
            slice0: 1,
            time0: 1,
        })
        .collect::<m![C / 8], m![C % 8 # 32]>()
        .commit_trim::<m![C % 8]>()
        .commit();
    let rhs_vrf: VrfTensor<i32, Chip, Cluster, m![A / 8], m![C / 8, C % 8]> = device
        .sub
        .begin(rhs_broadcasted.view())
        .fetch::<m![C / 8], m![C % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![C / 8], m![C % 8]>()
        .to_vrf();

    // Perform elementwise mul
    // The B dimension is in Chip (=4), which will be reduced later via reduce_over_chip
    // Result shape: [Chip=4, Cluster=2, Slice=A/8, Element=[C, A%8]]
    let mul_result: DmTensor<i32, Chip, Cluster, m![A / 8], m![C, A % 8]> = device
        .main
        .begin(lhs.view())
        .fetch::<m![C / 8, C % 8], m![A % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![C / 8, C % 8], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::MulInt, &rhs_vrf)
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit();

    // Now reduce over the Chip dimension using ReduceScatter pattern (for 4 chips)
    let reduced = reduce_over_chip(device, &mul_result);

    // Write back to HBM
    reduced.to_hbm(&mut device.tdma)
}

/// Reduce over chip axis using ReduceScatter pattern for Chip=4.
fn reduce_over_chip(
    device: &mut Device,
    tensor: &DmTensor<i32, Chip, Cluster, m![A / 8], m![C, A % 8]>,
) -> DmTensor<i8, m![C / 512], Cluster, m![A / 8], m![C % 512, A % 8]> {
    // ReduceScatter for Chip=4
    // Input:  Chip=[B=4], Cluster=2, Slice=[A/8], Element=[C, A%8]
    // Output: Chip=[C/512], Cluster=2, Slice=[A/8], Element=[C%512, A%8]
    //
    // The slice must address `Element`'s outermost axis, so it is C's outermost factor: each chip
    // picks one of the four C/256 positions and keeps that quarter.

    // Step 1: Create T0 with Slice(0,1,2,3)
    let sliced0: DmTensor<i32, Chip, Cluster, m![A / 8], m![C % 512, A % 8]> =
        tensor.asymmetric_chip_slice::<m![C / 512], _>(&mut device.sub, &[0, 1, 2, 3]);

    // Step 2: Create T1 with Slice(3,0,1,2) + ChipShuffle(1,2,3,0)
    let sliced1: DmTensor<i32, Chip, Cluster, m![A / 8], m![C % 512, A % 8]> =
        tensor.asymmetric_chip_slice::<m![C / 512], _>(&mut device.sub, &[3, 0, 1, 2]);

    let shuffled1: DmTensor<i32, Chip, Cluster, m![A / 8], m![C % 512, A % 8]> =
        sliced1.view().chip_shuffle([1, 2, 3, 0]).to_dm(&mut device.tdma);

    // Step 3: Create T2 with Slice(2,3,0,1) + ChipShuffle(2,3,0,1)
    let sliced2: DmTensor<i32, Chip, Cluster, m![A / 8], m![C % 512, A % 8]> =
        tensor.asymmetric_chip_slice::<m![C / 512], _>(&mut device.sub, &[2, 3, 0, 1]);

    let shuffled2: DmTensor<i32, Chip, Cluster, m![A / 8], m![C % 512, A % 8]> =
        sliced2.view().chip_shuffle([2, 3, 0, 1]).to_dm(&mut device.tdma);

    // Step 4: Create T3 with Slice(1,2,3,0) + ChipShuffle(3,0,1,2)
    let sliced3: DmTensor<i32, Chip, Cluster, m![A / 8], m![C % 512, A % 8]> =
        tensor.asymmetric_chip_slice::<m![C / 512], _>(&mut device.sub, &[1, 2, 3, 0]);

    let shuffled3: DmTensor<i32, Chip, Cluster, m![A / 8], m![C % 512, A % 8]> =
        sliced3.view().chip_shuffle([3, 0, 1, 2]).to_dm(&mut device.tdma);

    // Step 5: Add T0 + T1 + T2 + T3 using two-stage binary addition
    let mut sum01: DmTensor<i32, Chip, Cluster, m![A / 8], m![C % 512, A % 8]> = DmTensor::new();
    let mut sum012: DmTensor<i32, Chip, Cluster, m![A / 8], m![C % 512, A % 8]> = DmTensor::new();

    // Add T0 + T1
    device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(sliced0.view(), shuffled1.view())
        .fetch::<m![C % 512, I], m![A % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![C % 512, I], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C % 512]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        // After vector_clip_zip, result is implicitly filtered to Group 1 only
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit_view(sum01.view_mut());

    // Add (TO + T1) + T2
    device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(sum01.view(), shuffled2.view())
        .fetch::<m![C % 512, I], m![A % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![C % 512, I], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C % 512]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        // After vector_clip_zip, result is implicitly filtered to Group 1 only
        .vector_final()
        .commit_trim::<m![A % 8]>()
        .commit_view(sum012.view_mut());

    // Stage 2: Add ((T0 + T1) + T2) + T3 to get final result
    let mut reduced: DmTensor<i8, Chip, Cluster, m![A / 8], m![C % 512, A % 8]> = DmTensor::new();

    device
        .main
        .begin_interleaved::<I, _, _, _, _, _>(sum012.view(), shuffled3.view())
        .fetch::<m![C % 512, I], m![A % 8]>()
        .fetch_cast::<i32>()
        .collect::<m![C % 512, I], m![A % 8]>()
        .vector_init()
        .vector_intra_slice_unzip::<I, m![C % 512]>()
        .vector_clip_zip(ClipBinaryOpI32::AddFxp)
        // After vector_clip_zip, result is implicitly filtered to Group 1 only
        .vector_final()
        .cast::<i8, m![A % 8 # 32]>()
        .commit_trim::<m![A % 8]>()
        .commit_view(reduced.view_mut());

    // SAFETY: B and C / 512 both have size 4; every other mapping is unchanged.
    let reshaped: DmTensor<i8, m![C / 512], Cluster, m![A / 8], m![C % 512, A % 8]> = unsafe { reduced.reshape() };

    reshaped
}
