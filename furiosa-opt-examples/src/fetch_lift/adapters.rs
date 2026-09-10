use furiosa_opt_std::prelude::*;

use super::{A, Chip, Cluster, H, Row, Slice};

axes![Oct = 8];

/// Lifts `H` onto `Slice` before widening `i8` values to `i32`.
#[device(chip = 1)]
pub fn fetch_slice_lift_then_cast(
    device: &mut Device,
    input: &HbmTensor<i8, Chip, m![A, H, Oct]>,
) -> HbmTensor<i32, Chip, m![A, H, Oct]> {
    let dm: DmTensor<i8, Chip, Cluster, Slice, m![H, Oct]> =
        input.to_dm::<Cluster, Slice, m![H, Oct]>(&mut device.tdma);

    let result: DmTensor<i32, Chip, Cluster, m![A, H], m![Oct]> = device
        .main
        .begin(dm.view())
        .fetch::<m![H], m![Oct]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![Oct]>()
        .commit_trim::<m![Oct]>()
        .commit();

    result.to_hbm::<m![A, H, Oct]>(&mut device.tdma)
}

axes![Code = 16];

/// Lifts `H` onto `Slice`, decodes `f4e2m1`, and widens the result to `f32`.
#[device(chip = 1)]
pub fn fetch_slice_lift_table_lookup(
    device: &mut Device,
    input: &HbmTensor<f4e2m1, Chip, m![A, H, Code]>,
) -> HbmTensor<f32, Chip, m![A, H, Code]> {
    let dm: DmTensor<f4e2m1, Chip, Cluster, Slice, m![H, Code]> =
        input.to_dm::<Cluster, Slice, m![H, Code]>(&mut device.tdma);

    let result: DmTensor<f32, Chip, Cluster, m![A, H], m![Code]> = device
        .main
        .begin(dm.view())
        .fetch::<m![H], m![Code]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .fetch_table_lookup::<f8e4m3>()
        .fetch_cast::<f32>()
        .collect::<m![Code / 8], m![Code % 8]>()
        .commit_trim::<m![Code % 8]>()
        .commit();

    result.to_hbm::<m![A, H, Code]>(&mut device.tdma)
}

axes![Red = 4, Z = 2];

/// Lifts `Z` onto `Cluster` before reducing `Red` across slices.
#[device(chip = 1)]
pub fn fetch_cluster_lift_inter_slice_reduce(
    device: &mut Device,
    input: &HbmTensor<i32, Chip, m![Row, Red, Z, Oct]>,
) -> HbmTensor<i32, Chip, m![Z, Row, Oct]> {
    let dm: DmTensor<i32, Chip, m![2], m![Row, Red], m![Z, Oct]> =
        input.to_dm::<m![2], m![Row, Red], m![Z, Oct]>(&mut device.tdma);

    let result: DmTensor<i32, Chip, m![Z], m![Row, 1 # 4], m![Oct]> = device
        .main
        .begin(dm.view())
        .fetch::<m![Z], m![Oct]>()
        .fetch_cluster_lift::<m![Z], m![1]>()
        .collect::<m![1], m![Oct]>()
        .vector_init()
        .vector_inter_slice_reduce::<m![Row, 1 # 4], m![1]>(InterSliceReduceOpI32::AddSat)
        .vector_final()
        .commit_trim::<m![Oct]>()
        .commit();

    result.to_hbm::<m![Z, Row, Oct]>(&mut device.tdma)
}

axes![P128 = 128, Zp = 2, Act = 8, Dot = 32, Out = 8];

type ZeroPointSlice = m![P128, 2];
type LiftedZeroPointSlice = m![P128, Zp];
type ZeroPointLane = m![Out];

/// Lifts `Zp` onto `Slice` before zero-point subtraction and contraction.
#[device(chip = 1)]
pub fn lift_zero_point_sub_contract(
    device: &mut Device,
    input: &HbmTensor<i8, Chip, m![P128, Zp, Act, Dot]>,
    weight: &HbmTensor<i8, Chip, m![P128, Zp, Out, Dot]>,
) -> HbmTensor<i32, Chip, m![P128, Zp, Act, Out]> {
    let input_dm = input.to_dm::<Cluster, ZeroPointSlice, m![Zp, Act, Dot]>(&mut device.tdma);
    let weight_dm = weight.to_dm::<Cluster, LiftedZeroPointSlice, m![Out, Dot]>(&mut device.tdma);

    let trf: TrfTensor<i8, Chip, Cluster, LiftedZeroPointSlice, ZeroPointLane, m![Dot]> = device
        .sub
        .begin(weight_dm.view())
        .fetch::<m![Out], m![Dot]>()
        .fetch_cast::<i8>()
        .collect::<m![Out], m![Dot]>()
        .to_trf();

    let result: DmTensor<i32, Chip, Cluster, LiftedZeroPointSlice, m![Act, Out]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![Zp, Act], m![Dot]>()
        .fetch_slice_lift::<LiftedZeroPointSlice, m![Act]>()
        .fetch_zero_point_sub::<i9>(3)
        .collect::<m![Act], m![Dot]>()
        .contract_outer::<m![Act], m![Dot], _, _, i8>(&trf)
        .contract_packet::<m![1]>()
        .contract_time::<m![Act]>()
        .contract_lane::<m![Act], m![Out]>(LaneMode::Interleaved)
        .commit_trim::<m![Out]>()
        .commit();

    result.to_hbm::<m![P128, Zp, Act, Out]>(&mut device.tdma)
}
