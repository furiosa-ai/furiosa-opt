//! Fetch axis-lifting examples for `Chip`, `Cluster`, and `Slice`.

use furiosa_opt_std::prelude::*;

mod adapters;
pub mod mix;
pub mod reshape;
mod sub_context;
pub mod trf;

pub use adapters::{
    Act, Code, Dot, Oct, Out, P128, Red, Z, Zp, fetch_cluster_lift_inter_slice_reduce, fetch_slice_lift_table_lookup,
    fetch_slice_lift_then_cast, lift_zero_point_sub_contract,
};
pub use reshape::{HV, fetch_slice_lift_reshaped_broadcast, fetch_slice_lift_reshaped_named_broadcast};
pub use sub_context::{
    Beat, Part, Step, Win, Word, fetch_slice_lift_dynamic_view, fetch_sub_chip_lift_axis, fetch_sub_cluster_lift_axis,
    fetch_sub_slice_lift_axis, fetch_sub_slice_lift_bare_to_vrf, fetch_sub_slice_lift_custom_switch,
    fetch_sub_slice_lift_dynamic_view, fetch_sub_slice_lift_to_vrf,
};

axes![A = 128, H = 2, Q = 4, V = 16];

type Chip = m![1];
type Cluster = m![1 # 2];
type Slice = m![A, 2];

/// Lifts `H` from fetch time onto `Slice`.
#[device(chip = 1)]
pub fn fetch_slice_lift_axis(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![A, H, V]>,
) -> HbmTensor<bf16, Chip, m![A, H, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, Slice, m![H, V]> = input.to_dm::<Cluster, Slice, m![H, V]>(&mut device.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![A, H], m![V]> = device
        .main
        .begin(dm.view())
        .fetch::<m![H], m![V]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![A, H, V]>(&mut device.tdma)
}

#[device(chip = 1)]
pub fn fetch_slice_lift_axis_twice(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![A, H, V]>,
) -> (HbmTensor<bf16, Chip, m![A, H, V]>, HbmTensor<bf16, Chip, m![A, H, V]>) {
    let dm: DmTensor<bf16, Chip, Cluster, Slice, m![H, V]> = input.to_dm::<Cluster, Slice, m![H, V]>(&mut device.tdma);
    let output0: DmTensor<bf16, Chip, Cluster, m![A, H], m![V]> = device
        .main
        .begin(dm.view())
        .fetch::<m![H], m![V]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();
    let output1: DmTensor<bf16, Chip, Cluster, m![A, H], m![V]> = device
        .main
        .begin(dm.view())
        .fetch::<m![H], m![V]>()
        .fetch_slice_lift::<m![A, H], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();
    (
        output0.to_hbm::<m![A, H, V]>(&mut device.tdma),
        output1.to_hbm::<m![A, H, V]>(&mut device.tdma),
    )
}

/// Lifts the outer binary digit of `Q` from fetch time onto `Slice`.
#[device(chip = 1)]
pub fn fetch_slice_lift_digit(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![A, Q, V]>,
) -> HbmTensor<bf16, Chip, m![A, Q, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, Slice, m![Q, V]> = input.to_dm::<Cluster, Slice, m![Q, V]>(&mut device.tdma);

    let result: DmTensor<bf16, Chip, Cluster, m![A, Q / 2], m![Q % 2, V]> = device
        .main
        .begin(dm.view())
        .fetch::<m![Q], m![V]>()
        .fetch_slice_lift::<m![A, Q / 2], m![Q % 2]>()
        .collect::<m![Q % 2], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![A, Q, V]>(&mut device.tdma)
}

axes![Row = 64, G = 2];
type BroadcastSlice = m![2, Row, 2];
type LiftedBroadcasts = m![H, Row, G];

/// Lifts `H` and `G` onto two `Slice` broadcast slots.
#[device(chip = 1)]
pub fn fetch_slice_lift_broadcasts(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![Row, H, G, V]>,
) -> HbmTensor<bf16, Chip, m![H, Row, G, V]> {
    let dm: DmTensor<bf16, Chip, Cluster, BroadcastSlice, m![H, G, V]> =
        input.to_dm::<Cluster, BroadcastSlice, m![H, G, V]>(&mut device.tdma);

    let result: DmTensor<bf16, Chip, Cluster, LiftedBroadcasts, m![V]> = device
        .main
        .begin(dm.view())
        .fetch::<m![H, G], m![V]>()
        .fetch_slice_lift::<LiftedBroadcasts, m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![H, Row, G, V]>(&mut device.tdma)
}

type PaddedSlice = m![A, 1 # 2];

/// Lifts `H` from fetch time onto `Cluster`.
#[device(chip = 1)]
pub fn fetch_cluster_lift_axis(
    device: &mut Device,
    input: &HbmTensor<bf16, Chip, m![A, H, V]>,
) -> HbmTensor<bf16, Chip, m![H, A, V]> {
    let dm: DmTensor<bf16, Chip, m![2], PaddedSlice, m![H, V]> =
        input.to_dm::<m![2], PaddedSlice, m![H, V]>(&mut device.tdma);

    let result: DmTensor<bf16, Chip, m![H], PaddedSlice, m![V]> = device
        .main
        .begin(dm.view())
        .fetch::<m![H], m![V]>()
        .fetch_cluster_lift::<m![H], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![H, A, V]>(&mut device.tdma)
}

type FourChips = m![4];

/// Lifts `Q` from fetch time onto `Chip`.
#[device(chip = 4)]
pub fn fetch_chip_lift_axis(
    device: &mut Device,
    input: &HbmTensor<bf16, FourChips, m![A, Q, V]>,
) -> HbmTensor<bf16, m![Q], m![A, V]> {
    let dm: DmTensor<bf16, FourChips, Cluster, PaddedSlice, m![Q, V]> =
        input.to_dm::<Cluster, PaddedSlice, m![Q, V]>(&mut device.tdma);

    let result: DmTensor<bf16, m![Q], Cluster, PaddedSlice, m![V]> = device
        .main
        .begin(dm.view())
        .fetch::<m![Q], m![V]>()
        .fetch_chip_lift::<m![Q], m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![A, V]>(&mut device.tdma)
}

axes![C = 4, L = 2];
type ThreeLifted = m![A, H];

/// Lifts `C`, `L`, and `H` onto `Chip`, `Cluster`, and `Slice` in one read.
#[device(chip = 4)]
pub fn fetch_lift_every_dimension(
    device: &mut Device,
    input: &HbmTensor<bf16, m![4], m![A, C, L, H, V]>,
) -> HbmTensor<bf16, m![C], m![L, A, H, V]> {
    let dm: DmTensor<bf16, m![4], m![2], Slice, m![C, L, H, V]> =
        input.to_dm::<m![2], Slice, m![C, L, H, V]>(&mut device.tdma);

    let result: DmTensor<bf16, m![C], m![L], ThreeLifted, m![V]> = device
        .main
        .begin(dm.view())
        .fetch::<m![C, L, H], m![V]>()
        .fetch_chip_lift::<m![C], m![L, H]>()
        .fetch_cluster_lift::<m![L], m![H]>()
        .fetch_slice_lift::<ThreeLifted, m![1]>()
        .collect::<m![1], m![V]>()
        .commit_trim::<m![V]>()
        .commit();

    result.to_hbm::<m![L, A, H, V]>(&mut device.tdma)
}
