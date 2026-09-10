//! Rejection fixtures for DM-producing slice shape and alignment constraints.

use furiosa_opt_std::prelude::*;

pub use crate::chip_reduce::{A, B, C, D};

type Chip = m![A];
type Cluster = m![1 # 2];
type Slice = m![1 # 256];
type Full = m![B, C, D];
type SlicedFull = m![1 # 4, C, D];
type InnermostSliced = m![B, C];

/// Keeps the sliced axis in `Element2` instead of removing it.
#[device(chip = 4)]
pub fn invalid_slice_output(device: &mut Device, hbm: &HbmTensor<i32, Chip, Full>) -> HbmTensor<i32, Chip, Full> {
    let full: DmTensor<i32, Chip, Cluster, Slice, Full> = hbm.to_dm(&mut device.tdma);
    full.view()
        .chip_shuffle([0, 1, 2, 3])
        .chip_slice::<m![B], SlicedFull>([0, 1, 2, 3])
        .to_dm::<SlicedFull>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}

/// Slices an i32 axis whose one-element stride is 4 bytes rather than the required 8 bytes.
#[device(chip = 4)]
pub fn unaligned_slice_axis_stride(
    device: &mut Device,
    hbm: &HbmTensor<i32, Chip, Full>,
) -> HbmTensor<i32, Chip, m![B, C]> {
    let full: DmTensor<i32, Chip, Cluster, Slice, Full> = hbm.to_dm(&mut device.tdma);
    full.view()
        .chip_shuffle([0, 1, 2, 3])
        .chip_slice::<m![D], InnermostSliced>([0, 1, 2, 3])
        .to_dm::<InnermostSliced>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}

/// Uses an inner axis with ParallelCopy, which can only narrow the leading axis.
#[device(chip = 4)]
pub fn asymmetric_inner_axis(
    device: &mut Device,
    hbm: &HbmTensor<i32, Chip, m![C, B, D]>,
) -> HbmTensor<i32, Chip, m![C, D]> {
    let full: DmTensor<i32, Chip, Cluster, Slice, m![C, B, D]> = hbm.to_dm(&mut device.tdma);
    full.asymmetric_chip_slice::<m![B], m![C, D]>(&mut device.sub, &[0, 1, 2, 3])
        .to_hbm(&mut device.tdma)
}

/// Leaves must-not-write padding below the axis narrowed by ParallelCopy.
#[device(chip = 4)]
pub fn asymmetric_bottom_hole_tail(
    device: &mut Device,
    hbm: &HbmTensor<i32, Chip, m![B, 1 #{!} 2, C, D]>,
) -> HbmTensor<i32, Chip, m![1 #{!} 2, C, D]> {
    let full: DmTensor<i32, Chip, Cluster, Slice, m![B, 1 #{!} 2, C, D]> = hbm.to_dm(&mut device.tdma);
    full.asymmetric_chip_slice::<m![B], m![1 #{!} 2, C, D]>(&mut device.sub, &[0, 1, 2, 3])
        .to_hbm(&mut device.tdma)
}

/// Selects a position outside `AxisToSlice` for the last chip.
#[device(chip = 4)]
pub fn asymmetric_index_out_of_range(
    device: &mut Device,
    hbm: &HbmTensor<i32, Chip, Full>,
) -> HbmTensor<i32, Chip, m![C, D]> {
    let full: DmTensor<i32, Chip, Cluster, Slice, Full> = hbm.to_dm(&mut device.tdma);
    full.asymmetric_chip_slice::<m![B], m![C, D]>(&mut device.sub, &[0, 1, 2, 4])
        .to_hbm(&mut device.tdma)
}

/// Routes each live cluster to a padded source cluster.
#[device(chip = 4)]
pub fn padded_cluster_swap_source(device: &mut Device, hbm: &HbmTensor<i32, Chip, Full>) -> HbmTensor<i32, Chip, Full> {
    let full: DmTensor<i32, Chip, Cluster, Slice, Full> = hbm.to_dm(&mut device.tdma);
    full.view()
        .cluster_swap()
        .to_dm::<Full>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}

/// Uses runtime values as chip-shuffle source indices.
#[device(chip = 4)]
pub fn dynamic_chip_shuffle_indices(
    device: &mut Device,
    hbm: &HbmTensor<i32, Chip, Full>,
) -> HbmTensor<i32, Chip, Full> {
    let full: DmTensor<i32, Chip, Cluster, Slice, Full> = hbm.to_dm(&mut device.tdma);
    let mut result = DmTensor::<i32, Chip, Cluster, Slice, Full>::new();
    for i in 0..2 {
        let source = if i == 0 { 0 } else { 1 };
        result = full
            .view()
            .chip_shuffle([source, 1, 2, 3])
            .to_dm::<Full>(&mut device.tdma);
    }
    result.to_hbm(&mut device.tdma)
}

/// Uses runtime values as chip-slice indices.
#[device(chip = 4)]
pub fn dynamic_chip_slice_indices(
    device: &mut Device,
    hbm: &HbmTensor<i32, Chip, Full>,
) -> HbmTensor<i32, Chip, m![C, D]> {
    let full: DmTensor<i32, Chip, Cluster, Slice, Full> = hbm.to_dm(&mut device.tdma);
    let mut result = DmTensor::<i32, Chip, Cluster, Slice, m![C, D]>::new();
    for i in 0..2 {
        let index = if i == 0 { 0 } else { 1 };
        result = full
            .view()
            .chip_slice::<m![B], m![C, D]>([index, 1, 2, 3])
            .to_dm::<m![C, D]>(&mut device.tdma);
    }
    result.to_hbm(&mut device.tdma)
}

/// Uses runtime values as cluster-slice indices.
#[device(chip = 4)]
pub fn dynamic_cluster_slice_indices(
    device: &mut Device,
    hbm: &HbmTensor<i32, Chip, Full>,
) -> HbmTensor<i32, Chip, m![C, D]> {
    let full: DmTensor<i32, Chip, Cluster, Slice, Full> = hbm.to_dm(&mut device.tdma);
    let mut result = DmTensor::<i32, Chip, Cluster, Slice, m![C, D]>::new();
    for i in 0..2 {
        let index = if i == 0 { 0 } else { 1 };
        result = full
            .view()
            .cluster_slice::<m![B], m![C, D]>([index, 1])
            .to_dm::<m![C, D]>(&mut device.tdma);
    }
    result.to_hbm(&mut device.tdma)
}

axes![PaddedSliceAxis = 4, Row = 8, Col = 8];

type PaddedSliceChip = m![PaddedSliceAxis];
type PaddedSliceCluster = m![1 # 2];
type PaddedSliceStream = m![1 # 256];
type PaddedSliceFull = m![PaddedSliceAxis, Row, Col];

/// Slices a padded axis at a padding position: any in-bounds index is a valid selection, live or
/// not, but the sequencer currently cannot pair the resulting duplicate-tagged source and
/// destination streams.
#[device(chip = 4)]
pub fn padded_slice_axis_index(
    device: &mut Device,
    hbm: &HbmTensor<i32, PaddedSliceChip, PaddedSliceFull>,
) -> HbmTensor<i32, PaddedSliceChip, m![Row, Col]> {
    let full: DmTensor<i32, PaddedSliceChip, PaddedSliceCluster, PaddedSliceStream, m![PaddedSliceAxis # 8, Row, Col]> =
        hbm.to_dm(&mut device.tdma);
    full.view()
        .chip_shuffle([0, 1, 2, 3])
        .chip_slice::<m![PaddedSliceAxis # 8], m![Row, Col]>([0, 1, 2, 6])
        .to_dm::<m![Row, Col]>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}
