//! Rejection fixtures for invalid SPM index placement.

use furiosa_opt_std::prelude::*;

pub use crate::scatter_gather::{D, G, K, SC};

axes![SI = 2];

type Chip = m![1];
type PaddedCluster = m![1 # 2];
type BroadcastCluster = m![2];
type PlacedPE = m![G / 128];

/// Rejects an SPM indirect index partitioned across PEs instead of kept on PE 0.
#[device(chip = 1)]
pub fn invalid_gather_index_across_pes(
    device: &mut Device,
    table: &HbmTensor<bf16, Chip, m![K, D]>,
    index: &HbmTensor<i32, Chip, m![G]>,
) -> HbmTensor<bf16, Chip, m![G, D]> {
    let index_dm: DmTensor<i32, Chip, BroadcastCluster, m![G / 2], m![G % 2]> = index.to_dm(&mut device.tdma);
    let index_spm: SpmTensor<i32, Chip, BroadcastCluster, PlacedPE, m![G % 128]> = index_dm.to_spm(&mut device.tdma);
    let values_dm: DmTensor<bf16, Chip, PaddedCluster, m![G / 2], m![G % 2, D]> = table
        .gather::<m![K], m![D]>()
        .by_positions::<m![G % 128]>(&index_spm)
        .to_dm::<PaddedCluster, m![G / 2], m![G % 2, D]>(&mut device.tdma);

    values_dm.to_hbm(&mut device.tdma)
}

/// Rejects an index distributed over chips of its own, which names rows the table holds on
/// another chip. There is no inter-chip indirect DMA to fetch them.
#[device(chip = 2)]
pub fn invalid_gather_index_on_other_chips(
    device: &mut Device,
    table: &HbmTensor<bf16, m![SC], m![K, D]>,
    index: &HbmTensor<i32, m![SI], m![G]>,
) -> HbmTensor<bf16, m![SC], m![G, D]> {
    let values: DmTensor<bf16, m![SC], PaddedCluster, m![G / 2], m![G % 2, D]> = table
        .gather::<m![K], m![D]>()
        .by_byte_offsets::<m![G]>(index)
        .to_dm(&mut device.tdma);

    values.to_hbm(&mut device.tdma)
}
