//! Rejection fixtures for unsupported Transpose Engine packing modes.

use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1];
axes![P = 64, A = 16, C = 32];

/// Attempts doubled input packing for 4-bit elements, which the hardware does not support.
#[device(chip = 1, pe = 1)]
pub fn transpose_i4_doubled_tu(
    device: &mut Device,
    input: &HbmTensor<i4, Chip, m![P, A, C]>,
) -> HbmTensor<i4, Chip, m![P, C, A]> {
    let input_dm: DmTensor<i4, Chip, Cluster, m![P], m![A, C]> = input.to_dm(&mut device.tdma);

    let output_dm: DmTensor<i4, Chip, Cluster, m![P], m![C, A]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![A], m![C]>()
        .collect::<m![A], m![C # 64]>()
        .transpose::<m![C], m![A # 64]>()
        .commit_trim::<m![A]>()
        .commit();

    output_dm.to_hbm(&mut device.tdma)
}
