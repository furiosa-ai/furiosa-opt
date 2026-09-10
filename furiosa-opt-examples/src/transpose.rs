//! Transpose examples.

use furiosa_opt_std::prelude::*;

type Chip = m![1];
type Cluster = m![1];
axes![P = 64, A = 8, B = 16, C = 32, D = 4];

/// Device function that transposes a tensor from shape [A, B, C] to [C, A, B].
/// it is divided into two steps for demonstration purposes.
#[device(chip = 1)]
pub fn transpose_simple(
    device: &mut Device,
    input: &HbmTensor<f32, Chip, m![A, B, C]>,
) -> HbmTensor<f32, Chip, m![C, A, B]> {
    // transpose: [A, B, C] -> [A, C, B]
    let intermediate: HbmTensor<f32, Chip, m![A, C, B]> = input.to_hbm(&mut device.tdma);

    // transpose: [A, C, B] -> [C, A, B]
    intermediate.to_hbm(&mut device.tdma)
}

/// Transposes an i8 tensor from `[P, A, B]` to `[P, B, A]` with the TU
/// transpose engine.
///
/// The tensor is staged in SRAM as `[P=64] [A=8, B=16]`, then streamed as:
///
/// ```text
/// collect:   [A=8]  [B=16 # 32]
/// transpose: [B=16] [A=8  # 32]
/// commit:    [B=16] [A=8]
/// ```
#[device(chip = 1, pe = 1)]
pub fn transpose_i8_tu(
    device: &mut Device,
    input: &HbmTensor<i8, Chip, m![P, A, B]>,
) -> HbmTensor<i8, Chip, m![P, B, A]> {
    let input_dm: DmTensor<i8, Chip, Cluster, m![P], m![A, B]> = input.to_dm(&mut device.tdma);

    let output_dm: DmTensor<i8, Chip, Cluster, m![P], m![B, A]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![A], m![B]>()
        .collect::<m![A], m![B # 32]>()
        .transpose::<m![B], m![A # 32]>()
        .commit_trim::<m![A]>()
        .commit();

    output_dm.to_hbm(&mut device.tdma)
}

/// Transposes an i16 tensor with doubled input packing.
#[device(chip = 1, pe = 1)]
pub fn transpose_i16_tu(
    device: &mut Device,
    input: &HbmTensor<i16, Chip, m![P, D, B]>,
) -> HbmTensor<i16, Chip, m![P, B, D]> {
    let input_dm: DmTensor<i16, Chip, Cluster, m![P], m![D, B]> = input.to_dm(&mut device.tdma);

    let output_dm: DmTensor<i16, Chip, Cluster, m![P], m![B, D]> = device
        .main
        .begin(input_dm.view())
        .fetch::<m![D], m![B]>()
        .collect::<m![D], m![B]>()
        .transpose::<m![B], m![D # 16]>()
        .commit_trim::<m![D]>()
        .commit();

    output_dm.to_hbm(&mut device.tdma)
}
