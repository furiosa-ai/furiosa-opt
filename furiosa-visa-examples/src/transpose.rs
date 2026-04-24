//! Transpose examples.

use furiosa_visa_std::prelude::*;

type Chip = m![1];
axes![A = 8, B = 16, C = 32];

/// Device function that transposes a tensor from shape [A, B, C] to [C, A, B].
/// it is divided into two steps for demonstration purposes.
#[device(chip = 1)]
pub fn transpose_simple(
    ctx: &mut Context,
    input: &HbmTensor<f32, Chip, m![A, B, C]>,
) -> HbmTensor<f32, Chip, m![C, A, B]> {
    // transpose: [A, B, C] -> [A, C, B]
    let intermediate: HbmTensor<f32, Chip, m![A, C, B]> = input.to_hbm(&mut ctx.tdma, 0);

    // transpose: [A, C, B] -> [C, A, B]
    intermediate.to_hbm(&mut ctx.tdma, 0x1000)
}
