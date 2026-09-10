//! Kernels that exercise `#[unroll]` lowering.

use furiosa_opt_std::prelude::*;

axes![A = 512, B = 32];

/// One row of the transpose: DMA row `b` of `input` into column `b` of `output`.
pub(crate) fn transpose_row(
    device: &mut Device,
    input: &HbmTensorView<'_, i8, m![1], m![A, B]>,
    output: &mut HbmTensor<i8, m![1], m![B, A]>,
    b: usize,
) {
    transpose_row_into(device, input, output, b, b)
}

fn transpose_row_into(
    device: &mut Device,
    input: &HbmTensorView<'_, i8, m![1], m![A, B]>,
    output: &mut HbmTensor<i8, m![1], m![B, A]>,
    from: usize,
    into: usize,
) {
    let input_slice = input.tile::<m![B], 1, m![A, 1 # 32]>(from);
    let output_slice = output.view_mut().tile::<m![B], 1, m![1 #{!} 32, A]>(into);
    input_slice.to_hbm_view(&mut device.tdma, output_slice);
}

fn transpose_four_rows_in_helper(
    device: &mut Device,
    input: &HbmTensorView<'_, i8, m![1], m![A, B]>,
    output: &mut HbmTensor<i8, m![1], m![B, A]>,
) {
    #[unroll]
    for b in 0..4 {
        transpose_row(device, input, output, b);
    }
}

/// Transposes four rows while retaining the loop.
#[device(chip = 1)]
pub fn transpose_four_rows_rolled(
    device: &mut Device,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    for b in 0..4 {
        transpose_row(device, &input, &mut output, b);
    }
    output
}

/// Transposes four rows with an unrolled loop.
#[device(chip = 1)]
pub fn transpose_four_rows_unrolled(
    device: &mut Device,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    #[unroll]
    for b in 0..4 {
        transpose_row(device, &input, &mut output, b);
    }
    output
}

/// Transposes four rows in an unrolled loop inside a reachable helper.
#[device(chip = 1)]
pub fn transpose_four_rows_helper_unrolled(
    device: &mut Device,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    transpose_four_rows_in_helper(device, &input, &mut output);
    output
}

/// Runs a nested loop with both levels unrolled.
#[device(chip = 1)]
pub fn nested_both_unrolled(
    device: &mut Device,
    input: HbmTensorView<'_, i8, m![1], m![A, B]>,
) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = HbmTensor::<i8, m![1], m![B, A]>::new();
    #[unroll]
    for i in 0..2 {
        #[unroll]
        for _j in 0..2 {
            transpose_row(device, &input, &mut output, i);
        }
    }
    output
}
