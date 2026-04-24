use furiosa_visa_std::prelude::*;

axes![A = 512, B = 32];

/// Device function that transposes a tensor from shape [A, B] to [B, A].
#[device(chip = 1)]
pub fn tile_simple(ctx: &mut Context, input: HbmTensorView<'_, i8, m![1], m![A, B]>) -> HbmTensor<i8, m![1], m![B, A]> {
    let mut output = unsafe { HbmTensor::<i8, m![1], m![B, A]>::from_addr(0x3000) };
    for b in 0..32 {
        // TODO: replace with <m![B]>::SIZE
        let input_slice = input.tile::<m![B], 1, m![A, 1 # 32]>(b);
        let output_slice = output.view_mut().tile::<m![B], 1, m![1 # 32, A]>(b);
        input_slice.to_hbm_view(&mut ctx.tdma, output_slice);
    }
    output
}
