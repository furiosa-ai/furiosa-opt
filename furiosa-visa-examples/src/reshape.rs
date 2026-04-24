#![expect(clippy::type_complexity)]

use furiosa_visa_std::prelude::*;

axes![A = 256, B = 4096, C = 4, D = 2, E = 16, F = 16, G = 16, H = 2, I = 16,];

#[device(chip = 4)]
pub fn reshape(
    ctx: &mut Context,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![C], m![D, E, F, G, H, I]> {
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> =
        hbm_tensor.to_dm(&mut ctx.tdma, 0);

    let reshaped: DmTensor<i32, m![C], m![D], m![E, F], m![G, H, I]> = unsafe { dm_tensor.reshape() };

    reshaped.to_hbm(&mut ctx.tdma, 0x2000)
}

// Note: This test uses different shape definitions, so it's in a separate module
pub mod different_axes {
    use furiosa_visa_std::prelude::*;

    axes![A = 256, B = 4096, C = 4, D = 2, E = 256, F = 512,];

    #[device(chip = 4)]
    pub fn reshape_different_num_axes(
        ctx: &mut Context,
        hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
    ) -> HbmTensor<i32, m![C], m![D, E, F]> {
        let dm_tensor: DmTensor<
            i32,
            m![A / 4 % 4],
            m![A / 2 % 2],
            m![B % 16, B / 16 % 16],
            m![B / 256, A % 2, A / 16],
        > = hbm_tensor.to_dm(&mut ctx.tdma, 0);

        let reshaped: DmTensor<i32, m![C], m![D], m![E], m![F]> = unsafe { dm_tensor.reshape() };

        reshaped.to_hbm(&mut ctx.tdma, 0x2000)
    }
}
