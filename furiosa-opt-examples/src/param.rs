//! Tuple- and struct-typed `#[device]` parameter examples.
//!
//! Each passthrough returns its input tensor unchanged (HBM -> SRAM -> HBM), so the
//! tests can confirm a tuple and a struct parameter both reach the kernel by checking
//! the data survives the round-trip.

use furiosa_opt_std::prelude::*;

axes![A = 4096, B = 8];

type Chip = m![1];
type Cluster = m![1 # 2];

#[derive(DeviceSend)]
pub struct Inputs<'a> {
    pub x: &'a HbmTensor<i8, Chip, m![A, B]>,
}

#[device(chip = 1)]
pub fn tuple_passthrough(
    ctx: &mut Context,
    inputs: (&HbmTensor<i8, Chip, m![A, B]>,),
) -> HbmTensor<i8, Chip, m![A, B]> {
    inputs
        .0
        .to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut ctx.tdma, 0)
        .to_hbm(&mut ctx.tdma, 0x3000)
}

#[device(chip = 1)]
pub fn struct_passthrough(ctx: &mut Context, inputs: Inputs<'_>) -> HbmTensor<i8, Chip, m![A, B]> {
    inputs
        .x
        .to_dm::<Cluster, m![A / 16], m![A / 8 % 2, A % 8, B]>(&mut ctx.tdma, 0)
        .to_hbm(&mut ctx.tdma, 0x3000)
}
