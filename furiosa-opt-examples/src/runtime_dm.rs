//! Kernels borrowing runtime-owned DM tensors: `&Dm` and `&mut Dm` are the only DM parameter
//! shapes the device signature accepts, so this pins that both lower.

use furiosa_opt_std::prelude::*;

axes![S = 64, E = 8, P = 16448];

type Chip = m![1];
type Dm = DmTensor<i32, Chip, m![1], m![S], m![E]>;
type Hbm = HbmTensor<i32, Chip, m![S, E]>;

#[device(chip = 1, pe = 1)]
pub fn stage(ctx: &mut Device, input: &Hbm, output: &mut Dm) {
    input.view().to_dm_view(&mut ctx.tdma, output.view_mut());
}

#[device(chip = 1, pe = 1)]
pub fn readback(ctx: &mut Device, input: &Dm) -> Hbm {
    input.to_hbm(&mut ctx.tdma)
}

#[device(chip = 1, pe = 1)]
pub fn increment(ctx: &mut Device, input: &Dm, output: &mut Dm) {
    ctx.main
        .begin(input.view())
        .fetch::<m![1], m![E]>()
        .fetch_cast::<i32>()
        .collect::<m![1], m![E]>()
        .vector_init()
        .vector_intra_slice_tag(TagMode::Zero)
        .vector_fxp(FxpBinaryOp::AddFxp, 1)
        .vector_final()
        .commit_trim::<m![E]>()
        .commit_view(output.view_mut());
}

type PageDm = DmTensor<i32, Chip, m![1], m![S], m![P]>;
type PageHbm = HbmTensor<i32, Chip, m![S, P]>;

#[device(chip = 1, pe = 1)]
pub fn stage_pages(ctx: &mut Device, input: &PageHbm, output: &mut PageDm) {
    input.view().to_dm_view(&mut ctx.tdma, output.view_mut());
}

#[device(chip = 1, pe = 1)]
pub fn read_pages(ctx: &mut Device, input: &PageDm) -> PageHbm {
    input.to_hbm(&mut ctx.tdma)
}

#[device(chip = 1, pe = 1)]
pub fn loop_read(ctx: &mut Device, input: &Dm, output: &mut Hbm) {
    for _ in 0..2 {
        input.view().to_hbm_view(&mut ctx.tdma, output.view_mut());
    }
}

#[device(chip = 1, pe = 1)]
pub fn conditional_write(ctx: &mut Device, input: &Hbm, output: &mut Dm) {
    for i in 0..2 {
        if i == 1 {
            input.view().to_dm_view(&mut ctx.tdma, output.view_mut());
        }
    }
}

#[device(chip = 1, pe = 1)]
pub fn borrowed_dm_to_hbm(
    ctx: &mut Device,
    input: &Dm,
    output: &mut Dm,
) -> (HbmTensor<i32, Chip, m![S, E]>, HbmTensor<i32, Chip, m![S, E]>) {
    (input.to_hbm(&mut ctx.tdma), output.to_hbm(&mut ctx.tdma))
}
