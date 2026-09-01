#![expect(clippy::type_complexity)]

use furiosa_opt_std::prelude::*;

axes![A = 256, B = 4096];

#[device(chip = 4)]
pub fn chip_shuffle(
    ctx: &mut Context,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> {
    let hbm_tensor = hbm_tensor.to_hbm::<{ Dma::Tensor }, m![B, A % 4, A / 16]>(&mut ctx.tdma);
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> =
        hbm_tensor.to_dm(&mut ctx.tdma);

    let shuffled: DmTensor<i32, _, _, _, _> = dm_tensor.view().dm_chip_shuffle::<4>(&mut ctx.tdma, &[1, 2, 3, 0]);

    shuffled.to_hbm(&mut ctx.tdma)
}

#[device(chip = 4)]
pub fn cluster_swap(
    ctx: &mut Context,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> {
    let hbm_tensor = hbm_tensor.to_hbm::<{ Dma::Tensor }, m![B, A % 4, A / 16]>(&mut ctx.tdma);
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> =
        hbm_tensor.to_dm(&mut ctx.tdma);
    let shuffled = dm_tensor.view().dm_cluster_swap(&mut ctx.tdma);
    shuffled.to_hbm(&mut ctx.tdma)
}

/// A swap whose in-slice axis carries a padded tail, driving the padded-extent shape path.
#[device(chip = 4)]
pub fn cluster_swap_padded(
    ctx: &mut Context,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A % 2, B / 1024]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A % 2, B / 1024]> {
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A % 2], m![1 # 256], m![B / 1024 # 6]> =
        hbm_tensor.to_dm(&mut ctx.tdma);
    let shuffled = dm_tensor.view().dm_cluster_swap(&mut ctx.tdma);
    shuffled.to_hbm(&mut ctx.tdma)
}

#[device(chip = 4)]
pub fn cluster_swap_read_only_padded(
    ctx: &mut Context,
    hbm_tensor: &HbmTensor<i32, m![1 # 4], m![A % 2, B / 1024]>,
) -> HbmTensor<i32, m![1 # 4], m![A % 2, B / 1024]> {
    let dm_tensor: DmTensor<i32, m![1 # 4], m![A % 2], m![1 # 256], m![B / 1024 #{!} 6]> =
        hbm_tensor.to_dm(&mut ctx.tdma);
    let shuffled = dm_tensor.view().dm_cluster_swap(&mut ctx.tdma);
    shuffled.to_hbm(&mut ctx.tdma)
}
