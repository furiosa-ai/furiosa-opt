#![expect(clippy::type_complexity)]

use furiosa_visa_std::prelude::*;

axes![A = 256, B = 4096];

#[device(chip = 4)]
pub fn chip_slice(
    ctx: &mut Context,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B / 2048, B % 512]> {
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> =
        hbm_tensor.to_dm(&mut ctx.tdma, 0);

    let sliced: DmTensor<i32, _, _, _, m![B / 2048, B / 256 % 2, A % 2, A / 16]> = ctx
        .sub
        .parallel_copy_chip_slice::<4, m![B / 512 % 4], m![B / 2048, 1 # 4, B / 256 % 2, A % 2, A / 16], _, _, _, _, _, _>(
            dm_tensor.view(),
            &[0, 1, 2, 3],
        );

    sliced.to_hbm(&mut ctx.tdma, 0x2000)
}

#[device(chip = 4)]
pub fn cluster_slice(
    ctx: &mut Context,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B / 1024, B % 512]> {
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> =
        hbm_tensor.to_dm(&mut ctx.tdma, 0);

    let sliced: DmTensor<i32, _, _, _, m![B / 1024, B / 256 % 2, A % 2, A / 16]> = ctx
        .sub
        .parallel_copy_cluster_slice::<2, m![B / 512 % 2], m![B / 1024, 1 # 2, B / 256 % 2, A % 2, A / 16], _, _, _, _, _, _>(
            dm_tensor.view(),
            &[0, 1],
        );

    sliced.to_hbm(&mut ctx.tdma, 0x2000)
}

#[device(chip = 4)]
pub fn chip_shuffle(
    ctx: &mut Context,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> {
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> =
        hbm_tensor.to_dm(&mut ctx.tdma, 0);

    let shuffled: DmTensor<i32, _, _, _, _> = ctx
        .tdma
        .dm_chip_shuffle::<_, 4, _, _, _, _>(dm_tensor.view(), &[1, 2, 3, 0]);

    shuffled.to_hbm(&mut ctx.tdma, 0x2000)
}

#[device(chip = 4)]
pub fn cluster_shuffle(
    ctx: &mut Context,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> {
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> =
        hbm_tensor.to_dm(&mut ctx.tdma, 0);

    // Shuffle pattern [1, 0]: cluster 1->0, cluster 0->1
    let shuffled: DmTensor<i32, _, _, _, _> = ctx
        .tdma
        .dm_cluster_shuffle::<_, 2, _, _, _, _>(dm_tensor.view(), &[1, 0]);

    shuffled.to_hbm(&mut ctx.tdma, 0x2000)
}
