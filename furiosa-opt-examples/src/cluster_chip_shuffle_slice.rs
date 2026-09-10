#![expect(clippy::type_complexity)]

use furiosa_opt_std::prelude::*;

axes![A = 256, B = 4096];

/// Redistributes HBM chip slots by DMA.
#[device(chip = 4)]
pub fn hbm_chip_shuffle(
    device: &mut Device,
    hbm: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> {
    hbm.hbm_chip_shuffle(&mut device.tdma, &[1, 2, 3, 0])
}

#[device(chip = 4)]
pub fn chip_slice(
    device: &mut Device,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B / 2048, B % 512]> {
    let hbm_tensor = hbm_tensor.to_hbm::<_, m![B, A % 4, A / 16]>(&mut device.tdma);
    // `B / 512 % 4` leads `Element`: an asymmetric slice can only address the outermost axis, so the
    // axis to slice is put there at load time rather than reached for afterwards.
    let dm_tensor: DmTensor<
        i32,
        m![A / 4 % 4],
        m![A / 2 % 2],
        m![B % 16, B / 16 % 16],
        m![B / 512 % 4, B / 2048, B / 256 % 2, A % 2, A / 16],
    > = hbm_tensor.to_dm(&mut device.tdma);

    let sliced: DmTensor<i32, _, _, _, m![B / 2048, B / 256 % 2, A % 2, A / 16]> =
        dm_tensor.asymmetric_chip_slice::<m![B / 512 % 4], _>(&mut device.sub, &[3, 0, 1, 2]);

    sliced.to_hbm(&mut device.tdma)
}

#[device(chip = 4)]
pub fn cluster_slice(
    device: &mut Device,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B / 1024, B % 512]> {
    let hbm_tensor = hbm_tensor.to_hbm::<_, m![B, A % 4, A / 16]>(&mut device.tdma);
    // `B / 512 % 2` leads `Element`, for the reason given in `chip_slice`.
    let dm_tensor: DmTensor<
        i32,
        m![A / 4 % 4],
        m![A / 2 % 2],
        m![B % 16, B / 16 % 16],
        m![B / 512 % 2, B / 1024, B / 256 % 2, A % 2, A / 16],
    > = hbm_tensor.to_dm(&mut device.tdma);

    let sliced: DmTensor<i32, _, _, _, m![B / 1024, B / 256 % 2, A % 2, A / 16]> =
        dm_tensor.asymmetric_cluster_slice::<m![B / 512 % 2], _>(&mut device.sub, &[1, 0]);

    sliced.to_hbm(&mut device.tdma)
}

/// Selects one local axis position per cluster with Tensor DMA.
#[device(chip = 4)]
pub fn dma_cluster_slice(
    device: &mut Device,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B / 1024, B % 512]> {
    let hbm_tensor = hbm_tensor.to_hbm::<_, m![B, A % 4, A / 16]>(&mut device.tdma);
    let dm_tensor: DmTensor<
        i32,
        m![A / 4 % 4],
        m![A / 2 % 2],
        m![B % 16, B / 16 % 16],
        m![B / 512 % 2, B / 1024, B / 256 % 2, A % 2, A / 16],
    > = hbm_tensor.to_dm(&mut device.tdma);

    dm_tensor
        .cluster_slice::<m![B / 512 % 2], m![B / 1024, B / 256 % 2, A % 2, A / 16]>([1, 0])
        .to_dm::<m![B / 1024, B / 256 % 2, A % 2, A / 16]>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}

/// Fuses a chip permutation with a different local selection in each cluster.
#[device(chip = 4)]
pub fn chip_shuffle_cluster_slice(
    device: &mut Device,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B / 1024, B % 512]> {
    let hbm_tensor = hbm_tensor.to_hbm::<_, m![B, A % 4, A / 16]>(&mut device.tdma);
    let dm_tensor: DmTensor<
        i32,
        m![A / 4 % 4],
        m![A / 2 % 2],
        m![B % 16, B / 16 % 16],
        m![B / 512 % 2, B / 1024, B / 256 % 2, A % 2, A / 16],
    > = hbm_tensor.to_dm(&mut device.tdma);

    dm_tensor
        .chip_shuffle([1, 2, 3, 0])
        .cluster_slice::<m![B / 512 % 2], m![B / 1024, B / 256 % 2, A % 2, A / 16]>([1, 0])
        .to_dm::<m![B / 1024, B / 256 % 2, A % 2, A / 16]>(&mut device.tdma)
        .to_hbm(&mut device.tdma)
}

#[device(chip = 4)]
pub fn chip_shuffle(
    device: &mut Device,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> {
    let hbm_tensor = hbm_tensor.to_hbm::<_, m![B, A % 4, A / 16]>(&mut device.tdma);
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> =
        hbm_tensor.to_dm(&mut device.tdma);

    let shuffled = dm_tensor
        .chip_shuffle([1, 2, 3, 0])
        .to_dm::<m![B / 256, A % 2, A / 16]>(&mut device.tdma);

    shuffled.to_hbm(&mut device.tdma)
}

#[device(chip = 4)]
pub fn cluster_swap(
    device: &mut Device,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> {
    let hbm_tensor = hbm_tensor.to_hbm::<_, m![B, A % 4, A / 16]>(&mut device.tdma);
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A / 2 % 2], m![B % 16, B / 16 % 16], m![B / 256, A % 2, A / 16]> =
        hbm_tensor.to_dm(&mut device.tdma);

    let shuffled = dm_tensor
        .cluster_swap()
        .to_dm::<m![B / 256, A % 2, A / 16]>(&mut device.tdma);

    shuffled.to_hbm(&mut device.tdma)
}

/// A swap whose in-slice axis carries a padded tail, driving the padded-extent shape path.
#[device(chip = 4)]
pub fn cluster_swap_padded(
    device: &mut Device,
    hbm_tensor: &HbmTensor<i32, m![A / 4 % 4], m![A % 2, B / 1024]>,
) -> HbmTensor<i32, m![A / 4 % 4], m![A % 2, B / 1024]> {
    let dm_tensor: DmTensor<i32, m![A / 4 % 4], m![A % 2], m![1 # 256], m![B / 1024 # 6]> =
        hbm_tensor.to_dm(&mut device.tdma);
    let shuffled = dm_tensor
        .view()
        .cluster_swap()
        .to_dm::<m![B / 1024 # 6]>(&mut device.tdma);
    shuffled.to_hbm(&mut device.tdma)
}

/// Exercises read-only outer padding while keeping the live inner run 8-byte aligned.
#[device(chip = 4)]
pub fn cluster_swap_read_only_padded(
    device: &mut Device,
    hbm_tensor: &HbmTensor<i32, m![1 # 4], m![A % 2, B / 1024, B % 2]>,
) -> HbmTensor<i32, m![1 # 4], m![A % 2, B / 1024, B % 2]> {
    let dm_tensor: DmTensor<i32, m![1 # 4], m![A % 2], m![1 # 256], m![B / 1024 #{!} 6, B % 2]> =
        hbm_tensor.to_dm(&mut device.tdma);
    let shuffled = dm_tensor
        .view()
        .cluster_swap()
        .to_dm::<m![B / 1024 #{!} 6, B % 2]>(&mut device.tdma);
    shuffled.to_hbm(&mut device.tdma)
}
