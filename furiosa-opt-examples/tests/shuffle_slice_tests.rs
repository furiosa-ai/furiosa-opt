#![expect(clippy::type_complexity)]

use furiosa_opt_examples::chip_reduce::{
    redistribute_element_tile, shuffle_slice_identity, shuffle_slice_middle, shuffle_slice_middle_place,
    shuffle_slice_noncontiguous_axes, shuffle_slice_padded_tail,
};
use furiosa_opt_examples::cluster_chip_shuffle_slice::{
    chip_shuffle, chip_shuffle_cluster_slice, chip_slice, cluster_slice, cluster_swap, cluster_swap_padded,
    cluster_swap_read_only_padded, dma_cluster_slice, hbm_chip_shuffle,
};
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_chip_shuffle() {
    use furiosa_opt_examples::cluster_chip_shuffle_slice::{A, B};

    let mut device = Device::new(chip_shuffle.topology()).unwrap();

    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..A::SIZE * B::SIZE).map(|value| value as i32).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut device.pdma)
            .await
            .unwrap();

    let output = launch(chip_shuffle, (&mut device, &hbm_tensor)).await.unwrap();

    assert_eq!(
        output.to_host::<m![A, B]>(&mut device.pdma).await.unwrap().into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(
            (0i32..A::SIZE as i32 * B::SIZE as i32)
                .map(|x| {
                    let index_a_16 = x / B::SIZE as i32 / 16;
                    let index_a_4 = ((x / B::SIZE as i32) % 16) / 4;
                    let index_a_1 = x / B::SIZE as i32 % 4;
                    let index_b = x % B::SIZE as i32;

                    let out_index_a_4 = [1, 2, 3, 0][index_a_4 as usize];

                    index_a_16 * B::SIZE as i32 * 16
                        + out_index_a_4 * B::SIZE as i32 * 4
                        + index_a_1 * B::SIZE as i32
                        + index_b
                })
                .collect::<Vec<_>>()
        ),
    );
}

#[tokio::test]
async fn test_hbm_chip_shuffle() {
    use furiosa_opt_examples::cluster_chip_shuffle_slice::{A, B};

    let mut device = Device::new(hbm_chip_shuffle.topology()).unwrap();
    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..A::SIZE * B::SIZE).map(|value| value as i32).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut device.pdma)
            .await
            .unwrap();
    let output = launch(hbm_chip_shuffle, (&mut device, &hbm_tensor)).await.unwrap();

    assert_eq!(
        output.to_host::<m![A, B]>(&mut device.pdma).await.unwrap().into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(
            (0i32..A::SIZE as i32 * B::SIZE as i32)
                .map(|x| {
                    let index_a_16 = x / B::SIZE as i32 / 16;
                    let target_chip = ((x / B::SIZE as i32) % 16) / 4;
                    let index_a_1 = x / B::SIZE as i32 % 4;
                    let index_b = x % B::SIZE as i32;
                    let source_chip = [1, 2, 3, 0][target_chip as usize];

                    index_a_16 * B::SIZE as i32 * 16
                        + source_chip * B::SIZE as i32 * 4
                        + index_a_1 * B::SIZE as i32
                        + index_b
                })
                .collect::<Vec<_>>(),
        ),
    );
}

#[tokio::test]
async fn test_chip_slice() {
    use furiosa_opt_examples::cluster_chip_shuffle_slice::{A, B};

    let mut device = Device::new(chip_slice.topology()).unwrap();

    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..A::SIZE * B::SIZE).map(|value| value as i32).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut device.pdma)
            .await
            .unwrap();

    let output = launch(chip_slice, (&mut device, &hbm_tensor)).await.unwrap();

    assert_eq!(
        output
            .to_host::<m![A, B % 512, B / 2048]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A, B % 512, B / 2048], CurrentBackend>::from_vec(
            (0..A::SIZE * (B::SIZE / 4))
                .map(|x| {
                    let index_a = x / (B::SIZE / 4);
                    let index_b_1 = (x % (B::SIZE / 4)) / 2;
                    let index_b_2048 = x % 2;

                    let index_a_4 = (index_a % 16) / 4;

                    let out_index_a = index_a;

                    let index_b_512 = [3, 0, 1, 2][index_a_4];
                    let out_index_b = index_b_2048 * (B::SIZE / 2) + index_b_512 * (B::SIZE / 8) + index_b_1;

                    (out_index_a * B::SIZE + out_index_b) as i32
                })
                .collect::<Vec<_>>()
        ),
    );
}

#[tokio::test]
async fn test_cluster_slice() {
    use furiosa_opt_examples::cluster_chip_shuffle_slice::{A, B};

    let mut device = Device::new(cluster_slice.topology()).unwrap();

    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..A::SIZE * B::SIZE).map(|value| value as i32).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut device.pdma)
            .await
            .unwrap();

    let output = launch(cluster_slice, (&mut device, &hbm_tensor)).await.unwrap();

    assert_eq!(
        output
            .to_host::<m![A, B % 512, B / 1024]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A, B % 512, B / 1024], CurrentBackend>::from_vec(cluster_slice_expected()),
    );
}

#[tokio::test]
async fn test_dma_cluster_slice() {
    use furiosa_opt_examples::cluster_chip_shuffle_slice::{A, B};

    let mut device = Device::new(dma_cluster_slice.topology()).unwrap();
    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..A::SIZE * B::SIZE).map(|value| value as i32).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut device.pdma)
            .await
            .unwrap();

    let output = launch(dma_cluster_slice, (&mut device, &hbm_tensor)).await.unwrap();

    assert_eq!(
        output
            .to_host::<m![A, B % 512, B / 1024]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A, B % 512, B / 1024], CurrentBackend>::from_vec(cluster_slice_expected()),
    );
}

fn cluster_slice_expected() -> Vec<i32> {
    use furiosa_opt_examples::cluster_chip_shuffle_slice::{A, B};

    (0..A::SIZE * (B::SIZE / 2))
        .map(|x| {
            let index_a = x / (B::SIZE / 2);
            let index_b_1 = (x % (B::SIZE / 2)) / 4;
            let index_b_1024 = x % 4;
            let index_a_2 = (index_a % 4) / 2;
            let index_b_512 = [1, 0][index_a_2];
            let out_index_b = index_b_1024 * (B::SIZE / 4) + index_b_512 * (B::SIZE / 8) + index_b_1;

            (index_a * B::SIZE + out_index_b) as i32
        })
        .collect()
}

#[tokio::test]
async fn test_chip_shuffle_cluster_slice() {
    use furiosa_opt_examples::cluster_chip_shuffle_slice::{A, B};

    let mut device = Device::new(chip_shuffle_cluster_slice.topology()).unwrap();
    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..A::SIZE * B::SIZE).map(|value| value as i32).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut device.pdma)
            .await
            .unwrap();

    let output = launch(chip_shuffle_cluster_slice, (&mut device, &hbm_tensor))
        .await
        .unwrap();

    assert_eq!(
        output
            .to_host::<m![A, B % 512, B / 1024]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A, B % 512, B / 1024], CurrentBackend>::from_vec(
            (0..A::SIZE * (B::SIZE / 2))
                .map(|x| {
                    let target_a = x / (B::SIZE / 2);
                    let target_chip = target_a % 16 / 4;
                    let target_cluster = target_a % 4 / 2;
                    let source_chip = [1, 2, 3, 0][target_chip];
                    let source_a = target_a / 16 * 16 + source_chip * 4 + target_cluster * 2 + target_a % 2;
                    let b_inner = x % (B::SIZE / 2) / 4;
                    let b_outer = x % 4;
                    let b_slice = [1, 0][target_cluster];
                    let source_b = b_outer * (B::SIZE / 4) + b_slice * (B::SIZE / 8) + b_inner;

                    (source_a * B::SIZE + source_b) as i32
                })
                .collect::<Vec<_>>(),
        ),
    );
}

#[tokio::test]
async fn test_shuffle_slice_middle_axis() {
    use furiosa_opt_examples::chip_reduce::{A, B, C, D};

    let mut device = Device::new(shuffle_slice_middle.topology()).unwrap();
    let hbm_tensor: HbmTensor<i32, m![A], m![B, C, D]> = HostTensor::<i32, m![A, B, C, D]>::from_vec(
        (0..A::SIZE * B::SIZE * C::SIZE * D::SIZE)
            .map(|x| x as i32)
            .collect::<Vec<_>>(),
    )
    .to_hbm::<m![A], m![B, C, D]>(&mut device.pdma)
    .await
    .unwrap();

    let output = launch(shuffle_slice_middle, (&mut device, &hbm_tensor)).await.unwrap();

    assert_eq!(
        output
            .to_host::<m![A, C, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A, C, D], CurrentBackend>::from_vec(
            (0..A::SIZE * C::SIZE * D::SIZE)
                .map(|x| {
                    let target_chip = x / (C::SIZE * D::SIZE);
                    let c = x / D::SIZE % C::SIZE;
                    let d = x % D::SIZE;
                    let source_chip = [1, 2, 3, 0][target_chip];

                    (((source_chip * B::SIZE + target_chip) * C::SIZE + c) * D::SIZE + d) as i32
                })
                .collect::<Vec<_>>()
        ),
    );
}

#[tokio::test]
async fn test_shuffle_slice_identity() {
    use furiosa_opt_examples::chip_reduce::{A, B, C, D};

    let mut device = Device::new(shuffle_slice_identity.topology()).unwrap();
    let hbm_tensor: HbmTensor<i32, m![A], m![B, C, D]> = HostTensor::<i32, m![A, B, C, D]>::from_vec(
        (0..A::SIZE * B::SIZE * C::SIZE * D::SIZE)
            .map(|x| x as i32)
            .collect::<Vec<_>>(),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let output = launch(shuffle_slice_identity, (&mut device, &hbm_tensor))
        .await
        .unwrap();

    assert_eq!(
        output
            .to_host::<m![A, C, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        (0..A::SIZE * C::SIZE * D::SIZE)
            .map(|x| {
                let chip = x / (C::SIZE * D::SIZE);
                let c = x / D::SIZE % C::SIZE;
                let d = x % D::SIZE;
                (((chip * B::SIZE + chip) * C::SIZE + c) * D::SIZE + d) as i32
            })
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn test_redistribute_element_tile() {
    use furiosa_opt_examples::chip_reduce::{A, B, C, D, X};

    let mut device = Device::new(redistribute_element_tile.topology()).unwrap();
    let hbm: HbmTensor<i32, m![A], m![B, X, C, D]> = HostTensor::<i32, m![A, B, X, C, D]>::from_vec(
        (0..A::SIZE * B::SIZE * X::SIZE * C::SIZE * D::SIZE)
            .map(|x| x as i32)
            .collect::<Vec<_>>(),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let output = launch(redistribute_element_tile, (&mut device, &hbm)).await.unwrap();

    assert_eq!(
        output
            .to_host::<m![A, C, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        (0..A::SIZE * C::SIZE * D::SIZE)
            .map(|output_index| {
                let chip = output_index / (C::SIZE * D::SIZE);
                let c = output_index / D::SIZE % C::SIZE;
                let d = output_index % D::SIZE;
                ((((chip * B::SIZE + chip) * X::SIZE + 1) * C::SIZE + c) * D::SIZE + d) as i32
            })
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn test_shuffle_slice_middle_place() {
    use furiosa_opt_examples::chip_reduce::{A, B, C, D, MIDDLE_TARGET_POSITION};

    let mut device = Device::new(shuffle_slice_middle_place.topology()).unwrap();
    let hbm_tensor: HbmTensor<i32, m![A], m![B, C, D]> = HostTensor::<i32, m![A, B, C, D]>::from_vec(
        (0..A::SIZE * B::SIZE * C::SIZE * D::SIZE)
            .map(|x| x as i32)
            .collect::<Vec<_>>(),
    )
    .to_hbm(&mut device.pdma)
    .await
    .unwrap();
    let output = launch(shuffle_slice_middle_place, (&mut device, &hbm_tensor))
        .await
        .unwrap();
    let source_chips = [1, 2, 3, 0];

    assert_eq!(
        output
            .to_host::<m![A, C, B, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        (0..A::SIZE * C::SIZE * B::SIZE * D::SIZE)
            .map(|x| {
                let target = x / (C::SIZE * B::SIZE * D::SIZE);
                let c = x / (B::SIZE * D::SIZE) % C::SIZE;
                let b = x / D::SIZE % B::SIZE;
                let d = x % D::SIZE;
                if b == MIDDLE_TARGET_POSITION {
                    (((source_chips[target] * B::SIZE + target) * C::SIZE + c) * D::SIZE + d) as i32
                } else {
                    0
                }
            })
            .collect::<Vec<_>>()
    );
}

#[tokio::test]
async fn test_shuffle_slice_noncontiguous_axes() {
    use furiosa_opt_examples::chip_reduce::{A, B, C, D, X};

    let mut device = Device::new(shuffle_slice_noncontiguous_axes.topology()).unwrap();
    let hbm_tensor: HbmTensor<i32, m![A], m![B, X, C, D]> = HostTensor::<i32, m![A, B, X, C, D]>::from_vec(
        (0..A::SIZE * B::SIZE * X::SIZE * C::SIZE * D::SIZE)
            .map(|x| x as i32)
            .collect::<Vec<_>>(),
    )
    .to_hbm::<m![A], m![B, X, C, D]>(&mut device.pdma)
    .await
    .unwrap();

    let output = launch(shuffle_slice_noncontiguous_axes, (&mut device, &hbm_tensor))
        .await
        .unwrap();
    let source_chips = [1, 2, 3, 0];
    let coordinates = [(0, 3), (1, 0), (2, 5), (3, 7)];

    assert_eq!(
        output
            .to_host::<m![A, X, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A, X, D], CurrentBackend>::from_vec(
            (0..A::SIZE * X::SIZE * D::SIZE)
                .map(|output_index| {
                    let target_chip = output_index / (X::SIZE * D::SIZE);
                    let x = output_index / D::SIZE % X::SIZE;
                    let d = output_index % D::SIZE;
                    let source_chip = source_chips[target_chip];
                    let (b, c) = coordinates[target_chip];

                    ((((source_chip * B::SIZE + b) * X::SIZE + x) * C::SIZE + c) * D::SIZE + d) as i32
                })
                .collect::<Vec<_>>(),
        ),
    );
}

#[tokio::test]
async fn test_shuffle_slice_padded_tail() {
    use furiosa_opt_examples::chip_reduce::{A, B, C, D};

    let mut device = Device::new(shuffle_slice_padded_tail.topology()).unwrap();
    let hbm_tensor: HbmTensor<i32, m![A], m![B, C, D]> = HostTensor::<i32, m![A, B, C, D]>::from_vec(
        (0..A::SIZE * B::SIZE * C::SIZE * D::SIZE)
            .map(|x| x as i32)
            .collect::<Vec<_>>(),
    )
    .to_hbm::<m![A], m![B, C, D]>(&mut device.pdma)
    .await
    .unwrap();

    let output = launch(shuffle_slice_padded_tail, (&mut device, &hbm_tensor))
        .await
        .unwrap();

    assert_eq!(
        output
            .to_host::<m![A, C, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A, C, D], CurrentBackend>::from_vec(
            (0..A::SIZE * C::SIZE * D::SIZE)
                .map(|output_index| {
                    let chip = output_index / (C::SIZE * D::SIZE);
                    let c = output_index / D::SIZE % C::SIZE;
                    let d = output_index % D::SIZE;
                    (((chip * B::SIZE + chip) * C::SIZE + c) * D::SIZE + d) as i32
                })
                .collect::<Vec<_>>(),
        ),
    );
}

#[tokio::test]
async fn test_cluster_swap() {
    use furiosa_opt_examples::cluster_chip_shuffle_slice::{A, B};

    let mut device = Device::new(cluster_swap.topology()).unwrap();
    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..256 * 4096).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut device.pdma)
            .await
            .unwrap();
    let output = launch(cluster_swap, (&mut device, &hbm_tensor)).await.unwrap();

    assert_eq!(
        output.to_host::<m![A, B]>(&mut device.pdma).await.unwrap().into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(
            (0i32..256 * 4096)
                .map(|x| {
                    let index_a_4 = (x / 4096) / 4;
                    let index_a_2 = ((x / 4096) % 4) / 2;
                    let index_a_1 = (x / 4096) % 2;
                    let index_b = x % 4096;

                    let out_index_a_2 = [1, 0][index_a_2 as usize];

                    index_a_4 * 4096 * 4 + out_index_a_2 * 4096 * 2 + index_a_1 * 4096 + index_b
                })
                .collect::<Vec<_>>()
        ),
    );
}

#[tokio::test]
async fn test_cluster_swap_padded() {
    use furiosa_opt_examples::cluster_chip_shuffle_slice::{A, B};

    let mut device = Device::new(cluster_swap_padded.topology()).unwrap();
    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A % 2, B / 1024]> =
        HostTensor::<i32, m![A / 4 % 4, A % 2, B / 1024]>::from_vec((0..32).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A % 2, B / 1024]>(&mut device.pdma)
            .await
            .unwrap();
    let output = launch(cluster_swap_padded, (&mut device, &hbm_tensor)).await.unwrap();

    assert_eq!(
        output
            .to_host::<m![A / 4 % 4, A % 2, B / 1024]>(&mut device.pdma)
            .await
            .unwrap()
            .into_inner(),
        Tensor::<_, m![A / 4 % 4, A % 2, B / 1024], CurrentBackend>::from_vec(
            (0i32..32).map(|x| x / 8 * 8 + (x + 4) % 8).collect::<Vec<_>>()
        ),
    );
}

#[tokio::test]
async fn test_cluster_swap_read_only_padded() {
    use furiosa_opt_examples::cluster_chip_shuffle_slice::{A, B};

    let mut device = Device::new(cluster_swap_read_only_padded.topology()).unwrap();
    let hbm_tensor: HbmTensor<i32, m![1 # 4], m![A % 2, B / 1024, B % 2]> =
        HostTensor::<i32, m![1, A % 2, B / 1024, B % 2]>::from_vec((0..16).collect::<Vec<_>>())
            .to_hbm::<m![1 # 4], m![A % 2, B / 1024, B % 2]>(&mut device.pdma)
            .await
            .unwrap();
    let output = launch(cluster_swap_read_only_padded, (&mut device, &hbm_tensor))
        .await
        .unwrap();

    assert_eq!(
        output
            .to_host::<m![1, A % 2, B / 1024, B % 2]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        (0i32..16).map(|x| (x + 8) % 16).collect::<Vec<_>>()
    );
}
