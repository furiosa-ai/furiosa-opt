#![expect(clippy::type_complexity)]

use furiosa_visa_examples::cluster_chip_shuffle_slice::{chip_shuffle, chip_slice, cluster_shuffle, cluster_slice};
use furiosa_visa_std::prelude::*;

#[tokio::test]
async fn test_chip_slice() {
    use furiosa_visa_examples::cluster_chip_shuffle_slice::{A, B};

    let mut ctx = Context::acquire();

    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_buf((0..256 * 4096).map(Opt::Init).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut ctx.pdma, 0x1000)
            .await;

    let output = launch(chip_slice, (&mut *ctx, &hbm_tensor)).await;

    assert_eq!(
        output.to_host::<m![A, B % 512, B / 2048]>(&mut ctx.pdma).await.to_buf(),
        Tensor::<_, m![A, B % 512, B / 2048]>::from_buf(
            (0..256 * 1024)
                .map(|x| {
                    let index_a = x / 1024;
                    let index_b_1 = (x % 1024) / 2;
                    let index_b_2048 = x % 2;

                    let index_a_4 = (index_a % 16) / 4;

                    let out_index_a = index_a;

                    let index_b_512 = index_a_4;
                    let out_index_b = index_b_2048 * 2048 + index_b_512 * 512 + index_b_1;

                    out_index_a * 4096 + out_index_b
                })
                .map(Opt::Init)
                .collect::<Vec<_>>()
        )
        .to_buf(),
    );
}

#[tokio::test]
async fn test_cluster_slice() {
    use furiosa_visa_examples::cluster_chip_shuffle_slice::{A, B};

    let mut ctx = Context::acquire();

    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_buf((0..256 * 4096).map(Opt::Init).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut ctx.pdma, 0x1000)
            .await;

    let output = launch(cluster_slice, (&mut *ctx, &hbm_tensor)).await;

    assert_eq!(
        output.to_host::<m![A, B % 512, B / 1024]>(&mut ctx.pdma).await.to_buf(),
        Tensor::<_, m![A, B % 512, B / 1024]>::from_buf(
            (0..256 * 2048)
                .map(|x| {
                    let index_a = x / 2048;
                    let index_b_1 = (x % 2048) / 4;
                    let index_b_1024 = x % 4;

                    let index_a_2 = (index_a % 4) / 2;

                    let out_index_a = index_a;

                    let index_b_512 = index_a_2;
                    let out_index_b = index_b_1024 * 1024 + index_b_512 * 512 + index_b_1;

                    out_index_a * 4096 + out_index_b
                })
                .map(Opt::Init)
                .collect::<Vec<_>>()
        )
        .to_buf(),
    );
}

#[tokio::test]
async fn test_chip_shuffle() {
    use furiosa_visa_examples::cluster_chip_shuffle_slice::{A, B};

    let mut ctx = Context::acquire();

    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_buf((0..256 * 4096).map(Opt::Init).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut ctx.pdma, 0x1000)
            .await;

    let output = launch(chip_shuffle, (&mut *ctx, &hbm_tensor)).await;

    assert_eq!(
        output.to_host::<m![A, B]>(&mut ctx.pdma).await.to_buf(),
        Tensor::<i32, m![A, B]>::from_buf(
            (0i32..256 * 4096)
                .map(|x| {
                    let index_a_16 = x / 4096 / 16;
                    let index_a_4 = ((x / 4096) % 16) / 4;
                    let index_a_1 = x / 4096 % 4;
                    let index_b = x % 4096;

                    let out_index_a_4 = [1, 2, 3, 0][index_a_4 as usize];

                    index_a_16 * 4096 * 16 + out_index_a_4 * 4096 * 4 + index_a_1 * 4096 + index_b
                })
                .map(Opt::Init)
                .collect::<Vec<_>>()
        )
        .to_buf(),
    );
}

#[tokio::test]
async fn test_cluster_shuffle() {
    use furiosa_visa_examples::cluster_chip_shuffle_slice::{A, B};

    let mut ctx = Context::acquire();

    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_buf((0..256 * 4096).map(Opt::Init).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut ctx.pdma, 0x1000)
            .await;

    let output = launch(cluster_shuffle, (&mut *ctx, &hbm_tensor)).await;

    assert_eq!(
        output.to_host::<m![A, B]>(&mut ctx.pdma).await.to_buf(),
        Tensor::<i32, m![A, B]>::from_buf(
            (0i32..256 * 4096)
                .map(|x| {
                    let index_a_4 = (x / 4096) / 4;
                    let index_a_2 = ((x / 4096) % 4) / 2;
                    let index_a_1 = (x / 4096) % 2;
                    let index_b = x % 4096;

                    let out_index_a_2 = [1, 0][index_a_2 as usize];

                    index_a_4 * 4096 * 4 + out_index_a_2 * 4096 * 2 + index_a_1 * 4096 + index_b
                })
                .map(Opt::Init)
                .collect::<Vec<_>>()
        )
        .to_buf(),
    );
}
