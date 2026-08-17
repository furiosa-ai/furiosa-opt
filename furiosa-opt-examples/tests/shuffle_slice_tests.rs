#![expect(clippy::type_complexity)]

use furiosa_opt_examples::cluster_chip_shuffle_slice::chip_shuffle;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_chip_shuffle() {
    use furiosa_opt_examples::cluster_chip_shuffle_slice::{A, B};

    let mut ctx = Context::acquire();

    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_vec((0..256 * 4096).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut ctx.pdma)
            .await;

    let output = launch(chip_shuffle, (&mut *ctx, &hbm_tensor)).await;

    assert_eq!(
        output.to_host::<m![A, B]>(&mut ctx.pdma).await.into_inner(),
        Tensor::<_, m![A, B], CurrentBackend>::from_vec(
            (0i32..256 * 4096)
                .map(|x| {
                    let index_a_16 = x / 4096 / 16;
                    let index_a_4 = ((x / 4096) % 16) / 4;
                    let index_a_1 = x / 4096 % 4;
                    let index_b = x % 4096;

                    let out_index_a_4 = [1, 2, 3, 0][index_a_4 as usize];

                    index_a_16 * 4096 * 16 + out_index_a_4 * 4096 * 4 + index_a_1 * 4096 + index_b
                })
                .collect::<Vec<_>>()
        ),
    );
}
