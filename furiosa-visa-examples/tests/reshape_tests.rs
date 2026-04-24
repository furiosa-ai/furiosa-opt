#![expect(clippy::type_complexity)]

use furiosa_visa_examples::reshape::{different_axes::reshape_different_num_axes, reshape};
use furiosa_visa_std::prelude::*;

#[tokio::test]
async fn test_reshape() {
    use furiosa_visa_examples::reshape::{A, B, C, D, E, F, G, H, I};

    let mut ctx = Context::acquire();

    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_buf((0..256 * 4096).map(Opt::Init).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut ctx.pdma, 0x1000)
            .await;

    let output = launch(reshape, (&mut *ctx, &hbm_tensor)).await;

    assert_eq!(
        output.to_host::<m![C, D, E, F, G, H, I]>(&mut ctx.pdma).await.to_buf(),
        (0..256 * 4096)
            .map(|x| {
                let index_a_4 = x / (2 * 16 * 16 * 16 * 2 * 16);
                let index_a_2 = x % (2 * 16 * 16 * 16 * 2 * 16) / (16 * 16 * 16 * 2 * 16);
                let index_b_1 = x % (16 * 16 * 16 * 2 * 16) / (16 * 16 * 2 * 16);
                let index_b_16 = x % (16 * 16 * 2 * 16) / (16 * 2 * 16);
                let index_b_256 = x % (16 * 2 * 16) / (2 * 16);
                let index_a_1 = x % (2 * 16) / 16;
                let index_a_16 = x % 16;

                let out_index_a = index_a_16 * 16 + index_a_4 * 4 + index_a_2 * 2 + index_a_1;
                let out_index_b = index_b_256 * 256 + index_b_16 * 16 + index_b_1;

                out_index_a * 4096 + out_index_b
            })
            .map(Opt::Init)
            .collect::<Vec<_>>(),
    );
}

#[tokio::test]
#[ignore = "TODO: support reshape between different number of axes"]
async fn test_reshape_different_num_axes() {
    use furiosa_visa_examples::reshape::different_axes::{A, B, C, D, E, F};

    let mut ctx = Context::acquire();

    let hbm_tensor: HbmTensor<i32, m![A / 4 % 4], m![A / 16, A % 4, B]> =
        HostTensor::<i32, m![A, B]>::from_buf((0..256 * 4096).map(Opt::Init).collect::<Vec<_>>())
            .to_hbm::<m![A / 4 % 4], m![A / 16, A % 4, B]>(&mut ctx.pdma, 0x1000)
            .await;

    let output = launch(reshape_different_num_axes, (&mut *ctx, &hbm_tensor)).await;

    assert_eq!(
        output.to_host::<m![C, D, E, F]>(&mut ctx.pdma).await.to_buf(),
        (0..256 * 4096).map(Opt::Init).collect::<Vec<_>>(),
    );
}
