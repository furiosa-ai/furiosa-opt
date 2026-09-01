//! Tests for transpose operations.

#![expect(clippy::type_complexity, reason = "HbmTensor generics are inherently long")]

use furiosa_opt_examples::transpose::{transpose_i8_tu, transpose_i16_tu, transpose_simple};
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_transpose_simple() {
    use furiosa_opt_examples::transpose::{A, B, C};

    let _ = env_logger::try_init();

    let mut ctx = Context::acquire();

    // Create input tensor with values 0..A*B*C
    let total = 8 * 16 * 32;
    let input_data: Vec<f32> = (0..total).map(|x| x as f32).collect();

    let hbm_input: HbmTensor<f32, m![1], m![A, B, C]> = HostTensor::<f32, m![A, B, C]>::from_vec(input_data.clone())
        .to_hbm::<m![1], m![A, B, C]>(&mut ctx.pdma)
        .await;

    let output = launch(transpose_simple, (&mut *ctx, &hbm_input)).await;

    let actual = output.to_host::<m![C, A, B]>(&mut ctx.pdma).await.into_inner();

    let mut expected = vec![0f32; total];
    let mut idx = 0;
    for c in 0..32 {
        for a in 0..8 {
            for b in 0..16 {
                expected[idx] = (a * 16 * 32 + b * 32 + c) as f32;
                idx += 1;
            }
        }
    }
    assert_eq!(actual, Tensor::<_, m![C, A, B], CurrentBackend>::from_vec(expected));
}

#[tokio::test]
async fn test_transpose_i8_tu() {
    use furiosa_opt_examples::transpose::{A, B, P};

    let mut ctx = Context::acquire();
    let total = 64 * 8 * 16;
    let input_data: Vec<i8> = (0..total).map(|x| (x % 127) as i8).collect();
    let hbm_input = HostTensor::<i8, m![P, A, B]>::from_vec(input_data.clone())
        .to_hbm::<m![1], m![P, A, B]>(&mut ctx.pdma)
        .await;

    let output = launch(transpose_i8_tu, (&mut *ctx, &hbm_input)).await;
    let actual = output.to_host::<m![P, B, A]>(&mut ctx.pdma).await.into_inner();

    let mut expected = Vec::with_capacity(total);
    for p in 0..64 {
        for b in 0..16 {
            for a in 0..8 {
                expected.push(input_data[p * 8 * 16 + a * 16 + b]);
            }
        }
    }
    assert_eq!(actual, Tensor::<_, m![P, B, A], CurrentBackend>::from_vec(expected));
}

#[tokio::test]
async fn test_transpose_i16_tu() {
    use furiosa_opt_examples::transpose::{B, D, P};

    let mut ctx = Context::acquire();
    let total = 64 * 4 * 16;
    let input_data: Vec<i16> = (0..total).map(|x| x as i16).collect();
    let hbm_input = HostTensor::<i16, m![P, D, B]>::from_vec(input_data.clone())
        .to_hbm::<m![1], m![P, D, B]>(&mut ctx.pdma)
        .await;

    let output = launch(transpose_i16_tu, (&mut *ctx, &hbm_input)).await;
    let actual = output.to_host::<m![P, B, D]>(&mut ctx.pdma).await.into_inner();

    let mut expected = Vec::with_capacity(total);
    for p in 0..64 {
        for b in 0..16 {
            for d in 0..4 {
                expected.push(input_data[p * 4 * 16 + d * 16 + b]);
            }
        }
    }
    assert_eq!(actual, Tensor::<_, m![P, B, D], CurrentBackend>::from_vec(expected));
}
