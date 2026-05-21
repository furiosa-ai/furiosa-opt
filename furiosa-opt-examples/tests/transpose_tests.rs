//! Tests for transpose operations.

#![expect(clippy::type_complexity, reason = "HbmTensor generics are inherently long")]

use furiosa_opt_examples::transpose::transpose_simple;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_transpose_simple() {
    use furiosa_opt_examples::transpose::{A, B, C};

    let _ = env_logger::try_init();

    let mut ctx = Context::acquire();

    // Create input tensor with values 0..A*B*C
    let total = 8 * 16 * 32;
    let input_data: Vec<f32> = (0..total).map(|x| x as f32).collect();

    let hbm_input: HbmTensor<f32, m![1], m![A, B, C]> = HostTensor::<f32, m![A, B, C]>::from_buf(input_data.clone())
        .to_hbm::<m![1], m![A, B, C]>(&mut ctx.pdma, 0x0)
        .await;

    let output = launch(transpose_simple, (&mut *ctx, &hbm_input)).await;

    let actual = output.to_host::<m![C, A, B]>(&mut ctx.pdma).await.into_raw();

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
    assert_eq!(
        actual,
        <CurrentBackend as Backend>::RawTensor::from_buf::<m![C, A, B]>(expected)
    );
}
