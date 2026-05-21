use furiosa_opt_examples::fetch_commit::fetch_commit_simple;
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn test_fetch_commit_simple_host() {
    use furiosa_opt_examples::fetch_commit::{A, B};

    let mut ctx = Context::acquire();

    // Create input tensor with shape (A=4096)(B=8).
    let input = HostTensor::<i8, m![A, B]>::from_buf((0..32768).map(|x| x as i8).collect::<Vec<_>>())
        .to_hbm::<m![1], m![A, B]>(&mut ctx.pdma, 0)
        .await;

    // Call the device function.
    let output = launch(fetch_commit_simple, (&mut *ctx, &input)).await;

    let mut expected = vec![0i32; 4096 * 8];
    let mut idx = 0;
    for b in 0..8 {
        for a in 0..4096 {
            expected[idx] = ((a * 8 + b) as i8) as i32;
            idx += 1;
        }
    }

    assert_eq!(
        output.to_host::<m![B, A]>(&mut ctx.pdma).await.into_raw(),
        <CurrentBackend as Backend>::RawTensor::from_buf::<m![B, A]>(expected)
    );
}
