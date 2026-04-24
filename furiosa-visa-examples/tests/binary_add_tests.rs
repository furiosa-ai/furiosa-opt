use furiosa_visa_examples::binary_add::{A, binary_add_2048};
use furiosa_visa_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::test]
async fn test_binary_add_2048() {
    let mut ctx = Context::acquire();

    // Generate random input tensors and allocate output tensor.
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![A]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out = launch(binary_add_2048, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    assert_eq!(
        lhs.into_inner()
            .map(|x| x.map(|x| x as i32))
            .zip_with(&rhs.into_inner().map(|x| x.map(|x| x as i32)), |a, b| a + b)
            .to_buf(),
        out.to_host::<m![A]>(&mut ctx.pdma).await.to_buf()
    );
}
