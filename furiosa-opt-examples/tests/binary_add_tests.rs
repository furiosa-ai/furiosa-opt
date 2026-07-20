use furiosa_opt_examples::binary_add::{A, binary_add_2048};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::test]
async fn test_binary_add_2048() {
    let mut ctx = Context::acquire();

    // Generate random input tensors and allocate output tensor.
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![A]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma).await;

    let out = launch(binary_add_2048, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    assert_eq!(
        lhs.into_inner()
            .map(|x| x as i32)
            .zip_with(&rhs.into_inner().map(|x| x as i32), |x, y| x + y)
            .into_vec(),
        out.to_host::<m![A]>(&mut ctx.pdma).await.into_vec()
    );
}
