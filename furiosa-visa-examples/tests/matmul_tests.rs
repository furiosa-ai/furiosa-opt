use furiosa_visa_examples::matmul::{
    matmul_4096, matmul_16384, matmul_chip_reduce, matmul_cluster_reduce, matmul_with_split_reduce,
    matmul_with_split_reduce2, matmul_wo_broadcast,
};
use furiosa_visa_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::test]
#[ignore = "takes too much time to run"]
async fn test_matmul_16384() {
    use furiosa_visa_examples::matmul::matmul_16384::{A, B, C};

    let mut ctx = Context::acquire();

    // Generate random input tensors and allocate output tensor.
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![B, C]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    // Call the device function.
    let out = launch(matmul_16384, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    // Verify the output tensor content.
    let expected: Tensor<_, m![A, C]> = Tensor::contraction::<m![A, B, C], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(expected.to_buf(), out.to_host::<m![A, C]>(&mut ctx.pdma).await.to_buf());
}

#[tokio::test]
async fn test_matmul_4096() {
    use furiosa_visa_examples::matmul::matmul_4096::{A, B};

    let mut ctx = Context::acquire();

    // Generate random input tensors and allocate output tensor.
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![B]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    // Call the device function.
    let out = launch(matmul_4096, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    // Verify the output tensor content.
    let expected: Tensor<_, m![A]> = Tensor::contraction::<m![A, B], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(expected.to_buf(), out.to_host::<m![A]>(&mut ctx.pdma).await.to_buf());
}

#[tokio::test]
async fn test_matmul_wo_broadcast() {
    use furiosa_visa_examples::matmul::matmul_wo_broadcast::{A, B};

    let mut ctx = Context::acquire();

    // Generate random input tensors and allocate output tensor.
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out = launch(matmul_wo_broadcast, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let expected: Tensor<_, m![1]> = Tensor::contraction::<m![A, B], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(expected.to_buf(), out.to_host::<m![1]>(&mut ctx.pdma).await.to_buf());
}

#[tokio::test]
async fn test_matmul_with_split_reduce() {
    use furiosa_visa_examples::matmul::matmul_split_reduce::{A, B};

    let mut ctx = Context::acquire();

    // Generate random input tensors and allocate output tensor.
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![B]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let out = launch(matmul_with_split_reduce, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    let expected: Tensor<_, m![A]> = Tensor::contraction::<m![A, B], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(expected.to_buf(), out.to_host::<m![A]>(&mut ctx.pdma).await.to_buf());
}

#[tokio::test]
async fn test_matmul_with_split_reduce2() {
    use furiosa_visa_examples::matmul::matmul_split_reduce2::{K, M, N};

    let mut ctx = Context::acquire();

    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<bf16, m![M, K]>::rand(&mut rng);
    let rhs = HostTensor::<bf16, m![K, N]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    let acc_zero = HostTensor::<bf16, m![M, N]>::zero();
    let acc_zero_hbm = acc_zero.to_hbm(&mut ctx.pdma, 2 << 28).await;

    let out = launch(
        matmul_with_split_reduce2,
        (&mut *ctx, &lhs_hbm, &rhs_hbm, &acc_zero_hbm),
    )
    .await;

    let expected =
        Tensor::<bf16, m![M, N]>::contraction::<m![M, K, N], _, _>(&lhs.into_inner(), &rhs.into_inner()).to_buf();
    let out = out.to_host::<m![M, N]>(&mut ctx.pdma).await.to_buf();

    assert_eq!(expected.len(), out.len(), "output length mismatch");
    for (expected, out) in expected.into_iter().zip(out) {
        match (expected, out) {
            (Opt::Init(expected), Opt::Init(out)) => {
                let (x, y) = (f64::from(expected.to_f32()), f64::from(out.to_f32()));
                let diff = (x - y).abs();
                let norm = (x.abs() + y.abs()).min(f32::MAX as f64);
                assert!(
                    (expected.to_f32().is_nan() && out.to_f32().is_nan()) || diff <= 1.5_f64.max(norm * 0.2),
                    "{expected:?} != {out:?}",
                );
            }
            (Opt::Uninit, Opt::Uninit) => (),
            _ => panic!("{expected:?} != {out:?}"),
        }
    }
}

#[tokio::test]
async fn test_matmul_with_cluster_reduce() {
    use furiosa_visa_examples::matmul::matmul_cluster_reduce::{A, B, C};

    let mut ctx = Context::acquire();

    // Generate random input tensors and allocate output tensor.
    // [A, B] * [B, C] -> [A, C] where B=2 is mapped to Cluster
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![B, C]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    // Call the device function.
    let out = launch(matmul_cluster_reduce, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    // Verify the output tensor content.
    let expected: Tensor<_, m![A, C]> = Tensor::contraction::<m![A, B, C], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(expected.to_buf(), out.to_host::<m![A, C]>(&mut ctx.pdma).await.to_buf());
}

#[tokio::test]
async fn test_matmul_with_chip_reduce() {
    use furiosa_visa_examples::matmul::matmul_chip_reduce::{A, B, C};

    let mut ctx = Context::acquire();

    // Generate random input tensors and allocate output tensor.
    // [A, B] * [B, C] -> [A, C] where B=4 is mapped to Chip
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![B, C]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut ctx.pdma, 0 << 28).await;
    let rhs_hbm = rhs.to_hbm(&mut ctx.pdma, 1 << 28).await;

    // Call the device function (4 chips).
    let out = launch(matmul_chip_reduce, (&mut *ctx, &lhs_hbm, &rhs_hbm)).await;

    // Verify the output tensor content.
    let expected: Tensor<_, m![A, C]> = Tensor::contraction::<m![A, B, C], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(expected.to_buf(), out.to_host::<m![A, C]>(&mut ctx.pdma).await.to_buf());
}
