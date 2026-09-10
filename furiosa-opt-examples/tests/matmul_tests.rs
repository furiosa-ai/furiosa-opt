use furiosa_opt_examples::matmul::{
    matmul_4096, matmul_16384, matmul_chip_reduce, matmul_cluster_reduce, matmul_with_split_reduce, matmul_wo_broadcast,
};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::test]
async fn test_matmul_4096() {
    use furiosa_opt_examples::matmul::matmul_4096::{A, B};

    let mut device = Device::new(matmul_4096.topology()).unwrap();
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![B]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();
    let out = launch(matmul_4096, (&mut device, &lhs_hbm, &rhs_hbm)).await.unwrap();

    let expected: Tensor<_, m![A]> = Tensor::contraction::<m![A, B], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(
        expected.into_vec(),
        out.to_host::<m![A]>(&mut device.pdma).await.unwrap().into_vec()
    );
}

#[tokio::test]
#[ignore = "the 16K contraction is too slow for a regular test"]
async fn test_matmul_16384() {
    use furiosa_opt_examples::matmul::matmul_16384::{A, B, C};

    let mut device = Device::new(matmul_16384.topology()).unwrap();
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![B, C]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let output = launch(matmul_16384, (&mut device, &lhs_hbm, &rhs_hbm)).await.unwrap();
    let expected: Tensor<_, m![A, C]> = Tensor::contraction::<m![A, B, C], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(
        expected.into_vec(),
        output.to_host::<m![A, C]>(&mut device.pdma).await.unwrap().into_vec()
    );
}

// ANCHOR: split_reduce_test
#[tokio::test]
async fn test_matmul_with_split_reduce() {
    use furiosa_opt_examples::matmul::matmul_split_reduce::{A, B};

    let mut device = Device::new(matmul_with_split_reduce.topology()).unwrap();
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![B]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();
    let output = launch(matmul_with_split_reduce, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let expected: Tensor<_, m![A]> = Tensor::contraction::<m![A, B], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(
        expected.into_vec(),
        output.to_host::<m![A]>(&mut device.pdma).await.unwrap().into_vec()
    );
}
// ANCHOR_END: split_reduce_test

#[tokio::test]
async fn test_matmul_with_cluster_reduce() {
    use furiosa_opt_examples::matmul::matmul_cluster_reduce::{A, B, C};

    let mut device = Device::new(matmul_cluster_reduce.topology()).unwrap();
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![B, C]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();
    let out = launch(matmul_cluster_reduce, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let expected: Tensor<_, m![A, C]> =
        Tensor::<_, m![A, C]>::contraction::<m![A, B, C], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(
        expected.into_vec(),
        out.to_host::<m![A, C]>(&mut device.pdma).await.unwrap().into_vec()
    );
}

#[tokio::test]
async fn test_matmul_with_chip_reduce() {
    use furiosa_opt_examples::matmul::matmul_chip_reduce::{A, B, C};

    let mut device = Device::new(matmul_chip_reduce.topology()).unwrap();
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![B, C]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();
    let out = launch(matmul_chip_reduce, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let expected: Tensor<_, m![A, C]> =
        Tensor::<_, m![A, C]>::contraction::<m![A, B, C], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(
        expected.into_vec(),
        out.to_host::<m![A, C]>(&mut device.pdma).await.unwrap().into_vec()
    );
}

#[tokio::test]
// Oracle debt: neither backend checks the result, only that the kernel compiles.
#[cfg_attr(not(backend = "npu"), ignore = "Failing on cpu")]
#[cfg_attr(
    backend = "npu",
    ignore = "TODO: HostTensor::rand / Tensor::contraction not implemented for Npu BufRawTensor"
)]
async fn test_matmul_wo_broadcast() {
    use furiosa_opt_examples::matmul::matmul_wo_broadcast::{A, B};

    let mut device = Device::new(matmul_wo_broadcast.topology()).unwrap();

    // Generate random input tensors and allocate output tensor.
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![A, B]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out = launch(matmul_wo_broadcast, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    let expected: Tensor<_, m![1]> = Tensor::contraction::<m![A, B], _, _>(&lhs.into_inner(), &rhs.into_inner());
    assert_eq!(
        expected.into_vec(),
        out.to_host::<m![1]>(&mut device.pdma).await.unwrap().into_vec()
    );
}
