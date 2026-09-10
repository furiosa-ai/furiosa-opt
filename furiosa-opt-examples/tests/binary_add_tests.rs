use furiosa_opt_examples::binary_add::{A, binary_add_2048};
use furiosa_opt_std::prelude::*;
use rand::SeedableRng;
use rand::rngs::SmallRng;

#[tokio::test]
async fn test_binary_add_2048() {
    let mut device = Device::new(binary_add_2048.topology()).unwrap();

    // Generate random input tensors and allocate output tensor.
    let mut rng = SmallRng::seed_from_u64(42);
    let lhs = HostTensor::<i8, m![A]>::rand(&mut rng);
    let rhs = HostTensor::<i8, m![A]>::rand(&mut rng);
    let lhs_hbm = lhs.to_hbm(&mut device.pdma).await.unwrap();
    let rhs_hbm = rhs.to_hbm(&mut device.pdma).await.unwrap();

    let out = launch(binary_add_2048, (&mut device, &lhs_hbm, &rhs_hbm))
        .await
        .unwrap();

    assert_eq!(
        lhs.into_inner()
            .map(|x| x as i32)
            .zip_with(&rhs.into_inner().map(|x| x as i32), |x, y| x + y)
            .into_vec(),
        out.to_host::<m![A]>(&mut device.pdma).await.unwrap().into_vec()
    );
}

/// A pinned host tensor round-trips through HBM into a pinned output, both transfers reusing the
/// allocations the caller handed in.
#[tokio::test]
async fn pinned_round_trip_reuses_output() {
    let mut device = Device::new(binary_add_2048.topology()).unwrap();
    let logical = (0..<m![A]>::SIZE).map(|i| i as i8).collect::<Vec<_>>();

    let input = HostTensor::<i8, m![A]>::from_vec(logical.clone()).pinned().unwrap();
    let mut hbm: HbmTensor<i8, m![1], m![A]> = HostTensor::<i8, m![A]>::zero().to_hbm(&mut device.pdma).await.unwrap();
    input.to_hbm(&mut device.pdma).output(&mut hbm).await.unwrap();

    let mut output = HostTensor::<i8, m![A]>::zero().pinned().unwrap();
    hbm.to_host(&mut device.pdma).output(&mut output).await.unwrap();
    assert_eq!(output.into_vec(), logical);
}
