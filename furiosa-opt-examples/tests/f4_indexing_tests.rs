use furiosa_opt_examples::f4_indexing::{N, f4_indexing};
use furiosa_opt_std::prelude::*;
use rand::rngs::SmallRng;
use rand::{Rng, SeedableRng};

#[tokio::test]
async fn test_f4_indexing() {
    let mut rng = SmallRng::seed_from_u64(42);
    let input: Vec<u8> = (0..32).map(|_| rng.random::<u8>()).collect();

    let mut device = Device::new(f4_indexing.topology()).unwrap();
    let input_hbm = HostTensor::<f4e2m1, m![N]>::from_buf(input.clone())
        .to_hbm::<m![1], m![N]>(&mut device.pdma)
        .await
        .unwrap();

    let output = launch(f4_indexing, (&mut device, &input_hbm)).await.unwrap();
    let output = output
        .to_host::<m![N]>(&mut device.pdma)
        .await
        .unwrap()
        .into_inner()
        .into_buf();

    let mut expected = input[16..].to_vec();
    expected.extend_from_slice(&input[..16]);
    assert_eq!(output, expected);
}
