use furiosa_opt_examples::to_vrf_assertions::{A, to_vrf_fills_file};
use furiosa_opt_std::prelude::*;

type Chip = m![1];

/// An operand filling the register file exactly is accepted and runs. The rejected case is a
/// compile-time error, so it lives as a `compile_fail` example in the module's docs instead.
#[tokio::test]
async fn test_to_vrf_fills_file() {
    let mut device = Device::new(to_vrf_fills_file.topology()).unwrap();

    let input = HostTensor::<i32, m![A]>::from_vec((0..<m![A]>::SIZE).map(|x| x as i32).collect::<Vec<_>>())
        .to_hbm::<Chip, m![A]>(&mut device.pdma)
        .await
        .unwrap();

    launch(to_vrf_fills_file, (&mut device, &input)).await.unwrap();
}
