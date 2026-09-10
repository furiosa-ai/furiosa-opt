//! Answer-key test for the scalar-`if` value form (a runtime tile index runs and copies correctly).
//!
//! The negative form ([`runtime_if_scalar_immediate`]) is NOT tested here: the VISA simulator runs
//! the loop with concrete indices, so a runtime `if`-selected immediate simulates fine; the
//! compile-time-constant requirement is a codegen constraint. Its rejection is pinned by the
//! `runtime_if_scalar::runtime_if_scalar_immediate` entry in the `npu-visa-test` snapshot gate,
//! which asserts it fails at the `mir` stage with the expected diagnostic.

use furiosa_opt_examples::runtime_if_scalar::{A, B, runtime_if_scalar_index};
use furiosa_opt_std::prelude::*;

/// A runtime `if`-selected tile index is a valid view index: the kernel copies both `B`-halves and
/// `output == input`.
#[tokio::test]
async fn test_runtime_if_scalar_index() {
    let mut device = Device::new(runtime_if_scalar_index.topology()).unwrap();
    let input = HostTensor::<i8, m![A, B]>::from_vec((0..<m![A, B]>::SIZE).map(|x| x as i8).collect::<Vec<_>>());
    let input_hbm = input.to_hbm(&mut device.pdma).await.unwrap();

    let out = launch(runtime_if_scalar_index, (&mut device, &input_hbm))
        .await
        .unwrap();

    assert_eq!(
        input.clone().into_inner().into_vec(),
        out.to_host::<m![A, B]>(&mut device.pdma).await.unwrap().into_vec()
    );
}
