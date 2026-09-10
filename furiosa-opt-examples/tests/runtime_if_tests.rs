//! Answer-key tests for the runtime-`if` device functions: run them on the VISA
//! simulator and check the computed output against the oracle in each doc.

use furiosa_opt_examples::runtime_if::{W, runtime_if_two_outputs};
use furiosa_opt_std::prelude::*;

fn input_host() -> HostTensor<i32, m![W]> {
    HostTensor::<i32, m![W]>::from_vec((0..<m![W]>::SIZE as i32).collect::<Vec<_>>())
}

/// Statement form: the two arms write two separate output tensors, `input + 1` and `input + 2`.
#[tokio::test]
async fn test_runtime_if_two_outputs() {
    let mut device = Device::new(runtime_if_two_outputs.topology()).unwrap();
    let input = input_host();
    let input_hbm = input.to_hbm(&mut device.pdma).await.unwrap();

    let (out_then, out_else) = launch(runtime_if_two_outputs, (&mut device, &input_hbm)).await.unwrap();

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out_then.to_host::<m![W]>(&mut device.pdma).await.unwrap().into_vec()
    );
    assert_eq!(
        input.clone().into_inner().map(|x| x + 2).into_vec(),
        out_else.to_host::<m![W]>(&mut device.pdma).await.unwrap().into_vec()
    );
}
