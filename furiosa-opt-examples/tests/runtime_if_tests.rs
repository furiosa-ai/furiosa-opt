//! Answer-key tests for the runtime-`if` device functions: run them on the VISA
//! simulator and check the computed output against the oracle in each doc.

use furiosa_opt_examples::runtime_if::{W, runtime_if_chain, runtime_if_const, runtime_if_two_outputs};
use furiosa_opt_std::prelude::*;

fn input_host() -> HostTensor<i32, m![W]> {
    HostTensor::<i32, m![W]>::from_vec((0..<m![W]>::SIZE as i32).collect::<Vec<_>>())
}

/// Statement form: the two arms write two separate output tensors, `input + 1` and `input + 2`.
#[tokio::test]
async fn test_runtime_if_two_outputs() -> eyre::Result<()> {
    let mut device = Device::new(runtime_if_two_outputs.topology())?;
    let input = input_host();
    let input_hbm = input.to_hbm(&mut device.pdma).await?;

    let (out_then, out_else) = launch(runtime_if_two_outputs, (&mut device, &input_hbm)).await?;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out_then.to_host::<m![W]>(&mut device.pdma).await?.into_vec()
    );
    assert_eq!(
        input.clone().into_inner().map(|x| x + 2).into_vec(),
        out_else.to_host::<m![W]>(&mut device.pdma).await?.into_vec()
    );

    Ok(())
}

/// Branch result consumed by a further op (`mid` then `+10`) over a live carry: `output == input + 23`.
#[tokio::test]
async fn test_runtime_if_chain() -> eyre::Result<()> {
    let mut device = Device::new(runtime_if_chain.topology())?;
    let input = input_host();
    let input_hbm = input.to_hbm(&mut device.pdma).await?;

    let out = launch(runtime_if_chain, (&mut device, &input_hbm)).await?;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 23).into_vec(),
        out.to_host::<m![W]>(&mut device.pdma).await?.into_vec()
    );

    Ok(())
}

/// Constant condition (`if true`): the then-arm is always taken, so `output == input + 1`.
#[tokio::test]
async fn test_runtime_if_const() -> eyre::Result<()> {
    let mut device = Device::new(runtime_if_const.topology())?;
    let input = input_host();
    let input_hbm = input.to_hbm(&mut device.pdma).await?;

    let out = launch(runtime_if_const, (&mut device, &input_hbm)).await?;

    assert_eq!(
        input.clone().into_inner().map(|x| x + 1).into_vec(),
        out.to_host::<m![W]>(&mut device.pdma).await?.into_vec()
    );

    Ok(())
}
