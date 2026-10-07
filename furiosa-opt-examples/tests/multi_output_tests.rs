use furiosa_opt_examples::dma::{PA, PD, dup_two};
use furiosa_opt_std::prelude::*;

#[tokio::test]
async fn tuple_output_writes_to_output() -> eyre::Result<()> {
    let mut device = Device::new(dup_two.topology())?;
    let first = (0..<m![PA, PD]>::SIZE).map(|x| x as i32).collect::<Vec<_>>();
    let second = first.iter().map(|x| x + 1).collect::<Vec<_>>();
    let first_hbm = HostTensor::<i32, m![PA, PD]>::from_vec(first.clone())
        .to_hbm::<m![1], m![PA, PD]>(&mut device.pdma)
        .await?;
    let second_hbm = HostTensor::<i32, m![PA, PD]>::from_vec(second.clone())
        .to_hbm::<m![1], m![PA, PD]>(&mut device.pdma)
        .await?;
    let mut output = (
        HostTensor::<i32, m![PA, PD]>::zero().to_hbm(&mut device.pdma).await?,
        HostTensor::<i32, m![PA, PD]>::zero().to_hbm(&mut device.pdma).await?,
    );

    launch(dup_two, (&mut device, &first_hbm, &second_hbm))
        .output(&mut output)
        .await?;

    assert_eq!(
        output.0.to_host::<m![PA, PD]>(&mut device.pdma).await?.into_vec(),
        first
    );
    assert_eq!(
        output.1.to_host::<m![PA, PD]>(&mut device.pdma).await?.into_vec(),
        second
    );

    Ok(())
}
