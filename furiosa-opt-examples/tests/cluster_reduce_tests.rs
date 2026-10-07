#[path = "support/reduce.rs"]
mod reduce_common;

use furiosa_opt_examples::cluster_reduce::{
    A, B, C, D, all_gather, all_gather_after_reduce_scatter, all_reduce, reduce_scatter,
};
use furiosa_opt_std::prelude::*;
use reduce_common::{all_reduced, all_reduced_shards, gathered_shards, reduce_scattered, sequential_i32};

fn input_data() -> Vec<i32> {
    sequential_i32(A::SIZE * B::SIZE * C::SIZE * D::SIZE)
}

async fn input(device: &mut Device) -> eyre::Result<HbmTensor<i32, m![1], m![A, B, C, D]>> {
    Ok(HostTensor::<i32, m![A, B, C, D]>::from_vec(input_data())
        .to_hbm(&mut device.pdma)
        .await?)
}

async fn partials(device: &mut Device) -> eyre::Result<HbmTensor<i32, m![1], m![A, C, D]>> {
    Ok(
        HostTensor::<i32, m![A, C, D]>::from_vec(sequential_i32(A::SIZE * C::SIZE * D::SIZE))
            .to_hbm(&mut device.pdma)
            .await?,
    )
}

#[tokio::test]
async fn test_reduce_scatter() -> eyre::Result<()> {
    let mut device = Device::new(reduce_scatter.topology())?;
    let input = input(&mut device).await?;
    let output = launch(reduce_scatter, (&mut device, &input)).await?;

    assert_eq!(
        output.to_host::<m![A, C, D]>(&mut device.pdma).await?.into_vec(),
        reduce_scattered(A::SIZE, B::SIZE, C::SIZE, D::SIZE)
    );

    Ok(())
}

#[tokio::test]
async fn test_all_gather() -> eyre::Result<()> {
    let mut device = Device::new(all_gather.topology())?;
    let input = partials(&mut device).await?;
    let output = launch(all_gather, (&mut device, &input)).await?;

    assert_eq!(
        output.to_host::<m![A, C, D]>(&mut device.pdma).await?.into_vec(),
        gathered_shards(1, A::SIZE, C::SIZE, D::SIZE)
    );

    Ok(())
}

#[tokio::test]
async fn test_all_reduce() -> eyre::Result<()> {
    let mut device = Device::new(all_reduce.topology())?;
    let input = partials(&mut device).await?;
    let output = launch(all_reduce, (&mut device, &input)).await?;

    assert_eq!(
        output.to_host::<m![A, C, D]>(&mut device.pdma).await?.into_vec(),
        all_reduced(A::SIZE, C::SIZE, D::SIZE)
    );

    Ok(())
}

#[tokio::test]
async fn test_all_gather_after_reduce_scatter() -> eyre::Result<()> {
    let mut device = Device::new(all_gather_after_reduce_scatter.topology())?;
    let input = input(&mut device).await?;
    let output = launch(all_gather_after_reduce_scatter, (&mut device, &input)).await?;

    assert_eq!(
        output.to_host::<m![B, C, D]>(&mut device.pdma).await?.into_vec(),
        all_reduced_shards(1, A::SIZE, B::SIZE, C::SIZE, D::SIZE)
    );

    Ok(())
}
