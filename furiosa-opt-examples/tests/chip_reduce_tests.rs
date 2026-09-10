#[path = "support/reduce.rs"]
mod reduce_common;

use furiosa_opt_examples::chip_reduce::{
    A, B, C, D, all_gather, all_gather_after_reduce_scatter, all_reduce, butterfly_all_reduce, reduce_scatter,
};
use furiosa_opt_std::prelude::*;
use reduce_common::{all_reduced, all_reduced_shards, gathered_shards, reduce_scattered, sequential_i32};

fn full_input_data() -> Vec<i32> {
    sequential_i32(A::SIZE * B::SIZE * C::SIZE * D::SIZE)
}

fn shard_input_data() -> Vec<i32> {
    sequential_i32(A::SIZE * C::SIZE * D::SIZE)
}

async fn full_input(device: &mut Device) -> HbmTensor<i32, m![A], m![B, C, D]> {
    HostTensor::<i32, m![A, B, C, D]>::from_vec(full_input_data())
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

async fn shard_input(device: &mut Device) -> HbmTensor<i32, m![A], m![C, D]> {
    HostTensor::<i32, m![A, C, D]>::from_vec(shard_input_data())
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

#[tokio::test]
async fn test_reduce_scatter() {
    let mut device = Device::new(reduce_scatter.topology()).unwrap();
    let input = full_input(&mut device).await;
    let output = launch(reduce_scatter, (&mut device, &input)).await.unwrap();

    assert_eq!(
        output
            .to_host::<m![A, C, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        reduce_scattered(A::SIZE, B::SIZE, C::SIZE, D::SIZE)
    );
}

#[tokio::test]
async fn test_all_gather() {
    let mut device = Device::new(all_gather.topology()).unwrap();
    let input = shard_input(&mut device).await;
    let output = launch(all_gather, (&mut device, &input)).await.unwrap();

    assert_eq!(
        output
            .to_host::<m![4, A, C, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        gathered_shards(A::SIZE, A::SIZE, C::SIZE, D::SIZE)
    );
}

#[tokio::test]
async fn test_all_reduce() {
    let mut device = Device::new(all_reduce.topology()).unwrap();
    let input = shard_input(&mut device).await;
    let output = launch(all_reduce, (&mut device, &input)).await.unwrap();

    assert_eq!(
        output
            .to_host::<m![A, C, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        all_reduced(A::SIZE, C::SIZE, D::SIZE)
    );
}

#[tokio::test]
async fn test_all_gather_after_reduce_scatter() {
    let mut device = Device::new(all_gather_after_reduce_scatter.topology()).unwrap();
    let input = full_input(&mut device).await;
    let output = launch(all_gather_after_reduce_scatter, (&mut device, &input))
        .await
        .unwrap();

    assert_eq!(
        output
            .to_host::<m![4, B, C, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        all_reduced_shards(A::SIZE, A::SIZE, B::SIZE, C::SIZE, D::SIZE)
    );
}

#[tokio::test]
async fn test_butterfly_all_reduce() {
    let mut device = Device::new(butterfly_all_reduce.topology()).unwrap();
    let input = shard_input(&mut device).await;
    let output = launch(butterfly_all_reduce, (&mut device, &input)).await.unwrap();

    assert_eq!(
        output
            .to_host::<m![A, C, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        all_reduced(A::SIZE, C::SIZE, D::SIZE)
    );
}
