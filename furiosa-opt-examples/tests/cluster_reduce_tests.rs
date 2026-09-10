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

async fn input(device: &mut Device) -> HbmTensor<i32, m![1], m![A, B, C, D]> {
    HostTensor::<i32, m![A, B, C, D]>::from_vec(input_data())
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

async fn partials(device: &mut Device) -> HbmTensor<i32, m![1], m![A, C, D]> {
    HostTensor::<i32, m![A, C, D]>::from_vec(sequential_i32(A::SIZE * C::SIZE * D::SIZE))
        .to_hbm(&mut device.pdma)
        .await
        .unwrap()
}

#[tokio::test]
async fn test_reduce_scatter() {
    let mut device = Device::new(reduce_scatter.topology()).unwrap();
    let input = input(&mut device).await;
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
    let input = partials(&mut device).await;
    let output = launch(all_gather, (&mut device, &input)).await.unwrap();

    assert_eq!(
        output
            .to_host::<m![A, C, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        gathered_shards(1, A::SIZE, C::SIZE, D::SIZE)
    );
}

#[tokio::test]
async fn test_all_reduce() {
    let mut device = Device::new(all_reduce.topology()).unwrap();
    let input = partials(&mut device).await;
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
    let input = input(&mut device).await;
    let output = launch(all_gather_after_reduce_scatter, (&mut device, &input))
        .await
        .unwrap();

    assert_eq!(
        output
            .to_host::<m![B, C, D]>(&mut device.pdma)
            .await
            .unwrap()
            .into_vec(),
        all_reduced_shards(1, A::SIZE, B::SIZE, C::SIZE, D::SIZE)
    );
}
